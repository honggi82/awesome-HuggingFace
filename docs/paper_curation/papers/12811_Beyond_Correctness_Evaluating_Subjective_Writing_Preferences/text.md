# arXiv:2510.14616v2[cs.CL]28Jan2026

[Figure 1]

## Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures

ByteDance Seed, M-A-P Full author list in Contributions

#### Abstract

Current preference learning methods achieve high accuracy on standard benchmarks but exhibit significant performance degradation when objective quality signals are removed. We introduce WritingPreferenceBench, a dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where responses are matched for objective correctness, factual accuracy, and length. On this benchmark, sequence-based reward models—the standard architecture for RLHF—achieve only 52.7% mean accuracy, while zero-shot language model judges perform at 53.9%. In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%. This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification.

Date: January 29, 2026 Correspondence: Ge Zhang at zhangge.eli@bytedance.com Project Page: https://WritingPreferenceBench.github.io/

#### 1 Introduction

Reinforcement learning from human feedback (RLHF) has become the dominant paradigm for aligning language models with human values [3, 9, 23]. Reward models trained via RLHF achieve 95% accuracy on RewardBench’s objective tasks [14], which emphasize safety violations, factual errors, and instructionfollowing. However, our benchmark reveals a critical limitation: when we systematically remove objective quality signals (grammatical errors, factual mistakes, length differences), sequence-based reward models—the dominant architecture in production RLHF systems—collapse to 52.7% accuracy on writing preference tasks, barely above random chance. This 42-percentage-point degradation indicates that current preference learning primarily optimizes for error detection rather than recognition of subjective creative quality—a fundamental misalignment between training objectives and the aesthetic judgment required for creative tasks.

However, writing tasks constitute over 40% of language model interactions [2, 22], spanning creative fiction, persuasive essays, and personal expression where subjective quality matters more than objective correctness. Yet our evaluation infrastructure remains anchored in verifiable metrics. RewardBench [14] conflates safety with preference; WritingBench mixes creative with functional tasks [31]; LitBench uses Reddit upvotes as quality proxies [11]. Moreover, existing benchmarks predominantly focus on English, leaving cross-lingual

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

EN 1200 Pairs ZH 600 Pairs

[Figure 7]

[Figure 8]

[Figure 9]

### Ignore factors Writing Preference Bench

[Figure 10]

[Figure 11]

[Figure 12]

- • Grammer Error
- • Factual Mistakes
- • Length Gaps

[Figure 13]

Cross-Cultural Subjective Writing preference benchmark

###### Isolation → Focus

[Figure 14]

[Figure 15]

Emotional Resonance

Stylistic Flair

Model Instability Across Genres

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Peotry Prose Abstract Interview Novel Niche Drama Advertise Other

[Figure 25]

Best Reward Model acc: 73.3(zh),81.8(en) Generative Reward Model

Best LLM Judge acc: 62.7(zh),68.7(en)

[Figure 26]

Gemini-2.5-pro,Doubao-1.5-pro SC-RM < GenRM

- Figure 1 WritingPreferenceBench isolates subjective writing quality by neutralizing objective confounds (grammar, factuality, length). Across 1,800 human-validated preference pairs, standard sequence classifiers (SC-RM) perform near-randomly while generative reward models (GenRM) achieve 30% higher accuracy—but both architectures exhibit catastrophic instability across genres, exposing the brittleness of current preference learning.

preference evaluation, particularly for languages with distinct rhetorical traditions like Chinese, largely unexplored. Recent theoretical work warns of “reward hacking” where models exploit spurious correlations rather than learning genuine preferences [24].

We introduce WritingPreferenceBench, a cross-lingual dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres where both responses are grammatically correct, factually accurate, and length-matched. We focus on three dimensions of creative quality: creativity (original ideas and novel perspectives), stylistic sophistication (narrative techniques and linguistic elegance), and emotional resonance (capacity to evoke authentic responses). By neutralizing objective confounds, our benchmark tests whether models can recognize these aesthetic qualities that distinguish compelling from merely competent writing.

Our evaluation of 21 models, comprising 7 reward models and 14 language models as judges spanning opensource and proprietary families, reveals fundamental limitations in current preference learning approaches. Sequence-based reward models, the dominant architecture in production RLHF systems [25], achieve only 52.7% mean accuracy across both languages, while zero-shot language model judges [32] reach 53.9%. Both results are statistically indistinguishable from random chance. In stark contrast, generative reward models that produce explicit reasoning chains [6] achieve 81.8% accuracy. This 29-percentage-point gap indicates that subjective preference modeling requires structured intermediate reasoning rather than direct pattern matching. Moreover, all architectures exhibit severe genre instability (mean standard deviation 10.9%), with individual models ranging from 18.2% to 81.8% accuracy across categories, suggesting reliance on superficial heuristics rather than generalizable aesthetic principles. Notably, these failures persist across model scales: 27B parameter models show no consistent improvement over 8B variants, and reasoning-enhanced LLMs (Claude-4-Opus-thinking, OpenAI-o3) provide no advantage over standard architectures.

Contributions. We make three contributions to understanding preference learning:

• Benchmark: WritingPreferenceBench provides 1,800 validated preference pairs with systematic signal isolation across English and Chinese, enabling reproducible cross-lingual evaluation of subjective preference modeling.

Pair Generation

Query

[Figure 27]

Low Score Response

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

###### …

High Score Response

Peotry Prose Abstract Interview Novel Niche

Drama Advertise

| | |
|---|---|
| | |

Literature

Human Preference Pair

104 Chinese Queries - 51 types, Write 4 for each type 217 English Queries - 51 types, Write 7 for each type

600 Chinese Pairs 1200 English Pairs

Model’s Response

Reward Model Evaluation

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

… 20 Models

Pairs（1800 Total）

Input

[Figure 40]

2080 Chinese Responses - From 104 queries and 20 models 2604 English Responses - From 217 queries and 12 models

Reward … Model

Qwen WorldPM

Skywork Reward

[Figure 41]

Model Response--MR Human Preference--HP

[Figure 42]

[Figure 43]

Human Evaluation

[Figure 44]

[Figure 45]

Output

[Figure 46]

No Ordinary Good Excellent

[Figure 47]

|2|
|---|

|0|
|---|

|1|
|---|

|3|
|---|

1800

1 𝑖𝑖𝑖𝑖 𝑀𝑀𝑅𝑅𝑖𝑖 = 𝐻𝐻𝑅𝑅𝑖𝑖 0 𝑖𝑖𝑖𝑖 𝑀𝑀𝑅𝑅𝑖𝑖 ≠ 𝐻𝐻𝑅𝑅𝑖𝑖

𝑆𝑆𝑆𝑆𝑆𝑆𝑤𝑤𝑤𝑤𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚𝑚=

𝛿𝛿𝑖𝑖 ,𝑤𝑤𝑤𝑤𝑤𝑤𝑤𝑤𝑤 𝛿𝛿𝑖𝑖 =

Annotator

𝑖𝑖=1

Score Scale

- Figure 2 The data curation pipeline of WritingPreferenceBench. Our multi-stage process begins with expert-crafted queries across 51 genres, generates diverse responses using 20 state-of-the-art models, and culminates in rigorous human evaluation by trained annotators. Quality control mechanisms operate throughout to ensure preference pairs reflect genuine subjective quality distinctions rather than objective differences.

- • Empirical findings: Comprehensive evaluation establishes that (i) sequence classifiers fail systematically on subjective tasks, (ii) generative reward models with reasoning achieve 30% higher accuracy, and (iii) zero-shot LLM judges cannot reliably assess creative quality despite instruction tuning.
- • Architectural insights: Evidence that successful preference learning requires intermediate reasoning representations, not just pattern matching, with implications for next-generation RLHF systems.

#### 2 WritingPreferenceBench

The fundamental challenge in evaluating subjective writing is not merely collecting human judgments, but ensuring those judgments isolate genuine aesthetic and stylistic preference from objective quality signals. We present WritingPreferenceBench, a meticulously constructed benchmark that addresses this challenge through 1,800 preference pairs spanning English and Chinese creative writing. The construction process, illustrated in

- Figure 2, was guided by rigorous design principles and implemented through a human-in-the-loop pipeline designed to systematically eliminate confounding variables.

##### 2.1 Benchmark Construction Pipeline

We implemented a multi-stage pipeline that translates our design principles into a concrete, reproducible workflow. This process, depicted in Figure 2, first generates a diverse and culturally-rich set of candidate responses and then applies a rigorous, human-led filtering protocol to isolate pairs that represent genuine subjective preference.

- Phase 1: Architecting Diverse and Representative Queries. The benchmark’s foundation is a taxonomy of 51 creative writing categories, developed by merging taxonomies from established writing communities. To ensure both representational diversity and practical relevance, these categories range from classical literary traditions (e.g., poetry, fiction) to contemporary forms (e.g., advertising copy, social media content). Query

development followed a dual-expertise workflow where two experienced creative writing instructors drafted and aligned on creative blueprints for each category. For the English subset, queries were designed to be culturally neutral, avoiding references specific to any single Anglophone culture. For the Chinese subset, queries were adapted to reflect rhetorical conventions and genre expectations common in Mandarin writing traditions, while maintaining structural parallelism with English queries to enable cross-lingual comparison.

To scale query generation while maintaining quality, we employed a human-AI collaborative approach: the expert-authored blueprints (typically 2-3 sentences outlining the creative task and key constraints) were expanded into full queries using Gemini 2.5 Pro. Each generated query was then independently reviewed by both instructors, who validated it for creative intent, appropriate difficulty, and evaluative granularity. Queries requiring revision underwent iterative refinement (3-5 rounds on average) until both experts reached consensus on quality and alignment with the blueprint specifications.

- Phase 2: Generating a Spectrum of Responses. To create a rich and varied corpus for curation, we utilized a diverse suite of 20 state-of-the-art language models, including GPT-4.1, Claude-4, Gemini-2.5-Pro, and Doubao-

- 1.5-Pro. For each query, every model produced 5 outputs with temperature sampling set to T = 0.8. This strategy ensured the generation of a wide spectrum of quality—from formulaic to highly original—providing the necessary variance to identify the controlled preference gaps central to our benchmark’s design.

- Phase 3: Human-in-the-Loop Annotation and Quality Control. This phase is the cornerstone of our methodology, operationalizing a focus on subjective quality through a rigorous, integrated annotation and filtering protocol.

- • Initial Triage: Filtering for Objective Correctness. Before subjective assessment, an automated screening process removed responses with objective deficiencies. This filter eliminated approximately 15% of the raw responses, discarding outputs with comprehension-impeding grammatical errors, factual inconsistencies, or clear prompt violations. This crucial step ensures our benchmark tests for subjective quality, not basic error detection.
- • Expert Evaluation with a Calibrated Rubric. We recruited 11 expert annotators through a professional annotation service with demonstrated expertise in creative content evaluation. Annotator selection criteria included:

- – Language proficiency: English subset annotators (n=4) demonstrated native or near-native fluency; Chinese subset annotators (n=7) were native Mandarin speakers.
- – Writing competency: All annotators passed a qualification assessment requiring them to critique sample creative texts and provide detailed justifications aligned with our evaluation rubric.
- – Genre familiarity: Annotators demonstrated knowledge across multiple writing genres through a pre-screening questionnaire covering the 8 macro-categories in our taxonomy.

All annotators completed an 8-hour training program consisting of: (1) instruction on the evaluation rubric and subjective quality dimensions, (2) practice annotation of 50 consensus examples with group discussion of disagreements, and (3) a final calibration phase where inter-annotator agreement was measured. Each annotator then independently scored responses on a 4-point scale designed to distinguish levels of creative quality:

- – 3 (Creative): Demonstrates genuine creativity, stylistic flair, and deep engagement; suitable for publication.
- – 2 (Competent): Well-structured and complete, but predictable and lacking in originality.
- – 1 (Formulaic): Technically correct but lacks creative engagement; follows a rigid template.
- – 0 (Elementary): Fundamentally flawed or off-topic.

The framework includes universal quality criteria and detailed genre-specific standards (see Appendix C for complete guidelines).

- • Statistical Validation and Final Pair Curation. A final preference pair was curated only if it met three strict criteria for reliability and validity. A pair was accepted only if:

- 1. It had directional agreement from at least 2 of the 3 annotators.
- 2. It showed a minimum score gap of ∆ ≥ 1.
- 3. It passed a check for absence of confounding factors, such as significant length disparities.

Finally, a separate team of bilingual experts performed cross-lingual validation on a sample of pairs to confirm that scoring standards were applied equivalently across both languages.

Chinese Dataset English Dataset

| |
|---|

345

350

300

261

250

SampleCount

205 208

200

184

150

120

109 105

104

100

47

50

29 25 23

13 11 11

0

Fiction Functional Documents

Funny Promotional & Communication

Non-Fiction Role-Playing Poetry Scriptwriting

Genre Categories

- Figure 3 Distribution of preference pairs across a sample of the 8 writing macro-categories for both English and Chinese in WritingPreferenceBench. The dataset maintains balanced coverage across diverse writing genres, with deliberate oversampling of underrepresented categories to ensure comprehensive evaluation of preference modeling capabilities.

##### 2.2 Dataset Statistics

WritingPreferenceBench comprises 1,800 human-validated preference pairs (1,200 English, 600 Chinese) that establish a new standard for evaluating subjective writing preferences. Unlike existing benchmarks that conflate objective correctness with aesthetic quality, our dataset isolates genuine stylistic preference through careful statistical design.

Compositional Structure and Coverage.

- Figure 3 reveals the deliberate architectural choices underlying our 51-category taxonomy. The dataset’s statistical properties reflect three critical design decisions:

- • Cross-lingual parity: While maintaining a 2:1 English-Chinese ratio due to annotator availability, we ensure equivalent statistical power across languages with a minimum of 20 pairs per category in each language.
- • Genre equilibrium: Each category contains 20-40 preference pairs (mean=35.3, std=7.2), a distribution engineered to prevent the genre collapse observed in web-scraped datasets.
- • Compositional diversity: The taxonomy spans 8 macro-categories with deliberate oversampling of traditionally underrepresented genres (e.g., poetry, scriptwriting) to stress-test models’ preference modeling capabilities beyond dominant web text distributions.

Statistical Validation of Subjective Quality Gaps. Figure 4 and Table 3 reveal the empirical signature of subjective preference in our dataset. The length distributions expose a critical phenomenon: chosen responses exhibit significantly higher variance (English: SD=1801.9 vs. 593.4; Chinese: SD=1967.5 vs. 1311.0) and right-skewness compared to rejected responses. This asymmetry reflects a fundamental property of creative excellence—while mediocrity converges toward formulaic patterns, creativity manifests across diverse scales.

Chosen Rejected

400

| |
|---|

300

Frequency

200

100

0

0 1K 2K 3K 4K 5K 6K

Response Length

(a) English dataset

140

Chosen

120

Rejected

100

Frequency

80

60

40

20

0

0 1K 2K 3K 4K 5K 6K

Response Length

(b) Chinese dataset

- Figure 4 Length distributions of chosen and rejected responses reveal the statistical signature of creative quality in WritingPreferenceBench. Distributions truncated at 6K for visualization; dashed lines indicate means.

The score distributions validate our annotation protocol’s effectiveness. The median scores (English: 3 vs.

- 2; Chinese: 3 vs. 1) align precisely with our rubric’s creative-competent and creative-formulaic boundaries, demonstrating that our benchmark captures the most informative preference contrasts.
- 3 Experiments

We evaluate 21 models on WritingPreferenceBench: 7 reward models and 14 language models serving as zero-shot judges. This section describes our evaluation protocols and experimental setup.

##### 3.1 Evaluation Protocols

- Protocol 1: Reward Model Scoring. For each preference pair (Rchosen,Rrejected), reward models assign scalar scores. A prediction is correct if RM(Rchosen) > RM(Rrejected). We compute accuracy as:

Accuracy =

1 N

N

i=1

I[RM(Rchosen(i) ) > RM(Rrejected(i) )]

where N denotes total preference pairs and I[·] is the indicator function.

- Protocol 2: Pairwise Preference Judgment. Language models receive both responses with instructions to select the preferred text based on creativity, style, and emotional resonance. We use deterministic decoding (T = 0) and extract preferences from model outputs. This protocol tests whether general-purpose models can perform zero-shot preference evaluation without specialized training.

- 3.2 Models Reward Models. We evaluate 7 models spanning different architectures and scales:

- • Sequence Classifiers: Nvidia/AceMath-7B-RM [19], RM-Mistral-7B [10], Skywork-Reward-Llama-3.18B [17], Skywork-Reward-Gemma-2-27B [17]
- • Generative RMs: RM-R1-DeepSeek-Qwen-7B, RM-R1-DeepSeek-Qwen-14B, RM-R1-Qwen2.5-7B [6, 13, 30]

Language Model Judges. We evaluate 14 models including reasoning-enhanced variants (Claude-4-{Opus, Sonnet}-thinking [1], Doubao-1.6-thinking [26], OpenAI-o3-high [21]) and standard models (Gemini-2.5{Flash, Pro} [29], DeepSeek-R1 [13], ByteDance-Seed-1.6 [28], Doubao-{1.5-Lite, 1.5-Pro, 1.6-flash} [5], Qwen-3-235B [30], OpenAI-o4-mini).

##### 3.3 Implementation Details

All experiments use the same prompt templates across models to ensure fair comparison. For reward models, we use the default inference configuration from their respective repositories. For LLM judges, we employ a standardized prompt format that presents both responses and requests a preference judgment with justification. We evaluate on the full WritingPreferenceBench dataset (1,200 English, 600 Chinese pairs) without subsampling.

4 Results

##### 4.1 Reward Model Performance

We evaluate seven reward models across WritingPreferenceBench. Following RewardBench [14], models divide into sequence classifiers (discriminative heads on language models) and generative reward models. However, our results reveal that the RM-R1 series [6] represents a distinct category—generative models that produce reasoning chains before preference judgments, diverging from both traditional classifiers and DPO-based approaches evaluated in RewardBench. Table 1 presents accuracy across eight writing genres in English and Chinese.

Sequence classifiers fail on subjective preference. Traditional sequence-based reward models achieve 52.7% mean accuracy across both languages (ranging from 46.8% to 62.6% on individual language subsets), statistically indistinguishable from random chance (binomial test, p > 0.05). This aggregate masks severe instability: all four models exhibit catastrophic failures on at least one genre, with Nvidia/AceMath-7B dropping to 18.2% on Chinese Poetry while reaching 61.5% on Role-Playing—a 43.3 percentage point swing within a single model. Three of four sequence classifiers fall below 50% on their weaker language, indicating systematic rather than random failures.

Generative architecture enables strong performance. Generative reward models achieve substantially higher accuracy than sequence classifiers. RM-R1-Qwen2.5-7B reaches 81.8% (EN)—the highest performance across all models and 30 percentage points above the sequence classifier mean. All three generative models exceed 50% accuracy, while three of four sequence classifiers fall below random chance. This architectural advantage persists across languages: best generative performance reaches 64.5% (ZH) versus 53.5% for sequence classifiers.

Scale improves stability, not just accuracy. Scaling from 7B to 14B in RM-R1-DeepSeek yields distinct benefits: accuracy improves (50.3%→62.6% ZH), but more critically, variance drops (9.8→5.5). This stability gain does not transfer to sequence classifiers—Skywork-Gemma-27B shows no improvement over 8B variants despite 3.4× parameters. The 14B model’s low variance (5.5) represents the most consistent performance across genres, suggesting scale enables robust preference representations in generative architectures.

Sequence classifiers exhibit catastrophic genre failure. All sequence classifiers demonstrate extreme performance swings: Nvidia/AceMath-7B ranges from 18.2% to 61.5% (43.3 percentage point gap), while Skywork-Gemma-27B varies from 21.7% to 81.8%. Mean within-model standard deviation reaches 10.1% for discriminative models versus 9.2% for generative. These genre-specific failures—often below 40% accuracy—indicate fundamental instability rather than minor variations.

Cross-lingual consistency reveals architectural robustness. Generative models maintain more consistent cross-lingual performance than sequence classifiers. RM-R1-DeepSeek-Qwen-14B achieves 62.6% (ZH) and 62.5% (EN), while sequence classifiers show larger gaps: Nvidia/AceMath-7B scores 53.5% (ZH) versus 46.8% (EN). This consistency in generative models, particularly at larger scales, suggests that reasoning-based architectures learn more language-agnostic preference representations.

##### 4.2 Language Model Judge Performance

Table 2 presents the performance of 14 state-of-the-art language models serving as zero-shot preference judges on WritingPreferenceBench, revealing systematic underperformance compared to specialized reward models.

- Table 1 Reward model accuracy (%) on WritingPreferenceBench by architecture. Colors: >70% , <50% . Best overall in bold.

Model Lang Func. Promo. Non-Fic. Fiction Funny Poetry Script Role Avg Std Sequence Classifiers (Discriminative with scalar output) Nvidia/AceMath-7B

ZH 56.7 50.5 55.3 54.9 52.3 18.2 54.6 61.5 53.5 11.2 EN 48.0 53.9 59.6 33.9 55.1 36.0 21.7 51.7 46.8 12.4

ZH 60.8 46.7 63.8 55.9 54.1 54.5 72.7 46.1 55.6 9.1 EN 65.2 60.0 54.8 62.9 64.9 72.0 78.3 44.8 62.6 10.1

RM-Mistral-7B

ZH 56.7 43.8 61.7 52.2 50.5 63.6 54.6 38.5 52.0 8.2 EN 53.6 56.3 60.6 49.0 52.2 56.0 65.2 41.4 53.1 7.3

Skywork-Llama-3.1-8B

ZH 50.8 54.3 55.3 53.3 40.4 81.8 45.5 53.9 51.2 11.3 EN 49.0 53.9 59.6 33.9 55.1 36.0 21.7 51.7 46.8 12.4

Skywork-Gemma-2-27B

Generative Reward Models (Reasoning before scoring) RM-R1-DeepSeek-Qwen-7B

ZH 50.8 45.7 57.5 56.5 45.0 27.3 36.4 46.2 50.3 9.8 EN 64.8 50.9 58.7 57.4 54.4 52.0 52.2 37.9 56.8 7.2

ZH 59.2 45.7 57.5 71.7 66.7 90.0 54.6 69.2 62.6 13.2 EN 71.3 61.5 63.5 58.6 59.0 64.0 52.2 65.5 62.5 5.5

RM-R1-DeepSeek-Qwen-14B

ZH 69.1 56.1 78.7 86.4 67.9 81.8 72.7 84.6 73.3 10.9 EN 79.3 76.0 91.4 89.6 72.9 92.0 85.9 65.5 81.8 9.5

RM-R1-Qwen2.5-7B

LLM judges systematically underperform reward models. General-purpose language models achieve mean accuracy of 53.9%, compared to 58.2% for reward models—a 4.3% degradation despite orders of magnitude more parameters. The best LLM judge (Doubao-1.5-Pro: 68.7% EN) remains 13.1% below the top generative reward model (RM-R1-Qwen2.5-7B: 81.8% EN). This gap persists across all model families and scales, indicating that zero-shot preference evaluation cannot match task-specific training.

Reasoning capabilities provide no systematic advantage. Models with explicit reasoning mechanisms show no consistent improvement over standard architectures. Claude-4-Opus-thinking achieves 61.0% (EN) while non-reasoning Doubao-1.5-Pro reaches 68.7%. Similarly, OpenAI-o3-high with advanced reasoning scores only 48.1%, performing worse than simpler models like Gemini-2.5-Flash (57.5%). The correlation between reasoning capability and preference accuracy is negligible (r=0.08, p>0.5), suggesting that chain-of-thought processing does not inherently improve subjective quality assessment.

Genre instability exceeds that of reward models. LLM judges exhibit extreme performance variance across genres, surpassing even sequence classifiers. Gemini-2.5-Pro ranges from 80.0% on English Poetry to 34.8% on Scriptwriting—a 45.2% gap. OpenAI-o3-high shows similar instability: 72.0% on Poetry versus 21.7% on Scriptwriting. Mean within-model standard deviation reaches 11.4%, with 9 of 14 models showing standard deviations exceeding 10%. This variance pattern suggests that LLMs rely on superficial genre markers rather than genuine quality assessment.

Cross-lingual performance reveals model-specific biases. LLM judges demonstrate inconsistent cross-lingual patterns. Doubao models maintain relative consistency (1.5-Pro: 62.5% ZH, 68.7% EN), while others show severe degradation: OpenAI-o3-high drops from 48.1% (EN) to 42.0% (ZH). These disparities do not correlate with known multilingual capabilities, suggesting that preference evaluation activates different model behaviors across languages.

Implications for LLM-as-judge paradigm. The systematic underperformance of LLM judges relative to specialized reward models challenges the widespread adoption of LLM-as-judge evaluation [32]. Mean accuracy of 53.9%—barely above random—indicates that zero-shot prompting cannot elicit reliable preference judgments for subjective tasks. The failure of reasoning-enhanced models further suggests that the limitation is not computational but representational: without explicit preference training, even advanced LLMs default to surface-level heuristics rather than genuine quality assessment.

- Table 2 Language model judge accuracy (%) on WritingPreferenceBench using pairwise preference evaluation. Colors: >70% , <50% . Best overall in bold.

Model Lang Func. Promo. Non-Fic. Fiction Funny Poetry Script Role Avg Std

ZH 42.1 32.2 52.5 59.9 38.5 45.5 36.4 38.5 45.5 9.4 EN 54.6 54.2 49.2 41.1 47.3 68.0 17.4 41.4 48.3 13.4

ByteDance-Seed-1.6

ZH 55.1 36.4 57.6 73.3 49.5 54.6 72.7 46.2 56.0 12.1 EN 65.7 64.3 64.1 60.1 54.2 64.0 43.5 51.7 61.0 7.3

Claude-4-Opus-thinking

ZH 46.7 38.1 62.7 65.7 48.6 54.6 63.6 46.2 52.8 9.9 EN 62.4 58.6 58.7 53.9 50.3 50.0 31.8 48.2 55.7 9.3

Claude-4-Sonnet-thinking

ZH 46.7 41.5 61.0 61.6 48.6 45.5 63.6 46.2 52.0 8.8 EN 57.4 61.7 43.8 40.8 42.9 72.0 17.4 51.7 49.3 15.4

DeepSeek-R1

- Doubao-1.5-Lite

ZH 44.9 42.4 64.4 62.8 44.0 36.4 72.7 38.5 51.5 13.0 EN 62.8 63.9 42.2 50.5 45.4 52.0 47.8 48.3 53.7 8.1

- Doubao-1.5-Pro

ZH 54.2 47.5 72.9 79.7 54.1 54.6 63.6 69.2 62.5 10.8 EN 74.4 74.5 64.8 71.0 57.1 68.0 56.5 58.6 68.7 7.2

- Doubao-1.6-flash

ZH 39.3 30.5 55.9 60.5 38.5 54.6 63.6 38.5 45.8 11.9 EN 57.0 53.7 49.2 43.6 46.3 52.0 30.4 44.8 49.3 7.7

- Doubao-1.6-thinking

ZH 46.7 35.6 66.1 73.8 43.1 72.7 54.6 46.2 54.2 13.9 EN 64.1 66.5 60.9 57.9 48.8 72.0 30.4 41.4 58.9 12.8

ZH 48.6 40.7 66.1 71.5 49.5 63.6 63.6 53.9 56.2 10.3 EN 64.9 61.7 61.7 55.5 47.3 68.0 39.1 48.3 57.6 9.6

Doubao-1.6-thinking-agent

ZH 47.7 34.8 61.0 68.0 45.0 63.6 63.6 38.5 52.2 12.1 EN 59.1 57.7 62.5 59.8 52.2 56.0 34.8 51.7 57.5 8.1

Gemini-2.5-Flash

ZH 53.3 44.9 71.2 80.2 56.9 72.7 72.7 61.5 62.7 11.3 EN 70.3 66.5 68.0 65.7 58.5 80.0 34.8 72.4 65.7 12.6

Gemini-2.5-Pro

ZH 43.0 33.9 54.2 50.6 35.8 45.5 63.6 38.5 43.5 9.9 EN 58.3 58.6 60.9 55.5 53.2 68.0 30.4 55.2 56.6 10.0

OpenAI-o4-mini

ZH 36.5 28.0 52.5 50.6 40.4 63.6 54.6 38.5 42.0 11.1 EN 55.4 45.8 54.7 46.7 41.0 72.0 21.7 41.4 48.1 14.0

OpenAI-o3-high

ZH 45.8 39.0 59.3 62.2 42.2 54.6 63.6 53.9 50.5 9.2 EN 57.9 46.7 45.3 43.0 40.5 64.0 17.4 41.4 46.4 13.3

Qwen-3-235B

#### 5 Discussion

Our findings reveal fundamental limitations in current preference learning paradigms when applied to subjective domains and expose a critical gap between model capabilities and genuine human aesthetic judgment.

A Performance Ceiling on Subjectivity. Even the best-performing models struggle to surpass a modest accuracy threshold on purely subjective tasks. The top bilingual reward model achieves only 62.5% accuracy, suggesting that current methods are more adept at identifying objective flaws (which we filtered out) than they are at capturing nuanced stylistic and creative preferences. This indicates a potential ceiling for architectures trained primarily on objective-centric data.

Explicit Reasoning is No Panacea. The systematic underperformance of advanced LLM judges compared to simpler, specialized reward models is a striking result. It challenges the prevailing assumption that enhanced reasoning capabilities, such as chain-of-thought, naturally lead to better alignment with complex human values. Our results suggest the problem is not one of logic but of representation; models lack the underlying framework to encode and weigh aesthetic qualities like originality, emotional resonance, or stylistic flair.

Preference Functions are Brittle and Unstable. The most concerning discovery is the extreme performance variance of individual models across genres. Swings of over 50 percentage points between categories like Poetry

and Scriptwriting reveal that models are not learning generalizable principles of "good writing." Instead, they appear to be memorizing brittle, genre-specific heuristics. This instability has profound implications for RLHF, as optimizing against such a volatile reward signal could introduce unpredictable and undesirable biases into model behavior, rewarding stylistic mimicry over genuine quality.

Cross-Lingual Performance Reflects Training Data Imbalances. Our cross-lingual analysis found no consistent patterns in performance differences between English and Chinese across models. Instead, the gaps appear to be model-specific artifacts, likely stemming from imbalances in training corpora rather than systematic differences in linguistic structure or aesthetic principles. For instance, RM-R1-Qwen2.5-7B achieves 81.8% on English but 73.3% on Chinese, while Nvidia/AceMath-7B shows the reverse pattern (53.5% Chinese, 46.8% English). This variability suggests that current models do not learn language-invariant preference representations, but instead encode language-specific biases from their pre-training and fine-tuning data. Achieving robust cross-lingual preference modeling will require training approaches that explicitly encourage language-agnostic aesthetic understanding.

#### 6 Related Work

Preference Learning and Evaluation Benchmarks. Modern preference learning originated with Christiano et al. [9], scaled through InstructGPT [23] and RLHF [9]. Subsequent benchmarks [3, 7, 12, 25] and comprehensive evaluations like RewardBench [14], AlpacaEval [16], and MT-Bench [32] measure conversation, reasoning, and instruction-following. However, these benchmarks conflate objective correctness with subjective preference. RewardBench achieves 95% on safety but cannot evaluate aesthetic judgment; MT-Bench measures factual accuracy, not creative quality. Our work reveals that models excelling on these benchmarks fail catastrophically (52.7% accuracy) when objective signals are neutralized.

Evaluating Creative and Subjective Writing. Creative writing evaluation faces inherent subjectivity challenges [4, 8, 15, 20, 27]. Early reference-based metrics (BLEU, ROUGE) fail for open-ended generation. Recent benchmarks make progress but retain critical limitations. LitBench [11] uses Reddit upvotes—confounding preference with popularity and timing—and covers only English. WritingBench [31] spans six domains but mixes subjective creative tasks with objective functional ones (Academic & Engineering, Finance & Business). AlignBench [18] evaluates Chinese LLM alignment but focuses on general capabilities rather than creative preference. WritingPreferenceBench advances beyond these by: (1) systematically neutralizing objective confounds through human validation, (2) providing cross-lingual coverage with consistent methodology, and (3) isolating purely subjective quality discrimination where prior benchmarks conflate multiple signals.

#### 7 Conclusion

We introduced WritingPreferenceBench, a benchmark isolating subjective writing preference through systematic neutralization of objective quality signals. Our empirical evaluation demonstrates that sequence-based reward models—the dominant RLHF architecture—achieve 52.7% accuracy on subjective preference tasks. In contrast, generative reward models incorporating explicit reasoning achieve 81.8% accuracy, suggesting that intermediate representations are necessary for subjective quality assessment. All evaluated models exhibit high variance across genres (σ=10.1-14.0%), with individual models ranging from 18.2% to 92% accuracy across categories, indicating reliance on genre-specific heuristics rather than generalizable preference functions.

These results have theoretical and practical implications for preference learning. The 30 percentage point performance gap between architectures challenges the direct preference optimization paradigm [25] and suggests that subjective domains require fundamentally different inductive biases than objective tasks. The failure of scale to improve performance (27B models underperform 7B variants) and the inability of reasoningenhanced LLMs to surpass task-specific training indicate that current scaling laws may not apply to subjective preference modeling. Future work should investigate hybrid architectures combining the computational efficiency of discriminative models with the representational capacity of generative reasoning, and develop training objectives that explicitly encourage genre-invariant preference learning.

#### Contributions and Acknowledgements

Multimodal Art Projection (M-A-P) is a non-profit open-source AI research community, ran by donation. The community members are working on research topics in a wide range of spectrum, including but not limited to the pre-training paradigm of foundation models, large-scale data collection and processing, and the derived applications on coding, reasoning and music generation.

Core Contributors (Equal Contribution)

- • Shuangshuang Ying, M-A-P
- • Yunwen Li, CUHK-Shenzhen and M-A-P
- • Xingwei Qu, ByteDance Seed, China and The University of Manchester

Contributors

- • Xin Li, Nanyang Technological University
- • Sheng Jin, Zhejiang University
- • Minghao Liu, 2077AI and M-A-P
- • Zhoufutu Wen, ByteDance Seed, China
- • Xeron Du, M-A-P
- • Tianyu Zheng, M-A-P
- • Yichi Zhang, Nanyang Technological University
- • Letian Ni, Beijing University of Posts and Telecommunications
- • Yuyang Cheng, Peking University
- • Zhenzhu Yang, M-A-P
- • Qiguang Chen, Harbin Institute of Technology
- • Jingzhe Ding, ByteDance Seed, China
- • Shengda Long, ByteDance Seed, China
- • Wangchunshu Zhou, OPPO
- • Jiazhan Feng, ByteDance Seed, China
- • Wanjun Zhong, ByteDance Seed, China

Advisors

- • Libo Qin, Central South University
- • Wenhao Huang, ByteDance Seed, China
- • Wanxiang Che, Harbin Institute of Technology
- • Chenghua Lin, The University of Manchester

Corresponding Authors

- • Ge Zhang, ByteDance Seed, China and M-A-P
- • Chenghua Lin, The University of Manchester

#### References

- [1] Anthropic. Introducing claude 4: Opus 4 and sonnet 4. https://www.anthropic.com/news/claude-4, 2025. Accessed: 2025-05-23.
- [2] Anthropic. Anthropic economic index: Understanding ai’s effects on the economy, 2025. URL https://www. anthropic.com/economic-index. Accessed: 2025-09-16.
- [3] Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022.

- [4] Jill Burstein, Daniel Marcu, and Kevin Knight. Finding the write stuff: Automatic identification of discourse structure in student essays. IEEE Intelligent Systems, 18(1):32–39, 2003.

- [5] ByteDance Seed. Doubao-1.5-pro: Exploring the balance between performance and reasoning. https://seed. bytedance.com/en/special/doubao_1_5_pro/, 2025. Accessed: 2025-01-22.
- [6] Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, et al. Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387, 2025.

- [7] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating llms by human preference. In Forty-first International Conference on Machine Learning, 2024.

- [8] Martin Chodorow, Joel Tetreault, and Na-Rae Han. Detection of grammatical errors involving prepositions. In Proceedings of the fourth ACL-SIGSEM workshop on prepositions, pages 25–30, 2007.

- [9] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

- [10] Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023.

- [11] Daniel Fein, Sebastian Russo, Violet Xiang, Kabir Jolly, Rafael Rafailov, and Nick Haber. Litbench: A benchmark and dataset for reliable evaluation of creative writing, 2025. URL https://arxiv.org/abs/2507.00769.
- [12] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.

- [13] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [14] Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating reward models for language modeling. arXiv preprint arXiv:2403.13787, 2024.

- [15] Xia Li, Minping Chen, Jianyun Nie, Zhenxing Liu, Ziheng Feng, and Yingdan Cai. Coherence-based automated essay scoring using self-attention. In China National Conference on Chinese Computational Linguistics, pages 386–397. Springer, 2018.

- [16] Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. https://github. com/tatsu-lab/alpaca_eval, 5 2023.
- [17] Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024.

- [18] Xiao Liu, Xuanyu Lei, Shengyuan Wang, Yue Huang, Zhuoer Feng, Bosi Wen, Jiale Cheng, Pei Ke, Yifan Xu, Weng Lam Tam, et al. Alignbench: Benchmarking chinese alignment of large language models. arXiv preprint arXiv:2311.18743, 2023.

- [19] Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling. arXiv preprint, 2024.

- [20] Eleni Miltsakaki and Karen Kukich. Automated evaluation of coherence in student essays. In Proceedings of LREC, volume 2000, 2000.

- [21] OpenAI. Openai o3-high (o3 / o3-mini high reasoning effort variant). https://openai.com/index/ introducing-o3-and-o4-mini/, April 2025. “High reasoning effort” configuration of o3 / o3-mini; see “Introducing OpenAI o3 and o4-mini” and system card.
- [22] OpenAI. How people are using chatgpt, 2025. URL https://openai.com/index/ how-people-are-using-chatgpt/. Accessed: 2025-09-15.
- [23] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

- [24] Alexander Pan, Kush Bhatia, and Jacob Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. arXiv preprint arXiv:2201.03544, 2022.

- [25] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

- [26] ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.

- [27] Yi Tay, Minh Phan, Luu Anh Tuan, and Siu Cheung Hui. Skipflow: Incorporating neural coherence features for end-to-end automatic text scoring. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

- [28] ByteDance Seed Team. Seed 1.6: A general-purpose multimodal model with adaptive thinking and a 256k context window. https://seed.bytedance.com/en/seed1_6, June 2025. Incorporates multimodal capabilities, Adaptive Chain-of-Thought (AdaCoT), supports 23B active parameters / 230B total parameters, 256K context. :contentReference[oaicite:0]index=0.
- [29] Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024. URL https://arxiv.org/abs/2403.05530.
- [30] Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/blog/ qwen2.5/.
- [31] Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, et al. Writingbench: A comprehensive benchmark for generative writing. arXiv preprint arXiv:2503.05244, 2025.

- [32] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

#### A Use of Large Language Models

In accordance with ICLR 2026 policies, we disclose that a large language model was used during the manuscript preparation process to polish and refine the text. The LLM assisted in improving sentence fluency, enhancing clarity of expression, and standardizing language to align with academic writing conventions. All original academic arguments, experimental design, data analysis, and logical structure were developed solely by the authors. The authors independently verified all factual claims and technical content, and take full responsibility for the accuracy and validity of all statements in this paper.

#### B Full Taxonomy of Writing Categories

The benchmark spans 51 distinct writing categories, which are grouped into the 8 high-level domains shown below. This comprehensive taxonomy ensures a diverse and representative evaluation of model capabilities across a wide spectrum of writing tasks.

Functional Documents

- • Abstract for Academic Paper
- • Experiment Report
- • Meeting Minutes
- • Resume / Cover Letter
- • Thank You / Apology Letter
- • Product Manual
- • Proposal / Plan
- • Interview Questions
- • Open Letter
- • Argumentative Essay
- • Eulogy / Memorial Text

Promotional & Communication Documents

- • Speech Transcript
- • Advertisement Copy / Marketing Email
- • Slogan / Tagline
- • Social Media Content
- • Blog Post
- • Product Review
- • Popular Science Article
- • Tutorial / Guide
- • Debate Script

Non-Fiction Writing

- • Prose / Essay
- • Biography
- • Travelogue

- • Book / Film / Music Review

Fiction

- • Fantasy / Magic
- • Science Fiction
- • Suspense / Mystery
- • Historical Story
- • Fairy Tale / Fable
- • Slice of Life Story
- • Emotional / Romance Story
- • Wuxia
- • Military Fiction
- • Historical Fiction (Costume)
- • Xuanhuan
- • Xianxia
- • Gaming Fiction
- • Sports Fiction
- • General Fiction / Story

Funny

- • ACGN Funny Literature / Doujin (Fan Fiction)
- • Fandom Funny Literature
- • Esports / Gaming Funny Literature
- • Hip-hop / Rap Culture Funny Literature
- • Internet Slang Systems
- • Anti-Mainstream Consumer Culture
- • Cross-national / Cross-lingual Funny Literature
- • Subculture Identity Expression
- • Role-Playing (as a sub-genre)
- • Funny Literature / Subculture

Poetry

• Poetry

Scriptwriting

• Play / Script

Role-Playing

• Role-Playing

#### C Genre-Specific Scoring Guidelines

This appendix details the comprehensive framework provided to our expert annotators for evaluating model responses. The process is designed to be rigorous and consistent, combining a general quality rubric with a detailed hierarchy of universal and genre-specific standards.

##### C.1 General Scoring Rubric (4-Point Scale)

Each response was assigned a holistic quality score from 0 to 3. The rubric was anchored with descriptive levels and analogies to everyday standards to ensure annotator calibration.

- • Score 3: Creative / Professional The response is creative, stylistically fluent, and feels natural. It is a complete, well-crafted article on par with professionally published work (e.g., in a literary magazine). It is original, engaging, and often exceeds the prompt’s expectations in a surprising way.
- • Score 2: Competent / Predictable The response is good overall but lacks originality. The structure is sound and the content addresses the prompt, but the narrative or arguments are predictable. This level is analogous to a well-written but standard university-level essay or a competent product manual.
- • Score 1: Formulaic / Flawed The response exhibits significant issues. It may be written in a language different from the one requested, or it follows a rigid, unnatural template (e.g., every paragraph starting with a subheading). The word choice can be awkward or inappropriately technical (e.g., using "quantum" in a non-scientific context). This is comparable to a middle-school-level essay.
- • Score 0: Incoherent / Irrelevant The response is fundamentally unusable. It is nonsensical, completely fails to address the prompt’s genre or topic, or consists mostly of a direct repetition of the query. This is analogous to an elementary-school or illiterate level of writing.

##### C.2 Universal Evaluation Criteria

Beyond the holistic score, annotators assessed responses against a set of universal criteria applicable to all forms of writing.

###### • Prompt Adherence and Intent:

- – Does the response satisfy all explicit constraints in the query (e.g., themes, content, word count)?
- – Does it avoid vague, grandiose statements and focus on the core task?
- – Is the overall reading experience fluent, not sacrificed for overly ornate or complex sentences?

###### • Structure and Coherence:

- – Is the overall structure complete and are paragraphs divided logically?
- – Is the line of reasoning clear and are the ideas logically self-consistent?
- – For narratives, is the pacing effective (i.e., a clear beginning, development, climax, and conclusion)?

###### • Content and Substance:

- – Is the content rich and specific, avoiding empty, generic statements?
- – Does the chosen material effectively support the overall theme or argument?
- – Where applicable, are environmental descriptions vivid and effective at creating the desired atmosphere?

###### • Language and Expression:

- – Is the language accurate, precise, and grammatically correct?
- – Is the expression clear and unambiguous?
- – Does the writing style match the requirements of the prompt, genre, and intended audience?

##### C.3 Genre-Specific Evaluation Criteria

To account for the diverse nature of writing, annotators also applied specific standards for each category. The following are representative examples.

###### C.3.1 Fiction (e.g., Sci-Fi, Fantasy, Mystery)

- • Characters: Are characters consistent throughout the narrative? Are their relationships (e.g., friendship, rivalry) authentic and do they drive the plot?
- • Narrative Technique: Does the author "show" the story through action, dialogue, and detail, rather than simply "telling" the reader what is happening?
- • Creativity: Does the story demonstrate originality in its premise, characters, or plot? Does it effectively use narrative devices like foreshadowing and callbacks?

###### C.3.2 Scriptwriting

- • Dialogue: Is the dialogue believable for the characters, reflecting their personality, background, and emotional state?
- • Action & Staging: Does the script include stage directions (e.g., tone, emotion, action) for dialogue? Does it incorporate elements like set design, sound effects, and props?
- • Motivation: Do the main characters have clear, understandable motivations that drive the plot forward?

###### C.3.3 Non-Fiction (e.g., Essays, Biographies, Reviews)

- • Accuracy: Are all factual claims, data, quotes, and historical details accurate?
- • Authenticity: Is the author’s emotion and experience conveyed in a genuine and credible manner?
- • Depth: Does the writing go beyond surface-level description to offer deeper analysis of causes, meanings, or connections?

###### C.3.4 Functional Documents (e.g., Resumes, Proposals, Memos)

- • Purpose: Is the core purpose of the document (e.g., to inform, persuade, request) immediately clear?
- • Completeness: Does the document include all information necessary to achieve its goal?
- • Format & Logic: Does it follow the conventional format for its type? For persuasive documents, are the arguments clear, well-supported, and logical?

###### C.3.5 Funny (e.g., Internet Memes, Copypasta) This category evaluates a model’s grasp of niche, often non-literal communication styles.

- • Form: Does the response deliberately break conventional logic for humorous or absurd effect?
- • Technique: Does it correctly use techniques specific to the subculture, such as puns, homophones, context-dependent slang, or "serious nonsense"?
- • Tone: Does it successfully capture a specific ironic or satirical tone, potentially with multiple layers of meaning?

#### D Dataset Statistics

Table 3 Distributional properties of preference pairs in WritingPreferenceBench.

Dataset Metric Type Mean (STD) Median

Chosen 1450.3 (1801.9) 961.5 Rejected 839.9 (593.4) 792.0

Length (words)

English

Chosen 2.913 (0.296) 3.000 Rejected 1.602 (0.553) 2.000

Quality Score

Chosen 1873.5 (1967.5) 1340.5 Rejected 1458.3 (1311.0) 1218.5

Length (words)

Chinese

Chosen 2.560 (0.589) 3.000 Rejected 1.115 (0.567) 1.000

Quality Score

#### E Examples of Benchmark Queries

To illustrate the nature of the tasks in WritingPreferenceBench, this section provides several examples of the expert-crafted queries given to the models. These queries are designed to be specific, evocative, and challenging, pushing models beyond generic text generation.

##### E.1 Example 1: Poetry Query:

Please help me write a modern poem on the theme of the old refrigerator in my grandmother’s kitchen. It no longer cools and is now used as a storage cabinet; there are faded stickers and an old shopping list still on its door. The poem needs to start with sensory details like its sound, its smell, and its appearance. It should be depicted as a guardian of family memories. Please use rhetorical devices like personification or metaphor to express a sense of nostalgia and affection for the old days.

##### E.2 Example 2: Product Review Query:

Write a professional product review for a high-end outdoor shell jacket. The article must be well-structured and centered on its core performance metrics. The review should include at least: 1. Design & Workmanship: Analyze the jacket’s fit and cut, fabric technology, seam sealing process, zipper configuration, and overall weight. 2. Core Functionality Test: Objectively evaluate its waterproofing, breathability, and windproofing performance in simulated or real-world conditions. 3. Details & Usability: Review the hood’s range of adjustment, the logic of the pocket layout, and the adjustment systems for the cuffs and hem. The article’s conclusion must clearly summarize the product’s pros and cons and, in conjunction with its price, provide clear purchasing advice and an analysis of suitable user groups.

###### E.3 Example 3: Funny Category: ACGN Abstract Literature / Doujin (Fan Fiction)

Query:

Write an abstract fanfiction piece set against the backdrop of the "Human Instrumentality Project" from Neon Genesis Evangelion. The "protagonist" of this piece is not a specific person, but the very moment in which all human consciousnesses dissolve, merge, and collide within the Sea

of LCL. The core theme is "the boundary between the Self and the Other," aiming to explore which is more terrifying: absolute loneliness or the loss of self through fusion. You do not need to construct a plot. Instead, use a fragmented, multi-vocal "stream-of-consciousness" style to weave together the internal monologues, memory fragments, and sensory perceptions of different characters—Shinji’s inferiority complex, Asuka’s arrogance, Rei’s emptiness—and iconic sensory details (like "the metallic taste of orange juice" or "the sweetness of watermelon") to form a chaotic yet harmonious sea of consciousness.

##### E.4 Example 4: Short Story Query:

Write a short story about a conflict between neighbors in a city. The protagonist is a young person who works from home and is constantly bothered by a strange, rhythmic noise coming from their new upstairs neighbor late at night. At the moment they can’t stand it anymore and decide to confront the neighbor, they discover an unexpected and poignant truth about the source of the noise. The core of the story is the dramatic turn caused by this revelation, aiming to explore the alienation, misunderstanding, and eventual reconciliation between people in a modern city.

##### E.5 Example 5: Argumentative Essay Query:

On our journey through life, we often face the choice between "looking back" and "moving forward." Some believe that dwelling on the past hinders progress, and thus one must resolutely "move forward." Others are convinced that by frequently "looking back" and drawing wisdom from past experiences and lessons, we can walk the future path more steadily. These two attitudes, seemingly contradictory, are in fact dialectically unified, jointly shaping the trajectory of our lives. Please write an argumentative essay titled "‘Looking Back’ and ‘Moving Forward’". Your viewpoint should be distinct and your essay well-structured. The article must not only discuss why we should "look back" and why we must "move forward," but more importantly, it must delve into how a balance and unity can be achieved between the two. You are required to cite at least one historical figure as positive or negative evidence and analyze it in conjunction with a contemporary social phenomenon or a personal experience. When narrating the case, you must include rich descriptive details that reflect the character’s internal journey and emotional changes when faced with the choice between "looking" and "moving." The essay should be no less than 800 words.

##### E.6 Example 6: Speech Query:

You are about to graduate after three years of high school and, as the student representative, you need to deliver a speech at the graduation ceremony. Your audience includes not only the classmates you’ve spent every day with, but also the hardworking teachers and the parents who have come to attend. Please write a speech manuscript centered on the theme of "Gratitude and Responsibility." The speech should not be a collection of empty slogans or a simple farewell. You must include two specific, detailed stories: first, recount the profound impact a particular teacher had on you, describing a teaching moment or interaction that you still remember vividly; second, share a story of friendship and mutual growth with your classmates. Please create your own title. The speech must be no less than 600 words, with a closing signed by "Graduate Representative, Wang Chen" and the date.

###### E.7 Example 7: Advertising Copy Category: Advertisement Copy / Marketing Email

Query:

Write a marketing email for the new "Pathfinder 30L" backpack from the outdoor brand "Nomad’s Gear." The email must use a vivid user story (e.g., a summit experience) to highlight the backpack’s benefits, such as being lightweight and durable, in order to spark the reader’s desire for adventure. A catchy subject line, an immersive story, and a clear call to action at the end are required.

##### E.8 Example 8: Biography Query:

Please help me write a biography of the Mexican painter Frida Kahlo, with a suggested title of "The Burning Thorn Bush: Frida’s Pain and Creation." The core theme of this biography should not be a simple chronological account of her life, but an in-depth exploration of "how pain became the core fuel for her artistic creation." You need to closely link and analyze key events in her life (such as the bus accident, her marriage to Diego Rivera, and her miscarriages) with her representative paintings. Provide a detailed interpretation of how she transformed physical disability and emotional turmoil into the powerful symbols and visual language of her artwork.

