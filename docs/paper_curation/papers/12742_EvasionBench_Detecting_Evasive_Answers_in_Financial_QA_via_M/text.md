# arXiv:2601.09142v2[cs.LG]4Feb2026

## EvasionBench: A Large-Scale Benchmark for Detecting Managerial Evasion in Earnings Call Q&A

Shijian MA1,†, Yan LIN2,†,∗, Yi YANG1 1Hong Kong University of Science and Technology 2University of Macau mas8069@foxmail.com, yanlin@um.edu.mo, imyiyang@ust.hk †Equal contribution ∗Corresponding author

### Abstract

We present EvasionBench, a comprehensive benchmark for detecting evasive responses in corporate earnings call question-and-answer sessions. Drawing from 22.7 million Q&A pairs extracted from S&P Capital IQ transcripts, we construct a rigorously filtered dataset and introduce a three-level evasion taxonomy: direct, intermediate, and fully evasive. Our annotation pipeline employs a Multi-Model Consensus (MMC) framework, combining dual frontier LLM annotation with a three-judge majority voting mechanism for ambiguous cases, achieving a Cohen’s Kappa of 0.835 on human inter-annotator agreement. We release: (1) a balanced 84K training set, (2) a 1K goldstandard evaluation set with expert human labels, and (3) Eva-4B, a 4-billion parameter classifier fine-tuned from Qwen3-4B that achieves 84.9% Macro-F1, outperforming Claude Opus 4.5, GPT-5.2, and Gemini 3 Flash. Our ablation studies demonstrate the effectiveness of multimodel consensus labeling over single-model annotation. EvasionBench fills a critical gap in financial NLP by providing the first large-scale benchmark specifically targeting managerial communication evasion.

### 1 Introduction

Evasive communication—responding to questions without truly answering them—is a pervasive phenomenon across human discourse. Politicians deflect uncomfortable questions in interviews (Bull and Mayer, 1993; Clayman, 2001), witnesses hedge in legal depositions (Bachenko et al., 2008), and corporate executives sidestep analyst inquiries during earnings calls (Hollander et al., 2010; Gow et al., 2021). Detecting such evasion has concrete implications: in politics, it undermines democratic accountability; in legal contexts, it obscures truthfinding; and in financial markets, evasive management communication predicts subsequent earnings misses and stock underperformance (Barcel-

los, 2025; Larcker and Zakolyukina, 2012).

Despite evasion’s ubiquity, it remains underexplored in NLP. Unlike sentiment analysis—which has canonical benchmarks like SST and Financial PhraseBank—or question answering with SQuAD, no large-scale benchmark exists for evasion detection. This gap stems from two challenges: (1) the inherently subjective nature of “evasiveness,” which lies on a spectrum from direct to fully evasive, and (2) the prohibitive cost of expert annotation at scale. While sentiment captures what is said, evasion detection addresses whether the question was actually answered—a fundamentally different, discourse-level phenomenon rooted in Gricean pragmatics (Grice, 1975).

Corporate earnings calls offer an ideal testbed for evasion research. They provide: (a) high-stakes, adversarial Q&A with clearly defined speaker roles (analysts vs. executives), (b) abundant publicly available transcripts spanning decades, and (c) welldefined question types (quantitative, temporal, binary, causal) that enable operationalization of “directness.” Prior work has examined non-answers in this domain (Gow et al., 2021; Bamber and Nappert, 2025), but datasets remain limited in scale.

In this paper, we introduce EvasionBench, the first large-scale benchmark for evasion detection, using earnings calls as our primary domain. While our data is domain-specific, we hypothesize that both our annotation framework and taxonomy could transfer to other adversarial Q&A settings (e.g., political interviews, legal depositions), though cross-domain validation remains future work. Our contributions are:

- • EvasionBench, a large-scale dataset comprising 84K balanced training samples and 1K human-validated evaluation samples, derived from 22.7M raw Q&A pairs spanning 2002– 2022.
- • A Multi-Model Consensus (MMC) frame-

work for scalable annotation that combines dual frontier LLM labeling with three-judge arbitration, achieving Cohen’s Kappa of 0.835 against human annotators.

- • Eva-4B, a 4-billion parameter classifier that achieves 84.9% Macro-F1, outperforming Claude Opus 4.5, GPT-5.2, and Gemini 3 Flash on this task.
- • Comprehensive analysis demonstrating that multi-model consensus labeling significantly outperforms single-model annotation, with ablations showing +4.3 pp Macro-F1 improvement.

### 2 Related Work

#### 2.1 Evasion Across Domains

Evasive communication has been studied extensively across multiple fields, though NLP benchmarks remain scarce. In political communication, Bull and Mayer (1993) identified 43 distinct techniques politicians use to avoid answering questions, while Clayman (2001) distinguished between overt evasion (explicit refusal) and covert evasion (appearing to answer while sidestepping the core). Bavelas et al. (1990) theorized that equivocation arises from avoidance-avoidance conflicts where all direct answers carry costs—a framework directly applicable to corporate executives facing analyst scrutiny.

In legal and forensic contexts, researchers have developed computational approaches to detect deceptive language in testimony (Bachenko et al., 2008; Vrij, 2008). Newman et al. (2003) identified linguistic markers of deception using LIWC, achieving 61–67% classification accuracy. Crucially, evasion differs from deception: while deception involves false statements, evasion represents strategic non-responsiveness where speakers appear cooperative but avoid the question core (PérezRosas and Mihalcea, 2015).

From a pragmatic perspective, evasion can be analyzed as a violation of Grice’s maxim of Relation (Grice, 1975), often realized through hedging devices (Lakoff, 1973; Hyland, 1998) and indirect speech acts (Searle, 1975). Brown and Levinson (1987) connected such indirectness to facethreatening act mitigation, explaining why speakers strategically employ vague language.

Despite this rich theoretical foundation across disciplines, NLP lacks a canonical benchmark for

evasion detection—unlike sentiment (Socher et al., 2013), stance (Mohammad et al., 2016), or factual QA (Rajpurkar et al., 2016). Our work addresses this gap by providing the first large-scale, systematically annotated evasion benchmark.

#### 2.2 Evasion in Financial Communication

Research on managerial evasion during earnings calls has grown substantially in accounting and finance. Hollander et al. (2010) provide empirical evidence that managers strategically choose what to disclose, with silence interpreted negatively by investors. Gow et al. (2021) find that approximately 11% of analyst questions receive non-answers, with product-related questions frequently evaded in competitive environments. Bamber and Nappert (2025) offer theoretical frameworks for understanding why managers evade questions, while Barcellos (2025) examine how investor suspicion interacts with evasive communication. Nuaimi et al. (2025) propose a psychological taxonomy for evasive answers; however, their dataset relies solely on single-model LLM annotation without human validation, raising concerns about label reliability and potential model-specific biases.

#### 2.3 Financial NLP

Financial text analysis has evolved from lexiconbased methods (Loughran and McDonald, 2011) to neural approaches. Araci (2019) introduced FinBERT for financial sentiment analysis, demonstrating the value of domain-specific pretraining. Koval et al. (2023) predict earnings surprises from conference call transcripts using LLMs. Huang et al. (2014) show that managers strategically manage linguistic tone. However, these works focus on sentiment or prediction rather than discourse-level phenomena like evasion, which requires assessing whether a response actually addresses the question asked.

#### 2.4 Scalable Annotation with LLMs

The prohibitive cost of expert annotation has driven interest in LLM-based labeling approaches. Using LLMs as evaluators has gained traction for scalable assessment (Zheng et al., 2023), though Chen et al. (2024) identify systematic biases including position bias and verbosity preference.

Several methodologies address annotation quality and consistency. Self-consistency (Wang et al., 2022) samples multiple reasoning paths from a single model. RLAIF (Lee et al., 2023) uses AI

###### Work Scale Labels Human Multi-M

Gow et al. (2021) 2.1K 2 ✓ – Nuaimi et al. (2025) 12K 7 – – EvasionBench 85K 3 ✓ ✓

Table 1: Comparison with prior evasion datasets. Labels = number of categories; Human = human validation; Multi-M = multi-model consensus annotation.

feedback to scale preference learning. Tri-training (Zhou and Li, 2005) leverages three classifiers to label unlabeled data through agreement. Recent surveys on LLM-based annotation (Tan et al., 2024) validate consensus-based approaches for improving label quality.

Our MMC framework draws on these insights, combining multiple heterogeneous frontier LLMs (rather than homogeneous classifiers) with randomized presentation order to mitigate individual model biases. This design choice is motivated by our finding that different LLMs exhibit distinct labeling tendencies (Section 5).

#### 2.5 Summary and Positioning

- Table 1 synthesizes prior work on evasion detection in financial communication. Three gaps emerge:

(1) existing datasets lack scale—the largest prior corpus contains fewer than 15K samples with limited human validation; (2) annotation often relies on single-model LLM labeling without consensus mechanisms, introducing systematic bias; (3) no prior work provides both large-scale training data and rigorously human-validated evaluation sets. EvasionBench addresses all three limitations.

3 Task Definition and Taxonomy

#### 3.1 Problem Formulation

Given a question q posed by a financial analyst and a response a from corporate management, the task is to classify the response into one of three evasion levels: direct, intermediate, or fully evasive.

#### 3.2 Evasion Taxonomy

Prior taxonomies range from binary classifications (evasive vs. non-evasive; Gow et al., 2021) to finegrained psychological categories (Nuaimi et al., 2025 propose seven types including “explicit refusal,” “deflection,” and “information flooding”). We adopt a three-level ordinal scale for three reasons:

- 1. Empirical grounding: Pilot annotation with five levels yielded low inter-annotator agreement (κ < 0.5); collapsing to three restored reliability (κ = 0.83).
- 2. Actionable granularity: Binary classification loses the distinction between “partial answer” and “complete deflection”—information critical for downstream applications like investor alerting.
- 3. Alignment with Gricean pragmatics: Our levels map to degrees of Relation maxim violation (Grice, 1975): full adherence (direct), partial violation (intermediate), and complete violation (fully evasive).

Our taxonomy definitions, refined through iterative pilot annotation:

Direct The response explicitly and completely addresses the question core. Typical features include specific figures, clear yes/no stances, or direct explanations.

Intermediate The response provides related context but sidesteps the specific ask. Indicators include hedging language (“I think,” “it’s possible”), conditional framing, or answering adjacent topics.

Fully Evasive The question is ignored, explicitly refused, or the response is entirely off-topic. Examples include explicit refusal (“we don’t provide that granularity”), information flooding, or silent pivoting.

#### 3.3 Illustrative Examples

Table 2 presents representative examples from EvasionBench for each evasion category, illustrating the subtle distinctions our taxonomy captures.

##### 3.4 Question Core Types We identify four primary question core types:

- • Quantitative: Specific numbers, percentages, or magnitudes
- • Temporal: Timelines, dates, or cadences
- • Binary: Clear yes/no regarding plans or status
- • Causal/Directional: Reasons for trends or future outlook

Label Question Answer Direct “What exactly did you do with respect to those em-

“No. So it was having good support for them, having coaches and supervisors... We’ve put recruiters in every district office... We’ve got better onboarding and training programs...”

ployees that dropped the turnover so much? Was it the wage increase?”

Interm. “How are you thinking about your balance sheet and opportunities in the space given joint ventures, M&A?”

“Yes, you’re mentioning things that we internally think about a lot... we have a very strong balance sheet with positive net cash. So I think we’re in a very strong position and we continue to evaluate opportunities...”

Fully Ev. “You mentioned an exciting pipeline of new products.

“Well, I probably can’t give you too much of a sneak preview. But all I can say is that historically, our business has been great at planning ahead... But you will have to see it when it comes out.”

Could you provide us with a sneak preview?”

- Table 2: Representative examples from EvasionBench. Direct: Manager explicitly lists specific actions. Intermediate: Response discusses balance sheet but avoids the M&A question. Fully Evasive: Manager explicitly refuses to preview products.

### 4 Data Collection and Filtering

#### 4.1 Raw Data Source

We source data from the S&P Capital IQ Full Text Dataset, containing 22.7M Q&A pairs from 1.38M earnings call transcripts spanning 420K unique speakers. After three-stage filtering, 11.27M highquality pairs remain.

- 4.2 Three-Stage Quality Filtering We apply a rigorous filtering pipeline:

- Stage 1: Q&A Pair Extraction Identify analyst questions (Type 3) and management answers (Type

- 4) with sequential component ordering. Remove operator instructions and pleasantries.

- Stage 2: Quality Filtering Questions must contain “?”; answers must exceed 30 characters; no transcription markers ([indiscernible], [ph], [inaudible]).
- Stage 3: Substantial Content Selection Combined question and answer length ≥ 500 characters.

After filtering, 11.27M Q&A pairs (49.6%) remain.

- 5 Multi-Model Consensus Annotation

- 5.1 Framework Overview

Our Multi-Model Consensus (MMC) framework leverages agreement between frontier LLMs as a signal of annotation reliability. Figure 1 illustrates the complete pipeline.

#### 5.2 Stage I: Dual LLM Annotation

We employ Claude Opus 4.5 and Gemini 3 Flash as primary annotators, each independently labeling samples with structured prompts.

#### 5.3 Stage II: Consensus Detection

Samples where both models agree form the consensus set. Disagreement samples proceed to arbitration.

#### 5.4 Stage III: Three-Judge Arbitration

For disagreement cases (3,645 samples, 16.1% of total), three judge models (Claude Opus 4.5, Gemini 3 Flash, GPT-5.2) independently evaluate which original annotation is more accurate. Majority voting determines the final label.

Why Multi-Model Consensus? Figure 2 reveals systematic differences in judge tendencies: Opus prefers direct labels (53.3%), Gemini assigns more fully_evasive (23.5%), while GPT-5.2 favors intermediate (56.7%). This demonstrates that singlemodel annotation introduces systematic bias, motivating our multi-model consensus approach.

Anti-Bias Mechanism To eliminate position bias (primacy effect), we randomize which model’s prediction appears first in the judge prompt (seed=42 for reproducibility).

#### 5.5 Final Dataset Statistics

Table 3 presents the complete EvasionBench dataset statistics. The training data spans 2002– 2022 with 8,081 unique companies, ensuring broad coverage of corporate communication patterns.

- Figure 1: Multi-Model Consensus (MMC) annotation pipeline. Stage I: Data distillation from raw transcripts. Stage II: Dual LLM annotation with Claude Opus 4.5 and Gemini 3 Flash. Stage III: Three-judge arbitration for disagreement cases. Stage IV: Final balanced datasets.

| |44<br><br>32.2%|53.3%<br><br>.4%|32.<br><br>56.|36.2%<br><br>1%<br><br>7%|10.5%<br><br>23.5%<br><br>11.1%<br>|
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 20 40 60 80 100

Label Distribution (%)

Opus 4.5

Gemini 3 Flash

GPT-5.2

Direct

Intermediate Fully Evasive

- Figure 2: Label distribution across three judges on 3,645 disagreement samples. Each model exhibits distinct tendencies: Opus (conservative, 53% direct), Gemini (strict, 24% fully evasive), GPT-5.2 (balanced, 57% intermediate).

Dataset Samples Comp. Q len A len Train-60K 60,000 6,943 77.5 149.8 Train-24K 24,000 4,597 70.0 136.3 Total Train 84,000 8,081 75.4 145.9 Gold-1K Eval 1,000 319 72.0 131.3

Table 3: EvasionBench dataset statistics. All splits are balanced (33.3% per class). Comp. = unique companies; Q/A len = avg words.

Metric Value Total Samples 100 Agreement Count 89 (89.0%) Cohen’s Kappa 0.835 Macro-F1 88.99%

### 6 Inter-Annotator Agreement

To validate annotation quality, a second human annotator independently labeled a balanced subset of 100 samples from the Gold 1K evaluation set.

The Cohen’s Kappa of 0.835 indicates “Almost Perfect” agreement according to Landis and Koch (1977). Notably, 10 of 11 disagreements involve the intermediate class, confirming it as the most ambiguous category.

### 7 Eva-4B Model

#### 7.1 Base Model Selection

We select Qwen3-4B-Instruct-2507 (Qwen Team, 2025) as our base model due to its strong instruction-following capability (IFEval: 83.4%)

Table 4: Inter-annotator agreement statistics.

and efficient parameter count enabling practical deployment.

7.2 Two-Stage Training Pipeline Figure 3 illustrates our two-stage training approach.

- Stage 1: Consensus Training Full fine-tuning on 60K consensus samples where both frontier LLMs agreed. Training uses MS-SWIFT framework with 2 epochs, learning rate 2e-5, and bfloat16 precision.
- Stage 2: Judge-Refined Training Continued fine-tuning on 24K samples including arbitrated disagreement cases. This stage incorporates the

- Figure 3: Two-stage training pipeline for Eva-4B. Stage 1 trains on 60K consensus data to obtain Eva-4B (Consensus). Stage 2 continues training on 24K samples with either three-judge majority labels (Full) or Opus-only labels (Opus Only).

harder boundary cases resolved through threejudge voting.

- 7.3 Model Variants We train three variants for ablation:

- • Eva-4B (Consensus): Stage 1 only
- • Eva-4B (Opus Only): Stage 1 + Stage 2 with Opus labels
- • Eva-4B (Full): Stage 1 + Stage 2 with majority voting

8 Experiments

- 8.1 Evaluation Setup

We evaluate 12 models on the Gold 1K evaluation set:

- • Closed-source: Claude Opus 4.5, GPT-5.2, Gemini 3 Flash
- • Open-source: GLM-4.7, Qwen3-Coder, MiniMax-M2.1, Kimi-K2, DeepSeek-V3.2
- • Eva-4B variants: Full, Opus Only, Consensus
- • Base model: Qwen3-4B (before fine-tuning)

Model M-F1 F1-D F1-I F1-E Eva-4B (Full) 84.9 82.2 80.1 92.4 Gemini 3 Flash 84.6 84.6 78.3 91.0 Claude Opus 4.5 84.4 82.4 79.3 91.5 GLM-4.7 82.9 84.4 74.7 89.6 Eva-4B (Cons.) 81.4 79.4 75.0 89.7 GPT-5.2 80.9 75.4 76.1 91.2 Eva-4B (Opus) 80.6 77.6 73.9 90.3 Qwen3-Coder 78.2 72.1 72.4 90.0 Qwen3-4B (Base) 34.3 7.3 33.3 62.3

Table 5: Model performance on Gold 1K. M-F1 = Macro-F1; F1-D/I/E = F1 for Direct/Intermediate/Evasive. Bold = best in column. All models struggle most with Intermediate (F1-I).

100

Qwen3-4B (Base)

80

Macro-F1(%)

Eva-4B (Consensus)

60

Eva-4B (Opus Only)

40

Eva-4B (Full)

20

0

Figure 4: Ablation study: Macro-F1 comparison of Eva4B variants. Full fine-tuning with three-judge consensus achieves 84.9%, a +50.6 pp improvement over the base model.

#### 8.2 Main Results

Table 5 presents the full evaluation results with per-class F1 scores.

#### 8.3 Ablation Study

Figure 4 shows the ablation results. Our ablation demonstrates the value of multi-model consensus:

- • Base → Consensus: +47.1 pp Macro-F1
- • Consensus → Full: +3.5 pp Macro-F1
- • Full vs. Opus Only: +4.3 pp Macro-F1

The three-judge majority voting (Full) outperforms single-model labeling (Opus Only), validating the MMC approach.

Training Dynamics Figure 5 reveals a striking difference in training convergence. Eva-4B (Full) achieves a final loss of 0.007, while Eva-4B (Opus Only) converges to 0.56—an 80× difference. This suggests that three-judge consensus labels produce more consistent training signals, while singlemodel labels contain noise that impedes learning. This finding provides strong evidence for the effectiveness of the MMC annotation framework.

Stage 1: Consensus Training (60K)

Stage 2: Judge Training (24K)

3.0

Eva-4B (Consensus) Stage 1

Eva-4B (Full) Stage 2

Eva-4B (Opus Only) Stage 2

- 0

- 1

- 2

- 3

2.5

2.0

Loss

Loss

1.5

1.0

0.5

0.0

0 1000 2000 3000 Steps

0 500 1000 1500 Steps

- Figure 5: Training loss curves for Eva-4B variants. Stage 2 training shows dramatically different convergence: Eva-4B (Full) with three-judge labels reaches loss 0.007, while Eva-4B (Opus Only) plateaus at 0.56, indicating noisier single-model annotations.

|251 (75.1%)| |79 (23.7%)| |4 (1.2%)| |
|---|---|---|---|---|---|
|23 (6.9%)| |[Figure 1]<br><br>292 (87.7%)| |18 (5.4%)| |
|3 (0.9%)| |25 (7.5%)| |305 (91.6%)| |
| | | | | | |

Direct Intermediate Fully Evasive

Predicted

Direct

Intermediate

Fully Evasive

Gold

- Figure 6: Confusion matrix for Eva-4B (Full) on the Gold 1K evaluation set. The model achieves 84.8% accuracy with most errors occurring between adjacent classes.

#### 8.4 Error Analysis

- Figure 6 shows the confusion matrix for Eva-4B (Full). Analysis reveals:

- • 52.0% of errors: direct → intermediate
- • Adjacent class confusion accounts for 95.4% of errors
- • Fully evasive is easiest to detect (F1: 92.4%)

### 9 Discussion

Why is Intermediate difficult? The intermediate class represents subtle evasion where executives appear to address questions while actually sidestepping the core ask. This ambiguity challenges both humans (IAA disagreements) and models.

Qualitative Error Analysis We manually examined Eva-4B’s 152 errors on the Gold 1K set. Three patterns emerge:

#### (1) Hedging-induced over-prediction of eva-

sion. The dominant error type (52%) involves classifying direct responses as intermediate. These cases typically feature executives providing clear answers but using hedging language (“we do expect,” “soon”) that triggers false evasion signals. For example, when asked “Is that still expected to occur?” the response “We do expect it to occur soon” is gold-labeled direct but predicted intermediate—the model mistakes temporal vagueness for evasion.

#### (2) Qualitative vs. quantitative directness.

When analysts ask quantitative questions (“how much,” “what percentage”), executives sometimes provide qualitative explanations listing specific actions without numbers. The model classifies these as direct due to their specificity, while human annotators label them intermediate for failing to address the numeric core.

(3) Shared difficulty across models. Only 10.5% of Eva-4B errors are unique to our model; 33.6% of error samples are misclassified by 5–6 of the top models, suggesting these represent genuinely ambiguous cases at the boundary of human annotation reliability.

Position Bias in LLM-as-Judge To validate our anti-bias mechanism, we conducted a controlled experiment comparing fixed-position vs. randomizedposition judge assignments on 5,541 samples. Figure 7 shows the results: randomization increased Opus’s win rate from 63.5% to 68.6% (+5.1%), demonstrating substantial position bias in LLM-asjudge settings. This finding validates the necessity of randomized presentation order in our methodology.

Practical Applications EvasionBench enables: (1) automated screening for investor relations, (2) regulatory monitoring of corporate disclosure quality, (3) academic research on strategic communication.

### 10 Conclusion

We present EvasionBench, the first large-scale benchmark for detecting managerial evasion in earnings calls. Our Multi-Model Consensus framework provides a scalable approach to annotation that outperforms single-model labeling. Eva-4B demonstrates that a 4B parameter model can match or exceed frontier LLMs on this task when trained with high-quality consensus data. We release all

80

Fixed Position

Randomized Position

70

60

WinRate(%)

50

40

30

20

10

0

Opus Gemini

- Figure 7: Position bias analysis: Fixed position (always Opus first) vs. randomized position. Randomization reveals a +5.1% win rate shift, confirming position bias in LLM judges.

data, models, and code to facilitate future research.

### Limitations

Domain Specificity Our dataset and models are trained exclusively on earnings call transcripts. Generalization to other domains (political interviews, legal depositions) requires further validation.

Annotation Scale While our Gold 1K set is validated, single-annotator labeling with 100-sample IAA verification represents a limitation. Largerscale human annotation would strengthen validity.

Temporal Coverage Our data spans 2002-2022. Language patterns and evasion strategies may evolve, requiring periodic updates.

English Only The current benchmark covers only English-language transcripts from U.S.-listed companies.

### Ethics Statement

This research uses publicly available earnings call transcripts. No private or sensitive personal information is included. The evasion detection system is intended for research and analytical purposes; we caution against using model outputs as sole evidence for legal or regulatory actions. The dataset will be released under appropriate academic licenses.

### References

Dogu Araci. 2019. FinBERT: Financial sentiment analysis with pre-trained language models. arXiv preprint arXiv:1908.10063.

Joan Bachenko, Eileen Fitzpatrick, and Michael Schonwetter. 2008. Verification and implementation of language-based deception indicators in civil and criminal narratives. In Proceedings of the 22nd International Conference on Computational Linguistics (COLING), pages 41–48.

Matthew Bamber and Pier-Luc Nappert. 2025. Can we explain managerial non-answers during conference call Q&As? Contemporary Accounting Research, 42(2):1079–1105.

Leonardo P. Barcellos. 2025. Overcoming deceptive evasions in earnings calls: The role of investor suspicion. Management Science. Forthcoming.

Janet Beavin Bavelas, Alex Black, Nicole Chovil, and Jennifer Mullett. 1990. Equivocal Communication. Sage Publications, Newbury Park, CA.

Penelope Brown and Stephen C. Levinson. 1987. Politeness: Some Universals in Language Usage. Studies in Interactional Sociolinguistics. Cambridge University Press.

Peter Bull and Kate Mayer. 1993. How not to answer questions in political interviews. Political Psychology, 14(4):651–666.

Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. 2024. Humans or LLMs as the judge? a study on judgement biases. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8301–8327, Miami, Florida, USA. Association for Computational Linguistics.

Steven E. Clayman. 2001. Answers and evasions. Language in Society, 30(3):403–442.

Ian D. Gow, David F. Larcker, and Anastasia A. Zakolyukina. 2021. Non-answers during conference calls. Journal of Accounting Research, 59(4):1349– 1384.

H. Paul Grice. 1975. Logic and conversation. In Peter Cole and Jerry L. Morgan, editors, Syntax and Semantics 3: Speech Acts, pages 41–58. Academic Press, New York.

Stephan Hollander, Maarten Pronk, and Erik Roelofsen. 2010. Does silence speak? an empirical analysis of disclosure choices during conference calls. Journal of Accounting Research, 48(3):531–563.

Xuan Huang, Siew Hong Teoh, and Yinglei Zhang.

2014. Tone management. The Accounting Review, 89(3):1083–1113.

Ken Hyland. 1998. Hedging in Scientific Research Articles, volume 54 of Pragmatics & Beyond New Series. John Benjamins.

Ross Koval, Nicholas Andrews, and Xifeng Yan. 2023. Forecasting earnings surprises from conference call transcripts. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8197–8209, Toronto, Canada. Association for Computational Linguistics.

George Lakoff. 1973. Hedges: A study in meaning criteria and the logic of fuzzy concepts. Journal of Philosophical Logic, 2(4):458–508.

J. Richard Landis and Gary G. Koch. 1977. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174.

David F. Larcker and Anastasia A. Zakolyukina. 2012. Detecting deceptive discussions in conference calls. Journal of Accounting Research, 50(2):495–540.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. 2023. RLAIF: Scaling reinforcement learning from human feedback with AI feedback. arXiv preprint arXiv:2309.00267.

Tim Loughran and Bill McDonald. 2011. When is a liability not a liability? textual analysis, dictionaries, and 10-Ks. The Journal of Finance, 66(1):35–65.

Saif Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu, and Colin Cherry. 2016. SemEval-2016 task 6: Detecting stance in tweets. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pages 31– 41, San Diego, California. Association for Computational Linguistics.

Matthew L. Newman, James W. Pennebaker, Diane S. Berry, and Jane M. Richards. 2003. Lying words: Predicting deception from linguistic styles. Personality and Social Psychology Bulletin, 29(5):665–675.

Khaled Al Nuaimi, Gautier Marti, Alexis Marchal, and Andreas Henschel. 2025. Detecting evasive answers in financial Q&A: A psychological discourse taxonomy and lightweight baselines. In Proceedings of The 10th Workshop on Financial Technology and Natural Language Processing, pages 191–196.

Verónica Pérez-Rosas and Rada Mihalcea. 2015. Experiments in open domain deception detection. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1120– 1125.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

John R. Searle. 1975. Indirect speech acts. In Peter Cole and Jerry L. Morgan, editors, Syntax and Semantics 3: Speech Acts, pages 59–82. Academic Press, New York.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.

Zhen Tan and 1 others. 2024. Large language models for data annotation and synthesis: A survey. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Miami, Florida, USA. Association for Computational Linguistics.

Aldert Vrij. 2008. Detecting Lies and Deceit: Pitfalls and Opportunities, 2nd edition. Wiley-Interscience.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS).

Zhi-Hua Zhou and Ming Li. 2005. Tri-training: Exploiting unlabeled data using three classifiers. IEEE Transactions on Knowledge and Data Engineering, 17(11):1529–1541.

### A Training Hyperparameters

###### Parameter Stage 1 Stage 2

Framework MS-SWIFT MS-SWIFT Training Type Full Full Base Model Qwen3-4B Eva-4B-Cons. Dataset Size 60K 24K Epochs 2 2 Learning Rate 2e-5 2e-5 Batch Size (per GPU) 8 8 Gradient Accumulation 2 1 Effective Batch Size 32 32 Max Length 2500 2048 Precision bfloat16 bfloat16 Warmup Ratio 3% 3%

Table 6: Training hyperparameters for two-stage finetuning.

### B Full Model Results

Table 7 presents complete evaluation results for all 12 models on the Gold 1K evaluation set.

# Model Cat. Acc M-F1 F1-D F1-I F1-E

- 1 Eva-4B (Full) Eva 84.8 84.9 82.2 80.1 92.4
- 2 Gemini 3 Flash Closed 84.6 84.6 84.6 78.3 91.0
- 3 Claude Opus 4.5 Closed 84.1 84.4 82.4 79.3 91.5
- 4 GLM-4.7 Open 83.1 82.9 84.4 74.7 89.6
- 5 Eva-4B (Cons.) Eva 81.0 81.4 79.4 75.0 89.7
- 6 GPT-5.2 Closed 80.8 80.9 75.4 76.1 91.2
- 7 Eva-4B (Opus) Eva 80.6 80.6 77.6 73.9 90.3
- 8 Qwen3-Coder Open 78.0 78.2 72.1 72.4 90.0
- 9 MiniMax-M2.1 Open 71.8 71.3 72.2 59.6 82.1
- 10 DeepSeek-V3.2 Open 66.7 66.9 61.4 64.3 75.0
- 11 Kimi-K2 Open 67.8 66.7 66.8 53.6 79.6
- 12 Qwen3-4B (Base) Base 42.3 34.3 7.3 33.3 62.3

Table 7: Full evaluation results for all 12 models. Cat. = Category (Eva = Eva-4B variants, Closed = closedsource, Open = open-source, Base = base model). All metrics in %. Bold = best in column.

