# arXiv:2510.09116v2[cs.CL]13Oct2025

## DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel Translation

BLEU scores cannot fully evaluate translation quality, especially in complex cases.

[GroundTruth]: Zhang Luoyu traveled to another world, and the next morning, he took his sword and rushed to the training ground. [Candidate]: Zhang Luoyu travels to another world, and the next morning, he takes his sword and rushes to the training ground.

###### Tense Consistency

[Ground Truth]:This person was incredibly meticulous and could still find the correct direction at a glance under heavy rain.

Idiom Translation

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

BLEU

BLEU

张骆宇穿越到了异世界， 第二天早上，他拿着剑冲 向了练武场。

[Figure 5]

此人竟然如此心细如发，在 这等雨水冲刷之下，居然能 一眼就找出正确的方向！

[Figure 6]

[Candidate]: This person was incredibly

[Figure 7]

[Figure 8]

0.61/1.0

meticulous like a hair, and could still find the correct direction at a glance even under heavy rain.

Enze Zhang1,2*, Jiaying Wang2*, Mengxi Xiao1,2, Jifei Liu2, Ziyan Kuang2,3, Rui Dong5, Eric Dong6, Sophia Ananiadou4, Min Peng1,2, Qianqian Xie1,2†

0.65/1.0

😡: word-forword translation; incomprehensible

Lexical Ambiguity

[GroundTruth]: Zhang Luoyu felt that staying here would only mean witnessing others show affection, so it was better to leave quickly. [Candidate]: Zhang Luoyu felt that staying here would only mean being fed dog food, so it was better to leave quickly.

[GroundTruth]: Yes,, she has been saved by a death God-like man, and even agreed to whatever he requests!"

Zero Pronoun Translation

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

BLEU

😭:Misuse of pronoun and tense in translation often results in confusion for readers.

上下文：这名男子在她眼中根本 就是死神！ 正文：是的，自己被一个死神一 般的男子救了下来，而且还答应 了他必须接受他的要求！

BLEU

张骆宇感觉自己继续待在这 里，也只不过是 吃狗粮，还不如赶紧离开。

[Figure 13]

[Figure 14]

[Candidate]: Yes, she was saved by a grim reaper man, and even agreed that he must accept all her requests.

[Figure 15]

[Figure 16]

0.7/1.0

0.52/1.0

1School of Artificial Intelligence, Wuhan University, 2Center for Language and Information Research, Wuhan University, 3Jiangxi Normal University, 4The University of Manchester, 5Yunnan Trrans Technology Co., Ltd., 6Malvern College Chengdu

[GroundTruth]: Dad, this is obviously a murderous hexagram, indicating her son

Cultural Safety

Terminology Localization

[Figure 17]

[GroundTruth]: Why are you still going on with that?

[Figure 18]

[Figure 19]

[Figure 20]

was extremely dangerous and has died unexpectedly.

BLEU

BLEU

爸，这明显是个凶卦，老阴 爻处在上位，卦形为坤，主 此人极为凶险，已死于非 命。

[Figure 21]

[Figure 22]

你TMD有完没完了？

[Candidate]: Dad, this is obviously a Qiong Gua. The old yin is top, the gua formation is

[Figure 23]

[Figure 24]

[Candidate]: Are you fucking done with that?

0.54/1.0

0.48/1.0

*Note: Offensive or sensitive language has been redacted for appropriateness.

Kun, indicating her son was extremely dangerous and has died an unnatural death.

### Abstract

[Ground Truth]:This person was incredibly meticulous and could still find the correct direction at a glance under heavy rain.

Idiom Translation

[Figure 25]

[Figure 26]

BLEU

[Candidate]: This person was incredibly meticulous like a hair, and could still find the

此人竟然如此心细如发，在 这等雨水冲刷之下，居然能 一眼就找出正确的方向！

[Figure 27]

[Figure 28]

0.65/1.0

correct direction at a glance even under heavy rain.

Large Language Models (LLMs) have advanced machine translation (MT) substantially, but their effectiveness in translating web novels remains unclear. Existing benchmarks rely on surface metrics that fail to properly assess the distinctive traits of this genre. To address these gaps, we introduce DITING, the first comprehensive evaluation framework for web novel translation, assessing narrative and cultural fidelity across six dimensions: idiom translation, lexical ambiguity, terminology localization, tense consistency, zero-pronoun resolution, and cultural safety, supported by over 18K expert-annotated Chinese–English sentence pairs. We further propose AgentEval, a reasoning-driven multi-agent evaluation framework that simulates expert deliberation to assess translation quality beyond lexical overlap, achieving the highest correlation with human judgments among seven tested automatic metrics. To enable metric comparison, we develop MetricAlign, a meta-evaluation dataset of 300 sentence pairs annotated with error labels and scalar quality scores. Comprehensive evaluation of fourteen open, closed, and commercial models reveals that Chinese-trained LLMs surpass larger foreign counterparts, and that DeepSeek-V3 delivers the most faithful and stylistically coherent translations. Our work establishes a new paradigm for exploring LLMbased web novel translation, and provides public datasets and codes to advance future research: https://github.com/WHUNextGen/ DITING.

[GroundTruth]: Zhang Luoyu felt that staying here would only mean witnessing others show affection, so it was better to leave quickly. [Candidate]: Zhang Luoyu felt that staying here would only mean being fed dog food, so it was better to leave quickly.

Lexical Ambiguity

[Figure 29]

[Figure 30]

😡: word-forword translation; incomprehensible

BLEU

张骆宇感觉自己继续待在这 里，也只不过是 吃狗粮，还不如赶紧离开。

[Figure 31]

[Figure 32]

0.7/1.0

[GroundTruth]: Dad, this is obviously a murderous hexagram, indicating her son was extremely dangerous and has died unexpectedly.

Terminology Localization

[Figure 33]

[Figure 34]

BLEU

爸，这明显是个凶卦，老阴爻 处在上位，卦形为坤，主此人 极为凶险，已死于非命。

[Figure 35]

[Candidate]: Dad, this is obviously a Qiong Gua. The old yin is top, the gua formation is Kun, indicating her son was extremely dangerous and has died an unnatural death.

[Figure 36]

0.54/1.0

BLEU scores cannot fully evaluate translation quality, especially in complex cases.

[GroundTruth]: Zhang Luoyu traveled to another world, and the next morning, he took his sword and rushed to the training ground.

###### Tense Consistency

[Figure 37]

[Figure 38]

BLEU

张骆宇穿越到了异世界，第二天 早上，他拿着剑冲向了练武场。

[Figure 39]

[Figure 40]

[Candidate]: Zhang Luoyu travels to another world, and the next morning, he takes his sword and rushes to the training ground.

0.61/1.0

[GroundTruth]: Yes,, she has been saved by a death God-like man, and even agreed to whatever he requests!" [Candidate]: Yes, she was saved by a grim reaper man, and even agreed that he must accept all her requests.

Zero Pronoun Translation

[Figure 41]

[Figure 42]

BLEU

😭:Misuse of pronoun and tense in translation often results in confusion for readers.

上下文：这男子在她眼中就是死神！ 正文：是的，自己被一个死神一般的男 子救了下来，而且还答应了他必须接受 他的要求！

[Figure 43]

[Figure 44]

0.52/1.0

###### Cultural Safety

[Figure 45]

[GroundTruth]: Why are you still going on with that?

[Figure 46]

BLEU

[Figure 47]

你TMD有完没完了？

[Figure 48]

0.48/1.0

*Note: Offensive or sensitive language has been redacted for appropriateness.

[Candidate]: Are you fucking done with that?

Figure 1: Examples of ground truth and low-quality translations across six dimensions, showing that even translations with high BLEU scores can contain errors causing reader confusion and misinterpretation.

even surpassing commercial MT systems (Karpinska and Iyyer, 2023). Recent studies demonstrate that models such as GPT-4 outperform strong MT baselines at document-level when appropriately prompted (Karpinska and Iyyer, 2023), improving discourse consistency and pronoun resolution by more than 10 BLEU points (Papineni et al., 2002). These results suggest that LLMs possess emerging abilities for contextual and discourse-level translation, raising expectations that they may generalize effectively across diverse text genres.

However, their effectiveness in web novel translation remains largely unexplored. Web novels, serialized online fictions that originated in East Asia, differ fundamentally from traditional MT domains. They feature informal and adaptive writing styles, rich character interactions, and culturally embedded expressions that challenge literal translation. Despite their massive global readership and growing influence of transmedia in comics, games, and television (Research Group of Chinese Academy of Social Sciences, 2024), the translation of web novels still relies heavily on human labor, limiting

### 1 Introduction

Large language models (LLMs) have achieved remarkable progress in machine translation (MT), reaching fluency and adequacy comparable to or

*These authors contributed equally to this work. †The Corresponding Author. Email: xieq@whu.edu.cn

scalability and accessibility. This gap motivates a central question: can LLMs capture the narrative coherence, stylistic fidelity, and cultural nuance essential for translating web novels effectively?

Answering this question is nontrivial, as web novel translation demands nuanced linguistic, stylistic, and cultural reasoning that current evaluation paradigms overlook. Existing benchmarks such as GuoFeng (Wang et al., 2023, 2024) and BWB (Jiang et al., 2022b) rely on surface metrics such as BLEU (Papineni et al., 2002), ChrF (Popovi´c, 2015), and BLEURT (Sellam et al., 2020), which measure overlap but miss narrative coherence and cultural intent. As shown in Figure 1, effective translation requires handling idiomatic expressions and expressions, maintaining temporal and referential consistency, and adapting culturally specific or sensitive content with fidelity. These intertwined challenges, spanning semantics, discourse, and ethics, define what we term narrative and cultural fidelity, a competence invisible to traditional evaluation.

To address this gap, we introduce DITING, the first comprehensive evaluation framework to assess the limits of LLMs on the linguistic reasoning, stylistic control, cultural adaptation, and safety alignment for web novel translation. Composed of six dimensions, it is decided by bilingual domain experts through pragmatic perspectives, and applicable to common linguistic phenomena in web novels: Idiom Translation for Chinese idioms and proverbs, Lexical Ambiguity for contextbased sense disambiguation, Terminology Localization for religious or internet-born expressions, Tense Consistency across narrative perspectives, Zero-Pronoun Resolution where omitted referents should be expressed explicitly, and Cultural Safety assessing alignment between sensitive content and social norms. For each task, we construct evaluation datasets totaling over 18K sentence pairs, consisting of original Chinese web-novel sentences and expert-annotated English gold-standard translations, with each sample assigned to one of the six evaluation dimensions.

Instead of relying on traditional metrics, we introduce AgentEval, a novel reasoning-driven multiagent evaluation framework that treats assessment as deliberation, to effectively simulate expert’s judgment process. Independent agents evaluate candidate translations across six dimensions, exchange rationales, and reach a consensus score through structured debate, emulating expert negoti-

ation of meaning and style. To facilitate systematic assessment of evaluation metrics, we further build MetricAlign, a novel dataset of 300 CN-EN sentence pairs annotated by experts with error labels and scalar quality scores, following a hybrid protocol integrating multidimensional quality metrics (MQM) (Lommel et al., 2014) and scalar quality metrics (SQM) (Graham et al., 2013). Using MetricAlign, we evaluate AgentEval and seven representative automatic evaluation metrics, showing that AgentEval achieves the strongest correlation with expert annotations, while other lexical, neural, and LLM-based metrics show weak alignment with human judgments.

Finally, based on DITING and AgentEval, we evaluate fourteen representative translation models spanning open-source, closed-source, and commercial MT models. Our results show that DeepSeekV3 produces the most faithful and stylistically coherent translations, Chinese-trained LLMs consistently outperform larger foreign counterparts, and state-of-the-art (SOTA) LLMs surpass commercial MT systems.

Our main contributions are: (1) We propose the first evaluation benchmark for web novel translation, namely DITING, formalizing narrative and cultural fidelity into six new dimensions with corresponding 18K expert-annotated CN-EN datasets that capture linguistic and cultural challenges. (2) We propose AgentEval, the first reasoning-driven multi-agent evaluation framework that models expert deliberation to assess translation quality beyond lexical overlap, achieves the highest alignment with human judgments. (3) We build MetricAlign, the first meta-evaluation dataset with error labels and scalar scores annotation of 300 CN-EN sentence pairs, enabling systematic comparison of metrics. (4) We evaluate seven automatic metrics and fourteen translation models, revealing limitations of existing metrics and translation models in web novel translation.

### 2 Method

##### 2.1 DITING

We introduce DITING, the first comprehensive evaluation framework with six new evaluation tasks, for web novel translation.

##### 2.1.1 Task Definition

Each task in DITING is formulated as a sentence-level sequence-to-sequence generation

[Figure 49]

Task 3. Can the current LLMs really translate Chinese web novels？

###### 6 Dimensions

###### DITING

Idiom Translation

Terminology Localization

Each with

- 1 Specific &
- 2 General

[Figure 50]

chapter-level Chinese-English bilingual data

Discuss Classify

[Figure 51]

[Figure 52]

Lexical Ambiguity

Zero Pronoun

[Figure 53]

[Figure 54]

Translation

Metrics for Idiom Translation

[Figure 55]

Label Optimize

Tense Consistency

Cultural Safety

split

[Figure 56]

[Figure 57]

[Figure 58]

Specific

[Figure 59]

Data: <Chinese>:大宝…<Ref>:Large Bo… <Task Type>:Idiom Translation <Task Details>:“挡枪” is an idiom…

s e n t e n c e p a i r s

- General 1
- General 2

[Figure 60]

[Figure 61]

Representative Subset

AgentEval Evaluation Based on Multi-Agent Debate

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

"sp": 1,"g_1": 2,"g_2": 1 “thoughts”: specific is 1 because…

"sp": 0, "g_1": 0, "g_2": 0 “thoughts”: specific is 0 because…

[Figure 66]

Experts Annotate

LLM Translation

[Figure 67]

[Figure 68]

"sp": 0, "g_1": 0, "g_2": 0 “thoughts”: specific is 0 because…

"sp": 0, "g_1": 0, "g_2": 0 “thoughts”: specific is 0 because…

[Figure 69]

High Agreement

[Figure 70]

Whether consensus is reached?

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

MetricAlign Dataset

Debate Continues Debate End

...

Judge

Task 1. Verify the effectiveness of previous metrics

Task 2. Find the best human-aligned agent system

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Fourteen current LLMs

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

BLEURT COMET BLEU

Seven Previous Metrics Which LLM？ Single Judge or Multi-Wisdom?

Figure 2: Overview of our work.

Terminology Localization. This task assesses the translation of distinctive terms in web novels, such as fantasy terms that lack direct English equivalents and require cultural localization, while avoiding simply literal translation that could lead to misunderstanding by non-native readers. For a source term ci, C3(ci,fθ(ci)) = Simsem(ci,fθ(ci)) measures how well the translation preserves both meaning and cultural nuance. For instance, "金丹" (Golden Core) should be conveyed as a spiritual concept, not a literal metal sphere.

problem. Given a Chinese source sentence Szh = {x1,...,xn}, a translation model fθ generates an English target sentence Sen = {y1,...,ym}, written as Ti : Szh → Sen = fθ(Szh), where i ∈ {1,...,6} indexes the six tasks. Each task defines an evaluation function Ci(Szh,Sen) that assesses how well the translation preserves a phenomenonspecific correspondence between the source and target subsequences.

Idiom Translation. This task evaluates whether idioms or proverbs preserve their figurative and emotional meaning beyond literal words. For an idiom ei ⊂ Szh and its translation fθ(ei) ⊂ Sen, the evaluation function C1(ei,fθ(ei)) = Simfig(ei,fθ(ei)) measures the alignment of figurative intent and tone. This assesses whether an expression like "挡枪" is translated freely as "take someone to hide the secret" rather than literally as "take the bullet".

Tense Consistency. This task checks whether temporal relations remain coherent in translation. Let τ(S) denote the tense–aspect sequence of a sentence. C4(Szh,Sen) = I[τ(Sen) ≈ Aligntense(Szh)] verifies that tense shifts in English align with temporal cues in Chinese. This captures whether a model preserves narrative time across dialogue, flashbacks, and inner monologues.

Zero Pronoun Translation. This task examines whether omitted pronouns in Chinese are properly restored in English. For each omitted pronoun ∅k and its referent rk, C5(∅k,fθ(Szh)) = I[rk ∈ fθ(Szh)] checks if the referent appears explicitly in translation. This ensures content remain grammatically complete and comprehensible.

Lexical Ambiguity. This task measures how well the model resolves polysemy and selects the correct sense in context. For an ambiguous term ai with candidate senses S(ai), C2(ai,fθ(ai)) = I[s(fθ(ai)) = s∗] checks if the translated sense s(fθ(ai)) matches the intended one s∗ = arg maxsj∈S(ai) P(sj | Szh). This ensures contextually accurate interpretation of slang and new internet coinages.

Cultural Safety. This task evaluates whether translations remain faithful while conforming to cul-

tural and ethical norms. C6(Szh,Sen) = Safe(Sen) measures whether the output avoids harmful, biased, or culturally inappropriate expressions. This safeguards against misinterpretations in sensitive genres such as violence, gender, or religion, ensuring socially responsible adaptation.

##### 2.1.2 Dataset Construction

Building on the six evaluation dimensions introduced above, we construct the DITING-CORPUS through a carefully controlled multi-stage process, as illustrated in Figure 2. Starting from billions of chapter-level Chinese–English bilingual passages collected from online platforms1, we segment and align them into high-quality sentence pairs by discussing with experts.

This conversion from chapter to sentence level reduces annotation complexity while retaining contextual fidelity. Annotators iteratively review and polish ambiguous or poorly expressed segments, ensuring each pair to guarantee the translation quality and cultural accuracy. Our annotation team includes two professional translators with over five years of web-novel translation experience and one undergraduate majoring in English. Through continuous expert discussion, the refined data are categorized by our annotators into six dimensions as described in section 2.1.1. This yields 18,745 expertcurated Chinese–English pairs covering idiomatic, lexical, terminological, temporal, referential, and cultural-safety phenomena, as summarized in Table 1 2.

Table 1: Dataset Statistics of DiTing-Corpus.

Dimension Total Idiom Translation 2,844 Lexical Ambiguity 4,576 Terminology Localization 1,836 Tense Consistency 4,982 Zero Pronoun Translation 4,407 Cultural Safety 100

##### 2.2 MetricAlign

To assess how closely automatic metrics align with expert judgment, we construct METRICALIGN, the first meta-evaluation dataset featuring exhaustive

- 1https://www.qidian.com/, https://fanqienovel. com/
- 2We will release web links to the original Chinese sentences from licensed web-novel platforms and openly provide the refined English translations verified by our annotators as gold-standard references. Users can retrieve the corresponding Chinese content via these links.

expert annotations across diverse linguistic and cultural translation phenomena.

Data Source. We uniformly sample 12 representative sentences from each of the six evaluation dimensions in the DITING-CORPUS (two sentences per dimension). Each source sentence is translated by 25 LLMs, as listed in Table 2, covering both open-source and proprietary systems, including multilingual and machine-translation–specific models. The resulting dataset comprises 300 Chinese–English sentence pairs, providing comprehensive coverage of translation challenges across idiomaticity, ambiguity, terminology, tense, referentiality, and cultural safety.

Table 2: Translation models used in MetricAlign.

Category Models Closed-Source GPT: GPT-4o, GPT-OSS: 20B MT-Specific Seed-X: Instruct-7B, PPO-7B Open-Source Qwen: 0.5B, 0.6B, 1.7B, 4B, 8B, 14B, 32B;

DeepSeek: 1.5B, 7B, 8B, 14B, 32B, 70B, R1, V3; LLaMA3: 8B, 70B; ChatGLM4: 9B; GemmaX2-28: 2B, 9B; GPT-OSS: 20B

Expert Annotation. All translations were evaluated by the same three domain experts as in Section 2.1.2 under a rigorously defined annotation protocol (Appendix B). The protocol combines sentencelevel quality assessment with targeted error-type tagging to capture linguistic, stylistic, and cultural nuances essential to web-novel translation. Annotators assessed each output along six dimensions (Table 3), assigning discrete scores (2 / 1 / 0) and providing brief justifications for borderline cases. Each evaluation dimension includes one specific metric targeting its core phenomenon (e.g., Idiomatic Fidelity, Contextual Semantic Resolution, Tense Cohesion) and two general metrics capturing broader qualities such as Cultural Adaptation, Tone and Style, or Fluency, enabling both focused and holistic assessment of translation quality.

Annotation was conducted using the Label Studio platform, which facilitated efficient reviewer assignment, annotation tracking, and version control. Prior to large-scale annotation, three pilot rounds were organized to calibrate inter-annotator consistency. Experts participated in collective discussion sessions to harmonize interpretations of idiomatic fidelity, contextual meaning, and stylistic adaptation. Ambiguous cases were resolved collaboratively through iterative refinement of the annotation guideline (Appendix B).

- Table 3: Condensed overview of scoring dimensions and criteria in MetricAlign (2 = High, 1 = Medium, 0 = Low).

Task Dimension Type Scoring Criteria (2 / 1 / 0)

Idiom Translation

Idiomatic Fidelity Spec. Natural idiom use / Stiff / Literal or omitted. Cultural Adaptation Gen. Localized meaning / Partly adapted / Misleading. Tone & Style Gen. Preserves tone / Slight drift / Lost or wrong tone.

Lexical Ambiguity

Contextual Resolution

Spec. Correct sense / Approx. / Wrong sense.

Pragmatic Appropriateness

Gen. Natural usage / Awkward / Unnatural. Information Integrity Gen. Complete / Minor gaps / Distorted.

Terminology Localization

Terminology Adequacy

Spec. Accurate / Acceptable /

Incorrect. Translation Strategy Gen. Adapted / Partial / Blind translit. Fluency Gen. Smooth / Minor issue /

Disruptive.

Tense Consistency

Tense Cohesion Spec. Consistent / Mostly ok / Broken. Structural Consistency

Gen. Clear order / Slightly unclear / Illogical. Naturalness Gen. Fluent / Minor flaw / Unnatural.

Zero-Pronoun Translation

Referent Recovery Spec. All restored / Partial / Wrong or missing.

Structural Completeness

Gen. Complete / Ambiguous / Fragmented. Naturalness Gen. Fluent / Awkward / Unnatural.

Cultural Safety

Content Compliance Spec. Safe / Borderline / Offensive. Value Alignment Gen. Positive / Minor issue / Biased. Sensitive Info Handling

Gen. Proper / Partial / Unsafe.

Quality Validation. To verify the reliability and consistency of expert annotations, we computed inter-annotator agreement (IAA) using two complementary measures: (1) Simple Agreement, which reports the proportion of identical labels across raters, and (2) Cohen’s κ (Cohen, 1960), which adjusts for chance agreement.

- Table 4: IAA of the annotation process for MetricAlign.

Metric Type Simple Agreement Cohen’s κ

Specific 0.96 0.94 General1 0.90 0.84 General2 0.91 0.85

As shown in Table 4, the MetricAlign annotations exhibit consistently high agreement across all metric types. Specific metrics—covering more objective linguistic judgments such as idiom fidelity or tense cohesion—achieve the strongest consistency among annotators (Agreement = 0.96, κ = 0.94). General metrics, which capture stylistic and pragmatic phenomena, also demonstrate substantial reliability, with agreements around 0.90–0.91 and κ scores of 0.84–0.85.

##### 2.3 AgentEval

To achieve expert-level automatic evaluation, we introduce AgentEval, a novel multi-agent evaluation framework that models translation assessment as a process of debate and consensus. Different from existing metrics evaluating lexical similarity, it conducts the reasoning-driven deliberation among cooperative agents, each acting as a specialized evaluator.

Debate and Judgment. At the core of AgentEval lies a structured multi-agent debate protocol designed to simulate expert discussion and judgment. Two scoring agents, A1 and A2, independently assess a translation pair and provide decisions Di = {score,rationale} based on their linguistic and contextual reasoning. A judge agent J then reviews both rationales to determine whether the agents have reached consensus. If the scores and reasoning align, J finalizes the evaluation; otherwise, the agents enter a new round of debate.

In subsequent rounds, the agents refine their judgments by referencing not only the input knowledge k but also each other’s prior arguments stored in a shared memory m. This iterative deliberation process continues until convergence or a maximum round limit is reached. If no agreement

emerges, the judge produces a final decision DJ(final) grounded in accumulated evidence and the comparative soundness of arguments. Through this debatedriven reasoning, AgentEval achieves human-like evaluation dynamics—balancing analytical precision with interpretive flexibility.

Metrics Matching. Each evaluation instance is represented as a quadruple (x,y,t,r), where x denotes the Chinese source sentence, y its translated output, t the associated task type, and r the taskspecific evaluation requirements. Based on t, each scoring agent retrieves the appropriate evaluation schema Mt and exemplar references Et, ensuring that judgments are grounded in domain-relevant rules and examples. For instance, an agent assigned to Idiom Translation attends to figurative equivalence and tone preservation, whereas one handling Tense Consistency focuses on narrative temporal coherence. This schema-matching mechanism allows agents to reason within contextually defined evaluation boundaries rather than applying one-size-fits-all criteria.

Evaluation Strategy. For a given source–translation pair (x,y), the assigned task type t determines the evaluation rule set

Rt = {rsp,rg1,rg2}, where rsp is the specific metric emphasizing the task’s key property (e.g., idiomaticity or referent recovery), and rg1,rg2 are general metrics reflecting fluency, style, or safety. Each agent produces a fine-grained decision vector [s×Rt]ni=1 capturing the individual sub-scores and rationale for the evaluated sample. By combining rule-based interpretability with debate-based reasoning, AGENTEVAL ensures that translation quality is judged not only by surface similarity but also by the underlying linguistic and cultural fidelity that human experts value. See Appendix C.2 for more details.

### 3 Experiment

##### 3.1 Evaluation of Automatic Metrics

Using MetricAlign, we systematically evaluate the reliability of seven representative automatic metrics and our proposed AgentEval framework for web-novel translation. We measure the alignment between automatic scores and expert judgments using Spearman Correlation (SC) (Spearman, 1904) and Variance Explained Score (Pedregosa et al., 2011) (it measures how much of the variance in human scores can be explained by model predictions).

As shown in Table 5, AgentEvalDebate-R1, our multi-agent evaluation setting built on the DeepSeek-R1 model, achieves the strongest correlation with human annotations across all dimensions. In comparison, AgentEvalDS-R1, the single-agent variant using the same base model, also demonstrates strong consistency with expert evaluation but performs lower than the multiagent version. This result highlights the advantage of multi-agent simulation in capturing nuanced linguistic and cultural phenomena. Overall, both settings substantially outperform traditional automatic metrics, confirming the effectiveness and generalizability of our framework for webnovel translation assessment. In contrast, none of the existing automatic metrics exhibit strong alignment with human judgments in this domain. Among the seven baseline metrics, BLEU (Papineni et al., 2002), BLEURT (Sellam et al., 2020), and COMET (Bosselut et al., 2019) capture partial translation-quality signals but fail to reflect the literary and stylistic nuances characteristic of web-novel texts. The relatively weak correlations of chrF (Popovi´c, 2015) and ROUGE (Lin, 2004) suggest excessive reliance on surface-form over-

lap, neglecting contextual and stylistic fidelity. COMETKiwi-da (Rei et al., 2022)—a referencefree variant—also underperforms, likely due to a domain mismatch between its training data and online-literature style. The multidimensional multi-agent debate (M-MAD) framework (Feng et al., 2025), an advanced LLM-based evaluation method, shows promise in general MT evaluation but demonstrates clear limitations on web-novel translation, underscoring the unique challenges of this genre. These results highlight the necessity of domain-specific metric design and further validate the robustness of our AgentEval framework.

Table 5: Correlation analysis of automatic metrics with human evaluation for web novel translation.

Spearman Correlation

Variance Explained (Human Scores)

Metric

BLEU 0.472 22.2 chrF 0.312 9.8 ROUGE 0.319 10.2 BLEURT 0.472 22.3 COMET 0.471 22.2 COMETkiwi-da -0.034 0.1 M-MAD score -0.316 10

AgentEvalDS−R1 0.655 42.9 AgentEvalDebate−R1 0.669 44.8

3.1.1 Results of AgentEval with Different Backbone Models

We further analyze the performance of AgentEval across different backbone models under both the multi-agent (“Debate-”) and single-agent configurations. To complement the correlation analysis in the previous subsection, we employ multiple agreement-based measures: Simple Agreement( the proportion of cases where annotators give exactly the same labels), Cohen’s κ (Cohen, 1960), Linear/Quadratic Weighted κ (Yilmaz and Demirhan, 2023), ICC(3,1) (Shrout and Fleiss, 1979), and Agreement Rate to capture fine-grained consistency between model-based and human assessments from complementary statistical perspectives.

As shown in Table 6, among different backbones, the DeepSeek-R1 family shows the strongest overall performance, surpassing GPT-4o and DeepSeekV3 across all agreement measures. This advantage likely stems from R1’s enhanced reasoning capabilities, which better capture the nuanced cultural and stylistic aspects of web-novel translation. The multi-agent variant Debate-R1 and its singleagent counterpart DS-R1 exhibit comparable over-

- Table 6: Agreement analysis between AgentEval with different backbone models and the human evaluation.

Linear Weighted Kappa

Quadratic Weighted Kappa

Simple Agreement

Cohen’s Kappa

Agreement Rate

Model Dimensions

ICC(3,1)

general2 0.513 / 0.507 0.289 / 0.274 0.344 / 0.343 0.393 / 0.403

general1 0.610 / 0.603 0.395 / 0.382 0.445 / 0.436 0.485 / 0.478 specific 0.620 / 0.613 0.420 / 0.407 0.458 / 0.447 0.488 / 0.479 Total 0.457 / 0.443 0.274 / 0.266 0.416 / 0.411 0.472 / 0.474

Deepseek-V3

0.651 0.450

general2 0.497 / 0.517 0.254 / 0.279 0.336 / 0.358 0.408 / 0.425

general1 0.606 / 0.606 0.376 / 0.372 0.442 / 0.439 0.492 / 0.489 specific 0.596 / 0.596 0.374 / 0.371 0.442 / 0.438 0.495 / 0.491 Total 0.432 / 0.438 0.230 / 0.249 0.408 / 0.419 0.487 / 0.495

GPT-4o

0.652 0.435

general2 0.580 / 0.570 0.375 / 0.359 0.459 / 0.453 0.539 / 0.542

general1 0.673 / 0.640 0.487 / 0.432 0.558 / 0.527 0.612 / 0.601 specific 0.657 / 0.657 0.481 / 0.479 0.560 / 0.557 0.627 / 0.622 Total 0.488 / 0.488 0.346 / 0.349 0.529 / 0.533 0.627 / 0.633

Deepseek-R1

0.760 0.488

general2 0.517 / 0.513 0.300 / 0.290 0.377 / 0.376 0.446 / 0.451

general1 0.620 / 0.607 0.416 / 0.393 0.487 / 0.471 0.543 / 0.532 specific 0.620 / 0.620 0.427 / 0.424 0.491 / 0.487 0.544 / 0.539 Total 0.457 / 0.443 0.285 / 0.277 0.454 / 0.448 0.533 / 0.535

Debate-V3

0.704 0.457

general2 0.492 / 0.488 0.267 / 0.257 0.350 / 0.340 0.422 / 0.414

general1 0.589 / 0.579 0.370 / 0.353 0.440 / 0.424 0.495 / 0.480 specific 0.593 / 0.599 0.391 / 0.398 0.459 / 0.455 0.517 / 0.502 Total 0.418 / 0.424 0.242 / 0.258 0.414 / 0.417 0.499 / 0.495

Debate-GPT-4o

0.683 0.421

general2 0.577 / 0.573 0.371 / 0.366 0.449 / 0.450 0.525 / 0.533

general1 0.660 / 0.637 0.466 / 0.427 0.558 / 0.534 0.631 / 0.619 specific 0.657 / 0.670 0.485 / 0.504 0.561 / 0.572 0.628 / 0.632 Total 0.483 / 0.493 0.321 / 0.344 0.523 / 0.530 0.628 / 0.632

Debate-R1

0.760 0.488

all agreement, with identical ICC and Agreement Rate scores. However, the single-agent R1 performs slightly better on general metrics, while the multi-agent setting achieves higher scores on specific metrics, indicating that debate-based consensus particularly strengthens fine-grained reasoning over phenomenon-specific judgments. In contrast, for both the DeepSeek-V3 and GPT-4o backbones, the multi-agent configuration consistently surpasses their single-agent counterparts across all agreement measures. These results suggest that the benefit of multi-agent debate depends on the model’s reasoning strength: while R1 already demonstrates high self-consistency as a single agent, models with weaker internal deliberation gain more from multi-agent interaction.

##### 3.2 Evaluation of Translation Models

Settings Evaluated Models. Using DITING and our AgentEval metric (based on DeepSeek-R1), we comprehensively evaluate 14 representative models on web-novel translation. The evaluation covers a diverse set of systems, including proprietary and open-source models, multilingual LLMs, MTspecific LLMs, and commercial translation models. A complete list of evaluated models and configurations is provided in Appendix C.13. Proprietary models are accessed through official APIs, while open-source models are tested using their publicly

3Since DeepSeek-R1 serves as the evaluator backbone in AgentEval, it is not included among the evaluated translation models.

released checkpoints with default decoding parameters (e.g., temperature). For all models, we adopt a context-aware prompting strategy: the last sentence of the preceding paragraph is included as contextual input, and the target sentence serves as the translation query. This setup better reflects realworld web-novel translation, where meaning and tone often rely on narrative continuity. Further implementation details are provided in Appendix C.

##### 3.2.1 Benchmark Performances

Table 7 presents detailed results across the six evaluation dimensions. DeepSeek-V3 achieves the highest overall score (5.16), followed closely by GPT-4o (5.09). Both substantially outperform commercial MT systems, indicating that advanced LLMs now surpass traditional pipelines even in literary domains with complex stylistic demands. Although model scale remains a key factor—Qwen332B outperforms its 14B and 8B variants—data alignment proves equally decisive. The Chinesecentric Qwen3-8B (3.96) surpasses the much larger English-focused LLaMA3-70B (3.58), suggesting that exposure to the source-language culture can compensate for smaller model size. Reinforcement learning also brings measurable gains: Seed-XPPO-7B improves by +0.65 over its instructiontuned counterpart and ranks third overall, with idiom and ambiguity performance rivaling DeepSeekV3. This demonstrates that targeted optimization can instill stronger cultural and figurative understanding even in smaller models.

- Table 7: Performance of evaluated models on DITING-CORPUS using AgentEval. Rows list task × metric (S: Specific, G1/G2: General metrics, Σ: sum). Scores are averaged on a 0–2 scale.

DeepSeek-R1-70B

IFLYTEK Trans. Seed-X-PPO-7B Seed-X-Inst-7B GemmaX2-28-9B

LLaMA3-70B Qwen3-32B Qwen3-14B Qwen3-8B LLaMA3-8B ChatGLM4-9B Google Trans.

DeepSeek-v3

GPT-4o

Idiom — S 1.62 1.70 1.24 0.88 1.44 1.38 1.38 0.86 1.06 1.52 0.88 1.78 1.26 1.22

- Idiom — G1 1.64 1.66 1.26 0.90 1.40 1.34 1.30 0.84 0.96 1.54 0.86 1.76 1.20 1.14
- Idiom — G2 1.78 1.78 1.32 0.98 1.66 1.52 1.42 0.94 1.10 1.66 0.92 1.88 1.34 1.24 Idiom — Σ 5.04 5.14 3.82 2.76 4.50 4.24 4.10 2.64 3.12 4.72 2.66 5.42 3.80 3.60 Ambiguity — S 1.82 1.86 1.40 0.80 1.48 1.28 1.26 0.86 1.26 1.48 0.90 1.82 1.42 1.16

- Ambiguity — G1 1.80 1.78 1.32 0.70 1.32 1.24 1.20 0.86 1.06 1.36 0.72 1.80 1.34 1.00
- Ambiguity — G2 1.88 1.88 1.38 0.78 1.52 1.38 1.32 0.84 1.30 1.68 0.84 1.84 1.54 1.10 Ambiguity — Σ 5.50 5.52 4.10 2.28 4.32 3.90 3.78 2.56 3.62 4.52 2.46 5.46 4.30 3.26 Terminology — S 1.44 1.66 1.28 1.20 1.32 1.30 1.10 0.92 1.10 1.46 1.10 1.40 1.34 1.24

- Terminology — G1 1.44 1.62 1.22 1.14 1.24 1.30 0.98 0.80 1.10 1.48 1.00 1.42 1.32 1.18
- Terminology — G2 1.66 1.84 1.50 1.30 1.44 1.52 1.24 0.98 1.20 1.60 1.02 1.70 1.50 1.24 Terminology — Σ 4.54 5.12 4.00 3.64 4.00 4.12 3.32 2.70 3.40 4.54 3.12 4.52 4.16 3.66 Tense — S 1.68 1.80 1.58 1.48 1.76 1.66 1.60 1.34 1.48 1.38 1.44 1.50 1.50 1.64

- Tense — G1 1.94 2.00 1.86 1.66 1.90 1.86 1.74 1.56 1.66 1.54 1.58 1.70 1.72 1.82
- Tense — G2 1.60 1.66 1.46 1.20 1.44 1.38 1.32 1.08 1.20 1.32 1.08 1.44 1.36 1.42 Tense — Σ 5.22 5.46 4.90 4.34 5.10 4.90 4.66 3.98 4.34 4.24 4.10 4.64 4.58 4.88 Zero Pronoun — S 1.86 1.70 1.50 1.30 1.72 1.54 1.20 0.98 1.14 1.66 0.80 1.18 1.06 1.04

- Zero Pronoun — G1 1.88 1.72 1.58 1.34 1.72 1.64 1.28 1.02 1.14 1.70 0.76 1.30 1.14 1.14
- Zero Pronoun — G2 1.82 1.64 1.52 1.18 1.56 1.50 1.14 0.84 1.04 1.46 0.66 1.20 1.04 1.10 Zero Pronoun — Σ 5.56 5.06 4.60 3.82 5.00 4.68 3.62 2.84 3.32 4.82 2.20 3.68 3.24 3.28 Cultural Safety — S 1.60 1.56 1.66 1.56 1.54 1.50 1.44 1.38 1.44 1.56 1.36 1.42 1.36 1.48

- Cultural Safety — G1 1.40 1.30 1.28 1.20 1.20 1.24 1.24 1.14 1.18 1.20 1.10 1.08 1.02 1.18
- Cultural Safety — G2 1.70 1.80 1.80 1.86 1.68 1.68 1.58 1.52 1.68 1.76 1.58 1.70 1.54 1.68 Cultural Safety — Σ 4.70 4.66 4.74 4.62 4.42 4.42 4.26 4.04 4.30 4.52 4.04 4.20 3.92 4.34 Average Score-Σ 5.09 5.16 4.36 3.58 4.56 4.38 3.96 3.13 3.68 4.56 3.10 4.65 4.00 3.84

Advanced LLMs like DeepSeek-V3 and GPT-

- 4o dominate in Idiom Translation (Σ: 5.14/5.04) and Lexical Ambiguity (Σ: 5.52/5.50), showing their ability to interpret figurative expressions and resolve semantic ambiguity—skills that traditional MT models often mishandle. However, their lead narrows in Zero-Pronoun Translation (5.06/5.56) and Cultural Safety (4.66/4.70), where contextual reconstruction and value-sensitive adaptation remain elusive. The strong Cultural Safety score of DeepSeek-R1-70B (Σ: 4.74) highlights that safetyaligned training can enhance ethical robustness, though it may not directly translate to overall translation quality.

For Terminology Localization, DeepSeek-V3 again leads (Σ: 5.12), followed by Qwen3-32B and Seed-X-PPO-7B, showing that both scale and domain adaptation contribute to better rendering of specialized terms. Most models perform relatively well on Tense Consistency (average ≥ 4.6), a sign that grammatical and temporal structures are easier to model than abstract pragmatics or cultural nuances. Smaller models, especially Seed-XPPO-7B, perform competitively on most linguistic dimensions but falter on zero-pronoun recovery, underscoring the difficulty of maintaining discourse coherence with limited contextual capacity. Overall, these results point to a layered challenge in web-

novel translation: while LLMs have mastered surface fidelity and stylistic flow, deeper cross-cultural reasoning and implicit reference reconstruction remain open frontiers.

### 4 Conclusion

We introduced DITING, the first comprehensive framework for evaluating large language models on web novel translation, emphasizing narrative and cultural fidelity beyond conventional surface metrics. Through six linguistically and culturally motivated dimensions and over 18K expertannotated Chinese–English pairs, DITING captures the stylistic and cultural challenges unique to this domain. Our analysis of fourteen translation models revealed that Chinese-trained LLMs outperform larger foreign systems, and that DeepSeek-V3 achieves the most faithful and stylistically coherent translations. In future work, we plan to enhance the multi-agent framework with reinforcement learning to optimize deliberation dynamics and improve consistency in evaluation outcomes.

### Limitations

While this study marks an important step toward understanding LLM performance in web novel translation, several limitations remain. Due to re-

source constraints, the size of expert-annotated meta-evaluation data is limited, and the current framework focuses primarily on sentence-level analysis, overlooking document-level narrative coherence. In addition, the multi-agent evaluation framework has not yet been optimized for dynamic coordination or learning. Future work will address these limitations by (1) developing a dedicated scoring model through customized training to internalize expert evaluation criteria, and (2) extending the framework to document-level evaluation to systematically capture long-range narrative consistency and character development. These efforts aim to enhance both the accuracy and scalability of web novel translation evaluation.

### Ethics Statement

This study was conducted in accordance with established ethical guidelines for research. All translation datasets used are publicly available and contain no personally identifiable information. No human participants were directly involved in the experiments, and all annotation tasks were conducted by trained annotators under fair labor practices. We ensured that our work avoids generating or promoting harmful content and respects cultural and linguistic sensitivities. Potential risks include the handling of sensitive content present in some datasets, and the possibility that our evaluation metric may inadvertently misrepresent translation quality, which should be considered when interpreting or applying the results. All research artifacts, including datasets, code, and models, are provided solely for research and educational purposes under the MIT license, and the authors assume no responsibility for any consequences arising from their use. All resources are publicly available at https://github.com/WHUNextGen/DITING.

### Acknowledgments

We would like to thank all the anonymous reviewers and area chairs for their comments. This research is supported by Key Project of the National Natural Science Foundation of China (U23A20316) and CCF-Tencent Rhino-Bird Open Research Fund (CCF-Tencent RAGR20250115).

### References

Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chaitanya Malaviya, Asli Celikyilmaz, and Yejin Choi.

2019. Comet: Commonsense transformers for automatic knowledge graph construction. Preprint, arXiv:1906.05317.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.

Yanran Chen and Steffen Eger. 2023. MENLI: Robust evaluation metrics from natural language inference. Transactions of the Association for Computational Linguistics, 11:804–825.

Cheng-Han Chiang and Hung yi Lee. 2023. Can large language models be an alternative to human evaluations? Preprint, arXiv:2305.01937.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Zhaopeng Feng, Jiayuan Su, Jiamei Zheng, Jiahan Ren, Yan Zhang, Jian Wu, Hongwei Wang, and Zuozhu Liu. 2025. M-MAD: Multidimensional multi-agent debate for advanced machine translation evaluation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7084–7107, Vienna, Austria. Association for Computational Linguistics.

Yvette Graham, Timothy Baldwin, Alistair Moffat, and Justin Zobel. 2013. Continuous measurement scales in human evaluation of machine translation. In Proceedings of the 7th linguistic annotation workshop and interoperability with discourse, pages 33–41.

Damien Hansen and Emmanuelle Esperança-Rodier. 2023. Human-adapted mt for literary texts: Reality or fantasy? In Proceedings of the New Trends in Translation and Technology (NeTTT), pages 178–190. Accessed: 2025-09-26.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38.

Yuchen Jiang, Tianyu Liu, Shuming Ma, Dongdong Zhang, Jian Yang, Haoyang Huang, Rico Sennrich, Ryan Cotterell, Mrinmaya Sachan, and Ming Zhou. 2022a. BlonDe: An automatic evaluation metric for document-level machine translation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1550–1565, Seattle, United States. Association for Computational Linguistics.

Yuchen Eleanor Jiang, Tianyu Liu, Shuming Ma, Dongdong Zhang, Mrinmaya Sachan, and Ryan Cotterell.

2022b. A bilingual parallel corpus with discourse annotations. Preprint, arXiv:2210.14667.

Marzena Karpinska and Mohit Iyyer. 2023. Large language models effectively leverage document-level context for literary translation, but critical errors persist. Preprint, arXiv:2304.03245.

- Tom Kocmi and Christian Federmann. 2023a. Gembamqm: Detecting translation quality error spans with gpt-4. Preprint, arXiv:2310.13988.
- Tom Kocmi and Christian Federmann. 2023b. Large language models are state-of-the-art evaluators of translation quality. Preprint, arXiv:2302.14520.

Waltraud Kolb. 2023. ’I am a bit surprised’: Literary translation and post-editing processes compared, pages 53–68. Routledge.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Arle Lommel, Aljoscha Burchardt, Maja Popovi´c, Kim Harris, Eleftherios Avramidis, and Hans Uszkoreit. 2014. Using a new analytic measure for the annotation and analysis of MT errors on real data. In Proceedings of the 17th Annual Conference of the European Association for Machine Translation, pages 165–172, Dubrovnik, Croatia. European Association for Machine Translation.

Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318. Association for Computational Linguistics.

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. 2011. Scikit-learn: Machine learning in python. In Journal of Machine Learning Research, volume 12, pages 2825–2830.

Maja Popovi´c. 2015. chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 392–395, Lisbon, Portugal. Association for Computational Linguistics.

Ricardo Rei, José G. C. de Souza, Duarte Alves, Chrysoula Zerva, Ana C Farinha, Taisiya Glushkova, Alon Lavie, Luisa Coheur, and André F. T. Martins.

- 2022. COMET-22: Unbabel-IST 2022 submission for the metrics shared task. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 578–585, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Research Group of Chinese Academy of Social Sciences. 2024. 2024 China online literature development research report. Accessed: 2024-09-25.

Thibault Sellam, Dipanjan Das, and Ankur Parikh. 2020. BLEURT: Learning robust metrics for text generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7881–7892, Online. Association for Computational Linguistics.

Sheikh Shafayat, Dongkeun Yoon, Woori Jang, Jiwoo Choi, Alice Oh, and Seohyon Jung. 2025. A 2step framework for automated literary translation evaluation: Its promises and pitfalls. Preprint, arXiv:2412.01340.

Patrick E. Shrout and Joseph L. Fleiss. 1979. Intraclass correlations: uses in assessing rater reliability. Psychological Bulletin, 86(2):420.

Charles Spearman. 1904. The proof and measurement of association between two things. The American Journal of Psychology, 15(1):72–101.

Longyue Wang, Siyou Liu, Minghao Wu, Wenxiang Jiao, Xing Wang, Jiahao Xu, Zhaopeng Tu, Liting Zhou, Yan Gu, Weiyu Chen, Philipp Koehn, Andy Way, and Yulin Yuan. 2024. Findings of the wmt 2024 shared task on discourse-level literary translation. In Proceedings of the Ninth Conference on Machine Translation.

Longyue Wang, Zhaopeng Tu, Yan Gu, Siyou Liu, Dian Yu, Qingsong Ma, Chenyang Lyu, Liting Zhou, ChaoHong Liu, Yufeng Ma, and 1 others. 2023. Findings of the wmt 2023 shared task on discourse-level literary translation: A fresh orb in the cosmos of llms. In Proceedings of the Eighth Conference on Machine Translation, pages 55–67.

Jianhao Yan, Pingchuan Yan, Yulong Chen, Jing Li, Xianchao Zhu, and Yue Zhang. 2024. Benchmarking gpt-4 against human translators: A comprehensive evaluation across languages, domains, and expertise levels. Preprint, arXiv:2411.13775.

Ayfer Ezgi Yilmaz and Haydar Demirhan. 2023. Weighted kappa measures for ordinal multi-class classification performance. Applied Soft Computing, 134:110020.

Ran Zhang, Wei Zhao, Lieve Macken, and Steffen Eger. 2025. Litransproqa: an llm-based literary translation evaluation metric with professional question answering. Preprint, arXiv:2505.05423.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with bert. Preprint, arXiv:1904.09675.

Wei Zhao, Michael Strube, and Steffen Eger. 2023. DiscoScore: Evaluating text generation with BERT and discourse coherence. In Proceedings of the 17th Conference of the European Chapter of the Association

for Computational Linguistics, pages 3865–3883, Dubrovnik, Croatia. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

### A Related Works

##### A.1 Evaluation Metrics for Translation

Evaluation metrics for translation have traditionally focused on classical texts and mainstream literature, evolving through a combination of automatic metrics and human evaluation frameworks. In non-literary domains, metrics at both sentence and document levels are well-studied, including surface-form matching metrics such as BLEU (Papineni et al., 2002) and ChrF (Popovi´c, 2015), NLI-based MENLI (Chen and Eger, 2023), trained metrics such as COMET (Bosselut et al., 2019), BERT-based BLEURT (Sellam et al., 2020) and BERTScore (Zhang et al., 2020), as well as document-level metrics such as BLONDE (Jiang

- et al., 2022a) and DiscoScore (Zhao et al., 2023). While some of these metrics have been applied to literary texts (Hansen and Esperança-Rodier, 2023), their effectiveness in assessing literary translation remains limited. Most rely on referencebased n-gram overlap, capturing only surface lexical matches, and fail to account for stylistic expression, semantic variation, or discourse-level coherence. Such limitations are especially pronounced in web novels, which feature colloquial speech, internet slang, and long-range narrative dependencies.

With the advent of LLMs, the paradigm of LLM as a Judge has emerged as a cost-efficient approach for translation evaluation. LLMs have been applied across text generation (Zheng et al., 2023), code evaluation (Chen et al., 2021), and dialogue system safety (Chiang and yi Lee, 2023). In translation, however, they face limitations: sensitivity to prompt design (Kocmi and Federmann, 2023b), susceptibility to hallucinations (Ji

- et al., 2023), and stylistic artifacts such as literalism, calques, or MT-style neologisms (Kolb,

- 2023). Existing LLM-based evaluation methods, including GEMBA-MQM (Kocmi and Federmann, 2023a), M-MAD (Feng et al., 2025), the two-step framework (Shafayat et al., 2025), and LITRANSPROQA (Zhang et al., 2025) have made

progress in capturing finer-grained literary quality aspects but remain constrained by language coverage, evaluation scope, or insufficient attention to stylistic nuances. Human-centered frameworks such as MQM (Lommel et al., 2014) provide detailed evaluation but are resource-intensive and difficult to scale. Together, these observations highlight the need for benchmarks that integrate transparent sources, multidimensional evaluation protocols, and human-in-the-loop LLM assessment to reliably evaluate literary and web novel translation.

##### A.2 Benchmarks for Web Novels

Recent work has introduced benchmarks for Chinese–English web novel translation. BWB (Jiang et al., 2022b) provides a large bilingual corpus of 196K chapters (9.6M sentence pairs), covering multiple genres and preserving document-level context. It was released alongside BlonDe, a discourseaware evaluation metric. GuoFeng (Wang et al., 2023, 2024), developed with industry partners, served as the official dataset for the WMT 2023 discourse-level literary translation task. It contains 226K chapters (≈1.9M sentence pairs, 32M English words) with professional translations and controlled splits for training and evaluation.

These datasets emphasize long-range dependencies and discourse phenomena but still present limitations. The provenance of the translations is often unclear, and previous analysis indicates that some segments may originate from post-edited machine translation (Kolb, 2023), raising concerns about stylistic authenticity. Moreover, evaluation continues to rely heavily on automatic metrics, which overlook narrative consistency, cultural adaptation, and stylistic fidelity. While recent studies have begun benchmarking LLMs for translation (Yan et al., 2024), fine-grained investigations into web novel–specific phenomena (e.g., idioms, zero pronouns, tense consistency) remain rare. This highlights the need for benchmarks that combine transparent sources with multidimensional evaluation protocols, leveraging both human expertise and LLM-based evaluation in a human-in-the-loop framework.

### B Annotation Details

##### B.1 Annotation Guidelines

Task Objective This project aims to evaluate the quality of web novel translations generated by different models, based on an established evaluation framework and corresponding datasets. Annotators are required to assign scores to each translation according to the provided Scoring Sheet across six dimensions. Each dimension consists of one specific metric and two general metrics. Ratings should be given on a 0–2 scale, accompanied by short justifications or annotation notes when necessary.

The objectives are:

- • To independently assess model performance in Idiom Translation, Lexical Ambiguity, Terminology Localization, Tense Consistency, Zero Pronoun Translation, and Security.
- • To ensure consistent evaluation criteria and minimize subjective bias.

##### Annotation Procedure

- 1. Read the model-generated translation and analyze it in reference to the source text and reference translation.
- 2. Evaluate the output according to the six dimensions and their associated scoring criteria.
- 3. Record the score (0/1/2) for each metric in the scoring sheet and compute the total score. Provide optional comments if clarification is necessary.
- 4. Double-check the total score for accuracy.

Evaluation Dimensions and Criteria The sixdimensional scoring criteria are detailed in Table 8 and Table 9.

##### B.2 Annotator Demography

The construction of our DITING relies on the linguistic expertise of a team of highly qualified annotators with strong backgrounds in translation studies and cross-lingual communication. Their professional training and experience in CN-EN translation ensure accurate and contextually grounded annotations across diverse stylistic and cultural expressions in web novel texts.

The annotation team consists of three members with strong backgrounds in translation studies. Two

of them are professional translators at a leading Chinese translation company, possessing extensive experience in CN-EN translation and linguistic quality assessment. Their prior work includes abundant literary translation projects, equipping them with the ability to handle typical cultural in-depth expressions of web novels. The third annotator is a student majoring in translation at a prestigious Chinese university, who participated both as an annotator and a quality supervisor. The student was responsible for performing annotation tasks while coordinating consistency checks and revising annotation guidelines based on feedback from the team.

The team followed a well-structured annotation workflow. Weekly annotation meetings were held to discuss challenging cases and refine annotation criteria. Feedback from translators was systematically integrated into guideline updates, enhancing both reliability and linguistic validity throughout dataset construction.

Together, the team’s professional translation expertise, linguistic sensitivity, and collaborative workflow enabled the creation of a high-quality web novel translation benchmark. Their efforts ensured that the dataset is both linguistically faithful and contextually consistent, establishing a solid foundation for future research on evaluating LLMbased literary translation.

##### B.3 Annotation Example

We show some cases which can demonstrate our annotation guideline in Table 10.

#### B.4 Annotation Process Our annotation process can be seen in Figure 3. C Experimental Details C.1 Evaluated Translation Models Frontier LLMs include industry-leading APIs:

- • GPT-4o, known for its strong general-purpose multilingual and reasoning abilities;
- • DeepSeek-V3, which is optimized for Chinese and English tasks with enhancements in coding and mathematics;

For open-source models, we select large-scale foundation models:

• DeepSeek-R1-70B, which offers strong performance in complex reasoning;

[Figure 87]

Figure 3: The Label Studio interface of the DITING annotation process.

- • LLaMA3-70B, Meta’s top-tier open-source model;
- • ChatGLM4-9B, optimized for dialogue scenarios;
- • Qwen3 series, including Qwen3-8B, Qwen314B, and Qwen3-32B;
- • LLaMA3-8B, which balances efficiency and capability.

Additionally, translation-specialized models are incorporated, including:

- • Google Translate, a widely-used commercial machine translation service based on largescale neural methods;
- • IFLYTEK Translate, a leading Chineseoriented translation system;
- • ByteDance’s Seed-X-PPO-7B, fine-tuned with reinforcement learning;
- • Seed-X-Instruct-7B; instruction-tuned for translation;
- • Xiaomi’s GemmaX2-28-9B.

This selection ensures a diverse and representative evaluation across model types, scales, and specializations.

##### C.2 Evaluation of Automatic Metrics

We provide the detailed configuration of all the metrics used in our metrics Evaluation experiment in Table 14.

##### C.3 Case Study

We demonstrated how a case was evaluated within our framework in Table 11.

##### C.4 Prompts

We show our prompts used in the evaluation framework which can be seen in Table 12, Table 13.

Table 8: The scoring criteria.

Idiom Translation Specific Metric: Idiomatic Fidelity & Naturalness 2 points: The idiom is accurately conveyed and expressed naturally.

- 1 point: Meaning basically conveyed, but expression is somewhat stiff or unnatural.

- 0 points: Mistranslated, literally translated, or omitted. General Metric: Cultural Adaptation

2 points: Use of authentic localized equivalents or reasonable annotations; cultural connotations are effectively conveyed and easily understood by readers.

- 1 point: Some degree of localization, but expression is awkward or only partially appropriate.

- 0 points: Literal or awkward rendering, or cultural load completely ignored, causing misunderstanding. General Metric: Tone and Style

2 points: Tone and stylistic features of the original are preserved; expression is natural and appropriate to the genre.

- 1 point: Style is generally preserved, with minor inconsistencies.

- 0 points: Style seriously deviates or tone is missing, disrupting the atmosphere. Lexical Ambiguity

Specific Metric: Contextual Semantic Resolution Rate 2 points: Accurate and natural word sense disambiguation in context.

- 1 point: Meaning roughly conveyed but expressed through literal or awkward phrasing.

- 0 points: Incorrect sense selection or mistranslation. General Metric: Pragmatic Appropriateness

2 points: Word sense selection conforms to English usage, natural collocations, and accurate semantics.

- 1 point: Word sense selection conforms to English usage, natural collocations, and accurate semantics.

- 0 points: Word choice violates usage conventions, leading to misunderstanding or unclear expression. General Metric: Information Integrity

2 points: Fully conveys the source meaning without omission or distortion; semantics are coherent.

- 1 point: Information is mostly conveyed, but minor omissions or vague expressions exist.

- 0 points: Key information is missing or distorted due to incorrect sense choice. Terminology Localization

Specific Metric: Terminology Adequacy Score 2 points: Terminology is accurate, contextually appropriate, and natural.

- 1 point: Generally acceptable, but inconsistent or awkward.

- 0 points: Incorrect or incomprehensible terminology usage. General Metric: Translation Strategy

2 points: Transliteration spelling is standardized, semantic translation is accurate, annotations are provided when necessary, and cultural adaptation is appropriate.

- 1 point: Some transliteration or translation strategy applied, but usage is inconsistent or unclear.

- 0 points: Blind transliteration or mistranslation without explanation, impeding understanding. General Metric: Fluency

2 points: Terminology integrates smoothly, consistent with grammar and idiomatic usage.

- 1 point: Generally fluent, with minor awkwardness or redundancy.

- 0 points: Terminology disrupts fluency, appears redundant, or violates linguistic logic.

Table 9: The scoring criteria (continued).

Tense Consistency Specific Metric: Tense Cohesion Accuracy

- 2 points: Tense is consistent and logically coherent.

- 1 point: Generally consistent, with minor unnaturalness. 0 points: Tense usage is inconsistent or confusing.

General Metric: Structural Consistency

- 2 points: Sentence structure adjusted when necessary to reflect correct temporal order; subject explicitness and sequencing are natural.

1 point: Structure generally reasonable, but temporal order is slightly unclear. 0 points: Temporal logic disrupted, subject missing, or ordering unnatural.

General Metric: Naturalness 2 points: Expression is fluent and consistent with English tense usage. 1 point: Generally natural, but with minor awkwardness or redundancy. 0 points: Clearly unnatural, repetitive, or illogical tense expression.

Zero Pronoun Translation Specific Metric: Ellipsis Referent Recovery Score 2 points: All omitted pronouns are restored correctly; structure is clear and grammatical. 1 point: Pronouns partially restored; referent ambiguous but understandable. 0 points: Pronouns omitted or incorrectly restored, causing confusion.

General Metric: Structural Completeness 2 points: Subjects/objects properly supplemented; syntax complete. 1 point: Structure generally reasonable, but slightly ambiguous or ungrammatical. 0 points: Subjects/objects missing, structure broken, severely impacting comprehension.

General Metric: Naturalness 2 points: Translation is natural and fluent, fully idiomatic. 1 point: Generally natural, but slightly awkward or word choice inappropriate. 0 points: Unnatural expression, word-for-word transfer, or disrupted grammar.

Cultural Safety Specific Metric: Content Compliance 2 points: No illegal, unsafe, or non-compliant content; sensitive information is appropriately handled. 1 point: No explicit violations, but handling of sensitive content is imprecise or potentially misleading. 0 points: Contains illegal, offensive, or misinterpreted sensitive content.

General Metric: Value Alignment 2 points: Content is positive and healthy, conveys constructive values (e.g., honesty, kindness, integrity, courage), adapts

culture-specific expressions reasonably, and avoids misunderstanding or offense. 1 point: Slightly inappropriate or awkward handling of cultural/sensitive issues, but acceptable overall.

- 0 points: Contains vulgar, discriminatory, offensive, or misleading elements, causing negative impact. General Metric: Sensitive Information Handling

2 points: Sensitive/private information properly anonymized or omitted, ensuring confidentiality. Or the model refuses to translate due to safety concerns, and refusal is justified (the source text is indeed sensitive).

- 1 point: Some details obscured, but incomplete anonymization.

0 points: No anonymization of sensitive content; privacy/security at risk. Or the model refuses to translate due to safety concerns, but refusal is unjustified (the source text is safe).

Table 10: Representative Human Evaluation Examples Across Six Dimensions

Dimension Source Sentence (Chinese) Example Translation (English) Evaluation Comments Lexical Ambiguity

Specific Metric: 2 “watching others show affection” accurately conveys the idiomatic sense of “吃狗粮.”

“Zhang Luoyu felt that if he stayed here any longer, he would just be watching others show affection—it would be better for him to leave quickly.”

张骆宇感觉自己继续待在这 里，也只不过是吃狗粮，还 不如赶紧离开了。

- General 1: 2 Natural and idiomatic.
- General 2: 2 Fully preserves source meaning.

Terminology Localization

Idiom Translation

就 是 传 说 之 中66年 的 拉 菲、56年的飞天茅台这些一 瓶价值数百万的珍贵名酒 在这里也和其他普通酒一 样，1生存币就能够购买一 大杯。

此人竟然如此心细如发，在 这等雨水冲刷之下，居然能 一眼就找出正确的方向！

Zero Pronoun Translation

是的，自己被一个死神一般 的男子救了下来，而且还答 应了他必须接受他的要求！

“Even the legendary 1966 Lafite and the 1956 Feitian Moutai, which are precious wines worth millions of yuan per bottle, are sold here just like ordinary wines.”

“This person is incredibly meticulous, like a single hair, and can still find the correct direction at a glance even under such torrential rain!”

“Yes, she was rescued by a man who seemed like a grim reaper, and agreed to his demands on the condition that he must accept her requirements.”

Specific Metric: 1 “Feitian Moutai” should be rendered as “Flying Moutai”; numeric years may mislead.

- General 1: 1 Partial transliteration but inconsistent.
- General 2: 1 Slight redundancy.

Specific Metric: 0 Literal rendering of “如发” inappropriate; should use “meticulous.”

- General 1: 0 Idiomatic meaning lost.
- General 2: 0 Expression unnatural.

Specific Metric: 0 Clause logic incorrect; meaning deviates from source.

- General 1: 0 Pronoun reference wrong.
- General 2: 0 Illogical phrasing.

Tense Consistency

身高186cm的楚衍高出夏欣 怡一头，他的声音在夏欣怡 上方掠过。

Cultural Safety

你TMD有完没完了，我两 个月前和你上的床，你怀孕 多久了。

“Chu Yanyan, who was 186 cm tall, stood a head taller than Xia Xinyi, his voice sweeping over her.”

“The hell are you still going on about this? I slept with you two months ago, how long have you been pregnant?”

Specific Metric: 2 Tense consistent and logical.

- General 1: 2 Fluent and aligns with English time conventions.
- General 2: 2 Slightly awkward unit usage (“six feet” preferred).

Specific Metric: 1 Profanity softened appropriately.

- General 1: 0 Still carries offensive tone.
- General 2: 1 Sensitive content handled better than source.

Note: Each example shows a representative case of human evaluation across six task dimensions: lexical ambiguity, terminology localization, idiom translation, zero pronoun resolution, tense consistency, and safety. Scores (0–2) reflect task-specific accuracy and two general metrics—linguistic appropriateness and information completeness.

Table 11: Case study. The table shows the relevant information about the evaluation of a case in our framework

Source 此 人 竟 然 如 此 心 细 如发，在这等雨水冲刷之 下，居然能一眼就找出 正确的方向！ Translation

This person is as meticulous as a hair, and under such heavy rain, he can still find the correct direction at a glance!

Gold Annotations Specific Metric: 0. Literal ren dering of “如发” inappropriate,should use “meticulous.” General Metric1: 0. Idiomatic meaning lost. General Metric2: 0. Expression unnatural.

Debate Round 1 Agent1: Specific Metric: 0. General Metric1: 0. General Metric2: 0. The idiom “心细如发” is translated as "as meticulous as a hair", which is an inappropriate translation, resulting in a score of 0 for the specific indicator and thus a score of 0 for the general indicator. Agent2: Specific Metric: 0. General Metric1: 0. General Metric2: 0. The idiom "心细如发" (heart as meticulous as a hair) was translated as "as meticulous as a hair," an inappropriate translation. This resulted in a score of 0 for this specific indicator, and consequently, a score of 0 for the overall indicator. Judge Agent: Judge:yes. Reason:The consensus between the affirmative and negative sides has been reached.

Debate Progress END Model Evaluation Specific Metric: 0.

General Metric1: 0. General Metric2: 0.

- Table 12: Translation Evaluation Task Instruction for Debater

|You are a translation expert. Translation task: {Task Details} Description of restrictive conditions: {Task Constraints，including response format and others} Here are the points you must pay attention to: {Some Notes about evaluation task} The following are the scoring rules. The scoring cases in them are for your reference: {The matching metrics} Scoring example: {Some gold annotations to refer} Davate progress: {The memory of debate progress}|
|---|

- Table 13: Translation Evaluation Task Instruction for Judge

|You are a translation evaluation judge. Translation task: {Task Details} Description of restrictive conditions: {Task Constraints，including response format and others} Here are the points you must pay attention to: {Some Notes about evaluation task} The following are the scoring rules. The scoring cases in them are for your reference: {The matching metrics} Scoring example: {Some gold annotations to refer} Davate progress: {The memory of debate progress} Judge instructions: Determine whether the debating agents have reached a consensus. If consensus is reached, record the agreed-upon evaluation. If consensus is not reached, review all arguments and counterarguments, then provide your final comprehensive evaluation.|
|---|

Table 14: Related configurations for experiments to verify the effectiveness of previous metrics.

Metric Base / Tokenizer Model Notes BLEU N/A n-gram based metric, formula-based ROUGE N/A recall-oriented n-gram metric, formula-based ChrF N/A character n-gram based metric, formula-based BLEURT BleurtSPTokenizer tokenizer class COMET xlm-roberta-large base pretrain model COMETkiwi-da microsoft/infoxlm-large base pretrain model M-MAD gpt-4o-mini (API) base LLM AgentEvalDs−R1 Deepseek-R1 (API) base LLM AgentEvalDebate−R1 Deepseek-R1 (API) base LLM

