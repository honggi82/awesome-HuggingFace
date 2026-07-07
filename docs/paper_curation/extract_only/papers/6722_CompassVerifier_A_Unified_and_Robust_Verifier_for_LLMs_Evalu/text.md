# arXiv:2508.03686v1[cs.CL]5Aug2025

## CompassVerifier: A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward

Shudong Liu1,2,∗, Hongwei Liu1,∗, Junnan Liu1, Linchen Xiao1, Songyang Gao1, Chengqi Lyu1, Yuzhe Gu1, Wenwei Zhang1, Derek F. Wong2,†, Songyang Zhang1,†,ℵ, Kai Chen1,† 1Shanghai AI Laboratory 2NLP2CT Lab, University of Macau

#### Abstract

Answer verification is crucial not only for evaluating large language models (LLMs) by matching their unstructured outputs against standard answers, but also serves as the reward model to guide LLM optimization. Most evaluation frameworks rely on regularized matching or employ general LLMs for answer verification, which demands extensive, repetitive customization for regex rules or evaluation prompts. Two fundamental limitations persist in current methodologies: 1) the absence of comprehensive benchmarks that systematically evaluate verification capabilities across different LLMs; and 2) the nascent stage of verifier development, where existing approaches lack both the robustness to handle complex edge cases and the generalizability across different domains. In this work, we develop CompassVerifier, an accurate and robust lightweight verifier model for evaluation and outcome reward. It demonstrates multi-domain competency spanning math, knowledge, and diverse reasoning tasks, with the capability to process various answer types, including multi-subproblems, formulas, and sequence answers, while effectively identifying abnormal/invalid responses. We introduce VerifierBench benchmark comprising model outputs collected from multiple data sources, augmented through manual analysis of meta error patterns to enhance CompassVerifier. We anticipate that CompassVerifier and VerifierBench will facilitate answer verification, evaluation protocols, and reinforcement learning research. Code and dataset are available at https://github.com/open-compass/CompassVerifier.

#### 1 Introduction

Answer verification plays a critical role in the evaluation and training of large language models (LLMs), particularly for objective questions with verifiable answers (Achiam et al., 2023; Yang et al., 2024; Liu et al., 2024a;b). At the evaluation level, it enables precise measurement of performance differences across models (Chang et al., 2024); at the training level, it serves as a quality check for self-improvement (Hosseini et al., 2024; Song et al., 2025). With the rapid development of large reasoning models (LRMs) and reinforcement learning (RL), answer verification has further become a key component in constructing rule-based rewards, providing feedback signals to guide model optimization and iteration (Guo et al., 2025; OpenAI, 2024c; Luong et al., 2024; Wang et al., 2025a; Zhong et al., 2025).

Existing answer verification methods can be broadly categorized into two types. The first type relies on regularized string matching, such as extracting content following “The answer is” to compare with reference answers, or using tools like math-verify (huggingface, 2024) to check formula equivalence in mathematical tasks. The second type employs general LLMs for consistency judgment, where a specific prompt is designed to instruct the model to evaluate the alignment between candidate and reference answers. However, both approaches

∗Equal contribution. ℵProject lead. †Corresponding authors. Work done during Shudong’s internship at Shanghai AI Laboratory. Email: nlp2ct.shudong@gmail.com; {liuhongwei,zhangsongyang}@pjlab.org.cn

[Figure 1]

[Figure 2]

[Figure 3]

Questions

###### Ground Truth

Models

[Figure 4]

Question

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

LLM Response

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Formula

Short Answer

Math

Response Generation

Reasoning

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Ground Truth

[Figure 22]

Science

Knowledge

Sequence

[Figure 23]

Multi Choice

[Figure 24]

Stage 3 Annotation and Analysis

Stage 2 Multi-prompt Voting

Stage 1 Multi-expert Voting

###### Error Patterns

| | |
|---|---|
|Inco|nsist|
| | |

- A (Correct)

- - Complex formulas equivalence
- - …

- B (Incorrect)

- - Wrong option content
- - …

- C (Invalid)

[Figure 25]

Question LLM Response Ground Truth

- A (Correct)
- B (Incorrect)
- C (Invalid) Judgment Reason

[Figure 26]

[Figure 27]

- A (Correct)
- B (Incorrect)
- C (Invalid)

- A (Correct)
- B (Incorrect)
- C (Invalid)

[Figure 28]

Manual

[Figure 29]

Inconsist

| | |Consist|
|---|---|---|
| | | |

Deduplication and filtering

[Figure 30]

A (Correct) Consist B (Incorrect)

[Figure 31]

[Figure 32]

[Figure 33]

Easy Data

- - Repeat
- - …

Training Data Pool

VerifierBench Test

Figure 1: Overview of VerifierBench pipeline. Using OpenCompass (OC-Contributors, 2023), we collected more than 1 million LLM responses, applying multi-stage, multi-model verification with tool-assisted cleaning and filtering to create VerifierBench’s test/base training sets and catalog common verification error patterns.

suffer from significant limitations: the former requires repetitive customization of matching rules for different tasks and is prone to verification failures due to extraction errors; the latter demands frequent prompt adjustments to accommodate diverse tasks, domains, and answer types, while also facing the risk of misjudgment caused by model hallucination. Meanwhile, there is still no challenging benchmark available to evaluate and distinguish the verification capabilities of different models, nor to guide the development and iteration of verifiers.

In this paper, we establish a systematic framework for evaluating and training answer verification systems. We first introduce VerifierBench, a challenging benchmark for answer verification that aggregates numerous samples where rule-based methods frequently err or LLMs tend to produce incorrect judgments or hallucinations. We integrated over one million data samples through the OpenCompass (OC-Contributors, 2023) evaluation framework, encompassing responses from more than 50 models across 15 carefully selected datasets. Following large-scale data collection, each sample underwent a multi-stage filtering pipeline culminating in rigorous domain expert review and calibration. VerifierBench facilitates precise measurement of verification capabilities across diverse models, addressing complex scenarios where both rule-based matching and general models often fail, and offering manually analyzed summaries of prevalent error patterns.

We further present CompassVerifier, a series of lightweight yet robust and accurate verification models. The training data originates from three key sources: 1) The original training set from VerifierBench, which undergoes multi-model validation with simple, easily verifiable samples removed; 2) Formula-enhanced data, where we leverage the powerful DeepSeek-V3 model to generate numerous equivalent complex formulas with corresponding reasoning processes to improve formulaic answer evaluation; 3) Hallucination-specific data, where we systematically analyze failure patterns from human validation cases and synthesize targeted training samples to address common hallucination errors.

Our contributions are threefold:

- • We propose VerifierBench, a novel and challenging benchmark meticulously designed for fine-grained evaluation of verification abilities.
- • We develop CompassVerifier, a series of robust and efficient verification models enhanced through our three proposed techniques, achieving state-of-the-art performance across diverse domains and tasks. CompassVerifier can also effectively serve as a reward model in RL training, delivering more precise and reliable feedback signals for policy optimization.

- • Through a systematic analysis of prevalent failure modes in LLM-based verification, including characteristic hallucination phenomena and error propagation, we derive actionable insights aimed at advancing the design and robustness of future verification systems.

#### 2 Related Work

##### 2.1 Answer Verification

Unlike traditional discriminative models with well-defined classification labels, the unstructured outputs of generative LLMs pose unique verification challenges (Cobbe et al., 2021). Current approaches to verifying LLM-generated answers can be broadly categorized into outcome verification and process verification (Kawabata & Sugawara, 2024; Zhang et al., 2025).

Outcome verification focuses on assessing the correctness of final answers, typically through string-based pattern matching (OC-Contributors, 2023; Gao et al., 2024; OpenAI, 2023). Common practice instructs LLMs to output answers in predefined formats for characterlevel comparison with ground truth. For formulaic answers, specialized tools like MathVerify (huggingface, 2024) have been developed to handle equivalence checking. However, due to the inherent unpredictability of LLM outputs, such methods often suffer from matching failures or inaccuracies. Many studies thus employ general LLMs as verifiers via tailored prompts. While effective, both methods demand task-specific customization through either regex patterns or verified prompts, creating labor-intensive workflows. Process verification, requiring detection of reasoning errors in intermediate steps, has seen recent advances in both LLM-based verifiers and evaluation benchmarks (Lu et al., 2024; o1 Team, 2024; Lightman et al., 2023; Zheng et al., 2024; Zhou et al., 2024). However, process verifiers remain less frequently adopted in evaluations due to instability and high resource costs, and have not demonstrated substantially superior performance compared to outcome verifiers in RL.

We focus on scalable and robust outcome verification by developing a unified verifier that serves dual purposes: 1) as an evaluation model for benchmarking model performance, and 2) as a real-time reward model for RL training. By addressing the limitations of existing methods, such as ad-hoc prompt engineering and brittleness to output variations, CompassVerifier prioritizes efficiency, generalizability, and reliability across diverse tasks.

##### 2.2 LLM-as-a-Judge

The comprehensive capabilities of LLMs enable them to serve as cost-effective alternatives to human experts in evaluation tasks, a concept known as “LLM-as-a-Judge” (Gu et al.,

- 2024; Li et al., 2024a), which can be categorized into two approaches: subjective judgment and objective judgment.

Subjective judgment typically operates in scenarios without ground-truth answers, where LLMs score individual responses (Pointwise) (Zhu et al., 2025) or express preferences between paired responses (Pairwise) (Wang et al., 2024a). This requires the LLM to evaluate various aspects of responses, including usefulness, harmlessness, and creativity, and even identify reasoning stepwise errors in the responses (Cao et al., 2024; Li et al., 2024c; 2023). Recent studies also employ RL and inference-time scaling like generative critiques, long-CoT, and multi-sampling voting for judgment, albeit with high computational costs (Liu et al.,

- 2025; Shi & Jin, 2025). objective judgment is a more straightforward approach, requiring only the evaluation of response correctness against ground-truth. Beyond simple string matching, the prevalent method employs large-scale LLMs with carefully designed evaluation prompts for judgment. Recently, to enable smaller models to achieve comparable verification capabilities to large LLMs, Chen et al. (2025) proposes xVerify and its accompanying benchmark, which trains smaller verifier models by distilling GPT-4o’s capabilities. Other concurrent studies have also focused on distilling verification capabilities from large models to smaller ones to achieve better cost-effectiveness (Ma et al., 2025; Su et al., 2025).

We claim that objective judgment with ground-truth has yet to reach maturity, lacking both challenging benchmarks to discriminate model abilities and robust unified models. To address these gaps, we are committed to developing VerifierBench to rigorously test different models’ verification capabilities and CompassVerifier to provide the research community with an accurate evaluation tool.

#### 3 VerifierBench

The primary challenge in verifier development lies in the lack of comprehensive benchmarks and rigorous evaluation methodologies. Large-scale commercial models are often preferred for answer-matching tasks due to the prevailing assumption of scaling laws. However, critical questions remain unanswered: 1) To what extent do answer matching and objective judgment tasks adhere to scaling laws? 2) How should we balance model performance against computational costs in verification?

To answer these questions, in this work, we present VerifierBench, a systematic benchmark for evaluating diverse models’ judgment and verification capabilities. VerifierBench addresses this gap through: 1) Large-scale data collection for answer matching (3.1); 2) Multi-round validation involving multiple LLMs and human annotators (3.2); 3) Case analysis of typical error patterns to identify failure modes (3.3).

##### 3.1 Data Collection

The crux of the answer verification task hinges on its capacity to encompass a comprehensive range of verifiable answer types and heterogeneous model responses. To comprehensively gather such data, we employed the OpenCompass framework (OC-Contributors, 2023) to conduct large-scale evaluations across multiple models and datasets. Our systematic approach yielded more than 1,325,293 samples covering four key domains: knowledge, mathematics, science, and general reasoning. The collected data features:

- • Answer Type Diversity: Multiple response formats including multiple-choice question options, mathematical formulations, short texts, multi-subproblem items, and long-sequence responses, etc.
- • Prompt Variability: Input prompts covering few-shot, zero-shot, and dataset-specific formatting requirements.
- • Response Characteristics: Model outputs ranging from short and long chain-of-thought (CoT) answers to direct responses and anomalous outputs (e.g., repetitions, truncations).
- • Diverse Model Coverage: Comprehensive representation across commercial LLMs, opensource LLMs, and emerging LRMs, spanning diverse model scales.

Formally, our collected data consists of triplets: D = {(qi, ai∗,rim)}iN=1, where qi ∈ Q represents the i-th question, ai∗ ∈ A denotes the corresponding reference answer, rim ∈ R is the response generated by model m ∈ M. The primary objective of VerifierBench construction is to augment these triplets with verification labels, resulting in verified quadruples:

DVerifierBench = {(qi, ai∗,rim, vi)}iN=1, (1)

where vi ∈ {Correct,Incorrect,Invalid} is the verification label indicating the correctness of rim with respect to ai∗. Notably, during data collection and curation, we identified numerous responses exhibiting abnormal or exceptional behaviors. These include abruptly truncated outputs, excessive repetition, and cases where models refused to answer due to ethical considerations or other constraints. We therefore categorize such instances as invalid responses to enable a more fine-grained evaluation.

##### 3.2 Data Construction Pipeline

Our multi-stage verification pipeline, integrating LLMs, human annotators, and rule-based tools, efficiently identifies high-value training and testing samples from a large collected dataset.

Multi-Expert Voting. Initially, samples undergo direct verification (no CoT reasoning) by Qwen2.5-Instruct models (7B, 14B, 32B). Samples with consensus are deemed trivial cases reliably handled by weaker models and are removed, offering minimal value. For mathematical domains (Math, GSM8K, and AIME datasets), we also incorporated MathVerify (huggingface, 2024) as an additional expert verifier.

Multi-prompt Voting. Disputed samples advance to a second verification stage, where DeepSeek-V3 is employed with multiple prompts to generate diverse CoT reasoning paths. Consensus samples from this stage, representing moderately challenging instances, constitute our training pool. Our experiments revealed significant challenges in developing a universal verification prompt applicable across all datasets, evidenced by substantial residual disagreements after the second verification round. To address this, we implemented an additional verification phase for selected datasets, featuring domain-optimized prompts. For instance, the Chinese SimpleQA dataset required specially crafted Chinese-language prompts to achieve reliable verification outcomes.

Human Annotation and Analysis. The remaining disputed samples are human-annotated, with high-value cases primarily allocated to the test set. For the VerifierBench test set, we systematically excluded proof-based questions, open-ended problems, and numerical answers with ambiguous acceptability thresholds. These non-binary judgment cases, requiring specialized verification tools or domain expertise, are deferred to future work, ensuring VerifierBench focuses on clearly verifiable samples. Finally, we get the VerifierBench dataset, and we also make sure there is no overlap between VerifierBench test set and the train set for training CompassVerifier model.

Identification of Flawed Samples. Human annotation also identified a distinct category: “flawed samples”. Errors in these samples stem not from model deficiencies in problemsolving but from issues inherent to the questions (e.g., ambiguity, incorrect standard answers) or external factors (e.g., improper output truncation, generation of meaningless repetitive text, model refusal to answer). Such flawed samples, if not distinguished, can skew model capability assessment and hinder effective model iteration. These issues are often overlooked in traditional evaluation paradigms. Consequently, we explicitly label these samples as “Invalid” and integrate them into the VerifierBench test set. This approach enables a more granular, multi-dimensional, and realistic perspective for model performance verification.

##### 3.3 Statistics and Analysis

Statistics. We present the statistical characteristics of the VerifierBench test set across three dimensions: label categories (Table 4), problem domains (Table 5), and answer types (Table 6). After filtering and balancing, the dataset composition shows an approximate 4:6 ratio between Category A and B samples, with Category C representing about 7% of the total. Regarding problem domains, general reasoning, and mathematical reasoning constitute the majority, aligning with the current needs of RL training on LLMs. Classified by DeepSeekV3, the answer types comprise seven categories: multiple-choice, numerical values, short answers, formulas, multi-subproblem, sequences, and binary answers. The detailed dataset sources are provided in Table 3, with concrete examples illustrated in Section 7.6.

Error Analysis and Patterns. VerifierBench is designed not merely as a benchmark dataset for model evaluation, but as a comprehensive framework incorporating extensive human analysis and case studies. During annotation, we required annotators to provide detailed judgment rationales in addition to final labels. Through systematic collection and analysis of these rationales, we identified and categorized over 30 meta error patterns (Section 7.4), which represent fundamental causes of mistakes and hallucinations in LLM-based answer

Equivalent transformation

[Figure 34]

Judgment and Filtering

- GT A

- GT B

- GT C

Judgment and Filtering

- GT A

- GT B

- Prompt A

- Prompt B

- Prompt C

- COT A + Judgment

- COT B + Judgment

- COT C + Judgment

- Prompt D Direct Judgment

Response Generation

Ground Truth

[Figure 35]

[Question, Response, GT] Augmented data

Augmented

Complex Formula Augmentation

[Figure 36]

Shuffle Mixing Perturbation

[Figure 37]

Base

Training

Meta Error Patterns

[Figure 38]

[Figure 39]

- • Reference answer lists multiple options, LLM only gave one → Incorrectly marked as wrong
- • LLM appears to select the right option, but the chosen content is wrong → Incorrectly marked as correct
- • In simplification problem, LLM give unsimplified but equivalent formula → Incorrectly marked as correct

[Figure 40]

[Figure 41]

Template-guided Data Synthesis

Judgment and Filtering

Synthetic

CompassVerifier

All Training Data Pool

[Question, Response, GT] Synthetic data

Generalizability Augmentation

Error-Driven Adversarial Augmentation

Figure 2: Overview of CompassVerifier training pipeline.

verification. For example, while mathematically equivalent formulas are conventionally accepted as correct answers by LLMs or tools, they should be rejected for expression simplification problems. Similarly, for questions admitting multiple valid answers listed in the reference answer, a model response matching any one option should be considered correct, rather than complete coverage. We have found these meta patterns invaluable for both diagnostic analysis and targeted model improvement, and have incorporated them into our training framework.

#### 4 CompassVerifier

CompassVerifier is designed to deliver efficient, high-performance, and robust answer verification. The system leverages filtered (question, reference answer, model response) triples from VerifierBench with golden judgments as training supervision. We also propose three key techniques to drive its performance: Complex Formula Augmentation enhances formula variants verification, Error-Driven Adversarial Augmentation fortifies against failures, and Generalizability Augmentation ensures cross-domain and cross-prompt applicability.

- Figure 2 shows the whole pipeline of training CompassVerifier. Details of the composition of the training Data in Section 7.8.

##### 4.1 Error-Driven Adversarial Augmentation

To address potential annotation inaccuracies in our filtered data (see Section 3.2), we employ a three-phase adversarial augmentation strategy.

Human-in-the-Loop Analysis. Domain experts manually verify 5,000 annotated samples, identify and document failure rationales such as LLM misunderstandings of task constraints, misinterpretation of critical information in questions, and divergent penalty thresholds among judge models.

Pattern Clustering. We apply density-based clustering to these rationales, revealing over 20 high-impact error categories, particularly vulnerabilities in perspective-taking and format adherence. Analysis and details are shown in Section 7.4.

Meta-Judge Template Generation. For each error cluster, we develop structured templates that encode: 1) Question Characteristics (domain-specific requirements, content/format constraints) and 2) Response Error Patterns (failure types, localization, severity).

This aligns model judgments with human values and improves robustness against: (1) overstrict format-based rejection, (2) underpenalization of conceptual errors in fluent responses, and (3) context-sensitive scoring variations.

##### 4.2 Complex Formula Augmentation

Verifying answers in domains such as the natural sciences is challenging due to the prevalence of complex expressions. These expressions often exhibit diverse notational conventions

(e.g., symbolic, algebraic, floating-point, integer). Consequently, automated verifiers lacking robust mathematical equivalence understanding may erroneously reject semantically correct responses that differ superficially from reference solutions.

To address this issue, we introduce a Complex Formula Augmentation strategy that systematically generates multiple, notation-variant answers for each problem instance. Our procedure is as follows:

Reference Normalization. For each original question–answer pair in our dataset, we first convert the reference answer into a canonical representation, normalizing numeric precision and symbolic structure.

Variant Generation. We leverage the DeepSeek-v3 (Ma et al., 2025) to produce between one and three alternative formulations of the canonical answer. These variants include:

- 1) Symbolic rearrangements (e.g., rationalizing denominators, applying algebraic identities).
- 2) Precision-preserving floating-point expansions. 3) Equivalent integer or fraction representations. We enforce strict constraints to avoid precision loss and ensure each variant remains mathematically equivalent to the original answer within the problem context.

Quality Control. All generated variants are automatically checked for equivalence using a symbolic algebra engine, and a subset is manually reviewed by subject-matter experts to confirm correctness and naturalness of presentation.

By exposing the verifier to diverse but equivalent formulae, we markedly improve its ability to recognize correct answers regardless of notational differences, thereby reducing false negative rates in formula-intensive tasks.

##### 4.3 Generalizability Augmentation

Existing verifier models often rely on task-specific prompts, limiting their generalizability across different problems and subtle answer variations (e.g., numerical precision in TheoremQA (Chen et al., 2023)). To address this, we propose a Generalizability Augmentation strategy to enhance adaptability by systematically expanding prompt and response diversity in training data. We collect diverse prompts from public datasets (e.g., TheoremQA, GPQA (Rein et al., 2024), GAOKAOBench (Zhang et al., 2023)) and real-world scenarios, covering over 20 task types. For each prompt type, we design multiple variants, varying questioning styles, context lengths, linguistic registers, and instruction granularity. Our augmentation employs two key techniques:

Prompt Rewriting and Perturbation. We use LLMs (e.g., DeepSeek-v3) to automatically generate paraphrases, structural modifications, and detail-enriched prompt variants, while maintaining consistency with the final judgment. Furthermore, during training, we introduce prompt random sampling, dynamic mixing, and a prompt-invariance mechanism to prevent overfitting and encourage consistent judgments across different prompt formulations, thereby enhancing generalization.

Long-context Generalization. To improve robustness in long-context scenarios, we apply various perturbations to responses collected from LRMs (e.g., DeepSeek-R1 and its distilled variants) in the training set, including truncating different portions (e.g., first 20%, 40%, 60%) of the thinking process, replacing thinking tags (e.g., <think> or </think>) with alternative labels, or removing them entirely, while ensuring the final judgment remained consistent with the original response.

#### 5 Experiments

Baselines and Setup. We conduct comprehensive evaluations on VerifierBench across various model scales of CompassVerifier, ranging from 3B to 32B parameters. Baseline models include: (1) general LLMs such as Qwen2.5 (Yang et al., 2024), Qwen3 (Yang et al.,

- 2024), DeepSeek-V3 (Guo et al., 2025), and GPT-4o (OpenAI, 2024a); and (2) two recently

- Table 1: Main results on the VerifierBench benchmark. For fair comparison, we treat the “Invalid” instances in VerifierBench as incorrect labels, presenting results in a binary classification framework. We report Accuracy and F1 scores (%) across four categories and their average.

Math General Reasoning Knowledge Science Average Acc. F1 Acc. F1 Acc. F1 Acc. F1 Acc. F1

Model

General LLMs

|Qwen2.5-7B-Instruct 53.0 30.0 58.9 51.1 55.8 50.7 64.0 36.6 Qwen2.5-14B-Instruct 51.6 37.4 57.3 44.9 50.9 37.8 70.0 47.9 Qwen2.5-32B-Instruct 53.1 31.6 64.6 42.2 60.0 46.4 77.4 48.8 Qwen2.5-72B-Instruct 57.0 37.5 61.4 49.0 70.0 68.5 77.9 60.5<br><br>|57.9 42.1 57.4 42.0 63.8 42.2 66.6 53.9|
|---|---|
|Qwen3-8B 53.0 51.6 61.6 61.8 63.8 69.4 57.9 42.9 Qwen3-14B 65.1 44.1 76.8 66.7 69.8 66.7 81.6 56.8 Qwen3-30B-A3B 59.7 62.4 63.4 63.2 61.5 64.4 59.5 48.7 Qwen3-32B 64.4 54.6 74.9 70.3 68.7 69.5 74.7 52.8 Qwen3-235B-A22B 64.2 53.9 78.5 73.7 67.4 73.1 74.0 50.0<br><br>|59.1 56.4 73.3 58.6 61.0 59.7<br><br>70.7 61.8<br>71.0 62.7<br>|
|GPT-4.1-2025-04-14 66.6 42.0 85.4 79.5 84.0 82.9 88.4 75.0 GPT-4o-2024-08-06 63.9 34.9 78.7 68.2 79.8 78.3 83.2 54.9 DeepSeek-V3-0324 69.4 54.7 81.5 76.6 80.6 81.2 84.7 68.5|81.1 69.8 76.4 59.1 79.1 70.3|

Verifier Models

xVerify-0.5B-I 61.7 42.6 84.0 78.5 87.1 86.2 86.3 72.6 79.8 70.0

- xVerify-8B-I 64.3 42.6 84.3 78.9 86.1 85.1 88.7 74.9 80.8 70.4
- xVerify-9B-C 64.3 48.0 82.8 77.0 82.7 81.7 86.3 69.8 79.0 69.1 Tencent-Qwen2.5-7B-RLVR 71.2 55.3 80.9 73.8 78.0 76.8 84.0 62.6 78.5 67.1

CompassVerifiers

CompassVerifier-3B 76.3 71.0 88.9 85.9 87.9 87.7 86.8 77.1 85.0 80.4 CompassVerifier-7B 79.4 74.8 89.9 87.7 92.8 92.6 87.9 78.5 87.5 83.4 CompassVerifier-32B 84.1 80.8 92.1 90.3 95.1 94.8 91.8 84.7 90.8 87.7

90

| |CompassVerifier-3B<br><br>CompassV|erifier-7B<br><br>CompassVerifier-3|2B|
|---|---|---|---|
|xVerify-0.5|B<br><br>xVerify-|8B xVerify-9B<br><br>|DeepSeek-671B|
| |Qw<br><br>Tencent-Qwen2.5-7B|en3-14B<br><br>Qwen3-32B|Qwen3-235B|
| |Qwen3-8|Qwen<br><br>B<br><br>Qwen3-30B|2.5-72B|
| |Qwen2.5Qwen2.5|7B<br><br>-14B<br><br>Qwen2.5-32B| |
| | | | |

80

70

F1Score

60

50

40

100 101 102

Model Size (Billions of parameters) - Log Scale

- Figure 3: Model performances with size on VerifierBench. We show the F1 score in main results.

proposed specialized verifier models: xVerify (Chen et al., 2025) and Tencent-Qwen2.5-7BInstruct-RLVR (Su et al., 2025). We ask the model directly generate the final judgment of the given response and report F1 and Accuracy as metrics. More evaluation and training details are shown in Section 7.3.

##### 5.1 Main Results

From the Perspective of the Domain. We show the main results of VerifierBench in Table 1. Our CompassVerifier establishes new state-of-the-art performance across all VerifierBench

Qwen3-8B

GPT-4o

- Qwen2.5-7B

- Qwen3-8B

Tencent-Qwen2.5-7B-RLVR

CompassVerifier-7B-Base

Qwen3-32B

CompassVerifier-7B

CompassVerifier-7B

+ Complex Formula Augmentation

| |
|---|

DeepSeek-V3

CompassVerifier-32B

xVerify-8B-I

+ Error-Driven Adversarial Augmentation

| |
|---|

Average

Average

+ Both Augmentations

90

|25<br><br>50<br><br>75|0 C<br><br>|
|---|---|
| | |

3.5%

Boolean

Sequence

80

2.7%

2.4%

Performance

85

40

3.6%

3.0%

2.2%

Multi-subQ

Multi-choice

A (Correct)

(Invalid)

0

80

Numerical

Formula

75

B (Incorrect)

Short Text

Accuracy F1

(a) Seven answer types.

(b) Ternary class labels.

Figure 5: Ablation study on CompassVerifier-7B with different training technologies.

- Figure 4: Results (F1) on VerifierBench across 7 answer types and 3 correctness labels.

categories, achieving 84.1–95.1% accuracy and 80.8–94.8% F1-score in the 32B configuration. Three findings emerge: 1) As shown in Figure 3, verification capability exhibits progressive improvement with increasing scale, demonstrating accuracy gains from 85.0% to 90.8% and F1-score improvements from 80.4% to 87.7% as parameters scale from 3B to 32B. 2) Verification-specific architectures yield substantial gains: CompassVerifier-7B surpasses the similarly-sized original Qwen2.5-7B-Instruct by an absolute F1-score improvement of 41.3%. 3) Despite progress, mathematical verification remains challenging (80.8% best F1score vs. 94.8% for knowledge), highlighting persistent gaps in stepwise logical validation. Our smallest 3B variant outperforms GPT-4.1 by an absolute F1-score improvement of 10.6%, demonstrating parameter efficiency. Consistent performance across domains further underscores the model’s robustness. For instance, our CompassVerifier-32B model achieves high F1-scores across all evaluated categories. Such consistency indicates a well-generalized verification capability, effectively handling diverse types of information and reasoning processes.

From the Perspective of the Answer Type. Figure 4a demonstrates the performance comparison of similarly-sized models across different answer/question types. Notably, CompassVerifier-7B achieves consistent improvements across all categories. As evident from the results, multiple-choice questions emerge as the easiest category, with most models attaining strong performance, a finding attributable to their prevalence in evaluation benchmarks. However, baseline models show marked deficiencies in handling formulabased answers, multi-subquestions, and sequential answers, particularly struggling with sequential answers where none exceed 40 F1-score. This likely stems from the inherent complexity of sequential answers, which often require element-by-element matching of multiple components, significantly increasing verification difficulty. These challenging cases represent precisely the focus of CompassVerifier and constitute critical directions for future research. The complete results are presented in Table 7.

##### 5.2 Analysis of CompassVerifier

Beyond Binary Verification: Identifying Invalid Responses. Figure 4b presents the threeclass classification performance of six top-performing models. Notably, even advanced general LLMs like GPT-4o and DeepSeek-V3 without task-specific training exhibit significant performance bias, demonstrating substantially better results on categories A and B compared to C. Our manual analysis reveals that general models show particular insensitivity to duplicated patterns or truncated responses. To address this, we implemented a duplicate string detection script during data filtering (Section 3.2). Crucially, we argue that Category C requires distinct treatment as they are particularly susceptible to reward hacking in RL training scenarios. Full results of the ternary classification performance are shown in Table 8.

Impact of Data Augmentation Components. Figure 5 details the impact of our data augmentation strategies on CompassVerifier-7B. The baseline model (CompassVerifier-7B-Base) achieves 84.0% accuracy and 79.8% F1. Introducing Complex Formula Augmentation alone

- Table 2: Performance on VerifyBench using different prompt strategies. We report Accuracy and F1 scores (%) for both model-specific prompts and the standard VerifyBench prompts.

Model-specific Prompt VerifyBench Prompt Acc F1 Acc F1

Model

General LLMs

Qwen2.5-7B-Instruct 65.4 39.8 60.9 45.0 Qwen2.5-32B-Instruct 78.8 58.9 72.0 55.8 Qwen2.5-72B-Instruct 78.5 61.7 63.0 50.0 DeepSeek-V3 81.8 62.2 78.6 60.9

Verifier Models

xVerify-0.5B-I 77.9 66.2 - -

- xVerify-8B-I 83.2 70.7 - -
- xVerify-9B-C 83.2 71.0 - Tencent-Qwen2.5-7B-RLVR 82.4 68.9 - -

CompassVerifiers

CompassVerifier-3B 87.4 77.4 86.2 75.0 CompassVerifier-7B 88.1 79.0 86.0 73.3 CompassVerifier-32B 89.7 81.1 86.8 74.3

Before RL (Qwen3-4B-Base) Math-Verify Tencent-RLVR Qwen3-14B Qwen2.5-32B-Instruct CompassVerifier-7B CompassVerifier-32B

25

25

|+17.1 +16.9<br><br>+18.5+18.5<br><br>|
|---|
|+14.7|
| |
|+6.2|
| |

| |
|---|
|+14.4<br><br>+14.8<br><br>+13.6<br><br>+15.5+15.4<br><br>|
| |
|+5.4|
| |

|+46.4 +47.1 +47.5 +48.1+49.2<br><br>|
|---|
| |
|+29.0|
| |
| |
| |
| |

80

20

20

Performance(avg@32)

70

15

15

60

10

10

50

40

5

5

30

0

0

AIME 2024

AIME 2025

MATH500

Figure 6: Experimental results of CompassVerifier as a reward model. We employ MathVerify along with various general LLMs and verifier models as reward models for RL training, reporting the avg@32 performance on AIME24, AIME25, and MATH500.

improves accuracy to 86.7% (+2.7) and F1 to 82.8% (+3.0). This demonstrates the strategy’s effectiveness in enhancing the model’s capability to handle diverse formulaic expressions. Similarly, Error-Driven Adversarial Augmentation alone boosts accuracy to 86.4% (+2.4) and F1 to 82.0% (+2.2), underscoring its utility in fortifying the model against previously identified failure modes. Combining both strategies yields the best performance, with accuracy reaching 87.5% (+3.5) and F1 at 83.4% (+3.6), demonstrating their complementary and synergistic contributions to overall verification capabilities. Details are shown in Table 9.

Generalization of CompassVerifier. To evaluate the generalization capability of CompassVerifier, we also conduct tests on the hard subset of VerifyBench (Yan et al., 2025), a recent concurrent work for benchmarking verification abilities. This subset primarily contains standard answers that involve long reasoning COT, making it particularly challenging to verify. Table 2 presents the performance comparison across different models. Here, “Model-specific Prompt” indicates that xVerify/Tencent-RLVR employs their respective training prompts while other models use ours, whereas “VerifyBench Prompt” denotes that all models utilize the same prompt provided with the VerifyBench dataset. Our analysis leads to the following findings: 1) CompassVerifier still outperforms both general LLMs of similar size, specialized verifier models, and even DeepSeek-V3; 2) Due to our Generalizability Augmentation, even under VerifyBench’s prompt (deeper out-of-distribution setting), CompassVerifier maintains robust performance (score >86), while xVerify and Tencent-Qwen2.5-7B-RLVR completely fail to follow instructions.

##### 5.3 CompassVerifier as Reward Model

To validate the efficacy of CompassVerifier as a reward model in RL training, we examine its influence on enhancing the reasoning performance of models trained using RL. Specifically, we utilize GRPO (Shao et al., 2024) to train base LLMs with rule-based verifier Math-Verify (huggingface, 2024) and CompassVerifier and rigorously evaluate the reasoning capabilities of the trained models. We use the challenging Open-S1 (Dang & Ngo, 2025) as the RL training corpus, which can also be considered an out-of-distribution dataset for CompassVerifiers. More experimental settings are provided in Section 7.9.

Comparative results are shown in Figure 6 (Details in Table 10). Experimental results demonstrate that models trained with CompassVerifier outperform the base model, surpass those trained with the rule-based verifier (Math-Verify), and exceed models using general LLMs or alternative verifiers as reward models. This highlights CompassVerifier’s superior potential as a reward model, providing more precise evaluation for rollout trajectories generated in RL training. Additionally, CompassVerifier’s enhanced capacity to provide more effective signals (i.e., rewards) during training substantially improves the convergence efficiency of RL training. The results also reveal a noticeable performance gap between rule-based and model-based verifiers. As the data types and disciplines covered by Reinforcement Learning from Verifiable Rewards (RLVR) (Wang et al., 2025b) training continue to expand, rule-matching tools have become increasingly inadequate, which precisely motivated the development of CompassVerifier.

#### 6 Conclusion

To address the critical gap in large-scale answer verification evaluation, we present VerifierBench, featuring a meticulously designed pipeline for large-scale data collection, filtering, and annotation. We also introduce CompassVerifier, a novel verification model specifically engineered to handle multi-domain scenarios, diverse answer types, varied prompt formats, and irregular responses. CompassVerifier achieves superior accuracy and robustness compared to larger general LLMs and baseline verifier models. We anticipate that VerifierBench and CompassVerifier would significantly advance research in answer verification for evaluation frameworks and reward modeling for RLVR.

#### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

AI-MO. Aime 2024, 2024. URL https://huggingface.co/datasets/AI-MO/ aimo-validation-aime.

Anthropic. Claude 3.5 sonnet, 2024. URL https://www.anthropic.com/news/ claude-3-5-sonnet.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. Internlm2 technical report. arXiv preprint arXiv:2403.17297, 2024.

Maosong Cao, Alexander Lam, Haodong Duan, Hongwei Liu, Songyang Zhang, and Kai Chen. Compassjudger-1: All-in-one judge model helps model evaluation and evolution. arXiv preprint arXiv:2410.16256, 2024.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology, 15(3):1–45, 2024.

Ding Chen, Qingchen Yu, Pengyuan Wang, Wentao Zhang, Bo Tang, Feiyu Xiong, Xinchi Li, Minchuan Yang, and Zhiyu Li. xverify: Efficient answer verifier for reasoning model evaluations. arXiv preprint arXiv:2504.10481, 2025.

Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. TheoremQA: A theorem-driven question answering dataset. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 7889–7901, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.489. URL https://aclanthology.org/2023.emnlp-main.489/.

Francois Chollet, Mike Knoop, Gregory Kamradt, and Bryan Landers. Arc prize 2024: Technical report. arXiv preprint arXiv:2412.04604, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. https://github.com/ InternLM/xtuner, 2023.

Quy-Anh Dang and Chris Ngo. Reinforcement learning for reasoning in small llms: What works and what doesn’t, 2025. URL https://arxiv.org/abs/2503.16219.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Proc. of NAACL, 2019.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/records/12608602.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024.acl-long.211/.

Yancheng He, Shilong Li, Jiaheng Liu, Yingshui Tan, Weixun Wang, Hui Huang, Xingyuan Bu, Hangyu Guo, Chengwei Hu, Boren Zheng, et al. Chinese simpleqa: A chinese factuality evaluation for large language models. arXiv preprint arXiv:2411.07140, 2024b.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. V-star: Training verifiers for self-taught reasoners. arXiv preprint arXiv:2402.06457, 2024.

huggingface. Math-verify: A robust mathematical expression evaluation system designed for assessing large language model outputs in mathematical tasks. https://github.com/ huggingface/Math-Verify, 2024.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Akira Kawabata and Saku Sugawara. Rationale-aware answer verification by pairwise self-evaluation. arXiv preprint arXiv:2410.04838, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In SIGOPS, 2023.

Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al. From generation to judgment: Opportunities and challenges of llm-as-a-judge. arXiv preprint arXiv:2411.16594, 2024a.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath tir. Hugging Face repository, 2024b. URL https://huggingface.co/AI-MO/NuminaMath-TIR. Dataset documentation available at https://github.com/project-numina/aimo-progress-prize/ blob/main/report/numina_dataset.pdf.

Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, hai zhao, and Pengfei Liu. Generative judge for evaluating alignment. In The Twelfth International Conference on Learning Representations, 2024c. URL https://openreview.net/forum?id=gtkFw6sZGS.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval, 5 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024a.

Junnan Liu, Hongwei Liu, Linchen Xiao, Ziyi Wang, Kuikun Liu, Songyang Gao, Wenwei Zhang, Songyang Zhang, and Kai Chen. Are your llms capable of stable reasoning? CoRR, abs/2412.13147, 2024b.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025.

Jianqiao Lu, Zhiyang Dou, Hongru Wang, Zeyu Cao, Jianbo Dai, Yunlong Feng, and Zhijiang Guo. Autopsv: Automated process-supervised verifier. Advances in Neural Information Processing Systems, 37:79935–79962, 2024.

Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. arXiv preprint arXiv:2401.08967, 3, 2024.

Kaijing Ma, Xinrun Du, Yunran Wang, Haoran Zhang, Zhoufutu Wen, Xingwei Qu, Jian Yang, Jiaheng Liu, Minghao Liu, Xiang Yue, et al. Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks. arXiv preprint arXiv:2410.06526, 2024.

Xueguang Ma, Qian Liu, Dongfu Jiang, Zejun Ma, and Wenhu Chen. General-reasoner: Advancing llm reasoning across all domains. https://github.com/TIGER-AI-Lab/ General-Reasoner, 2025.

Skywork o1 Team. Skywork-o1 open series. https://huggingface.co/Skywork, November

2024. URL https://huggingface.co/Skywork. OC-Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023. OpenAI. Openai evals: Evals is a framework for evaluating llms and llm systems, and an

open-source registry of benchmarks. https://github.com/openai/evals, 2023. OpenAI. Gpt-4o, 2024a. URL https://openai.com/index/hello-gpt-4o/. OpenAI. Gpt-4o mini, 2024b. URL https://openai.com/index/

gpt-4o-mini-advancing-cost-efficient-intelligence. OpenAI. O1-preview, 2024c. URL https://openai.com/index/ introducing-openai-o1-preview.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. In First Conference on Language Modeling, 2024.

ByteDance Seed. Doubao-1.5-pro, 2025. URL https://seed.bytedance.com/en/special/ doubao_1_5_pro.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. In EuroSys, pp. 1279–1297. ACM, 2025.

Wenlei Shi and Xing Jin. Heimdall: test-time scaling on the generative verification. arXiv preprint arXiv:2504.10337, 2025.

Yuda Song, Hanlin Zhang, Carson Eisenach, Sham M. Kakade, Dean Foster, and Udaya Ghai. Mind the gap: Examining the self-improvement capabilities of large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=mtJSMcF3ek.

Yi Su, Dian Yu, Linfeng Song, Juntao Li, Haitao Mi, Zhaopeng Tu, Min Zhang, and Dong Yu. Expanding rl with verifiable rewards across diverse domains. arXiv preprint arXiv:2503.23829, 2025.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging bigbench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv:2312.11805, 2023.

Rui Wang, Hongru Wang, Boyang Xue, Jianhui Pang, Shudong Liu, Yi Chen, Jiahao Qiu, Derek Fai Wong, Heng Ji, and Kam-Fai Wong. Harnessing the reasoning economy: A survey of efficient reasoning for large language models. arXiv preprint arXiv:2503.24377, 2025a.

Yidong Wang, Zhuohao Yu, Wenjin Yao, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. PandaLM: An automatic evaluation benchmark for LLM instruction tuning optimization. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=5Nn2BLV7SB.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025b.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024b.

Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024.

Yuchen Yan, Jin Jiang, Zhenbang Ren, Yijun Li, Xudong Cai, Yang Liu, Xin Xu, Mengdi Zhang, Jian Shao, Yongliang Shen, et al. Verifybench: Benchmarking reference-based reward systems for large language models. arXiv preprint arXiv:2505.15801, 2025.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Xiaotian Zhang, Chunyang Li, Yi Zong, Zhengyu Ying, Liang He, and Xipeng Qiu. Evaluating the performance of large language models on gaokao benchmark. arXiv preprint arXiv:2305.12474, 2023.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. arXiv preprint arXiv:2501.07301, 2025.

Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Processbench: Identifying process errors in mathematical reasoning. arXiv preprint arXiv:2412.06559, 2024.

Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao, Hua Lu, Yicheng Chen, Yang Zhang, Wei Zhou, Jinjie Gu, and Lei Zou. A comprehensive survey of reward models: Taxonomy, applications, challenges, and future. arXiv preprint arXiv:2504.12328, 2025.

Zihao Zhou, Shudong Liu, Maizhen Ning, Wei Liu, Jindong Wang, Derek F Wong, Xiaowei Huang, Qiufeng Wang, and Kaizhu Huang. Is your model really a good math reasoner? evaluating mathematical reasoning with checklist. arXiv preprint arXiv:2407.08733, 2024.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. JudgeLM: Fine-tuned large language models are scalable judges. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=xsELpEPn4A.

#### 7 Appendix

##### 7.1 Details of VerifierBench Statistics

- Table 3: Dataset source distribution.

Source Count Percentage (%) BBH 639 22.68 GaokaoBench 201 7.14 Math 182 6.46 MMLU Pro 172 6.11 GPQA Diamond 51 1.81 GSM8K 14 0.50 AIME2024 3 0.11 SimpleQA 97 3.44 Numina Train 106 3.76 HLE 355 12.60 KorBench 395 14.02 OlympiadBench 345 12.25 ARC Prize Public Evaluation 175 6.21 TheoremQA 82 2.91

- Table 4: Category distribution.

Category Count Percentage (%)

- A 1092 38.76
- B 1526 54.17
- C 199 7.06

- Table 5: Domain distribution.

Domain Count Percentage (%)

General Reasoning 1151 40.86 Mathematical Reasoning 900 31.95 Knowledge 387 13.74 Scientific Reasoning 379 13.45

- Table 6: Answer type distribution.

Answer Type Count Percentage (%)

Multiple Choice 891 31.63 Short Text 354 12.57 Numerical 434 15.41 Formula 343 12.18 Multi-subproblem 281 9.98 Sequence 468 16.61 Boolean Answer 46 1.63

##### 7.2 Details of VerifierBench Construction

Data Collection. Our experimental evaluation encompasses a comprehensive collection of 53 LLMs, including representative examples such as Qwen-2.5 (Yang et al., 2024),

LLaMA3 (Grattafiori et al., 2024), DeepSeek-V3 (Liu et al., 2024a), DeepSeek-R1 (Guo et al.,

- 2025), GPT-4o (OpenAI, 2024a), GPT-4o-mini (OpenAI, 2024b), Gemini (Team et al., 2023), claude3-5 (Anthropic, 2024), Doubao-1.5-Pro (Seed, 2025), InternLM (Cai et al., 2024) and Mixtral (Jiang et al., 2024). All specific models are listed in Table 12. These models are evaluated across sixteen diverse benchmarks: GSM8K (Hosseini et al., 2024), Math (Hendrycks et al., 2021), AIME2024 (AI-MO, 2024), BBH (Suzgun et al., 2022), GaokaoBench (Zhang et al., 2023), HLE (Phan et al., 2025), KorBench (Ma et al., 2024), GPQA (Rein et al., 2024), SimpleQA (Wei et al., 2024), ChineseSimpleQA (He et al., 2024b), MMLU-Pro (Wang et al., 2024b), ARC (Chollet et al., 2024), OlympiadBench (He et al., 2024a), TheoremQA (Chen et al., 2023), NuminaMath (Li et al., 2024b), and Drop (Dua et al., 2019). Through the OpenCompass (OC-Contributors, 2023) framework, we collected more than 1.32 million response models, creating the most comprehensive response datasets to date.

VerifierBench Construction Details. For samples with inconsistent verification results across multiple models and prompts, we identified numerous cases that were either redundant or unworthy of human annotation. We employed a string-matching script to detect and remove duplicate responses, which predominantly belonged to category C (invalid responses). Additionally, we utilized DeepSeek-V3 to identify problematic cases, including: (1) questions with obvious open-ended nature, (2) incomplete reference answers, and (3) proof-based problems - all of which cannot be objectively evaluated solely based on reference answers and may introduce ambiguity in test set evaluation. After deduplication, approximately 5,000 samples underwent human annotation, where annotators further flagged the aforementioned problematic types. Annotation results revealed that most of the inconsistent samples were ultimately labeled as category B (incorrect responses), suggesting a potential tendency of LLM judges toward false positives. To maintain better label balance, we further applied similarity-based filtering to remove redundant samples within the category B subset. This rigorous filtering process yielded a final high-quality dataset of 2,817 samples.

##### 7.3 Details of CompassVerifier Experiments

Evaluation Setup. We use OpenCompass (OC-Contributors, 2023) and employ both F1 score and Accuracy as evaluation metrics, with particular emphasis on the F1 score, as it provides a more comprehensive assessment considering the precision, recall, and balance of the class distribution simultaneously. For all open-source models, we use vllm Kwon et al. (2023) for the acceleration of inference. For all models, we employ temperature=1.0 for data synthesis and temperature=0.0 for evaluation/verification, with both max_gen_len and max_model_len set to their maximum values. We use the official prompt for Xverify and Tencent-Qwen2.5-7B-Instruct-RLVR, and a general non-cot prompt for CompassVerifier and general LLMs, which can be found in the first prompt in Section 7.7.

Training Setup. We use XTuner (Contributors, 2023) for training our CompassVerifier model on Qwen2.5 (Yang et al., 2024) series models, largely adhering to the original hyperparameters. Fine-tuning is conducted using a learning rate of 2 × 10−5 with a max sequence length 32768. A multiplicative learning rate decay is applied after each epoch, with a gamma value of 0.85. The batch sizes are set to 32. All models are trained for one epoch on the training set and fully fine-tuned on 8×A100 80GB GPUs.

- Table 7: Detailed results on VerifierBench across different question types. We report Accuracy (Acc.) and F1 scores (%) for various problem categories and their average. Bold numbers indicate the best performance in each column.

Model

Boolean Multi-sub Numerical Short Text Formula Multi-choice Sequence Average Acc. F1 Acc. F1 Acc. F1 Acc. F1 Acc. F1 Acc. F1 Acc. F1 Acc. F1

|Qwen2.5-7B-Instruct 63.0 41.4 45.9 40.2 49.5 11.3 65.0 38.0 53.5 18.4 62.0 65.0 59.2 23.9 Qwen2.5-14B-Instruct 63.0 66.7 54.5 45.0 57.4 39.3 59.9 42.3 53.8 26.9 49.0 45.9 68.8 34.8 Qwen2.5-32B-Instruct 58.7 53.7 65.8 37.7 56.7 33.9 61.3 27.7 59.3 19.5 55.8 52.5 80.6 19.5 Qwen2.5-72B-Instruct 73.9 71.4 65.8 46.7 62.0 36.8 57.9 47.7 57.0 27.5 61.9 62.4 74.8 40.4|56.9 34.0 58.0 43.0 62.6 34.9 64.8 47.6<br><br>|
|---|---|
|Qwen3-8B 73.9 77.8 50.2 48.5 52.5 44.3 52.3 47.4 54.7 47.7 70.4 76.8 53.0 30.4 Qwen3-14B 69.6 66.7 69.8 52.0 64.8 39.0 76.6 56.1 66.6 27.7 72.4 73.8 84.6 39.0 Qwen3-30B-A3B 71.7 69.8 45.9 44.9 66.1 66.4 53.7 47.4 48.8 51.4 74.9 79.8 55.1 28.1 Qwen3-32B 80.4 80.9 63.4 55.9 64.8 51.4 68.6 57.1 64.2 44.3 74.3 77.8 78.4 46.0 Qwen3-235B-A22B 67.4 57.1 60.9 52.6 63.8 48.9 67.8 56.1 62.5 43.5 79.0 82.6 83.3 50.4|58.1 53.3 72.0 50.6<br>59.5 55.4 70.6 59.1 69.2 55.9<br><br><br>|
|GPT-4.1-2025-04-14 80.4 80.0 68.3 44.7 64.1 31.6 83.1 64.7 68.6 22.9 89.4 91.0 88.3 43.3 GPT-4o-2024-08-06 65.2 63.6 63.7 37.0 63.6 29.5 79.7 54.4 67.2 11.0 80.0 81.9 86.8 35.4 DeepSeek-V3-0324 63.0 56.4 61.2 52.0 68.2 48.9 81.6 66.3 69.5 39.3 85.4 87.6 85.5 54.1|77.4 54.0<br><br>72.3 44.7<br>73.5 57.8<br>|

xVerify-0.5B-I 67.4 59.5 66.9 25.6 63.6 37.8 64.7 36.6 60.8 22.0 95.7 96.6 85.5 35.0 72.1 44.7 xVerify-8B-I 71.7 71.1 73.0 51.3 65.2 36.3 65.3 28.1 66.6 24.8 92.6 94.0 88.3 35.3 74.7 48.7 xVerify-9B-C 67.4 70.6 76.9 50.4 65.2 40.8 58.8 34.8 63.4 30.0 92.3 93.6 85.9 29.8 72.8 50.0 Tencent-Qwen2.5-7B-Instruct-RLVR 71.7 71.1 69.0 51.4 74.9 59.2 71.2 28.2 69.8 40.2 84.2 86.5 85.0 27.1 75.1 52.0

CompassVerifier-3B 87.0 86.4 80.8 69.3 75.8 65.1 78.8 59.9 68.8 57.4 95.7 96.6 87.6 52.5 82.1 69.3 CompassVerifier-7B 91.3 91.7 85.1 75.0 77.0 67.5 87.6 79.1 71.1 61.2 95.6 96.6 90.2 67.1 85.4 76.0 CompassVerifier-32B 95.7 95.8 93.6 89.2 80.9 74.7 88.4 79.8 79.9 71.4 96.2 97.0 93.2 74.6 89.2 83.0

General LLMs

Verifier Models

CompassVerifiers

- Table 8: Three-label classification performance on VerifierBench. Beyond binary correctness (correct/incorrect), this evaluation requires models to identify invalid responses. We report Accuracy and macro-F1 scores (in %) across four distinct categories and their overall average.

Model

Math General Reasoning Knowledge Science Average Acc. macro-F1 Acc. macro-F1 Acc. macro-F1 Acc. macro-F1 Acc. macro-F1

|Qwen2.5-7B-Instruct 39.6 29.2 49.2 37.8 45.2 34.6 50.3 34.2 Qwen2.5-14B-Instruct 44.2 37.7 50.9 40.1 42.9 37.6 57.1 44.1 Qwen2.5-32B-Instruct 46.0 35.7 59.8 47.8 55.6 45.7 70.8 52.5 Qwen2.5-72B-Instruct 51.1 43.0 57.3 48.6 67.4 52.2 72.9 58.8|46.1 34.0 48.8 39.9 58.0 45.4 62.2 50.7<br><br>|
|---|---|
|Qwen3-8B 48.2 35.8 54.0 42.3 56.1 41.1 47.9 36.5 Qwen3-14B 61.3 57.3 72.3 63.5 65.4 54.7 74.7 61.9 Qwen3-30B 53.3 45.6 49.6 42.1 54.8 50.2 45.0 39.0 Qwen3-32B 57.2 54.2 61.6 54.4 60.2 51.7 58.7 50.0 Qwen3-235B-A22B 58.8 42.8 73.8 55.0 65.4 48.6 67.6 52.4|51.5 38.9 68.4 59.4 50.7 44.2 59.4 52.6 66.4 49.7<br><br>|
|GPT-4.1-2025-04-14 61.7 59.6 78.1 73.6 78.3 69.7 79.5 68.4 GPT-4o-2024-08-06 57.9 53.9 68.3 62.9 73.4 66.0 71.1 57.1 DeepSeek-V3-0324 63.2 49.1 77.4 66.2 76.5 60.3 80.5 67.8|74.4 67.8 67.7 60.0 74.4 60.9|

CompassVerifier-3B 73.4 68.8 87.4 85.6 86.3 87.1 87.6 80.8 83.7 80.6 CompassVerifier-7B 77.7 74.3 88.1 87.6 91.5 92.6 86.0 79.1 85.8 83.4 CompassVerifier-32B 82.0 79.6 90.0 90.7 94.3 95.9 91.3 86.8 89.4 88.3

General LLMs

CompassVerifiers

- Table 9: Ablation study on CompassVerifier-7B with different augmentation strategies on VerifierBench main results. Complex Formula Augmentation enhances formula variants verification, Error-Driven Adversarial Augmentation fortifies against failure cases.

Setting Accuracy (%) ∆ Acc (%) F1 (%) ∆ F1 (%) CompassVerifier-7B-Base 84.0 - 79.8 -

+ Complex Formula Augmentation 86.7 +2.7 82.8 +3.0 + Error-Driven Adversarial Augmentation 86.4 +2.4 82.0 +2.2 + Both Augmentations 87.5 +3.5 83.4 +3.6

- Table 10: Experimental results of CompassVerifier as a reward model. We report the avg@32 performance on AIME24, AIME25, and MATH500.

Model AIME24 AIME25 MATH500 Original Model Performance

Qwen3-4B-Base 2.7 1.8 34.1 RL with Rule-based Verifier Math-Verify 8.9 7.2 63.1 RL with Model-based Verifier

Tencent-RLVR 17.4 16.2 80.5 Qwen3-14B 19.8 16.6 81.2 Qwen2.5-32B 19.6 15.4 81.6 CompassVerifier-7B 21.2 17.3 82.2 CompassVerifier-32B 21.2 17.2 83.3

- 7.4 Details of Meta Error Patterns

We display the meta error patterns in three categories: A (Correct), B (Incorrect), and C (Invalid) as shown in the following figures.

Meta Pattern: A (Correct)

- • The units in the LLM Response differ from those in the final answer, resulting in different numerical expressions, but they are consistent upon conversion, should be judged as Correct.
- • The reference answer is an extremely complex formula, and the LLM Response appears very different in form but simplifies to an equivalent expression, with no explicit requirement for simplification in the question, should be judged as Correct.
- • The question requires calculating a numerical decrease, and the LLM Response has the opposite sign of the reference answer because either uses negative signs to represent decrease, but they are equivalent, should be judged as Correct.
- • The reference answer provides multiple candidate answers without requiring all possibilities. The LLM Response provides one of them, should be judged as Correct.
- • The question doesn’t explicitly specify answer format (numerical or formula). The LLM Response and reference answer differ in form but are equivalent when calculated, should be judged as Correct.
- • The question requires specific formatting (order, capitalization, etc.). While the LLM Response appears different from the reference answer in formatting, upon inspection it fully complies, should be judged as Correct.
- • When calculating values with units, the reference answer and LLM Response may differ in unit representation or numerical values, but are equivalent after unit conversion, should be judged as Correct.
- • For multiple-choice or true/false questions, the LLM Response ultimately gives the correct answer despite showing significant uncertainty, should be judged as Correct.
- • The question requires expressions meeting simple conditions (sum, product, logical relations, etc.), and the reference answer may include multiple valid forms. The LLM Response differs in form but meets all requirements, should be judged as Correct.
- • The LLM initially provides an incorrect answer but corrects it after reflection, should be judged as Correct.

- • The reference answer consists of multiple sub-questions. The LLM answers all sub-questions correctly during reasoning, even if not presented together at the end, should be judged as Correct.

Meta Pattern: B (Inorrect)

- • For multiple-choice questions, the LLM Response selects the correct option but follows with unrelated option content, should be judged as Incorrect.
- • The question requires formula simplification. The LLM answer isn’t fully simplified to minimal form, even if equivalent to the reference answer, should be judged as Incorrect.
- • The reference answer is a formula with specified output format. The LLM answer doesn’t comply with this format, even if equivalent, should be judged as Incorrect.
- • The question requires an expression where the sum equals a certain value with each number used once. The LLM Response repeats numbers while satisfying the sum, should be judged as Incorrect.
- • The reference answer is an un-simplified logical formula after substitution. The LLM Response is incorrect due to simplification causing format errors, should be judged as Incorrect.
- • The LLM Response only provides solution code without final results, should be judged as Incorrect.
- • The LLM Response (formula/numerical) and reference answer aren’t equivalent when calculated, should be judged as Incorrect.
- • When describing numerical intervals, the reference answer and LLM Response differ in endpoint inclusion (open/closed), should be judged as Incorrect.
- • For sequence decryption requiring exact matching, the LLM Response doesn’t match the reference answer, should be judged as Incorrect.
- • The reference answer is a long sequence requiring exact correspondence. The LLM Response has minor differences with some errors, should be judged as Incorrect.
- • The question explicitly requires multiple candidate answers (provided in reference), but the LLM Response gives only one, should be judged as Incorrect.
- • The LLM initially provides a correct answer but changes to incorrect or "unanswerable" after reflection, should be judged as Incorrect.
- • For symbolic sequences, the LLM Response contains garbled characters, should be judged as Incorrect.
- • The reference answer is numerical, and the LLM Response provides more decimal places but rounds differently, should be judged as Incorrect.
- • The reference answer is an extremely large number, and the LLM Response provides a high-order power expression that doesn’t match after calculation, should be judged as Incorrect.
- • After detailed reasoning, the LLM Response fails to provide a clear answer or states the question is unanswerable, should be judged as Incorrect.
- • For multi-part questions, the number of final answers in the LLM Response doesn’t match the reference answer, should be judged as Incorrect.

Meta Pattern: C (Invalid)

- • The question contains multiple sub-questions, but the number of reference answers doesn’t match, indicating quality issues, should be judged as Invalid.
- • The reference answer has serious omissions, truncation, or formatting issues, should be judged as Invalid.
- • The question itself has serious omissions, truncation, or formatting issues, should be judged as Invalid.
- • The LLM doesn’t answer normally, stating it needs more information or internet access, should be judged as Invalid.
- • The LLM Response is clearly truncated and incomplete, should be judged as Invalid.

- • The LLM Response is mostly garbled text with no valuable information extractable, should be judged as Invalid.
- • The LLM Response contains extensive meaningless repetition, making correct answers unidentifiable, should be judged as Invalid.

##### 7.5 Meta-Judge Template Generation Fields

Table 11: Meta-Judge Template Generation Fields (Academic Disciplines and Subfields)

Category Discipline Subfields

Mathematics Differential calculus, Integral calculus, Probability statistics, Operations research, Mathematical logic, Financial mathematics, Topology, Algebraic geometry

Natural Sciences

Physics Theoretical physics, Quantum mechanics, Condensed matter physics, Astrophysics, Nuclear physics, Optics, Acoustics

Chemistry Analytical chemistry, Organic chemistry, Inorganic chemistry, Physical chemistry, Materials chemistry, Environmental chemistry, Chemical biology

Biology Molecular biology, Genetics, Ecology, Cell biology, Biochemistry, Microbiology Earth Sciences Geology, Geophysics, Atmospheric sciences, Oceanography, Environmental science, Paleontology Statistics Data science, Biostatistics, Economic statistics, Machine learning algorithms, Bayesian analysis

Mechanical Engineering Mechanical design & manufacturing, Automatic control, Robotics, Vehicle engineering, Thermal & power engineering, MEMS Computer Science & Technology Artificial intelligence, Computer networks, Software engineering, Computer vision, Cybersecurity, Big data analytics Electronic Information Engineering Communication engineering, IC design, Optoelectronic technology, Wireless sensor networks, Smart grid Civil Engineering Structural engineering, Bridge & tunnel design, Geotechnical engineering, Hydraulic engineering, Urban planning Materials Science & Engineering Nanomaterials, Metallic materials, Polymer materials, Composite materials, Material processing Chemical Engineering Chemical process design, Petroleum refining, Biochemical engineering, Catalytic reaction engineering, Separation technology Environmental Engineering Pollution control technology, Environmental monitoring, Ecological restoration, Solid waste treatment, Clean energy development Aerospace Engineering Aircraft design, Propulsion systems, Aerodynamics, Satellite navigation, Aerospace materials Biomedical Engineering Medical imaging technology, Biomaterials, Artificial organs, Biosensors, Rehabilitation engineering Energy & Power Engineering Nuclear technology, Wind energy development, Solar energy utilization, Fuel cells, Thermal system optimization

Engineering

##### 7.6 Cases in VerifierBench

Example I: Verification as Correct (Gold Judgment: A) Question Let f(x) = √x and g(x) = x2. Find f(g(f(g(f(8))))). Gold Answer 2√2

LLM Response To solve the problem f(g(f(g(f(8))))), we need to evaluate the functions step by step, starting from the innermost function and working our way out.

- 1. Evaluate f(8): f(x) = √x

f(8) =

√8

- 2. Evaluate g(f(8)): g(x) = x2

g(

√8) = (

√8)2 = 8

- 3. Evaluate f(g(f(8))):

- f(8) =

√8

- g(

√8) = 8 f(8) =

√8

- 4. Evaluate g(f(g(f(8)))):

- f(8) =

√8

- g(

√8) = 8

- f(8) =

√8

- g(

√8) = 8

- 5. Evaluate f(g(f(g(f(8))))):

- f(8) =

√8

- g(

√8) = 8

- f(8) =

√8

- g(

√8) = 8 f(8) =

√8

After evaluating the functions step by step, we find that:

√8 Thus, the final answer is:

f(g(f(g(f(8))))) =

|√8<br><br>|
|---|

- Gold Judgment A (Correct) Meta Data

- • Data Source: math
- • Domain: Math
- • Answer Type: Numerical

Example II: Verification as Incorrect (Gold Judgment: B)

Question Sort the following words alphabetically: List: marshmallow doge alpenstock ocean accept angus drool jutish resistive chromium vociferous castigate prize octennial stonewort concision elizabethan

Gold Answer accept alpenstock angus castigate chromium concision doge drool elizabethan jutish marshmallow ocean octennial prize resistive stonewort vociferous LLM Response So the answer is accept, alpenstock, angus, castigate, chromium, concision, doge, drool, elizabethan, jutish, marshmallow, octennial, ocean, prize, resistive, stonewort, vociferous.

##### Gold Judgment B (Incorrect) Meta Data

- • Data Source: BBH/bbh-word_sorting
- • Domain: General Reasoning
- • Answer Type: Sequence

Example III: Verification as Invalid (Gold Judgment: C)

Question A droplet of molten lead of average 1/8 inch diameter, at 750°F, falls from a height of 60 ft, and solidifies by the cooling effect of the surrounding air. The surrounding temperature is 70°F. If the lead solidifies at 621°F, calculate the coefficient of heat transfer.

Given properties for lead:

- • Cp = 0.031 Btu/lbm-°F
- • ρ = 710 lbm/ft3
- • hfusion = 10.6 Btu/lbm

Options:

- • A. 59.2 Btu/hr-ft2-°F
- • B. 45.7 Btu/hr-ft2-°F
- • C. 52.8 Btu/hr-ft2-°F
- • D. 63.4 Btu/hr-ft2-°F

Gold Answer A LLM Response To solve this problem, we need to calculate the coefficient of heat transfer (h) using the formula: Q = h * A * ∆T where: - Q is the heat transferred (in Btu) - A is the surface area of the droplet (in ft2)

- ∆T is the temperature difference (in °F) First, let’s find the volume (V) of the droplet using the given average diameter (1/8 inch): V = π * (d/2)3 V = π * (1/16)3 V = π * (1/4096) ft3 Now, let’s find th

##### Gold Judgment C (Invalid) Meta Data

- • Data Source: mmlu_pro/mmlu_pro_engineering
- • Domain: Science
- • Answer Type: Multiple Choice

##### 7.7 Prompt List

Please as a grading expert , judge whether the final answers given by the candidates below are consistent with the standard answers , that is, whether the candidates answered correctly. Here are some evaluation criteria:

- 1. Please refer to the given standard answer. You don't need to regenerate the answer to the question because the standard answer has been given. You only need to judge whether the candidate 's answer is consistent with the standard answer according to the form of the question

. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.

- 2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.

- 3. Some answers may be expressed in different ways , such as some answers may be a mathematical expression , some answers may be a textual description , as long as the meaning expressed is the same. Before making a judgment , please understand the question and the standard answer first ,

and then judge whether the candidate 's answer is correct.

- 4. Some answers may consist of multiple items , such as multiple -choice questions , multiple -select questions , fill -in-the -blank questions , etc. Regardless of the question type , the final answer will be considered correct as long as it matches the standard answer , regardless of whether the reasoning process is correct. For multiple -select questions and multi

-blank fill -in-the-blank questions , all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.

- 5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed {{}} and only judge whether the candidate 's answer is consistent with the

standard answer.

- 6. If the candidate 's answer is invalid (e.g., incomplete (cut off mid response), lots of unnormal repetitive content , or irrelevant to the question , saying it can 't answer the question because some irresistible factors , like ethical issues , no enough information , etc.), select option

C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:

- A: CORRECT

- B: INCORRECT

- C: INVALID Just return the letters "A", "B", or "C", with no text around it. Here is your task. Simply reply with either CORRECT , INCORRECT , or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer. <Original Question Begin >: {question} <Original Question End> <Standard Answer Begin >: {gold_answer} <Standard Answer End> <Candidate 's Answer Begin >: {llm_response} <Candidate 's Answer End> Judging the correctness of the candidate 's answer:

Prompt 1: Prompt for general LLM evaluation

As a grading expert , your task is to determine whether the candidate 's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:

- 1. Reference Standard:

- - The standard answer is definitive and always correct

- - The question is perfectly valid - never question them

- - Do not regenerate answers; only compare with the given standard

- 2. Comparison Method:

- - Carefully analyze the question 's requirements and the standard answer 's structure

- * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.

- * This determination must be made based on the question 's phrasing and the nature of the standard answer.

- - Compare ONLY the candidate 's final answer (ignore all reasoning/ explanation errors)

- - Disregard any differences in formatting or presentation style

- - For mathematical expressions: calculate step by step whether the two formulas are equivalent

- - For multiple -choice questions: compare only the final choice and corresponding option content

- 3. Multi -part Answers:

- - For questions requiring multiple responses (e.g., multi -select):

- - All parts must match the standard answer exactly.

- - Compare each sub-answer step by step. Partial matches are considered incorrect.

- 4. Validity Check:

- - Reject answers that are:

- * Incomplete (cut off mid -sentence in the final sentence , lacking a complete response) - Label as INCOMPLETE

- * Repetitive (repetition of words or phrases in a loop) - Label as REPETITIVE

- * Explicit refusals (e.g., directly return "I cannot answer/provide/ access ...") - Label as REFUSAL

- - For invalid answers , specify the type in the judgment (e.g., \boxed{ C} - INCOMPLETE).

Grading Scale:

- \boxed{A} - CORRECT:

- - Answer matches standard exactly (including equivalent expressions)

- - For numerical answers: consider as equivalent if values match when rounded appropriately

- - Semantically equivalent responses

- \boxed{B} - INCORRECT:

- - Any deviation from standard answer

- - Partial matches for multi -part questions

- \boxed{C} - INCOMPLETE/REPETITIVE/REFUSAL:

- Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/ REFUSAL)

Execution Steps and Output Formats: Analysis step by step: [ Thoroughly evaluate the candidate 's answer including:

- (1) First check if the answer is INCOMPLETE (cut off mid -sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \boxed{C} with the corresponding type.

- (2) Analyze the question 's core requirements and the standard answer 's structure , for example:

- - Strict requirements: Identify mandatory constraints (e.g., simplification , answer order , multi -part completeness)

- - Tolerant allowances: Ignore non -critical deviations (e.g., missing option labels in MCQs , equivalent but unformatted expressions)

- - Required answer type , precision level , etc.

- (3) Perform a detailed comparison between the candidate 's final answer and the standard answer , for example:

- - Content equivalence

- - Permitted variations in numerical precision

- - Allowed expression formats] Final Judgment: \boxed{A/B/C} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/ REFUSAL >

Here is your task. <Original Question Begin > {question} <Original Question End> <Standard Answer Begin > {gold_answer} <Standard Answer End> <Candidate 's Answer Begin > {llm_response} <Candidate 's Answer End> Analysis step by step and Final Judgment:

Prompt 2: Prompt A for CoT answer verification

As a grading expert , your task is to determine whether the candidate 's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:

- 1. Reference Standard:

- - The standard answer is definitive and always correct

- - The question is perfectly valid. Never question them

- - Do not regenerate answers; only compare with the given standard answer

- 2. Thoroughly evaluate the candidate 's answer follow these steps

- - Carefully analyze the question 's content and requirements

- * Strict requirements: Identify mandatory constraints (e.g., simplification , answer order , multi -part completeness)

- * Tolerant requirements: Ignore non -critical deviations (e.g., missing option labels in MCQs , equivalent but unformatted expressions)

- - Carefully analyze the standard answer 's content and structure. Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components

- - Validity Check for the candidate 's answer. Reject answers that are:

- * Incomplete (cut off mid -sentence in the final sentence , lacking a complete response) - Label as INCOMPLETE

- * Repetitive (repetition of words or phrases in a loop) - Label as REPETITIVE

- * Explicit refusals (e.g., directly return "I cannot answer/provide/ access ...") - Label as REFUSAL

- - Perform a detailed comparison between the candidate 's final answer and the standard answer

- * Compare ONLY the candidate 's final answer (ignore all reasoning/ explanation errors)

- * Disregard any differences in formatting or presentation style

- * For mathematical expressions: calculate step by step whether the two formulas are equivalent

- * For multiple -choice questions: compare only the final choice and the corresponding option content

- * For questions requiring multiple sub -answers (e.g., multi -select): All parts must match the standard answer exactly. Compare each sub -

answer step by step. Partial matches are considered incorrect.

- 3. Grading Scale:

- \boxed{A} - CORRECT:

- - Answer matches standard exactly (including equivalent expressions )

- - For numerical answers: consider as equivalent if values match when rounded appropriately

- - Semantically equivalent responses

- \boxed{B} - INCORRECT:

- - Any deviation from standard answer

- - Partial matches for multi -part questions

- \boxed{C} - INCOMPLETE/REPETITIVE/REFUSAL:

- Fails validity criteria above (must specify: INCOMPLETE/ REPETITIVE/REFUSAL)

Output Formats: Analysis: [Analysis and evaluate step by step here.] Final Judgment: \boxed{A/B/C} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/ REFUSAL >

Here is your task. <Original Question Begin > {question} <Original Question End> <Standard Answer Begin > {gold_answer} <Standard Answer End> <Candidate 's Answer Begin > {llm_response} <Candidate 's Answer End> Analysis: Final Judgment:

Prompt 3: Prompt B for CoT answer verification

As a grading expert , your task is to determine whether the candidate 's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:

- 1. Reference Standard:

- - The standard answer is definitive and always correct

- - The question is perfectly valid - never question them

- - Do not regenerate answers; only compare with the given standard

- 2. Comparison Method:

- - Extract ONLY the candidate 's final answer (ignore all reasoning/ explanation errors)

- - If no complete final answer exists (e.g., response is cut off or contains only reasoning) - INVALID

- - Compare this directly with the standard answer

- - Disregard any differences in formatting or presentation style

- - For mathematical expressions: compare semantic equivalence , not syntax

- - For \boxed{} format: ignore the \boxed notation when comparing

- 3. Multi -part Answers:

- - For questions requiring multiple responses (e.g., multi -select):

- - All parts must match the standard answer exactly

- - Partial matches are considered incorrect

- 4. Validity Check:

- Reject answers that are:

- * Incomplete (cut off mid -response or missing final answer)

- * Purely reasoning without final answer

- * Repetitive or uninterpretable

- * Irrelevant to the question

- * Explicit refusals (e.g., "I cannot answer/provide/access ...")

Grading Scale:

- \boxed{A} - CORRECT:

- - Answer matches standard exactly (including equivalent expressions)

- - For numerical answers: allow 1% tolerance for floating -point variations

- - Semantically equivalent responses

- \boxed{B} - INCORRECT:

- - Any deviation from standard answer

- - Partial matches for multi -part questions

- \boxed{C} - INVALID:

- Fails validity criteria above

Execution Steps and Output Formats: Analysis:

- 1. Completeness and Validity Check: [confirm if candidate 's answer is complete and include the final answer]

- 2. Extracted Final Answer: [state what was identified as final answer]

- 3. Standard Comparison: [describe how it matches/mismatches] Final Judgment: [\boxed{A/B/C}]

Here is your task. <Original Question Begin > {question} <Original Question End> <Standard Answer Begin > {gold_answer} <Standard Answer End> <Candidate 's Answer Begin > {llm_response} <Candidate 's Answer End> Analysis and Final Judgment:

Prompt 4: Prompt C for CoT answer verification

Table 12: List of models used in the experiment with response counts

Model Family Model Name Response Count Yi Yi-Lightning 18496

Yi-1.5-9B-Chat 17722

GPT GPT-4o 18495 GPT-4o-mini 44502 GPT-4-1-2025-0414 2673 GPT-4.5-preview-2025-02-27 18381

Doubao Doubao-Pro-32k-241215 6378 Doubao-Pro-1.5-32k-250115 18517 Doubao-Pro-32k-240828 5692

Qwen Qwen-Max-0919 18434 Qwen-Max-2025-01-25 29173 Qwen2.5-Max 18320 Qwen2.5-7B-Instruct 49003 Qwen2.5-14B-Instruct 32116 Qwen2.5-32B-Instruct 37477 Qwen2.5-72B-Instruct 37568 QwQ-32B 20623

Gemini Gemini-2.0-Flash-Exp 17303 Gemini-1.5-Pro 18429 Gemini-2-5-Pro-03-25 669

DeepSeek-R1 DeepSeek-Chat-R1 16556 DeepSeek-R1-distill-Qwen-1.5B 16012 DeepSeek-R1-distill-Qwen-7B 16364 DeepSeek-R1-distill-Llama-8B 15731 DeepSeek-R1-distill-Qwen-14B 16671 DeepSeek-R1-distill-Qwen-32B 16042 DeepSeek-R1-distill-Llama-70B 15772

Llama Llama-3-1-8B-Instruct 44857

- Llama-3-1-70B-Instruct 18018
- Llama-3-2-3B-Instruct 28618
- Llama-3-3-70B-Instruct 28307

Mixtral Mistral-Small-Instruct-2409 18233 Mistral-Small-3.1-24B-Instruct 14331 Ministral-8B-Instruct-2410 17962 Mixtral-Large-Instruct-2411 18381

Claude Claude-3-5-Sonnet-20241022 18521 Claude-3-7-Sonnet-20250219 18474 Claude-3-7-Sonnet-20250219-Thinking 4723

Gemma Gemma-2-9B-It 34541 Gemma-2-27B-It 34704 Gemma3-27B-It 13120

DeepSeek-Chat DeepSeek-V2.5 31896 DeepSeek-Chat-V3 31950

InternLM InternLM2.5-7B-Chat 43336 InternLM2.5-20B-Chat 37594 InternLM3-8B-Instruct 15976

Phi Phi-4 18360 GLM GLM-4-9B-Chat 17537

GLM-4-Plus 18486 MiniMax MiniMax-Text-01 39570 Moonshot Moonshot-V1-32k 18067 Hunyuan Hunyuan-Standard-256K 18082 StepFun Step-2-16k 18405

##### 7.8 Details of Training Data for CompassVerifier

For the composition of CompassVerifier train dataset, we use 54420 consist samples from the VerifierBench pipeline as shown in Figure 1 as the base train set, we then use Error-Driven Adversarial Augmentation and Complex Formula Augmentation to construct extra data comprehensively enhance the capabilities of the CompassVerifier model. The composition of our train data list in Table 13.

Table 13: Composition of CompassVerifier training data

Data Source Number of Samples Percentage (%) Base Train Set (VerifierBench) 54,420 56.20 Error-Driven Adversarial Augmentation 24,294 25.09 Complex Formula Augmentation 18,118 18.71 Total 96,832 100.00

Error-Driven Adversarial Augmentation. Using DeepSeek-v3, we generate 34 Meta-Judge Templates covering common and extreme error scenarios then generate 224294 synthetic examples that emphasize decision boundary cases, especially where human judges tolerate minor errors that baseline verifiers over-penalize.

Complex Formula Augmentation. Applying this augmentation pipeline, we have synthesized approximately 18118 enhanced examples spanning 14 distinct scientific and engineering disciplines.

7.9 Details of CompassVerifier-as-Reward Experimental Settings Base LLMs. We utilize Qwen3-4B-Base (Yang et al., 2025) as the base LLM for the GRPO training.

Training Template. We utilize the following training template to prompt the base LLM to generate a response for each question. We only verify the format correctness to ensure the final answer is encapsulated within ‘\boxed{...final answer...}’.

Training Template of CompassVerifier

A conversation between a User and an Assistant. The User poses a question, and the Assistant provides a solution. The Assistant’s response follows these structured steps:

- 1. **Reasoning Process**: The Assistant comprehensively thinks about the problem through a reasoning process.
- 2. **Conclusion**: The Assistant reaches a conclusion, which is enclosed within ‘<conclusion>’ and ‘</conclusion>’ tags. The final answer is highlighted within ‘\boxed{...final answer...}’.
- 3. **Response Format**: The complete response should be formatted as:

...reasoning process... <conclusion>

...conclusion... The answer is \boxed{...final answer...} </conclusion>

Training Data. We utilize the challenging mathematical reasoning dataset Open-S1 (Dang & Ngo, 2025) as the RL training corpus. To increase the difficulty of our validation, we curate the final training set by specifically excluding problems with integer solutions from the original Open-S1 dataset.

Evaluation. We employ Math-Verify (huggingface, 2024) as our evaluation tool since the answers in these three benchmarks are readily verifiable, making them particularly well-suited for Math-Verify’s verification mechanism.

Reward Design. We design a simple reward scheme: 0 for answer errors, and 1 for correct responses.

Training Parameters. We utilize the following loss function, with Table 14 detailing the training parameters:

L = E(q,a)∼D,{o

i}iG=1∼πθold(·|q)

|oi|

G

πθ (oi,t|q, oi,<t) πθold (oi,t|q, oi,<t)

πθ (oi,t|q, oi,<t) πθold (oi,t|q, oi,<t)

1 ∑iG=1 |oi|

### ∑

### ∑

min

ai,t,clip

,1 − ϵmin,,1 − ϵmax ai,t ,

t=1

i=1

(2) where D denotes the training data, (q, a) represents the question-answer pair, G signifies the group size, and

ai,t = ri − mean({ri}iG=1). (3) In this context, ai,t signifies the advantage of response oi at the t-th position, and ri denotes the reward of response oi. Essentially, the KL penalty of the original GRPO loss is omitted, and zero mean normalization is employed to estimate the advantage.

Table 14: Training parameters of CompassVerifier as reward experiments.

##### Parameters Value

train batch size 256 train epochs 2 learning rate 1e-6 max prompt length 4096 max response length 12288 G 8

ϵmin 0.2 ϵmax 0.28

Hardware. All experiments are conducted on clusters equipped with 8 NVIDIA A800-SXM4-80GB GPUs and Intel(R) Xeon(R) Platinum 8336C CPUs, implementing with veRL (Sheng et al., 2025).

