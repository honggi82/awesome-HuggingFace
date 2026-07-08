# arXiv:2508.04017v1[cs.CV]6Aug2025

## Can Large Multimodal Models Actively Recognize Faulty Inputs? A Systematic Evaluation Framework of Their Input Scrutiny Ability

### Haiqi Yang1*, Jinzhe Li1,3*, Gengxu Li1, Yi Chang1,2,3, Yuan Wu1†

1School of Artificial Intelligence, Jilin University 2Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China 3International Center of Future Science, Jilin University {yanghaiqi24, lijz2121, lgx22}@mails.jlu.edu.cn, yichang@jlu.edu.cn, yuanwu@jlu.edu.cn

###### Abstract

Large Multimodal Models (LMMs) have witnessed remarkable growth, showcasing formidable capabilities in handling intricate multimodal tasks with exceptional performance. Recent research has underscored the inclination of large language models to passively accept defective inputs, often resulting in futile reasoning on invalid prompts. However, the same critical question of whether LMMs can actively detect and scrutinize erroneous inputs still remains unexplored. To address this gap, we introduce the Input Scrutiny Ability Evaluation Framework (ISEval), which encompasses seven categories of flawed premises and three evaluation metrics. Our extensive evaluation of ten advanced LMMs has identified key findings. Most models struggle to actively detect flawed textual premises without guidance, which reflects a strong reliance on explicit prompts for premise error identification. Error type affects performance: models excel at identifying logical fallacies but struggle with surface-level linguistic errors and certain conditional flaws. Modality trust variesGemini 2.5 pro and Claude Sonnet 4 balance visual and textual info, while aya-vision-8b over-rely on text in conflicts. These insights underscore the urgent need to enhance LMMs’ proactive verification of input validity and shed novel insights into mitigating the problem. The code is available at https://github.com/ MLGroupJLU/LMM_ISEval.

### Introduction

The rapid advancement of Large Multimodal Models (LMMs) has profoundly transformed the approach to complex, multimodal tasks. These models, demonstrating remarkable aptitude for integrating information across diverse modalities such as text, images, and audio (Li et al. 2024a), have consequently unlocked new possibilities in various applications, from enhanced human-computer interaction to more sophisticated automated agent systems (Hu et al. 2025; Tang et al. 2025). However, as LMM capabilities grow, their reliability and trustworthiness have become critical con-

*These authors contributed equally. †Corresponding authors.

cerns, demanding thorough investigation and robust solutions.

For LMMs to be truly reliable, they must actively scrutinize inputs and identify potential errors (He et al. 2025; Zhao et al. 2025), rather than simply accepting them and generating flawed reasoning (Wang et al. 2024b). This proactive stance is essential for preventing the propagation of errors and ensuring the integrity of the model’s outputs. This means the model not only remains unaffected by noisy or perturbed inputs but also actively identifies, diagnoses, and reports those errors to the user. This capability goes beyond simply being resilient in the face of flawed data; it enables the model to provide valuable feedback, helping users understand why a particular input might be problematic and guiding them toward more accurate or well-formed queries.

In the domain of Large Language Models (LLMs), existing research has already revealed their frequent failure to proactively question erroneous or logically flawed inputs, often leading to verbose and unnecessary over-reasoning on invalid questions (Li et al. 2025; Fan et al. 2025). This highlights the paramount importance of instilling active input validation capabilities in these models, as their passive acceptance of flawed information can undermine their utility and credibility.

Despite current studies evaluating LMMs’ modal preferences when encountering conflicting inputs (Yan et al. 2025; Zhang et al. 2025a; Hua, Yun, and Pavlick 2025) or their error detection abilities when explicitly instructed (Yan et al. 2024), there remains a notable absence of targeted and systematic research exploring the question: Can Large Multimodal Models Actively Recognize Faulty Inputs?

To address this critical gap, we introduce the Input Scrutiny Ability Evaluation Framework (ISEval). This innovative framework features seven meticulously designed categories of erroneous premises, comprehensively covering the diverse forms of errors prevalent in multimodal inputs, ranging from expression inaccuracies to logical inconsistencies. Furthermore, we establish three robust evaluation metrics to quantitatively and qualitatively assess LMMs’ input scrutiny abilities, providing a multifaceted perspective on their performance. Leveraging ISEval, we conduct a systematic evaluation across 10 of the latest LMMs and identify three key findings: (1) Most models have limited autonomous abil-

#### Example

[Figure 1]

Query with False Premise Q:builtThethisringswayshownwhich alsoare partiallycontainsinterlinked.the thick light Therering?\n<image1>are only four rings in total. How long is the longest chain

With Input Scrutiny Ability

Without Input Scrutiny Ability

......the longest chain is **4 rings long...... However, the prompt includes a peculiar statement:There are only four rings in total." This contradicts the image, which clearly shows 8 rings. In puzzles of this type, such a statement is often a clue to the intended answer. ......Therefore, the intended answer is likely 4. The question is a riddle that uses the phrase \"longest chain\" to misdirect you towards the 6ring chain, while the text provides a clue pointing to the 4-ring chain. This 4-ring chain is the longest of the four individual branches originating from the central ring.

Looking at how the rings are interlinked in the image, I can trace the possible chains that include the thick light ring. The longest possible chain would be 3 rings long. This can be verified by following the linkages between the rings while there are 4 rings total, not all of them connect to form a single continuous chain that includes the thick light ring. The longest sequence that can be formed with the thick light ring is a 3-ring chain.

###### Type of False Premise

Expression Error

Conditional Error

Reasoning Error

Unclear Citation:

Irrelevant or Incorrect Condition::

Misguided Logic：

Misunderstandings caused by unclear citations

Containing redundant content or conditions

Misleading with faulty logic

Lacking Condition:

Grammatical or Wording Error:

Lacking the conditions for obtaining the correct answer

Grammatical errors or improper wording

Exclusive Condition：

Misuse Confusion:

Existing mutually exclusive conditions

Misuse or confusion of terms

- Figure 1: Example comparison of model responses with and without input scrutiny ability. The query contains a false premise about the total number of rings (claiming ”only four rings” while the image shows 8 rings). The response ”Without Input Scrutiny Ability” accepts the flawed premise and generates reasoning based on it, while the response ”With Input Scrutiny Ability” proactively identifies the contradiction between the text and the image, demonstrating active input validation.

ity to detect flawed premises, with low Spontaneous Error Detection Rates (SEDR), yet they demonstrate significantly improved Guided Error Detection Rates (GEDR) when provided with explicit prompts, indicating that their latent critique capabilities rely heavily on external guidance to be activated. (2) Error types significantly affect detection performance: models achieve peak proficiency in identifying logical fallacies , but struggle with spontaneous recognition of Surface-Level Linguistic Errors and show consistently poor results in detecting Irrelevant or Incorrect Conditions and Exclusive Conditions. (3) Under cross-modal inconsistency, all models increase their reliance on visual input, with most closed-source models exhibiting a vision preference exceeding 50%, while most open-source models remain text-skewed. In contrast, with no cross-modal inconsistency, all LMMs consistently default to a preference for text. Our main contributions are as follows:

- • We introduce ISEval, a novel and comprehensive evaluation framework specifically engineered to assess the input scrutiny abilities of Large Multimodal Models (LMMs), which is built upon a meticulously curated dataset incorporating seven distinct categories of erroneous premises.
- • We conducted a systematic evaluation of 10 state-of-theart LMMs against the ISEval benchmark. This provides a

detailed and nuanced understanding of their capabilities in scrutinizing input validity.

• Our in-depth analysis of model performance yields three significant findings. These insights illuminate crucial limitations in LMMs’ proactive assessment of input validity and shed light on how their modal preferences influence their responses to faulty information.

### Related Works

##### Error Detection

Unimodal Extensive research has explored large language models’ (LLMs) error detection and critique capabilities across domains. In mathematical reasoning error detection, (Li et al. 2024b) identified four error types, built an annotated dataset, and showed prompting these types improves correction accuracy, with calculation errors being most challenging. (Liang et al. 2025) proposed MathClean, revealing limitations in strong models like GPT-o1 for detecting flaws in math questions and answers. (Shen et al. 2025) developed MathQ-Verify to filter ill-posed math questions, boosting verification performance.

For broader critique and correction abilities, two studies introduced CriticBench with distinct focuses. (Lin et al. 2024) examined Generation-Critique-Correction (GQC)

[Figure 2]

Question Question

Datasets

Question with False Premises

[Figure 3]

Answers Generated by

Question

[Figure 4]

[Figure 5]

Gemini Claude

[Figure 6]

[Figure 7]

Grok Qwen

[Figure 8]

Rewrite by GPT4o

[Figure 9]

- 1. o3 evaluation

[Figure 10]

- 2. Human Manually Check

Filtering by Human Annotators

Evaluation Results

Question with False Premises

A: Question Synthesis Pipeline B: Dataset Distribution

C: Evaluation Pipeline

- Figure 2: Schematic illustration of the dataset construction and evaluation pipeline. (A) The question synthesis pipeline: original questions are rewritten by GPT-4o to implant predefined false premises, followed by filtering by human annotators to ensure quality. (B) Dataset distribution across error types and variants. (C) The evaluation pipeline: model-generated answers are first evaluated by o3 and then verified through manual checks to ensure accurate assessment of input scrutiny ability.

across five domains, finding critique ability tied to training directions, logic tasks more amendable to correction, and strong models outperforming weaker ones in cross-critique (with occasional reverse in self-critique). (Luo et al. 2023) focused on critique ability itself, showing it emerges with model size, self-critique remains hard for top models, and accuracy drops with uncertain problems; they proposed a ”self-check” baseline.

Recent work shifted to proactive premise critique. (Fan et al. 2025) found reasoning models ”overthink” premisemissing questions, while non-reasoning models better identify such irrationality. (Li et al. 2025) proposed PCBench for premise critique evaluation, noting most models rely on explicit prompts, lack autonomy, and reasoning ability does not stably correlate with premise critique ability—highlighting the need to enhance proactive input examination.

Multimodal Studies on multimodal inconsistency detection include (Yan et al. 2025)’s MMIR benchmark, targeting visual-text mismatches in web pages and slides, with models like o1 performing strongly versus open-source models’ struggles. (Yan et al. 2024)’s ErrorRadar, focused on multimodal math reasoning errors, notes GPT-4o and similar models trail human experts by around 10%. (Zhang et al. 2025b)’s MMMC dataset analyzes visual-text conflict hallucinations and compares mitigation strategies like prompt engineering and fine-tuning. (Liu et al. 2025) evaluated visionlanguage models’ (VLMs) robustness to scientific questionanswering distractions, finding most VLMs are more sensitive to text-based ones. Previous efforts also have shed light on how LMMs handle inconsistent or distracting inputs (Shu et al. 2025). However, these works rarely check if these models can autonomously identify cross-modal flaws, inconsistencies, or gaps without explicit prompting. Such an omission hinders a complete understanding of their reliability in real-world settings, where inputs are often noisy,

incomplete, or contradictory. Our work aims to bridge this gap by establishing a framework to evaluate proactive input scrutiny, thereby fostering the development of more robust and trustworthy multimodal systems.

##### Modality Preference in LMMs

Existing studies have shown that multimodal large models suffer from significant modality preferences, hindering cross-modal information integration for reasoning. This limitation leads to issues like hallucination generation (Chen et al. 2024; Zhou et al. 2024; Min et al. 2024) and reduced accuracy in detecting multimodal input errors (Yan et al.

- 2024). Consequently, research on modality preferences in these

models has become a crucial area of inquiry. (Zhang et al.

- 2025a) developed the MC2 benchmark to assess modality preference under controlled evidence conflict scenarios. It found that 18 tested Multimodal Large Language Models (MLLMs) showed clear modality bias and proposed a representation engineering-based method to control this preference. (Dong et al. 2025) examined foundation models (FMs) in cross-modal conflict situations, revealing that while FMs achieved a recognition rate of 90% in unimodal scenarios, this rate dropped significantly in multimodal contexts due to cross-modal attention imbalance. (Zheng et al.

2025) investigated the impact of modality bias on MLLMs, diagnosing its current state, proposing a research roadmap, identifying key factors, and experimentally validating their influence.

Unlike previous studies that focused on image-text inconsistencies, our dataset covers a wider range of error types for more detailed analysis. We confirm that most large models exhibit modality preferences, consistent with prior findings. Additionally, we conduct an in-depth analysis of the relationship between modality preferences and models’ input scrutiny ability.

### The ISEval Framework

This section details the design and methodology of the Input Scrutiny Evaluation Framework (ISEval). We begin by presenting the preliminaries, which define essential terms and concepts for evaluating LMMs. This is followed by an introduction to the False Premise Taxonomy, categorizing the types of errors used in our framework. Next, we explain the data construction pipeline, detailing how our erroneous inputs are generated. Finally, the evaluation metrics are introduced, outlining how model performance is measured.

##### Preliminaries

The input to an LMM is denoted as I, with textual input specified as It and visual input as Iv. To evaluate the input scrutiny capability of LMMs, we construct erroneous inputs Ie by rewriting It and implanting seven types of predefined errors e into it separately.

Notably, for certain error types, inconsistencies may arise between visual and textual inputs when ∃c ∈ Iv,c′ ∈ It where c ⊥ c′. This condition indicates that at least one semantic concept c from the visual input logically contradicts a semantic concept c′ from the textual input. Such cases are categorized as Cross-Modal Inconsistency, a specific error type characterized by direct conflicts in semantic or factual information between Iv and It.

A model is considered to possess input scrutiny capability

under current flawed input Ie when its output A identifies the implanted error e without relying on explicit prompting to check for errors.

##### False Premise Taxonomy

To comprehensively assess LMMs’ proactive input scrutiny capabilities, we developed a broad-ranging error classification system based on MathClean (Liang et al. 2025), which comprises three major categories and seven sub-categories of errors for constructing erroneous inputs Ie.

Expression Error Expression errors pertain to issues in the formulation or clarity of It’s language or references, preventing the model from correctly interpreting the given information.

- • Unclear Citation: In multimodal prompts, the failure of It to explicitly specify the referent object (specific entities mentioned in text, particular elements within Iv) prevents the model from accurately identifying the target subject. This ambiguity leads to comprehension defects, such as vague understanding or multiple interpretations of the prompt’s intent. It does not involve direct conflicts between modalities but merely obscures the textual basis for problem-solving.
- • Grammatical or Wording Error: It containing grammatical inaccuracies (faulty sentence structures, incorrect unit conversions) or inappropriate word choices (semantically contradictory expressions) impede the model’s accurate understanding of the stated preconditions. Consequently, the model is unable to derive correct answers due

to misinterpreting Ie. This error type belongs to CrossModal Inconsistency because the flawed expressions in

It can conflict with the semantic concepts presented in Iv.

• Misuse Confusion: This category specifically highlights instances where It uses terms improperly (including professional terms and basic concept terms), describing objects with incorrect terminology, leading to premise errors and interfering with the model’s understanding. This error type is classified as Cross-Modal Inconsistency as the incorrect terminology in It contradicts the actual concepts reflected in Iv.

Conditional Error Conditional errors arise when the conditions provided in It are flawed, incomplete, or contradictory, making it impossible for the model to establish a valid basis for its response.

- • Irrelevant or Incorrect Condition: It includes content or conditions extraneous to problem-solving (supplementary information that does not influence the final answer), which can interfere with the model’s ability to

identify core conditions in Ie, potentially leading to misdirection. It does not constitute a cross-modal contradiction.

- • Lacking Condition: In Ie lacks necessary conditions for deriving the correct answer (information missing from It but present in Iv, or entirely absent from both It and Iv), rendering it impossible for the model to directly compute or infer the required solution. This error type is a form of Cross-Modal Inconsistency. From the perspective of content integrity, it conflicts with the conditions required for normal problem-solving, and thus can be considered conflicting with the situation where problem-solving is carried out based on complete conditions combined with images.
- • Exclusive Condition: It presents two or more conditions that cannot hold simultaneously (conflicting values for the same attribute), creating contradictions that prevent

the model from establishing a consistent premise in Ie and thus obtaining a valid answer. This error type falls under Cross-Modal Inconsistency as the mutually exclusive conditions in It may clash with the unified information shown in Iv.

Reasoning Error Reasoning errors involve flaws in the logical structure or guidance provided in It, which can lead the model down an incorrect path of deduction or calculation.

• Misguided Logic: It contains erroneous reasoning steps or flawed logical guidance (incorrect formulas, inverse logical sequences) that mislead the model. This causes the model to perform calculations or deductions based on an incorrect logical framework for Ie, inevitably resulting in inaccurate outcomes. It does not involve crossmodal contradictions.

Ultimately, the False Premise Taxonomy offers a comprehensive set of errors designed to test a multimodal model’s ability to scrutinize its inputs. These errors, classified into Expression, Conditional, and Reasoning types,

Expression Error Conditional Error Reasoning Error

Grammatical or Wording Error

Irrelevant or Incorrect Condition

Model Metric

Unclear Avg Citation

Misuse Confusion

Lacking Condition

Exclusive Condition

Misguided Logic

SEDR 3.33 1.00 4.33 0.00 4.33 3.33 16.67 4.71 GEDR 55.33 56.33 64.00 39.00 48.00 39.00 84.33 55.14

GPT-4o

SEDR 6.00 4.00 3.00 2.67 2.67 4.00 36.67 8.43 GEDR 41.33 33.67 42.67 31.67 33.67 25.33 65.33 39.10

Claude Sonnet 4

SEDR 23.67 13.67 21.00 14.33 16.67 13.33 51.00 21.95 GEDR 66.00 50.33 61.67 69.00 45.67 27.67 83.67 57.72

Gemini 2.5 pro

SEDR 13.33 14.67 14.33 3.67 14.33 7.67 38.00 15.14 GEDR 61.00 55.67 61.67 60.67 58.00 29.00 81.00 58.14

Grok 3

SEDR 3.33 1.67 2.00 0.33 2.33 1.67 14.33 3.67 GEDR 26.33 26.33 27.33 10.67 24.33 17.00 55.00 26.72

InternVL3-38B-Instruct

SEDR 5.33 1.67 2.00 2.00 2.33 2.67 25.33 5.90 GEDR 24.00 15.00 23.33 14.67 8.00 9.33 41.00 19.33

Qwen2.5-VL-32B-Instruct

SEDR 3.00 4.67 3.67 1.33 3.33 1.67 21.67 5.62 GEDR 46.67 56.67 50.00 36.67 51.67 31.00 67.67 48.62

aya-vision-32b

SEDR 2.00 2.00 1.67 1.33 5.00 0.67 15.67 4.05 GEDR 16.00 20.00 23.00 14.00 22.33 16.67 40.00 21.71

Llama-3.2-11B-Vision-Instruct

SEDR 3.00 2.67 5.33 1.67 4.33 1.33 13.00 4.48 GEDR 31.67 32.33 31.67 22.00 34.00 15.00 49.33 30.86

Qwen2.5-VL-7B-Instruct

SEDR 3.67 4.00 4.00 2.33 5.00 1.00 17.33 5.33 GEDR 30.33 36.33 31.33 26.67 52.67 16.00 52.00 35.05

aya-vision-8b

- Table 1: Spontaneous Error Detection Rate (SEDR) and Guided Error Detection Rate (GEDR) of 10 Large Multimodal Models across seven error subcategories, encompassing Expression Errors (Unclear Citation, Grammatical or Wording Error, Misuse Confusion), Conditional Errors (Irrelevant or Incorrect Condition, Lacking Condition, Exclusive Condition), and Reasoning Error (Misguided Logic). The maximum value and the next largest value of each task are indicated by the bold and underlined text, respectively.

are intended to expose vulnerabilities in a model’s comprehension, consistency checking, and logical deduction. Notably, Grammatical or Wording Error, Misuse Confusion, Lacking Condition and Exclusive Condition can also present as Cross-Modal Inconsistencies, where conflicts arise between textual and visual information.

##### Overview of Data Construction

To systematically evaluate the premise-critical ability of LMMs when dealing with erroneous multimodal inputs, we construct the ISEval-dataset. The core details of this dataset are elaborated as follows:

Data Variants and Distribution In variant design, adhering to a comparative evaluation logic, each base question with a predefined error is generated into two types of erroneous input variants:

• Errorneous inputs without inslicit instructions (Ie−ins): This variant directly assess the model’s ability to autonomously identify erroneous information without instruction. Performance on Ie−ins intuitively reflects the model’s inherent premise-scrutiny capability.

• Errorneous inputs with inslicit instructions (Ie+ins): This variant append an inslicit prompt (”check for premise errors”) to the erroneous input, serving as a compare benchmark. By comparing results on Ie−ins and Ie+ins, we can determine whether the model relies on external guidance or possesses independent reasoning ability in premise evaluation, clarifying the logic underlying its analysis.

For dataset distribution, to ensure comprehensiveness and reliability, we synthesized 300 inputs for each error type. The total number of inputs in ISEval-dataset is thus calculated as: 7 (error types) × 300 (inputs per type) × 2 (variants: Ie−ins and Ie+ins) = 4200. This scale can not only cover diverse evaluation scenarios but also meet the confidence requirements of statistical analysis.

Data Sampling and Synthesis We employed two commonly used datasets, MathVision (Wang et al. 2024a) and MathVista (Lu et al. 2023), as the basic data sources. For each error type, we randomly sampled from these two datasets and then used a few-shot prompting method to drive the large model to generate samples corresponding to the

error type. All synthesized samples have undergone strict manual review to ensure they conform to the defined error type and the expected evaluation standards until the predetermined number of questions is achieved.

##### Evaluation Metrics

To systematically evaluate model responses to instructions with false premises, we define the following metrics for evaluating models’ outputs:

Spontaneous Error Detection Rate (SEDR) This metric denotes the proportion of cases where a model independently identifies and flags inaccuracies in input premises without external guidance, calculated as:

NSE NI−ins e

SEDR =

(1)

where NSE represents the number of instances with successful spontaneous error identification, and NI−ins

denotes the total number of erroneous inputs without explicit guidance.

e

Guided Error Detection Rate (GEDR) This metric measures the percentage of scenarios where a model successfully recognizes and specifies problematic premises upon explicit instructions to ”verify premise accuracy,” with the formula:

NGE NI+ins e

(2)

GEDR =

where NGE stands for the number of instances with successful error identification under prompting, and NI+ins

corresponds to the total number of erroneous inputs with explicit instructions.

e

Modality Trust Preference Score (MTPS) This quantifies a model’s tendency to prioritize visual or textual information amid image-text inconsistencies, expressed as symmetric scores for modality preferences:

###### MTPS = (PV ,PT) (3)

For calculation, an LMM evaluator categorizes model responses to erroneous inputs into three types: image preference, text preference, or no preference. PV and PT respectively represent the proportions of image-preferring and text-preferring responses relative to the total number of erroneous inputs with inter-modal contradictions (NI

):

e

NV NI

PV =

e

(4)

NT NI

(5)

PT =

e

Here, NV and NT denote the counts of image-preferring and text-preferring responses respectively, while NI

encompasses all such erroneous inputs (including those classified as ”no preference”).

e

### Experiment

In this section, we evaluate a range of LMMs using our proposed benchmark. The evaluation covers both closed-source and open-source models under few-shot settings. We begin by introducing the evaluated models and the evaluation protocol. We then present a summary of performance results across different model types and tasks.

##### Evaluation Setup

Evaluated Models. We evaluate a total of 10 LMMs, including 4 closed-source and 6 open-source models spanning various architectures and parameter scales. For the closedsource models, we include GPT-4o (Hurst et al. 2024), Claude Sonnet 4 (Anthropic 2025), Gemini 2.5 pro (Google 2025), Grok 3 (xAI 2025). For open-sourced models, we consider InternVL3-38B-Instruct (Zhu et al. 2025), Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-7B-Instruct (Bai et al. 2025), aya-vision-32b, aya-vision-8b (Cohere 2025) and Llama-3.2-11B-Vision-Instruct (Meta 2024). Response evaluation is performed with o3 (OpenAI 2025) as an automated evaluator. For closed-source models (e.g., GPT-4o, Claude Sonnet 4), we adopt their latest official versions with the temperature set to 0.0 and all other configurations kept at default.

Evaluation Protocols. The benchmark includes questions with two distinct response formats: multiple-choice and open-ended. For open-source models, we use the versions available on the ModelScope platform, with generation settings adjusted for consistency across evaluations. Specifically, InternVL3-38B-Instruct is configured with a temperature of 0.0 to ensure deterministic output. For Qwen2.5VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, Aya-vision-8B, and Aya-vision-32B, we set the temperature to 0.3, disable streaming responses, and adopt random sampling where applicable. All other configuration settings for these models follow their default settings as released by the respective developers. Llama-3.2-11B-Vision-Instruct is used with its default configuration. Additional details about the evaluated models are provided in Appendix.

##### Main Results

Overall Results. We systematically evaluated three core capabilities of LMMs: Spontaneous Error Detection Rate (SEDR), Guided Error Detection Rate (GEDR) and Modality Trust Preference Score (MTPS). For SEDR in Table 1, most models exhibited limited autonomous scrutiny of flawed premises. GPT-4o achieved only 4.71% SEDR, and InternVL3-38B-Instruct scored 3.67%—indicating minimal proactive identification of errors without explicit prompting. Top performers like Gemini 2.5 pro (21.95%) and Grok 3 (15.14%) showed marginal improvements but still reflected restricted spontaneous critical reasoning. In contrast, GEDR shows marked performance gains when models received explicit verify premise accuracy prompts. Grok 3 (58.14% GEDR) and Gemini 2.5 pro (57.72% GEDR) demonstrated stronger critique abilities under guidance, with GPT-4o reaching 55.14% GEDR. This proactive - assisted performance gap reveals a critical shortfall: most LMMs pos-

Cross-Modal Inconsistency

no Cross-Modal Inconsistency

Model

SEDR GEDR MTPS SEDR GEDR MTPS

GPT-4o 3.25 51.83 54.84/44.00 6.67 59.55 35.67/63.56 Claude Sonnet 4 3.42 33.84 59.58/39.42 15.11 46.11 38.34/61.11 Gemini 2.5 pro 16.17 46.34 63.42/35.50 29.67 72.89 46.00/53.34

Grok 3 12.75 51.08 41.00/56.17 18.33 67.56 28.11/71.00 InternVL3-38B-Instruct 1.92 23.75 46.92/48.83 6.00 30.67 35.33/61.78

Qwen2.5-VL-32B-Instruct 2.17 13.92 51.25/46.75 10.89 26.56 33.11/65.89

aya-vision-32b 3.34 47.34 30.92/66.25 8.67 50.34 21.89/77.00 Llama-3.2-11B-Vision-Instruct 2.34 20.50 38.58/59.67 6.33 23.33 24.11/74.55

Qwen2.5-VL-7B-Instruct 3.42 28.25 49.17/46.58 5.89 34.33 32.22/64.33 aya-vision-8b 3.50 34.08 26.67/69.59 7.78 36.33 15.33/82.67

- Table 2: 10 Large Multimodal Models’ performance under cross-modal inconsistency and no cross-modal inconsistency, including SEDR, GEDR, and MTPS. The maximum value and the next largest value of each task are indicated by the bold and underlined text, respectively.

sess latent critique capabilities but fail to activate them autonomously, relying heavily on explicit prompting to identify flawed inputs.

Error Types Performance. Table 1 details the SEDR and GEDR for seven error subcategories, revealing variations in LMMs ability to identify different input flaws. Models demonstrate peak proficiency in identifying logical fallacies in both spontaneous and guided detection, with top performers achieving over 80% success in detecting Misguided Logic when prompted. This divergence reveals that sophisticated logical analysis capacity remains inaccessible without explicit instruction. Performance declines moderately for Surface-Level Linguistic Errors, where guided detection proves reasonably effective but spontaneous recognition remains the lowest among all categories, confirming that grammatical nuances rarely trigger autonomous scrutiny despite their rule-based nature. Detection rates drop substantially for Irrelevant or Incorrect Conditions, which exhibit the weakest guided performance across models, while Exclusive Conditions show consistently poor results in both detection modes.

Modality Trust Preferences. The Modality Trust Preference Score (MTPS) results in Table 2 reveal systematic and context-dependent shifts in modality reliance across models. Under cross-modal inconsistency—where image and text content diverge—most models increase their reliance on visual input, suggesting an effort to resolve semantic conflict by prioritizing image-based grounding. For example, Gemini 2.5 Pro allocates 63.42% of its attention to vision in conflicting contexts, while Claude Sonnet 4 and GPT-4o also exhibit visual-preferred MTPS distributions. However, this trend is not universal. Several models, particularly those with smaller architectures or limited training data, display persistent textual dominance even under contradiction. Notably, aya-vision-8b maintains a strong text preference under inconsistency, while Qwen2.5-VL-7B-Instruct and Llama3.2-11B-Vision-Instruct also show near-balanced or textskewed MTPS. In contrast, under no cross-modal inconsistency, all models shift toward greater textual reliance, re-

gardless of their prior visual weighting. This includes models previously more balanced or visual-biased. These shifts indicate a general tendency to treat text as the primary reference modality in congruent input scenarios. Taken together, the MTPS suggest that higher-capacity models are more likely to modulate modality trust in response to semantic context—favoring vision for disambiguation during inconsistency and defaulting to text when input modalities agree. Conversely, smaller or less adaptive models tend to apply fixed modality weights, limiting their ability to resolve multimodal contradictions effectively.

##### Detailed Analysis

A striking disparity emerges between spontaneous and guided error detection performance across all models. Without explicit prompts to verify inputs, even top-performing models like Gemini-2.5-Pro achieve a SEDR of only 21.95%, while most models (e.g., GPT-4o, InternVL3-38B) score below 5%. This gap mirrors prior observations in LLMs, where models passively accept flawed premises without challenge (Gao et al. 2024; Li et al. 2025). Our work systematically extend this finding to the multimodal domain, showing that LMMs also struggle to identify flawed inputs in the absence of external guidance—even when given access to visual context. This suggests that the added complexity of cross-modal alignment may further suppress spontaneous scrutiny.

While (Deng et al. 2025) revealed that vision-language models often exhibit a default bias toward textual input—even when textual and visual modalities conflict—suggesting a tendency toward blind faith in text, our findings uncover a more nuanced picture. Specifically, our analysis reveals that while most models do favor text in nonconflict scenarios, some large, closed-source models exhibit dynamic shifts in modality trust when faced with image-text contradictions. For example, Gemini 2.5 pro increases its reliance on visual information in contradiction-rich tasks, indicating a capacity for activating visual scrutiny. In contrast, smaller models like aya-vision-8b remain text-dominant regardless of conflict, supporting the idea that architectural scale and training sophistication play key roles in adaptive modality trust. This divergence from prior work highlights the importance of evaluating models not only for static biases but also for context-sensitive trust adjustments, which are critical in real-world applications requiring cross-modal validation.

### Conclusion

This study addresses the underexplored question of whether Large Multimodal Models (LMMs) can actively recognize faulty inputs by introducing the Input Scrutiny Ability Evaluation Framework (ISEval), which includes seven flawed premise categories and relevant metrics to assess LMMs’ input scrutiny capabilities . Our evaluation of 10 advanced LMMs via ISEval reveal key limitations: most models show low Spontaneous Error Detection Rates (SEDR) but improved Guided Error Detection Rates (GEDR) with explicit prompts, indicating reliance on external guidance. Modality

trust varies—Gemini 2.5 pro and Claude 4 balance visual and textual info, while aya-vision-8b and Grok 3 over-rely on text in conflicts. These findings highlight the need to enhance LMMs’ proactive input validation. ISEval provides a benchmark, offering insights to guide development of more reliable multimodal systems .

### References

Anthropic. 2025. Claude 4 Sonnet.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Chen, J.; Zhang, T.; Huang, S.; Niu, Y.; Zhang, L.; Wen, L.; and Hu, X. 2024. ICT: Image-Object CrossLevel Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models. arXiv preprint arXiv:2411.15268v1 [cs.CV].

Cohere. 2025. aya.

Deng, A.; Cao, T.; Chen, Z.; and Hooi, B. 2025. Words or Vision: Do Vision-Language Models Have Blind Faith in Text? arXiv:2503.02199.

Dong, H.; Liu, M.; Zhou, K.; Chatzi, E.; Kannala, J.; Stachniss, C.; and Fink, O. 2025. Advances in Multimodal Adaptation and Generalization: From Traditional Approaches to Foundation Models. arXiv:2501.18592.

Fan, C.; Li, M.; Sun, L.; and Zhou, T. 2025. Missing Premise exacerbates Overthinking: Are Reasoning Models losing Critical Thinking Skill? arXiv:2504.06514.

Gao, J.; Gan, L.; Li, Y.; Ye, Y.; and Wang, D. 2024. Dissecting Dissonance: Benchmarking Large Multimodal Models Against Self-Contradictory Instructions. arXiv:2408.01091.

Google. 2025. gemini-2.5-pro-exp-03-25.

He, Y.; Li, S.; Liu, J.; Wang, W.; Bu, X.; Zhang, G.; Peng, Z.; Zhang, Z.; Zheng, Z.; Su, W.; and Zheng, B. 2025. Can Large Language Models Detect Errors in Long Chain-ofThought Reasoning? arXiv:2502.19361.

Hu, X.; Xiong, T.; Yi, B.; Wei, Z.; Xiao, R.; Chen, Y.; Ye, J.; Tao, M.; Zhou, X.; Zhao, Z.; Li, Y.; Xu, S.; Wang, S.; Xu, X.; Qiao, S.; Wang, Z.; Kuang, K.; Zeng, T.; Wang, L.; Li, J.; Jiang, Y. E.; Zhou, W.; Wang, G.; Yin, K.; Zhao, Z.; Yang, H.; Wu, F.; Zhang, S.; and Wu, F. 2025. OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use. In Che, W.; Nabende, J.; Shutova, E.; and Pilehvar, M. T., eds., Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 7436–7465. Vienna, Austria: Association for Computational Linguistics. ISBN 979-8-89176-251-0.

Hua, T.; Yun, T.; and Pavlick, E. 2025. How Do VisionLanguage Models Process Conflicting Information Across Modalities? arXiv:2507.01790.

Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Radford, A.; et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Li, J.; Li, G.; Chang, Y.; and Wu, Y. 2025. Don’t Take the Premise for Granted: Evaluating the Premise Critique Ability of Large Language Models. arXiv:2505.23715.

Li, M.; Chen, K.; Bi, Z.; Liu, M.; Peng, B.; Niu, Q.; Liu, J.; Wang, J.; Zhang, S.; Pan, X.; Xu, J.; and Feng, P. 2024a. Surveying the MLLM Landscape: A Meta-Review of Current Surveys. arXiv:2409.18991.

Li, X.; Wang, W.; Li, M.; Guo, J.; Zhang, Y.; and Feng, F. 2024b. Evaluating Mathematical Reasoning of Large Language Models: A Focus on Error Identification and Correction. arXiv:2406.00755.

Liang, H.; Qiang, M.; Li, Y.; He, Z.; Guo, Y.; Zhu, Z.; Zhang, W.; and Cui, B. 2025. MathClean: A Benchmark for Synthetic Mathematical Data Cleaning. arXiv:2502.19058.

Lin, Z.; Gou, Z.; Liang, T.; Luo, R.; Liu, H.; and Yang, Y. 2024. CriticBench: Benchmarking LLMs for CritiqueCorrect Reasoning. arXiv:2402.14809.

Liu, M.; Chen, H.; Wang, J.; and Zhang, W. 2025. On the robustness of multimodal language model towards distractions. arXiv:2502.09818.

Lu, P.; Bansal, H.; Xia, T.; Liu, J.; Li, C.; Hajishirzi, H.; Cheng, H.; Chang, K.-W.; Galley, M.; and Gao, J.

- 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255.

Luo, L.; Lin, Z.; Liu, Y.; Shu, L.; Zhu, Y.; Shang, J.; and Meng, L. 2023. Critique Ability of Large Language Models. arXiv:2310.04815. Meta. 2024. Meta. Min, K.; Kim, M.; il Lee, K.; Lee, D.; and Jung, K.

- 2024. Mitigating Hallucinations in Large Vision-Language Models via Summary-Guided Decoding. arXiv preprint arXiv:2410.13321v3 [cs.AI]. OpenAI. 2025. o3.

Shen, C.; Wong, Z. H.; He, R.; Liang, H.; Qiang, M.; Meng, Z.; Zhao, Z.; Zeng, B.; Zhu, Z.; Cui, B.; and Zhang, W. 2025. Let’s Verify Math Questions Step by Step. arXiv:2505.13903.

Shu, D.; Zhao, H.; Hu, J.; Liu, W.; Payani, A.; Cheng, L.; and Du, M. 2025. Large Vision-Language Model Alignment and Misalignment: A Survey Through the Lens of Explainability. arXiv:2501.01346.

Tang, F.; Xu, H.; Zhang, H.; Chen, S.; Wu, X.; Shen, Y.; Zhang, W.; Hou, G.; Tan, Z.; Yan, Y.; Song, K.; Shao, J.; Lu,

- W.; Xiao, J.; and Zhuang, Y. 2025. A Survey on (M)LLMBased GUI Agents. arXiv:2504.13865.

Wang, K.; Pan, J.; Shi, W.; Lu, Z.; Ren, H.; Zhou, A.; Zhan, M.; and Li, H. 2024a. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37: 95095–95169.

Wang, Z.; Han, Z.; Chen, S.; Xue, F.; Ding, Z.; Xiao, X.; Tresp, V.; Torr, P.; and Gu, J. 2024b. Stop Reasoning! When Multimodal LLM with Chain-of-Thought Reasoning Meets Adversarial Image. arXiv:2402.14899.

xAI. 2025. grok3.

Yan, Q.; Fan, Y.; Li, H.; Jiang, S.; Zhao, Y.; Guan, X.; Kuo, C.-C.; and Wang, X. E. 2025. Multimodal Inconsistency Reasoning (MMIR): A New Benchmark for Multimodal Reasoning Models. arXiv:2502.16033.

Yan, Y.; Wang, S.; Huo, J.; Li, H.; Li, B.; Su, J.; Gao,

- X.; Zhang, Y.-F.; Xu, T.; Chu, Z.; Zhong, A.; Wang, K.; Xiong, H.; Yu, P. S.; Hu, X.; and Wen, Q. 2024. ErrorRadar: Benchmarking Complex Mathematical Reasoning of

Multimodal Large Language Models Via Error Detection. arXiv:2410.04509.

- Zhang, Y.; Ma, J.; Hou, Y.; Bai, X.; Chen, K.; Xiang, Y.; Yu, J.; and Zhang, M. 2025a. Evaluating and Steering Modality Preferences in Multimodal Large Language Model. arXiv:2505.20977.
- Zhang, Z.; Zhou, W.; Zhao, J.; and Li, H. 2025b. Robust Multimodal Large Language Models Against Modality Conflict. arXiv:2507.07151.

Zhao, Y.; Gan, G.; Zhao, C.; and Cohan, A. 2025. Are multimodal LLMs robust against adversarial perturbations? RoMMath: A systematic evaluation on multimodal math reasoning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 11653–11665.

Zheng, X.; Liao, C.; Fu, Y.; Lei, K.; Lyu, Y.; Jiang, L.; Ren, B.; Chen, J.; Wang, J.; Li, C.; Zhang, L.; Paudel, D. P.; Huang, X.; Jiang, Y.-G.; Sebe, N.; Tao, D.; Gool, L. V.; and Hu, X. 2025. MLLMs are Deeply Affected by Modality Bias. arXiv:2505.18657.

Zhou, G.; Yan, Y.; Zou, X.; Wang, K.; Liu, A.; and Hu, X. 2024. Mitigating Modality Prior-Induced Hallucinations in Multimodal Large Language Models via Deciphering Attention Causality. arXiv preprint arXiv:2410.04780v2 [cs.CV]. Zhu, J.; Wang, W.; Chen, Z.; Liu, Z.; Ye, S.; Gu, L.; Tian, H.; Duan, Y.; Su, W.; Shao, J.; Gao, Z.; Cui, E.; Wang, X.; Cao, Y.; Liu, Y.; Wei, X.; Zhang, H.; Wang, H.; Xu, W.; Li, H.; Wang, J.; Deng, N.; Li, S.; He, Y.; Jiang, T.; Luo, J.; Wang, Y.; He, C.; Shi, B.; Zhang, X.; Shao, W.; He, J.; Xiong, Y.; Qu, W.; Sun, P.; Jiao, P.; Lv, H.; Wu, L.; Zhang, K.; Deng, H.; Ge, J.; Chen, K.; Wang, L.; Dou, M.; Lu, L.; Zhu, X.; Lu, T.; Lin, D.; Qiao, Y.; Dai, J.; and Wang, W. 2025. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv:2504.10479.

### Appendix Prompt Template

Figures 1 to 7 present prompts for generating erroneous inputs corresponding to seven distinct error types. Figure 8 showcases the images associated with these prompts. Figures 9 to 11 provide prompts for evaluating the input scrutiny ability of models and their modality preferences.

[Figure 11]

###### Figure 3: Prompt for generating errorneous input of the Unclear Citation type

[Figure 12]

###### Figure 4: Prompt for generating errorneous input of the Grammatical or Wording Error type

[Figure 13]

###### Figure 5: Prompt for generating errorneous input of the Misuse Confusion type

[Figure 14]

###### Figure 6: Prompt for generating errorneous input of the Irrelevant or Incorrect Condition type

[Figure 15]

###### Figure 7: Prompt for generating errorneous input of the Lacking Condition type

[Figure 16]

###### Figure 8: Prompt for generating errorneous input of the Exclusive Condition type

[Figure 17]

###### Figure 9: Prompt for generating errorneous input of the Unclear Citation type

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Grammatical or Wording Error

Irrelevant or Incorrect Condition

Unclear Citation

Misuse Confusion

[Figure 22]

[Figure 23]

[Figure 24]

Exclusive Condition

Lacking Condition

Misguided Logic

Figure 10: Prompt Template Images

[Figure 25]

- Figure 11: Prompt for evaluating LMMs’ Spontaneous Error Detection

[Figure 26]

###### Figure 12: Prompt for evaluating LMMs’ Guided Error Detection

[Figure 27]

###### Figure 13: Prompt for evaluating LMMs’ Modality Trust Preference

|Model<br><br>|Size|Model Link|
|---|---|---|
|o3|N/A<br><br>|https://openai.com/index/introducing-o3-and-o4-mini/|
|GPT-4o<br><br>|N/A|https://platform.openai.com/docs/models#gpt-4o<br><br>|
|Claude Sonnet 4<br><br>|N/A<br><br>|https://claude.ai/|
|Gemini 2.5 Pro|N/A<br><br>|https://gemini.google.com/app|
|Grok 3|N/A<br><br>|https://console.x.ai/|
|InternVL3-38B-Instruct<br><br>|38B<br><br>|https://modelers.cn/models/Models_Ecosystem/InternVL3-38B|
|Qwen2.5-VL-32B-Instruct|32B|https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct<br><br>|
|Aya-Vision-32B<br><br>|32B<br><br>|https://huggingface.co/CohereForAI/aya-vision-32b|
|Llama-3.2-11B-Vision-Instruct|11B<br><br>|https://huggingface.co/meta-llama/llama-3.2-11b-vision-instruct|
|Qwen2.5-VL-7B-Instruct|7B<br><br>|https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct|
|Aya-Vision-8B<br><br>|8B|https://huggingface.co/collections/CohereForAI/c4ai-aya-vision|

###### Table 3: List of AI Models with Sizes and Links

