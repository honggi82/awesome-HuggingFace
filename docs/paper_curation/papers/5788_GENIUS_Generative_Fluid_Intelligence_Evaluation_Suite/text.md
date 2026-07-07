## GENIUS: Generative Fluid Intelligence Evaluation Suite

Ruichuan An*1 Sihan Yang*1 Ziyu Guo2 Wei Dai1 Zijun Shen1 Haodong Li3 Renrui Zhang†2 Xinyu Wei4 Guopeng Li3 Wenshan Wu5 Wentao Zhang‡1

# arXiv:2602.11144v1[cs.LG]11Feb2026

### Abstract

Unified Multimodal Models (UMMs) have shown remarkable progress in visual generation. Yet, existing benchmarks predominantly assess Crystallized Intelligence, which relies on recalling accumulated knowledge and learned schemas. This focus overlooks Generative Fluid Intelligence (GFI): the capacity to induce patterns, reason through constraints, and adapt to novel scenarios on the fly. To rigorously assess this capability, we introduce GENIUS (GENerative Fluid Intelligence EvalUation Suite). We formalize GFI as a synthesis of three primitives. These include Inducing Implicit Patterns (e.g., inferring personalized visual preferences), Executing Ad-hoc Constraints (e.g., visualizing abstract metaphors), and Adapting to Contextual Knowledge (e.g., simulating counter-intuitive physics). Collectively, these primitives challenge models to solve problems grounded entirely in the immediate context. Our systematic evaluation of 12 representative models reveals significant performance deficits in these tasks. Crucially, our diagnostic analysis disentangles these failure modes. It demonstrates that deficits stem from limited context comprehension rather than insufficient intrinsic generative capability. To bridge this gap, we propose a training-free attention intervention strategy. Ultimately, GENIUS establishes a rigorous standard for GFI, guiding the field beyond knowledge utilization toward dynamic, general-purpose reasoning. Our dataset and code will be released at: https://github.com/arctanxarc/GENIUS.

### 1. Introduction

Unified Multimodal Models (UMMs) have witnessed remarkable progress recently (Team, 2024; Chen et al., 2025; Xie et al., 2024), delivering impressive results across diverse

*Equal contribution †Project leader ‡Corresponding author 1Peking University 2CUHK 3StepFun 4PolyU 5MSRA. Correspondence to: Wentao Zhang <wentao.zhang@pku.edu.cn>.

Preprint. February 12, 2026.

tasks (An et al., 2025; Li et al., 2025a; Jiang* et al., 2025). Benefiting from the fusion of understanding, UMMs are capable of processing complex, interleaved contexts and exhibiting extensive world knowledge to reshape the generative paradigm. Consequently, they are widely regarded as a milestone on the path toward Artificial General Intelligence (AGI). However, this rapid advancement invites a natural question: How far are current UMMs from achieving true general intelligence regrading visual generation?

To investigate this problem, drawing upon existing literature (Cattell, 1963; Schipolowski et al., 2014; Kent, 2017), we deconstruct General Intelligence in visual generation into two primary components: Crystallized Intelligence (CI) and Fluid Intelligence (FI). Current development and evaluation focus of UMMs mainly targets CI (i.e., the capacity for memorization and retrieval of pre-trained knowledge). For instance, a model’s ability to generate a flawless “cat” often stems from exposure to billions of instances during training, followed by probabilistic reproduction during inference. However, this trend has severely masked a critical but long-ignoring deficiency concerning FI of visual generation skills, termed Generative Fluid Intelligence (GFI), (i.e., the ability to perform inducing, reasoning and ad-hoc adaptation in novel scenarios). As shown in Fig. 1, the “Simple Constraint” task requires the model to identify ad-hoc rules (e.g., abstract symbol denotes “rain”) and apply them to the visual output, instead of just retrieving static concepts.

Despite its critical importance, research along this direction remains limited (shown in Tab. 1):

- • First, a formal definition is absent. This theoretical void impedes the foundational guidance, which is necessary for steering UMMs toward general intelligence.
- • Second, benchmarks are inadequate. Current evaluations predominantly assess model memorization and retrieval, failing to disentangle static knowledge to probe the true bounds of general intelligence.
- • Third, systematic analyses are lacking. The lack of investigations into the failure modes leaves critical questions of why models fail and how to improve unanswered.

To bridge these gaps, we introduce GENIUS (GENerative Fluid Intelligence EvalUation Suite), the first framework dedicated to the systematic evaluation of GFI. Drawing

Implicit Pattern Induction

Task: Implicit Pattern Generation (86 Samples)

Overall Style Visual Feature Spatial Relationship Palette Entity

Context: Elena prefers the patterns of the first image and dislikes the second.<image 1><image 2>

Context: The artist is fascinated by the first image and finds the second image lack of aesthetic appeal.<image 1><image 2>

Context: Alexander prefers the character in this image.<image 1> Instruction: Modify this image by trans-forming the character into one that matches the style Alexander prefers.<image 2>

Context: I prefer the palette in this image.<image 1> I dislike the palette in this image.<image 2>

Instruction: Please fill this image with my preferred palette, and show the characteristics of the palette.<image 3>

Instruction: Adapt this image into the specific style that the artist loves.<image 3>

Instruction: Update to match Elena's specific preferences.<image 3>

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

- <image 1><image 2> Instruction: Please draw a squirrel during the daytime.

|[Figure 5]|
|---|

|[Figure 6]|
|---|

Context: On this planet, objects do not wear out over time. This is a newly bought piece of clothing,<image 1> and this is the same clothing after being worn for a year.<image 2>

Instruction: Please draw the appearance of this pants after being worn for a year. <image 3>

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

Context: In this world, the natural environment is affected by the positive and negative emotions of humans.<image 1><image 2>

Instruction: Please generate the the weather outside the laboratory. <image 3>

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

[Figure 29]

| |
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

Instruction: Please generate a a photo of <WN> working, dressing in a white lab coat, holding a test tube in his hand. He appears to be looking anxious.

Context: Smith likes the spatial relationship between the two objects in this image.<image 1>

Instruction: Modify this image to the spatial relationship

that Smith likes.

- <image 2>

Ad-hoc Constraint Execution

Task: Symbolic Constraint Generation (153 Samples)

Operation Implementation Visual Metaphor Layout Visual Feature Instance Binding

Context: The transformation from this image to this image depicts the execution of operation <f>.<image 1><image 2> The transformation from this image to this image depicts the execution of operation <g>.<image 3><image 4>

Context: This visual pattern indicates that the trumpet in the image is 'Quiet/Calm'.<image 1> While this visual pattern indicates that the trumpet is 'Loud/Noisy'.<image 2>

Context: This is <WN>, a cheerful young man with short wavy brown hair and fair skin. <WN> has a blue T-shirt. <WN> enjoys eating street food while casually dressed. <WN> wears his green athletic short-sleeved shirt when he goes running. <WN> will wear an afroshaped wig at the party. <WN> works in a biological laboratory.<image 1>

Context: The spatial configuration in this image is A.<image 1> The one in this image is B.<image 2>

Context: Feature A is the object in this image.<image 1> Feature D is the environ of this image.<image 2> Instruction: Please depict Feature A within Feature D.

Instruction: Redistribute the elements in this image into configuration B.<image 3>

Instruction: Based on the relationship between visual patterns and perception described above, make the dog in this image seem quiet.<image 3>

Instruction: Please generate the appearance of the object in this image after first performing operation <f> and then performing operation <g>.<image 5>

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

Task: Visual Constraint Generation (60 Samples)

Task: Multi-Semantic Generation (110 Samples)

Simple Constraint Complex Constraint Noun Phrase Verb Phrase Adjective Phrase

Context: This is a visual description of over the moon.<image 1>

Context: Editing rules: If there are trees in the image, please plant flowers when seeing this animal's picture.<image 1> If there are flowers in the image, please plant trees when seeing this animal's picture.<image 2> If the condition does not fall into any of the above situations, do not perform any operation.

Context: This symbol in this image can make raining happen.<image 1> Whereas the symbol in this image can make snowing happen.<image 2>

Context: This is a picture of eager beaver.<image 1> Instruction: Generate an image of a big cheese following the same semantic interpretation (literal vs. figurative) as the context image.

Context: This is a visual description of climb the walls. <image 1>

Instruction: Generate an image of a loose cannon following the same semantic interpretation (literal vs. figurative) as the context image.

Instruction: Generate an image of a couch potato following the same semantic interpretation (literal vs. figurative) as the context image.

Instruction: Under the current circumstances, the city in this image encounters the symbol in this image.<image 3> <image 4> Please illustrate what happens next in the city.

Instruction: I show you this animal in this image.<image 3>

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

|[Figure 48]|
|---|

Please edit this image according to the rules above.<image 4>

Contextual Knowledge Adaptation

Task: Prior-Conﬂicting Generation (101 Samples)

Abnormal Biological Growth Gravity Anomaly Abnormal Animal Behavior Time Reversal Weather Anomaly

Context: In this world, living things grow according to their own rules. A young lion's size is like the one in this image,<image 1> while an elderly lion's size is like this.<image 2>

Context: On this planet, gravity is determined by color.<image 1><image 2>

Context: The rhythms of animals in this world all conform to what is shown in the following figures.

Instruction: Please generate the appearance of the white earphones in this image on this planet.<image 3>

Instruction: The wolf in this world grows like a bear. Please generate the appearance of an elderly wolf in this world.

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

- Figure 1. An overview of GENIUS benchmark. It is hierarchically structured into three dimensions, five tasks, and diverse sub-tasks.

from the Cattell-Horn-Carroll (CHC) theory (Schneider & McGrew, 2012), we distill three core primitives of FI: (I) Inductive Inference, (II) Abstract Dynamic Reasoning and (III) Adaptive Inhibition. We materialize these theoretical concepts into three corresponding dimensions within GENIUS. To ensure a fine-grained assessment, we further construct five novel and well-designed tasks that specify concrete capabilities within each dimension. For each task, we employ a hybrid evaluation comprising three metrics: (I) Rule Compliance, which challenges the model’s precision in following ad-hoc rules; (II) Visual Consistency, which assesses the stability of generated attributes under logical constraints; and (III) Aesthetic Quality, which demands that the model maintains fundamental aesthetic standards. Through manual curation, our suite features tasks well designed by multimodal experts. Unlike traditional benchmarks that prioritize static world knowledge, generation quality or safety, we ensure that every sample presents a dynamic and novel rule, strictly decoupling static knowledge to offer a pure quantification of the model’s GFI capabilities.

With GENIUS, we systematically evaluate 12 representative open-source and proprietary models. To ensure evalu-

ation robustness, we provide manually annotated hints for each test case, which have undergone at least three rounds of cross-validation to support unbiased hybrid evaluation. Overall, our results reveal clear gaps between current stateof-the-art (SOTA) models and general intelligence. Surprisingly, pre-planning and post-reflection yield marginal gains. These findings expose under-explored deficiencies in current generative models, highlighting the urgent need to advance fluid intelligence in the next generation of UMMs.

Building on these findings, we move beyond evaluation to investigate the underlying mechanisms of failure. Taking Bagel (Deng et al., 2025) as an example, by visualizing the attention distribution of it, it surprisingly shows irregular noise and spikes across the multimodal context. This indicates that the model struggles to accurately focus on critical new rules in the context. Inspired by the theoretical perspective of In-Context Learning (ICL) as Implicit FineTuning (Dherin et al., 2025), we demonstrate that ICL process is mathematically equivalent to update specific model parameters while generation. Then, we offer a possible view: the imbalanced attention distribution results in insufficient guidance during implicit gradient descent, which causes the

###### Table 1. Comparison of representative benchmarks. * denotes understanding tasks. ✔

indicates partial satisfaction (e.g., For Manual Curated/Annotation, it implies a combination of human curation and automatic methods). GENIUS pioneers Fluid Intelligence evaluation, featuring multi-image input, multimodal interleaved context, hybrid metrics, and pure manually curated and annotated testcases.

✗

Multimodal Interleaved Context

Task Dimension

Manual Curated Annotation

Multi-Image Input

Fluid Intelligence

Hybrid Evaluation

Benchmark # Samples

Implicit Pattern Induction

Explicit Constraint Execution

Contextual Knowledge Adaptation

GenEval (Ghosh et al., 2023) 2.2k ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ WISE (Niu et al., 2025) 1.0k ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✔

✗

RISE (Zhao et al., 2025) 360 ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✓ DPG-Bench (Hu et al., 2024) 1.0k ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗

DreamBooth (Ruiz et al., 2023) 3.0k ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ UnifyBench (An et al., 2025) 0.1k ✗ ✗ ✗ ✗ ✔

✗

✗ ✗ ✓ Tiif-Bench (Wei et al., 2025) 5.0k ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✔

✗

OpenING* (Zhou et al., 2025) 5.4k ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✔

✗

UniEval* (Li et al., 2025d) 4.2k ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✗ MME-Unify* (Xie et al., 2025b) 4.1k ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✔

✗

RealUnify* (Shi et al., 2025) 1.0k ✓ ✗ ✗ ✗ ✗ ✗ ✗ ✓ ROVER (Liang et al., 2025) 1.3k ✓ ✗ ✗ ✗ ✗ ✗ ✓ ✓ WEAVE (Chow et al., 2025) 0.1k ✓ ✗ ✔

✗

✗ ✗ ✗ ✓ ✓ GENIUS (Ours) 0.5k ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

gradient direction vague or stochastic, failing to overcome the inertia of pre-trained priors. Based on this, we design a training-free mechanism as a strong baseline. The results show consistent performance gains across all tasks, which not only validates the effectiveness of our method but also corroborates the rationality of our theoretical framework.

In summary, our core contributions are as follows:

- • We formally define Generative Fluid Intelligence (GFI), filling a theoretical void to provide foundational guidance for steering UMMs toward general intelligence.
- • We introduce GENIUS, the first benchmark designed to purely quantify GFI. It features 510 expert-curated samples spanning three dimensions and five tasks, supported by a robust hybrid evaluation protocol.
- • We systematically evaluate 12 representative open-source and proprietary models. Results reveal significant deficits in SOTA models, underscoring the clear performance gap.
- • We trace GFI failures via theoretical and empirical analysis. Then, we propose a training-free strategy that boosts performance, effectively activating the model’s GFI.

### 2. GENIUS

#### 2.1. Benchmark Overview

Grounded in the Cattell-Horn-Carroll (CHC) theory (Schneider & McGrew, 2012), General Intelligence is defined not merely by Crystallized Intelligence but more fundamentally by Fluid Intelligence, which is the capacity to induce, reason, and adapt in novel scenarios, independent or even contrary to prior knowledge. Drawing from this, we formally define Generative Fluid Intelligence (GFI) for visual generation as the synthesis of three core primitives, where humans effortlessly demonstrate these capabilities through:

• Induce Implicit Pattern (Inductive Inference): Distilling implicit patterns and intrinsic attributes from observations.

- • Execute Ad-hoc Constraints (Abstract Dynamic Reasoning): Executing logical reasoning within the bound of ad-hoc defined visual or symbolic constraints.
- • Adapt based on Contextual Knowledge (Adaptive Inhibition): Adjusting based on contextual cues, even when necessitating a deviation from established common sense.

However, these capabilities still present significant challenges for current UMMs. To objectively assess model performance in these areas and pinpoint limitations, we introduce GENIUS (GENerative Fluid Intelligence EvalUation Suite), the first benchmark dedicated to evaluate Generative Fluid Intelligence. As illustrated in Fig. 5, the suite comprises a total of 510 expert-curated samples spanning 20 diverse sub-tasks, structurally distributed across three core dimensions: 86 for Implicit Pattern Induction, 213 for Adhoc Constraint Execution, and 211 for Contextual Knowledge Adaptation. Unlike previous benchmarks that prioritize static world knowledge, generation quality or safety, GENIUS is constructed to strictly exclude prior knowledge. It systematically evaluates models across (I) Inductive Inference, (II) Abstract Ad-hoc Reasoning, and (III) Adaptive Inhibition, as aforementioned, purely quantifying their aptitude for solving novel problems.

#### 2.2. Benchmark Construction

For each category of GENIUS, we curate a diverse set of high-quality, expert-designed test cases. Each instance comprises a multi-modal interleaved context and removal of any single modality from context makes the task unsolvable.

Implicit Pattern Induction: This dimension mainly contains a novel task: Implicit Pattern Generation, which assesses the capacity to deduce unstated visual preferences from context and apply them for generation. As shown in Fig. 1, the interleaved input presents images of varying styles alongside specific user preferences. During testing,

the model is required to induce the target stylistic pattern based on these preferences and manifest it in the generated output. This task necessitates the integration of both modalities: relying solely on images would cause the model to blindly conflate visual features, while relying solely on text would leave the stylistic preference undefined, causing the model to collapse into its pre-trained distribution.

Ad-hoc Constraint Execution: A significant ability of FI is to perform dynamic reasoning under newly defined, ad-hoc rules. To systematically assess this, we construct two complementary tasks: (I) Visual Constraint Generation (II) Symbolic Constraint Generation, where novel meanings are assigned to symbols or images within the context. Crucially, we deliberately select elements that are devoid of pre-existing semantic associations, such as an image patch (e.g., “defining a plain blue square as an operation to remove the specifc object”) or a mathematical symbol (e.g., “defining a function f as an instruction to make the object melt”). This design reflects the model’s capability to solve novel problems by reasoning abstractly. The absence of either visual or textual modalities would fundamentally compromise the establishment of these ad-hoc rules, making it impossible to link the element to its new definition.

Contextual Knowledge Adaptation: A model possessing FI must exhibit flexible adaptation rather than rigid adherence to pretrained knowledge. We introduce two novel tasks to evaluate this: (I) Prior-Conflicting Generation: The model must reason based on newly defined common sense presented in the context (e.g., “object weight is determined by color”), even if it contradicts established facts. (II) MultiSemantic Generation: This task requires the model to discern whether to interpret a concept literally or metaphorically (e.g., distinguishing “a green hand” as a novice vs. green skin) based on the specific multimodal contexts. In both tasks, missing any modality prevents the precise definition of new knowledge or the clarification of semantics, causing the model to fail in dynamic adaptation.

#### 2.3. Evaluation Metric

Evaluating the ability of GFI remains a challenging task. Recent studies (Gu et al., 2024) have shown that large multimodal models (LMMs) exhibit strong visual reasoning and alignment capabilities, making them suitable as evaluators. Following this, we construct a robust pipeline comprising three orthogonal metrics. We utilize the frontier LMM (Gemini-3-Pro (Google DeepMind, 2025)) as the judge, employing structured prompts (Detailed in the Appendix A.2) and a hybrid evaluation strategy that incorporates manuallycurated hints to ensure rigorous quantification. All metrics are assigned a score of 0 (fail), 1 (partial), or 2 (perfect).

Rule Compliance This metric measures the model’s accuracy in executing strict ad-hoc rules. Since GFI tasks

involve precise constraints (e.g., specific symbolic, layouts or palette), relying on the LMM’s unguided interpretation is unreliable. We therefore adopt a hybrid protocol that grounds automated evaluation in human-verified truth. For each sample, we provide a manually curated eval-hint serving as the “gold standard”. The LMM evaluator strictly compares the output against the specific nouns, adjectives and spatial predicates defined in this eval-hint.

Visual Consistency For some GFI tasks, the visual identity of original objects must remain unchanged (e.g., specific characters or objects). This metric assesses the model’s ability to preserve context during dynamic reasoning. We provide specific hybrid eval-hints for each sample that identify the key visual elements from the reference images. The evaluator verifies the stability of these elements in the generated image, ensuring the model does not hallucinate or discard critical context while following instructions.

Aesthetic Quality Generative outputs must maintain physical coherence even when processing novel or conflicting semantic inputs. We employ a specific prompt to evaluate aesthetic quality, focusing on anatomical logic, lighting, and the presence of AI artifacts (e.g., distorted limbs). This metric ensures that the model’s adaptation to new rules does not come at the cost of basic visual realism.

### 3. Experiment

We conduct a comprehensive evaluation of 12 representative open-source and proprietary models. The open-source model comprises Qwen-Image-Edit-2511 (Wu et al., 2025b), GLM-Image (Zhipu AI Team, 2026), FLUX.2-dev (Labs, 2025), NextStep-1 (Team et al., 2025), Emu3.5-Image (Cui et al., 2025) and Bagel (Deng et al., 2025). The proprietary category includes leading commercial models: Nano Banana (Google, 2025a) and its Pro variant (Google, 2025b), SeeDream series (4.0 & 4.5) (Seedream et al., 2025) and GPT-Image (OpenAI Team, 2025).

As outlined in Sec. 2.3, we employ Gemini-3-Pro (Google DeepMind, 2025) as the evaluator. To mitigate stochastic variance and ensure robustness, we report the final scores as the average of three independent runs for each sample. The quantitative results are shown in Tab. 2. Given the diversity of multimodal input formats, we adopt interleaved inputs for models capable of processing them, while utilizing a decoupled format for those that do not. Further ablation studies concerning interleaved formats are in the Appendix D.1.

#### 3.1. Main Results

Generative Fluid Intelligence (GFI) remains a significant bottleneck for current models. Our results reveal a stark reality: even the state-of-the-art proprietary model, Nano Banana Pro, achieves an overall score of only 57.19, falling

- Table 2. Main Results. We evaluate models across dimensions. The Overall column represents weighted score across all tasks, calculated using a ratio of RC:VC:AQ = 6:3.5:0.5. Ours is implemented on Bagel. The best and second best performances are highlighted.

Implicit Pattern Induction Ad-hoc Constraint Execution Contextual Knowledge Adaptation

Method Interleaved Overall

Implicit Pattern Symbolic Constraint Visual Constraint Prior-Conflicting Multi-Semantic

RC VC AQ RC VC AQ RC VC AQ RC VC AQ RC VC AQ

Proprietary Models Nano Banana Pro ✓ 57.19 66.86 44.59 96.51 71.38 50.00 92.11 76.67 66.67 96.67 52.97 41.38 90.59 35.45 - 95.00 Nano Banana ✓ 50.66 56.47 39.04 94.12 60.46 51.91 90.20 68.33 79.17 93.33 35.50 39.47 91.00 30.28 - 93.12 GPT-Image ✗ 47.15 58.14 41.92 93.60 58.82 32.82 93.79 49.17 62.50 92.50 43.50 33.33 90.00 28.64 - 85.45 SeeDream 4.0 ✗ 21.26 12.05 0.70 96.39 21.57 3.44 84.64 40.00 4.17 76.67 30.69 10.34 82.67 30.73 - 80.00 SeeDream 4.5 ✗ 52.84 70.00 59.59 97.06 62.91 41.09 94.37 58.33 62.50 86.67 40.10 41.38 92.57 35.00 - 86.82

Open-Source Models Qwen-Image ✗ 30.58 36.18 27.69 71.05 36.18 27.69 71.05 26.67 45.83 55.83 27.72 20.69 71.78 25.91 - 69.55 GLM-Image ✗ 24.71 32.94 19.86 93.53 22.37 21.15 87.50 27.50 12.50 70.83 20.30 15.52 71.29 17.73 - 70.91 FLUX.2-dev ✗ 34.39 34.30 27.70 88.95 35.76 31.01 87.09 39.17 50.00 59.17 25.25 30.17 84.16 29.82 - 79.82 NextStep-1 ✗ 10.44 10.74 0.40 25.12 11.33 2.54 21.67 21.50 4.20 29.17 15.49 7.55 28.71 12.80 - 20.28 Emu3.5-Image ✗ 36.67 41.86 35.81 83.72 34.97 39.31 86.93 24.17 29.17 42.50 26.24 37.93 82.18 32.87 - 75.46 Omini-Gen2 ✗ 27.87 29.07 26.35 76.16 25.33 30.38 77.96 11.67 41.67 52.50 23.76 34.48 69.80 19.27 - 63.76 Bagel ✓ 26.74 26.74 27.03 84.30 29.61 16.03 76.32 22.50 12.50 49.17 22.28 17.24 74.75 33.49 - 53.67

Ours ✓ 32.92 39.54 44.92 66.71 36.54 26.73 67.11 30.45 35.11 47.84 23.67 36.75 57.78 34.22 - 52.75

short of a passing grade. Meanwhile, representative opensource models like Bagel fall significantly behind, scoring a mere 26.74. These tasks demand ad-hoc reasoning and dynamic adaptation to novel rules, which are less directly grounded by the models’ pre-trained parametric knowledge. Together, these quantitative deficits suggest that while current UMMs have acquired robust capabilities for crystallized reproduction, they remain fundamentally distant from the fluid adaptability required for general-purpose generation.

Current models fail to effectively arbitrate the conflict between pre-trained priors and the given context. As shown in Tab. 2, this deficiency is most pronounced in the Contextual Knowledge Adaptation dimension, where performance consistently drops below other task categories. When ad-hoc instructions explicitly contradict world knowledge (e.g., counter-intuitive physical laws or remapped semantics), models exhibit a strong “cognitive inertia”, frequently defaulting to their pre-trained priors. This suggests that existing architectures lack a robust mechanism to inhibit intrinsic priors, failing to dynamically adapt to the context.

Aesthetic fidelity masks deep logical deficiencies. Our hybrid metric analysis uncovers a pervasive “illusion of competence”: models consistently maintain high Aesthetic Quality scores, yet their performance on Rule Compliance lags substantially behind. This discrepancy suggests that previous model optimization has disproportionately focused on surface-level visual plausibility at the expense of deep context interpretation and logical adherence. By exposing this, GENIUS signals a necessary paradigm shift for next generation of models: moving beyond merely generating “beautiful” pixels to achieving profound context comprehen-

sion and ensuring logically correct visual synthesis.

#### 3.2. Discussion and Analysis

Pre-planing and post-reflection yield marginal gains. We investigated various inference-time enhancement strategies to potentially mitigate performance deficits. Taking Nano Banana Pro and Bagel as examples, we implemented preplanning (activating reasoning mode) and post-reflection (an iterative process where initial generations are evaluated and re-fed as context for refinement). However, as illustrated in Fig. 2(a), empirical results across both Nano Banana Pro and Bagel indicate that these strategies yield only marginal gains. This suggests that current architectures struggle to effectively leverage explicit reasoning for generation.

Context comprehension is the key to solve GFI problems. To isolate the source of failure, we introduced humancurated hints to guide the generation process. Specifically, we employed a progressive intervention strategy: initially utilizing text-only hints, and subsequently constructing multimodal hints to ensure information completeness, thereby explicitly guiding the model’s generation. The results are illustrated in Fig. 2(a). This intervention resulted in substantial performance improvements; however, the degree of improvements varied significantly: Nano Banana Pro exhibited a much more boost compared to Bagel. This observation highlights that accurate context comprehension acts as a critical factor in solving GFI tasks. Meanwhile, solving GFI problems requires not only accurate context comprehension to decode ad-hoc rules but also robust intrinsic model capabilities to execute them, implying that comprehension aids cannot fully compensate for a weaker

[Figure 52]

[Figure 53]

|Context: Feature A is the cloth in this image.<br><br><image 1> Feature B is the style of this image.<br><image 2> Instruction: Please reshape this picture using Feature A and Feature B.<image 3><br><br><br>|[Figure 54]|
|---|
<br><br>|[Figure 55]|
|---|
<br><br>|[Figure 56]|
|---|
<br><br>A. The generated image should feature the character from <image 3> wearing the blue fitness tank top from <image 1>, rendered in a pixel art style like <image 2>.<br>B. The generated image should feature the character from <image 3> wearing the blue fitness tank top from <image 1>.<br>C. The generated image should feature the character from <image 1> wearing the cloth from <image 3>, rendered in a pixel art style like <image 2>.<br>D. The generated image should feature the character from <image 1> rendered in a pixel art style like <image 2>.<br><br><br>VQA Sample|
|---|

|26.74<br><br>65.68<br><br>57.19<br><br>94.11<br><br>VQA Accuracy GENIUS Overall Score<br><br>|
|---|

Ground Truth: A Bagel Nano Banana Pro

(a) The Influence of Different Context (b) VQA Sample & VQA Accuracy (c) Pearson Correlation

- Figure 2. Diagnostic analysis and metric validation. (a) Performance comparison across different context settings. (b) Analysis of the gap between context comprehension (VQA) and generation capabilities. (c) Correlation analysis validating the LMM-as-a-Judge metric.

base model’s generative limitations.

Generative failure primarily stems from an execution gap rather than comprehension deficits. To investigate the root cause, we reformulated the generative tasks into comprehension-oriented Visual Question Answering (VQA) probes. Specifically, we structured these probes as multiplechoice questions that query the model regarding the expected visual appearance of the target image. We utilized our expert hints for Rule Compliance as the ground truth answers, while simultaneously constructing three distractors for each sample to facilitate evaluation. The results are shown in Fig. 2(b). Empirical results reveal a significant disparity: models frequently demonstrate an accurate understanding of the context’s intent but fail to translate this into compliant visual outputs. This suggests that the model’s current cognitive processing of the context, while sufficient for discriminative understanding tasks, lacks the granularity required for generative reconstruction. We hypothesize this stems from two factors: first, the high information density of interleaved contexts, where fine-grained visual nuances (e.g., specific textures) are difficult to fully capture and articulate through limited modalities; and second, a structural inefficiency in current UMM architectures, where rich semantic understanding from the encoder is not effectively propagated to the generative decoder, resulting in a “knowbut-cannot-draw” phenomenon. We further discuss how to enhance this critical contextual comprehension in Sec. 4.1.

- 3.3. Validity of LMM-as-a-Judge

adhering to the same metrics used by the LMM evaluator to compare the consistency between human and LMM scoring.

As shown in Fig. 2(c), the Pearson correlation between human expert ratings and LMM-based scores demonstrates a high degree of alignment. Our analysis reveals exceptionally strong global consistency across all samples: the Pearson correlation coefficient (r) reaches 0.9630 for NanoBanana Pro and 0.9659 for Bagel. Such high linear correlation indicates that the LMM evaluator accurately captures the underlying logic of human judgment in image generation tasks. Furthermore, dimension-specific analysis shows that the Mean Absolute Error (MAE) remains consistently low across multiple metrics, ranging from 0.06 to 0.11. Relative to the 0–2 scoring scale, these errors are quite small, further validating the robustness of the evaluation framework across different models and task dimensions. In conclusion, the LMM-as-a-Judge framework serves as a reliable and effective alternative to human evaluation.

To further ensure the reproducibility and cross-model robustness, we extended our validation to include the open-source Qwen2.5-VL-72B (Bai et al., 2025) as the judge. Empirical results shown in the Appendix C indicate that while Qwen2.5-VL-72B tends to assign systematically lower absolute scores compared to Gemini-3-Pro, suggesting a stricter evaluation criterion. The relative performance trends and model rankings remain identical. This consistency across proprietary and open-source evaluators confirms that the observed performance gaps are intrinsic to the models being tested rather than artifacts of a specific judge, thereby reinforcing the reliability and generalizability of the results.

To verify the reliability of using LMMs as a judge, we conducted an analysis of the correlation between LMM-based automated scoring and human expert judgment. We performed a study by randomly and uniformly sampling 100 output images across various dimensions from two representative models: Nano Banana Pro and Bagel. Five human experts were invited to independently rate these samples,

### 4. A Potential Solution

Evaluation on GENIUS reveals a clear gap between current SOTA models and general intelligence. To diagnose the potential causes of this deficit, we conduct a comprehensive analysis from both theoretical and empirical perspectives,

[Figure 57]

[Figure 58]

| | |
|---|---|
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |

AttentionScore(0-1)

AttentionScore(0-1)

Text Tokens

Text Tokens

VAE Tokens

VAE Tokens

VIT Tokens

VIT Tokens

image 1 image 2 image 3 generated image image 1 image 2 image 3 generated image

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

Context: Feature A is the cloth in this image.<image 1> Feature B is the style of this image.<image 2>

Context: Feature A is the cloth in this image.<image 1> Feature B is the style of this image.<image 2>

Instruction: Please reshape this picture using Feature A and Feature B.<image 3>

Instruction: Please reshape this picture using Feature A and Feature B.<image 3>

[Figure 67]

###### Figure 3. Visualization of attention scores (range [0, 1]). Left: Existing models. Right: Ours.

[Figure 68]

[Figure 69]

###### Current Status: Noisy Gradient

1. Keyword Distillation

| |
|---|

Context: Feature A is the cloth in this image.<image 1> Feature B is the style of this image.<image 2> Instruction: Please reshape this picture using Feature A and Feature B.<image 3>

Signal Tokens

Model:

Noisy Implicit Gradient

(e.g., Area to remain)

- Image 1: Cloth; Cloth style
- Image 2: Pixel Art Style
- Image 3: Character

Implicit Gradient Calculation

| |
|---|

Noise Tokens

|[Figure 70]|
|---|

|[Figure 71]|
|---|

|[Figure 72]|
|---|

(e.g., Background)

[Figure 73]

| |
|---|

Noise Tokens

(e.g., Useless area)

Attention Weight

[Figure 74]

[Figure 75]

[Figure 76]

|[Figure 77]| |
|---|---|
| | |

|[Figure 78]| |
|---|---|
| | |

|[Figure 79]| |
|---|---|
| | |

2. Relevance Mapping

[Figure 80]

[Figure 81]

Ours: Clean Gradient

| |
|---|

Signal Tokens

Clean Implicit Gradient

(e.g., Area to remain)

Implicit Gradient Calculation

| |
|---|

3. Bias Injection

Noise Tokens

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Selected Layers &Steps Only

(e.g., Background)

| |
|---|

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Noise Tokens

###### Physically Dampening Gradient Noise Softmax

(e.g., Useless area)

Attention Weight

(a) Theoretical Impact of Attention on Implicit Gradients (b) The Method Pipeline

Figure 4. Method overview. Guided by the theoretical insight that attention magnitude dictates gradient norms (a), we implement a three-stage pipeline (b) to explicitly suppress noise tokens and rectify the implicit optimization direction.

focusing on the widely applicable Bagel framework.

#### 4.1. Experimental Observation

To investigate the underlying mechanism of failure, we visualized the attention distribution over the entire context, using the image tokens generated during the process as the query. Surprisingly, as shown in the left part of Fig. 3, we found the attention distribution is unreasonable: it exhibits irregular noise and stochastic spikes across the multimodal context. This indicates the model struggles to precisely capture pivotal ad-hoc rules from the context. Instead of pinpointing the critical definition, the attention is spread out indiscriminately across the input. As a result, the model fails to extract the specific signal needed for adaptation and simply falls back to its pre-trained priors.

#### 4.2. Theoretical Analysis

To explain this phenomenon, we adopt the theoretical perspective of In-Context Learning (ICL) as Implicit FineTuning from (Dherin et al., 2025; Ahn et al., 2023; Dai et al., 2023; von Oswald et al., 2023). we perform a derivation on Bagel, which adopts a Mixture-of-Transformer architecture. Since GENIUS targets the generative task, we redefine the function A(u,g) from (Dherin et al., 2025), based on the Bagel model, where A denotes the network layer component responsible for context processing, u repre-

sents the encoding of context and instructions, and g denotes the encoding of intermediate noisy tokens of images. We suppose in the t-th step and the l-th Decoder blocks we have

g(t,l+1) = L(Upl),b(u(l),g(t,l)), where Up is a projection layer in the decoder block, b is the bias of Down layer, and L rep-

resents the l-th block’s forward propagation. And then we can formalize the relationship between u and the (Up,b):

Theorem 4.1. The layer update satisfies following property:

LUp+∆Up, b+∆b(u′,g) = LUp,b(u,g) (1) where the bias perturbation is defined as:

∆b = A(u,g) − A(u′,g) (2) and the upsampling operator perturbation be defined as:

Up(δA)N(A(u′,g))⊤ ∥N(A(u′,g))∥2

x RMS(x)

∆Up =

,N(x) =

And the normalized attention difference is given by:

(3)

##### δA = N(A(u,g)) − N(A(u′,g)) (4)

According to Thm. 4.1, we can demonstrate that in multimodal generation, the ICL process is mathematically equivalent to updating specific model parameters. This successfully extends the conclusions of (Dherin et al., 2025):

We first formalize the vector notations for clarity: let u = (u1,...,un), and u(i) = (u1,...,ui). Then we hope:

i,bi(u(i),g) = LUp,b(u,g) ∀i = 1,2,...,n (5)

LUp

From this objective, we derive expressions for Upi and bi:

Up(δAi)N(A(g))⊤ ∥N(A(g))∥2

(6)

Upi = Up + ∆Upi = Up +

bi = b + ∆bi = b + A(u(i),g) − A(g) (7) where the attention difference term δAi is defined as:

δAi = A(u(i),g) − A(g). (8)

Based on the above derivations, we present the key theorem for iterative parameter updates:

Theorem 4.2. For the (i + 1)-th iteration, Up and b follow the gradient descent update rules below:

Distillation phase, leveraging the semantic reasoning capability of Bagel, we prompt the model to distill task-critical visual cues into a set of region-specific keywords K (the prompt is detailed in the Appendix F.1). Subsequently, during Relevance Mapping, we compute a semantic relevance map S by evaluating the alignment between these keywords and the visual context tokens, where S serves as a proxy for the token’s contribution to the effective gradient signal. Finally, via Bias Injection, we inject a spatial bias F(S) directly into the attention logits:

Attention = softmax

A + λ · F(S)

√

d

V (10)

This formulation ensures tokens with high relevance are emphasized while noise is suppressed. The detailed mathematical formulation is provided in the Appendix F.2. By rectifying the attention landscape, we re-weight the implicit gradient updates, deterministically steering the optimization trajectory to overcome pre-trained priors.

Upi+1 = Upi − h∇UpLi(Upi), bi+1 = bi − ∇b tr δi⊤bi

(9)

where h = 1/∥N(A(g))∥2 denotes the learning rate. Li(Up) = tr ∆⊤i Up is loss function, among which ∆i = Up δ ˆi N(A(g))⊤, δˆi = N(A(u(i),g)) − N(A(u(i+1),g)), and δi = A(u(i),g) − A(u(i+1),g). Combining empirical observations with theoretical analysis, we offer a hypothesis for the deficit in GFI: The imbalanced attention distribution results in a lack of guidance during implicit gradient descent. Consequently, the descent direction becomes stochastic, failing to overcome the pre-trained priors. Full proof of both theorems is in the Appendix G.

#### 4.3. Attention Adjustment Mechanism

Guided by Thm. 4.2, we recognize that the magnitude of attention assigned to the context directly dictates the norm of the implicit gradient update. The irregular attention distribution within the context images, as previously observed in Sec. 4.1, implies that irrelevant “noise” tokens currently contribute significant, erroneous gradient components, thereby diverting the optimization trajectory away from the optimal path. To counteract this, we propose a training-free adjustment mechanism to recalibrate the update direction. By explicitly suppressing the attention weights of noise tokens, we mathematically dampen their corresponding gradient norms (i.e., ∥∆Upnoise∥ → 0), ensuring the implicit fine-tuning is driven solely by critical context signals.

Specifically, we implement this mechanism through a threestage pipeline shown in Fig. 4(b). First, in the Keyword

#### 4.4. Experimental Results

As visualized in Fig. 3 (Right), our mechanism successfully rectifies the originally disordered attention landscape into a sharpened distribution with distinct peaks focused on critical tokens. Quantitative results in Tab. 2 further demonstrate consistent performance gains across nearly all dimensions compared to the baseline Bagel (e.g., boosting the Overall score of 6.18%). This validates that deterministically steering the implicit gradient trajectory effectively activates the model’s latent GFI without requiring parameter updates. Consequently, this mechanism establishes a strong baseline, offering a simple paradigm for improving GFI capabilities.

### 5. Conclusion

In this paper, we introduced GENIUS, the first benchmark dedicated to systematically quantifying Generative Fluid Intelligence (GFI). By grounding in the Cattell-Horn-Carroll (CHC) theory, we formalized GFI into three core dimensions, including Implicit Pattern Induction, Ad-hoc Constraint Execution, and Contextual Knowledge Adaptation, providing a rigorous standard for assessing model capability in novel, reasoning-intensive scenarios. Through systematic evaluation of 12 representative open-source and proprietary models, we reveal a stark reality: even state-of-the-art models like Nano Banana Pro fall short of a passing grade, while open-source models exhibit significant performance deficits. Our analysis exposes a critical “execution gap”, where models struggle to arbitrate conflicts between pre-trained priors and ad-hoc context, often prioritizing aesthetic fidelity over logical rule compliance. Furthermore, we partially trace these failures to attention mechanism defects during infer-

ence and propose a training-free adjustment strategy that effectively activates latent GFI capabilities. We hope that GENIUS will serve as a pivotal testbed for future research, guiding the evolution of next-generation models from crystallized memorization toward true general intelligence.

### Impact Statement

This paper presents a benchmark and theoretical framework aimed at advancing the evaluation of Fluid Intelligence in generative models. By distinguishing Generative Fluid Intelligence (GFI) from standard crystallized knowledge retrieval, our work intends to shift the community focus toward developing systems that possess true adaptability and logic-grounded control. Our contributions align with the goal of creating more robust AI systems by highlighting the “illusion of competence,” a phenomenon where aesthetic quality masks logical deficiencies. This focus encourages transparent evaluation and prevents the deployment of models that appear capable but fail in critical, rule-bound scenarios. Furthermore, improved GFI capabilities may contribute to versatile creative tools and scientific visualization assistants that can accurately follow complex, ad-hoc instructions without hallucination. We do not foresee any unique negative societal consequences beyond those already recognized in the broader field of generative AI.

### References

Ahn, K., Cheng, X., Daneshmand, H., and Sra, S. Transformers learn to implement preconditioned gradient descent for in-context learning, 2023. URL https:// arxiv.org/abs/2306.00297.

An, R., Yang, S., Zhang, R., Shen, Z., Lu, M., Dai, G., Liang, H., Guo, Z., Yan, S., Luo, Y., et al. Unictokens: Boosting personalized understanding and generation via unified concept tokens. arXiv preprint arXiv:2505.14671, 2025.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Barak, T. and Loewenstein, Y. Investigating learningindependent abstract reasoning in artificial neural networks. arXiv e-prints, pp. arXiv–2407, 2024.

Cattell, R. B. Theory of fluid and crystallized intelligence: A critical experiment. Journal of educational psychology, 54(1):1, 1963.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

Chollet, F. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.

Chow, W., Pan, J., Liang, Y., Zhou, M., Song, X., Jia, L., Zhang, S., Tang, S., Li, J., Zhang, F., et al. Weave: Unleashing and benchmarking the in-context interleaved comprehension and generation. arXiv preprint arXiv:2511.11434, 2025.

Cui, Y., Chen, H., Deng, H., Huang, X., Li, X., Liu, J., Liu, Y., Luo, Z., Wang, J., Wang, W., Wang, Y., Wang, C., Zhang, F., Zhao, Y., Pan, T., Li, X., Hao, Z., Ma,

- W., Chen, Z., Ao, Y., Huang, T., Wang, Z., and Wang,
- X. Emu3.5: Native multimodal models are world learners, 2025. URL https://arxiv.org/abs/2510. 26583.

Dai, D., Sun, Y., Dong, L., Hao, Y., Ma, S., Sui, Z., and Wei, F. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers, 2023. URL https://arxiv.org/abs/2212.10559.

Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Dherin, B., Munn, M., Mazzawi, H., Wunder, M., and Gonzalvo, J. Learning without training: The implicit dynamics of in-context learning, 2025. URL https: //arxiv.org/abs/2507.16003.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Google. Introducing Gemini 2.5 Flash Image, our state-of-the-art image model. https://developers.googleblog.com/ introducing-gemini-2-5-flash-image/, 2025a.

Google. Introducing Nano Banana Pro. https: //blog.google/innovation-and-ai/ products/nano-banana-pro/, 2025b.

Google DeepMind. Gemini 3 Pro. https://deepmind. google/models/gemini/pro/, 2025.

Gu, J., Jiang, X., Shi, Z., Tan, H., Zhai, X., Xu, C., Li, W., Shen, Y., Ma, S., Liu, H., et al. A survey on llm-as-ajudge. The Innovation, 2024.

Guo, Z., Zhang, R., Li, H., Zhang, M., Chen, X., Wang, S., Feng, Y., Pei, P., and Heng, P.-A. Thinking-whilegenerating: Interleaving textual reasoning throughout visual generation. arXiv preprint arXiv:2511.16671, 2025.

Guo*, Z., Zhang*, R., Tong*, C., Zhao*, Z., Gao, P., Li, H., and Heng, P.-A. Can we generate images with cot? let’s verify and reinforce image generation step by step. CVPR 2025, 2025.

Hu, X., Wang, R., Fang, Y., Fu, B., Cheng, P., and Yu, G. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Jaeggi, S. M., Buschkuehl, M., Jonides, J., and Perrig, W. J. Improving fluid intelligence with training on working memory. Proceedings of the National Academy of Sciences, 105(19):6829–6833, 2008.

Jiang*, D., Guo*, Z., Zhang*, R., Zong, Z., Li, H., Zhuo, L., Yan, S., Heng, P.-A., and Li, H. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025.

Jin, W., Niu, Y., Liao, J., Duan, C., Li, A., Gao, S., and Liu, X. Srum: Fine-grained self-rewarding for unified multimodal models. arXiv preprint arXiv:2510.12784, 2025.

Kent, P. Fluid intelligence: A brief history. Applied Neuropsychology: Child, 6(3):193–203, 2017.

Koh, J. Y., Fried, D., and Salakhutdinov, R. R. Generating images with multimodal language models. Advances in Neural Information Processing Systems, 36:21487– 21506, 2023.

Labs, B. F. FLUX.2: Frontier Visual Intelligence. https: //bfl.ai/blog/flux-2, 2025.

Li, C., Wu, W., Zhang, H., Xia, Y., Mao, S., Dong, L., Vuli´c, I., and Wei, F. Imagine while reasoning in space: Multimodal visualization-of-thought. arXiv preprint arXiv:2501.07542, 2025a.

Li, J., Zhang, D., Wang, X., Hao, Z., Lei, J., Tan, Q., Zhou, C., Liu, W., Yang, Y., Xiong, X., et al. Chemvlm: Exploring the power of multimodal large language models in chemistry area. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 415–423, 2025b.

Li, Q., Ye, Z., Feng, X., Zhong, W., Qin, L., Chen, R., Li, B., Jiang, K., Wang, Y., Liu, T., et al. Cai: Captionsensitive attention intervention for mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2506.23590, 2025c.

Li, Y., Wang, H., Zhang, Q., Xiao, B., Hu, C., Wang, H., and Li, X. Unieval: Unified holistic evaluation for unified multimodal understanding and generation. arXiv preprint arXiv:2505.10483, 2025d.

Li, Y., Yang, J., Li, B., and Tang, R. Cama: Enhancing multimodal in-context learning with context-aware modulated attention. arXiv preprint arXiv:2505.17097, 2025e.

Liang, Y., Chow, W., Li, F., Ma, Z., Wang, X., Mao, J., Chen, J., Gu, J., Wang, Y., and Huang, F. Rover: Benchmarking reciprocal cross-modal reasoning for omnimodal generation. arXiv preprint arXiv:2511.01163, 2025.

Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Feng, C., Ning, K., Zhu, B., et al. Wise: A world knowledge-informed semantic evaluation for textto-image generation. arXiv preprint arXiv:2503.07265, 2025.

OpenAI Team. New ChatGPT Images is Here. https://openai.com/index/ new-chatgpt-images-is-here/, 2025.

Qin, J., Wu, J., Chen, W., Ren, Y., Li, H., Wu, H., Xiao, X., Wang, R., and Wen, S. Diffusiongpt: Llmdriven text-to-image generation system. arXiv preprint arXiv:2401.10061, 2024.

Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., and Aberman, K. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22500–22510, 2023.

Schipolowski, S., Wilhelm, O., and Schroeders, U. On the nature of crystallized intelligence: The relationship between verbal ability and factual knowledge. Intelligence, 46:156–168, 2014.

Schneider, W. J. and McGrew, K. S. The cattell-horn-carroll model of intelligence. 2012.

Seedream, T., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

Shi, Y., Dong, Y., Ding, Y., Wang, Y., Zhu, X., Zhou, S., Liu, W., Tian, H., Wang, R., Wang, H., et al. Realunify: Do unified models truly benefit from unification? a comprehensive benchmark. arXiv preprint arXiv:2509.24897, 2025.

Sun, Q., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, Y., Gao, H., Liu, J., Huang, T., and Wang, X. Emu: Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.

Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Wang, Y., Rao, Y., Liu, J., Huang, T., and Wang, X. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14398–14409, 2024.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Team, N., Han, C., Li, G., Wu, J., Sun, Q., Cai, Y., Peng, Y., Ge, Z., Zhou, D., Tang, H., Zhou, H., Liu, K., Huang, A., Wang, B., Miao, C., Sun, D., Yu, E., Yin, F., Yu, G., Nie, H., Lv, H., Hu, H., Wang, J., Zhou, J., Sun, J., Tan, K., An, K., Lin, K., Zhao, L., Chen, M., Xing, P., Wang, R., Liu, S., Xia, S., You, T., Ji, W., Zeng, X., Han, X., Zhang, X., Wei, Y., Xu, Y., Jiang, Y., Wang, Y., Zhou, Y., Han, Y., Meng, Z., Jiao, B., Jiang, D., Zhang, X., and Zhu, Y. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025.

von Oswald, J., Niklasson, E., Randazzo, E., Sacramento, J., Mordvintsev, A., Zhmoginov, A., and Vladymyrov, M. Transformers learn in-context by gradient descent, 2023. URL https://arxiv.org/abs/2212.07677.

Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

Wei, X., Zhang, J., Wang, Z., Wei, H., Guo, Z., and Zhang, L. Tiif-bench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.

Xie, J., Darrell, T., Zettlemoyer, L., and Wang, X. Reconstruction alignment improves unified multimodal models. arXiv preprint arXiv:2509.07295, 2025a.

Xie, W., Zhang, Y.-F., Fu, C., Shi, Y., Nie, B., Chen, H., Zhang, Z., Wang, L., and Tan, T. Mme-unify: A comprehensive benchmark for unified multimodal understanding and generation models. arXiv preprint arXiv:2504.03641, 2025b.

Zhang, D., Lei, J., Li, J., Wang, X., Liu, Y., Yang, Z., Li, J., Wang, W., Yang, S., Wu, J., et al. Critic-v: Vlm critics help catch vlm errors in multimodal reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 9050–9061, 2025.

Zhao, S., Hao, S., Zi, B., Xu, H., and Wong, K.-Y. K. Bridging different language models and generative vision models for text-to-image generation. In European Conference on Computer Vision, pp. 70–86. Springer, 2024.

Zhao, X., Zhang, P., Tang, K., Zhu, X., Li, H., Chai, W., Zhang, Z., Xia, R., Zhai, G., Yan, J., et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.

Zhipu AI Team. https://z.ai/blog/glm-image, January 2026.

Zhou, P., Peng, X., Song, J., Li, C., Xu, Z., Yang, Y., Guo, Z., Zhang, H., Lin, Y., He, Y., et al. Opening: A comprehensive benchmark for judging open-ended interleaved image-text generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 56–66, 2025.

Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12966–12977, 2025a.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., and Liu, Z. Qwen-image technical report, 2025b. URL https://arxiv.org/abs/2508.02324.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

### A. Benchmark Details

- A.1. Data Statistics

I

mplicit Pa

tern

Induction

(86)

Ad-hoc Constraint Execution (213)

C

ontextual

K

- n
- o

wled

ge

Adaptation

(211)

Implicit

Pattern Gen.

(86)

Symbolic Constraint Gen.

(153)

Visual Constraint Gen.

(60)

Multi-Se

ma

ntic

Gen.

(110)

Prior-Conflicting Gen.

(101)

(1 SO5t)yvleeral

Spatial Relationship

(14)

Visual Feature

(18)

Palette

(20)

E ntity

(19)

O

peration I

m

ple

mentation

(27)

Visual Metaphor

(29)

Layout (30)

Visual Feature

(38)

Instance Binding (29)

Co(nS2si6mt)rpailnet

Complex Constraint

(34)

Nou n

P hrase

(50)

Ver

b

P

hrase

(30)

Adjective

Phrase (30)

Abnormal Biological Growth

(21)

Gravity Anomaly

(20)

Abnormal Animal Behavior

(20)

Time Reversal (20)

AWnoematahleyr

(20)

[Figure 82]

Figure 5. Data composition pie chart. GENIUS comprises 3 dimensions, 5 tasks, and 20 sub-tasks.

- A.2. Evaluation Prompt

As illustrated in the previously presented prompt templates, we developed a systematic evaluation framework using Large Multimodal Models (LMMs) to assess three key dimensions of generative quality:

Rule Compliance (RC): For each GENIUS sample, an audit of textual-visual alignment is conducted. This process rigorously verifies nouns, adjectives, and spatial constraints to ensure 100% compliance with specific modification requests. Details of the prompt template are provided in Fig. 8.

Visual Consistency (VC): For each GENIUS sample, Visual Consistency may be evaluated multiple times or not at all, depending on how many reference images (objects in the image) need to remain visually consistent. Since we have observed that many open-source models directly copy reference images to cheat (e.g., Bagel, GLM-Image, etc.), a dedicated antiplagiarism screening is conducted prior to the Visual Consistency audit. The LMM first performs a pixel-level identity check; if the target image is found to be an exact pixel-for-pixel duplicate of the reference without any generative modifications, the consistency score is automatically set to 0. Details of the prompt template are provided in Fig. 9.

Aesthetic Quality (AQ): Assesses visual logic, rendering clarity, and realism. It rewards commercial-grade outputs while penalizing structural collapses or AI hallucinations. Details of the prompt template are provided in Fig. 10.

### B. Detailed Qualitative Examples and Model Outputs

To provide a more granular view of the GENIUS benchmark, we present comprehensive qualitative examples for each sub-task. For every data sample, we showcase a complete data instance that includes: (1) the full input content (comprising both context and instruction); (2) the specific evaluation hints utilized for assessing Rule Compliance (RC) and Visual Consistency (VC); and (3) the corresponding generated outputs from six representative models: Nano Banana Pro (Google, 2025b), Nano Banana (Google, 2025a), SeeDream4.5 (Seedream et al., 2025), FLUX.2-dev (Labs, 2025), Bagel (Deng et al., 2025) and ours. These detailed comparisons, which highlight the capabilities and failure modes of different architectures, are illustrated in Fig. 11 and Fig. 12.

### C. Evaluation using Qwen2.5-VL-72B as Judge

To further validate the robustness of our evaluation framework, we employed Qwen2.5-VL-72B (Bai et al., 2025) as the judge model to assess GENIUS benchmark. The results are summarized in Tab. 3.

Table 3. Benchmark Results by Qwen2.5-VL-72B.The Overall column represents the weighted score across all tasks, calculated using a metric ratio of RC:VC:AQ = 6:3.5:0.5. The best and second best performances are highlighted.

Implicit Pattern Induction Ad-hoc Constraint Execution Contextual Knowledge Adaptation

Method Interleaved Overall

Implicit Pattern Symbolic Constraint Visual Constraint Prior-Conflicting Multi-Semantic

RC VC AQ RC VC AQ RC VC AQ RC VC AQ RC VC AQ

Proprietary Models Nano Banana Pro ✓ 48.35 62.21 37.84 89.53 69.93 34.35 82.68 74.14 54.17 88.33 28.22 42.24 89.11 27.27 - 67.27 Nano Banana ✓ 42.88 52.72 41.89 80.81 58.52 35.50 80.72 66.95 45.83 89.83 22.24 28.45 85.64 23.00 - 68.64 GPT-Image ✗ 40.94 53.15 41.27 90.35 59.15 29.77 82.35 42.50 33.33 69.17 27.72 35.34 89.60 17.73 - 58.18 SeeDream 4.0 ✗ 17.74 8.72 1.35 92.44 19.93 5.73 76.47 37.50 8.33 76.67 11.39 8.62 85.64 22.82 - 68.18 SeeDream 4.5 ✗ 44.17 64.79 40.32 89.65 62.11 25.08 80.39 60.83 39.50 89.17 26.80 40.66 84.16 26.18 - 62.27

Open-Source Models Qwen-Image ✗ 25.67 29.81 17.24 79.65 31.33 24.66 74.18 20.83 25.00 59.17 12.58 30.17 78.71 15.82 - 68.18 GLM-Image ✗ 17.45 23.23 17.22 80.23 15.99 21.81 71.76 23.33 18.67 69.17 6.44 15.93 79.21 10.09 - 48.64 FLUX.2-dev ✗ 27.37 33.40 17.57 85.47 33.32 27.36 77.06 30.33 37.70 63.75 12.87 30.38 80.69 19.27 - 63.18 NextStep-1 ✗ 9.90 0.38 20.02 11.98 1.56 15.22 20.54 3.19 19.32 14.11 6.98 20.21 10.08 12.28 - 13.57 Emu3.5-Image ✗ 28.80 43.02 26.35 80.81 33.66 26.72 81.70 20.83 37.50 45.00 10.89 24.14 84.65 20.45 - 62.73 Omini-Gen2 ✗ 21.12 24.42 18.24 81.40 20.59 22.90 82.03 8.33 20.83 62.50 11.39 31.90 82.67 8.18 - 58.64 Bagel ✓ 18.97 14.53 20.27 80.23 16.01 14.89 80.72 16.67 25.00 57.50 6.93 16.38 82.67 22.73 - 60.45

Ours ✓ 23.91 26.45 30.71 70.54 22.01 20.24 75.53 25.27 26.61 46.37 7.89 27.55 75.92 22.35 - 47.24

As illustrated, utilizing Qwen2.5-VL-72B as the evaluator results in a universal decrease in the Overall Scores across all tested models. This suggests that Qwen2.5-VL-72B may impose a stricter standard for rule and visual compliance compared to the primary evaluator. Crucially, despite the shift in absolute scores, the relative performance trends and the ranking order of the models remain largely consistent. This consistency reinforces the reliability of GENIUS benchmark, demonstrating that the observed performance gaps are intrinsic to the models themselves rather than artifacts of a specific judge.

### D. Additional Experiments and Analysis

#### D.1. Ablation on Interleaved Format

In the context of the GENIUS Benchmark, multimodal interleaved data can be presented in various input formats. Since models exhibit varying degrees of compatibility with these formats, we investigate the impact of input structure on performance by defining three distinct paradigms, as illustrated in Fig. 6(a). First, in the Edit Mode, the visual and textual modalities are decoupled. Images are provided separately (e.g., appended at the end or beginning) and are referenced within the text using placeholders like “image i”. Second, the Interleaved Mode corresponds to the standard setting used in our main experiments. Here, images are interleaved with text but are inserted at the boundaries of complete semantic units (typically at the end of a sentence), preserving the syntactic integrity of the text strings. Third, the Fine-Grained Interleaved Mode inserts images precisely at their point of reference, even within a sentence. In this mode, visual tokens act as intrinsic parts of the syntax and can interrupt the textual flow, requiring the model to handle fine-grained multimodal dependencies.

We conducted evaluations on the Nano Banana series and Bagel, as they are among the few models capable of supporting all three formats. The Overall scores are reported in Fig. 6(b). The results indicate that performance trends vary across models, likely due to differences in model architecture. Notably, we observe a significant performance gap between Edit Mode and the two interleaved modes (Interleaved and Fine-Grained), while the disparity between the two interleaved formats is relatively marginal. This variability suggests that current multimodal models possess limited robustness regarding input formatting, exhibiting a strong sensitivity to how visual information is integrated with text.

#### D.2. Discussion on the Composition of Input

To verify the necessity of contextual information for high-fidelity generation, we conducted an ablation study on the Nano Banana Pro model by removing the context component and relying solely on the final instruction. The comparative Rule Compliance scores across different tasks are reported in Fig. 6(c). As observed, removing context leads to a precipitous decline in performance across the board, underscoring its indispensable role. Specifically, the Implicit Pattern Generation,

[Figure 83]

[Figure 84]

Edit Mode

<image 1><image 2><image 3> Context: Math rules image 1 = 1; image 2 = 3; image 3 = 5.

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

Instruction: Following the math rules above, generate a picture of the number of apples represented by pear.

RuleCompliance

Interleaved Mode

Context: Math rules: This image represents 1<image 1>; this image represents 3<image 2>; and this image represents 5<image 3>.

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

Instruction: Following the math rules above, generate a picture of the number of apples represented by pear.

Fine-Grained Interleaved Mode

Context: Math rules <image 1> = 1; <image 2> = 3; <image 3> = 5.

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

Instruction: Following the math rules above, generate a picture of the number of apples represented by pear.

Implicit Pattern

Symbolic Constraint

Visual Constraint

PriorConflicting

MultiSemantic

Nano Banana Pro Nano Banana Bagel

(a) Different Interleaved Format (b) Overall Scores for Different Interleaved Formats (c) Rule Compliance for Different Input Composition

Figure 6. Additional Experiments and Analysis. This figure presents the definition of three input formats (a) and their corresponding performance impact (b), followed by an ablation study assessing the importance of contextual information in instruction following (c).

Symbolic Constraint Generation, and Visual Constraint Generation tasks suffer the most severe degradation. This is anticipated, as these tasks require the model to inductively reason or extract specific visual-textual mappings defined solely within the context; without these definitions, the model lacks the necessary premises to execute the instruction. Similarly, the Prior-Conflicting Generation task exhibits a significant drop, as the model inevitably reverts to its pre-trained priors in the absence of an explicit counter-factual context to override them. Interestingly, the decline in the Multi-Semantic Generation task is less pronounced. This relative stability can be attributed to the task’s inherent difficulty (resulting in a lower baseline performance) and the probability that the model might fortuitously align with the target semantics even without disambiguating context. Nevertheless, the consistent performance gap confirms that context is not merely supplementary but a critical foundation for accurate generation in complex scenarios.

### E. Related Work

Fluid Intelligence Originating from the Cattell-Horn-Carroll (CHC) theory of cognitive abilities (Schneider & McGrew, 2012), general intelligence is structurally divided into Crystallized Intelligence (Gc) and Fluid Intelligence (Gf). While Gc relies on the utilization of accumulated knowledge, Gf represents the innate capacity to solve novel problems through inductive reasoning and dynamic reasoning, independent of prior knowledge, which is often considered as more indicative of general intelligence (Jaeggi et al., 2008; Chollet, 2019; Barak & Loewenstein, 2024). In the field of understanding, evaluating Gf has traditionally focused on logical reasoning and abstract pattern completion. Prominent benchmarks such as the ARC Bench (Chollet, 2019) assess a model’s ability to induce rules from few-shot examples and generalize to new scenarios. However, these evaluations are predominantly discriminative or symbolic, targeting problem-solving in restricted domains (e.g., grid worlds). In the context of Unified Multimodal Models (UMMs), current assessments remain largely confined to Gc, testing the model’s capability on static world knowledge.

Unified Multimodal Models (UMMs) Recent years have witnessed a paradigm shift from modular composition towards native fusion in multimodal models. Early approaches primarily bridged pre-trained Large Language Models (Qin et al., 2024; Esser et al., 2024; Zhao et al., 2024; Li et al., 2025b; Zhang et al., 2025) with diffusion decoders to enable visual synthesis capabilities (Koh et al., 2023). However, the latest wave of UMMs, represented by Chameleon (Team, 2024), Show-o (Xie et al., 2024; Guo* et al., 2025) and Emu Series (Sun et al., 2023; 2024; Wang et al., 2024), marks a fundamental departure by discretizing visual signals into discrete tokens. Janus (Wu et al., 2025a) and its improvements (Guo et al., 2025; Jiang* et al., 2025) claims that understanding and generation require distinct information, employing different tokenizers for each task. Among these architectures, Bagel (Deng et al., 2025) and its improvements (Xie et al., 2025a; Jin et al., 2025) have demonstrated remarkable versatility in both understanding and generation tasks, becoming one of the most representative open-sourced models. Despite these architectural breakthroughs, current research predominantly evaluates these models on task-specific benchmarks (e.g., VQA or standard T2I generation), leaving their intrinsic Generative Fluid Intelligence (i.e., the capacity to reason and adaptively generate under novel, ad-hoc constraints) largely unexplored.

Generative Evaluation of UMMs With the rapid progress of UMMs, numerous benchmarks (Ghosh et al., 2023; Zhao

et al., 2025; Wei et al., 2025; Li et al., 2025d; Chow et al., 2025) have been proposed to assess their capabilities, yet most remain confined to traditional evaluation paradigms. Early benchmarks like GenEval (Ghosh et al., 2023), WISE (Niu et al., 2025), and DPG-Bench (Hu et al., 2024) primarily focus on single-image generation tasks, assessing static world knowledge or basic text-to-image alignment without involving complex, interleaved contexts. OpenIng (Zhou et al., 2025) focuses on in-context visual generation, while it primarily targeting interleaved text-and-image generation and Crystallized Intelligence. While more recent suites such as MME-Unify (Xie et al., 2025b), RealUnify (Shi et al., 2025), and ROVER (Liang et al., 2025) have begun to incorporate multi-image inputs, they predominantly target Crystallized Intelligence, evaluating the model’s ability to recall pre-trained information rather than adapting novel rules. Crucially, none of the existing benchmarks systematically evaluate Generative Fluid Intelligence (GFI). As shown in Table 1, current methods lack comprehensive coverage across key GFI dimensions, including Implicit Pattern Induction, Explicit Constraint Execution, and Contextual Knowledge Adaptation. Furthermore, many rely heavily on synthetic data or purely LLM-as-Judge that fail to capture failure cases. In contrast, GENIUS fills this critical void by being the first benchmark to feature a fully multimodal interleaved context, purely manually curated annotations, and a hybrid evaluation protocol to quantify FI in generative scenarios.

### F. Details of Method

#### F.1. Prompt Template for Keyword Generation

To extract task-critical visual cues, we employ following prompt to guide Bagel in identifying key regions within the context images. The template is shown in Fig. 7. The generated keywords are subsequently used to compute the relevance map.

#### F.2. Mathematical Formulation of Attention Modulation

In this section, we detail the implementation of the Bias Injection stage. Our modulation strategy is mathematically inspired by (Li et al., 2025e;c), adapted to our keyword-based relevance scoring.

The modulation is applied selectively to a subset of decoder layers Lselected and generation steps Tselected. For a targeted head h in layer l ∈ Lselected at step t ∈ Tselected, let Al,h ∈ RN×N denote the original attention logits (before Softmax). Let S ∈ RN be the relevance score vector computed in the Relevance Mapping stage.

To enforce the model’s focus on critical signals, we inject a dynamic bias term into the attention mechanism. The modulated attention logits Aˆ l,h are computed as follows:

##### Aˆ l,h(i,j) = Al,h(i,j) + λ · F(Sj) (11)

where i denotes the query token index, j denotes the key token index, and λ is a scalar hyperparameter controlling the modulation intensity. The function F(·) maps the raw relevance scores to a bipolar bias distribution:

Sj − µS σS + ϵ

F(Sj) =

(12)

Here, µS and σS are the mean and standard deviation of the relevance scores across the current context window. The final attention weights are obtained via the standard Softmax operation:

Attention(Q,K,V ) = softmax

A ˆ √

d

V (13)

This formulation ensures that the gradient norm contribution from noise tokens is effectively dampened by the exponential suppression of the Softmax function.

### G. Theorem Part

#### G.1. Exact Definition of A

Let xt denote the noisy intermediate variable at a certain time step. Let C1 represent the context and instruction (text modality), and C2 represent the image modality. Then for t + 1-th step, we have:

xt+1 = F(u1,u2,gt) (14) cause Bagel uses MoE architecture, the intermediate variables are defined as:

- u1 = Und Encoder(C1 ∥ C2), (15)

- u2 = Gen Encoder(C2), (16) gt = Gen Encoder(xt). (17)

For the (l + 1)-th Decoder layer, Bagel employs a Pre-Layer Normalization (Pre-LN) structure. Let u = (u1||u2) where ∥ denotes matrix concatenation operation. We then obtain the detailed update rule of the decoder layer:

g(t,l+1) = A(u(l),g(t,l)) + f Up N(A(u(l),g(t,l)) + b = L(Upl),b(u(l),g(t,l)) (18)

where the initial bias is set as binitial = 0, RMS denotes the Root Mean Square operation, and Up denotes the Up layer in the decoder block.

The core attention function A(u,g) is formulated as:

A(u,g) = MoE attn((U1 ∥ U2),G) + g (19)

- U1 = Und(RMSNorm(u1)) (20)
- U2 = Gen(RMSNorm(u2)) (21) G = Gen(RMSNorm(g)) (22)

where

Gquery (Ukey ∥ Gkey)⊤ √dattn × (Uvalue ∥ Gvalue), (23)

MoE attn(U,G) = Softmax

where U = (Uquery,Ukey,Uvalue) and G = (Gquery,Gkey,Gvalue) denote the query/key/value components of U and G respectively, dattn is the dimension of attention heads. Und (Understanding) and Gen (Generation) denote the Q, K, and V projection layers for different experts in the MoE architecture.

#### G.2. Proof of Thm. 4.1

Our goal is to prove:

LUp+∆Up, b+∆b(u′,g) = LUp,b(u,g) (24) According to the definition of L, we have:

LUp+∆Up,b+∆b(u′,g) = A(u′,g) + f((Up + ∆Up)( A(u′,g) RMS(A(u′,g))

)) + b + ∆b (25) Cause:

Up(δA)N(A(u′,g))⊤ ∥N(A(u′,g))∥2

,∆b = A(u,g) − A(u′,g) (26) We proceed to expand Equation (25):

∆Up =

LUp+∆Up,b+∆b(u′,g) = A(u,g) + f(UpN(A(u′,g)) + Up(δA)) + b (27) = A(u,g) + f(UpA(u,g))) + b (28) = LUp,b(u,g) (29)

where

δA = N(A(u,g)) − N(A(u′,g)) (30) Thus, we complete the theorem’s proof.

- G.3. Proof of Thm. 4.2 Our goal is to prove the following conclusion:

##### Upi+1 = Upi − h∇UpLi(Upi), bi+1 = bi − ∇b tr δi⊤bi

For Up, we first expand the update rule directly and obtain:

(31)

Upi+1 − Upi = ∆Upi+1 − ∆Upi

Up(δAi)(N(A(g)))⊤ ∥N(A(g))∥2

Up(δAi+1)(N(A(g)))⊤ ∥N(A(g))∥2

−

=

Up(δAi+1 − δAi)(N(A(g)))⊤ ∥N(A(g))∥2

=

According to the main text, if we define:

 

h = ∥N(A1(g))∥2 ∆i = Up(δAi − δAi+1)(N(A(g)))⊤ Li(Up) = ∆⊤i Up



and notice the following trivial property:

(32)

(33)

∇Up trace ∆⊤i Up = ∆i (34) then we can conclude that:

Upi+1 = Upi − h∇UpLi(Upi) (35) For b, we have:

##### bi+1 − bi = ∆bi+1 − ∆bi = A(u(i+1),g) − A(u(i),g) (36) so according to property Equation (34):

bi+1 = bi − ∇b tr δi⊤bi (37) Thus, we complete the theorem’s proof.

#### Prompt Template for Keyword Generation

You are an EXPERT IMAGE GENERATION PLANNER. Your goal is to parse multimodal instructions and map every provided image to its specific role in the generation process.

###### ◎ Task:

Analyze the user’s Instruction and the list of image num provided images. Determine precisely what information needs to be extracted or retained from each image.

###### Focus Definition Rules:

1. TARGET CANVAS / BASE IMAGE Target Canvas / Base Image: If an image serves as the foundation to be edited, reshaped, or modified (e.g., "change the background of this image", "add a hat to this person"), the value must be "all". Feature Extraction: If only a specific part is needed (e.g., "swap face", "use this shirt", "holding this object"), output the specific noun (e.g., "face", "shirt", "cup"). Style/Attribute Reference: If the image provides abstract attributes (e.g., "use this lighting", "copy this art style", "follow this pose"), output the attribute name (e.g., "lighting", "art style", "pose"). Irrelevant: If an image is mentioned but contributes no visual content to the result, output empty string "".

###### Output Format:

Return ONLY a strictly valid JSON object: { "<image X>": "focus string" }

###### Few-Shot Examples:

- Example 1: Input: "Transfer the style of <image 1> to the car in <image 2>, but make sure the background matches <image 3>." Output: {

- "<image 1>": "art style",
- "<image 2>": "car",
- "<image 3>": "background" }

- Example 2: Input: "Take <image 1> and remove the person from it." Output: { "<image 1>": "all" }

###### Current Input Context: Images Count: {image num} Instruction:

"""

{content} """

###### Figure 7. Prompt Template for Keyword Generation.

#### Rule Compliance Evaluation

Your task is to act as a PRECISION VISUAL AUDITOR in Zero-Tolerance Fidelity Mode.

###### ◎ Mission Statement:

Rigorously evaluate the alignment between the HINT and the MODEL OUTPUT IMAGE. You must prioritize Technical Precision over Perceptual Plasticity. This mode is designed for high-stakes instruction following where \close enough" is considered a failure.

Scoring Hierarchy:

- 1. SCORE 2 [PERFECT EXECUTION]

Standard: The image is a flawless visual manifestation of the HINT. Every explicit and implicit constraint must be met with 100% accuracy. Identity and Nouns: All requested subjects are present, anatomically/structurally correct, and positioned exactly as described. Adjective/Attribute Fidelity: Every single descriptor (color, texture, material, style, state, quantity) is rendered without deviation.

Strict Detail Check: If the hint specifies \three buttons" and there are two or four, it is NOT a 2. If the hint says \crimson" and the output is \bright red," it is NOT a 2.

Rule: Award only if there is ZERO discrepancy between text and pixels.

- 2. SCORE 1 [PARTIAL COMPLETION / ACCURACY DRIFT]

Standard: The core subject/action is present, but the execution fails on specific details, modifiers, or secondary constraints. Minor Omissions:A secondary adjective is ignored (e.g., \vintage" style is missing, but the object is there). Count/Scale Errors: Wrong number of objects or incorrect relative sizing. Color Drift: The color is in the correct family but lacks the specific shade or intensity requested. Rule: Award 1 if the viewer can tell what was intended, but the \fine print" of the instruction was neglected.

- 3. SCORE 0 [FAILURE / GROSS ERROR] Standard: Total failure to execute the primary intent.

Subject Error: The wrong object was added, the target was removed, or the primary subject is unrecognizable. Non-Action: No change was made to the image despite a request for modification. Semantic Inversion/Incoherence: The model did the opposite of the hint or produced a visual hallucination/glitch. Rule: Award 0 if the primary goal of the HINT is unfulfilled.

###### Evaluation Steps:

- 1. SCAN: List every Noun, Adjective, Count, and Spatial relation in the HINT.
- 2. VERIFY PRIMARY: Is the core action/subject present? (If No → 0).
- 3. VERIFY EXHAUSTIVE: Check every item from Step 1. Any error → 1.
- 4. FINAL SCORE: Award 2 ONLY if Step 3 yields a perfect match. Evaluation Data:

HINT: {{hint}} TARGET: {{output image tag}}

###### Output Format:

Return ONLY the following line: Rule Compliance: X

###### Figure 8. Prompt Template for Rule Compliance Evaluation.

#### Visual Consistency Evaluation

Your task is to act as a VISUAL FIDELITY ANALYST in Strict Evaluation Mode.

###### ◎ Mission Statement:

Your mission is to evaluate the consistency between the REFERENCE IMAGE and the TARGET based on the HINT. Your goal is to distinguish between \Great Work" (2), \Acceptable Effort" (1), and \Total Failure" (0).

Scoring Hierarchy:

- 1. SCORE 2 [HIGH FIDELITY (SUCCESSFUL)]

Standard: The TARGET is a high-quality implementation of the HINT. The core subject remains stable and the image feels professional. Strong Identity: The main subject (person, object, or scene) is clearly the same as in the Reference. Smooth Transformation: The changes requested by the HINT are integrated. Minor Tolerance: Small shifts in color, slight facial softening, or minor background variations are PERFECTLY ACCEPTABLE. Rule: If the image is good, and follows the HINT, give it a 2.

- 2. SCORE 1 [RECOGNIZABLE DERIVATION]

Standard: The TARGET is clearly related to the Reference, even if the execution is imperfect. This is the catch-all category for images that \get the idea right" but lose some detail. Recognizable Link: You can still tell it is based on the same subject or concept, even if the face looks a bit different or background have shifted. Moderate Drift: The HINT was attempted, but the model may have simplified original details or introduced AI blurring/messiness. High Tolerance: Even if the image has lost some \Visual DNA," as long as it isn’t a completely different subject, it stays in this category. Rule: If the model tried to follow the HINT and the result is \okay" or \recognizable," award a 1.

- 3. SCORE 0 [TOTAL FAILURE]

Standard: Total failure to execute the primary intent or maintain subject connection.

Subject Swap: The model generated a completely different person or object. Ignored Instruction: The model provided a generic image that ignores both the Reference and the HINT entirely. Broken Output: The image is a corrupted, unidentifiable mess of pixels. Rule: Do NOT award a 0 if there is any link to the Reference Image.

###### Evaluation Steps:

- 1. RELATION: Is the Target even remotely related to the Reference subject? (If No → Score 0).
- 2. QUALITY & FIDELITY: Does the Target look stable, professional, and closely follow the HINT without distracting errors? (If Yes → Score 2).
- 3. DRIFT ASSESSMENT: For everything else in between (minor drift, detail loss, background shifts, but same subject) → Score 1.

###### Evaluation Data:

REFERENCE IMAGE: {{reference image}} HINT: {{hint}} TARGET: {{output image tag}}

###### Output Format:

Return ONLY the following line: Visual Consistency: X

###### Figure 9. Prompt Template for Visual Consistency Evaluation.

#### Aesthetic Quality Evaluation

Your task is to act as a PROFESSIONAL VISUAL ARTS CURATOR in High-Visual Harmony Mode.

◎ Mission Statement: Your mission is to categorize images into Masterpiece Level (2), Standard Work

(1), and Technical Failure (0). You must reward images that achieve high visual harmony and logical consistency, prioritizing Visual Cohesion and professional merit.

Scoring Hierarchy:

- 1. SCORE 2 [EXCEPTIONAL / MASTERPIECE LEVEL]

Standard: High-end, commercial quality. The image is indistinguishable from professional work. Structural Logic: All objects and characters follow the laws of physics and anatomy. Rendering Clarity: Textures look intentional and sharp; lighting creates a convincing sense of depth. Visual Appeal: The overall composition is professional and free of distracting AI hallucinations. Tolerance Clause: Award a 2 even if there is a tiny, non-distracting flaw, provided the overall impact is professional.

- 2. SCORE 1 [STANDARD / THE GOOD EFFORT ZONE]

Standard: The DEFAULT category. Visually acceptable and logically sound but contains visible "AI-isms." Visible Flaws: Slightly soft hands, plastic skin, or shadows that do not perfectly align with the light source. Minor Perspective Issues: Background elements that are slightly tilted or out of scale. Rule: Use this as the safety score for any image that is "pretty good" but not perfect.

- 3. SCORE 0 [FAILED / TECHNICAL FAILURE] Standard: Images that are visually nonsensical or structurally broken. Logical Collapse: Extra limbs that are gross and distracting, or faces that have lost basic human structure. Extreme Noise: Grain or artifacts so heavy they obscure the main subject. Rule: Only award 0 if the image is unusable. If the parts are mostly in the right place, do NOT give a 0.

Evaluation Steps:

- 1. SCAN: Identify the visual logic, realism, and presence of any AI-generated artifacts.
- 2. VERIFY LOGIC: Does the image maintain basic structural integrity? (If No → 0).
- 3. ASSESS QUALITY: Check for professional rendering, sharp textures, and lighting depth.
- 4. FINAL SCORE: Award 2 for commercial quality, 1 for standard AI output, and 0 for unusable failure.

###### Evaluation Data:

METRIC: Aesthetic Quality (Visual Logic and Realism) TARGET: {{output image tag}}

###### Output Format:

Return ONLY the following line: Aesthetic Quality: X

###### Figure 10. Prompt Template for Aesthetic Quality Evaluation.

###### Implicit Pattern Induction

Implicit Pattern Generation

Context: I prefer the palette in this image.<image 1> \nI dislike the palette in this image. <image 2> Instruction: Please fill this image <image 3> with my preferred palette, and show the characteristics of the palette.

Nano Banana Pro Nano Banana SeeDream4.5 FLUX.2-dev Bagel Ours

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

RC Hint: The generated image should be: a picture of overlapping mountains in an autumn wilderness, dis-playing a brown color pattern.

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

VC Hint: <image 3> The lines and basic shape should be preserved, and the entire background should be preserved.

RC: 2 VC: 1 AQ: 2 RC: 1 VC: 1 AQ: 2 RC: 1 VC: 1 AQ: 2 RC: 1 VC: 0 AQ: 1 RC: 0 VC: 1 AQ: 1 RC: 1 VC: 1 AQ: 1

Context: Evan really likes the accessories in this image.<image 1>\nThe person on the right of this image is Evan.<image 2> Instruction: Create an image that Evan is dressed in his favorite style.

- VC Hint: <image 2> The appearance of the boy in this image should be preserved.

RC Hint: The generated image should be a little boy wearing a knitted hat.

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

|[Figure 109]|
|---|

|[Figure 110]|
|---|

- RC: 2 VC: 2 AQ: 1 RC: 1 VC: 2 AQ: 2 RC: 2 VC: 2 AQ: 2 RC: 0 VC: 0 AQ: 2 RC: 0 VC: 0 AQ: 2 RC: 0 VC: 1 AQ: 2

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

- RC: 2 VC: 2 AQ: 2 RC: 2 VC: 0 AQ: 2 RC: 2 VC: 2 AQ: 2 RC: 1 VC: 0 AQ: 1

|[Figure 116]|
|---|

RC: 1 VC: 1 AQ: 2 RC: 1 VC: 1 AQ: 2

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

RC: 2 AQ: 2 RC: 2 AQ: 2 RC: 2 AQ: 2 RC: 0 AQ: 2 RC: 1 AQ: 0 RC: 1 AQ: 0

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

- RC: 1 VC: 2 AQ: 2 RC: 1 VC: 0 AQ: 2 RC: 2 VC: 1 AQ: 2 RC: 0 VC: 0 AQ: 2 RC: 0 VC: 0 AQ: 2 RC: 1 VC: 1 AQ: 2

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

- RC: 2 VC: 1 AQ: 2 RC: 1 VC: 1 AQ: 1 RC: 1 VC: 1 AQ: 2 RC: 1 VC: 0 AQ: 2 RC: 1 VC: 1 AQ: 1 RC: 1 VC: 1 AQ: 1

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

|[Figure 139]|
|---|

|[Figure 140]|
|---|

- RC: 2 VC: 2 AQ: 2 RC: 1 VC: 1 AQ: 2 RC: 2 VC: 1 AQ: 2 RC: 1 VC: 1 AQ: 1 RC: 0 VC: 0 AQ: 2 RC: 1 VC: 1 AQ: 1

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

|[Figure 144]|
|---|

|[Figure 145]|
|---|

|[Figure 146]|
|---|

RC: 2 VC: 2 AQ: 2 RC: 0 VC: 1 AQ: 2 RC: 0 VC: 1 AQ: 2 RC: 0 VC: 1 AQ: 2 RC: 0 VC: 0 AQ: 2 RC: 0 VC: 1 AQ: 2

|[Figure 147]|
|---|

|[Figure 148]|
|---|

|[Figure 149]|
|---|

|[Figure 150]|
|---|

|[Figure 151]|
|---|

|[Figure 152]|
|---|

- RC: 2 VC: 2 AQ: 2 RC: 2 VC: 2 AQ: 2 RC: 2 VC: 1 AQ: 2 RC: 1 VC: 1 AQ: 2

RC: 0 VC: 0 AQ: 1

RC: 0 VC: 2 AQ: 1

Context: The artist is fascinated by this image <image 1> and finds this image lack of aesthetic appeal.<image 2>

Instruction: Adapt this image into the specific style that the artist loves.<image 3>

RC Hint: The generated image should consist of the sun, trees, mountains, and rivers; the overall style is Van Gogh's abstract art.

- VC Hint: <image 3> All items displayed should remain unchanged, only the painting style should be altered.

|[Figure 153]|
|---|

|[Figure 154]|
|---|

|[Figure 155]|
|---|

Context: I like the inner object but dislike the outer object in this image <image 1>. I dislike the inner object but like the outer object in this image <image 2>.

Instruction: Create an image that both objects are ones I like. RC Hint: The generated image should show a small bird inside a basket.

|[Figure 156]|
|---|

|[Figure 157]|
|---|

Context: I like the pattern on the vase in this image <image 1>, but I don't like the pattern on the scroll in this image <image 2>.

Instruction: Convert <image 3> according to my aesthetic preference.

RC Hint: The generated image should be: a classical Chinese ink painting scroll depicting mountains and rivers.

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

VC Hint: <image 3> The scroll's style should be preserved, and the entire background should be retained.

Ad-hoc Constraint Execution Symbolic Constraint Generation

Context: The transition from this image to that image is operation <f>.

- <image 1><image 2>

|[Figure 161]|
|---|

|[Figure 162]|
|---|

|[Figure 163]|
|---|

Context: This visual pattern indicates that the trumpet in the image is 'Quiet/Calm'<image 1>,while this visual pattern indicates that the trumpet is 'Loud/Noisy'.<image 2>

Instruction: Based on the relationship between visual patterns and perception described above, make the dog in this image seems noisy and loud.<image 3>

VC Hint: <image 3> The appearance of the dog in the image should remain unchanged.

- <image 2> The yellow explode lines in the image should remain unchanged.

Instruction: Please generate the appearance of this image after performing operation <f>.<image 3>

RC Hint: The generated image should depict a neatly trimmed green shrub extending towards the distance.

|[Figure 164]|
|---|

|[Figure 165]|
|---|

|[Figure 166]|
|---|

VC Hint: <image 3> The background and environment in the image should not be altered. The overall position of the shrub belt should remain unchanged.

RC Hint: The generated image should depict a dog surrounded by explosive yellow line patterns.

Context: Layout style A is in <image 1>. Layout style B is in <image 2>. Instruction: Shift objects in this image to layout style B.<image 3> RC Hint: The generated image should depict a cozy vintage study scene. On a wooden desk, a silver flute stands vertically on the left side.

VC Hint: <image 3> The style and relative size of the flute and music stand should remain unchanged, and the background of the image should also stay the same.

|[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

Context: Feature C is the pose in <image 1>. Feature B is the material of <image 2>. Instruction: Please reshape this picture using Feature C and Feature B.<image 3>

RC Hint: The generated image should depict a Mickey Mouse snow sculpture with its hands extended.

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

VC Hint: <image 3> The character in the image should remain unchanged.

Context: This is Dex, a hacker who always wears cat-ear headphones when operating a computer. <image 1> His desk is covered with various encryption and decryption devices, his cat-ear headphones can block out all external interference, and he can quickly crack complex program codes.

|[Figure 173]|
|---|

|[Figure 174]|
|---|

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

|[Figure 178]|
|---|

Instruction: Please generate a scene of Dex in a dimly lit room, wearing his headphones .... RC Hint: The generated image should depict a scene of a hacker in a dimly lit room wearing his cat-ear headphones ... VC Hint: <image 1> The appearance of the hacker should remain unchanged.

|[Figure 179]|
|---|

RC: 2 VC: 2 AQ: 2 RC: 2 VC: 2 AQ: 2 RC: 1 VC: 2 AQ: 2 RC: 1 VC: 2 AQ: 2 RC: 1 VC: 0 AQ: 2 RC: 1 VC: 1 AQ: 2

###### Figure 11. Detailed Qualitative Examples and Model Outputs. (1/2)

###### Ad-hoc Constraint Execution

Visual Constraint Generation

Nano Banana Pro Nano Banana SeeDream4.5 FLUX.2-dev Bagel Ours

Context: Math rules: This image represents 1<image 1>; this image represents 3<image 2>; and this image represents 5<image 3>. Instruction: Following the math rules above, generate a picture of the number of apples represented by pear. RC Hint: The generated image should be three apples.

|[Figure 180]|
|---|

|[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

|[Figure 188]|
|---|

RC: 2 AQ: 2

RC: 1 AQ: 2 RC: 2 AQ: 2 RC: 0 AQ: 1 RC: 0 AQ: 1 RC: 0 AQ: 1

Context: Math rules: \nCondition A (this icon):<image 1> this image equals 3<image 2>;\nCondition B (this icon):<image 3> this image equals 1<image 2>. Instruction: Current condition is this icon<image 1>. Under this condition, generate a picture of the number of cats represented by this image<image 2>. RC Hint: The generated image should have three cats.

|[Figure 189]|
|---|

|[Figure 190]|
|---|

|[Figure 191]|
|---|

|[Figure 192]|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

|[Figure 195]|
|---|

|[Figure 196]|
|---|

|[Figure 197]|
|---|

RC: 2 AQ: 2

RC: 1 AQ: 2 RC: 2 AQ: 1 RC: 0 AQ: 1 RC: 0 AQ: 2 RC: 1 AQ: 1

Contextual Knowledge Adaptation Multi-Semantic Generation

Context: This is a picture of top banana <image 1>. Instruction: Generate an image of a bookworm following the same semantic interpretation (literal vs. figurative) as the context image.

|[Figure 198]|
|---|

|[Figure 199]|
|---|

|[Figure 200]|
|---|

|[Figure 201]|
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

RC Hint: The generated image should show an actual worm crawling between the pages of a book. There should be no person completely absorbed and focused deeply on reading a book in the image. (All satisfied: 2 score; Satisfied one sentence: 1 score; Not satisfied: 0 score)

|[Figure 204]|
|---|

RC: 2 AQ: 2 RC: 1 AQ: 2 RC: 1 AQ: 2 RC: 0 AQ: 2 RC: 0 AQ: 1 RC: 1 AQ: 1

Context: This is a visual description of break a leg <image 1>. Instruction: Generate an image of a couch potato following the same semantic interpretation (literal vs. figurative) as the context image.

|[Figure 205]|
|---|

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

|[Figure 209]|
|---|

|[Figure 210]|
|---|

|[Figure 211]|
|---|

RC Hint: The generated image should show a large raw potato sitting comfortably on a sofa. There should be no person in the image. (All satisfied: 2 score; Satisfied one sentence: 1 score; Not satisfied: 0 score)

RC: 1 AQ: 2 RC: 1 AQ: 1

RC: 0 AQ: 1 RC: 0 AQ: 0 RC: 0 AQ: 1

RC: 2 AQ: 2

Context: This is a visual description of on pins and needles <image 1>. Instruction: Generate an image of a loose cannon following the same semantic interpretation (literal vs. figurative) as the context image.

|[Figure 212]|
|---|

|[Figure 213]|
|---|

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

RC Hint: The generated image should show a realistic old-fashioned cannon rolling uncontrollably because it is not tied down. There should not be any person acting unpredictably in the image. (All satisfied: 2 score; Satisfied one sentence: 1 score; Not satisfied: 0 score)

|[Figure 218]|
|---|

- RC: 1 AQ: 2 RC: 1 AQ: 2
- RC: 2 AQ: 2 RC: 2 AQ: 2 RC: 1 AQ: 2 RC: 0 AQ: 1 RC: 1 AQ: 1 RC: 1 AQ: 1

RC: 0 AQ: 1 RC: 1 AQ: 1 RC: 1 AQ: 1 RC: 1 AQ: 2

Contextual Knowledge Adaptation Prior-Conﬂicting Generation

Context: In this world, living things grow according to their own rules. A young carp's size is like this image<image 1>, while an elderly carp's size is like this image<image 2>.

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

Instruction: The goldfish in this world grows like a carp. Please generate the appearance of a young goldfish in this world. RC Hint: The generated image should be a goldfish about the size of a seal.

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

Context: On this planet, gravity is determined by color.<image 1> <image 2>

Instruction: Please generate the appearance of the yellow pear on this planet. RC Hint: The generated image should be a pear in the air.

|[Figure 233]|
|---|

|[Figure 234]|
|---|

RC: 2 AQ: 2

RC: 2 AQ: 2 RC: 2 AQ: 2 RC: 2 AQ: 1 RC: 0 AQ: 2 RC: 1 AQ: 2

Context: In this world, the relationship between insects and insectivorous birds is shown in the following figures.<image 1><image 2> Instruction: Please draw an earthworm and a chickenappearance of a young goldfish in this world.

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

RC Hint: The generated image should depict a strong, fierce earthworm bullying a weak chicken.

RC: 2 AQ: 2

RC: 0 AQ: 2 RC: 0 AQ: 2 RC: 0 AQ: 2 RC: 0 AQ: 1 RC: 0 AQ: 2

Context: On this planet, objects do not wear out over time. This is a newly bought light bulb: <image 1>, and this is the bulb after being turned on for a year: <image 2>.

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

Instruction: Please draw the appearance of this light bulb after being turned on for a year.<image 3>

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

RC Hint: The generated image should be a brand new light bulb.

VC Hint: <image 3> The style of light bulb in this image should be maintained.

RC: 2 VC: 2 AQ: 2

RC: 1 VC: 2 AQ: 2 RC: 1 VC: 2 AQ: 2 RC: 1 VC: 0 AQ: 2 RC: 0 VC: 0 AQ: 2 RC: 0 VC: 2 AQ: 2

Context: In this world, the natural environment is affected by the positive and negative emotions of humans.<image 1><image 2> Instruction: Please generate the appearance of the weather outside the restaurant window.<image 3> RC Hint: The generated image should be a background with fog. VC Hint: <image 3> The foreground

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

|[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|
|---|

people of this image should be preserved.

RC: 0 VC: 2 AQ: 1 RC: 0 VC: 0 AQ: 2 RC: 0 VC: 1 AQ: 1 RC: 0 VC: 0 AQ: 1 RC: 0 VC: 0 AQ: 1 RC: 0 VC: 1 AQ: 1

###### Figure 12. Detailed Qualitative Examples and Model Outputs. (2/2)

