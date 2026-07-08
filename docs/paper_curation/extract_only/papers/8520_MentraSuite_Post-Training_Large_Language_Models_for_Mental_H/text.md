# arXiv:2512.09636v2[cs.CL]16Dec2025

## MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment

Mengxi Xiao

School of Artificial Intelligence, Wuhan University; Center for Language and Information Research, Wuhan University China

Enze Zhang

School of Artificial Intelligence, Wuhan University; Center for Language and Information Research, Wuhan University China

Kailai Yang

The University of Manchester United Kingdom

Ziyan Kuang

Center for Language and Information Research, Wuhan University China

Pengde Zhao

School of Computer Science, Wuhan University China

Zhiwei Liu

The University of Manchester United Kingdom

Weiguang Han

School of Computer Science, Wuhan University China

Shu Liao

Center for Language and Information Research, Wuhan University China

Lianting Huang

Mount Holyoke College United States

Jinpeng Hu

Hefei University of Technology China

Min Peng

School of Artificial Intelligence, Wuhan University; Center for Language and Information Research, Wuhan University China pengm@whu.edu.cn

Qianqian Xie∗

School of Artificial Intelligence, Wuhan University; Center for Language and Information Research, Wuhan University China xieq@whu.edu.cn

Sophia Ananiadou

The University of Manchester United Kingdom sophia.ananiadou@manchester.ac.uk

#### Abstract

Mental health disorders affect hundreds of millions globally, and the Web now serves as a primary medium for accessing support, information, and assessment. Large language models (LLMs) offer scalable and accessible assistance, yet their deployment in mentalhealth settings remains risky when their reasoning is incomplete,

∗Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

inconsistent, or ungrounded. Existing psychological LLMs emphasize emotional understanding or knowledge recall but overlook the step-wise, clinically aligned reasoning required for appraisal, diagnosis, intervention planning, abstraction, and verification. To address these issues, we introduce MentraSuite, a unified framework for advancing reliable mental-health reasoning. We propose MentraBench, a comprehensive benchmark spanning five core reasoning aspects, six tasks, and 13 datasets, evaluating both task performance and reasoning quality across five dimensions: conciseness, coherence, hallucination avoidance, task understanding, and internal consistency. We further present Mindora, a post-trained model optimized through a hybrid SFT–RL framework with an inconsistency-detection reward to enforce faithful and coherent reasoning. To support training, we construct high-quality trajectories using a novel reasoning trajectory generation strategy, that

strategicallyfiltersdifficultsamplesandappliesastructured,consistencyoriented rewriting process to produce concise, readable, and wellbalanced trajectories. Across 20 evaluated LLMs, Mindora achieves the highest average performance on MentraBench and shows remarkable performances in reasoning reliability, demonstrating its effectiveness for complex mental-health scenarios.

reasoningthroughalarge-scalesynthesispipelineandhybridGRPOSFT training. Psy-Interpreter [8] enhances implicit mental-state inference using expert-annotated scenarios and a trajectory-aware reinforcement learning framework that imitates clinician-like reasoning.PsychCounsel-Bench[40]assesseswhetherLLMsmeetcounseling knowledge standards, finding that only frontier models surpass certification-level performance [31].

• Computing methodologies → Causal reasoning and diagnostics; Natural language generation.

Despite these advances, current methods still have important limitations in both method design and evaluation as showin in Table 1. Most existing approaches focus on emotional understanding, knowledge tests, or supervised reasoning tailored to a narrow set of tasks, without systematically modeling core stages of clinical reasoning. Yet effective mental-health support requires reasoning across several interconnected processes: appraisal (recognizing maladaptive thought patterns), diagnosis (identifying likely conditions), intervention (selecting appropriate therapeutic strategies), abstraction (synthesizing structured evidence), and verification (detecting inaccurate or misleading mental-health information). Furthermore, existing methods focus primarily on task accuracy while overlooking the quality and reliability of the reasoning process. Reliable mental-health support requires LLMs to produce transparent, coherent, and context-grounded reasoning across several key dimensions: reasoning conciseness (avoiding unnecessary complexity or repetition), logical coherence (providing stepwise, case-specific justification), hallucination avoidance (not introducing unsupported facts), task understanding (following the intended objective without drift), and internal consistency (maintaining non-contradictory reasoning throughout). Addressing these aspects is essential for assessing and developing LLMs that can perform the step-wise, integrative reasoning that underlies appraisal, diagnosis, intervention, abstraction, and verification in real mental-health practice.

CCS Concepts

#### Keywords

Mental Health Reasoning, Large Language Models, Post-training

ACM Reference Format:

Mengxi Xiao, Kailai Yang, Pengde Zhao, Enze Zhang, Ziyan Kuang, Zhiwei Liu, Weiguang Han, Shu Liao, Lianting Huang, Jinpeng Hu, Min Peng, Qianqian Xie, and Sophia Ananiadou. 2018. MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym ’XX). ACM, New York, NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX

#### 1 Introduction

Mentalhealthdisordersaffecthundredsofmillionsofpeopleworldwide and remain a leading contributor to disability, social inequality, and unmet clinical needs [9, 21, 33]. As individuals increasingly turn to online platforms for information, support, and selfassessment, the Web has become a critical medium for expanding access to mental health care and advancing social good. Artificial intelligence (AI) plays a growing role in this shift: assisting individuals understand their conditions [30, 36, 39], supporting counselors in developing treatment strategies [23, 34], and helping clinicians in decision-making [35]. Recently, large language models (LLMs) [11, 18] have been rapidly adopted across web-based mental health applications, due to their strong linguistic capabilities, broad world knowledge, and ability to engage in natural, empathetic dialogue. Their potential impact is considerable: LLMs can provide scalable, always-available guidance and reach populations that traditional services often fail to serve.

Tofillthisgap,wepresentMentraSuite1,aunifiedsuiteofbenchmarks, datasets, and models for advancing reliable mental-health reasoning. We introduce MentraBench, a comprehensive benchmark that evaluates five essential aspects of clinical and counseling cognition: appraisal, diagnosis, intervention, abstraction, and verification. MentraBench spans six tasks and 13 datasets, built by refining existing resources and constructing new ones. Unlike prior benchmarks focused primarily on accuracy, MentraBench emphasizes the quality of reasoning trajectories, assessing five key dimensions: reasoning conciseness, logical coherence, hallucination avoidance, task understanding, and internal consistency. We then introduce Mindora, a post-trained model optimized for diverse mental-health reasoning tasks and more reliable reasoning processes. Mindora adopts a novel hybrid supervised fine tuningreinforcement learning (SFT–RL) training strategy with an LLMbased inconsistency-detection reward that dynamically enforces internal consistency while enhancing reasoning depth and generalization to unseen cases. To support Mindora’s training, we construct high-quality reasoning data through a Reasoning Trajectory Generation (RTG) strategy. RTG filters samples by difficulty and appliesastructuredrewritingprocedurethatproducesconcise,readable, and well-balanced reasoning trajectories, directly mitigating issues such as over-elaboration and improving reasoning clarity.

However,deployingLLMsinsuchsensitive,high-stakessettings requires more than producing fluent responses or superficially accuratepredictions.Effectivementalhealthsupportdependsontransparent,coherent,andcontext-groundedreasoningthatreflectshow human clinicians interpret complex, subjective narratives. When LLMs misread user self-reports, rely on incomplete reasoning, or accept subjective statements as factual, they may exaggerate symptoms, provide misleading feedback, or inadvertently amplify users’ anxiety [27]. At scale, these risks threaten public trust and can undermine the social-good promise of AI-enabled mental health support systems. These risks underscore the urgent need for LLMs capableofresponsible,clinicallyalignedreasoningratherthanmerely generating plausible answers [10, 17].

Recent efforts [4, 8, 40], have begun exploring how LLMs can better support mental-health tasks. Such as, Psyche-R1 [4] jointly integratesempathy,psychologicalknowledge,andchain-of-thought

1The code and data are available in MentraSuite.

###### Appraisal

Cognitive Error Identification

###### Psychiatry QA

Identify what cognitive pattern is present.

This year, one of my best friends has gone through a lot of issue with his other friends. He really feels like he has no one left besides me. All he ever does is work or stay at home. He rarely comes to church anymore... Q: The most likely cognitive error is? <choices>

... Her vital signs are within normal limits. On mental status examination, she is oriented to person, place, and time. She reports feeling anxious. Physical examination shows no abnormalities. Q: The most appropriate next step is? <choices>

CognitiveReframing PatternReframe Therapist Q&A

[Figure 1]

[Figure 2]

###### Diagnosis

###### Abstraction

###### Determine what the evidence shows.

###### Mental Health Condition Detection

Determine what mental condition is likely.

Psychiatry Systematic Review Summarization

It's kinda always been an issue. I wouldn't say it's as bad as my peers but it's definitely bad. There's a lot that I obviously can't write cause it would take a while but I'll just say the grand scheme of things. So basically I'm not really happy. I haven't been happy... Q: What is the depression severity (0/1/2)?

Title: Family-based interventions versus standard care for people with schizophrenia Abstract: ... shifting from high to low expressed emotion (RR 3.90, 95% CI 1.11 to 13.71; 2 RCTs, 72 participants... Q: The main findings of the systematic review is?

DepSign SWMH T-SID

[Figure 3]

[Figure 4]

PSRS*

###### Intervention

###### Verification

Counseling Strategy Formulation

###### Mental Health Misinfomation Identification

###### Decide what to do therapeutically.

Determine whether the mental-health information is accurate.

Video title: waves of depression #depression

Mrs. Smith has recently returned home after a period away, likely due to health challenges, and is confronted with mixed emotions about her current situation. She expresses happiness about being home... Q: select appropriate counseling strategies. <choices>

PsyDTCorpusM AnnoMIM

#mentalhealth #mentalilness #shorts #youtube

[Figure 5]

[Figure 6]

Audio transcript: wanted to be famous, just being real... Q: whether the video contains mental health misinformation?

MentalMisinfo

Figure 1: Tasks and datasets included in MentraBench.

Table 1: Comparison of MentraBench with existing mental-health reasoning works.

Statistics Task Choice Evaluation Aspects Training Strategy Task Dataset Appraisal Diagnosis Intervention Multi-step Abstraction Verification Correctness Reasoning Chain Ability Reward CoT Construction

Psyche-R1 [4] 3 4 × × × × × empathy, logic format, correctness prompt-rationale optimization Psy-Interpreter [8] 3 6 × × × × × × × knowledge bilateral reward knowledge injection PsychCounsel-Bench [40] 1 1 × × × × × × × \ \ \ MentraBench (ours) 6 13 logic, interpretability, consistency format, length, correctness, consistency structured, generation-verifier refinement

Extensive experiments on MentraBench covering 20 evaluated LLMs,showthatMindoraachievesthehighestaverageperformance across all 13 datasets, outperforming strong baselines such as GPT4o-mini and DeepSeek-R1. Trajectory-level analysis further confirmsremarkableperformanceacrossallfivereasoningdimensions, demonstrating Mindora’s superior ability to reason concisely, accurately, and coherently in complex mental-health scenarios.

In summary, our contribution can be summarized as follows:

(1) We present MentraBench, the first comprehensive benchmark designed to evaluate LLMs’ reasoning abilities across five key aspects of mental-health practice: appraisal, diagnosis, intervention, abstraction, and verification. It includes 6 tasks, and 13 datasets, emphasizing both task accuracy and reasoning quality.

- (2) We develop Mindora, a post-trained model that combines su-

pervised fine-tuning and reinforcement learning with a novel llm based consistency detection rewards to enhance reasoning conciseness, consistency, and factual grounding.

- (3) We propose the Reasoning Trajectory Generation strategy,

which produces structured and concise reasoning data through difficulty filtering and structured transformation, improving interpretability and mitigating reasoning redundancy.

(4) Extensive experiments demonstrate Mindora’s superior performance across task performance and all reasoning dimensions, surpassingstate-of-the-artmodelssuchasGPT-4o-miniandDeepSeekR1.

#### 2 Methods 2.1 MentraBench

Mental health reasoning requires a range of cognitive and decisionmaking skills, from recognizing distorted thought patterns to interpreting clinical evidence and selecting interventions. To systematically evaluate these capabilities in LLMs, we design MentraBench around six complementary dimensions. Appraisal tests cognitivepatternreasoning,identifyingsubtlemaladaptivethoughtprocesses. Diagnosisassessesconditionclassification,determininglikelymentalhealthproblemsfromclientdata.Interventionevaluatestherapeuticplanningskills,generatingappropriatecounselingstrategies.Multi-

step probes multi-step reasoning, integrating knowledge, diagnosis, and intervention in complex scenarios. Abstraction examines evidencesynthesis,summarizingfindingsfromstructuredresearch or experimental data. Verification challenges models to discern accurate from misleading mental health information. Each dimension is instantiated through a representative task paired with corresponding datasets. Figure 1 provides an overview of tasks and datasets included in MentraBench, and Table 2 summarizes key dataset statistics. MentraBench comprises six tasks and 13 datasets, constructed through refinements of existing resources and the creation of new ones. In Table 2, datasets marked with 𝑀 indicate those processed in this work, while those marked with ∗ denote newly curated and annotated datasets.

2.1.1 Appraisal:Cognitive-PatternReasoning. TheAppraisaldimension evaluates an LLM’s ability to identify what cognitive error is

###### Table 2: Dataset statistics.

Dataset Train/Valid/Test Source Annotation Metric Cognitive Error Identification CognitiveReframing [28] 751/173/284 synthetic, MHA Human MicroF1 PatternReframe [15] 1302/661/300 synthetic Human MicroF1 Therapist Q&A [29] 1662/409/507 counseling dialogues (real) Human MicroF1 Mental Health Condition Detection DepSign [22] 1198/300/600 Reddit weak supervision MicroF1 SWMH [12] 1200/300/500 Reddit weak supervision MicroF1 T-SID [12] 1200/300/942 Twitter weak supervision MicroF1 Counseling Strategy Formulation

PsyDTCorpus𝑀 1200/300/- counseling dialogues (syn) Human Jaccard AnnoMI𝑀 -/-/133 counseling dialogues (real) Human Jaccard Psychiatry QA MHQA [25] 1200/300/717 PubMed Human/LLM MicroF1 MedQA𝑀 464/58/121 Medical exams Human MicroF1 MedMCQA 318/120/446 Medical exams Human MicroF1 PubMedQA𝑀 -/-/89 PubMed Human MicroF1 Psychiatry Systematic Review Summarization

PSRS* -/-/108 Cochrane Library Human Recall

Mental Health Misinformation Identification MentalMisinfo [16] 123/31/130 Youtube, Bitchute Human MacroF1

Note: Datasets with 𝑀 are processed in this work. Dataset with * is newly curated and annotated in this work, where the ”recall” metric means the coverage of annotated scoring points.

present in a client’s self-reported statement. This dimension targets fine-grained cognitive-pattern reasoning, requiring the model to recognize subtle distorted appraisal cues and distinguish among similar forms of maladaptive thinking. We instantiate this dimension through the cognitive error identification task, evaluated usingthreehigh-qualitydatasetsthatspanmorethanadozencognitivedistortioncategories.CognitiveReframing[28]combinessimulated negative thoughts from the Thought Records Dataset [2] and selfreports from the Mental Health America website2, with distortions annotated by 15 trained mental-health professionals. PatternReframe [15], constructed from PERSONA-CHAT personas [41], contains statements crafted to manifest specific distortions and labeled by five independent raters. Therapist Q&A [29], derived from real therapist–client interactions in the Therapist Q&A corpus3, provides naturally occurring distorted statements annotated by two clinical raters. These three datasets adopt slightly different taxonomies of cognitive errors. In our prompt design, we follow each dataset’soriginaldefinitionsandexamples.Thecognitive-errorcategoriessharedacrossallthreedatasetsinclude:All-or-NothingThinking, Overgeneralization, Labeling, Fortune Telling, Mind Reading, Should Statements, and Personalization. Unique categories in CognitiveReframinginclude:EmotionalReasoning,ComparingandDespairing, Blaming, Negative Feeling or Emotion, Catastrophizing, and Discounting the Positive. Unique categories in PatternReframe include:MentalFiltering,Catastrophizing,andDiscountingthePositive. Unique categories in Therapist Q&A include: Emotional Reasoning, Mental Filtering, Magnification, and No Distortion.

- 2.1.2 Diagnosis:Mental-ConditionReasoning. TheDiagnosisdimension assesses an LLM’s ability to determine what mental condition is likely given a client’s textual expression. This type of reasoning requires the model to make calibrated, clinically aligned judgments about potential psychological problems, distinguishing between similar symptom presentations, avoiding overpathologizing, and assessing severity with nuance. We instantiate this dimension

- 2https://screening.mhanational.org/
- 3https://www.kaggle.com/arnmaud/therapist-qa

through the mental health condition detection task, which evaluates whether a model can identify and classify possible mental health issues from real-world social-media posts. To benchmark diagnostic performance, we employ one depression-specific dataset, DepSign[22],andtwomulti-conditionmental-healthclassification datasets collected from different platforms: SWMH [12] from Reddit and T-SID [12] from Twitter.

- 2.1.3 Intervention:Therapeutic-ActionReasoning. TheIntervention dimension evaluates an LLM’s ability to determine what counseling action should be taken in response to a client’s situation. This dimensiontargetstherapeutic-actionreasoning,requiringthemodel toanalyzetheclient’spresentationandselecttheinterventionstrategythatismostcontextuallyappropriate,ratherthanofferinggeneric, misplaced, or logically inconsistent responses. We instantiate this dimension through the counseling-strategy formulation task, covering thirteen commonly taught intervention types.4

To construct evaluation data, we use two high-quality counseling dialogue corpora. Client utterances are compressed into concise case summaries using GPT-4o prompts, and counselor utterances are annotated to extract strategy labels as reference answers. All constructed items are manually reviewed and verified by an expert counselor with over ten years of clinical experience, resulting in the final [dataset]𝑀 versions. PsyDTCorpus𝑀: Based on PsyDTCorpus [37], which contains 5,000 high-quality single-turn dialogues from SoulChatCorpus and 12 anonymized real counseling cases synthesized into multi-turn interactions. Due to its partially synthetic nature, this dataset is used to train Mindora but excluded from benchmark evaluation. AnnoMI𝑀: Based on AnnoMI [34], constructed from authorized motivational interviewing (MI) demonstration videos sourced from YouTube and Vimeo, transcribed and curated into high-quality multi-turn counseling interactions.

- 2.1.4 Multi-step: Multi-Step Clinical Reasoning. The Multi-step dimension assesses whether an LLM can perform multi-step clinical reasoning across appraisal, diagnosis, and intervention. Models must integrate symptom interpretation, condition identification, treatment selection, and research-level analysis within a single pipeline, mirroring the sequential decision-making process of mental-healthprofessionals.Weinstantiatethisdimensionthrough a psychiatry QA task and evaluate it using four psychiatry-focused datasets.MHQA[25]providesknowledge-intensivequestion-answer pairs from 471k PubMed abstracts covering major mental-health

conditions.FromMedQA𝑒𝑛 [13],MedMCQA[20],andPubMedQA[14], we manually extracted psychiatric-related questions to form the [𝑑𝑎𝑡𝑎𝑠𝑒𝑡]𝑀 version, ensuring that they test their diagnostic, intervention, and evidence-based reasoning abilities. Together, these datasets comprehensively benchmark an LLM’s capacity for endto-end, multi-step clinical reasoning grounded in both practice and research evidence.

4The strategies include Clarification, Paraphrasing, Reflection of Feeling, Summarizing, Questioning Skills, Immediacy, Use of Silence, Self-Disclosure, Confrontation, Encouragement, Repetition, Interpretation, and Guidance.

- 2.1.5 Abstraction: Evidence-Based Reasoning. The Abstraction dimension evaluates whether an LLM can determine what the evidence shows by interpreting and summarizing complex psychiatric research reports. This task targets free-text evidence reasoning, requiring models to process long, highly structured systematic review abstracts, extract key numerical and methodological information, and convert it into clinically meaningful conclusions, including effect direction and certainty levels. We instantiate this dimension through the Psychiatry Systematic Review Summarization (PSRS) dataset. Each case in PSRS was manually curated from the Cochrane Library [26] to cover a broad spectrum of psychiatric conditions, populations, and intervention types, and expert annotators generated scoring points for every instance to capture the main findings.

In this task, LLMs are required to summarize the main results reported in the studies—turning quantitative outcomes, confidence intervals, and methodological notes into concise, interpretable conclusions. This involves synthesizing statistical evidence into clinical interpretation, inferring effect direction, and assessing the certainty of evidence, rather than merely reproducing numerical results. All abstracts are publicly available and retain study identifiers to ensure traceability; no patient-level or individual trial data are included.

- 2.1.6 Verification: Misinformation Detection. The Verification dimensionevaluateswhetheranLLMcandeterminewhethermentalhealth information is accurate. This task targets misinformation detection reasoning, requiring models to identify misleading, anecdotal, or non-evidence-based claims in user-generated content while relying on authoritative clinical knowledge. We instantiate this dimension through the Mental Health Misinformation Identification task, where models must distinguish reliable mental-health information from potentially harmful narratives in textual scripts derived from social-media videos.

WeevaluatethistaskusingtheMentalMisinfodataset[16],which

contains video content from platforms such as YouTube Shorts and BitChute. The materials were systematically filtered, transcribed, and manually annotated to indicate whether each statement is accurate or misleading. By including this task, our benchmark assesses an LLM’s ability to perform evidence-based verification under real-world, noisy, and informal language conditions commonly encountered in online mental-health discourse.

#### 2.2 Mindora

To enable more reliable and clinically aligned mental-health reasoning, we propose Mindora, a post-trained large language model designed to improve both reasoning depth and reasoning fidelity. Mindora builds on a hybrid SFT–RL training framework that integrates high-quality reasoning trajectories with reinforcement signals targeting consistency.

- 2.2.1 ReasoningTrajectoryGeneration. TosupportMindora’straining with high-quality supervision signals, we propose a reasoning trajectory generation strategy that focuses on genuinely challenging reasoning steps rather than trivial pattern completion. Although the tasks and datasets described above are demonstrably

reasoning-intensive, directly using all training samples would dilute the supervision with many instances that models can already solve through surface-level cues. To obtain training data with sufficient difficulty, we perform zero-shot question answering on the training split using Llama-3-8B-Instruct and retain only the cases where the model produces incorrect answers. This filtering procedure ensures that the collected trajectories focus on problems that require deeper reasoning and are more informative for model training. The number of retained training cases is shown in Table 2.

To address the readability and structural incoherence of reasoning trajectories caused by backtracking in iterative search, we propose a structured reasoning trajectory generation method inspired by the search-based complex reasoning framework [3]. The core workflow involves two key stages: iterative optimal path search guided by a verifier and structured formatting of the optimal trajectory, ensuring both reasoning depth and interpretability.

- 2.2.2 Iterative Optimal Reasoning Path Search. We first leverage GPT-4o to explore and refine reasoning trajectories for verifiable mentalhealthproblems,followingafeedback-driveniterativesearch paradigm: Reasoning Generation. For a given verifiable problem 𝑥 with ground-truthanswer𝑦∗,GPT-4ogeneratesaninitialChain-of-Thought

(CoT), denoted as 𝑒0, and preliminary answer 𝑦0 by analyzing the case context and applying domain knowledge.

Verifier-GuidedRefinement.Amedicalverifierimplementedusing GPT-4o checks if 𝑦0 aligns with 𝑦∗. If the verification returns False, GPT-4o samples a search strategy to refine the trajectory: (1) Backtracking: Revisits earlier reasoning steps 𝑒𝑗 (𝑗 < 𝑖 − 1) to identify and resolve logical flaws. (2) Exploring New Paths: Develops an alternative reasoning approach distinct from prior attempts 𝑒0,...,𝑒𝑖−1. (3) Verification: Validates the logical consistency and factual accuracy of the current trajectory 𝑒𝑖−1. (4) Correction: Directly amends errors in the latest reasoning step𝑒𝑖−1 to align with domain principles.

Termination Conditions. The iteration continues until the verifier confirms the answer is correct. If the maximum number of iterations 𝑁 = 3 is reached without a correct answer, the search restarts for up to 𝑇 = 3 attempts and all failed trajectories are discarded.

This process ensures that the final trajectory [𝑒0,𝑦0,...,𝑒𝑖,𝑦𝑖] embodies iterative reflection and optimal reasoning, while avoiding stagnation in suboptimal paths.

- 2.2.3 StructuredReasoningFormats. Tomitigatereadabilitydegradation caused by backtracking and unstructured reflection, the optimal reasoning trajectory is formatted into a standardized structure with two mandatory phases. This formatting enforces clarity, consistency, and alignment between reasoning and answers, prohibiting deviations from the predefined schema.

In the reasoning phase: (1) All analytical content is enclosed within<think>tags.(2)Structuredsubtitles(e.g.,###SymptomAnal-

ysis, ###Differential Diagnosis) are used to segment reasoning steps, each on a separate line. (3) The phase concludes with a mandatory ###Final Conclusion section that summarizes the core logical chain and justifies the subsequent answer.

In the answer phase: (1) The final judgment is enclosed within <answer> tags. (2) The phase strictly ends with the format Answer:

[Figure 7]

### ?

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Inner Thinking

[Figure 12]

[Figure 13]

[Figure 14]

Case: A 60-year-old man is found roaming aimlessly by a shopkeeper and brought him to the hospital. He was unable to tell his address. The blood investigation assay revealed: Na+115 mEq/L, K+ 3.2 mEq/L. Q: Which of the following is the most likely diagnosis? <choices> a. Multi-infarct dementia b. Delirium c. Alzheimer's disease d. Dissociative fugue

Inner Thinking

GPT-4o

###### LLaMA3.1-8B

Final Conclusion

[Figure 15]

[Figure 16]

[Figure 17]

Agent-based reasoning trajectory generation

filter and retain only the hard cases where LLaMA fails training data

Inner Thinking

[Figure 18]

retained cases ( ) Conclusion

Final

[Figure 19]

[Figure 20]

[Figure 21]

sampling a batch from

[Figure 22]

trajectory structuring

<think> ###Analyzing Symptoms and Investigations: ... ###Differential Diagnosis of Confusion:... ###Assessing Impact of Electrolyte Imbalance: ... ###Final Conclusion: The most likely diagnosis for the patient's acute confusion due to severe hyponatremia is delirium. This conclusion aligns with the symptom profile and the effects of the electrolyte imbalance. </think> <answer>Answer: b</answer>

rollout BRL from current policy (K=8 candidates)

[Figure 23]

[Figure 24]

sampling a batch from

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Qwen3-32B

compute

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### SFT

GRPO

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

* SFT and RL are executed jointly in each training step, enabling dynamic loss fusion instead of a two-stage SFT→RL procedure.

iteration

[Figure 43]

joint training

after 463 iterations ...

Mindora

###### Figure 2: The framework of Mindora.

[option/result] to ensure unambiguous output. (3) The answer must be logically consistent with the conclusion derived in the Reasoning Phase.

a formalized reward mechanism to ensure the quality and validity of model outputs. The training framework integrates dual data streams, adaptive weight scheduling, and a multi-criteria reward function, as formally detailed below.

Training Framework and Notations. Let ℳ denote the base model (Qwen3-8B), ℳaux represent the auxiliary model (Qwen332B) for internal consistency detection, and 𝒟SFT ⊂ 𝒳 × 𝒴 denote the expert SFT dataset where 𝒳 is the set of mental health reasoning prompts and 𝒴 is the set of expert solutions. Let 𝒟RL ⊂ 𝒳 × 𝒴 be the RL exploration dataset generated by ℳ during rollout. The training objective is to optimize the policy 𝜋𝜃 (parameterized by 𝜃) via the CHORD algorithm, which dynamically fuses SFT loss ℒSFT and RL loss ℒRL using two-level weights: global weight 𝜇(𝑡) and token-level weight 𝜙(⋅).

An example of structured trajectory is shown in Figure 2.

- 2.2.4 Rationale for Structural Constraints. The mandatory formatting rules address several critical limitations of unstructured trajectories, specifically enhancing logical coherence, ensuring consistency and improving readability. Enhancing Logical Coherence. Unstructured trajectories often contain fragmented backtracking, such as Wait, earlier I forgot to check symptom duration, Let me revisit that which disrupts logical flow. By segmenting reasoning into titled modules and isolating backtracking within iterative search rather than the final output, the structured trajectory maintains a linear and coherent chain of logic. Ensuring Consistency. The mandatory ###Final Conclusion and Answer: [result] elements ensure that the answer directly reflects the reasoning process. This aims to avoiding inconsistencies, such as a conclusion favoring Major Depressive Disorder while the answer lists Generalized Anxiety Disorder. This alignment is crucial fortrainingmodelstoproducelogicallygroundedoutputsinmentalhealth tasks, where misalignment could lead to clinical misjudgments. Improving Interpretability. Structured subtitles (e.g., ###Differential Diagnosis Exclusion) teach the model to decompose complex mentalhealthjudgmentsintodomain-specificsub-tasks—mirroring how clinicians systematically analyze cases. This modular learning improvesthe model’s ability to replicate interpretable, professional reasoning patterns.
- 2.2.5 Training Procedure. The training of the mental health reasoning model adheres to the CHORD [42] algorithm’s core paradigm of dynamic balancing between sft and rl exploration, with

𝑠 (inputThe rewardprompt)𝑟(𝑠,𝑎)is a compositefor an actionfunction𝑎 (modelconsistingsolution)ofgivenfour sequen-a state tial validity and quality checks, formally defined as:

𝑟(𝑠,𝑎) = 𝕀(FormatValid(𝑎)) ⋅ 𝕀(LengthValid(𝑎)) ⋅ 𝕀(Consistency(𝑎)) ⋅ 𝑄Quality(𝑎) (1)

where: 𝕀(⋅) is the indicator function (1 if the condition holds, 0 otherwise). In detail, FormatValid(𝑎) verifies if 𝑎 adheres to the mandatory format <think>...</think><answer>...</answer>. LengthValid(𝑎) ensures the token length of the inner thinking trajectory 𝒯 in 𝑎 falls within a valid range [𝐿min,𝐿max], where 𝐿min = 10 tokens and 𝐿max = 2048 tokens. Consistency(𝑎) detects factual inconsistencies or errors in 𝒯 using the auxiliary model ℳaux.

𝑄Quality(𝑎) quantifies the correctness of 𝑎 based on task-specific benchmark criteria, with values in [0,1]. It is defined separately for three task types:

- (1) Single-choice questions: Let 𝑦∗ be the ground-truth answer.

Then:

𝑄Quality(𝑎) ={

1 if the final conclusion in 𝒯 = 𝑦∗, 0 otherwise. (2)

- (2) Multiple-choice questions: Let 𝑌∗ = {𝑦1∗,𝑦2∗,...,𝑦𝑘∗} be the

set of ground-truth answers, and 𝑌 = {𝑦1,𝑦2,...,𝑦𝑚} be the set of

𝜋sample(𝑎𝑗,𝑡|𝑥𝑗,𝑎𝑗,<𝑡) (importancesamplingratio,𝜋sample is the reference policy), 𝐴𝑗 = 𝑟(𝑥𝑗,𝑎𝑗)−𝜇𝑅

where:𝑟𝑗,𝑡(𝜃) = 𝜋𝜃(𝑎𝑗,𝑡|𝑥𝑗,𝑎𝑗,<𝑡)

MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment Conference acronym ’XX, June 03–05, 2018, Woodstock, NY

answers𝑌 and 𝑌∗in: 𝒯. The quality score is the Jaccard similarity between

|𝑌 ∩ 𝑌 |𝑌 ∪ 𝑌∗|

𝜎𝑅+𝜖𝑧 (normalized advantage, 𝜇𝑅/𝜎𝑅 are the mean/std of rewards in the rollout group, 𝜖𝑧 = 10−8 for stability), 𝜖 = 0.2. The policy parameters 𝜃 are updated by

𝑄Quality(𝑎) =

. (3)

(3) Short-answer questions: Let 𝒦 = {𝑘1,𝑘2,...,𝑘𝑛} be the set of key scoring points for the task, and 𝒦hit ⊆ 𝒦 be the subset of points covered in 𝒯. Then:

minimizing ℒtotal(𝜃) using the Adam optimizer with 𝛽1 = 0.9, 𝛽2 = 0.999, and learning rate 𝜂 = 2×10−6. Training checkpointing is performed every 10 steps to preserve intermediate model states.

|𝒦hit| |𝒦|

𝑄Quality(𝑎) =

. (4)

Training Pipeline. The training process proceeds in iterations 𝑡 = 1,2,...,𝑇max (where 𝑇max is the total number of training steps), with each iteration consisting of data sampling, weight scheduling, loss computation, and parameter update stages.

3 Experiments

Step1:DataSampling.Ateachstep𝑡,wesampletwomini-batches:

- 3.1 Experimental Settings

Evaluated LLMs. To assess whether model scale, distilled reasoning variants, and the gap between reasoning-oriented and chatorientedLLMsaffectperformanceonmentalhealthreasoningtasks, we evaluated a broad range of GPT, DeepSeek, Qwen, and LLaMA models.Forclosed-sourcesystems,weincludedthereasoningmodelsGPT-o(o4-mini),DeepSeek-R(DeepSeek-R1),andQwen’sQwQ series(QwQ-plus),aswellastheleadingchatmodelsGPT-4o,DeepSeekV3, and Qwen-plus5. For open-source models, we tested Qwen, LLaMA,andDeepSeek-distilledvariantsacrossmultiplescales,with the full list provided in Table 3. For psychology-focused LLMs, we evaluated EmoLLM6 , finetuned from Qwen2-7B-Instruct as a wellknown mental-health chat model, and Psyche-R17 , an SFT+GRPOtrained Qwen2.5-7B-Instruct model representing the latest mentalhealth-reasoning LLMs.

EvaluationSettings.Allclosed-sourceLLMsareaccessedthrough their official APIs, while open-source models are deployed on a single NVIDIA A800-SXM4-80GB GPU. Parameters such as temperature are kept at their default values. Prompts are aligned with the Mindora format to generate structured reasoning chains for consistent comparison.

- 3.2 Main Results

An SFT mini-batch ℬSFT ∼ 𝒟SFT with batch size 𝐵SFT = 64 (consistentwiththeexperimentalsetup),whereeachsampleis(𝑥𝑖,𝑦𝑖∗) ∈

𝒳×𝒴(𝑥𝑖 is the prompt,𝑦𝑖∗ is the expert solution). An RL mini-batch ℬRL ∼ 𝒟RL with dynamic batch size 𝐵RL, where each sample is (𝑥𝑗,𝑎𝑗) ∈ 𝒳 × 𝒴 ( 𝑎𝑗 is the model-generated solution via rollout). The rollout for 𝒟RL generation uses a temperature 𝜏 = 1.0, with 𝐾 = 8 candidate solutions sampled per prompt.

##### Step 2: Adaptive Weight Scheduling. The global weight 𝜇(𝑡)

(balancing ℒSFT and ℒRL) follows a warmup-decay schedule to transition from expert imitation to RL exploration:

(1) Warmup phase (1 ≤ 𝑡 ≤ 𝑡warmup, 𝑡warmup = 200 steps):

𝑡 𝑡warmup

𝜇(𝑡) = 𝜇valley + (𝜇peak − 𝜇valley) ⋅

, (5)

where 𝜇peak = 0.5 (maximum SFT influence) and 𝜇valley = 0.02 (minimum SFT influence).

(2) Decay phase (𝑡warmup < 𝑡 ≤ 𝑡warmup + 𝑡decay, 𝑡decay = 400 steps):

𝑡 − 𝑡warmup 𝑡decay

𝜇(𝑡) = 𝜇peak − (𝜇peak − 𝜇valley) ⋅

. (6)

The token-wise weight 𝜙(𝑦𝑡∗;𝜋𝜃) (for SFT loss modulation) is defined based on the policy’s probability of generating the expert token 𝑦𝑡∗ (given prompt 𝑥 and prefix 𝑦<𝑡∗ ):

𝜙(𝑦𝑡∗;𝜋𝜃) = 𝑝𝑡(1 − 𝑝𝑡), (7)

In this benchmark, we evaluate the reasoning performance of various models across 13 datasets covering five core aspects of mentalhealth practice. The results, presented in Table 3, highlight several key findings.

where 𝑝𝑡 = 𝜋𝜃(𝑦𝑡∗|𝑥,𝑦<𝑡∗ ). This parabolic function prioritizes learning for tokens where the policy is uncertain ( 𝑝𝑡 ≈ 0.5 ) while downweighting certain ( 𝑝𝑡 ≈ 1 ) or irrelevant ( 𝑝𝑡 ≈ 0 ) tokens.

In general, Mindora𝐶𝐻𝑂𝑅𝐷 achieves the highest average score across all datasets, followed by Mindora𝑆𝐹𝑇 + 𝑅𝐿, both surpassing leading proprietary reasoning models such as GPT-o4-mini and DeepSeek-R1.Thisdemonstratestheeffectivenessofourpost-training strategyinenhancingreasoningperformancewithincomplex,context-

lossℒtotal(𝜃)isaweightedcombinationoftheSFTloss(withtokenwise weighting) and the RL loss:

###### Step 3: Loss Computation and Parameter Update. The total

ℒtotal(𝜃) = (1 − 𝜇(𝑡)) ⋅ ℒGRPO(𝜃) + 𝜇(𝑡) ⋅ ℒSFT−𝜙(𝜃), (8)

SFTlosswithtoken-wiseweightingℒSFT−𝜙(𝜃)iscomputedover

sensitivemental-healthtasks.Acrosseverydataset,Mindora𝐶𝐻𝑂𝑅𝐷 outperform the backbone model Qwen3-8B, showing consistent

ℬSFT, minimizing the weighted negative log-likelihood of expert solutions:

gainsinbothaccuracyandreasoningquality.Furthermore,Mindora𝐶𝐻𝑂𝑅𝐷 alsoexceedstheseparatelytrainedMindora𝑆𝐹𝑇 andMindora𝑆𝐹𝑇+𝑅𝐿 variants, confirming that the joint SFT–RL training paradigm more effectively balances imitation and exploration, avoiding overfitting while improving generalization to unseen cases.

ℒSFT−𝜙(𝜃)

|𝑦∗

𝑖 |

1 ∑(𝑥𝑖,𝑦∗

∑

∑

𝜙(𝑦𝑖,𝑡∗ ;𝜋𝜃) ⋅ log𝜋𝜃(𝑦𝑖,𝑡∗ |𝑥𝑖,𝑦𝑖,<𝑡∗ ). (9)

= −

|𝑦𝑖∗|

(𝑥𝑖,𝑦∗

𝑡=1

𝑖 )∈ℬSFT

𝑖 )∈ℬSFT

GRPO loss ℒGRPO(𝜃) is optimized over ℬRL to maximize the expected reward, using a clipped surrogate objective (consistent with PPO-style updates):

Performance analysis across the five task categories reveals that

MindoraCHORD demonstrates strong appraisal and diagnostic reasoning, showing improved recognition of subtle cognitive patterns andaccuratedifferentiationofoverlappingsymptompresentations.

ℒGRPO(𝜃)

|𝑎𝑗|

1 ∑(𝑥𝑗,𝑎𝑗)∈ℬRL |𝑎𝑗|

∑

∑

min(𝑟𝑗,𝑡(𝜃)𝐴𝑗,clip(𝑟𝑗,𝑡(𝜃),1 − 𝜖,1 + 𝜖)𝐴𝑗), (10)

= −

5Qwen-plus is Qwen’s versatile chat model, also trained with reasoning capabilities.

𝑡=1

(𝑥𝑗,𝑎𝑗)∈ℬRL

- 6https://www.modelscope.cn/models/aJupyter/EmoLLM_Qwen2-7B-Instruct_lora/.
- 7https://huggingface.co/MindIntLab/Psyche-R1

It also achieves notable gains in intervention and abstraction tasks, reflecting enhanced multi-stage reasoning and evidence synthesis. The model’s consistent results in verification tasks further suggest reliable factual grounding and resistance to misinformation.

the thought Am I insane? after experiencing the situation of feeling watched by others. The correct reasoning requires recognizing that the cognitive error lies in the thought itself, specifically, labeling, as the client directly labels themselves as insane. However, most models mistakenly focused on the situation rather than the thought, interpreting the error as treating feelings as facts due to the client’s perception of being watched. Only our model correctly concentratedonthe thoughtitself, avoidingconfusion between the external situation and the internal self-labeling process.

We also observe several interesting phenomena. Among opensource LLMs, different versions of the same model (e.g., DSDistillQwen3-32B, Qwen3-32B, and QwQ-32B) show only minor differences in overall reasoning performance, regardless of whether they are distilled, chat, or reasoning variants. This suggests that mentalhealth reasoning tasks demand specialized reasoning abilities that general-purpose post-training cannot fully capture, underscoring the need for targeted reasoning optimization for mental health scenarios. Moreover, we find that open-source models ranging from 14B to 70B parameters achieve similar average scores around 0.6, while 8B-scale models remain near 0.55. In contrast, our Mindora series and the baseline Psyche-R1, both specifically optimized for mental-health reasoning, exceed the average performance of 8B models,demonstratingthestrongpotentialoftargetedpost-training in this domain.

#### 4 Related Work

Recent efforts have begun to explore psychological reasoning and counseling intelligence in LLMs. Psyche-R1 [4] represents a major step toward domain-specific psychological LLMs that combine empathy, expertise, and reasoning. It introduces a synthetic benchmark and a hybrid training pipeline that mixes supervised finetuning on easier samples with reinforcement learning on filtered hard cases, improving interpretive reasoning and emotional understanding. Psy-Interpreter [8] further advances psychological and social-cognitivereasoningthroughtheStimuliQAdataset,composed of expert-annotated narrative stimuli capturing emotions and collective cognition. Its bilateral reinforcement-learning design aligns model trajectories with expert reasoning patterns, enhancing interpretive and social-cognitive generalization. PsychCounsel-Bench [40] complements these modeling efforts by constructing a 2,200question benchmark derived from counselor-certification exams covering counseling methods, abnormal and developmental psychology, and ethics. Evaluations on leading models (e.g., GPT-4o, Llama 3.3-70B, Gemma 3-27B) suggest that current LLMs can already master exam-level psychological knowledge. Despite their contributions, existing works largely emphasize emotional understanding,socialinference,ortheoreticalknowledge,leavingdeeper reasoningprocessesunderexplored.Psyche-R1andPsy-Interpreter enhance empathy and emotion-related reasoning but do not assess multi-stagechainsthatintegrateappraisal,diagnosis,andintervention. PsychCounsel-Bench evaluates professional knowledge but is limited to exam-style multiple-choice questions, testing recall rather than open-ended, context-dependent reasoning. As a result, key abilities:such as synthesizing conflicting evidence, distinguishing overlapping symptoms, generating context-appropriate interventions, and verifying factual accuracy, remain unmeasured.

#### 3.3 Reasoning Trajectory Evaluation

In this section, we evaluate the reasoning trajectory quality of best performing models in MentraBench from each major family (GPT, DeepSeek, Qwen, and LLaMA), the mental-health-oriented PsycheR1, our backbone Qwen3-8B, and our proposed Mindora𝐶𝐻𝑂𝑅𝐷. To assess reasoning quality beyond task accuracy, we conduct a detailed reasoning trajectory evaluation based on five criteria: reasoning conciseness, logical coherence, hallucination, task understanding, and internal consistency. Each reasoning chain is manually evaluated following a binary guideline, where a score of 1 is assigned if no errors are observed and 0 otherwise, with detailed scoring criteria shown in Appendix A. The final reasoning trajectory score is computed as the average across all five dimensions, providing a comprehensive measure of reasoning reliability and transparency in mental-health tasks.

For each model and each dataset, we sample four representative cases, two where all models produce correct answers and two where all models fail, to ensure fairness in comparison.

Table 4 shows that Mindora𝐶𝐻𝑂𝑅𝐷 achieves remarkable average trajectory score, demonstrating balanced performance across multiple evaluation dimensions. Through refined training and balanced optimization, our model attains the best overall correctness and interpretability, highlighting its strength in both reasoning accuracy and clarity.

#### 5 Conclusion

Forredundancyandbacktrackingrelateddimensions,Mindora𝐶𝐻𝑂𝑅𝐷

In this work, we introduced MentraSuite, comprising the MentraBench benchmark and the Mindora model, to systematically advance and evaluate mental-health reasoning. Unlike prior studies that focus primarily on emotional understanding or knowledgebased assessment, our benchmark targets five clinically grounded reasoning aspects to capture the multi-stage and context-sensitive nature of real mental-health practice: appraisal, diagnosis, intervention,abstraction,andverification.WefurtherdevelopedastructuredReasoningTrajectoryGenerationmethodandthepost-trained model Mindora, which integrates supervised fine-tuning and reinforcement learning with consistency-aware reward design. Experimental results demonstrate that Mindora’s overall reasoning performance surpasses strong baselines such as GPT-4o-mini and

shows substantial improvement over the backbone model Qwen38B. This confirms the effectiveness of the structured trajectory generation step in our training data, which enables the model to produce reasoning chains that are more organized, concise, and logically coherent.

#### 3.4 Case Study

In this section, we analyze a challenging case of cognitive error identification shown in Appendix Figure 3, where all compared modelsproducedincorrectanswers.Inthiscase,theclientexpresses

###### Table 3: Experimental results on MentraBench.

Model CognitiveReframing PatternReframe Therapist Q&A 𝐴𝑣𝑔1 DepSign SWMH T-SID 𝐴𝑣𝑔2 AnnoMI𝑀 MHQA MedQA𝑀 MedMCQA𝑀 PubMedQA𝑀 𝐴𝑣𝑔4 PSRS MentalMisinfo 𝐴𝑣𝑔𝑎𝑙𝑙 close-source models

GPT-o4-mini[19] 0.7002 0.6607 0.4468 0.6026 0.3607 0.7374 0.7515 0.6165 0.2086 0.5462 0.8925 0.8102 0.7118 0.7402 0.8650 0.7781 0.6515 GPT-4o[18] 0.6791 0.6667 0.4543 0.6000 0.4064 0.7500 0.7114 0.6226 0.3358 0.4531 0.6138 0.7936 0.6949 0.6389 0.9065 0.6894 0.6273 Deepseek-R1[6] 0.7516 0.7069 0.4472 0.6352 0.4085 0.7624 0.7218 0.6309 0.2020 0.4984 0.8608 0.8988 0.7256 0.7459 0.9037 0.5689 0.6505 Deepseek-V3[7] 0.6540 0.6755 0.4131 0.5809 0.4211 0.7715 0.7581 0.6502 0.2140 0.4768 0.8184 0.8696 0.6848 0.7124 0.9296 0.6156 0.6386 Qwen-plus[38] 0.6821 0.6395 0.4057 0.5758 0.4043 0.7624 0.7662 0.6443 0.1866 0.4795 0.8242 0.8895 0.7199 0.7283 0.9048 0.6018 0.6387 QwQ-plus[24] 0.6698 0.6871 0.3698 0.5756 0.4085 0.7547 0.7330 0.6321 0.1848 0.4201 0.6405 0.8354 0.7226 0.6546 0.9335 0.4838 0.6034 70B+ open-source models LLaMA-4[1] 0.6588 0.6486 0.4031 0.5702 0.3914 0.7310 0.7884 0.6369 0.2066 0.4684 0.7414 0.8900 0.7239 0.7059 0.7634 0.6158 0.6178 LLaMA-3.3-70B[11] 0.6444 0.6301 0.3566 0.5437 0.4148 0.7163 0.7157 0.6156 0.2023 0.4148 0.7712 0.8904 0.7299 0.7016 0.6318 0.6894 0.6006 dsdistill-LLaMA-70B[6] 0.6667 0.6667 0.3905 0.5746 0.4000 0.7516 0.6953 0.6156 0.2367 0.4404 0.7097 0.7957 0.7452 0.6727 0.8914 0.6814 0.6209 Qwen2.5-72B[24] 0.6852 0.6207 0.3930 0.5663 0.3806 0.7639 0.8136 0.6527 0.2796 0.4022 0.7449 0.8402 0.7229 0.6775 0.9555 0.6091 0.6316 32B open-source models dsdistill-Qwen32B[6] 0.6540 0.6696 0.3539 0.5592 0.3979 0.7358 0.7373 0.6237 0.2178 0.4010 0.6697 0.7795 0.7229 0.6433 0.9225 0.6177 0.6061 Qwen3-32B[38] 0.6247 0.6395 0.4255 0.5632 0.3806 0.7685 0.8038 0.6510 0.2016 0.4355 0.7415 0.8206 0.7083 0.6765 0.9333 0.5025 0.6143 QwQ-32B[24] 0.6791 0.6842 0.3905 0.5846 0.4085 0.7500 0.7296 0.6294 0.1749 0.4305 0.6831 0.8470 0.7375 0.6745 0.9296 0.4642 0.6084 14B open-source models dsdistill-Qwen14B[6] 0.6635 0.5981 0.3404 0.5340 0.4169 0.7421 0.7750 0.6447 0.2470 0.3823 0.6198 0.6660 0.7497 0.6045 0.8641 0.5795 0.5880 Qwen3-14B[38] 0.6540 0.6486 0.4820 0.5949 0.3871 0.7484 0.7540 0.6298 0.1680 0.4215 0.7047 0.8102 0.6882 0.6562 0.8155 0.6600 0.6109 7∼8B open-source models

|LLaMA3.1-8B[11] dsdistill-LLaMA-8B[6] EmoLLM[32] Psyche-R1[5] Qwen3-8B[38]|0.5871 0.5714 0.1957 0.4514 0.5906 0.5507 0.2396 0.4603 0.6180 0.5472 0.3322 0.4991 0.5013 0.5222 0.2021 0.4085 0.5941 0.6395 0.4057 0.5464<br><br>|0.4476 0.7358 0.7581 0.6472 0.4190 0.7294 0.7490 0.6325 0.4334 0.7229 0.7931 0.6498 0.4169 0.6877 0.7962 0.6336 0.4169 0.7358 0.7515 0.6347<br><br>|0.2147 0.2001 0.1408 0.2424 0.1489|0.2771 0.5791 0.7925 0.7096 0.5896 0.3321 0.4430 0.6070 0.7587 0.5352 0.3373 0.4311 0.5749 0.6735 0.5042 0.5872 0.6989 0.8401 0.7194 0.7114 0.4009 0.5824 0.7927 0.6899 0.6165<br><br>|0.6805 0.8157 0.7906 0.8164 0.8115|0.6703 0.7617 0.6346 0.6954 0.4773<br><br>|0.5553 0.5536 0.5407 0.5943 0.5729|
|---|---|---|---|---|---|---|---|
|Mindora𝑆𝐹𝑇 Mindora𝑆𝐹𝑇+𝑅𝐿 Mindora𝐶𝐻𝑂𝑅𝐷|0.5693 0.6207 0.4472 0.5457 0.5975 0.6207 0.4566 0.5583 0.7293 0.6842 0.5088 0.6408<br><br>|0.4887 0.7437 0.7597 0.6640 0.5240 0.7715 0.8140 0.7032 0.4655 0.7760 0.8030 0.6815<br><br>|0.4193 0.3803 0.4016|0.5830 0.7263 0.8535 0.7376 0.7251 0.5996 0.7839 0.8593 0.8212 0.7660 0.6317 0.7590 0.8535 0.8442 0.7721<br><br>|0.7764 0.7159 0.8379<br><br>|0.5512 0.5681 0.7178|0.6367 0.6548 0.6933<br><br>|

###### Table 4: Reasoning trajectory evaluation.

Model R1 R2 R3 R4 R5 R𝑎𝑣𝑔

Annotation Scores GPT-o4-mini 1.0000 0.4135 1.0000 0.9519 1.0000 0.8731 Deepseek-R1 0.9519 0.9808 1.0000 0.9808 1.0000 0.9827 Qwen-plus 0.6538 1.0000 1.0000 0.9808 1.0000 0.9270 LLaMA-4 1.0000 0.2596 1.0000 0.8942 0.9423 0.8192 Psyche-R1 0.9423 0.9038 1.0000 0.9712 0.9038 0.9442 Qwen3-8B 0.5769 0.9712 1.0000 0.9808 0.9904 0.9039 Mindora𝐶𝐻𝑂𝑅𝐷 0.9423 0.9519 1.0000 0.9904 0.9808 0.9731 Inner-Annotator Agreement Gwet AC1 0.9607 0.8918 1.0000 0.9705 0.9855 0.9617 Cohen’s Kappa 0.8692 0.7906 1.0000 0.6033 0.7298 0.7986 Consistency 0.9698 0.9286 1.0000 0.9725 0.9863 0.9714

Note: R1: Reasoning Conciseness; R2: Logical Coherence; R3: No Hallucination; R4: Task Understanding; R5: Internal Consistency.

DeepSeek-R1, achieving balanced performance across datasets and superior reasoning-chain quality.

Beyond performance improvements, our findings highlight the importance of transparent, coherent, and context-grounded reasoning in clinical applications. Structured trajectory data and targeted post-training effectively reduce reasoning redundancy, enhance internal consistency, and improve interpretability, which are key steps toward reliable AI-assisted mental-health assessment. We believe this work provides a solid foundation for studying how reasoning-oriented alignment can enable LLMs to assist in clinical decision-making responsibly and ethically, aligning with the Web for Good vision of developing AI systems that serve human wellbeing with trustworthiness and social value.

#### Acknowledgments

This work is partially supported by Key Project of the National Natural Science Foundation of China (U23A20316), and CCF-Tencent Rhino-Bird Open Research Fund (CCF-Tencent RAGR20250115).

#### References

- [1] Meta AI. 2025. Introducing LLaMA 4: Advancing Multimodal Intelligence. https: //ai.meta.com/blog/llama-4-multimodal-intelligence/
- [2] Franziska Burger, Mark A Neerincx, and Willem-Paul Brinkman. 2021. Natural language processing for cognitive therapy: extracting schemas from thought records. PloS one 16, 10 (2021), e0257832.
- [3] Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. 2024. HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs. arXiv:2412.18925 [cs.CL] https://arxiv.org/abs/ 2412.18925
- [4] Chongyuan Dai, Jinpeng Hu, Hongchang Shi, Zhuo Li, Xun Yang, and Meng Wang. 2025. Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning. arXiv preprint arXiv:2508.10848 (2025).
- [5] Chongyuan Dai, Jinpeng Hu, Hongchang Shi, Zhuo Li, Xun Yang, and Meng Wang. 2025. Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning. arXiv:2508.10848 [cs.CL] https://arxiv.org/ abs/2508.10848
- [6] DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948 [cs.CL] https://arxiv.org/abs/ 2501.12948
- [7] DeepSeek-AI. 2025. DeepSeek-V3 Technical Report. arXiv:2412.19437 [cs.CL] https://arxiv.org/abs/2412.19437
- [8] Yichao Feng. 2025. From Stimuli to Minds: Enhancing Psychological Reasoning in LLMs via Bilateral Reinforcement Learning. arXiv preprint arXiv:2508.02458

(2025).

- [9] GBD 2019 Mental Disorders Collaborators. 2022. Global, regional, and national burden of 12 mental disorders in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. The Lancet Psychiatry 9, 2 (Feb. 2022), 137–150. doi:10.1016/S2215-0366(21)00395-3 Epub 2022 Jan 10.
- [10] Declan Grabb, Max Lamparth, and Nina Vasan. 2024. Risks from Language Models for Automated Mental Healthcare: Ethics and Structure for Implementation. arXiv:2406.11852 [cs.CY] https://arxiv.org/abs/2406.11852
- [11] Aaron Grattafiori, Abhimanyu Dubey, and et al Abhinav Jauhri. 2024. The Llama 3 Herd of Models. arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783
- [12] Shaoxiong Ji, Xue Li, Zi Huang, and Erik Cambria. 2022. Suicidal ideation and mental disorder detection with attentive relation networks. Neural Computing and Applications 34, 13 (2022), 10309–10319.
- [13] Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2020. What Disease does this Patient Have? A Large-scale Open

- Domain Question Answering Dataset from Medical Exams. arXiv preprint arXiv:2009.13081 (2020).
- [14] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu.

2019. PubMedQA: A Dataset for Biomedical Research Question Answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2567–2577.

- [15] Mounica Maddela, Megan Ung, Jing Xu, Andrea Madotto, Heather Foran, and Y-Lan Boureau. 2023. Training Models to Generate, Recognize, and Reframe Unhelpful Thoughts. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, 13641–13660. doi:10.18653/V1/2023. ACL-LONG.763
- [16] Viet Cuong Nguyen, Mini Jain, Abhijat Chauhan, Heather Jamie Soled, Santiago Alvarez Lesmes, Zihang Li, Michael L Birnbaum, Sunny X Tang, Srijan Kumar, and Munmun De Choudhury. 2025. Supporters and Skeptics: LLMbased Analysis of Engagement with Mental Health (Mis) Information Content on Video-sharing Platforms. In Proceedings of the International AAAI Conference on Web and Social Media, Vol. 19. 1329–1345.
- [17] Francis C. Ohu, Darrell Norman Burrell, and Laura A. Jones. 2025. Public Health Risk Management, Policy, and Ethical Imperatives in the Use of AI Tools for Mental Health Therapy. Healthcare 13, 21 (2025). doi:10.3390/healthcare13212721
- [18] OpenAI. 2024. GPT-4o System Card. https://cdn.openai.com/gpt-4o-systemcard.pdf
- [19] OpenAI. 2025. OpenAI o3 and o4-mini System Card. https://cdn.openai.com/ pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf
- [20] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering. arXiv:2203.14371 [cs.CL] https://arxiv.org/abs/2203. 14371
- [21] Vikram Patel, Shekhar Saxena, Crick Lund, Graham Thornicroft, Florence Baingana, Paul Bolton, Dan Chisholm, Pamela Y. Collins, Janice L. Cooper, John Eaton, Helen Herrman, Mohammad M. Herzallah, Y. Huang, Mark J. D. Jordans, Arthur Kleinman, Maria E. Medina-Mora, Emily Morgan, Uzma Niaz, Olayinka Omigbodun, Martin Prince, Atif Rahman, Benedetto Saraceno, Bidyut K. Sarkar, Mary De Silva, Indira Singh, Dan J. Stein, Charlene Sunkel, and Jürgen Unützer.

2018. The Lancet Commission on global mental health and sustainable development. The Lancet 392, 10157 (Oct. 2018), 1553–1598. doi:10.1016/S0140-6736(18) 31612-X Epub 2018 Oct 9; Erratum in: Lancet. 2018 Oct 27;392(10157):1518. doi: 10.1016/S0140-6736(18)32624-2.

- [22] Rafał Poświata and Michał Perełkiewicz. 2022. OPI@LT-EDI-ACL2022: Detecting Signs of Depression from Social Media Text using RoBERTa Pre-trained Language Models. In Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion, Bharathi Raja Chakravarthi, B Bharathi, John P McCrae, Manel Zarrouk, Kalika Bali, and Paul Buitelaar (Eds.). Association for Computational Linguistics, Dublin, Ireland, 276–282. doi:10.18653/v1/2022.ltedi1.40
- [23] Huachuan Qiu and Zhenzhong Lan. 2024. Interactive agents: Simulating counselor-client psychological counseling via role-playing llm-to-llm interactions. arXiv preprint arXiv:2408.15787 (2024).
- [24] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
- [25] Suraj Racha, Prashant Joshi, Anshika Raman, Nikita Jangid, Mridul Sharma, Ganesh Ramakrishnan, and Nirmal Punjabi. 2025. MHQA: A Diverse, Knowledge Intensive Mental Health Question Answering Challenge for Language Models. arXiv preprint arXiv:2502.15418 (2025).
- [26] E Ri, E Je, D Arikpo, M Mm, and C Ja. 2015. Cochrane Library Trusted evidence. Informed decisions. Better health. Cochrane Database of Systematic Reviews [Intervention Review] Hand washing promotion for preventing diarrhoea. DOI 10, 14651858 (2015), 1–94.
- [27] K. L. Rosen, M. Sui, K. Heydari, E. J. Enichen, and J. C. Kvedar. 2025. The perils of politeness: how large language models may amplify medical misinformation. NPJ Digital Medicine 8, 1 (Nov. 2025), 644. doi:10.1038/s41746-025-02135-7
- [28] Ashish Sharma, Kevin Rushton, Inna Lin, David Wadden, Khendra Lucas, Adam Miner, Theresa Nguyen, and Tim Althoff. 2023. Cognitive Reframing of Negative Thoughts through Human-Language Model Interaction. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 9977–10000.
- [29] Sagarika Shreevastava and Peter Foltz. 2021. Detecting Cognitive Distortions from Patient-Therapist Interactions. In Proceedings of the Seventh Workshop on

- Computational Linguistics and Clinical Psychology: Improving Access, Nazli Goharian, Philip Resnik, Andrew Yates, Molly Ireland, Kate Niederhoffer, and Rebecca Resnik (Eds.). Association for Computational Linguistics, Online, 151–158. doi:10.18653/v1/2021.clpsych-1.17
- [30] Gopendra Vikram Singh, Sai Vardhan Vemulapalli, Mauajama Firdaus, and Asif Ekbal. 2024. Deciphering cognitive distortions in patient-doctor mental health conversations: A multimodal llm-based detection and reasoning framework. In Proceedings of the 2024 conference on empirical methods in natural language processing. 22546–22570.
- [31] Hoyun Song, Huije Lee, Jisu Shin, Sukmin Cho, Changgeon Ko, and Jong C. Park.

2025. Does Rationale Quality Matter? Enhancing Mental Disorder Detection via Selective Reasoning Distillation. In Findings of the Association for Computational Linguistics: ACL 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 21738–21756. doi:10.18653/v1/2025.findings-acl.1119

- [32] EmoLLM Team. 2024. EmoLLM: Reinventing Mental Health Support with Large Language Models. https://github.com/SmartFlowAI/EmoLLM.
- [33] World Health Organization. 2025. Over a billion people living with mental health conditions – services require urgent scale-up. https://www.who.int/news/item/02-09-2025-over-a-billion-people-livingwith-mental-health-conditions-services-require-urgent-scale-up. News release; accessed 2025-12-01.
- [34] Zixiu Wu, Simone Balloccu, Vivek Kumar, Rim Helaoui, Ehud Reiter, Diego Reforgiato Recupero, and Daniele Riboni. 2022. Anno-MI: A Dataset of ExpertAnnotated Counselling Dialogues. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 6177–6181. doi:10. 1109/ICASSP43922.2022.9746035
- [35] Mengxi Xiao, Ben Liu, He Li, Jimin Huang, Qianqian Xie, Xiaofen Zong, Mang Ye, and Min Peng. [n.d.]. MoodAngels: A Retrieval-augmented Multi-agent Framework for Psychiatry Diagnosis. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.
- [36] Mengxi Xiao, Qianqian Xie, Ziyan Kuang, Zhicheng Liu, Kailai Yang, Min Peng, Weiguang Han, and Jimin Huang. 2024. HealMe: Harnessing Cognitive Reframing in Large Language Models for Psychotherapy. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 1707–1725. doi:10.18653/v1/ 2024.acl-long.93
- [37] Haojie Xie, Yirong Chen, Xiaofen Xing, Jingkai Lin, and Xiangmin Xu. 2025. PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 1081–1115. doi:10.18653/v1/2025.acl-long.55
- [38] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https://arxiv.org/abs/2505.09388
- [39] Kailai Yang, Tianlin Zhang, Ziyan Kuang, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. 2024. MentaLLaMA: interpretable mental health analysis on social media with large language models. In Proceedings of the ACM Web Conference 2024. 4489–4500.
- [40] Min Zeng. 2025. PsychCounsel-Bench: Evaluating the Psychology Intelligence of Large Language Models. arXiv preprint arXiv:2510.01611 (2025).
- [41] Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing Dialogue Agents: I have a dog, do you have pets too?. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2204–2213.
- [42] Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. 2025. On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting. arXiv:2508.11408 [cs.LG] https://arxiv.org/abs/2508.11408

#### A Reasoning Chain Evaluation Guideline

Assign 1 if no instances of the issue are present in the reasoning chain; assign 0 if any instance is observed.

Reasoning Conciseness. The reasoning chain should contain no unnecessary complexity, repetition, or backtracking. Error indicators include:

Over-elaborating a straightforward case (e.g., exhaustively evaluating all options when the answer is obvious). Repeating the same evidence or argument across multiple steps.

- • Reversing earlier conclusions without justification. LogicalCoherence.Eachstepshouldprovideclearandcase-specific reasoning, not merely labels or unsupported claims. Error indicators include:

Steps that function only as headings without substantive elaboration.

- • Claims presented without corresponding explanations or evidence.

HallucinationAvoidance.Thereasoningchainshouldaccurately reflect the case information and avoid hallucinations. Error indicators include Introducing facts not mentioned in the case.

Task Understanding. The reasoning chain should correctly follow the task objective and not drift to a different task. The model’s reply shouldn’t address a different task than instructed. For example, if a model misunderstands the counseling strategy formulation task, it may generate counselor utterances instead of selecting an appropriate counseling strategy.

Internal Consistency. The reasoning chain should exhibit no contradictions across steps. Error indicators include:

• Changing conclusions mid-chain without reconciling prior evi-

Later steps contradict earlier interpretations of symptoms, diagnoses, or risk levels.

dence.

#### B Case Study

The detail of case study is illustrated in Figure 3.

###### CognitiveReframing

Situation: In class I feel like others hear my thoughts and judge me. When some things really don't work(for example I have some things for school for tommorow and I can't find how to do that thing after searching hours on internet I get angry and sometimes results to self harm(punching). I sometimes feel like everybody is here just to watch me and see how I react to things(sounds insane). Thought: Am I insane? Q: The most likely cognitive error is?

- 1. Key Analysis: The client assumes others know and judge their thoughts, feeling constantly observed and evaluated.
- 2. Step Reasoning:

- • Client believes they can read others’ minds (“others hear my thoughts and judge me”)
- • They infer judgment without evidence, fitting the definition of mind reading

- 3. Final Answer:

###Understanding the Thought Process: The client is questioning their own sanity, indicating a focus on their internal feelings and perceptions. ###Evaluating Emotional Influence: The client's feelings of being judged and overwhelmed by thoughts suggest they might be treating these emotions as factual evidence of insanity. ###Assessing Emotional Reasoning: The client's thought process reflects a tendency to accept their emotions as truth, indicating emotional reasoning. ###Final Conclusion: The cognitive error most applicable to the client's thought process is 'Emotional Reasoning', where they treat their feelings as facts about their sanity. generated_answer:

[Figure 44]

GPT-o4-mini

Answer: Mind reading

reasoning: The client’s perception of others hearing their thoughts and judging them aligns with the [Mind Reading] cognitive error, where they assume others are aware of their inner thoughts and judgments. Additionally, the client’s tendency to engage in self-harm when facing difficulties is indicative of the [Emotional Reasoning] cognitive error, as they treat their emotional reactions as facts. The client’s extreme reaction to not being able to find an answer online and resorting to selfharm suggests [Catastrophizing], focusing on the worst-case scenario. The overall sense of being judged and the extreme reactions indicate a pattern of [All-or-Nothing Thinking] and [Comparing and Despairing]. generated_answer:

[Figure 45]

Answer: emotional reasoning Mindora

[Figure 46]

Key Analysis: The client believes others are judging them and feels watched, indicating a distorted belief about others' perceptions. Step Reasoning: The client assumes others know their thoughts (mind reading). Feeling judged without evidence aligns with mind reading. Self-harm stems from distress, not the primary cognitive error. Final Answer: Answer: [Mind reading]

Psyche-R1

Answer: [Mind reading]

- 1. Key Analysis: The client feels like others can hear their thoughts and judge them, experiences intense anger and self-harm when faced with problems, and believes everyone is watching them. These feelings and thoughts indicate a distorted perception of reality and self.
- 2. Step Reasoning:

- - The client assumes others can hear their thoughts and judge them, indicating an assumption about others' thoughts.
- - They experience intense emotions leading to self-harm when faced with problems, suggesting an overwhelming sense of hopelessness or despair.
- - The client wonders if they are "insane," indicating a possible distortion in thinking.

- 3. Final Answer: Answer: Mind reading

[Figure 47]

Qwen3-8B

[Figure 48]

- 1."In class I feel like others hear my thoughts and judge me.": This is a classic example of Mind Reading. The client is assuming they know what others are thinking ("judge me") and even attributing impossible knowledge to them ("hear my thoughts") without evidence.
- 2."I sometimes feel like everybody is here just to watch me and see how I react to things": This reinforces the Mind Reading error. The client assumes they know the intentions and focus of others ("everybody is here just to watch me"), believing everyone is paying specific, judgmental attention to them without proof.
- 3."(sounds insane)" / "Am I insane?": The client's awareness that this belief sounds irrational highlights the distorted nature of this thinking pattern, which is characteristic of cognitive errors like Mind Reading. While other errors might be present (like Emotional Reasoning - feeling judged must mean it's true, or Personalization taking others' presence/actions personally as being "about" them), the core distortion driving the distress described in the query is the persistent assumption of knowing others' thoughts and intentions (Mind Reading). Answer: Mind reading Deepseek-R1

LLaMA-4

- 1. Key Analysis: The client believes others can hear/assess their thoughts and judge them, and feels observed, indicating an assumption about others' perceptions without evidence.
- 2. Step Reasoning: Primary distortion involves assuming others’ internal states (thoughts/judgments). Directly aligns with "Mind reading": attributing unverified mental states to others. Other errors (e.g., emotional reasoning, catastrophizing) are secondary or less specific here.
- 3. Final Answer: Answer: [Mind reading]

[Figure 49]

[Figure 50]

Qwen-plus

###### Figure 3: A challenging case of cognitive error identification.

