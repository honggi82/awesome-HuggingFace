# arXiv:2503.13399v1[cs.CV]17Mar2025

## MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research

James Burgess*1 Jeffrey J Nirschl*1 Laura Bravo-S´anchez1,∗ Alejandro Lozano1 Sanket Rajan Gupte1 Jesus G. Galaz-Montoya1 Yuhui Zhang1 Yuchang Su2 Disha Bhowmik3 Zachary Coman3 Sarina M. Hasan4 Alexandra Johannesson5 William D. Leineweber1 Malvika G Nair3 Ridhi Yarlagadda3 Connor Zuraski1 Wah Chiu1 Sarah Cohen3 Jan N. Hansen1 Manuel D Leonetti6 Chad Liu6 Emma Lundberg1,5,6 Serena Yeung-Levy1,6

1Stanford University, 2Tsinghua University, 3University of North Carolina at Chapel Hill, 4Princeton University, 5KTH Royal Institute of Technology, 6Chan Zuckerberg Biohub Network

1 Expert Visual Understanding

What is unusual about the result?

Perception

"How is the Seipin localized withinthe

endoplasmic reticulum (ER)?"

[Figure 1]

"Do cells treated with BafA1 (left) express more p26 compared to control (right)?"

[Figure 2]

2 Hypothesis generation

Why does this happen in my experiment?

Assessment

"Which mechanism might explain why ASLV

particles show signs of merging in a CryoEM image?"

"Which glial cell is likely responsible for

this abnormal reticular ﬁber pattern?"

[Figure 3]

[Figure 4]

3 Experimental proposal

How do I test my hypothesis?

Action

[Figure 5]

"What experiment could you perform to test if CCNB1 protein levels relate to cell cycle stages in human A-431 cells?"

[Figure 6]

"What experimental change can increase

the likelihood of achieving a highresolution structure?"

Figure 1. A scientific experimentation workflow drives discovery: researchers analyze experiments, develop hypotheses, and design further experiments to test their ideas. We release MicroVQA, a visual question answering (VQA) benchmark to test these three tasks in the context of biological microscopy. Each of the 1,042 samples is created by a biology expert, and transformed into a multiple choice question (MCQ).

#### Abstract

Scientific research demands sophisticated reasoning over multimodal data, a challenge especially prevalent in biology. Despite recent advances in multimodal large language models (MLLMs) for AI-assisted research, existing multimodal reasoning benchmarks only target up to collegelevel difficulty, while research-level benchmarks emphasize lower-level perception, falling short of the complex multimodal reasoning needed for scientific discovery. To bridge this gap, we introduce MicroVQA, a visual-question answering (VQA) benchmark designed to assess three reasoning capabilities vital in research workflows: expert image understanding, hypothesis generation, and experiment pro-

Correspondence {jmhb,jnirschl,lmbravo}@stanford.edu ∗These authors contributed equally

posal. MicroVQA consists of 1,042 multiple-choice questions (MCQs) curated by biology experts across diverse microscopy modalities, ensuring VQA samples represent real scientific practice. In constructing the benchmark, we find that standard MCQ generation methods induce language shortcuts, motivating a new two-stage pipeline: an optimized LLM prompt structures question-answer pairs into MCQs; then, an agent-based ‘RefineBot’ updates them to remove shortcuts. Benchmarking on state-of-the-art MLLMs reveal a peak performance of 53%; models with smaller LLMs only slightly underperform top models, suggesting that language-based reasoning is less challenging than multimodal reasoning; and tuning with scientific articles enhances performance. Expert analysis of chain-ofthought responses shows that perception errors are the most frequent, followed by knowledge errors and then overgen-

eralization errors. These insights highlight the challenges in multimodal scientific reasoning, showing MicroVQA is a valuable resource advancing AI-driven biomedical research. MicroVQA is available here, project here.

#### 1. Introduction

The transformative potential of AI in scientific discovery lies in its ability to interpret and reason over complex, multimodal data while integrating specialist domain knowledge. Recently, multimodal large language models (MLLMs) have demonstrated notable capabilities in data comprehension, recall, and multi-step reasoning across various modalities on exam-style questions [1, 3, 6, 49, 69, 81]. This progress has generated enthusiasm for scientific AI applications, ranging from multimodal chat AI assistants [33, 70, 85, 96] to AI agents capable of orchestrating complex experiments [12, 16, 26, 55]. However, while MLLMs perform well on general exam tasks, they often lack the specialized, contextual reasoning needed for real scientific challenges, where tasks require advanced image-based reasoning, analysis, and hypothesis-driven experimentation.

In recent years, efforts to apply scientific MLLMs across a range of domains have laid an essential groundwork. Advances have been achieved in image understanding [41, 43, 47, 50, 57], literature analysis [41, 47, 66, 77], and problemsolving at school and early college levels [53, 87]. In applied settings, AI has facilitated scientific tasks in biology [28, 60, 63, 68], chemistry [12, 38, 55], software [63, 73], and diagnostic pathology and radiology [52]. However, there has been less progress on multimodal tasks requiring reasoning in research-level settings. The gap is due in part to a lack of multimodal benchmarks for scientific reasoning – unlike college-level tasks where existing exams can form benchmarks, research tasks require expert curation.

Microscopy provides a compelling use case for developing such a benchmark. It requires more than pattern recognition or factual recall – it demands the ability to synthesize visual information with experimental context, formulate hypotheses, and propose experimental follow-ups. To evaluate these advanced capabilities, we introduce MicroVQA, a visual-question answering (VQA) benchmark tailored for multimodal scientific reasoning. Grounded in the real-world scientific practice of biological microscopy, MicroVQA incorporates 1,042 multiple-choice questions (MCQs) manually created by biological researchers across diverse disciplines, with each question taking more than 30 minutes to generate. Careful expert curation ensures the benchmark is relevant to real scientific applications.

MicroVQA is designed to evaluate three key reasoning tasks crucial to scientific inquiry: (1) expert image understanding – the ability to recognize scientifically relevant features and integrate contextual information about sample preparation; (2) hypothesis generation – the formulation

of scientifically plausible explanations from experimental data; and (3) experiment proposal – the ability to suggest further experiments that could validate or refute hypotheses. These capabilities form a common workflow in scientific experimentation. They demand both abductive reasoning – inferring the best explanation from multiple possibilities – and deductive reasoning – moving from general principles to specific conclusions [81].

We tested standard methods to map open scientific VQA samples to multiple-choice questions (MCQs), but found that they induce language shortcuts – the MCQs failed to truly test multimodal abilities. To ensure question quality and difficulty, we developed a two-stage pipeline for MCQ generation: first, an optimized LLM prompt structures QA pairs into well-formatted MCQs; then, a novel agent-based ‘RefineBot’ system increases question difficulty by rewriting questions without language shortcuts. This ensures that MicroVQA’s MCQs better test true scientific capabilities, rather than simple pattern recognition.

We benchmarked frontier MLLMs on MicroVQA, finding that the best performers achieved a modest peak of 53%, which shows a significant gap between current models and expert-level scientific reasoning. The variation between different MLLMs was small; notably, smaller LLMs only slightly underperform their larger counterparts, suggesting that the key challenge for our MCQs is not languageonly reasoning, but multimodal reasoning or knowledge. We also find that finetuning MLLMs on scientific publications improves MicroVQA performance. Finally, a group of biomedical researchers performed a detailed qualitative analysis into MLLM failure modes, uncovering three major failure modes: Expert perception is the most common, followed by knowledge errors, and then overgeneralization reasoning errors. This further supports that visual reasoning is a key challenge in MicroVQA.

In summary, our contributions are as follows:

- • We introduce MicroVQA, a benchmark for multimodal reasoning in biological microscopy. We defined three key tasks for scientific research – expert image understanding, hypothesis generation, and experiment proposal. Then, expert researchers curated 1,042 diverse VQA samples.
- • We develop a novel two-stage MCQ generation pipeline to address the challenge of creating MCQs from longformat scientific questions without language shortcuts.
- • We provide quantitative and qualitative benchmarking of frontier MLLMs, highlighting areas for improvement.

#### 2. Related work

MLLM reasoning benchmarks While perception focuses on identifying an image’s visual features, visual reasoning often integrates prior knowledge to derive new conclusions using logical inference strategies like deduction, induction, abduction, and analogical reasoning [81].

Table 1. MicroVQA benchmark attributes.

Reasoning in MLLMs has focused on image-text VQA [25, 44, 53, 81, 87, 88]. The most prominent examples in science are MMMU [87, 88], ScienceQA, [53] and MathVista [54]. While they do test reasoning, they are derived from exams up to the college level, while our benchmark emphasizes research-level difficulty. Other popular multimodal benchmarks test commonsense inference [71, 90], spatial or relational awareness [35, 91] and integrating prior knowledge [56]. Another line of work is visual abductive reasoning [8, 32], or finding the most likely explanation for an observation from incomplete information; this relates closely to hypothesis generation in this work.

Dataset feature Value Total questions 1,042 Multi-image questions 423 Avg. MCQ question length 66 Avg. MCQ answer length 15 Avg. raw question length 158 Avg. raw answer length 52 Unique image sets 255 Image Modalities Light, Fluoro, Electron Image Scales Tissue, Cell, Subcell, Atomic Organisms 31 Research areas 33 Expert question creators 12 Time to create 1 question 30-40 mins Time to quality check 1 MCQ 5 mins

MLLMs in science Many works consider multimodal VQA in scientific or medical domains like microscopy images [50], biological images [57], medical images [30, 34], and chemical structure drawings [43]. Still, they mostly test more straightforward perception and recognition. Figure comprehension for scientific articles does require more reasoning with multimodal data [41, 45–47, 61]; however, the images here are curated for publication, and the emphasis is on understanding content, rather than performing practical tasks. On the other hand, our benchmark includes realistic experiment images and tests more advanced reasoning like hypothesis generation and experiment planning. Scientific tasks have received much more attention in language-only LLMs [9, 15, 22, 37, 41, 58, 62, 63, 66, 67, 73, 80, 95], with GPQA being a notable work targeting PhD-level multi-step reasoning [62], but these do not test multimodal reasoning, which is important to many scientific areas [26]. In terms of models, most MLLMs in the biomedical domain are targeted at clinical tasks [34, 42, 59, 65, 75, 92].

MicroVQA taxonomy

Expert Visual

1 Understanding 2 Hypothesis generation 3 Experimental Proposal

- 2.1 Causal mechanisms
- 2.2 Functional implications

- 3.1 Suggest new experiment
- 3.2 Address technical issues

- 1.1 Compare image sets
- 1.2 Identify abnormalities

E.g., What gene dysregulation could lead to the observed cytoplasm shape?

E.g., What new experiment could I do to test if Gene Y is causing these problems?

E.g., How is the mitochondrial morphology different in image 1 vs image 2?

E.g., Given the unexpected localization of the centrosome, what will be the impact on liver function?

E.g., Are the nuclei unhealthy, and what about the shape tells you that?

E.g., The staining did not target my objective, and the SNR was low. How can I improve it?

Figure 2. MicroVQA taxonomy of sub-tasks.

scientific reasoning by tackling two major evaluation challenges. First, defining tasks is challenging because scientific research encompasses many complex activities [33] – we therefore define three important tasks important to biomedical discovery. Second, VQA samples cannot be readily sourced from existing exams or classification datasets – we therefore assembled a team of biologists to curate challenging reasoning problems. In Sec. 3.1, we provide an overview of the MicroVQA benchmark; Sec. 3.2 details the key tasks; and finally, Sec. 3 demonstrates that MicroVQA is at the forefront of measuring research-level reasoning in MLLMs. In the later Sec. 4, we describe our approach for generating multiple-choice questions.

Multiple-choice question construction A challenge in creating exams for education research and benchmarks in machine learning is generating multiple-choice questions with quality distractors [4, 27]. Recent works have explored LLM-based generation methods, including zero-shot distractor prompting, few-shot prompting, LLM fine-tuning with high-quality examples, and prompting for known error types [10, 23, 24, 94]. However, a major challenge is that generated distractors are not sufficiently vision-centric – MLLMs can often select the correct answer from the distractors without access to the image [74]. While this is partly explained by language biases on truly vision-centric MCQs [7, 29], others likely do fail to test visual understanding due to poor MCQ construction [41] – this informs our RefineBot MCQ generation method in Sec. 4.2.

##### 3.1. Overview of MicroVQA

MicroVQA is a comprehensive dataset of 1,042 VQA triplets, manually curated by a team of expert biological researchers from diverse fields, and key attributes are summarized in Tab. 1. These expert-crafted questions test reasoning across three key tasks: expert understanding, hypothesis generation, and experiment proposal, and we additionally provide a taxonomy of sub-tasks in Sec. 3. The questions cover a broad spectrum, ranging from biological issues, such as “What is unusual about the cell shape?” – to technical imaging challenges like “Is this feature due to low signal-to-noise ratio (SNR) or is it real?”. We provide the ‘raw’ questions and answers written by experts, which

#### 3. The MicroVQA benchmark

Here, we present the MicroVQA benchmark for visualquestion-answering (VQA) which advances multimodal

are often long and nuanced, along with converted multiplechoice questions (MCQs) suitable for MLLM evaluation.

The images encompass the most common microscopy modalities used in human biology: brightfield, fluorescence, and electron microscopy. The sample types span the full range of microscopic scales – tissue, cellular, subcellular, and atomic – and emphasize organisms relevant to human biology and medically motivated tasks, namely human and mouse. Over 60% of the samples have multiple images because comparison is essential in microscopy research, and because multichannel fluorescence images cannot be represented in standard RGB. All these attributes are provided as metadata tags to facilitate deeper error analysis, thereby enhancing MicroVQA’s value as a resource for advancing AI capabilities in scientific research.

##### 3.2. MicroVQA scientific reasoning tasks

To define specific reasoning tasks aimed at advancing biomedical research, we established the following criteria: (1) the tasks should use image-text inputs and text outputs suitable for MLLMs, (2) they should require higher-order reasoning beyond simple image processing, and (3) they should emphasize core experimental activities such as experimentation, analysis, and action [33], rather than auxiliary tasks like literature review or writing. The tasks were developed through interviews with nine co-author PIs and postdocs specializing in microscopy research (Appendix D). We identified three essential capabilities that we now introduce, with examples shown in Sec. 3.

Expert visual understanding After performing an experiment, scientists must interpret data within its experimental context. In biological microscopy, this involves identifying patterns like protein distributions or cell morphologies, while also assessing technical aspects such as artifacts and noise levels. The task requires MLLMs to perform anomaly detection and image comparison, going beyond simple perception in two key ways: analysis must consider sample preparation context, and expert knowledge is needed to evaluate biological features and technical artifacts.

Hypothesis generation The next step is proposing mechanistic hypotheses to explain experimental data. For example, when a drug treatment causes mitochondrial fragmentation in cells, a hypothesis might be that the drug disrupted calcium homeostasis, activating the DRP1 protein through a multi-step pathway, leading to fragmentation. This requires abductive reasoning [8, 20, 32, 81] as one must select from many possible hypotheses given incomplete information. The process demands the MLLM to integrate relevant domain knowledge with experimental context and visual features to reason about which hypothesis best explains the observations.

Table 2. Comparing scientific multimodal benchmarks close to MicroVQA for in terms of reasoning level or difficulty beyond college level. We show difficulty level, scientific domain, dataset source, and size. Compared to others, MicroVQA either has higher difficulty level, or it has higher reasoning level (which is established in Fig. 3). Compared to others at the same reasoning level, namely MMMU-Pro, it has similar size.

Difficulty level Domain Source Size

Benchmark

MicroVQA research microscopy expert-curated 1,042 MMSci [47] research science paper figures 7132 LabBench [41] research biology webQA 181 PathVQA [30] graduate pathology texbooks 16.3k OmnimedVQA∗ [34] graduate medical classification dataset 127.9k Microbench [50] graduate microscopy classification dataset 17.2k MMMU [87] undergraduate general textbooks, webQA 11k MMMU Pro [88] undergraduate general MCQ dataset 1,730 Science QA [53] grade-school science exams 16.8k

Bloom’s Levels

5 - Evaluation 4 - Analysis 3 - Application 2 - Comprehension 1 - Recall

E.g. Which tissue sample appears healthier betweeen the two images?

E.g. Which experiment would conﬁrm the localization of the target protein to motile cilia in ependymal cells?

E.g. In the image, two cells are connected by a redcolored structure. What is this structure called?

Question Distribution of Bloom's Levels Across Benchmarks

|LabBench<br><br>MicroVQA<br><br>MMSci<br><br>2<br><br>2|41<br><br>.5<br><br>.9<br><br>8.3|.4<br><br>48.2|24.3<br><br>87.0|24<br><br>32.1|.3<br><br>9.2<br><br>8.8<br><br>1<br><br>8.5|.0 1.1<br><br>0.3|
|---|---|---|---|---|---|---|
|MicroBench (Perception)<br><br>OmniMed VQA<br><br>PathVQA| |48.4<br><br>71.9<br><br>71.1| |51.6|23.4<br><br>23.9<br><br>2.4<br><br>3.3 1|1.8<br><br>.2<br><br>0.5<br><br>0.5|
|MMMU<br><br>MMMU-Pro|32.9<br><br>21.3|26.5|45.8|45.5|14.9 6.0<br><br>5.9|0.4<br><br>0.8|
|ScienceQA<br><br>school|4|4.3| |40.9|9.7 4.6|0.5|

school Undergrad.ResearchGraduate

Grade-

0.0 20.0 40.0 60.0 80.0 100.0 Proportion of dataset (% )

Figure 3. Composition of scientific MLLM benchmarks regarding estimated Bloom’s taxonomy [11]. Higher levels are more cognitively challenging. MicroVQA has more questions at higher levels compared to other benchmarks, for example, MMMU [87] and ScienceQA [53], while perception-driven medical benchmarks like OmniMedVQA are at lower levels.

Experiment proposal The final step is designing experiments to validate hypotheses. In microscopy, this often involves selecting appropriate assays and controls, requiring knowledge of suitable protocols and deductive reasoning about whether proposed experiments will provide evidence for or against the hypothesis. This task also may involve adjusting a particular experiment to address technical issues, like how to ensure a good signal-to-noise ratio,

and this requires both image understanding and knowledge of protocols.

In Appendix F.7.1, we show an example question in each task with an expert-annotated ‘reasoning trace’ – it shows the type of reasoning required for typical questions.

- 3.3. Analysis of MicroVQA benchmark

Scientific discovery requires complex reasoning beyond basic perception and knowledge, and it requires expertise beyond school-level education. Here we show that MicroVQA addresses a gap that prior multimodal scientific benchmarks do not test high level reasoning at researchlevel difficulty.

To assess reasoning levels in MLLM benchmarks, we use Bloom’s taxonomy [11], which hierarchically classifies cognitive difficulty from recall to evaluation. We fine-tune GPT-4o to classify Bloom’s taxonomy on related MLLM benchmarks with results in Fig. 3; in Tab. 2 we additionally show key attributes of the same benchmarks. Fig. 3 shows that benchmarks having research-level and graduatelevel difficulty are dominated by lower reasoning questions (levels 1-2), while our MicroVQA has higher level reasoning (levels 3-4). This is not surprising since OmnimedVQA and MicroBench derive questions from prior classification datasets, while our task definition and data collection approach does reflect high-level reasoning.

On the other hand, some benchmarks have higher Bloom’s reasoning level closer to MicroVQA – MMMU, MMMU-Pro, and Science-QA – but their difficulty level is lower – undergraduate or grade-school. The most comparable benchmark for reasoning is the undergraduate-level MMMU-Pro, and its dataset size is 1,730 compared to our 1,042: this shows that for reasoning-intensive benchmarks, sizes beyond 1,000 are very high.

- 4. MCQ generation process

- 4.1. Creation of raw VQA samples by experts

Expert generation of raw VQA samples Creating VQA triples suitable for research-level reasoning tasks is timeconsuming and requires expertise. We recruited 12 human experts to each create approximately 90 VQA samples, taking about 30–40 minutes per sample. Each submission included an image set, a question, a detailed response, and contextual information like experimental context and motivation. They are raw VQA triples, (v0,q0,a0) – ‘raw’ means the question and answer can be much longer and more detailed than typical VQA benchmarks. This allowed expert annotators to include enough detail to match what is a valuable input and response for an MLLM. Appendix E shows materials that prompt question creators. The images were sourced from the contributor’s own experiments, image databases, or research papers published since 2024, and

all image licenses permit redistribution.

Quality control We had three quality control criteria: questions should be challenging and not solvable with undergraduate-level knowledge; they should be aligned with the defined tasks; and there should be diversity in image content. Questions were reviewed against these criteria for each contributor, and feedback was provided for further VQA samples. These review rounds occurred after submitting the first 10 questions, and the first 40 questions.

##### 4.2. MCQ generation from raw VQA samples

To benchmark MLLMs, we transform raw expert-created VQA samples to multiple-choice questions (MCQs) and propose new methods for MCQ generation.

Motivation: naive MCQ generation is inadequate The standard approach for VQA benchmarks is to zero-shot prompt an LLM with raw VQA samples [4], but this has two problems. The first is that generated MCQs do not follow established educational MCQ design principles (Appendix E.2), likely because the raw VQA questions and answers are often long and variable. Our stage 1 ‘exam alignment’, which we describe in the next section, addresses this challenge. The second challenge is that generated MCQs do not properly test MLLM capabilities. Evaluating GPT4o on these MCQs scores 93% even without the image, despite most of the dataset questions clearly requiring image analysis, and we investigate this by inspecting the chainof-thought outputs in Fig. 4 and Appendix E.2. Many distractors can be easily eliminated based on general biology knowledge, or because they are too vague compared to the correct option. We hypothesize that MCQ and distractor generation is challenging when the target answer string is long, and where the subject matter is specialized, suggesting that other scientific benchmarks may encounter similar issues. Overall, this motivates a more involved approach to distractor generation.

Stage 1: Exam-style MCQ generation Here we ensured that multiple-choice questions (MCQs) conformed to established design principles for biomedical exams (Fig. 4). This effort was led by a physician-scientist co-author trained in medical board exam item writing, who reviewed educational literature on MCQ design [64] and Bloom’s Taxonomy for assessing cognitive skills in biomedical exams [5, 19, 89]. We began by manually transforming 50 usersubmitted raw question-answer pairs, (q0,a0), into onebest-answer MCQs with k distractors, (q1,a1,d1). These ‘Gold Standard’ MCQs retained the original questions’ meanings while strictly adhering to NBME standards [64] and minimizing cues that enable test-wise guessing. We used these gold-standard MCQs in a supervised learning

###### 1 2 Agent-based MCQ refinement

###### 0 Raw VQA creation Exam-style MCQ generation

[Figure 7]

[Figure 8]

[Figure 9]

RefineBot

Define taxonomy with experts

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

###### +

Exam-style refinement

[Figure 18]

Evaluator + Reflector

[Figure 19]

[Figure 20]

Hard

[Figure 21]

[Figure 22]

Question collection

[Figure 23]

Raw dataset MCQ1

Easy

Expert review

MCQ {1,1*} MCQ2

[Figure 24]

Review and feedback

[Figure 25]

Optimized prompt

Feedback

[Figure 26]

Rewriter

Supervised prompt optimization (DSPy) Gold standard Optimized

[Figure 27]

[Figure 28]

Gold standard

[Figure 29]

Context

Generated Metrics

Raw dataset

Checker

MCQ1*

[Figure 30]

[Figure 31]

prompt

[Figure 32]

[Figure 33]

Estimated expert time: ~600 h

Removes unnecessary detail

MCQ stage 1 MCQ stage 2

###### Raw question

Unnecessary detail

Underspecified compared to answer

Question: “A fluorescence microscopy image showing eukaryotic cells with a target protein highlighted in the green

Question: “A fluorescence microscopy image of SiHa cells incubated with an antibody against a target protein (green channel) was analyzed. Based on the image, what can be inferred about the subcellular localization and expression levels of the target protein?

Question: The image shows SiHa cells that have been incubated with an antibody against a target protein (green channel) and imaged using fluorescence microscopy. What can you tell about the subcellular location and expression levels of the target protein based on this image? Answer: “The target protein is mainly expressed in the nucleus, and to a lesser degree, the cytoplasm. Its expression appears to differ between the cells, as there is single-cell variation in the signal intensity both in the nucleus and cytosol.

channel was reviewed. From the visual data, what can be inferred about the localization and expression levels of the protein within these cells?

Hard distractor: Preserves QA objective plausible and subtle

- A) The target protein is primarily observed in the nuclear region with limited cytoplasmic presence, with fluctuations in expression between cells.
- B) The protein exhibits predominant localization across the cell surface, maintaining uniform intensity in all cells.
- C) A notable cytoplasmic accumulation of the protein with consistent signal levels is observed across all cells.
- D) The protein is detected evenly throughout both the cytoplasm and nucleus but shows significant variability in the cellular signal intensity.
- E) The fluorescence strongly indicates localization in the cytoplasmic matrix, with a concurrent uniformity in expression intensity among the observed cells. Correct answer: A

- A) The target protein is exclusively localized in the nucleus with uniform expression across all cells.
- B) The target protein is predominantly expressed in the cytoplasm with variable expression levels across cells.
- C) The target protein is mainly expressed in the nucleus and less in the cytoplasm, with single-cell variation.
- D) The target protein is evenly distributed throughout the cell with consistent expression levels in all cells. Correct answer: C

Long answer

Easy distractor: too vague

Hard distractor

Answer more concise

Justification to answer

- Figure 4. Constructing the MicroVQA multiple choice questions. (0) We defined tasks with domain biological scientists and created 1,061 raw VQA samples. (1) The raw samples were aligned to an exam-style MCQ by manually transforming a small set and optimizing an LLM prompt to match that alignment. (2) MCQs are further improved using RefineBot, a new iterative method to make MCQs more challenging. The lower panel shows an example MCQ from raw VQA to final: the annotations highlight key changes that we further explore in Appendix E.2, where red indicates issues, and green indicates good attributes.

setup to develop a general LLM prompt, p, that maps raw question-answer pairs to exam-style MCQs. We leverage the DSPy framework [39, 40], which optimizes p to match the gold-standard MCQ outputs. Additionally, the optimization process incorporated supervision from LLMbased quality metrics: content similarity, NBME-aligned formatting, and the absence of extraneous clues. Further details are in Appendix E.2.

does not not occur inside the virus, and so does not match the question. It passes the reflection to the rewriter LLM agent. The rewriter revises the question stem and generates new distractors to invalidate the identified strategies: in this example it may create distractors referring to processes that do occur inside a virus. To prevent significant changes to the question-answer pair over iterations, we employ an LLM checker to ensure semantic consistency with the original pair. The revised MCQ is returned to the evaluator; if it still answers correctly, the loop continues. The process stops if the rewriter fails the checker or after n iterations. If it fails, RefineBot can be rerun with a different seed, often succeeding—a form of inference-time scaling [13]. All agents are GPT-4o-0806, but to mitigate potential bias against 4o in final evaluation, we also use Claude-3.5Sonnet-0620 as the evaluator agent; we assess biases in the experiments section.

Stage 2: RefineBot question refinement Although Stage 1 produces exam-aligned MCQs, many remain easily solvable due to language shortcuts (see Appendix D). To enhance difficulty, we introduce RefineBot, a method that increases MCQ complexity (Fig. 4). The key idea is that weaknesses in MCQ construction are revealed by the chainof-thought (CoT) answers. The first LLM agent, the evaluator/reflector, answers the MCQ with CoT and then reflects on the strategies used. For example, a question about processes inside a virus might have a distractor referring to surface processes – this can be eliminated without referencing the image. The LLM summarizes its solution strategy into a reflection text: in this case, that the distractor was implausible based on knowledge that the stated process

Final MCQ quality check Since MCQ generation involves LLM processing, we need to verify the correctness of the final answer. Each MCQ is manually reviewed by the same expert who created that particular question (between 80 and 120 per person). For any question with issues, that expert makes minimal changes to the question correct it.

#### 5. Experiments

##### 5.1. Benchmarking MLLMs with MicroVQA

We evaluate state-of-the art multimodal large language models (MLLMs) on the MicroVQA benchmark. We include open and closed models spanning the categories: reasoning, large, small, and medical – medical is the closest domain to microscopy with specialist MLLMs. We utilize standard chain-of-thought prompting [88] (deatails in Appendix F). We report mean accuracy in multiple-choice VQA for each tasks Tab. 3. We also perform and analyze no-image Appendix F.5.

MicroVQA is challenging for all MLLMs Our evaluation reveals a substantial gap between current MLLMs and the upper bound, with the leading model, o1, at 52.8. There is surprisingly little variation in performance between models, with most closed and open models all scoring above 40, with the lowest score (Llama-3.2-11b) likely due to instruction-following issues. A human baseline (Appendix F.3) scores only 50.3 – this shows that biology experts are specialized to subdomains, and are challenged by different subdomains. In fact this demonstrates that experts could benefit from MLLMs that can solve MicroVQA’s tasks. The performance variation across the 3 tasks is small – the highest score is 56.4 on ‘expert visual understanding (V)’, compared to the highest overall of 52.8. The ‘hypothesis generation’ task is the hardest for all models and the gap is strongest for smaller models. For example the gap between hypothesis generation and expert visual understanding is 3.7 for the strongest large model, but 10.7 on the strongest small model.

Smaller models are surprisingly competitive For all models with a lower-parameter equivalent – Gemini-1.5, QwenVL, VILA, and Llama – the drop in performance due to size is less than 3 for all except Llama. Moreover, these drops are smaller than on other multimodal reasoning benchmarks like MMMU, where for example, Gemini drops by 9.6 between Pro and Flash-8B [87]. Typically, smaller models have the same vision encoder, but a smaller LLM [79]. Since the size of the language model has a small impact, we hypothesize that for the ‘solved’ part of MicroVQA, the language-only reasoning is relatively simple compared to domains like math that require multi-step logic. This suggests that other aspects of MLLM reasoning are more challenging, like multimodal reasoning. If true, this suggests that future work could focus on stronger image representations, and this idea is supported by the qualitative error analysis in Sec. 5.2.

Specialist biomedical training does improve performance While no specialist MLLMs have been developed

- Table 3. VQA accuracy on MicroVQA by task: expert visual understanding (V), hypothesis generation (H), experiment proposal (E). Models marked ∗ were used in MCQ generation, which may affect comparative performance (see Sec. 5.1). The model † is the base LlaVA for LLaVA-Med. Best values are bolded.

Model Overall V H E

R

o1 [36] 52.8 55.4 50.2 53.0

largemodels

∗Claude-3.5-Sonnet [6] 51.7 54.1 50.2 50.4 Gemini-Pro-1.5 [69] 51.1 52.0 50.2 50.9 Pixtral-Large [2] 49.8 50.8 49.5 48.7 Grok-2-Vision [84] 48.4 50.3 46.4 48.7 Qwen-2-vl-72b-Instruct [79] 47.5 49.2 45.7 47.8 VILA1.5-40b [48] 47.5 47.2 47.9 47.4 ∗GPT-4o [1] 45.6 48.7 43.1 44.8

- Llama-3.1-Nemotron-70b-Instruct [83] 44.2 44.9 43.3 44.8
- Llama-3.2-90b-Vision-Instruct [21] 42.4 44.9 42.1 38.7

smallmodels

Qwen-2-VL-7b [79] 48.8 54.1 43.3 49.6 Claude-3.5-Haiku [6] 47.1 48.0 43.8 51.7 Gemini-Flash-1.5-8b [69] 46.7 48.7 43.6 49.1 GPT-4o-mini [1] 46.2 48.5 43.6 47.0 Pixtral-12b [2] 45.6 46.9 44.8 44.8 VILA1.5-13b [48] 41.8 41.8 47.5 40.9 Llama-3.2-11b-vision-instruct [21] 30.3 32.4 29.3 28.7

medical

LLaVA-Med-Mistral-7B [42] 43.0 37.3 47.1 41.6 †LLaVA-Mistral-7B [49] 39.8 31.6 43.1 37.1 Random 22.0 21.9 21.8 21.9 Human 50.3 52.7 47.5 51.4

- Table 4. Ablation study on MicroVQA MCQ generation stages (shown in Fig. 4). Accuracy is high because MCQs have shortcuts (Sec. 4.1) after ‘Stage 1’ exam alignment, but is lower after ‘Stage 2’. Final column is the relative decrease in accuracy. Models with ∗ were used in Stage 2 generation and have the biggest accuracy drops (bolded). They are grouped with different models from the same provider in ∗∗.

Stage 1 Stage 2 Relative drop

∗GPT-4o 79.7 46.8 -41.2 ∗∗GPT-4o-mini 75.6 46.2 -39.0 ∗Claude-3.5-Sonnet 82.2 51.7 -37.1 ∗∗Claude-3.5-Haiku 77.3 47.1 -39.0 o1 81.6 52.8 -35.3 Pixtral-Large 80.1 49.8 -37.8 Gemini-Pro-1.5 79.1 51.1 -35.4

for microscopy research in particular, LLaVA-Med is finetuned on scientific articles from PubMed that include data types overlapping with MicroVQA – especially (tissue) pathology. In Tab. 3, we compare LLaVA-Med against it’s base model, LLaVA-Mistral-7B and find overall stronger performance by 4.5 points. This suggests two clear opportunities for improving MicroVQA performance. The first is to instruct-tune with a dataset that better aligns with the MicroVQA data domain: not only tissue pathology, but also fluorescence and electron microscopy, which is available in scientific articles [93]. The second is to simply begin training from a stronger open model like Pixtral. Beyond that, there is more to explore in MLLM specialization, such as tuning the base image encoder for microscopy [51].

Models

Gemini-1.5-Pro VILA1.5-40B LlaVA-Med-Mistral-7B Sub-task Taxonomy Bloom’s Level

[Figure 34]

[Figure 35]

- 1.1 Compare image sets
- 1.2 Identify

- 2

- 3

- 4

- 5

abnormalities

- 2.1 Causal

mechanisms

- 2.2 Functional implications

- 3.1 Suggest experiment
- 3.2 Address

technical issue

Accuracy Accuracy

- Figure 5. Performance by sub-task and Bloom’s level for best models: Gemini-1.5-Pro (closed source), VILA1.5-40B (opensource), and LlaVA-Med-Mistral-7B (medical).

RefineBot MCQ generation is very effective but introduces small model biases We introduced a new method for refining MCQs to make them more challenging – RefineBot. Tab. 4 shows that between stage 1 and stage 2 (before and after RefineBot), the relative accuracy for all models drops by between 35% and 42%. Firstly, this validates that RefineBot is a valuable tool for making benchmarks harder. However, the most significant drops are for models used by RefineBot (GPT-4o & Claude-3.5-Sonnet, and their smaller versions, GPT-4o-mini & Claude-3-Haiku). The adversarial process introduced a small bias against these models (and we hypothesise the bias would be higher if using only one model instead of two). Despite making fair evaluation slightly more challenging, our use of frontier LLMs in the refinement process is well-motivated, as these models are best positioned to generate challenging, high-quality MCQs that probe the boundaries of current capabilities. Moreover, the bias seems modest – less than 10 points of relative drop – while the benefit is a huge increase in difficulty across all test MLLMs, as seen in Tab. 4.

Metadata-based analysis supports deeper error analysis Fig. 5 shows error rates based on attribute tag. In Appendix F we explore results further – namely that multiimage reasoning has surprisingly good performance, higher level Bloom’s questions are harder, and (not shown) rare image modalities perform worse.

##### 5.2. Error analysis on MicroVQA

To understand model errors, three biomedical experts reviewed the Chain-of-Thought (CoT) responses of 30 samples on Claude 3.5 Sonnet, with each review requiring at least 45 mins of careful analysis. Errors were classified into a major category and minor contributing errors were also

noted. Expert perception accounted for 50% of errors, 30% were knowledge errors, and 13% were due to overgeneralization; the final 7% were text hallucination or general reasoning error. Appendix F.6 shows examples for the most common types. We expand on this analysis with an automated version across the complete dataset in Appendix F.7.

Expert perception errors occurred when the model misinterpreted visual features, leading to incorrect inferences. For example, in Fig. 13, the model judges the small electron-dense objects as ribosomes, rather than stress granules – ribosomes tend to be smaller, have a more regular shape, and lower contrast. Language bias may have contributed to the preference for ribosomes since they are very commonly studied in EM literature available to LLM training. Overall, perception errors were the dominant error type (50%), which suggests that future methods should improve vision representations in MLLMs for microscopy.

Knowledge errors highlight gaps in nuanced biomedical knowledge. One example in Fig. 17 deals with understanding how protein localization changes during cell signaling. An experiment gives information about a transmembrane signaling protein, and the question asks to interpret the image and explain the function in these cells. The model correctly perceives a punctate distribution, but assumes the role of coordinating intracellular vesicle trafficking. This represents a shallow interpretation that could have been improved with specialized biomedical knowledge. These errors could be mitigated by improving knowledge in MLLMs, either in training or as a database at inference.

Overgeneralization errors (and simplification errors) reflect the model’s tendency to apply broad scientific principles without regard for specific context – arguably they are reasoning errors. In one example in Fig. 21, the question is to compare the risk of malignancy of a tumor, schwannoma, to other nerve sheath tumors. During reasoning, the model proposes a simplified question – what is true about schwannoma – which it answers while ignoring the comparison.

#### 6. Conclusion

MLLMs hold great potential for advancing scientific discovery. We aim to inspire the development of broadly applicable scientific AI systems. Future directions include training models for more robust image representations, integrating knowledge bases, exploring open evaluation methods, and studying LLMs’ reasoning over hypotheses. Beyond microscopy, we hope our benchmark serves as a blueprint for AI evaluation in expert-driven domains. More generally we aspire for different application domains – biomedicine, chemistry, materials science – to share methods towards the common goal of building scientific AI systems.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

- 2023. 2, 7

[2] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Devendra Chaplot, Jessica Chudnovsky, Saurabh Garg, Theophile Gervet, Soham Ghosh, Am´elie H´eliou, Paul Jacob, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073,

- 2024. 7

- [3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736,

2022. 2

- [4] Elaf Alhazmi, Quan Z Sheng, Wei Emma Zhang, Munazza Zaib, and Ahoud Alhazmi. Distractor generation for multiple-choice questions: A survey of methods, datasets, and evaluation. arXiv preprint arXiv:2402.01512, 2024. 3, 5
- [5] Lorin W. Anderson and David R. Krathwohl, editors. A Taxonomy for Learning, Teaching, and Assessing. A Revision of Bloom’s Taxonomy of Educational Objectives. Allyn & Bacon, New York, 2 edition, 2001. 5, 8
- [6] Anthropic. The claude 3 model family: Opus, sonnet, haiku. Model card, Anthropic, 2024. 2, 7
- [7] Nishant Balepur, Abhilasha Ravichander, and Rachel Rudinger. Artifacts or abduction: How do llms answer multiple-choice questions without the question? arXiv preprint arXiv:2402.12483, 2024. 3, 18
- [8] Chandra Bhagavatula, Ronan Le Bras, Chaitanya Malaviya, Keisuke Sakaguchi, Ari Holtzman, Hannah Rashkin, Doug Downey, Scott Wen-tau Yih, and Yejin Choi. Abductive commonsense reasoning. arXiv preprint arXiv:1908.05739,

2019. 3, 4

- [9] Owen Bianchi, Maya Willey, Chelsea X Alvarado, Benjamin Danek, Marzieh Khani, Nicole Kuznetsov, Anant Dadu, Syed Shah, Mathew J Koretsky, Mary B Makarious, et al. Cardbiomedbench: A benchmark for evaluating large language model performance in biomedical research. bioRxiv, pages 2025–01, 2025. 3
- [10] Semere Kiros Bitew, Johannes Deleu, Chris Develder, and Thomas Demeester. Distractor generation for multiplechoice questions with predictive prompting and large language models. arXiv preprint arXiv:2307.16338, 2023. 3
- [11] Benjamin S. Bloom. Taxonomy of Educational Objectives: The Classification of Educational Goals. Handbook I: The Cognitive Domain. David McKay Co Inc, New York, 1956. 4, 5
- [12] Daniil A Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, 2023. 2
- [13] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher R´e, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with

- repeated sampling. arXiv preprint arXiv:2407.21787, 2024. 6
- [14] James Burgess, Jeffrey J Nirschl, Maria-Clara Zanellati, Alejandro Lozano, Sarah Cohen, and Serena Yeung-Levy. Orientation-invariant autoencoders learn robust representations for shape profiling of cells and organelles. Nature Communications, 15(1):1022, 2024. 7
- [15] Hengxing Cai, Xiaochen Cai, Junhan Chang, Sihang Li, Lin Yao, Changxin Wang, Zhifeng Gao, Yongge Li, Mujie Lin, Shuwen Yang, et al. Sciassess: Benchmarking llm proficiency in scientific literature analysis. arXiv preprint arXiv:2403.01976, 2024. 3
- [16] Anne E Carpenter, Beth A Cimini, and Kevin W Eliceiri. Smart microscopes of the future. nature methods, 20(7):962– 964, 2023. 2
- [17] Nathan H Cho, Keith C Cheveralls, Andreas-David Brunner, Kibeom Kim, Andr´e C Michaelis, Preethi Raghavan, Hirofumi Kobayashi, Laura Savy, Jason Y Li, Hera Canaj, et al. Opencell: Endogenous tagging for the cartography of human cellular organization. Science, 375(6585):eabi6983, 2022. 7
- [18] Kenneth Clark, Bruce Vendt, Kirk Smith, John Freymann, Justin Kirby, Paul Koppel, Stephen Moore, Stanley Phillips, David Maffitt, Michael Pringle, et al. The cancer imaging archive (tcia): maintaining and operating a public information repository. Journal of digital imaging, 26:1045–1057,

2013. 7

- [19] Alison Crowe, Clarissa Dirks, and Mary Pat Wenderoth. Biology in bloom: implementing bloom’s taxonomy to enhance student learning in biology. CBE Life Sci. Educ., 7

(4):368–381, 2008. 5, 8

- [20] Igor Douven. Abduction, 2011. 4
- [21] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 7

- [22] Kehua Feng, Keyan Ding, Weijie Wang, Xiang Zhuang, Zeyuan Wang, Ming Qin, Yu Zhao, Jianhua Yao, Qiang Zhang, and Huajun Chen. Sciknoweval: Evaluating multilevel scientific knowledge of large language models. arXiv preprint arXiv:2406.09098, 2024. 3
- [23] Wanyong Feng, Jaewook Lee, Hunter McNichols, Alexander Scarlatos, Digory Smith, Simon Woodhead, Nancy Otero Ornelas, and Andrew Lan. Exploring automated distractor generation for math multiple-choice questions via large language models. arXiv preprint arXiv:2404.02124, 2024. 3
- [24] Nigel Fernandez, Alexander Scarlatos, Simon Woodhead, and Andrew Lan. Divert: Distractor generation with variational errors represented as text for math multiple-choice questions. arXiv preprint arXiv:2406.19356, 2024. 3
- [25] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 3
- [26] Shanghua Gao, Ada Fang, Yepeng Huang, Valentina Giunchiglia, Ayush Noori, Jonathan Richard Schwarz, Yasha

- Ektefaie, Jovana Kondic, and Marinka Zitnik. Empowering biomedical discovery with ai agents. arXiv preprint arXiv:2404.02831, 2024. 2, 3
- [27] Mark J Gierl, Okan Bulut, Qi Guo, and Xinxin Zhang. Developing, analyzing, and using distractors for multiplechoice tests in education: A comprehensive review. Review of educational research, 87(6):1082–1116, 2017. 3, 8
- [28] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, et al. Towards an ai co-scientist. arXiv preprint arXiv:2502.18864, 2025. 2
- [29] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 3, 13, 18
- [30] Xuehai He, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Pathvqa: 30000+ questions for medical visual question answering. arXiv preprint arXiv:2003.10286, 2020. 3, 4
- [31] Marco Y Hein, Duo Peng, Verina Todorova, Frank McCarthy, Kibeom Kim, Chad Liu, Laura Savy, Camille Januel, Rodrigo Baltazar-Nunez, Madhurya Sekhar, et al. Global organelle profiling reveals subcellular localization and remodeling at proteome scale. Cell, 188(4):1137–1155, 2025. 7
- [32] Jack Hessel, Jena D Hwang, Jae Sung Park, Rowan Zellers, Chandra Bhagavatula, Anna Rohrbach, Kate Saenko, and Yejin Choi. The abduction of sherlock holmes: A dataset for visual abductive reasoning. In European Conference on Computer Vision, pages 558–575. Springer, 2022. 3, 4
- [33] Tom Hope, Doug Downey, Daniel S Weld, Oren Etzioni, and Eric Horvitz. A computational inflection for scientific discovery. Communications of the ACM, 66(8):62–73, 2023. 2,

- 3, 4, 6

[34] Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. Omnimedvqa: A new large-scale comprehensive evaluation benchmark for medical lvlm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22170–22183, 2024. 3,

- 4, 2, 7

- [35] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 3
- [36] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 7
- [37] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. arXiv preprint arXiv:1909.06146, 2019. 3
- [38] Yeonghun Kang and Jihan Kim. ChatMOF: an artificial intelligence system for predicting and generating metal-organic frameworks using large language models. Nat. Commun., 15

(1):4705, 2024. 2

- [39] Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive NLP. arXiv preprint arXiv:2212.14024, 2022. 6
- [40] Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714,

2023. 6

- [41] Jon M Laurent, Joseph D Janizek, Michael Ruzo, Michaela M Hinks, Michael J Hammerling, Siddharth Narayanan, Manvitha Ponnapati, Andrew D White, and Samuel G Rodriques. Lab-bench: Measuring capabilities of language models for biology research. arXiv preprint arXiv:2407.10362, 2024. 2, 3, 4
- [42] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36, 2024. 3, 7
- [43] Junxian Li, Di Zhang, Xunzhi Wang, Zeying Hao, Jingdi Lei, Qian Tan, Cai Zhou, Wei Liu, Yaotian Yang, Xinrui Xiong, et al. Chemvlm: Exploring the power of multimodal large language models in chemistry area. CoRR, 2024. 2, 3
- [44] Lin Li, Guikun Chen, Hanrong Shi, Jun Xiao, and Long Chen. A survey on multimodal benchmarks: In the era of large ai models. arXiv preprint arXiv:2409.18142, 2024. 3
- [45] Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models. arXiv preprint arXiv:2403.00231, 2024. 3
- [46] Shengzhi Li and Nima Tajbakhsh. Scigraphqa: A large-scale synthetic multi-turn question-answering dataset for scientific graphs. arXiv preprint arXiv:2308.03349, 2023.
- [47] Zekun Li, Xianjun Yang, Kyuri Choi, Wanrong Zhu, Ryan Hsieh, HyeonJung Kim, Jin Hyuk Lim, Sungyoung Ji, Byungju Lee, Xifeng Yan, et al. Mmsci: A multimodal multi-discipline dataset for phd-level scientific comprehension. arXiv preprint arXiv:2407.04903, 2024. 2, 3, 4
- [48] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699, 2024. 7
- [49] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 2, 7
- [50] Alejandro Lozano, Jeffrey Nirschl, James Burgess, Sanket Rajan Gupte, Yuhui Zhang, Alyssa Unell, and Serena Yeung. Micro-bench: A microscopy benchmark for visionlanguage understanding. Advances in Neural Information Processing Systems, 37:30670–30685, 2024. 2, 3, 4
- [51] Alejandro Lozano, Min Woo Sun, James Burgess, Liangyu Chen, Jeffrey J Nirschl, Jeffrey Gu, Ivan Lopez, Josiah

- Aklilu, Austin Wolfgang Katzer, Collin Chiu, et al. Biomedica: An open biomedical image-caption archive, dataset, and vision-language models derived from scientific literature. arXiv preprint arXiv:2501.07171, 2025. 7
- [52] Ming Y ”Lu, Bowen Chen, Drew F K Williamson, Richard J Chen, Melissa Zhao, Aaron K Chow, Kenji Ikemura, Ahrong Kim, Dimitra Pouli, Ankush Patel, Amr Soliman, Chengkuan Chen, Tong Ding, Judy J Wang, Georg Gerber, Ivy Liang, Long Phi Le, Anil V Parwani, Luca L Weishaupt, and Faisal” Mahmood. ”a multimodal generative AI copilot for human pathology”. ”Nature”, 634(8033):”466–473”,

2024. 2

- [53] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521,

2022. 2, 3, 4

- [54] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023. 3
- [55] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature Machine Intelligence, pages 1–11, 2024. 2
- [56] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 3
- [57] M Maruf, Arka Daw, Kazi Sajeed Mehrab, Harish Babu Manogaran, Abhilash Neog, Medha Sawhney, Mridul Khurana, James P Balhoff, Yasin Bakis, Bahadir Altintas, et al. Vlm4bio: A benchmark dataset to evaluate pretrained visionlanguage models for trait discovery from biological images. arXiv preprint arXiv:2408.16176, 2024. 2, 3
- [58] Adrian Mirza, Nawaf Alampara, Sreekanth Kunchapu, Benedict Emoekabu, Aswanth Krishnan, Mara Wilhelmi, Macjonathan Okereke, Juliane Eberhardt, Amir Mohammad Elahi, Maximilian Greiner, et al. Are large language models superhuman chemists? arXiv preprint arXiv:2404.01475,

2024. 3

- [59] Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265, 2023. 3
- [60] Siddharth Narayanan, James D Braza, Ryan-Rhys Griffiths, Manu Ponnapati, Albert Bou, Jon Laurent, Ori Kabeli, Geemi Wellawatte, Sam Cox, Samuel G Rodriques, et al. Aviary: training language agents on challenging scientific tasks. arXiv preprint arXiv:2412.21154, 2024. 2
- [61] Shraman Pramanick, Rama Chellappa, and Subhashini Venugopalan. Spiqa: A dataset for multimodal question answering on scientific papers. arXiv preprint arXiv:2407.09413, 2024. 3

- [62] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. arXiv preprint arXiv:2311.12022,

2023. 3

- [63] Lo¨ıc A Royer. Omega—harnessing the power of large language models for bioimage analysis. nature methods, 21(8): 1371–1373, 2024. 2, 3
- [64] Billings Melissa S, DeRuchie Kristine, Hussie Kieran, Kulesher Allison, Merrell Jacquelyn, Morales Amy, Paniagua Miguel A, Sherlock Jennifer, Swygert Kimberly, and Tyson Julie. Nbme item-writing guide: Constructing written test questions for the health sciences, 2021. 5, 8
- [65] Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, et al. Capabilities of gemini models in medicine. arXiv preprint arXiv:2404.18416,

2024. 3

- [66] Michael D Skarlinski, Sam Cox, Jon M Laurent, James D Braza, Michaela Hinks, Michael J Hammerling, Manvitha Ponnapati, Samuel G Rodriques, and Andrew D White. Language agents achieve superhuman synthesis of scientific knowledge. arXiv preprint arXiv:2409.13740, 2024. 2, 3
- [67] Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, and Kai Yu. Scieval: A multi-level large language model evaluation benchmark for scientific research. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 19053–19061, 2024. 3
- [68] Kyle Swanson, Wesley Wu, Nash L Bulaong, John E Pak, and James Zou. The virtual lab: Ai agents design new sarscov-2 nanobodies with experimental validation. bioRxiv, pages 2024–11, 2024. 2
- [69] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 7
- [70] Surendrabikram Thapa and Surabhi Adhikari. Chatgpt, bard, and large language models for biomedical research: opportunities and pitfalls. Annals of biomedical engineering, 51

(12):2647–2651, 2023. 2

- [71] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visiolinguistic compositionality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238–5248, 2022. 3
- [72] Peter J Thul, Lovisa Akesson,˚ Mikaela Wiking, Diana Mahdessian, Aikaterini Geladaki, Hammou Ait Blal, Tove Alm, Anna Asplund, Lars Bj¨ork, Lisa M Breckels, et al. A subcellular map of the human proteome. Science, 356(6340): eaal3321, 2017. 7
- [73] Minyang Tian, Luyu Gao, Shizhuo Dylan Zhang, Xinan Chen, Cunwei Fan, Xuefei Guo, Roland Haas, Pan Ji, Kittithat Krongchon, Yao Li, et al. Scicode: A research coding benchmark curated by scientists. arXiv preprint arXiv:2407.13168, 2024. 2, 3

- [74] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860, 2024. 3, 18
- [75] Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. NEJM AI, 1(3):AIoa2300138, 2024. 3
- [76] Matheus P Viana, Jianxu Chen, Theo A Knijnenburg, Ritvik Vasan, Calysta Yan, Joy E Arakaki, Matte Bailey, Ben Berry, Antoine Borensztejn, Eva M Brown, et al. Integrated intracellular organization and its variations in human ips cells. Nature, 613(7943):345–354, 2023. 7
- [77] David Wadden, Kejian Shi, Jacob Morrison, Aakanksha Naik, Shruti Singh, Nitzan Barzilay, Kyle Lo, Tom Hope, Luca Soldaini, Shannon Zejiang Shen, et al. Sciriff: A resource to enhance language model instruction-following over scientific literature. arXiv preprint arXiv:2406.07835,

2024. 2

- [78] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024. 17
- [79] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 7
- [80] Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. arXiv preprint arXiv:2307.10635, 2023. 3
- [81] Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin, Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan, Quanzeng You, and Hongxia Yang. Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning. arXiv preprint arXiv:2401.06805, 2024. 2, 3, 4
- [82] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024. 7
- [83] Zhilin Wang, Alexander Bukharin, Olivier Delalleau, Daniel Egert, Gerald Shen, Jiaqi Zeng, Oleksii Kuchaiev, and Yi Dong. Helpsteer2-preference: Complementing ratings with preferences. arXiv preprint arXiv:2410.01257, 2024. 7
- [84] xAI. Bringing grok to everyone, 2024. Accessed: 2025-03-

13. 7

- [85] Tong Xie, Yuwei Wan, Wei Huang, Yufei Zhou, Yixuan Liu, Qingyuan Linghu, Shaozhou Wang, Chunyu Kit, Clara Grazian, Wenjie Zhang, et al. Large language models as master key: unlocking the secrets of materials science with gpt. arXiv preprint arXiv:2304.02213, 2023. 2

- [86] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006, 2024. 2
- [87] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 2, 3, 4, 7
- [88] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Ming Yin, Botao Yu, Ge Zhang, et al. Mmmu-pro: A more robust multidiscipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. 3, 4, 7, 15
- [89] Nikki B Zaidi, Charles Hwang, Sara Scott, Stefanie Stallard, Joel Purkiss, and Michael Hortsch. Climbing bloom’s taxonomy pyramid: Lessons from a graduate histology course. Anat. Sci. Educ., 10(5):456–464, 2017. 5, 8
- [90] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6720–6731,

2019. 3

- [91] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and SongChun Zhu. Raven: A dataset for relational and analogical visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5317– 5327, 2019. 3
- [92] Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan, Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davison, Hui Ren, et al. A generalist vision–language foundation model for diverse biomedical tasks. Nature Medicine, pages 1–13, 2024. 3
- [93] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al. Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915,

2023. 7, 3, 17

- [94] Yuhui Zhang, Yuchang Su, Yiming Liu, Xiaohan Wang, James Burgess, Elaine Sui, Chenyu Wang, Josiah Aklilu, Alejandro Lozano, Anjiang Wei, et al. Automated generation of challenging multiple-choice questions for vision language model evaluation. arXiv preprint arXiv:2501.03225, 2025. 3
- [95] Haiteng Zhao, Chang Ma, FangZhi Xu, Lingpeng Kong, and Zhi-Hong Deng. Biomaze: Benchmarking and enhancing large language models for biological pathway reasoning. arXiv preprint arXiv:2502.16660, 2025. 3
- [96] Zhiling Zheng, Oufan Zhang, Ha L Nguyen, Nakul Rampal, Ali H Alawadhi, Zichao Rong, Teresa Head-Gordon, Christian Borgs, Jennifer T Chayes, and Omar M Yaghi. Chatgpt research group for optimizing the crystallinity of mofs and cofs. ACS Central Science, 9(11):2161–2170, 2023. 2

## MicroVQA: A Multimodal Reasoning Benchmark for Microscopy-Based Scientific Research

### Supplementary Material

Appendices A-C include acknowledgments, author contributions, ethics, and discussion of limitations and future work. The remaining sections correspond to sections in the paper. Appendix D is benchmark details from the main Sec. 3. Appendix E is MCQ generation details from the main Sec. 4. Appendix F is the experiments details from the main Sec. 5.

## Table of Contents

- A. Acknowledgments and author contributions 1
- B. Ethics 2
- C. Limitations and future work 2
- D. Benchmark details 3

- D.1. Accessing MicroVQA benchmark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- D.2. Dataset Schema and Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- D.3. Training contamination mitigation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- D.4. Benchmark attribute details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- D.5. Related benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- D.6. Bloom’s taxonomy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- D.7. Methodology for proposing tasks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- D.8. Guidance for question creators . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- D.9. Microscopy image representations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- E. MCQ generation details 7

- E.1. Question collection process details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- E.2. Experiment on naive MCQ generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- E.3. Stage 1 MCQ generation (exam alignment) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- E.4. Stage 2 MCQ generation (RefineBot) details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- E.5. MCQ generation results: stage 1 exam alignment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- E.6. MCQ generation results: stage 2 RefineBot . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- F. Experiments details 15

- F.1. Evaluation prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- F.2. Model details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- F.3. Human baseline on MicroVQA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- F.4. More experiments on attribute-level information . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- F.5. Language shortcut ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- F.6. Error analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- F.7. Reasoning trace analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

#### A. Acknowledgments and author contributions

Acknowledgements We thank Christian Kindermann for feedback on motivation for benchmarking and reasoning taxonomies; Siddharth Doshi for feedback on motivation for applications in scientific chat systems; and Krisjanis Mazurs for feedback on structuring user interviews. We also thank Assaf Zaritsky, Andrew S Moore, Pedro Guedes Dias, Angus Toland, Helen Healy, and Andrew Kassianos, and Loic Royer for discussions on defining the VQA tasks.

Funding acknowledgements We gratefully acknowledge funding from HAI Hoffman-Yee (E.L., S.Y-L., J.M.H.B.), Schmidt Futures (E.L.), the Bridge2AI Program (NIH Common Fund; OT2 OD032742; E.L.,), G¨oran Gustafsson Foundation and Knut and Alice Wallenberg Foundation (KAW 2021.0346; E.L.) as well as to MU (HPA). S.C. received funding from NIH/NIGMS R35GM133460 and CZI A23-0264-001. J.N.H. was supported by a Postdoctoral Fellowship from EMBO Postdoctoral Fellowship (ALTF 556-2022).

###### Author contributions

- • Project conception: JB, JN, LBS, AL, SYL
- • Defining the VQA tasks: JB, JN, JGG, WC, SC, JNH, CL, SYL
- • Managing benchmark collection processes: JN, JB
- • Creating benchmark questions: DB, ZC, SMH, AJ, WDL, MGN, RY, CZ, JNH, CL
- • MCQ conversion pipeline: JB, JN, LBS
- • Evaluations: JB, SRG, YS
- • Qualitative error analysis: JN, JGG, CZ
- • Quantitative error analysis: LBS
- • Paper writing and figures: JB, JN, LBS
- • Supervision: SYL, EL, CL, MDL, CNH, SC, WC

#### B. Ethics

Ethical use of biomedical data: MicroVQA was developed with a commitment to ethical practices in handling biomedical research data. All microscopy images included in the dataset were either original, unpublished data or sourced from open-access articles (e.g., CC-BY licensed) published after January 2024. The dataset does not include patient-identifiable information, and MicroVQA adheres to applicable privacy and research ethics guidelines. The dataset does not constitute human subjects research.

Potential societal impacts: Biomedical AI models, including those trained on MicroVQA, have the potential to influence research and healthcare significantly, both positively and negatively. While these models can accelerate scientific discovery, they may also perpetuate biases in training data or result in uneven performance across demographic or biological groups. To mitigate these risks, MicroVQA includes metadata annotations to support the analysis of potential biases and emphasizes diverse data sourcing. We will engage with the research community to address emerging ethical concerns.

Data licensing and usage: MicroVQA is distributed under the Creative Commons Attribution-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-sa/4.0/) license to promote transparency and collaboration in the research community. This license reflects our efforts to balance open access with the ethical use of data, enabling both academic and commercial applications of the dataset while adhering to original license requirements. Full licensing details are available at https://creativecommons.org/licenses/by-sa/4.0/.

#### C. Limitations and future work

While MicroVQA advances research-level reasoning benchmark significantly, we can identify a number of limitations and avenues for future work.

Open vs closed evaluation Like most VQA benchmarks, we used multiple-choice (MCQ) evaluation. However downstream applications will obviously operate in an open setting, without options and requiring more detail. MLLM research does have some open benchmarks [86], though impartial evaluation is a significant challenge. Our early experiments with open prompting suggest that models like GPT-4o tend to give very vague answers, and tend to depend strongly on the text part of the input prompt.

Dataset scale Our final dataset had 1042 samples, which is smaller than some other VQA benchmarks like MicroBench [50] and OmniMedVQA [34]. MicroVQA’s scale sufficient to compare overall scores between models, however statistical power starts to become limiting when trying to compare different models on subsets of the dataset. How does this compare to other benchmarks? As we can see from Tab. 2 and Fig. 3, the larger datasets tend to be those with lower-level reasoning according to Bloom’s taxonomy. Their samples tend to be derived automatically from metadata tags or classification labels. Benchmarks with comparably high-level reasoning, like MMMU-Pro, have a similar scale to ours (1,700). In the languageonly domain, the research-level reasoning benchmark GPQA has 448 samples. This trend reflects a clear, practical limitation: it is very time-consuming for humans to create difficult reasoning questions, and the pool of capable experts is small.

Breadth of dataset attributes in microscopy We aimed for MicroVQA to cover a broad range of the microscopy discipline; for example, we have samples from all four relevant scales – tissues, cells, subcellular, atomic. However, each human annotator is an expert in a subset of microscopy and biology, so there are practical limits on dataset coverage. We, therefore, chose to focus on the most common imaging modalities – brightfield, fluorescence, electron – while skipping less common modalities like Raman spectroscopy. In terms of sample types, we emphasized those relevant to humans – human and mouse – while having only a few samples from rare organisms like bottlenose dolphin.

Coverage of scientific reasoning Looking even more broadly, the motivation of this work is to advance reasoning in scientific research generally. However, we use microscopy in particular. This is because of a clear trade-off in breadth vs depth. Our dataset strikes a good balance, given that microscopy represents a large portion of visual data in biomedicine and biology. Future work can reuse our framework to generate reasoning VQA benchmarks in other fields of visual biomedicine and biology, such as in medicine [34] or ecology [93].

Resolving MCQ shortcuts A key contribution in this paper is the RefineBot for generating multiple choice questions (MCQs) that are too challenging to solve using text-only shortcuts. Our approach used the outputs of existing MLLMs as signal to make distractors harder. However, this cannot guarantee the removal of shortcuts; more generally, harder distractors may exist. Therefore, the next generation of LLMs may be able to identify and exploit new shortcuts. If that occurs, we can run RefineBot again using the new LLM, creating MicroVQA-2. This strategy could be used for other VQA evaluations.

#### D. Benchmark details

##### D.1. Accessing MicroVQA benchmark

MicroVQA is an expert-curated benchmark for multimodal scientific reasoning, specifically in microscopy-based biological research. The dataset comprises 1,042 multiple-choice questions (MCQs) created by biological researchers to represent authentic scientific practice. Each MCQ is paired with an associated microscopy image and metadata describing the experimental context. A demonstration subset of 28 instances is available for initial exploration.

The MicroVQA dataset is publicly accessible on HuggingFace at https://huggingface.co/datasets/jmhb/microvqa. It is distributed under the Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license, allowing reuse and share-alike redistribution with attribution. The dataset is available in Apache Arrow and Parquet formats for efficient data handling. These formats support fast serialization, memory mapping, and streaming from the web, facilitating seamless integration into machine learning pipelines.

##### D.2. Dataset Schema and Structure

The dataset schema defines the fields included for each data instance. Figure 6 provides a schema summary, while a sample data instance is detailed below 7.

|{<br><br>"image_id": datasets.Value("string"), "image": datasets.Image(decode=True, id=None), "label": datasets.ClassLabel(), "label_name": datasets.Value("string"), "domain": datasets.Value("string"), "subdomain": datasets.Value("string"), "modality": datasets.Value("string"), "submodality": datasets.Value("string"), "stain": datasets.Value("string"), "microns_per_pixel": datasets.Value("string"), "questions": datasets.Value("string"),<br><br>}|
|---|

- Figure 6. Example data schema.

##### D.3. Training contamination mitigation

To mitigate contamination, all data in MicroVQA were sourced from original, unpublished microscopy images or open-access articles (e.g., CC-BY-licensed) published after January 2024. Additionally, canary strings were embedded in the dataset to detect potential contamination during fine-tuning or model evaluation.

|{<br><br>"image_id": "d54bb7ec-284f-4218-a47d-af87bb371de5", "image": datasets.Image(decode=True, id=None), "label": datasets.ClassLabel(), "label_name": "pathology", "domain": "pathology", "subdomain": "gastrointestinal pathology", "modality": "light microscopy", "submodality": "brightfield microscopy", "stain": "H&E", "microns_per_pixel": "Not provided", "questions": "Question:<br><br>‘‘‘A hematoxylin and eosin (H&E)-stained micrograph shows cellular formations within a gastrointestinal tissue sample with unique organization and cellular patterns. What pattern and morphology do these cells most closely align with?<br><br>A) Bundles of elongated cells with pale pink cytoplasm and uniform oval nuclei in a crisscross pattern<br>B) Circular groups of cells with mucin in central spaces and small, dark nuclei<br>C) Dense arrays of polygonal cells with small nucleoli and vacuolated cytoplasm<br>D) Rows of cells with conspicuous cytoplasm and rounded nuclei<br>E) Diffused arrangement of cells with transparent cytoplasm and irregularly clustered chromatin<br><br><br>Correct Answer: A) Bundles of elongated cells with pale pink cytoplasm and uniform oval nuclei in a crisscross pattern‘‘‘<br><br>}|
|---|

- Figure 7. Example data instance.

##### D.4. Benchmark attribute details

The MicroVQA benchmark is designed to evaluate multimodal reasoning capabilities in biomedical research contexts. The dataset incorporates diverse attributes such as organism, research subject, imaging modalities, and biological length scales (image scales) to ensure a comprehensive assessment1. These metadata not only enhance the scientific relevance but also enable analysis of reasoning across different biological and microscopy contexts. This section describes the methodology used to define, compute, and annotate these key attributes.

The benchmark comprises 1,042 total questions, including 423 questions containing multiple images combined into panels. A collage of sample images is shown in 25. The average MCQ token lengths were computed using the o200k_base tokenizer, with separate calculations for the raw input and formatted multiple-choice questions. Imaging modalities are broadly categorized into light, fluorescence, and electron microscopy. Biological length scales span four levels: tissue (millimeter scale), cellular (micrometer scale), subcellular (nanometer scale), and atomic (angstrom scale), enabling reasoning across diverse structural hierarchies.

To annotate attributes such as organism, specimen, and research subject, we manually reviewed random samples from the dataset and created a taxonomy to cover the most likely categories. A biomedical domain expert manually labeled these categories for 50 raw input question-answer pairs and performed initial prompt engineering. We used DSPy to optimize the prompt and few-shot examples on the labeled dataset. The input context included the taxonomy of organisms8 and research subjects 9.

These annotations were the gold standard to guide DSPy prompt engineering for automated attribute labeling on the full dataset. Prompt instructions included fallback rules (e.g., tagging ”None” for indeterminate attributes) to ensure accuracy and minimize hallucination. The optimized pipeline annotated all questions while ensuring the validity of the dataset attributes.

##### D.5. Related benchmarks

In section 3.3 and Tab. 2 we compare MicroVQA to related scientific multimodal benchmarks, focusing on its intersection of high-difficulty and advanced reasoning questions. Beyond these key attributes, we examine the sources of the questions. Most benchmarks rely on existing question repositories, such as exams, web-based QA datasets, textbooks, or other MCQ datasets. This approach bypasses the need for MCQ generation methods, as the questions and options are pre-tested and curated, making these benchmarks more representative of established tasks. Another common strategy involves reformatting classification datasets into MCQs using question templates, which increases task diversity but often limits scenario unique-

|Taxonomy of organisms in biology| # Taxonomy of Organisms in Biology and Biomedical Research: Apis mellifera (Honeybee)<br><br>Social insect with complex behavior. Keywords: honeybee, social behavior, ecology, pollination.<br><br>Arabidopsis thaliana (Thale Cress) Model plant organism. Keywords: Arabidopsis, plant genetics, photosynthesis.<br><br>Caenorhabditis elegans (C. elegans) Transparent nematode worm. Keywords: worm, development, aging, programmed cell death.<br><br>Danio rerio (Zebrafish) Fish with transparent embryos. Keywords: zebrafish, vertebrate development, embryology.<br><br>... Drosophila melanogaster (Fruit Fly)<br><br>Insect with rapid life cycle. Keywords: fruit fly, genetics, development.<br><br>Escherichia coli (E. coli) Bacterium used in genetics and molecular biology. Keywords: bacteria, gene expression, plasmids.<br><br>... Homo sapiens (Human Cells)<br><br>Cultured human cells (e.g., HeLa, HEK293, A431, U2OS). Keywords: human cell lines, immortalized cell lines<br><br>In vitro (None) Purified molecules and reactions in controlled environments. No living cells or organisms. Keywords: purified proteins, purified molecules, cell-free systems<br><br>...|
|---|

- Figure 8. Examples of taxonomy classes used as context to LLM to assign an organism to a question. A YAML file with the full taxonomy will be released with the code.

ness and language variability. In contrast, MicroVQA is built by directly querying experts for challenging, domain-specific research questions. By starting with long form questions and answers, our approach requires an innovative MCQ generation method, ensuring uniquely tailored questions that push the boundaries of domain-specific reasoning and knowledge assessment.

##### D.6. Bloom’s taxonomy

Bloom’s Taxonomy is a model for evaluating cognitive skill levels in educational assessments. It has six hierarchical levels of learning: recall, understand/comprehend, apply, analyze, evaluate, and create. Multiple-choice questions can test five of Bloom’s levels but not the sixth level, “create”. These levels provide a structured framework for assessing lower-order and higher-order thinking skills, which are essential for designing robust benchmarks like MicroVQA.

Initial attempts to use LLMs to classify Bloom’s levels for biomedical visual question answering (VQA) benchmarks revealed a systematic overestimation of cognitive skill levels. This discrepancy limits the naive use LLMs for automated Bloom’s classification, perhaps due to the nuances of applying Bloom’s taxonomy in domain-specific contexts. To ensure accurate classification of Bloom’s level, a physician-scientist with NBME training in item writing and familiarity with Bloom’s classification manually labeled Bloom’s level for 200 examples from public MCQ questions, MicroVQA, MicroBench, and ScienceQA. These examples were used to fine-tune GPT-4o-mini (gpt-4o-mini-2024-07-18) to classify Bloom’s level and align with expert judgments.

The fine-tuned Bloom’s classifier was applied to MicroVQA and numerous public science/biomedical benchmarks to assess the cognitive skills evaluated by each question. Each question was mapped to one of Bloom’s levels, providing insights into the distribution of cognitive tasks across the benchmark. This analysis revealed that while existing benchmarks predominantly tested lower-order skills (e.g., recall and comprehension), MicroVQA has a greater emphasis on questions that require higher-order reasoning (e.g., analyzing, evaluating), filling a critical gap in multimodal scientific reasoning

|Taxonomy of research subjects| #Taxonomy of Research Subjects in Biology: Anatomy<br><br>Study of the structure of organisms and their parts. Keywords: organs, tissues, morphology, physical organization.<br><br>Biochemistry Chemical processes within living organisms. Keywords: proteins, enzymes, nucleic acids, metabolic pathways.<br><br>... Cell and Molecular Biology<br><br>Structure and function of cells and their molecules. Keywords: cellular processes, gene expression, molecular pathways.<br><br>... Developmental Biology<br><br>Process by which organisms grow and develop. Keywords: embryology, morphogenesis, genetic regulation.<br><br>... Genetics<br><br>Study of heredity and gene function. Keywords: DNA, inheritance, genetic variation, gene expression.<br><br>Immunology Immune system in health and disease. Keywords: antibodies, immune response, pathogens, vaccines.<br><br>Microbiology Study of microorganisms. Keywords: bacteria, viruses, fungi, parasites, microbial ecology.<br><br>Neurobiology Structure and function of the nervous system. Keywords: neurons, brain, neural circuits, cognition.<br><br>... Structural Biology<br><br>Molecular structure of biological macromolecules. Keywords: protein folding, nucleic acid structures, crystallography.<br><br>...|
|---|

- Figure 9. Examples of taxonomy classes used as context to LLM to assign a research subject to a question. A YAML file with the full taxonomy will be released with the code.

assessments.

##### D.7. Methodology for proposing tasks

Here, we expand on the task proposal methodology introduced in Sec. 3.2. To define specific reasoning tasks aimed at advancing biomedical research, we conducted a comprehensive expert consultation process. Through structured interviews with nine Principal Investigators and postdocs specializing in microscopy research, including participants from both academia and industry, we developed a framework for tasks that would meaningfully integrate multimodal large language models (MLLMs) into microscopy research workflows.

The task selection process followed specific criteria: tasks must utilize image-text inputs and text outputs suitable for MLLMs; they should require higher-order reasoning beyond simple image processing; and they should emphasize core experimental activities (experimentation, analysis, and action) rather than auxiliary tasks like literature review or writing [33].

Our methodology involved a two-phase interview process. The first phase consisted of initial one-hour individual sessions with eight research groups featuring open-ended, unstructured discussions about key challenges in microscopy experiments, potential applications of MLLMs, current workflows and unmet needs, and possible AI system integration points. One week later, we then conducted follow-up interviews to consolidate findings and build consensus.

To avoid anchoring biases, we intentionally withheld predefined categories during initial interviews. This approach enabled unbiased task identification and allowed us to capture a broad spectrum of potential use cases. Through this iterative process, we identified three essential capabilities that align with the scientific life cycle for microscopy experiments: advanced image interpretation, hypothesis generation, and experimental design/hypothesis testing.

##### D.8. Guidance for question creators

Question creators were given detailed guidance as pdfs, that we share at https://github.com/jmhb0/microvqa under folder benchmark/human collection. During the submission process, we completed two rounds of quality control to ensure difficulty and alignment to the prompts, as described in Sec. 4.1.

##### D.9. Microscopy image representations

MLLMs are trained primarily on three-channel RGB images, however microscopy images are more diverse. To ensure alignment with how common MLLMs process images, we required that each image artifact for a single question was represented as no more than six RGB images. Where multiple images were needed, they could be multiple image files, or a single image that is concatenated. For videos, users could submit up to six frames; for 3d, they could submit slices or z-projections.

An important difference is multichannel images. Many fluorescence microscopy images are multichannel, where the image collection ensures that each channel represents a certain structure. For example channel 0 may be nucleus, and channel 1 may be mitochondria. There are a few approaches. Some questions represent this situation as separate grayscale images for each channel. Others show a ‘composite image’, which is a single image where one each channel is mapped to one of the RGB channels; e.g. a common visualization will put nucleus in blue and mitochondria in green. Some questions do both: grayscale images for each channel, along with a composite image merging them.

Finally, we allowed screenshots from research papers or image software, since this is a realistic way that future LLM-users would use AI tools.

#### E. MCQ generation details

##### E.1. Question collection process details

Data collection used a structured submission form shared with collaborators. The form required users to provide the following:

- • Image or Image Set: Users uploaded microscopy images, optionally including multiple images for comparative questions.
- • Experiment Details: A brief description of the experimental setup, imaging modalities, and any relevant biological context.
- • Question Text: Each question was aligned with one of three predefined use cases: identifying interesting features, explaining mechanisms, or proposing next steps.
- • Answer and Distractors: Contributors provided a correct answer and biologically plausible incorrect answers. Quality control was performed in multiple rounds. To familiarize users with the task and data requirements, each user

submitted an initial form with 4-6 questions for manual quality control and feedback. Feedback was provided to users on improving question specificity, ensuring the questions required image interpretation rather than relying solely on general knowledge. Contributors were encouraged to refine their submissions based on the feedback to better align with the tasks and goals of the MicroVQA benchmark.

Users were encouraged to submit data from multiple sources, imaging modalities (e.g., fluorescence, electron microscopy), and biological scales (e.g., tissue, cellular, subcellular) to promote diversity. Contributors could upload microscopy images from their experiments or use CC-BY publicly available images. Common image sources included preprints or publications released after January 2024 (to mitigate contamination), or from established image repositories [14, 17, 18, 31, 72, 76]. This ensured that all images were either original or sourced from open-license publications, enabling their use in an open-source benchmark.

##### E.2. Experiment on naive MCQ generation

In Sec. 4.2, we discuss how the naive approach to MCQ generation – simply zero-shot prompting GPT – leads to MCQs that are very easy for frontier models to solve. Which ultimately motivated our two-stage MCQ generation approach. Here, we discuss those experiments in more detail. We create a naive version of MicroVQA using a standard prompt [34, 82]:

You are an expert in molecular and cell biology, and in microscopy. I will give you an original biology-related question and its answer, your task is to rephrase an equivalent question with identical answer. The question related to an image, and we don’t show the image. Meanwhile, I want to transfer this QA-pair into a multi-choice question. Please generate 5 incorrect options to construct the candidate options. {{QUESTION}}

{{ANSWER}}

Then, we evaluated this naive MicroVQA on the top closed source models with the image, and to study the over-reliance on language shortcuts, we evaluated a text-only version. In the text-only setting, we add the following to the prompt: “If an image is mentioned, ignore this information and try your best to answer the question.”. Tab. 5 shows the results of this experiment.

Table 5. Performance on MicroVQA MCQs with naive MCQ generation. We report overall VQA and VQA without the image.

VQA VQA-no-image

GPT-4o 85.1 82.7 Claude-3.5-Sonnet 91.4 88.4 Gemini-1.5-Pro 88.5 82.4

All models score over 80% on questions that are from a specialized domain. We hypothesize that this reflects a problem with the non-trivial task of MCQ generation from raw VQAs for several reasons. First, the high ‘VQA-no-image’ scores across all models suggest that selecting the correct option does not require image understanding, despite the raw VQA samples being designed to incorporate image-based reasoning. An alternative explanation is that the questions themselves may be too easy for advanced models. Second, we test this alternate explanation by performing open evaluation on a subset of the raw VQA queries– that is, we queried GPT-4o with the questions but without the options. Here, the models rarely gave good responses, even after careful prompting. This suggests that models are leveraging test-taking strategies to eliminate distractors [27]. Third, we examined the chain-of-thought responses in the VQA-no-image case, which suggested the models often eliminate distractors based on information from the text question. Therefore, we conclude that there are issues with the improper generation of the MCQs using the naive strategy.

What exactly are the problems with MCQ generation? Qualitatively, the types of errors are the same as those we discuss in Appendix E.6, so we refer readers to that section to get a deeper understanding.

##### E.3. Stage 1 MCQ generation (exam alignment)

The first stage transformed raw, open-ended questions and long-form answers into well-structured MCQs aligned with established principles of biomedical exam design. A physician-scientist trained in medical board exam item writing initiated the process by reviewing biomedical educational literature, NBME guidelines [64], and Bloom’s taxonomy for assessing cognitive skills [5, 19, 89]. The expert manually converted 50 raw question-answer pairs into one-best-answer MCQs with biologically relevant distractors using the literature review and their prior training. This manual effort ensured that the MCQs preserved the original question’s intent and scientific validity while adhering to educational standards. Distractors were crafted to require reasoning and prevent reliance on superficial cues like word association or general knowledge, thus maintaining the depth and complexity of the original content.

To scale this process, the expert designed an initial LLM prompt for automated Stage 1 MCQ generation. This prompt and the manually curated training data were input for DSPy-based prompt optimization. Using the MIPROv2 optimizer, the prompt was refined within a Chain of Thought (CoT) framework combined with retrieval-augmented generation (RAG) pipelines powered by o1-mini. Optimization was guided by a custom metric evaluated by a separate LLM judge (o1-mini). The judge assessed each revised MCQ across three criteria: (1) semantic similarity to the original question and answer, (2) adherence to NBME formatting standards, and (3) minimization of extraneous information that could provide unintended clues. The optimization metric was weighted to prioritize preserving the original meaning and scientific validity, with lesser emphasis on formatting and extraneous text. Additional implementation details are provided in the source code.

##### E.4. Stage 2 MCQ generation (RefineBot) details

The goal of RefineBot is to take MCQs that are too easy for MLLMs – especially where they are leveraging language-only shortcuts – and to rewrite them to make them more challenging. It is the third panel in Fig. 4. For more details on what types of improvements are achieved, see Appendix E.6. The complete code for RefineBot is released.

A single RefineBot iteration Below, we show the pseudo-code for a single-question revision. The input is a question string, a list of options, and the index of the correct answer in the options. It returns a tuple (code, MCQ) where code is a

string on whether the run was successful, and MCQ is either the revised multiple choice question if the rewrite was successful or otherwise None.

- 1 def revise_mcq(question: str, choices: list, correct_idx: int, max_iters: int = 5):

- 2 # Store original version

- 3 original = {’question’: question, ’choices’: choices, ’correct_idx’: correct_idx}

- 4 reflections = []

- 5

- 6 for iter in range(max_iters):

- 7 # Try answering without image

- 8 result = evaluate_without_image(question, choices, correct_idx)

- 9

- 10 if not result.can_answer:

- 11 code = ’SUCCESS_NO_CHANGE’ if iter == 0 else ’SUCCESS_REWRITE’

- 12 return code, (question, choices, correct_idx)

- 13

- 14 if iter == max_iters - 1:

- 15 return ’FAIL_ITERATIONS’, None

- 16

- 17 # Reflect on strategies used to solve

- 18 reflection = analyze_why_answerable(result)

- 19 reflections += reflection

- 20

- 21 # Rewrite based on all reflections from all iterations

- 22 new_version = rewrite_question(reflections, original)

- 23

- 24 # Verify the rewrite didn’t change the meaning meaning preserved

- 25 if not check_meaning_preserved(original, new_version):

- 26 return ’FAIL_REWRITE’, None

- 27

- 28 # Update current version

- 29 question = new_version[’question’]

- 30 choices = new_version[’choices’]

- 31 correct_idx = new_version[’correct_idx’]

- 32

- 33 return ’FAIL_UNKNOWN’, None

This process is also described in the main text. One detail to note: the rewrite_question function receives the reflections from all prior iterations, not just the current one. In this way, the rewriting function can consider all possible strategies that have been identified. All LLM calls are to gpt-4o-2024-08-06.

To better illustrate the method, we step through a single example. This way, we can show the prompts and understand responses.

- 1. The evaluation of a question with the evaluate_without_image prompt and response:

Prompt

-------------------------------------------------------------------------------The following question is supposed to be paired with an image. We will not provide the image, so answer to the

best of your ability.

In an electron microscopy image of a mouse primary embryonic cortical neuron expressing full-length human mHTT (BACHD), "eye-ball" shaped features are observed in the top left corner that appear anomalous. What is the most likely cause of these features?

Think step by step and then output the answer in the format of "The answer is (X)" at the end.

- (a) Aggregation of human mHTT protein within the neuron

- (b) Ice contamination from liquid nitrogen during sample preparation

- (c) Formation of intracellular vesicles during neuron development

- (d) Artifacts arising from excessive heavy metal staining

-------------------------------------------------------------------------------Response (target answer is b)

-------------------------------------------------------------------------------To determine the cause of the "eye-ball" shaped features observed in the electron microscopy image, we can

analyze each option step by step:

- (a) Aggregation of human mHTT protein within the neuron: Aggregations of proteins can indeed appear as anomalous structures under electron microscopy. However, these typically have irregular shapes and may not consistently resemble "eye-ball" shapes.

- (b) Ice contamination from liquid nitrogen during sample preparation: Ice contamination can occur during sample preparation, particularly if samples are not handled properly when flash-frozen. This can lead to

anomalous features appearing in microscopy images. These features might appear as circular or irregular shapes, potentially looking like "eye-balls."

- (c) Formation of intracellular vesicles during neuron development: Intracellular vesicles are normal structures within neurons and other cells. Although they can appear during development, they are typically well

characterized and documented, and are less likely to be described as anomalous in the context given.

- (d) Artifacts arising from excessive heavy metal staining: Over staining can indeed produce artifacts in electron microscopy images. While staining artifacts are plausible, they might not specifically result in " eye-ball" shapes unless the staining process consistently targets specific round structures.

Considering the nature of the descriptors ("eye-ball" shaped, anomalous), option (b) ice contamination is often

a source of unexpected features in microscopy images that could fit the appearance described, as ice can form circular shapes.

Therefore, the most likely cause of these features is (b) Ice contamination from liquid nitrogen during sample preparation.

The answer is (b).

- 2. If the answer is incorrect, then the question is already hard enough, and we can exit. Otherwise, we call the analyze_why_answerable function, which reflects on how this answer was achieved. It continues in the same conversation:

-------------------------------------------------------------------------------User

-------------------------------------------------------------------------------That is correct.

Explain how you were able to answer the question without access to the image - only the question_stem and

choices. What strategies did you use? Then, briefly summarize these "language-only strategies" into a short list. Finally, say whether your reasoning was closer to:

- - ’deduction’: you answered with high confidence because you had enough information.

- - ’abduction’: you answered with lower confidence you made an informed guess. Be concise in your final response.

- -------------------------------------------------------------------------------Assistant

-------------------------------------------------------------------------------To answer the question without the image, I relied entirely on analyzing the given information and the

plausible characteristics of each option. Here’s a breakdown of my strategy:

- 1. **Keyword and Context Analysis**: I focused on key descriptors such as "eye-ball" shaped features and their being "anomalous" in an electron microscopy image of neurons expressing human mHTT. This guided me to consider characteristics commonly seen under electron microscopy.

- 2. **Knowledge of the Subject Matter**: I applied general knowledge about electron microscopy artifacts and protein behavior. This helped me weigh the likelihood of each option based on what typically causes specific image features.

- 3. **Process of Elimination**: By understanding the nature of each option-including potential artifacts or structures (such as vesicles or ice) in microscopy-I systematically assessed the validity of each scenario in relation to the described features.

- 4. **Logical Inference**: I inferred which scenario most commonly results in anomalies resembling "eye-ball" features, favoring the simplest explanation.

Summary of "language only strategies":

- - Analyze keywords and context

- - Apply relevant subject matter knowledge

- - Use process of elimination

- - Make logical inferences

The reasoning was closer to ’abduction’ as I made an informed guess based on patterns and typical causes, given that I lacked direct visual confirmation.

- 3. Next, we call rewrite_question. Here, the templated variable {{conversations}} has the full conversations produced by the function analyze_why_answerable in the previous call and from all other iterations. The {{n_chat}} just indicates how many conversations there are included:

Below, I will display {{n_chat}} chat conversations between a ’user’ and an LLM ’assistant’.

In each conversation

- - a user asks the assistant to answer a multichoice VQA question, however they do not provide the image. They only get the question_stem and choices.

- - the question_stem and correct answer are similar to other conversations, however the distractors are different

- - the assistant then answers correctly.

- - the user then asks the assistant to explain how it answered the question with only the text.

- - the assistant then summarizes what strategy they used to answer the question.

Altogether, these conversations give examples of language shortcuts that need to be avoided when constructing

the question_stem and choices. Here are the conversations: {{conversations}} Your task is to rewrite the question_stem and choices so that a different LLM ’assistant’ cannot use the

language-only strategies that were identified in these past conversations. Include an ’explanation’ about why your new set of distractors are better. Your revised choices should include the correct answer at the ’correct_index’.

Your revised question_stem and choices should not significantly change the meaning of the question and correct

answer. You are free to change the distractors a lot to achieve this task. Include {{n_choices}} choices. The choices should not be made too different in length and detail to the original question stem and choices. Note that the question stem may contain important cues that cannot be removed. E.g. if a question asks about

the "green stain" in an image, you cannot change it to "stain" because it introduces ambiguity. The choices should not include the letters in front like (a), (b), ... Return a json

The output of this function is a JSON and is:

{

"question_stem": "In an electron microscopy image of a neuron expressing human mHTT (BACHD) protein, distinct, spherical features are prominent and seem to be atypical. What is the most probable explanation for these features?",

"choices": [ "Concentration of cellular waste products leading to distinct morphological anomolies", "Formation of air bubbles introduced during sample immobilization processes", "Ice contamination from liquid nitrogen during sample preparation", "Inclusion bodies formed by aggregated proteins expressing during stress response", "Cellular morphological changes during mitotic division"

], "correct_index": 2, "explanation": "The revised options introduce plausible scientific phenomena that could also result in

spherical or distinct anomalies. This set of distractors is more varied and grounded in legitimate alternative explanations without leading the responder towards the correct answer through language patterns. For example, air bubbles and protein inclusion bodies might mimic sphere-like structures. Additionally, by adding an unrelated phenomenon like cellular changes during division, the set confuses the language pattern strategy previously used. Thus, the correct choice of \"Ice contamination from liquid nitrogen\" stands out less and necessitates either visual confirmation or real understanding of sample preparation outcomes to identify correctly."

}

Notice that the response includes an explanation of why it made the changes, and the changes relate to what was in the reflection, adding interpretability.

- 4. Finally, we check that the rewrite has not changed the semantic meaning of the question and answer.

Prompt

******************************************************************************** Below are two question-answer pairs. The question-answer pairs are part of VQA triplets, and both pairs use the same image.

Are these question-answer pairs semantically equivalent? Or are they significantly different? Give a true/false and also an explanation.

- QUESTION 1: In an electron microscopy image of a mouse primary embryonic cortical neuron expressing full-length human mHTT

(BACHD), "eye-ball" shaped features are observed in the top left corner that appear anomalous. What is the most likely cause of these features?

- ANSWER 1: Ice contamination from liquid nitrogen during sample preparation

- QUESTION 2:

In an electron microscopy image of a neuron expressing human mHTT (BACHD) protein, distinct, spherical features are prominent and seem to be atypical. What is the most probable explanation for these features?

- ANSWER 2: Ice contamination from liquid nitrogen during sample preparation

******************************************************************************** Response, is_equivalent: True

******************************************************************************** is_equivalent: Both question-answer pairs revolve around electron microscopy images depicting neurons

expressing the human mHTT (BACHD) protein. The main focus in each question is on the presence of unusual spherical (or ’eye-ball’ shaped) features. In both cases, the identical explanation is provided: ice contamination from liquid nitrogen during sample preparation.

While the wording differs slightly, the core components, contexts, and the ultimate explanation remain consistent between the two pairs. Thus, they can be considered semantically equivalent as they describe the same phenomenon with the same reasoning for the observed anomalies.

- 5. If the meaning check step fails, then we exit with a fail as we notice that once the meaning deviates it will become more distorted with further iterations. Otherwise, we update the MCQ, and start the loop again. Where the first function call in the new loop will check whether this revised MCQ is now too hard or needs to be rewritten.

Calibrating the preservation of meaning Since there is a risk that the rewriter will change the semantic meaning of the question and answer, it is crucial that the check_meaning_preserved is effective. For topics requiring domain expertise, this requires prompt engineering with a domain expert providing feedback. This is one of the key human bottlenecks in tuning the prompts in this system.

Other sources of MCQ invalidation While the check_meaning_preserved function checks that the question and answer is correct, an additional risk is that the rewrite introduces distractors that are more correct than the target answer. In our manual review evaluations (we reviewed all 1,042 MCQs), this was very rare. However, if this were a significant problem in future systems, it may be addressable by tuning the rewriter bot, with an expert human evaluator in the prompt engineering loop.

Extending RefineBot beyond language shortcuts We constructed RefineBot specifically to address the issue of language shortcuts – where the question is answerable without using the image. This is appropriate because we work with visioncentric VQA, meaning the questions should require image interpretation. However, the framework should work in other settings – namely language-only QA. This would require adjusting the prompts slightly. For example, the current reflection prompt asks how the question was successfully evaluated without using language; also, the current rewriter prompt directs the LLM to address language shortcuts in its rewrite. Therefore, adapting Refinebot to language-only MCQ should make these prompts more general.

On the other hand, the method could also be updated for the more general VQA setting – to improve questions that do not have language shortcuts, but are easy for some other reason. Here, the evaluator and reflector agents, which are currently LLMs, could be switched for an MLLM, which also ingests the image (though this would be expensive for current state-of-the-art models). We hypothesize that current MLLMs are so weak at image perception, that this is not necessary for MicroVQA – that simply removing language shortcuts will suffice for creating a challenging benchmark. As MLLMs improve at perceiving microscopy images, this general VQA RefineBot could become more useful for ensuring that the benchmark remains challenging.

Repeated RefineBot iterations A single run of RefineBot may exit as a FAIL, either because the rewrite changed the meaning of the question, or because the maximum number of iterations was reached. However, RefineBot can give very different outputs when changing the random seed, and we find that simply running again with a different random seed can lead to a successful run. Therefore, if the bot fails, we rerun it.

If an MCQ exited RefineBot successfully, then the evaluator LLM was fooled by the question (got the answer incorrect). Yet, the question could still be suboptimal in detectable ways: the LLM could be capable of narrowing the MCQ down to two options, but have guessed the wrong option; moreover, this question could be easier for different models. To ensure it is difficult enough overall, we require that the MCQ is answered incorrectly for two random seeds from two different models: gpt-4o-2024-08-06 and claude-3-5-sonnet-20241022. If not, then it fails, and we run the RefineBot again with a different random seed.

We discuss in the experiments Sec. 5.1 how this process leads to a small evaluation bias against the models used as evaluators. That section shows that despite the bias, this process makes the MCQs generally harder for all models. Future versions of RefineBot could mitigate this by including a larger family of evaluator models.

Finally, some MCQs fail to pass the RefineBot check. In these cases, we simply keep the stage 1 MCQ.

##### E.5. MCQ generation results: stage 1 exam alignment

The Stage 1 pipeline produced one-best-answer MCQs that preserved the content, testing focus, and validity of the expertgenerated question-answer pairs. It generated an initial set of distractors following best practices outlined in educational and NBME guidelines. A key design goal of Stage 1 was to preserve the original question content without altering or introducing new information. Secondary objectives included ensuring proper formatting according to NBME guidelines and minimizing information that might provide clues to the correct answer without reference to the image.

While the pipeline ensured alignment with these goals, preserving the original content occasionally resulted in MCQs containing extra information, such as image descriptions or specific biological terms (i.e., gene or protein name), that could inadvertently help narrow the options. For example, stage 1 question stems sometimes included protein names, disease descriptions, or descriptions of image features, which could reduce the challenge. Figure 11 illustrates Stage 1 outputs that are appropriately formatted but may preserve details from the original input, making questions less challenging. While the distractors were biologically plausible, some options were overly easy to eliminate, reducing the challenge. These limitations are addressed and refined in Stage 2.

The o1-mini-2024-09-12 LLM judge, employed during DSPy optimization, compared the Stage 1 MCQs to the original inputs to evaluate (1) semantic similarity and (2) adherence to MCQ formatting. The results showed that 96% of questions preserved the original meaning, and 100% adhered to NBME formatting guidelines. Manual analysis of the 4% labeled as at least partially dissimilar revealed that these cases typically involved open-ended original questions being reformulated into closed-ended MCQs. This transformation, a necessary step for creating one-best-answer MCQs, was not a failure but rather an expected outcome of Stage 1. In rare instances where the original submission included multiple correct answers, the output of Stage 1 retained one correct answer and excluded others to comply with the one-best-answer format. The LLM judge flagged these cases as partially dissimilar but these changes were essential for adhering to the MCQ structure.

A common issue in MCQ design is that correct answers tend to be longer and more detailed than distractors. To assess this, we computed the ratio of tokens in incorrect options to those in the correct answer, where values ≤ 1 indicate longer correct answers, and values ≥ 1 suggest shorter correct answers. Stage 1 MCQs showed a ratio of 0.87, indicating that correct answers were only slightly longer than distractors. By comparison, naive MCQ generation yielded a ratio of 0.46, meaning correct answers were nearly twice as long as distractors, making them easier to identify. Additionally, naive MCQ generation preserved the original meaning in only 89% of cases and met NBME formatting guidelines in just 70%, highlighting the efficacy of the Stage 1 process at preserving information and ensuring proper MCQ formatting.

##### E.6. MCQ generation results: stage 2 RefineBot

The second stage of our MCQ generation method focuses on increasing the difficulty of the questions and options. The key idea is that we want to remove language shortcuts – information in the MCQ that allows answering the question without access to the image. To aid in understanding, we construct a toy example in Fig. 10 with the three types of language shortcuts that we identify. The questions should be paired with a stain that is mitochondria, and we want the VQA to require actually inferring that from the image. In shortcut 1, the text ‘gives away’ the image content so it’s trivial to answer the question. In shortcut 2: The LLM generates implausible or weak distractors. In shortcut 3: ’Language bias’, is a known problem in VQA [29].

Moving to real MCQs, in Fig. 4 we illustrate the transformation of a question at each stage, highlighting issues and corresponding improvements. While stage 1 produces exam-style questions that preserve the QA objective, the wording of the questions and distractors often enable MLLMs to exploit shortcuts rather than demonstrate knowledge. To address this, stage 2 (RefineBot) employs a chain-of-thought process. We analyze additional examples of improvements achieved in stage 2, showing how RefineBot improves both the style and content. The key strategies correspond to the three shortcut types that we identified:

1. (Visual giveaway) Over-description of the image eliminates the need for image interpretation. For example, in Fig. 11 (middle row), stage 1 includes details that reveal the specifics of the image, allowing the model to propose common issues without analyzing the image. Stage 2 replaces these specifics with general terms in the question stem.

[Figure 36]

- Figure 10. Three types of language shortcut relevant to MicroVQA. The target VQA has an image that is fluorescence microscopy stained with TOMM20 which would show a pattern consistent with visualizing mitochondria.

- 2. (Weak distractors) Distractors unrelated to the specifics of the scenario described by the question. In Fig. 11 (bottom row), stage 2 increases distractor complexity by incorporating elements tied to the question’s context, making them less predictable.
- 3. (Language bias) Overly precise wording in the question stem: Precise phrasing can make it easy to identify the correct answer through pattern matching. For example, Fig. 11 (top row), changing ‘eye-ball shaped’ to ‘spherical’ retains the question’s intent while creating more challenging distractors referencing plausible alternative processes.

###### MCQ stage 1 MCQ stage 2

###### Image

Very specific wording

Question:

Question: In an electron microscopy image of a mouse primary embryonic cortical neuron expressing full-length human mHTT (BACHD), "eye-ball" shaped features are observed in the top left corner that appear anomalous. What is the most likely cause of these features?

In an electron microscopy image of a neuron expressing human mHTT (BACHD) protein, distinct, spherical features are prominent and seem to be atypical. What is the most probable explanation for these features?

Wording maintains meaning while relating to distractors

[Figure 37]

Plausible spherical structure

- A) Concentration of cellular waste products leading to distinct morphological anomalies
- B) Formation of air bubbles introduced during sample immobilization processes
- C) Ice contamination from liquid nitrogen during sample preparation
- D) Inclusion bodies formed by aggregated proteins expressing during stress response
- E) Cellular morphological changes during mitotic division Correct answer: C

- A) Aggregation of human mHTT protein within the neuron,
- B) Ice contamination from liquid nitrogen during sample preparation
- C) Formation of intracellular vesicles during neuron development
- D) Artifacts arising from excessive heavy metal staining Correct answer: B

Also involves sample processing, correct answer stands-out less

Plausible spherical structure

More general wording preserves content

###### Image MCQ stage 1 MCQ stage 2

Gives away image specifics

Question: In a confocal microscopy image of HEK293T cells, the gene PSMB7 is endogenously tagged with GFP. One cell appears significantly larger than the others and contains multiple nuclei. What is the most likely reason for the presence of multi-nucleated cells in this culture?

Question: In a confocal microscopy image that displays an immortalized line of epithelial cells tagged with a fluorescent protein, one cell is notably larger than its neighbors and contains multiple nuclei. Considering possible observations, what could be the leading cause of multinucleation in this scenario?

[Figure 38]

Easy to rule-out: unconnected to question

- A) Increased rates of apoptosis in the culture
- B) Cell fusion events leading to multinucleation
- C) Normal occurrence in immortalized cell lines
- D) Contamination with another cell type Correct answer: C

- A) Enhanced nuclear division without cytokinesis completion during cell replication
- B) Cellular response to an excessive nutrient-rich environment
- C) Abnormal chromosomal segregation during meiosis
- D) The accidental integration of another distinct cell line into the culture
- E) Natural behavior in certain transformed eukaryotic cells Correct answer: E

Relate better to question

###### Image MCQ stage 1 MCQ stage 2

Maintains key information

Gives away image modality

Question: Microscopy images of eukaryotic cells stained for nuclei (blue), cytoskeleton (red), and a nuclear-associated protein (green) show areas with unclear nuclear boundaries and faint cytoskeletal signals. The cells appear more compressed than anticipated. What might be the most plausible cause for this observation?

Question: Confocal microscopy images of U2OS cells were prepared by staining nuclei with DAPI (blue), microtubules with an αtubulin antibody (red), and the protein TREX1 with a green antibody. In one of the images, the cells in the upper and lower parts appear to have blurred nuclei and partially visible microtubules, making the cells look smaller than expected. What is the most likely reason for this observation?

[Figure 39]

- A) Inadequate washing leading to excess dye retention
- B) Slight misalignment of the optical components causing image distortion
- C) Improper focal settings creating unfocused regions
- D) Irregular light source intensity impacting image clarity
- E) Low temperature causing cellular contraction affecting image details Correct answer: C

- A) Overfixation of cells leading to excessive cross-linking
- B) Underexposure during image acquisition causing insufficient signal
- C) Out-of-focus sections resulting in blurred cellular structures
- D) Non-specific antibody binding causing background fluorescence Correct answer: C

Hard distractor: links question Easy to rule-out: unrelated to and image content

the question

- Figure 11. Examples of changes to questions and options between stage 1 and stage 2 (RefineBot) of our MCQ generation method. In red elements that need to be improved and in green improvements.

#### F. Experiments details

##### F.1. Evaluation prompts

All prompts except for LLaVA and LLaVA-Med used the chain-of-thought prompt template from the MMMU-Pro code [88]:

|The following is a multiple choice question (with answers). Think step by step and then output the answer in the format of \"The answer is (X)\" at the end.<br><br>{{QUESTION}} Options: {{CHOICES}}|
|---|

An example complete question is:

|The following is a multiple choice question (with answers). Think step by step and then output the answer in the format of "The answer is (X)" at the end.<br><br>A fluorescence microscopy image displays cultured mouse primary neurons stained with a specific marker localized<br><br>to mitochondria. The fluorescence signal exhibits an uneven distribution across the neuronal cell, and there is an absence of signal within the nucleus. What is the most probable cause for the uneven distribution of the mitochondrial signal?<br><br>Options:<br><br>(1): Mitochondrial transport along axonal and dendritic processes ensures uneven distribution based on cellular energy demands.<br>(2): Differential expression of mitochondrial proteins leads to variable fluorescence signal across the neuron.<br>(3): Mitochondrial biogenesis occurs at specific neuronal regions, causing localized fluorescence accumulation.<br>(4): Mitochondrial anchoring to the cytoskeleton restricts their movement, resulting in uneven distribution.<br>(5): Mitochondrial dynamics, including fission and fusion, allow mitochondria to move and distribute throughout the neuron.<br>|
|---|

The answer is then extracted with the regex pattern: answer is \(?([0-9])\)?.

For LLaVA-Med, we added an extra parsing strategy. This was because we observed that while LLaVA-Med often gave reasonable responses to questions, it failed to follow instructions well. First, it did not follow the CoT prompt consistently to think step by step. It also didn’t use the right answer formatting: "The answer is (X)\". Since we were more interested in biomedical reasoning adhering to VQA formatting, we decided to handle this model differently.

Specifically, to encourage CoT reasoning, we fixed the prefix of the response to:

Response: Let’s think through this step by step and evaluate each option.. Then we added a suffix to the end:

After carefully considering all the options, the most likely answer is (, Then we continued generation, and did regular regex parsing to get the final answer.

We also find that a few models make minor but consistent errors with formatting the answer and elect to fix them. For example x-ai/grok-2-vision-1212 commonly adds two asterisks before the answer like “The answer is **(4)**” instead of “The answer is (4)” (GPT-4o also did this sometimes). Some other common errors that we saw but did not handle (because it would be difficult to do rigorously) was shown by meta-llama/llama-3.2-11b-vision-instruct: the response would write one of the MCQ options but not state its number.

##### F.2. Model details

Our main benchmarking experiments in Tab. 3 refer to models that have multiple versions. The exact model versions used for inference are shown in Tab. 6

Model Name API Endpoint Source & Details o1 o1-2024-12-17 OpenAI API

Claude-3.5-Sonnet *anthropic/claude-3.5-sonnet-20240620 Openrouter API Gemini-Pro-1.5 google/gemini-pro-1.5 Openrouter API Pixtral-Large mistralai/pixtral-large-2411 Openrouter API Grok-2-Vision x-ai/grok-2-vision-1212 Openrouter API Qwen-2-vl-72b-Instruct qwen/qwen-2-vl-72b-instruct Openrouter API VILA1.5-40b VILA1.5-40b HuggingFace, local inference GPT-4o gpt-4o-2024-08-06 OpenAI API

- Llama-3.1-Nemotron-70b-Instruct nvidia/llama-3.1-nemotron-70b-instruct Openrouter API
- Llama-3.2-90b-Vision-Instruct meta-llama/llama-3.2-90b-vision-instruct Openrouter API Qwen-2-VL-7b qwen/qwen-2-vl-7b-instruct Openrouter API Claude-3.5-Haiku anthropic/claude-3.5-haiku Openrouter API Gemini-Flash-1.5-8b google/gemini-flash-1.5-8b Openrouter API GPT-4o-mini gpt-4o-mini-2024-07-18 OpenAI API Pixtral-12b mistralai/pixtral-12b Openrouter API VILA1.5-13b VILA1.5-13b HuggingFace, local inference Llama-3.2-11b-vision-instruct meta-llama/llama-3.2-11b-vision-instruct Openrouter API LLaVA-Med-Mistral-7B LLaVA-Med-Mistral-7B HuggingFace, local inference LLaVA-Mistral-7B LLaVA-Mistral-7B HuggingFace, local inference

Table 6. AI model API endpoints and their sources

For LLaVA and LLaVA-med, although the version number 1.6 is different from LLaVA-Med with 1.5, we do believe that LLaVA-Med was either trained from this LLaVA version, or from one that is very similar. They have identical architecture, and the version is very close; it is possible the LLaVA-Med copied the number incorrectly. The LLaVA-Med codebase or github issues does not clarify this at the time of writing.

Note that for OpenRouter function calls to Anthropic, the date is not included in the string. We believe they use timestamps for older model versions, but not for the latest version. So reproducing it may require checking if the model has been updated.

- F.3. Human baseline on MicroVQA To perform a human baseline, we recruited co-authors who created questions. The conditions were:

- • Questions were provided in a google form.
- • Participants did not view any question that they created.
- • Internet access was allowed, but LLM access was not. This is because realistic biological practice is knowledge-intensive, and may require looking up facts or references.
- • Time limit was 3 minutes per question. They were free to take as many breaks as they wanted.
- • They answered the multiple choice question without further written justification.

These conditions – 3 minutes and internet access – give sufficient time for an expert to make a reasonable attempt. Since the score was far below 100%, we claim that these questions could be of value to real experts as an LLM chat assistant. Each participant only took a subset of the questions (about 100 each). The final score is the aggregate.

- F.4. More experiments on attribute-level information

Models

Gemini-1.5-Pro

VILA1.5-40B

LlaVA-Med-Mistral-7B

[Figure 40]

Accuracy

Electron microscopy

Fluorescence microscopy

Light microscopy

Figure 12. Performance by image modality type for the best models: Gemini-1.5-Pro (closed source), VILA1.5-40B (open-source), and LlaVA-Med-Mistral-7B (medical).

We described our metadata features in Sec. 3, and we can now leverage them to benchmark models on different attributes. Fig. 5 (left) shows all sub-tasks, and shows that ‘compare image sets’, which is common in biology to compare ‘treatment’ and ‘control’ groups. All models perform relatively well, despite multi-image reasoning being understudied in MLLMs [78]. Fig. 5 (right) shows that for Gemini and LLaVa-Med, higher Bloom’s levels – corresponding to more advanced reasoning – leads to worse performance. This supports the intuitive result that higher-level reasoning is more challenging – although VILA does not follow this trend. A final result in Fig. 12 shows that all models performed worse on cryo-electron microscopy images than brightfield or fluorescence microscopy. This may be due to cryo-EM’s challenging features (low contrast, low signal-to-noise ratio) and their under-representation in pretraining datasets [93].

- F.5. Language shortcut ablations

In Tab. 4 we do language shortcut ablations for some top-performing models and some lower-performing models – both no-image and choices-only ablations.

Table 7. Ablation study on MicroVQA MCQs. Column 2 is overall accuracy. Cols 3-4 are the no-image ablation accuracy and drop compared to overall accuracy. Cols 5-6 are the choices-only ablation accuracy and drop compared to overall accuracy. See the text for discussion.

|Accuracy|No-image ablation<br><br>Accuracy Drop|Choices-only ablation<br><br>Accuracy diff<br><br>|
|---|---|---|
|o1 52.8 Claude-3.5-Sonnet 51.7 Gemini-Pro-1.5 51.1 Pixtral-Large 49.8 Grok-2-Vision 48.4 GPT-4o-mini 46.2 Pixtral-12b 45.6<br><br>|49.2 -3.6<br><br>46.0 -5.8<br>47.2 -3.8 46.3 -3.6 46.3 -2.1 44.2 -1.9 43.7 -1.9<br>|37.7 -15.1<br><br>44.0 -7.7<br><br>36.8 -14.3 36.7 -13.1 40.5 -7.9<br><br>34.3 -11.9 31.8 -13.8<br><br>|
|Random 22.0| | |

C G P G G P

The no-image ablation is a common test in VQA evaluations[29, 74]. The MLLM only receives the text prompt, and we add the following sentence to the prompt template:

If an image is mentioned ignore this information and try your best to answer the question. The choices-only ablation has been explored in (non-visual) question answering [7]. The model does not receive the image or the question and must guess an answer from only the choices. We add this sentence to the prompt template:

However I will not give you the question text or the images, I will only give you the choices, so please try your best to answer the question.

Both ablations lower the scores, but they are still above random. While this is initially surprising, it is actually consistent with other benchmarks. A key challenge in VQA benchmarks is for questions to be “vision-centric”: that the question requires using the image features to answer. We still have very strong evidence that MicroVQA is vision-centric, which we discuss below.

First, let’s recall the language shortcuts by considering a VQA sample with an image of a green field, with the question “what color is the grass”.

- • ‘Language bias’ is well-known for enabling models to make educated guesses about MCQs without access to the image [29]. For the question “what color is the grass?”, then without the image, “green” is a likely guess, although there are other possible answers, like “brown”.
- • ‘Visual giveaway’ means there are too many image details in the question, for example if the question were “what color is the green grass?” This removes any need to look at the image.
- • ‘Weak distractors’ gives other question choices that are easily eliminated. For example, for question “what color is the grass?”, a bad distractors would be “pink”. This also removes any need to look at the image.

(In Appendix E.6, we also discussed language shortcuts with biology examples as it related to MCQ construction.) Now we argue that MicroVQA is still vision-centric despite the ablation results.

###### 1. Language bias allows cheating on vision-centric questions If answerable due to a ‘language bias’, the question may

still require perception. A model may guess that the grass is green, but it is only a guess – the grass really could be brown [29]. Therefore correct answers under the ‘no-image’ ablation may still be vision-centric. On the other hand problems with ‘visual giveaway’ and ‘weak distractor’ shortcuts really do render the question not-vision-centric. The RefineBot method we employed in MCQ generation was designed to target language shortcut removal. Anecdotally, our qualitative analysis (from Appendix E.6) found that RefineBot tends to focus on fixing the visual giveaway and weak distractor shortcuts. Ideally we would perform a quantitative analysis to identify which language shortcuts are due to language bias versus the others, however analyzing these questions is very time-consuming.

###### 2. The unsolved questions are hard because of perception. The ‘no-image’ ablation only tells us about the ‘easy’

questions, but half the quesitons are too hard for current models. Our qualitative analysis of 30 failure cases demonstrates that half of them made critical perception errors, which strongly suggests they require image understanding

In contrast to the ‘vision-centric’ claims from the Cambrian paper [74]: They show that many VQA benchmarks have no-image ablation scores only slightly underperforming the with-image scores, and they conclude that such VQA benchmarks “do not properly measure vision-centric capabilities”. But that conclusion cannot be drawn from only the no-image ablation

(following our point 2 above). For benchmarks where MLLM’s score far from 100%, the failures could mostly be due to poor image understanding. This is very plausible since visual capabilities in MLLMs are viewed as lagging behind language capabilities. Likewise, our experience with our microscopy data was that visual understanding was poor, but that language knowledge was impressive.

In point 1 above, we argued that MLLMs can use language biases to pass the no-image ablation, even if they are visioncentric questions. This is actually in agreement with the claim from Cambrian that those questions don’t test vision-centric capabilities.

Finally, some non-vision centric questions are acceptable. The goal of MicroVQA is to source realistic problems from microscopy research. While this area does lean heavily on image analysis, not every question will realistically require perception, so this is acceptable.

##### F.6. Error analysis

Here we show samples from the manual qualitative error analysis (details in the main paper). To see all 30 examples, refer the github repo.

Image key 5

###### F.6.1 Perception errors

[Figure 41]

Q: A cryo-electron tomography (cryoET) slice of mammalian cells was acquired at a resolution of 1.3 Å and binned by 4 (resulting in a 5.3 Å per pixel). Within the tomogram, dark circular like structures are observed in an organelle. What is the modest likely identity of these structures?

- A) Ribosomes
- B) Phase-separated condensates such as stress granules
- C) Lysosomes
- D) Peroxisomes

VLM Answer: Let me think about this step by step:

- 1) First, let's look at what we see in the image:

- - A cryo-electron tomography slice showing cellular structures
- - Circular-like dark structures visible
- - The resolution is quite high (5.3 Å after binning)

- 2) Let's consider the size and appearance of these structures:

- - They appear as dense, small, roughly circular objects
- - The size is consistent with ribosomes (approximately 25-30 nm)
- - They show the characteristic electron density pattern of ribosomes

- 3) Let's analyze each option:

- - Ribosomes: Match the size, shape, and density pattern
- - Phase-separated condensates: Usually less defined, more amorphous
- - Lysosomes: Much larger, membrane-bound organelles
- - Peroxisomes: Also larger, membrane-bound organelles

- 4) Additional considerations:

Subject: Cell and molecular biology Bloom’s level: 4 (Analysis) Specimen: mammalian cells

This is a slice of cryoET tomogram of mammalian cells acquired at 1.3 A but binned by 4 (5.3 A/pix)

- - Ribosomes are often found associated with mitochondria
- - The resolution is sufficient to resolve ribosomal structures
- - The contrast and density are typical of ribosome imaging in cryoET The answer is (1)

Error type: Perception

Error reason: These are granules characterized by heterogeneous, electron-dense aggregates. Ribosomes would be smaller, more regular in shape, and not have as high contrast.

Figure 13

Image key 157

[Figure 42]

Q: A hematoxylin and eosin (H&E)-stained micrograph shows cellular formations within a gastrointestinal tissue sample with unique organization and cellular patterns. What pattern and morphology do these cells most closely align with?

- A) Bundles of elongated cells with pale pink cytoplasm and uniform oval nuclei in a crisscross pattern
- B) Circular groups of cells with mucin in central spaces and small, dark nuclei
- C) Dense arrays of polygonal cells with small nucleoli and vacuolated cytoplasm
- D) Rows of cells with conspicuous cytoplasm and rounded nuclei
- E) Diffused arrangement of cells with transparent cytoplasm and irregularly clustered chromatin

VLM Answer: Let me analyze this step by step:

- 1) Looking at the H&E-stained micrograph:

- - It shows a circular cross-section of gastrointestinal tissue
- - There's a clear organizational pattern along the periphery
- - The center shows pink-stained connective tissue (lamina propria)

- 2) Focusing on the cellular arrangement:

- - The cells are arranged in circular groups
- - These groups form glandular structures
- - There are clear central spaces (lumens) in these structures
- - The nuclei appear dark and are located at the base of the cells

- 3) Comparing with the options:

- - Option 1 describes elongated cells in crisscross pattern - doesn't match
- - Option 2 describes circular groups with mucin in central spaces and small, dark nuclei matches perfectly
- - Option 3 describes polygonal cells with vacuoles - doesn't match
- - Option 4 describes rows of cells - doesn't match
- - Option 5 describes diffused arrangement - doesn't match

- 4) The image shows typical intestinal glands (crypts) with:

Subject: Anatomy Bloom’s level: 4 (Analysis) Specimen: gastrointestinal lesion biopsy

H&E-stained; micrograph of gastrointestinal lesion biopsy

- - Circular arrangement
- - Central lumens containing mucin
- - Small, dark nuclei at the base of the cells The answer is (2)

Error type: Perception

Error reason: No. The intestinal glands are not the lesion. There is a pink spindle cell lesion with a fascicular arrangement of cells

###### Figure 14

[Figure 43]

Q: An H&E-stained image of a nerve sheath tumor displays unique cell arrangements and marked changes within its structure. Given these observations, what is the most likely reason for the observed morphological alternations?

- A) Indication of rapid cellular proliferation
- B) Sign of tumor shrinkage post-treatment
- C) Result of genetic changes within the tumor cells
- D) Outcome of a chronic process affecting the tumor
- E) Sign of metastasis to another location

VLM Answer: Let me analyze this step by step:

- 1) Looking at the H&E-stained images, I can observe:

- - Dense cellular proliferation
- - Distinctive "herringbone" or swirling pattern
- - High cellularity with numerous nuclei
- - Evidence of active growth pattern

- 2) Key characteristics visible:

- - The cells are arranged in fascicles
- - There's a high nuclear to cytoplasmic ratio
- - The pattern suggests active tumor growth
- - No signs of necrosis or treatment effect
- - No obvious signs of metastatic spread

- 3) Analysis of the options:

- - Option 1 matches with the high cellularity and growth pattern
- - Option 2 doesn't fit as there's no evidence of shrinkage
- - Option 3 while possible, isn't directly evident in morphology
- - Option 4 doesn't match the active growth pattern seen
- - Option 5 isn't supported by the histological features shown

- 4) The morphological features strongly suggest active tumor growth with:

[Figure 44]

- - Characteristic fascicular pattern
- - High cellular density
- - Well-preserved tumor architecture
- - Signs of active proliferation The answer is (1)

Subject: Pathology Bloom’s level: 4 (Analysis) Specimen: schwannoma tissue

Error type: Perception

Error reason: No. One image shows cystic change, which can be seen in chronic, longstanding schwannomas. The other shows Verocay bodies and fascicular growth, which do not indicate rapid cellular proliferation.

###### Figure 15

[Figure 45]

Q: In a fluorescence microscopy image comparing wounded and unwounded liver tissues stained to visualize markers of immune response – neutrophil derived migrasomes (Ly6G, yellow) and platelets (CD41, magenta), what is most likely observed about the distribution of these markers in unwounded tissue?

- A) Slight overlap without any notable biological consequences
- B) Substantial overlap implying persistent immune activation
- C) No overlap, indicating complete segregation of cellular functions
- D) Randomized distribution suggesting passive circulation
- E) Significant separation highlighting cellular non-interaction

VLM Answer: Let me think about this step by step:

- 1) The image shows two types of staining:

- - Yellow (Ly6G) for neutrophil-derived migrasomes
- - Magenta (CD41) for platelets

- 2) Looking at the unwounded tissue (presumably the left image):

- - There are distinct yellow dots (migrasomes)
- - There are magenta structures (platelets) - The distribution appears fairly separated - The markers rarely overlap

- 3) This makes biological sense because:

- - In healthy/unwounded tissue, immune cells and platelets typically maintain their distinct roles
- - Without injury, there's no strong stimulus for interaction
- - The cells are likely performing their normal surveillance functions

- 4) Analyzing the options:

- - Option 1 suggests slight overlap - more than what's visible
- - Option 2 suggests high overlap - not seen
- - Option 3 indicates complete segregation - matches the image
- - Option 4 suggests random distribution - the pattern looks more organized
- - Option 5 also suggests separation - similar to option 3

- 5) Option 3 best matches what we observe: "No overlap, indicating complete segregation of cellular functions" The answer is (3)

[Figure 46]

Subject: Physiology Bloom’s level: 2 (Comprehension) Specimen: mouse liver tissue (wounded and unwounded)

Error type: Perception

Error reason: The markers rarely overlap, but there is very focal colocalization in some puncta. The selected answer argues for “complete segregation of cellular functions”, which is a too extreme or absolute conclusion based on the image.

###### Figure 16

###### F.6.2 Knowledge errors

[Figure 47]

Q: A double immunofluorescence experiment was conducted on nerve cells, highlighting Sema4B (red marker) and Sox9, an astrocyte marker (green marker). Sema4B was found primarily surrounding the nuclear area dispersed in the cytoplasm. Based on this pattern, what potential role might Sema4B play in these cells?

- A) Sema4B contributes to the regulation of metabolic pathways within the cytoplasm.
- B) Sema4B is associated with intracellular trafficking processes.
- C) Sema4B acts as a signaling molecule interacting with cell surface receptors.
- D) Sema4B aids in transporting proteins across cell compartments.
- E) Sema4B is directly involved in gene transcription regulation inside the nucleus.

VLM Answer: Let me solve this step by step:

- 1) Looking at the immunofluorescence images:

- - Sox9 (green) marks astrocytes
- - Sema4B (red) shows a particular distribution pattern
- - The merged image shows both markers plus nuclear staining (blue)

- 2) Analyzing Sema4B's distribution:

- - It appears as red punctate staining
- - Located primarily around nuclear areas
- - Shows dispersed pattern in cytoplasm
- - Not concentrated in nucleus itself

- 3) This distribution pattern suggests:

- - The protein is involved in cytoplasmic processes
- - Shows vesicular-like pattern
- - Located in areas suggesting trafficking or transport

- 4) Looking at the options:

- - Option 4 mentions protein transport
- - This matches the vesicular-like pattern
- - The perinuclear and cytoplasmic distribution is consistent with protein trafficking
- - Other options don't align as well with the observed distribution pattern

- 5) Reasoning:

- - The punctate pattern and distribution is characteristic of proteins involved in cellular trafficking
- - The location around nucleus (where protein synthesis occurs) and throughout cytoplasm supports a transport role
- - This pattern is less consistent with metabolic, signaling, or transcriptional roles The answer is (4)

Subject: Neurobiology

- Bloom’s level: 3 (Application) Specimen: Nerve cells

Error type: Misconception (knowledge)

Error reason: Sema4B is a transmembrane protein that plays a role in cellular signaling. TM proteins can be internalized via endocytosis and does not mean that its main function is intracellular trafficking.

###### Figure 17

[Figure 48]

Q: A cryo-Electron Tomography image of fibroblasts from a 70-year-old patient with Alzheimer’s disease shows three mitochondria gathered together. What is the most likely reason for this mitochondrial clustering.

- A) Increased mitochondrial biogenesis
- B) Enhanced mitochondrial fusion leading to interconnected networks
- C) Mitochondrial fission due to cellular stress
- D) Impaired mitophagy causing accumulation of damaged mitochondria

VLM Answer: Let's solve this step by step:

- 1) First, let's consider the context:

- - This is from a 70-year-old patient with Alzheimer's disease
- - Alzheimer's is known to affect cellular function, including mitochondrial health
- - We're observing clustered mitochondria in the image

- 2) Analyzing each option:

- - Option 1 (Increased biogenesis): Would show new, healthy mitochondria, not clustering
- - Option 2 (Enhanced fusion): Would show elongated, connected mitochondria
- - Option 3 (Mitochondrial fission): Would show smaller, fragmented mitochondria
- - Option 4 (Impaired mitophagy): Would show accumulated, potentially damaged mitochondria

- 3) Looking at the image:

- - We see multiple mitochondria clustered together
- - They appear to be intact but gathered in groups
- - This suggests they're not being properly cleared from the cell

- 4) Considering Alzheimer's disease:

- - It's known to impair cellular cleanup mechanisms
- - Mitophagy is a key quality control process for mitochondria
- - When impaired, damaged mitochondria accumulate instead of being removed

- 5) Conclusion: The clustering is most likely due to impaired mitophagy, as this would explain why multiple mitochondria are gathering instead of being properly cleared from the cell, which is a known issue in Alzheimer's disease. The answer is (4)

Subject: Neurobiology

- Bloom’s level: 4 (Analysis) Specimen: Fibroblasts from an Alzheimer’s patient

Error type: Misconception (knowledge)

Error reason: Impaired mitophagy may result in larger overall mitochondria, not an increase in small mitochondrial aggregates. Fission is a better answer in this case.

###### Figure 18

[Figure 49]

Q: A microscopic analysis of tissue from an individual suffering from recurrent seizures involves using hematoxylin & eosin stain alongside a vimentin stain. Cells with stronger vimentin expression are significantly larger than surrounding entities. Which identity is most probable for these prominent vimentin-expressing cells?

- A) Dystrophic neurons typically found around glial scars
- B) Enlarged reactive astrocytes near injury sites C) Aberrant large cells characteristic of cortical malformation

D) Macrophages adapted to engulf cellular debris E) Fibroblast proliferation in response to brain injury

VLM Answer: Let me solve this step by step:

- 1) First, let's analyze what we're seeing in the images:

- - The H&E stain (top image) shows tissue with scattered cells and some enlarged cells
- - The vimentin stain (bottom image) shows strong brown staining of certain larger cells

- 2) Key information from the question:

- - Patient has recurrent seizures
- - Vimentin-positive cells are significantly larger
- - Looking for most probable identity of these cells

- 3) Important facts about vimentin:

- - Vimentin is an intermediate filament protein
- - Strongly expressed in reactive astrocytes
- - Also expressed in mesenchymal cells

- 4) In the context of seizures:

- - Seizures can cause brain injury
- - Brain injury triggers astrogliosis
- - Astrocytes become reactive and enlarged in response to injury

- 5) Looking at the options:

- - Option 1: Neurons don't typically express vimentin
- - Option 2: Reactive astrocytes do express vimentin and enlarge during injury
- - Option 3: Cortical malformation cells wouldn't necessarily express vimentin
- - Option 4: Macrophages are typically smaller
- - Option 5: Fibroblasts would be rare in this context

- 6) The morphology and staining pattern of the large cells, combined with the clinical context of seizures, most strongly suggests reactive astrocytes. The answer is (2)

[Figure 50]

Subject: Neurobiology

- Bloom’s level: 3 (Application) Specimen: Resection specimen for epilepsy

Error type: Misconception (knowledge)

Error reason: The AI is able to interpret the question and image, but lacks the knowledge to recognize the link between recurrent seizures, cortical dysplasia, and large ballooned cells.

###### Figure 19

[Figure 51]

Q: A certain specialized stain reveals unusual cell movement crossing boundaries typical of neuronal migration in a tissue resection from a case of structural brain anomaly. Which condition is primarily associated with this type of cellular pattern?

A) Focal cortical dysplasia B) Congenital Zika syndrome C) Schizencephaly

- D) Type II lissencephaly
- E) Walker-Warburg syndrome

VLM Answer: Let me solve this step by step:

- 1) The image shows a histological section with specialized staining that reveals abnormal cellular organization and migration patterns.
- 2) The question asks about unusual cell movement crossing typical neuronal migration boundaries.
- 3) Looking at the options:

- - Focal cortical dysplasia involves disorganized neurons in localized areas
- - Congenital Zika syndrome primarily affects neural progenitor cells
- - Schizencephaly involves clefts in the cerebral hemispheres
- - Type II lissencephaly shows abnormal neuronal migration and positioning
- - Walker-Warburg syndrome is a severe form of congenital muscular dystrophy with brain malformations

- 4) Key characteristics of focal cortical dysplasia include:

- - Abnormal neuronal migration
- - Disorganized cortical layering
- - Abnormal cell types crossing normal boundaries
- - Disrupted cerebral cortex architecture

- 5) The histological image shows features consistent with focal cortical dysplasia, including:

Subject: Neurobiology

- Bloom’s level: 4 (Analysis) Specimen: Brain parenchyma

- - Disrupted cellular organization
- - Abnormal migration patterns
- - Cells crossing typical boundaries The answer is (1)

Error type: Misconception (knowledge)

Error reason: Misconception about the scope and extent of neuronal migration patterns in focal cortical dysplasia (FCD) versus Type II lissencephaly. FCD will have abnormal cortical lamination, but the neurons are still within the expected gray/white matter of normal neuronal migration. They do not extend beyond and cross established neuronal boundaries such as the glial limitans.

###### Figure 20

###### F.6.3 Overgeneralization or oversimplification errors

Q: In a histological examination, an H&E-stained image shows Verocay bodies and cystic degeneration with pseudoepithelial metaplasia, indicative of schwannoma. How does the risk of malignant transformation in schwannoma compare to other peripheral nerve sheath tumors?

[Figure 52]

- A) Schwannomas exhibit a lower risk of malignant transformation compared to other peripheral nerve sheath tumors.
- B) Schwannomas and neurofibromas have similar risks of malignant transformation.
- C) The risk of malignant transformation in schwannomas is dependent on specific genetic factors.
- D) Schwannomas have an increased risk of malignancy in the presence of multiple tumors.
- E) Schwannomas rarely undergo malignant transformation, regardless of tumor location or size. VLM Answer: Let me think about this step by step:

- 1) The images show typical histological features of schwannoma with:

- - Verocay bodies (palisading nuclei)
- - Cystic degeneration
- - Pseudoepithelial metaplasia

- 2) Regarding malignant transformation in schwannomas:

- - Schwannomas are typically benign tumors
- - They rarely undergo malignant transformation - This is true regardless of their size or location - Even multiple schwannomas (as in schwannomatosis) rarely become malignant

- 3) Comparing to other peripheral nerve sheath tumors:

- - Neurofibromas, especially plexiform types, have a higher risk of malignant transformation
- - This is particularly true in the setting of NF1 (neurofibromatosis type 1)
- - Malignant peripheral nerve sheath tumors (MPNSTs) more commonly arise from neurofibromas than schwannomas

- 4) The statement about genetic factors or increased risk with multiple tumors isn't accurate for schwannomas, though it may be true for other nerve sheath tumors
- 5) Looking at the options, the key is that schwannomas maintain their benign nature in most cases, regardless of circumstances. The answer is (5)

[Figure 53]

Subject: Pathology Bloom’s level: 2 (Comprehension) Specimen: schwannoma tissue

Error type: Overgeneralization

Error reason: The selected choice does not address the question asked “compared to other peripheral nerve sheath tumors?”. The “best answer” for the question should compare to other tumors as specified in the question. The AI has “oversimplified” the question to ask a simpler question than the one asked.

Figure 21

[Figure 54]

Q: Confocal microscopy images depict live HEK293 cells subjected to CRISPR/Cas9-mediated knockdown of a specific gene, alongside wildtype cells. The cells are stained with BODIPY-C12 (cyan) to highlight lipid droplets and Hoechst to stain the nuclei. In wildtype cells, distinct organellar structures are visible, whereas these structures are absent in the gene knockdown cells. What is the most likely cellular outcome resulting from the knockdown of this gene in HEK293 cells?

- A) Disruption of mitochondrial fission
- B) Impaired peroxisome biogenesis
- C) Altered lipid metabolism pathways
- D) Reduced endoplasmic reticulum function
- E) Decreased nuclear transport processes

VLM Answer: Let me solve this step by step:

- 1) First, let's analyze what we're seeing in the images:

- - Live HEK293 cells are shown
- - BODIPY-C12 staining (cyan) shows lipid droplets
- - Hoechst staining shows nuclei (grey/white)
- - The images show a time series (left to right)

- 2) Key observations:

- - The cyan staining (BODIPY-C12) indicates the presence of lipid droplets
- - There's a difference between wildtype and knockdown cells in terms of organellar structures
- - The staining specifically targets lipids

- 3) Consider the options:

- - Mitochondrial fission would primarily affect energy metabolism
- - Peroxisome biogenesis affects lipid breakdown
- - Lipid metabolism directly relates to lipid droplet formation
- - ER function is broader than just lipid effects
- - Nuclear transport isn't directly related to lipid droplets

- 4) Reasoning:

- - Since BODIPY-C12 specifically stains lipids
- - The visible differences are in lipid-containing structures
- - The most direct connection would be to lipid metabolism
- - Changes in lipid metabolism would directly affect lipid droplet formation and distribution The answer is (3)

Subject: Cell and molecular biology Bloom’s level: 4 (Analysis) Specimen: HEK293 cells (wildtype and PEX19 knockdown)

Error type: Overgeneralization

Error reason: Assumes that the changes in lipid droplets are directly linked to altered lipid metabolism pathways, without fully integrating the implications of peroxisome biogenesis in lipid metabolism and organelle dynamics.

Figure 22

###### F.6.4 Hallucination or errors due to added details

[Figure 55]

Q: A CryoET tomographic slice image and its corresponding 3D annotation of cellulose fibers in vitro show differing amounts of cellulose fibers. Beyond the 2D nature of the slice versus the 3D nature of the annotation, which mechanism most likely accounts for the observed differences?

A) Errors in the alignment during image reconstruction leading to artifacts. B) Overlapping fibers being misinterpreted as single fibers due to resolution limits.

- C) Variability in manual annotation leading to subjective bias.
- D) Sample dehydration causing shrinkage of fibers in the image. E) Background noise in the image causing misidentification of fibers.

VLM Answer: Let me solve this step by step:

- 1) First, let's analyze what we're seeing:

- - Left image: CryoET tomographic slice showing cellulose fibers in grayscale
- - Right image: 3D annotation of the fibers in blue
- - There appears to be a difference in how many fibers are visible/annotated

- 2) Consider the limitations of CryoET imaging:

- - Resolution is limited
- - Multiple fibers can appear to overlap in 2D views - Depth information can be challenging to interpret

- 3) Evaluate each option:

- - (1) Alignment errors typically cause obvious artifacts, which aren't apparent here
- - (2) Resolution limits could cause overlapping fibers to appear as one in the 2D slice
- - (3) Manual annotation variability wouldn't explain systematic differences
- - (4) Sample dehydration would affect both views equally
- - (5) Background noise might affect fiber identification but isn't the primary issue

- 4) The key observation is that the 2D slice shows fewer distinct fibers than the 3D annotation, likely because:

- - Multiple fibers that overlap in the 2D view can be properly separated in 3D
- - The resolution limits of the microscope make it difficult to distinguish overlapping fibers in the 2D view The answer is (2)

Subject: Cell and molecular biology Bloom’s level: 4 (Analysis) Specimen: cellulose fibers in vitro

Error type: Hallucination

Error reason: The question does not state there are systematic differences. This detail is added during the AI CoT reasoning. A better answer would rule out easy or obvious causes of differences (random or systematic). Although fibers can overlap in Cryo-EM, it is generally considered to have a reasonably high resolution.

Figure 23

##### F.7. Reasoning trace analysis

###### F.7.1 Human step-by-step reasoning

To better understand how subject matter experts approach questions in MicroVQA, we have a cell biologist and microscopist manually annotate their reasoning for each of the three main use cases.

- Reasoning example for task 1 ‘expert visual understanding’.

|*Question* What subcellular structure could the bright green spots represent?*<br>*Answer* This staining pattern most likely refers to cytoplasmic bodies.<br>*Reasoning:*<br><br><br>- Find green spots (no reasoning required)<br>- Rule out staining artefacts<br><br>- Look at marker channels to assess whether cells appear intact<br>- Compare the green spots to the marker channels for nuclei or microtubules<br>- Check that green spots are always inside the cells<br><br><br>- Yes, inside cells, so these are no staining artefact.<br><br>- Look up which spotty staining patterns in cells exist<br><br>- Nuclear bodies<br>- Nuclear speckles<br>- Nucleoli<br>- Cytoplasmic bodies<br>- Centrosome<br>- Vesicles<br>- Intermediate filaments<br>- Spotty microtubule staining<br><br><br>- Explore which patterns can be excluded<br>- Check whether green spots are inside or outside of nuclei to know what patterns to explore<br><br>- Outside of nuclei, thus nuclear subcellular localizations can be excluded, and cytoplasmic spotty patterns need to be explored<br><br>- Check whether green spots are placed on microtubules<br><br>- No, thus cannot be spotty microtubule stain<br><br>- Assess localization > if accumulating around microtubule organizing center could be likely vesicles or centriolar satellites<br><br>- No, not accumulating at centrosome<br><br>- Assess size and shape of the spots<br><br>- Small size and round, can exclude intermediate filaments<br><br>- Count green spots per cell and check that not more than 10<br><br><br>- Less likely vesicles, likely cytoplasmic bodies|
|---|

- Reasoning example for task 2 ‘hypothesis generation’.

- *Question* The sperm cells swim in circles. Wouldn’t they need to swim straight to reach the egg?
- *Answer* Sperm motility and its relation to reaching the egg and fertility is very complex and still not fully understood.

Yet, it is known that sperm swimming in circles in a shallow observation chamber like you might have applied, may swim on helical or twisted-ribbon structures in an in vivo or 3D space and thereby might in fact efficiently swim forward.

- *Reasoning*

- - Explore the swimming trajectories to validate what the user means with swimming in circles.

- Detect trajectory, assess how it would be described: curved or circular swimming trajectory

- - Explore what is reported on swimming trajectories of sperm cells.

- - Sperm cell swimming trajectories are species dependent

- Validate species:

- - Assess whether the user mentioned the species in the prompt.
- - Assess look of cells in image (morphology, geometry) and compare to other images with known species eventually

- Likely human sperm.

- - Sperm cell swimming trajectories are experimental setup dependent: shallow versus deep observation chamber

- Validate setup:

- Brightfield microscopy, sperm cells are always swimming parallel, little to no off-focus blurring of the tail.

- Assess how images from different experimental setups look like for human sperm

- Likely shallow cover-slip-based observation chamber

- Consequences of setup for trajectory In shallow observation chambers sperm cannot freely beat and thus beat parallel to

the glass surface, with no 3D component.

- - Sperm cells hitting a glass surface align their beat with the glass and this way can get trapped if asymmetrically beating.
- - Curved trajectory in 2D may relate to a progressive/straight swimming path in 3D

- - Sperm cell swimming trajectories are stimulus dependent and maturation-state dependent.
- - Validate setup and explore metadata provided:

- - Check literature and compare image to available images of sperm from healthy donors (if available).
- - Compare detected swimming trajectory to swimming trajectories of sperm explored in a similar species and experimental setup where sperm cells are perturbed (if available).
- - Are there indications for any treatment under which the detected swimming trajectory would be abnormal and how would these look like (if available).

- - No indications for any treatment. Neither in prompt nor in image.

- - Explore what is known on sperm cell trajectories to reach the egg in humans.

- - Not much is known since in vivo studies are not possible.
- - Sperm cells need to be able to develop hyperactive motility to swim to the egg

- How do hyperactive sperm cells look?

- Asymmetric beating

- What trajectory would they swim in in a 2D observation chamber?

- Curved trajectory

- Does that fit the swimming trajectory presented?

- Yes

- - Assess whether when summarizing all the obtained information the conclusion from the observed swimming trajectory to sperm not reaching the egg is valid.

- No.

- - Compose summarizing reply to support the conclusion.

- Reasoning example for task 3 ‘experiment proposal’.

|*Question* Can you suggest a simple experiment to find out whether centrosomal multiplication is occurring here?<br>*Answer* One option could be to co-stain the EDC3 antibody with a centrosomal marker like PCNT or CEP250.<br>*Reasoning*<br><br><br>- Extract hypothesis to be tested from the question<br><br>- EDC3 is labeling centrosomes and thus, centrosomal multiplication is occuring in the pictured cells<br><br>- Detect the experimental setup<br><br>- Immunofluorescent antibody staining on a protein of interest (EDC3) in a human cultured cell type<br><br>- Compare to similar experimental setups that support such a hypothesis<br><br>- Immunofluorescent antibody staining of centrosomes, co-labeling cell markers, performing microscopy, counting centrosomes<br>- Acquiring or creating a transgenic cell line with a GFP-labeled centrosomal proteins, followed by microscopy<br><br><br>- Determine which methods are compatible with EDC3 staining<br><br>- Both compatible with co-staining EDC3<br><br>- Determine which experiment is simplest for someone performing the shown experimental setup and if equal which one can be realized in less time<br><br>- Immunofluorescent staining of EDC3 and centrosomes<br><br>- Find centrosome markers for human cells and assess suitability for this cell type<br>- Final decision on experimental setup<br>|
|---|

###### F.7.2 LLM-based error analysis

To better understand how the model approaches questions in MicroVQA, we extend our manual reasoning trace analysis from a subset of the dataset to all questions by leveraging zero-shot prompting of an LLM. The automatic error categorization achieves 63% accuracy compared to manually assigned categories. Although imperfect, automated error categorization is scalable and preserves the overall distribution of human-identified errors, enabling broader analysis. We also classify correctly answered questions to quantify overall changes across experiments. Below, we present the prompts and category definitions used for classification.

Below is a multiple choice question with options and the reasoning that lead a model to an incorrect response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace to tag the error type:

- - Perception: the image was not interpreted correctly.
- - Overgeneralization: the details of the question were ignored and the general case was applied.
- - Hallucination: details were added during reasoning that weren’t in the question or extracted from the image.
- - Other: the error does not fit the above categories. Question:

{{question}} Correct answer: {{correct_answer}} Resoning trace: {{reasoning}}

|Below is a multiple choice question with options and the reasoning that lead a model to a correct response. Originally the model was also shown an image with the question. Your task is to use the reasoning trace and determine if the question was answered because of these reasons:<br><br>- No image: the image is unnecessary because the correct answer doesn’t rely on interpreting visual cues.<br>- Visual giveaway: the image is unnecessary because critical visual information is already described in the question itself.<br>- Language bias: the question has information that makes the correct option obvious.<br>- Weak distractors: the distractors are easy to rule out according to the reasoning trace.<br>- Good question: the question is well-constructed and needs the image and specialized knowledge to answer.<br>- Other: the question is hard to answer or doesn’t fit the other classes.<br><br><br>Question: {{question}}<br><br>Correct answer: {{correct_answer}} Resoning trace: {{reasoning}}|
|---|

[Figure 56]

[Figure 57]

Without image (ablation) With image

Perception 0.6%

Hallucination Good question 5.0%

Other wrong 5.3%

Other wrong 7.4%

13.9%

Weak distractors 20.9%

Hallucination 8.2%

Weak distractors 13.5%

Perception 19.6%

No image 14.6%

No image 12.4%

Overgeneralization 44.9%

Good question 4.3%

Visual giveaway 1.0%

Language bias 0.7%

Language bias Other correct 0.9%

0.2%

Overgeneralization 27.3%

Visual giveaway 0.3%

Other correct 0.3%

Figure 24. Reasoning trace categorization with an LLM (GPT-4o): Correct (blue) vs. incorrect (red) answers. Comparison of ClaudeSonnet’s responses without (left) and with (right) image access.

We apply this system to examine the impact of providing or withholding the image for each question (Fig. 24). While overall accuracy remains comparable across both conditions, the distribution of errors and successful reasoning patterns differs meaningfully. As expected, hallucination errors decrease when the image is available, perception errors are rare when the model lacks visual input, and the proportion of questions that do not require an image remains stable.

Most errors arise from questions requiring visual and contextual understanding, highlighting the importance of multimodal reasoning. Overgeneralization errors decrease when the image is provided, as the model is better grounded in the

question context. However, many questions remain challenging even with visual input, as perception errors increase. This suggests that effectively solving our dataset requires models to both maintain the specifics of the question’s context and accurately interpret visual information.

Additionally, note that the proportion of questions answered correctly through language shortcuts (e.g., language bias and visual giveaways) remains low in both settings, suggesting the effectiveness of the RefineBot strategy. However, addressing weak distractors remains a persistent challenge for specialized datasets.

[Figure 58]

35

Figure 25. Collage of images from MicroVQA.

