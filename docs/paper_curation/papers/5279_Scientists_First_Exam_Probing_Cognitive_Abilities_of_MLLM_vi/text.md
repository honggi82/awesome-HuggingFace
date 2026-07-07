# arXiv:2506.10521v6[cs.AI]14Nov2025

## Scientists’ First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning

#### PrismaX Team∗

Shanghai Artificial Intelligence Laboratory Dataset: https://huggingface.co/datasets/PrismaX/SFE Website: https://prismax.opencompass.org.cn/ Code: https://github.com/PrismaX-Team/sfe

### Abstract

Scientific discoveries increasingly rely on complex multimodal reasoning that integrates information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists’ First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three cognitive levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries.

### 1 Introduction

Scientific discoveries rely on investing significant time in analyzing large-scale, complex, and diverse data. Researchers require domain-specific knowledge to interpret scientific data across various modalities, and apply problem-solving skills to address specific scientific challenges [64]. Recent advances in multimodal large language models (MLLMs) have achieved remarkable performance on a wide range of benchmarks, comparable to or even surpassing human-level understanding in both general-level (e.g., MMLU [28], SuperGLUE [67], TriviaQA [31]) and graduate-level (e.g., GPQA [60], HumanEval [12], GSM8K [15]) knowledge domains. As MLLMs continue to progress from general-purpose understanding to domain-specific knowledge, scientific discovery has emerged as a critical frontier for evaluating and extending their abilities [5, 10, 11, 35, 42, 44, 84].

The process of scientific discovery often involves specialized scientific analysis of data modalities (e.g., molecular structures, spectra, protein sequences) from various scientific fields. SuperGPQA [17] extends conventional domains by incorporating long-tail disciplines, ensuring accessibility to realworld professional expertise. CURIE [16] establishes a ten-task benchmark for evaluating scientific reasoning in long-context scenarios. HLE [59] is introduced to evaluate model capabilities through challenging and expert-authored questions. However, despite the growing interest in the scientific domain, most existing scientific benchmarks extract tasks from secondary sources such as academic materials [16, 27, 78] and textbooks [70, 78]. As a result, they inadequately probe the cognitive abilities (e.g., perception, understanding, and reasoning) required for analyzing scientific data

∗Please refer to Appendix A for all team members

39th Conference on Neural Information Processing Systems (NeurIPS 2025) Track on Datasets and Benchmarks.

[Figure 1]

[Figure 2]

[Figure 3]

###### Scientific Raw Data

###### ... Elaborate Cog-Level Design

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Signal Perception Attribute Understanding

Comparative Reasoning

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

###### ...

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Discipline: Chemistry

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Discipline: Life Format: MCQ Taskid: L007

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Format: Open Ended

Long Visual Contexting

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Taskid: C004

Discipline: Earth Format: Open Ended Question

Taskid: E005

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Question Based on given mass spectrum , select the molecular structure that best matches it.

[Figure 40]

Question Generate a natural language description for the given molecule.

cientists’ irst xam

[Figure 41]

[Figure 42]

Question The first image ..., the second image ..., the third image .... Please analyze the rainfall formation process in eastern China.

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### Answer Choices

[Figure 47]

[Figure 48]

Material Life Chemistry

Correct Answer The molecule is the hydrochloride salt of dorzolamide. It is used in ophthalmic solutions to lower increased intraocular pressure in the treatment of open-angle glaucoma and ocular hypertension. It has a role as an EC 4.2.1.1 (carbonic anhydrase) inhibitor, an antihypertensive agent and an antiglaucoma drug. It is a hydrochloride and an organoammonium salt. It contains a dorzolamide.

A. B. C. D. E. F.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Correct Answer The precipitation area is located ahead of a westerly trough, which extends southward from Gansu to Yunnan. The region is influenced by warm advection ahead of the trough and receives moisture transported from the South China Sea. The subtropical high over the northwestern Pacific ...

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Correct Answer [“B”]

Bilingual Expert-based

Astronomy Earth

- (a) Examples from our proposed SFE. SFE is designed to comprehensively evaluate the scientific capabilities of MLLMs in depth and breadth.

[Figure 63]

Chemistry

Astronomy

Earth

Life

Material

Protein Structure Analysis (4)

RNA Structure Analysis (5)

Proteomic Sequencing

(2)

Circulation Understanding

Multivariate (5) Understanding

(3)

Ground Data Comparison Reasoning

(4)

Remote Sensing Perception (2)

Structure Analysis

(3)

Property Prediction

(9)

Drug Screening

(1)

Molecular Design and Optimization (2)

Reaction Prediction

(4)

Crystal Structure Analysis

(5)

Electronic Property Analysis

(3)

XRD Pattern Inference

(3)

Spectral Analysis

(5)

Light Curve Analysis

(1)

Star Perception and Comparison

(2)

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Metabolomic Sequencing

(3)

- (b) The structure of SFE includes 5 disciplines, 19 scientific directions, and 66 tasks.

100

90 93

91

claude-3-sonnet 88 90

90

gpt-4.1

87

86

82

gpt-4o

grok-2

76

80

68

score(%)

66

60

56

60

50

46

40

32

31

27

25

20

SFE GPQA MATH MMLU ScienceQA

(c) Performance of mainstream MLLM on various benchmarks.

17

18.0

SFE HLE SciBench

#ofdataformat(%)

SFE HLE SciBench

15

15

#ofimages

CURIE

CURIE

10

10

ScienceQA

ScienceQA

7.0

MMMU

MMMU

5

5

2.3

0.51.1 1.01.0

1.0 0.2

1.0 0.3

1 1 1 1 1

0.1

0

0

mean max

# of data formats

(d) Number of images per QA on various benchmarks.

(e) Number of released data formats.

Figure 1: Overview of the Scientists’ First Exam (SFE) benchmark.

encountered in real-world research. Moreover, these benchmarks tend to focus exclusively on a single ability to interpret domain knowledge from the data, while neglecting the full spectrum from perception to reasoning. This capability gap exposes a fundamental challenge: How to granularly measure MLLMs’ scientific cognitive capabilities across multiple disciplines for scientific discovery?

To bridge this gap, we introduce the Scientists’ First Exam (SFE) benchmark, designed to comprehensively evaluate the scientific cognitive capabilities of MLLMs through three cognitive levels (cog-levels): Scientific Signal Perception (L1) characterizes the capacity to discern critical components within visualizations of scientific raw data; Scientific Attribute Understanding (L2) demonstrates the ability to interpret domain-expert knowledge; Scientific Comparative Reasoning (L3) manifests the ability to derive phenomenological insights through structured comparison of multiple scientific visual sources. SFE encompasses 66 expert-curated, high-value multimodal tasks across five disciplines: Astronomy, Chemistry, Earth, Life, and Materials Sciences (Fig. 1b). Each task is constructed from native scientific raw data formats (Fig. 1a) and formulated as visual question answering (VQA) pairs (Fig. 1e), designed to probe specific levels of scientific cognition. All tasks are bilingual (English & Chinese) to support broad accessibility. These tasks are designed not only to require a deep understanding of domain-specific knowledge and data analysis skills but also to significantly enhance research efficiency and facilitate advancements that benefit society.

We benchmark 16 state-of-the-art open and closed weight MLLMs using SFE, as illustrated in Fig. 1c. As observed, while these MLLMs perform well on benchmarks such as MMLU [78] and ScienceQA [48], they all exhibit suboptimal results on the SFE benchmark. This indicates that SFE serves as a challenging frontier for scientific MLLM development. Our contributions:

- 1. We propose the first benchmark to categorize scientific tasks by cognitive capacity, introducing a three-level taxonomy: Scientific Signal Perception (L1), Scientific Attribute Understanding (L2), and Scientific Comparative Reasoning (L3). This formulation enables fine-grained evaluation of how MLLMs engage with different layers of scientific research.

Table 1: Comparison of large language model (LLM) benchmarks related to science. Astro, Chem, Phy, Bio, CS, QC, Geo, and Mat are abbreviations for Astronomy, Chemistry, Physics, Biology, Computer Science, Quantum Computing, Geospatial Analysis, and Material, respectively. AD, Bus., Sci., Med., HSS, and TE refer to Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, respectively. The Question Types include MCQ (Multiple-Choice Questions), EM (Exact Match), and OQ (Open Questions). EN/ZH denotes the language of each benchmark (English/Chinese).

MultiModal

Question Type

Task #Count

Raw Data

Question Source Language

Benchmark Discipline

MCQ OQ .png ✗

Textbooks e-Resources

AD, Bus., Sci. Med., HSS, TE

MMMU[78]

EN

✓

Natural, social and language science

SCIENCEQA[48]

✓ MCQ .png ✗ Curricula EN

Math, Chem Physics

SCIBENCH[70]

✓ OQ .png ✗ Textbooks EN

7 School subjects

CMMU[27]

✓ OQ .png ✗ Exams ZH

Public datasets

EN SuperGPQA[17] 13 Disciplines ✗ MCQ - ✗ Experts EN

ChemBench[26] Chem ✗ EM - 8

MCQ, EM judgment

Knowledge base

SciEval[63] Chem, Phy, Bio ✗

- ✗

EN

Math, Phy, Bio HSS, CS, Chem,TE

✓ MCQ, EM .png ✗ Experts EN

HLE[59]

Mat, Phy, QC, Geo Bio, Proteins

CURIE[16]

✓ OQ .png 10 Experts EN

Astro,Chem, Earth Life,Mat

MCQ EM, OQ

.png,.mgf .sto,.txt

SFE

, ... 66 Experts EN, ZH

✓

- 2. We release the bilingual SFE benchmark, encompassing 66 expert-curated multimodal tasks across five scientific disciplines and covering three question types. All tasks are constructed from native scientific data formats and aligned with three cognitive capacity levels.
- 3. We comprehensively evaluate 16 state-of-the-art MLLMs, revealing that GPT-o3 achieves the best overall performance, and newer model versions show clear improvements in L3 tasks.

### 2 Related Works

Science LLMs / MLLMs. Recent advancements in domain-specific large language models (LLMs) have significantly impacted various scientific fields. In the biomedical domain, LLMs have been employed for tasks such as clinical documentation, information retrieval, and hypothesis generation [35, 36, 49–51, 56]. In chemistry, recent studies have focused on tasks such as drug property prediction, molecular discovery, chemical reaction extraction, and protein structure understanding [11, 39, 40, 43–45, 47, 52, 81, 87]. Notably, researchers have developed LLMs trained on scientific corpora to support molecular and protein discovery [5, 14, 20, 54, 66, 71]. For instance, iupacGPT [14] uses IUPAC nomenclature to effectively capture relationships among atoms and chemical groups. Similarly, Progen [54] and ProtGPT2 [20] are trained for protein sequence generation and understanding. Geosciences have benefited from LLMs through their applications in ocean science, extreme weather, and remote sensing [8, 38, 41, 53, 73, 77, 83, 84]. Models like EarthGPT [83] and CLLMate [38] further integrate multimodal knowledge to support scientific question answering.

Science Benchmarks. With the development of large language models (LLMs), recent efforts in LLM benchmarking have increasingly focused on evaluating scientific reasoning capabilities across diverse domains and modalities. Early benchmarks [46] such as ScienceQA [48] and CMMU [27], primarily focus on science level below high school. Several college-level benchmarks [17, 70, 72, 78, 79] have emerged to support deeper scientific understanding. As state-of-the-art multimodal LLMs rapidly improve, many of them now achieve strong performance on these benchmarks. This has motivated the creation of more expert-level datasets that feature high-difficulty questions such as HLE [59] and CURIE [16]. However, although some benchmarks offer broad domain coverage, they often lack clearly defined tasks, making it difficult to assess specific model limitations. To address this,

###### Structure Construction

###### Task Construction

###### Benchmark Construction

###### Task Define Rule

Scientific Frontier

[Figure 85]

Visualization

[Figure 86]

[Figure 87]

[Figure 88]

Dicipline Question Type

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Pymatviz Colormap

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Structure Overview

[Figure 105]

3D Render Rdkit

Raw Data

Astronomy Chemistry

Star Preception and Comparison Light Curve Analysis Spectral Analysis

Molucular Design and Optimization

Answer Annotation

Structure Analysis Property Prediction Reaction Preicion Drug Screening

[Figure 106]

###### Material

###### Task Formulation Review

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Cystal Structure Analysis XRD Pattern Inference Electronic Property Analysis

[Figure 112]

[Figure 113]

Align Propose Check Accept

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

###### Earth

[Figure 122]

[Figure 123]

[Figure 124]

###### Life

[Figure 125]

Circulation Understanding Multivariate Understanding Remote Sensing Perception Ground Data Comparison Reasoning

[Figure 126]

Protein Structure Analysis Metabolomic Sequencing Proteomic Sequencing

Translate Format

[Figure 127]

RNA Structure Analysis

[Figure 128]

###### Raw Data Collection

###### Expertised Supporting

[Figure 129]

###### Annotation Review

- ■ Phase Diagram
- ■ X-Ray Diffraction
- ■ RNA Structure
- ■ Proteomic Sequencing
- ■ Molecular Structure
- ■ SMILES
- ■ Vertical Integrate Liquid
- ■ Geopotential Height
- ■ Light Curve
- ■Spectrum Raw Dataset Data Format

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

What scientific tasks or questions are most valuable in your research area?

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Atmospheric circularion understanding is fundamental. It regulates the large-scale distribution of heat and moisture, influences the development of weather systems

[Figure 157]

Format Checking

Cross Validating

- Figure 2: Data collection framework of SFE. First, we define 18 scientific directions based on the scientific frontier and domain experts. Building upon this structure, we invite experts to propose tasks and contribute raw data based on three cognitive levels. Finally, we employ visualization techniques and further engage experts to annotate the resulting benchmark.

works like ChemLLMBench [26] and CURIE [16] have introduced smaller-scale but task-oriented benchmarks that enable more targeted evaluation of scientific capabilities. Nevertheless, nearly all of these benchmarks remain monolingual, limiting their applicability for comprehensively evaluating LLMs in global deployment contexts.

### 3 SFE Dataset and Tasks

#### Table 2: Statistics of SFE.

The SFE benchmark consists of 830 multimodal VQA pairs spanning 66 real-world scientific tasks across five disciplines: Astronomy, Chemistry, Earth, Life, and Materials Science (Fig. 1b). Each task is constructed from native disciplinary data formats such as molecular structures, spectra, and radar charts, and is carefully annotated by domain experts. On average, each question contains 2.3 scientific images (ranging from 1 to 18), and the dataset supports bilingual prompts and answers in both English and Chinese. In total, SFE spans 17 distinct scientific data formats and is motivated by scientific frontiers such as drug discovery, celestial body radial velocity estimation and peptide sequence inference, etc. Detailed dataset statistics are presented in Table 2, and a complete task list is provided in Appendix Table 6.

###### Statistic Number

# of VQA 830 # of Scientific Tasks 66

# of scientific signal perception VQA 202 # of scientific attribute understanding VQA 503 # of scientific comparative reasoning VQA 125

# of MCQ 284 # of Exact Match 420 # of Open Question 126

Average Question Tokens Length 88 (en) / 86 (zh) Average Answer Tokens Length 100 (en) / 106 (zh) Average # of Images 2.3 (1-18) Released # Scientific Data Format 17

#### 3.1 Data Collection

The construction of SFE is the result of extensive collaboration with domain experts across multiple scientific disciplines and research directions. As shown in Fig. 2, the overall data collection pipeline comprises three key stages: (1) Structure Construction, where we collaborate with experts to identify high-value challenges and define scientific directions; (2) Task Construction, where the scientific directions are instantiated into concrete tasks with specific question types and cognitive levels through expert-driven design and review; (3) Benchmark Construction, where scientific raw data are rendered, visualized, and used to construct expert-authored VQA with high-quality samples.

Structure Construction. The SFE is initiated by identifying foundational scientific directions across five core disciplines: Astronomy, Chemistry, Earth, Life, and Materials Science. These fields are selected for their broad scientific relevance and the need for specialized knowledge. Based on the experts’ consultations, we define a structured set of scientific directions, such as circulation

understanding in Earth and reaction prediction in chemistry. These directions serve as the foundation for downstream task formulation and reflect current research frontiers.

Task Construction. Building upon the established scientific directions, domain experts collaboratively define concrete tasks. We first formalize the task definition rules by mapping each scientific direction to an appropriate cognitive level and question type as shown in Fig. 2. This taxonomy ensures that each task is aligned with a distinct level of cognitive capacity and supports evaluation diversity. For each discipline, we collaborated with 2-5 experts to define benchmark tasks that align with high-priority issues in real-world scientific research. Each expert-designed task is expected to satisfy the following core criteria: (1) It reflects a meaningful problem that requires substantial domain knowledge and is commonly encountered in real-world research. (2) It must be solvable through expert-level reasoning that necessarily integrates multimodal scientific inputs such as structured visualizations and symbolic representations. (3) It aligns with one of the proposed cognitive levels (L1-L3). To support each task, experts also provide instructions for collecting raw data that would serve as visual inputs for the VQA pairs. These data sources span a diverse range of formats, including public datasets (e.g., ERA5 [29]), existing benchmarks (e.g., MoleculeNet [75]), and domain-specific databases (e.g., RCSB PDB [7], PDBbind [69], and PubChem [33]). For each task, domain experts identify suitable data entries that are typically used to analyze the corresponding scientific problems, such as RNA sequence files (.stockholm) and protein structure files (.pdb).

Benchmark Construction. In the final stage, each task is instantiated into a set of VQA pairs. To ensure accessibility and visual coherence, all data formats are rendered into image form. During the answer annotation phase, experts construct each VQA pair by composing: (1) rendered multimodal input, (2) a scientifically meaningful question, and (3) an answer grounded in expert reasoning. Then, all VQA pairs are translated into both Chinese and English to provide multilingual support and formatted in JSON for standardization. Finally, a two-stage validation process is applied for quality control. Each VQA pair is first cross-reviewed by domain experts for scientific correctness, clarity, and alignment with the intended reasoning level. This is followed by rule-based validation for format checking. Only VQA pairs that pass both stages are included in the final benchmark.

#### 3.2 Tasks

Astronomy. In Astronomy, analyzing diverse modalities such as spectra is essential for a wide range of scientific tasks, including property estimation and event detection. Therefore, we design 8 tasks to assess the cognitive abilities of MLLMs from 3 cognitive levels. L1 tasks include galaxy morphology classification, where MLLMs are required to differentiate perceptual features of galaxies based on the provided image. Second, L2 tasks include surface temperature estimation, gravitational constant estimation, light curve classification, metallicity estimation, etc. For example, in the metallicity estimation task, MLLMs need to infer the total metal abundance of a target celestial object based on its observed spectrum. Third, L3 tasks target transient detection. Given the pre-transient and post-transient images, along with difference images, MLLMs are expected to determine whether a transient has occurred during the process. Refer to the Appendix for more details.

Chemistry. We formulate 19 tasks in the Chemistry domain, spanning three cognitive levels to evaluate MLLMs’ understanding of molecular structures, properties, and interactions. L1 tasks include elemental composition recognition, molecular description generation, etc. In the molecular description generation task, MLLMs are required to produce descriptions of specific molecules, highlighting key attributes such as types of chemical bonds and the number of carbon atoms. L2 tasks involve Lipinski drug-likeness estimation, absorption property prediction, distribution property prediction, etc. For example, in the absorption property prediction task, MLLMs need to estimate properties of molecules, such as the plasma protein binding rate. L3 tasks include virtual screening, protein-ligand binding affinity prediction, etc. In the virtual screening task, MLLMs are expected to identify all molecules capable of binding to a given structure. Refer to the Appendix for more details.

Earth. To systematically evaluate the performance of MLLMs in the Earth science domain, we construct 14 tasks across three cognitive levels based on diverse weather variables and data sources. First, L1 tasks include thermocline depth recognition, perception of extreme precipitation distribution, SAR image grounding, etc. For example, in the perception of extreme precipitation distribution task, MLLMs are required to identify multiple locations of extreme precipitation. Second, L2 tasks include moisture source understanding, precipitation event analysis, convective weather types identification, etc. For example, the precipitation event analysis requires models to analyze the formation process

of precipitation by using information from geopotential height, moisture flux, vertical velocity, etc. Third, L3 tasks consist of differential prediction comparison, temperature sequence comparison, etc. In the temperature sequence comparison task, MLLMs need to compare annual temperature series from two different time periods and describe differences in statistical characteristics. Refer to the Appendix for more details.

Life. In the life science domain, we construct 14 tasks across three cognitive levels to comprehensively evaluate MLLMs, focusing on modalities such as RNA structures and protein sequences. L1 tasks include fragment ion peaks count, protein chain count, small molecule count, etc. For example, in the fragment ion peaks count task, MLLMs are asked to determine the number of specific ions present in a given MS/MS spectrum. L2 tasks consist of molecular composition inference, specified protein detection, etc. For instance, in the molecular composition inference task, MLLMs need to infer the elemental composition of the compound based on the provided spectrum. L3 tasks include spectrum matching, where MLLMs are required to identify the molecular structure that best corresponds to a given mass spectrum. Refer to the Appendix for more details.

Materials. To evaluate MLLMs’ performance in the realm of materials, we formulate 11 tasks spanning three cognitive levels. L1 tasks include atomic composition description, crystal group identification, crystal formula determination, etc. For example, in the atomic composition description task, models are required to select relevant properties of a target lattice from multiple descriptions. Second, L2 tasks cover band gap classification, stability estimation, energy band and DOS interpretation. For example, in the energy band and DOS interpretation task, MLLMs are required to infer whether a given material belongs to a metal or a semiconductor and estimate its band gap. Third, L3 tasks involve complicated comparative reasoning, such as phase identification. Specifically, given the XRD pattern of a composite material and candidate substances, MLLMs are required to identify three materials that form the composite material. Refer to the Appendix for more details.

### 4 Experiments and Evaluations

General Settings. We conduct a comprehensive evaluation of the state-of-the-art MLLMs on SFE. For models with open weights, we assess InternVL2.5-78B [13], InternVL3-78B [86], Qwen2.5VL-72B [6], LLaMaVision-90B [22], and LLaVa-Onevision-72B [37]. For models with closed weights, we evaluate GPT-4o-2024-11-20 [57], GPT-4.1-2025-04-14 [57], GPT-o1-2024-12-17 [57], GPT-o3 [57], Claude 3 Opus [4], Claude 3.7 Sonnet [4], Gemini-2.0-Flash [65], Gemini-2.5Flash [65], Gemini-2.5-Pro [65], Grok-2-Vision-12-12 [76] and Doubao-1.5-Vision-Pro [24]. When benchmarking, we configure all MLLMs’ temperatures to 0 for reduced randomness and employ a standard zero-shot prompt template across all tasks. Specifically, the template begins with a description of the task assigned to the model, followed by the inclusion of question texts with interleaved images. Additionally, we fix the maximal number of generated tokens to 1024, ensuring fairness and cost-effectiveness in our evaluations.

Metrics. We present the BERTScore [82] and the LLM-as-a-Judge score [23] for all tasks, except for the remote sensing perception task in Earth science, where we report the execution success rate and the Intersection over Union (IoU). For BERTScore, we use the F1 score. For LLM-as-a-Judge score, we employ GPT-4o-2024-11-20 as the judge, allowing us to semantically verify the correctness of answers against model predictions. Conversely, the execution success rate is utilized to assess whether MLLMs can accurately follow prompts to produce bounding boxes in the desired format. Finally, the IoU metric evaluates the precision of these bounding boxes compared to ground truths. Without further clarification, each experiment is conducted once to obtain the final results.

#### 4.1 Main Results

We evaluate the performance of 16 MLLMs across five scientific disciplines and two languages (English and Chinese) on SFE using the LLM-as-a-Judge score for evaluation metric. Our results in Table 3 demonstrate that SFE is capable of revealing fine-grained differences in model capabilities, visual grounding quality, and multilingual reasoning robustness. Below, we detail key observations.

- Observation 1. SFE is capable of distinguishing model capability across both proprietary and open-source MLLMs. Among all evaluated models, GPT-o3 achieves the best overall performance, with an average score of 34.08% in English and 31.60% in Chinese. It is consistently strong across all

- Table 3: Experimental results of all models on different disciplines using different languages. The LLM-as-a-Judge score is used as the evaluation metric. ‘Average’ represents the mean score.

Model

Astronomy Chemistry Earth Life Material Average en zh en zh en zh en zh en zh en zh

Closed Weight MLLMs

Grok-2-Vision-12-12 19.37 18.23 21.29 20.20 33.58 36.23 25.10 25.95 35.57 36.89 24.97 25.10 GPT-4o-2024-11-20 20.38 18.04 21.60 19.25 32.65 29.93 30.39 27.58 49.67 48.36 27.15 24.72 GPT-4.1-2025-04-14 24.18 25.50 24.01 22.11 40.40 44.43 34.90 33.66 47.70 48.85 30.88 31.05 GPT-o1-2024-12-17 22.03 21.84 27.41 27.21 38.61 36.56 33.99 31.31 61.15 61.64 32.19 31.24

GPT-o3 24.24 23.80 28.91 27.89 43.05 36.29 33.59 31.57 63.44 58.20 34.08 31.60 Gemini-2.0-Flash 16.14 12.78 27.82 24.69 34.24 32.91 32.48 27.32 52.79 50.49 29.49 26.33 Gemini-2.5-Flash 24.30 24.11 23.67 23.47 31.99 30.53 25.03 25.10 56.39 55.90 28.03 27.63 Gemini-2.5-Pro 5.13 6.08 2.07 2.28 2.52 3.84 19.73 22.35 28.69 27.70 8.04 8.96 Claude-3-Opus 14.68 16.08 19.63 17.45 36.62 32.12 22.55 23.66 36.72 32.13 23.64 22.15

Claude-3.7-Sonnet 25.89 22.34 27.79 25.14 38.21 37.09 31.24 29.48 49.51 46.72 31.62 29.23 Doubao-1.5-vision-pro 28.35 23.99 25.00 24.46 26.16 25.83 30.07 29.35 51.48 46.39 28.79 27.17

Open Weight MLLMs

Qwen2.5-VL-72b 26.46 20.89 18.33 15.27 25.03 24.57 25.29 22.75 41.47 42.46 24.17 21.51 InternVL-2.5-78B 15.76 17.09 18.54 15.58 34.77 37.22 27.06 25.49 43.11 39.84 24.43 23.54

InternVL-3-78B 27.09 24.62 19.80 16.29 28.81 28.94 30.85 27.58 40.98 42.30 26.52 24.30 Llama-3.2-Vision-90B 20.63 18.04 16.94 15.14 27.22 30.53 25.29 21.37 32.30 31.80 22.26 20.95 Llava-OneVision-72B 25.19 23.23 15.34 15.17 23.31 27.81 24.58 24.58 37.54 42.62 22.10 23.39

Average 21.23 19.79 21.14 19.51 31.07 30.93 28.27 26.82 45.54 44.52 26.15 24.93

- Table 4: Experimental results of all models on different cognitive levels and different question types in both Chinese and English.

Exact Match Open Question MCQ IoU LLM score Bertscore LLM score LLM score en zh en zh en zh en zh en zh en zh en zh en zh

L1 L2 L3

Model

###### Closed Weight MLLMs

Grok-2-Vision-12-12 28.31 25.66 23.10 25.81 27.44 21.44 6.11 8.30 17.32 17.39 0.707 0.718 35.48 35.0 32.68 33.13 GPT-4o-2024-11-20 35.50 32.70 26.46 23.32 17.28 18.32 2.66 2.13 20.42 17.58 0.698 0.710 36.74 39.23 33.87 30.21 GPT-4.1-2025-04-14 35.98 36.61 30.02 29.64 26.64 28.32 5.19 3.47 22.10 20.14 0.714 0.723 46.35 44.13 38.49 42.74 GPT-o1-2024-12-17 43.65 42.80 29.09 26.60 27.36 32.40 0.77 1.83 22.61 21.82 0.70 0.71 44.81 44.62 42.04 40.56

GPT-o3 42.54 41.27 30.30 28.27 36.48 31.60 5.17 5.94 23.75 22.12 0.694 0.693 48.85 44.33 44.26 41.27 Gemini-2.0-Flash 41.43 36.46 26.86 23.70 22.00 21.60 0.91 0.44 21.38 19.28 0.70 0.65 35.67 25.19 39.47 37.39 Gemini-2.5-Flash 37.04 37.57 27.89 26.98 14.96 15.20 0.99 0.96 24.45 24.45 0.688 0.688 30.00 30.10 32.71 31.51 Gemini-2.5-Pro 17.46 18.78 6.16 6.94 1.36 2.24 - - 8.11 8.60 0.548 0.558 12.50 15.96 6.30 6.94 Claude-3-Opus 25.71 23.54 23.80 22.09 19.84 20.32 0.87 0.08 15.99 14.71 0.700 0.708 30.58 30.87 32.64 30.21

Claude-3.7-Sonnet 37.46 33.39 29.80 28.15 30.08 27.28 2.27 1.38 21.40 19.09 0.710 0.671 46.06 46.73 41.76 38.13 Doubao-1.5-vision-pro 32.70 32.43 26.48 24.31 32.16 30.72 2.19 11.84 23.50 21.00 0.677 0.691 34.62 33.85 34.65 34.05

###### Open Weight MLLMs

Qwen2.5-VL-72b 31.11 26.08 23.36 21.91 16.96 12.96 24.49 15.37 20.49 17.55 0.645 0.667 20.58 22.12 31.06 27.25 InternVL-2.5-78B 25.08 22.91 25.33 24.35 19.84 21.20 3.79 3.65 16.53 14.92 0.690 0.692 36.06 36.06 32.11 31.97

InternVL-3-78B 26.67 27.30 26.10 22.82 28.00 25.68 5.25 7.16 19.39 15.99 0.69 0.70 37.5 38.85 33.27 31.51 Llama-3.2-Vision-90B 26.14 24.23 22.33 20.34 16.16 18.48 0.27 0.25 15.90 14.17 0.70 0.68 29.23 29.71 29.33 27.99 Llava-OneVision-72B 25.13 26.40 19.14 20.68 29.52 29.76 2.84 1.42 14.99 15.87 0.700 0.665 32.88 31.54 28.91 31.76

Average 32.00 30.49 24.76 23.49 22.89 22.34 4.26 4.25 19.27 17.79 0.689 0.683 34.87 34.27 33.35 32.29

disciplines, particularly in Earth and Materials, where scientific visual interpretation and structured multimodal reasoning are often required. Notably, Gemini-2.5-Pro registers the lowest average score among all evaluated models, with only 8.04% in English and 8.96% in Chinese due to excessive thinking, causing exhausted token budgets and clipped answers2. On the other hand, while GPT-o3 reasons as well, it balances reasoning and token usage effectively. The performance gap between GPT-o3 and Gemini-2.5-Pro exceeds 26% on average, showing SFE is comprehensive enough to differentiate the ability across the full range of models. The results also present a systematic divide between closed-weight proprietary models and open-weight models. On average, the best proprietary models (GPT-o3, GPT-o1-2024-12-17, Claude-3.7-Sonnet) outperform the strongest open alternatives (InternVL-3-78B) by 6-8%. Additionally, within SFE, the model series shows clear internal progress. For example, Claude-3.7-Sonnet outperforms Claude-3-Opus by over 7% in both English and Chinese, reflecting measurable architectural or training gains. A similar pattern is also observed within the InternVL series.

2Experiments with more generated tokens are conducted in Table 17 and Table 18 in the appendix.

- 24

- 25

37.5

| | | | | |27|.2| |
|---|---|---|---|---|---|---|---|
| |2|6.09<br><br>26|.57<br><br>26|.9<br><br>27.|06|27.3|3|
| | | | | |SFE|-en| |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

score(%)

score(%)

score(%)

37.25 37.75

- 26
- 27

score(%)

30.95 31.26

36.59

35.0

30.0

24.74

30.57

35.58

30.26

24.62

24.88 SFE-en

24.49

32.5

33.69

24.32

29.28

SFE-en

SFE-en

30.56

23.86

27.43

27.5

1 2 3 4 5 6 # of pass

1 2 3 4 5 6 # of pass

1 2 3 4 5 6 # of pass

1 2 3 4 5 6 # of pass

- 24

- 25

37.5

score(%)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |2|1.25<br><br>21.8|8<br><br>22.0|122.|04<br><br>22|.1 22.1|7|
| | | | | |SFE|-zh| |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

25.09 25.19

- 21
- 22

score(%)

score(%)

37.3 37.95

24.99

score(%)

31.02 31.2

36.64

35.0

30.0

24.91

30.79

35.33

30.23

24.53

32.5

SFE-zh

33.41

29.25

SFE-zh

23.93

SFE-zh

30.41

27.36

27.5

1 2 3 4 5 6 # of pass

1 2 3 4 5 6 # of pass

1 2 3 4 5 6 # of pass

1 2 3 4 5 6 # of pass

(a) GPT-4.1-2025-04-14

(b) Gemini-2.5-Flash

(c) QwenVL-2.5 -72B

(d) InternVL-3-78B

- Figure 3: Pass@k scores of four state-of-the-art MLLMs on SFE. Closed-weight MLLMs demonstrate superior initial performance and greater scalability than open weight MLLMs.

- Observation 2. SFE exhibits a clear performance gap of MLLMs between disciplines. Our findings reveal that across nearly all models, Material Science emerges as the most tractable domain. The top model, GPT-o3, reaches 63.44% in English and 58.20% in Chinese on it. And even open models like Qwen2.5-VL-72b and InternVL-3-78B achieve over 40% in this domain. This trend reflects the relatively structured visual inputs (e.g., phase diagram, X-Ray diffraction) which require the model to generate structured scientific outputs based on symbolic visual images that are aligned with current models’ comparatively strong skills. In contrast, Astronomy tasks present more substantial challenges. This domain involves spectral analysis tasks, where models estimate numerical astrophysical parameters (e.g., temperature and velocity) from raw or noisy spectral visualizations, which current MLLMs find challenging. This highlights SFE’s role in diagnosing which types of scientific reasoning MLLMs can currently handle and which are still out of reach.
- Observation 3. SFE reveals a potential shift in MLLM capabilities from knowledge understanding to high-order reasoning. Through SFE’s three-level cognitive framework, its results provide an analysis of how recent models’ cognitive improvements are distributed. Our findings (in Table 4) show that newer MLLMs exhibit notably higher performance on L3 tasks compared to earlier models, while their L2 performance remains largely similar. This trend aligns with the adoption of advanced reasoning techniques in state-of-the-art models. For example, GPT-o3 improves L3 performance from 26.64% (GPT-4.1-2025-04-14) to 36.48% without an obvious increase in L2 scores (from 30.30% to 30.02%). This pattern reflects OpenAI’s reports that emphasize scaling reinforcement learning for reasoning and learning tool-use strategies rather than knowledge expansion. Similarly, InternVL-3 outperforms InternVL-2.5 by 8% in L3 performance in English, despite only marginal gains in L2. This improvement can be attributed to InternVL-3’s architectural and training advances, particularly its native multimodal pretraining and Mixed Preference Optimization (MPO) to support Chain-of-Thought reasoning, which is a critical skill for success in L3 tasks such as comparative diagram analysis. The negligible gains in L2 tasks suggest that these improvements arise not from broader knowledge acquisition but from enhanced training innovations.
- 4.2 Analysis

Pass@k Analysis. The Pass@k metric [34] selects the highest quality answers from an MLLM as its final response to a question, indicating the model’s potential for improvement through post-training (e.g., RLHF [58], GRPO [25], etc.). In Fig. 3, we evaluate pass@k scores of GPT-4.1-2025-04-14, Gemini-2.5-Flash, Qwen2.5-VL-72b and InternVL-3-78B on SFE, with k ranging from 1 to 6. As shown, GPT-4.1-2025-04-14 and Gemini-2.5-Flash outperform state-of-the-art open-weight MLLMs by not only exhibiting superior initial performance (30.56% v.s. 26.09%) but also demonstrating strong scalability (30.56% → 37.75% v.s. 26.09 → 27.33%). This suggests that closed-weight MLLMs may leverage more diverse and expansive raw datasets during pre-training than open-weight MLLMs. Furthermore, their post-training phase may prioritize a balanced approach, emphasizing exploration alongside exploitation, rather than focusing exclusively on exploitation [80].

Scaling Law of Model Size. We benchmarked MLLMs of varying sizes to evaluate their impact on SFE, as depicted in Fig. 4. Although Llama-3.2-Vision and Llava-Onevision series improve with size, they underperform compared to the Qwen2.5-VL and InternVL series. The larger models, Qwen2.5-

SFE-en

25

score(%)

20

Qwen2.5-VL

Llama-3.2-Vision

15

InternVL-3

Llava-OV

0 20 40 60 80 model size (B)

25

SFE-zh

score(%)

20

Qwen2.5-VL

Llama-3.2-Vision

15

InternVL-3

Llava-OV

0 20 40 60 80 model size (B)

- Figure 4: The scaling law of model size on SFE. Clearly, the amount of scientific data has not been scaled proportionally with the increase in model size, resulting in minor overfit for QwenVL-2.5-72B.

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Question Generate a natural language description for the given molecule.

[Figure 163]

Ground Truth The molecule is a fatty acid ethyl ester of 3-hexenoic acid. It has a role as a metabolite. It derives from a 3-hexenoic acid.

[Figure 164]

GPT-o3 Ethyl 3-hexenoate - the ethyl ester of hex-3-enoic acid featuring a cixcarbon chain with a C3-C4 double bond.

[Figure 165]

GPT-4o The molecule shown is ethyl pent2-enoate, an ester with a double bond in the pentyl chain.

Question The first figure uses shading to represent precipitation. The second figure shows 500 hPa geopotential height with contour lines, 850 hPa wind field with vectors, and moisture flux with shading. What are the sources of moisture for precipitation in the East China region?

Question

Based on the XRD pattern of the composite material made from a mixture of three materialsand five XRD patterns of AlN , BaTe , Al₃Ir , AcOF , Ac₂CuIr (orderly), identify all three phases/materials present in the composite material.

[Figure 166]

Ground Truth This graph contains three distinct materials: Al₃Ir, AcOF, Ac₂CuIr.

GPT-o3 The XRD pattern confirms the presence of AlN (34°, 49°, 60°, 82°), BaTe (25–30°, 44°, 53°, 62°, 71°), and Al₃Ir (22–24°, 42°, 50°, 66–67°, 74–89°), with all major reflections accounted for by these three phases.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

GPT-4o AlN, BaTe, Al₃Ir

Signal Perception Attribute Understanding Comparative Reasoning

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

GPT-o1

- D and E. From the 850 hPa wind vectors and the shaded moisture flux fields (second panel), the low level flow feeding into eastern China is largely southsouthwesterly, tracing back to the South China Sea and Bay of Bengal.

[Figure 178]

Answer Choices A. Bohai Sea B. Yellow Sea C. East China Sea D. South China sea

- E. Bay of Bengal

Claude-3.7-Sonnet D. The moisture flux is particularly intense along the South China Sea-East China. The wind vectors show air masses moving from the South China Sea northward into the rainfall region. While the East China Sea is nearby, the primary moisture transport appears to be coming from the south.

Figure 5: Case studies on SFE across different cognitive levels.

VL-72B and InternVL-3-78B, do not significantly surpass their smaller counterparts, indicating a lack of proportional scientific data scaling during pre-training. Additionally, Qwen2.5-VL-72B’s performance is lower than Qwen2.5-VL-7B, suggesting potential overfitting. This emphasizes the need for balanced data scaling relative to model size in the scientific domain.

Table 5: Scores on different temperatures.

Model 0.0 0.2 0.4 0.6 0.8 1.0

GPT-4.1-2025-04-14 30.88 31.31 31.59 32.09 31.97 31.63 InternVL-3-78B 24.43 26.98 26.93 26.55 25.86 23.12

Impact on Temperatures. We analyze the effect of temperature settings on MLLMs’ performance, as shown in Table 5. The results indicate that both excessively high and low temperature values can lead to performance degradation for scientific discoveries. Empirically, maintaining the temperature within the range of 0.4 to 0.6 optimally balances the trade-off between exploration and exploitation, thereby enhancing MLLMs’ efficacy.

Case Study. Fig. 5 presents case studies evaluating the performance of MLLMs on SFE across different cognitive levels. In the signal perception task, both GPT-4o-2024-11-20 and GPT-o3 effectively recognize functional groups within molecular images. However, GPT-o3, equipped with reasoning capabilities, offers more precise results for fine-grained tasks such as counting carbon atoms. For attribute understanding, Claude-3.7-Sonnet and GPT-o1-2024-12-17 accurately identify colormaps and wind vectors in geographical images. Despite achieving approximately one-third accuracy in their responses, GPT-o1-2024-12-17 falsely identifies the Bay of Bengal with its wind strength misinterpreted, suggesting an incomplete geographical understanding. In comparative reasoning, GPT-o3 integrates information from multiple images but struggles with comparative selection, revealing an incomplete reasoning chain. Conversely, across similar tasks, MLLMs like GPT-4o-2024-11-20, Grok-2-Vision-12-12, and others primarily focus on the initial four images, indicating limitations in processing long visual contexts.

- 5 Conclusion

In this paper, we propose the Scientists’ First Example (SFE) benchmark, aiming to provide a granular assessment of MLLMs’ scientific cognitive capabilities from perception to reasoning. SFE includes 830 expert-verified VQA pairs across 66 tasks in five high-value disciplines, addressing critical needs

for more rigorous and diverse evaluation tools for scientific MLLMs. Extensive experiments reveal insights that could contribute to advancements in AI-driven scientific discoveries.

However, the benchmark’s scope and depth could be further enhanced, which we intend to improve in future work. Furthermore, while SFE has the potential to significantly advance these discoveries by offering a robust evaluation framework, it also raises concerns about increasing reliance on AI in scientific research. This might inadvertently undermine the value of human intuition and creativity.

### Acknowledgments and Disclosure of Funding

This work is supported by Intern Discovery.

### References

- [1] A. Aghamousa, J. Aguilar, S. Ahlen, S. Alam, L. E. Allen, C. A. Prieto, J. Annis, S. Bailey, C. Balland, O. Ballester, et al. The desi experiment part i: science, targeting, and survey design. arXiv preprint arXiv:1611.00036, 2016.
- [2] H. Aihara, N. Arimoto, R. Armstrong, S. Arnouts, N. A. Bahcall, S. Bickerton, J. Bosch, K. Bundy, P. L. Capak, J. H. Chan, et al. The hyper suprime-cam ssp survey: overview and survey design. Publications of the Astronomical Society of Japan, 2018.
- [3] E. Angeloudi, J. Audenaert, M. Bowles, B. M. Boyd, D. Chemaly, B. Cherinka, I. Ciuc˘a, M. Cranmer, A. Do, M. Grayling, et al. The multimodal universe: enabling large-scale machine learning with 100 tb of astronomical scientific data. In NeurIPS, 2024.
- [4] Anthropic. Anthropic ai, 2023. URL https://www.anthropic.com.
- [5] V. Bagal, R. Aggarwal, P. Vinod, and U. D. Priyakumar. Molgpt: molecular generation using a transformerdecoder model, 2021.
- [6] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [7] H. Berman, K. Henrick, and H. Nakamura. Announcing the worldwide protein data bank. Nature structural & molecular biology, 10(12):980–980, 2003.
- [8] Z. Bi, N. Zhang, Y. Xue, Y. Ou, D. Ji, G. Zheng, and H. Chen. Oceangpt: A large language model for ocean science tasks. arXiv preprint arXiv:2310.02031, 2023.
- [9] M. S. Binetti, C. Campanale, C. Massarelli, and V. F. Uricchio. The use of weather radar data: Possibilities, challenges and advanced applications. Earth, 2022.
- [10] H. Cao, Z. Liu, X. Lu, Y. Yao, and Y. Li. Instructmol: Multi-modal integration for building a versatile and reliable molecular assistant in drug discovery. arXiv preprint arXiv:2311.16208, 2023.
- [11] L. Chen, W. Wang, Z. Bai, P. Xu, Y. Fang, J. Fang, W. Wu, L. Zhou, R. Zhang, Y. Xia, et al. PharmaGPT: Domain-specific large language models for bio-pharmaceutical and chemistry. arXiv preprint arXiv:2406.18045, 2024.
- [12] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [13] Z. Chen, W. Wang, Y. Cao, Y. Liu, Z. Gao, E. Cui, J. Zhu, S. Ye, H. Tian, Z. Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [14] K.-H. Cho, K. T. No, et al. iupacgpt: Iupac-based large-scale molecular pre-trained model for property prediction and molecule generation. 2023.
- [15] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [16] H. Cui, Z. Shamsi, G. Cheon, X. Ma, S. Li, M. Tikhanovskaya, P. C. Norgaard, N. Mudur, M. B. Plomecka, P. Raccuglia, et al. CURIE: Evaluating llms on multitask scientific long-context understanding and reasoning. In ICLR, 2025.
- [17] X. Du, Y. Yao, K. Ma, B. Wang, T. Zheng, K. Zhu, M. Liu, Y. Liang, X. Jin, Z. Wei, et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. arXiv preprint arXiv:2502.14739, 2025.
- [18] Y. Du, A. R. Jamasb, J. Guo, T. Fu, C. Harris, Y. Wang, C. Duan, P. Liò, P. Schwaller, and T. L. Blundell. Machine learning-aided generative molecular design. Nature Machine Intelligence, 6(6):589–604, 2024.
- [19] L. L. Ferreira and A. D. Andricopulo. Admet modeling approaches in drug discovery. Drug discovery today, 24(5):1157–1165, 2019.
- [20] N. Ferruz, S. Schmidt, and B. Höcker. ProtGPT2 is a deep unsupervised language model for protein design. Nature communications, 2022.
- [21] J. Gong, L. Bai, P. Ye, W. Xu, N. Liu, J. Dai, X. Yang, and W. Ouyang. Cascast: Skillful high-resolution precipitation nowcasting via cascaded modelling. arXiv preprint arXiv:2402.04290, 2024.
- [22] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [23] J. Gu, X. Jiang, Z. Shi, H. Tan, X. Zhai, C. Xu, W. Li, Y. Shen, S. Ma, H. Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.
- [24] D. Guo, F. Wu, F. Zhu, F. Leng, G. Shi, H. Chen, H. Fan, J. Wang, J. Jiang, J. Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.
- [25] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [26] T. Guo, B. Nan, Z. Liang, Z. Guo, N. Chawla, O. Wiest, X. Zhang, et al. What can large language models do in chemistry? a comprehensive benchmark on eight tasks. In NeurIPS, 2023.
- [27] Z. He, X. Wu, P. Zhou, R. Xuan, G. Liu, X. Yang, Q. Zhu, and H. Huang. Cmmu: a benchmark for chinese multi-modal multi-type question understanding and reasoning. In IJCAI, 2024.
- [28] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [29] H. Hersbach, B. Bell, P. Berrisford, S. Hirahara, A. Horányi, J. Muñoz-Sabater, J. Nicolas, C. Peubey, R. Radu, D. Schepers, et al. The era5 global reanalysis. Quarterly journal of the royal meteorological society, 146(730):1999–2049, 2020.
- [30] K. Huang, T. Fu, W. Gao, Y. Zhao, Y. Roohani, J. Leskovec, C. W. Coley, C. Xiao, J. Sun, and M. Zitnik. Artificial intelligence foundation for therapeutic science. Nature chemical biology, 18(10):1033–1036, 2022.
- [31] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.
- [32] R. Kessler, G. Narayan, A. Avelino, E. Bachelet, R. Biswas, P. Brown, D. Chernoff, A. Connolly, M. Dai, S. Daniel, et al. Models and simulations for the photometric lsst astronomical time series classification challenge (plasticc). Publications of the Astronomical Society of the Pacific, 2019.
- [33] S. Kim and E. E. Bolton. Pubchem: A large-scale public chemical database for drug discovery. Open Access Databases and Datasets for Drug Discovery, pages 39–66, 2024.
- [34] S. Kulal, P. Pasupat, K. Chandra, M. Lee, O. Padon, A. Aiken, and P. S. Liang. Spoc: Search-based pseudocode to code. Advances in Neural Information Processing Systems, 32, 2019.
- [35] Y. Labrak, A. Bazoge, E. Morin, P.-A. Gourraud, M. Rouvier, and R. Dufour. Biomistral: A collection of open-source pretrained large language models for medical domains. arXiv preprint arXiv:2402.10373, 2024.
- [36] J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, and J. Kang. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 2020.

- [37] B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [38] H. Li, Z. Wang, J. Wang, A. K. H. Lau, and H. Qu. Cllmate: A multimodal llm for weather and climate events forecasting. arXiv preprint arXiv:2409.19058, 2024.
- [39] J. Li, Y. Liu, W. Fan, X.-Y. Wei, H. Liu, J. Tang, and Q. Li. Empowering molecule discovery for molecule-caption translation with large language models: A chatgpt perspective, 2024.
- [40] Y. Liang, R. Zhang, L. Zhang, and P. Xie. DrugChat: towards enabling chatgpt-like capabilities on drug molecule graphs. arXiv preprint arXiv:2309.03907, 2023.
- [41] Z. Lin, C. Deng, L. Zhou, T. Zhang, Y. Xu, Y. Xu, Z. He, Y. Shi, B. Dai, Y. Song, et al. Geogalactica: A scientific large language model in geoscience. arXiv preprint arXiv:2401.00434, 2023.
- [42] P. Liu, Y. Ren, J. Tao, and Z. Ren. Git-mol: A multi-modal large language model for molecular science with graph, image, and text. Computers in biology and medicine, 2024.
- [43] S. Liu, J. Wang, Y. Yang, C. Wang, L. Liu, H. Guo, and C. Xiao. Conversational drug editing using retrieval and domain feedback. In ICLR, 2024.
- [44] X. Liu, Y. Guo, H. Li, J. Liu, S. Huang, B. Ke, and J. Lv. DrugLLM: Open large language model for few-shot molecule generation. arXiv preprint arXiv:2405.06690, 2024.
- [45] Y. Liu, S. Ding, S. Zhou, W. Fan, and Q. Tan. Moleculargpt: Open large language model (llm) for few-shot molecular property prediction. arXiv preprint arXiv:2406.12950, 2024.
- [46] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024.
- [47] Z. Liu, W. Zhang, Y. Xia, L. Wu, S. Xie, T. Qin, M. Zhang, and T.-Y. Liu. Molxpt: Wrapping molecules with text for generative pre-training. arXiv preprint arXiv:2305.10688, 2023.
- [48] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022.
- [49] R. Luo, L. Sun, Y. Xia, T. Qin, S. Zhang, H. Poon, and T.-Y. Liu. BioGPT: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics, 2022.
- [50] Y. Luo, J. Zhang, S. Fan, K. Yang, Y. Wu, M. Qiao, and Z. Nie. Biomedgpt: Open multimodal generative pre-trained transformer for biomedicine. arXiv preprint arXiv:2308.09442, 2023.
- [51] R. K. Luu and M. J. Buehler. BioinspiredLLM: Conversational large language model for the mechanics of biological and bio-inspired materials. Advanced Science, 2024.
- [52] L. Lv, Z. Lin, H. Li, Y. Liu, J. Cui, C. Y.-C. Chen, L. Yuan, and Y. Tian. Prollama: A protein large language model for multi-task protein language processing. IEEE Transactions on Artificial Intelligence, 2025.
- [53] C. Ma, Z. Hua, A. Anderson-Frey, V. Iyer, X. Liu, and L. Qin. Weatherqa: Can multimodal language models reason about severe weather? arXiv preprint arXiv:2406.11217, 2024.
- [54] A. Madani, B. McCann, N. Naik, N. S. Keskar, N. Anand, R. R. Eguchi, P.-S. Huang, and R. Socher. Progen: Language modeling for protein generation. arXiv preprint arXiv:2004.03497, 2020.
- [55] A. McNaught. The iupac international chemical identifier. Chem. Int, 28(6):12–15, 2006.
- [56] G. Miolo, G. Mantoan, and C. Orsenigo. Electramed: a new pre-trained language representation model for biomedical nlp. arXiv preprint arXiv:2104.09585, 2021.
- [57] OpenAI. Gpt, 2023. URL https://openai.com/gpt. Large Language Model.
- [58] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [59] L. Phan, A. Gatti, Z. Han, N. Li, J. Hu, H. Zhang, C. B. C. Zhang, M. Shaaban, J. Ling, S. Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.
- [60] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

- [61] T. Schmit, M. Gunshor, G. Fu, T. Rink, K. Bah, and W. Wolf. GOES-R Advanced Baseline Imager (ABI) Algorithm Theoretical Basis Document for: Cloud and Moisture Imagery Product (CMIP). University of Wisconsin–Madison, 2010.
- [62] T. M. Smith, V. Lakshmanan, G. J. Stumpf, K. L. Ortega, K. Hondl, K. Cooper, K. M. Calhoun, D. M. Kingfield, K. L. Manross, R. Toomey, et al. Multi-radar multi-sensor (mrms) severe weather and aviation products: Initial operating capabilities. Bulletin of the American Meteorological Society, 2016.
- [63] L. Sun, Y. Han, Z. Zhao, D. Ma, Z. Shen, B. Chen, L. Chen, and K. Yu. Scieval: A multi-level large language model evaluation benchmark for scientific research. In AAAI, 2024.
- [64] R. Taconis, M. G. Ferguson-Hessler, and H. Broekkamp. Teaching science problem solving: An overview of experimental work. Journal of Research in Science Teaching: The Official Journal of the National Association for Research in Science Teaching, 2001.
- [65] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [66] T. Truong Jr and T. Bepler. Poet: A generative model of protein families as sequences-of-sequences. In NeurIPS, 2023.
- [67] A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 2019.
- [68] L. Wang, Y. Li, X. Xu, and F. Li. Characteristic analysis of dual-polarization weather radar echoes of convective precipitation and snowfall in the mount everest region. Atmosphere, 2021.
- [69] R. Wang, X. Fang, Y. Lu, and S. Wang. The pdbbind database: Collection of binding affinities for proteinligand complexes with known three-dimensional structures. Journal of medicinal chemistry, 47(12): 2977–2980, 2004.
- [70] X. Wang, Z. Hu, P. Lu, Y. Zhu, J. Zhang, S. Subramaniam, A. R. Loomba, S. Zhang, Y. Sun, and W. Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models. In ICML, 2024.
- [71] Y. Wang, H. Zhao, S. Sciabola, and W. Wang. cMolGPT: a conditional generative pre-trained transformer for target-specific de novo molecular generation. Molecules, 2023.
- [72] Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In NeurIPS, 2024.
- [73] N. Webersinke, M. Kraus, J. A. Bingler, and M. Leippold. Climatebert: A pretrained language model for climate-related text. arXiv preprint arXiv:2110.12010, 2021.
- [74] J. W. Wilson, N. A. Crook, C. K. Mueller, J. Sun, and M. Dixon. Nowcasting thunderstorms: A status report. Bulletin of the American Meteorological Society, 1998.
- [75] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, and V. Pande. Moleculenet: a benchmark for molecular machine learning. Chemical science, 9(2):513–530, 2018.
- [76] XAI. Grok, 2023. URL https://x.ai/grok. Large Language Model.
- [77] Y. Yan, H. Wen, S. Zhong, W. Chen, H. Chen, Q. Wen, R. Zimmermann, and Y. Liang. Urbanclip: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web. In WWW, 2024.
- [78] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.
- [79] X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, B. Yu, G. Zhang, H. Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.
- [80] Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, S. Song, and G. Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

- [81] D. Zhang, W. Liu, Q. Tan, J. Chen, H. Yan, Y. Yan, J. Li, W. Huang, X. Yue, W. Ouyang, et al. Chemllm: A chemical large language model. arXiv preprint arXiv:2402.06852, 2024.
- [82] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675, 2019.
- [83] W. Zhang, M. Cai, T. Zhang, Y. Zhuang, and X. Mao. Earthgpt: A universal multi-modal large language model for multi-sensor image comprehension in remote sensing domain. IEEE Transactions on Geoscience and Remote Sensing, 2024.
- [84] W. Zhang, M. Cai, Y. Ning, T. Zhang, Y. Zhuang, H. Chen, J. Li, and X. Mao. Earthgpt-x: Enabling mllms to flexibly and comprehensively understand multi-source remote sensing imagery. arXiv preprint arXiv:2504.12795, 2025.
- [85] Y. Zhang, M. Long, K. Chen, L. Xing, R. Jin, M. I. Jordan, and J. Wang. Skilful nowcasting of extreme precipitation with nowcastnet. Nature, 2023.
- [86] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, Y. Duan, H. Tian, W. Su, J. Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
- [87] L. Zhuo, Z. Chi, M. Xu, H. Huang, H. Zheng, C. He, X.-L. Mao, and W. Zhang. Protllm: An interleaved protein-language llm with protein-as-word pre-training. arXiv preprint arXiv:2403.07920, 2024.

- A PrismaX Team We provide the full list of PrismaX team members as follows:

- • Yuhao Zhou3* (Shanghai Artificial Intelligence Laboratory & Engineering Research Center of Machine Learning and Industry Intelligence, Ministry of Education & Sichuan University)
- • Yiheng Wang3* (Shanghai Artificial Intelligence Laboratory & Shanghai Jiao Tong University)
- • Xuming He3* (Shanghai Artificial Intelligence Laboratory & Zhejiang University)
- • Ao Shen (Shanghai Artificial Intelligence Laboratory & Fudan University)
- • Ruoyao Xiao (Chinese Academy of Meteorological Sciences)
- • Zhiwei Li (Shanghai Artificial Intelligence Laboratory)
- • Qiantai Feng (Shanghai Artificial Intelligence Laboratory)
- • Zijie Guo (Shanghai Artificial Intelligence Laboratory & Fudan University)
- • Yuejin Yang (Shanghai Artificial Intelligence Laboratory & Research Institute of Intelligent Complex Systems, Fudan University)
- • Hao Wu (Shanghai Artificial Intelligence Laboratory)
- • Wenxuan Huang (Shanghai Artificial Intelligence Laboratory)
- • Jiaqi Wei (Shanghai Artificial Intelligence Laboratory & Zhejiang University)
- • Dan Si (Sichuan University)
- • Xiuqi Yao (Shanghai Artificial Intelligence Laboratory)
- • Jia Bu (East China Normal University)
- • Haiwen Huang (East China Normal University)
- • Manning Wang (Fudan University)
- • Tianfan Fu (Shanghai Artificial Intelligence Laboratory & Nanjing University)
- • Shixiang Tang (Shanghai Artificial Intelligence Laboratory)
- • Ben Fei (Shanghai Artificial Intelligence Laboratory)
- • Dongzhan Zhou (Shanghai Artificial Intelligence Laboratory)
- • Fenghua Ling (Shanghai Artificial Intelligence Laboratory)
- • Yan Lu (Shanghai Artificial Intelligence Laboratory)
- • Siqi Sun (Fudan University & Shanghai Artificial Intelligence Laboratory )
- • Chenhui Li (East China Normal University)
- • Guanjie Zheng (Shanghai Jiao Tong University)
- • Jiancheng Lv (Sichuan University)
- • Wenlong Zhang† (Shanghai Artificial Intelligence Laboratory)
- • Lei Bai† (Shanghai Artificial Intelligence Laboratory)

† Corresponding author: zhangwenlong@pjlab.org.cn, bailei@pjlab.org.cn

3This work was primarily conducted during the author’s internship at the Shanghai Artificial Intelligence Laboratory.

### B Tasks Overview

Table 6: Details of tasks in SFE.

Discipline Task TaskID Subtask Question Type Astronomy

Star Perception and Comparison

A001 Galaxy morphology classification MCQ

(8)

A011 Transient detection MCQ

A003 Light curve classification MCQ

Light Curve Analysis

- A006 Surface temperature estimation Exact Match

- A007 Gravitational constant estimation Exact Match

- A008 Metallicity estimation Exact Match

- A009 Alpha-element abundance estimation

Exact Match

- A010 Radial velocity estimation Exact Match

Spectral Analysis

Chemistry

Structure Analysis

C001 Elemental composition recognition Exact Match

(19)

- C003 IUPAC name recognition Exact Match

- C004 Molecular description generation Open Question

Property Prediction

- C005 Lipinski drug-likeness estimation Exact Match

C007 Topological polar surface area calculation

Exact Match

- C023 Absorption property prediction Exact Match

- C024 Distribution property prediction Exact Match

- C025 Metabolism property prediction Exact Match

- C026 Excretion property prediction Exact Match

- C027 Toxicity property prediction MCQ

- C028 QM9 quantum chemical property prediction

Exact Match

- C029 Protein-ligand binding affinity prediction

Exact Match

Reaction Prediction

- C009 Product SMILES prediction Exact Match

- C010 Reaction classification MCQ MCQ

- C012 Reactant molecular recognition Exact Match

- C013 Reaction condition and catalyst prediction

Molecular Design and Optimization

C006 Synthetic accessibility estimation Exact Match

C018 Molecular property optimization MCQ Drug screening C030 Virtual screening MCQ

Earth

Circulation Understanding

E004 perception of extreme precipitation distribution

MCQ

(14)

- E006 Moisture source attribution MCQ

- E007 subtropical high ridge control region MCQ

- E008 thermocline depth recognition MCQ E017 Outlier analysis MCQ

Continued on next page

Continued Discipline Task Subtask

Subtask Question Type

ID

Multivariate Understanding

E005 Precipitation event analysis Open Question

- E021 Convective weather types identification

MCQ

- E022 Convective influence regions identification

MCQ

Remote Sensing Perception

- E011 SAR image grounding Exact Match

- E012 Infrared image grounding Exact Match

Ground Data Comparison Reasoning

E014 temperature sequence comparison MCQ

E016 Vertical profile comparison MCQ E018 differential prediction comparison Open Question E001 Satellite-radar matching MCQ

Life Science

Proteomic Sequencing

L001 Fragment ion peaks count Exact Match L003 De novo peptide sequence Exact Match

(14)

Metabolomic Sequencing

- L005 Atom count inference Exact Match

- L006 Molecular composition inference Exact Match

- L007 Spectrum Matching MCQ

Protein Structure Analysis

- L008 Protein chain count Exact Match

- L009 Small molecule count Exact Match

- L010 Specified protein detection Exact Match

- L019 Protein structure feature analysis Open Question

RNA Structure Analysis

- L014 Structural domains identification MCQ

- L015 Structural domain count Exact Match

- L016 RNA type identification Exact Match

- L017 RNA secondary structure inverse folding

Exact Match

- L020 Structural motifs and positions description

Open Question

Materials Science

Crystal Structure Analysis

- M001 Atomic composition description MCQ

- M002 Crystal group identification Exact Match

- M003 Crystal formula determination Exact Match

- M004 Elemental valence state prediction Exact Match M006 Stability estimation MCQ

(11)

Energy band and DOS interpretation

Electronic Property Analysis

M008

Open Question

M010 Band gap classification Open Question M018

Valence state and electronic orbital analysis

Open Question

- M020 Phase identification Open Question

- M021 Lattice constant estimation Open Question

- M022 Crystal grain size estimation Open Question

XRD Pattern Inference

### C Tasks Description

Astronomy. In the field of astronomy, raw astronomical observations span multiple modalities, including images, time series, spectra, and hyperspectral observations [1, 32, 2]. Specifically, data collected for celestial objects often covers multiple bands and includes rich attribute information. Therefore, inspired by [3], we divide astronomical tasks into three categories based on data modalities, including Celestial Image Understanding and Comparison, Time-series Light Curve Analysis, and Spectral Analysis, each of which is detailed below.

- • Star Perception and Comparison. The morphology and brightness of celestial objects provide crucial insights into galaxy formation, which is essential for studying the evolutionary history of stars. For instance, in the transient detection sub-task, given the pre-transient and post-transient images, as well as the difference between the two, MLLMs are required to determine whether a transient event has occurred during this process.
- • Light Curve Analysis. Photometric light curves record the brightness of celestial objects over time, and are crucial for estimating the intrinsic properties and predicting the dynamic behavior of stars. For example, in the Light Curve Classification sub-task, MLLMs are provided with time-series photometric data and are required to classify the type of celestial object.
- • Spectral Analysis Spectral data enables the inference of physical properties such as surface temperature and chemical composition. Therefore, we have designed 5 sub-tasks based on different physical properties. For example, in the Surface Temperature Estimation task, MLLMs are required to estimate the surface temperature of a target celestial object based on its spectral profile.

Chemistry. To systematically evaluate MLLMs’ performance in the field of chemistry, we organize our benchmark around four major capability categories: Structure Analysis, Molecular Property Prediction, Chemical Reaction Prediction, and Molecular Design and Optimization. These capabilities reflect fundamental scientific questions central to chemical research and application.

- • Structure Analysis. Structure Analysis tests a model’s grasp of symbolic chemistry, such as identifying elemental composition, describing molecules using natural language, and generating IUPAC (International Union of Pure and Applied Chemistry) names (e.g., methane, ethane) [55]. It examines the foundational ability to ensure that the model correctly recognizes the chemical structure and elements of the molecules.
- • Property Prediction. Molecular Property Prediction aims to assess a model’s ability to infer chemical and biological properties from molecular structures. It focuses on the properties of chemical molecules that scientists are most interested in, such as drug molecules’ ADMET (i.e., absorption, distribution, metabolism, excretion, toxicity) [19], synthetic accessibility, and quantum mechanics energy, etc.
- • Reaction Prediction. Chemical Reaction Prediction assesses the MLLM’s understanding of chemical reactions, including the products of the reaction, reaction centers (the region within a molecule where bond breaking and forming occur), and required reaction conditions (e.g., catalyst).
- • Molecular Design and Optimization. Molecular Design and Optimization evaluates the MLLM’s capacity to generate molecules with desirable chemical properties (e.g., pharmaceutical properties). These tasks focus on identifying novel and diverse molecules with desirable molecular properties (e.g., druglikeness, solubility, binding affinities to the target proteins), which are fundamental tasks in drug and material discovery [18].
- • Drug Screening. Drug Screening evaluates the MLLM’s ability to identify biologically relevant candidate molecules based on protein sequences and molecular structure information [30].

Earth. The Earth science domain includes various data modalities. First, variables can be either temporal or static. Meteorological time series are often used to analyze strong convective weather events [85, 21, 74]. Static attributes, such as long-term average temperature maps, are typically used to capture climate patterns. Second, strong correlations often exist between variables. For example, the position of the western Pacific subtropical high (i.e., the 5880 hPa line) is positively correlated with the coverage range of precipitation belts. Third, variables may originate from diverse sources, including ground-based weather radar [62], multi-spectral satellite imagers [61], and various remote

sensing instruments. To cater to such diverse scenarios, we establish a comprehensive Earth science task paradigm that focuses on 4 tasks, each comprising multiple sub-tasks.

- • Task1: Circulation Understanding. In atmospheric physics, large-scale circulation patterns play a central role in modulating regional weather phenomena, such as precipitation anomalies. For example, in the moisture source attribution sub-task, MLLMs are required to reason the origin of moisture contributing to the regional rainfall by integrating information from moisture fluxes and wind fields.
- • Task2: Multivariate Understanding. Complex weather phenomena often arise from the interplay among multiple meteorological variables. For example, in the precipitation event analysis sub-task, MLLMs are required to consider precipitation, temperature, wind direction, moisture flux, and geopotential height collectively to infer the underlying causes of rainfall events.
- • Task3: Remote Sensing Perception. Remote sensing imagery provides multispectral observations of Earth’s surface, enabling comprehensive monitoring of environmental and urban conditions. To assess the capability of MLLMs in identifying object categories and localizing their spatial positions from satellite data, we design two sub-tasks using different image modalities. MLLMs are required to first count the number of objects belonging to a specified category, and then identify the spatial location of each instance using the bounding box.
- • Task4: Ground Data Comparison Reasoning. Ground observational data have been widely used in scenarios such as global climatological analysis and extreme precipitation monitoring [68, 9]. To evaluate the capability of MLLMs in comparing and reasoning meteorological terminology and data distributions, we define 4 tasks based on different variables. For example, in the temperature sequence comparison task, MLLMs need to compare annual temperature series from two different time periods and describe differences of statistical characteristics

Life. To access the capabilities of MLLMs in the domain of Life Science, we define three core categories: Biomolecular Profiling, Sequence Reasoning, and Structure Interpretation. These capabilities reflect core scientific practices in the Life Science, where understanding arises from linking molecular measurements, symbolic sequences, and structural representations to functional biological meaning.

- • Biomolecular Profiling. Biomolecular Profiling involves extracting quantitative and compositional features of biological molecules. These profiling skills are fundamental in high-throughput biological workflows such as mass spectrometry and molecular diagnostics, where precise characterization of biomolecular species is critical for downstream analysis.
- • Sequence Reasoning. Sequence Reasoning addresses the functional interpretation of biological sequences, including proteins and RNAs. Such reasoning requires models to understand sequencefunction relationships, modularity, and constraints that are central to biological information flow and molecular design.
- • Structure Interpretation. Structure Interpretation targets the analysis of spatial and symbolic representations of biological macromolecules. These tasks emphasize spatial abstraction and multi-view understanding, as biological structures are often represented through hybrid formats that combine 2D, symbolic, and textual information.

Materials Science. Materials science presents unique challenges grounded in structural periodicity, quantum-level electronic properties, and empirical characterization signals. To reflect these, we design tasks around three core capabilities: Crystal Structure Analysis, Electronic Property Analysis, and XRD Pattern Inference.

- • Crystal Structure Analysis. Crystal Structure Analysis involves symbolic, compositional, and descriptive reasoning over crystalline systems. These tasks reflect how materials scientists represent, classify, and formally describe the structural foundations of solids, where a deep understanding of symmetry operations, periodicity, and the thermodynamic and structural stability conditions is required.
- • Electronic Property Analysis. Electronic Property Analysis aims at interpreting the electronic behavior of a material through visual and symbolic representations. These tasks require models to bridge diagrammatic understanding with functional inference, emulating expert workflows in electronic materials analysis and property prediction.

- • XRD Pattern Inference. XRD Pattern Inference focuses on scientific reasoning grounded in experimental diffraction signals. These tasks require models to connect peak pattern distributions with structural parameters through symbolic and numerical reasoning.

- D Question Format For multiple-choice questions, the question is formatted as:

Multiple-choice questions format example

You are an expert in discipline and need to solve the following question. The question is a multiple-choice question. Answer with the option letter from the given choices.

{task prompt} {question} {options}

For exact match questions, the question is formatted as: Exact match questions format example

You are an expert in discipline and need to solve the following question. The question is an exact match question. Answer the question using a single word or phrase.

{task prompt} {question}

For open-ended questions, the question is formatted as: Exact match questions format example

You are an expert in discipline and need to solve the following question. The question is an open-ended question. Answer the question using a phrase.

{task prompt} {question}

Finally, for bounding boxes extraction questions, the question is formatted as: bounding boxes extraction questions format example

You are an expert in discipline and need to solve the following question. The question is an open-ended question. Answer the question using a phrase. Each bounding box is represented by four numbers, corresponding to the positions x_min, y_min, x_max, and y_max. The coordinate origin is located at the top-left corner of the image, and the bottom-right corner has coordinates (1, 1). Therefore, both x and y range from 0 to 1. In the final output, two bounding boxes are separated by a semicolon, and all bounding boxes are enclosed in square brackets. Here is an example output: "[x1_min, y1_min, x1_max, y1_max; x2_min, y2_min, x2_max, y2_max]" (no quotation marks).

{task prompt} {question}

- E Responses Evaluation

Given the MLLM’s responses, the ground truth answer and the problem, we ask GPT-4o to judge the correctness with the following prompt.

LLM-as-a-Judge prompt

You are a strict evaluator assessing answer correctness. You must score the model’s prediction on a scale from 0 to 10, where 0 represents an entirely incorrect answer and 10 indicates a highly correct answer.

# Input Question: {question} Ground Truth Answer: {answer} Model Prediction: {prediction}

# Evaluation Rules

- - The model prediction may contain the reasoning process, you should spot the final answer from it.
- - For multiple-choice questions: Assign a higher score if the predicted answer matches the ground truth, either by option letters or content. Include partial credit for answers that are close in content.
- - For exact match and open-ended questions:

- * Assign a high score if the prediction matches the answer semantically, considering variations in format.
- * Deduct points for partially correct answers or those with incorrect additional information.

- - Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- - Treat numerical answers as correct if they match within reasonable precision
- - For questions requiring units, both value and unit must be correct

# Scoring Guide Provide a single integer from 0 to 10 to reflect your judgment of the answer’s correctness.

# Strict Output format example 4

### A Subtask Description

- A.1 Astronomy

- A.1.1 Galaxy Morphology Classification (A001)

In this subtask, we address the problem of galaxy morphology classification from single-band optical imagery. The goal is to determine the structural category of a galaxy—such as merging, spiral, or edge-on—based on its visual appearance. This task poses significant challenges due to the complex and diverse nature of galactic structures, along with variations in scale, orientation, and observational noise. We adopt a multi-class classification formulation with four expert-defined morphological labels. Accurate recognition requires capturing both global shape and fine-grained features, which motivates the need for models with strong spatial sensitivity and robustness to astrophysical variations.

Galaxy Morphology Classification (A001) Images:

[Figure 179]

Question: What is the structure of the galaxy in the image?

Options:

- (A) Disturbed Galaxies.
- (B) Merging Galaxies.
- (C) Unbarred Tight Spiral Galaxies.
- (D) Edge-on Galaxies with out Bulge.

Answer: A

- A.1.2 Light Curve Classification

This subtask focuses on the classification of astronomical light curves across multiple photometric bands. Each instance represents the temporal flux variation of a celestial source, visualized as band-wise curves. The objective is to identify the physical origin of the variability—ranging from active galactic nuclei (AGN) to microlensing or tidal disruption events. The problem is inherently noisy and sparse, with irregular sampling and band-dependent features. We pose this as a multiclass classification task, where models must learn to align temporal dynamics with astrophysical signatures. This setting tests the model’s capacity to extract semantic patterns from structured time-series representations under observational constraints.

Light Curve Classification (A003) Images:

[Figure 180]

Question: Based on the provided images of different bands, what type of light curve is it?

#### Options:

- (A) M-dwarf.
- (B) AGN.
- (C) TDE.
- (D) Microlens-Single.

Answer: B

- A.1.3 Surface Temperature Estimation

This subtask addresses the estimation of stellar surface temperature from spectral observations. Each input is a flux-calibrated spectrum plotted over wavelength, with red markers denoting the Line Spread Function sigma. The task is to predict the effective temperature (Teff) of the star, formulated as a regression problem. Accurate estimation relies on modeling subtle variations in spectral line profiles and continuum shapes, which encode temperature-sensitive features. The challenge lies in handling high-dimensional, noisy spectral data while preserving physically meaningful cues. This task benchmarks a model’s ability to extract quantitative astrophysical parameters from structured scientific measurements under realistic observational conditions.

Surface Temperature Estimation (A006) Images:

[Figure 181]

#### Question:

The image provided is a spectrum of a star. The X-axis represents wavelength in Angstroms (\u00c5), and the Y-axis represents flux. The red spots indicate the Line Spread Function Sigma. What is the estimated effective temperature(Teff, in Kelvin) of the star based on the spectrum? You only need to respond with a float value.

Answer: 5402.044921875

#### A.1.4 Gravitational Constant Estimation

This subtask focuses on estimating the surface gravity (log g) of a star from its observed spectrum. The input is a one-dimensional flux curve over wavelength, with red markers indicating the Line Spread Function sigma. Surface gravity estimation is formulated as a regression task, requiring precise inference from subtle line-broadening features and continuum patterns. The complexity arises from the interplay of noise, spectral resolution, and astrophysical diversity. Robust models must learn to extract informative cues while generalizing across varying stellar types. This task provides a controlled benchmark for evaluating a model’s ability to recover physical parameters from high-dimensional spectral data.

Gravitational Constant Estimation (A007) Images:

[Figure 182]

Question:

The image provided is a spectrum of a star. The X-axis represents wavelength in Angstroms (\u00c5), and the Y-axis represents flux. The red spots indicate the Line Spread Function Sigma. What is the estimated surface gravity (log g) of the star based on the spectrum? You only need to respond with a float value.

Answer: 4.1094794273376465

#### A.1.5 Metallicity Estimation

This subtask targets the estimation of stellar metallicity ([M/H]) from spectroscopic data. Each input is a high-resolution stellar spectrum, where the flux is plotted against wavelength, and red markers indicate the Line Spread Function sigma. The goal is to regress the overall metal abundance, a key indicator of stellar composition and evolution. Metallicity signatures manifest as fine absorption features, often shallow and blended, making the task sensitive to spectral resolution and noise. We cast this as a regression problem that requires models to extract and integrate subtle chemical patterns. This benchmark emphasizes precision in learning astrophysical properties from dense observational data.

Metallicity Estimation (A008) Images:

[Figure 183]

#### Question:

The image provided is a spectrum of a star. The X-axis represents wavelength in Angstroms (\u00c5), and the Y-axis represents flux. The red spots indicate the Line Spread Function Sigma. What is the estimated overall metallicity ([M/H]) of the stellar based on the spectrum? You only need to respond with a float value.

#### Answer:

-0.07942300289869308

#### A.1.6 Alpha-element Abundance Estimation

This subtask targets the estimation of stellar metallicity ([M/H]) from spectroscopic data. Each input is a high-resolution stellar spectrum, where the flux is plotted against wavelength and red markers indicate the Line Spread Function sigma. The goal is to regress the overall metal abundance, a key indicator of stellar composition and evolution. Metallicity signatures manifest as fine absorption features, often shallow and blended, making the task sensitive to spectral resolution and noise. We cast this as a regression problem that requires models to extract and integrate subtle chemical patterns. This benchmark emphasizes precision in learning astrophysical properties from dense observational data.

Alpha-element Abundance Estimation (A009) Images:

[Figure 184]

#### Question:

The image provided is a spectrum of a star. The X-axis represents wavelength in Angstroms (\u00c5), and the Y-axis represents flux. The red spots indicate the Line Spread Function Sigma. What is the estimated overall alpha abundance ([alpha/M]) of the star based on the

spectrum? You only need to respond with a float value. Answer: 0.12627530097961426

#### A.1.7 Radial Velocity Estimation

This subtask aims to estimate the radial velocity of a star from its observed spectrum. The input is a flux-versus-wavelength curve, with red markers indicating the Line Spread Function sigma. Radial velocity manifests as Doppler-induced shifts in spectral lines, often subtle and entangled with instrumental effects and stellar variability. We cast this as a regression problem that requires precise modeling of line positions under varying noise and resolution. Success in this task demands both global pattern recognition and fine-grained alignment sensitivity. This benchmark probes a model’s capability to infer stellar motion from spectral displacement, under realistic observational constraints.

Radial Velocity Estimation (A010) Images:

[Figure 185]

Question:

The image provided is a spectrum of a star. The X-axis represents wavelength in Angstroms (00c5),˘ and the Y-axis represents flux. The red spots indicate the Line Spread Function Sigma. What is the estimated radial velocity of the star based on the spectrum? You only need to respond with a float value.

Answer:

-5.426918983459473

#### A.1.8 Transient Detection

This subtask targets the detection of astrophysical transients through image differencing. Given a triplet of images—current observation, historical reference, and their pixel-wise difference—the objective is to determine the presence of a transient event, such as a supernova or variable star. The challenge arises from subtle photometric variations and observational artifacts that may obscure true events or generate false positives. We formulate the task as a binary classification problem, demanding models that can effectively learn discriminative temporal cues while suppressing noise and instrumental distortions. This setting provides a controlled yet realistic benchmark for evaluating temporal sensitivity in astronomical image analysis.

Transient Detection (A011) Images:

[Figure 186]

[Figure 187]

[Figure 188]

#### Question:

The first image is the latest reference image, the second image is the historical reference image, and the third image shows the difference between the two. Please determine whether a transient event has occurred based on these three images.

#### Options:

- (A) Yes, a transient event has occurred.
- (B) No, a transient event has not occurred.
- (C) I don’t know. Answer: B

- A.2 Chemistry

- A.2.1 Elemental Composition Recognition (C001)

The Elemental Composition Recognition task focuses on identifying and quantifying the atomic constituents in a given molecular structure image. Given a 2D chemical diagram, the task requires the model to enumerate all unique elements present and tally the number of atoms for each element. This subtask demands a comprehensive understanding of chemical notation and spatial relationships, challenging the system’s capability to parse complex visual and symbolic information. Precision in recognizing overlapping or ambiguous atoms is vital. This task serves as a fundamental benchmark for evaluating visuo-semantic reasoning in machine perception frameworks and sets a strong baseline for molecular understanding.

Elemental Composition Recognition (C001) Images:

[Figure 189]

Question: From the molecular structure <image>, list all the unique elements present in the compound and the number of atoms of each.

Answer: "C": 20, "H": 28, "N": 10, "O": 13, "P": 2, "S": 2

#### A.2.2 IUPAC Name Recognition (C003)

The IUPAC Name Recognition subtask aims to bridge molecular structure identification with systematic chemical nomenclature. Given a molecular diagram, the model is required to generate its precise IUPAC name, demonstrating both structural understanding and proficiency in chemical linguistics. This task rigorously tests the model’s ability to parse complex visual information and translate it into standardized nomenclature, reflecting the fundamentals of computational chemistry and pattern recognition. In essence, this subtask challenges the model to establish a robust mapping from visually represented molecules to their unique textual identifiers, a critical step toward comprehensive chemical informatics.

IUPAC Name Recognition (C003) Images:

[Figure 190]

Question: From the molecular structure <image>, provide the molecular IUPAC name.

Answer: N-[(5S,5aS,8aR,9R)-9-(4-hydroxy-3,5-dimethoxyphenyl)-8-oxo-5a,6,8a,9-tetrahydro-5H[2]benzofuro[5,6-f][1,3]benzodioxol-5-yl]-2-hydroxybenzamide

#### A.2.3 Molecular Description Generation (C004)

The subtask centers on the automatic generation of natural language descriptions for molecular structures from their graphical representations. Given an image depicting a chemical compound,

the model is expected to output a coherent and accurate textual summary detailing the molecule’s key structural features, constituent components, and functional groups. This task challenges the model’s understanding of visual molecular notation and its chemical semantics, thereby facilitating applications ranging from automated database curation to assistive chemical education. By bridging vision and language in the context of cheminformatics, this subtask provides a rigorous benchmark for evaluating multimodal reasoning and generation capabilities.

Molecular Description Generation (C004) Images:

[Figure 191]

Question: Generate a natural language description for the given molecule <image>.

#### Answer:

The molecule is a branched amino tetrasaccharide comprising N-acetyl-beta-D-glucosamine at the reducing end with a N-acetyl-alpha-D-galactosaminyl-(1->3)-[alpha-L-fucosyl-(1->2)]beta-D-galactosyl moiety attached at the 4-position. It has a role as an epitope.

#### A.2.4 Lipinski Drug-likeness Estimation (C005)

In this subtask, we evaluate the ability of models to estimate key drug-likeness properties according to Lipinski’s Rule of Five, a fundamental heuristic in medicinal chemistry for assessing oral bioavailability. Given a molecular structure image, the model is required to compute five physicochemical descriptors: molecular weight, XLogP (an estimate of octanol-water partition coefficient), hydrogen bond donor count, hydrogen bond acceptor count, and rotatable bond count. The outputs are standardized as a JSON object for consistency. This task demands not only a robust understanding of chemical structures but also precise quantitative reasoning, analogous in rigor to structure-property prediction challenges.

Lipinski Drug-likeness Estimation (C005) Images:

[Figure 192]

#### Question:

From the molecular structure <image>, calculate the five Lipinski rule indicators with values rounded to one decimal place: molecular weight, LogP, number of hydrogen bond donors,

number of hydrogen bond acceptors, and number of rotatable bonds. Please output as a JSON dict with these exact keys (no units):

{ "Molecular Weight": , "XLogP": , "Hydrogen Bond Donor Count": , "Hydrogen Bond Acceptor Count": , "Rotatable Bond Count": , }

#### Answer:

{ "Molecular Weight": 285.3, "XLogP": 0.8, "Hydrogen Bond Donor Count": 2.0, "Hydrogen Bond Acceptor Count": 4.0, "Rotatable Bond Count": 0.0 }

#### A.2.5 Topological Polar Surface Area Calculation (C007)

In this subtask, we focus on the quantitative assessment of a compound’s polarity via its Topological Polar Surface Area (TPSA). Given a molecular structure in image format, participants are required to accurately deduce and compute the TPSA, a widely adopted descriptor for predicting drug absorption and permeability. The answer, precise to one decimal place, reflects the sum of surface contributions from polar atoms, and is to be reported without units. This task demands not only rigorous chemical understanding but also proficiency in molecule-feature interpretation, mirroring real-world scenarios in cheminformatics and drug discovery pipelines.

Topological Polar Surface Area Calculation (C007) Images:

[Figure 193]

Question:

Calculate the Topological Polar Surface Area (TPSA) of the compound from the molecular structure <image>. Report the value in square angstroms (00c5˘ 00b2),˘ rounded to one decimal place. Please omit the unit (u00c5 u00b2) in your answer.

Answer: 38.3

#### A.2.6 Absorption Property Prediction (C023)

In this subtask, we investigate the prediction of key absorption-related properties directly from molecular structure images. The task requires estimating multiple pharmacokinetic attributes, including lipophilicity (logP), aqueous solubility, Caco-2 permeability, and hydration free energy, as

well as binary classifications on human intestinal absorption (HIA), bioavailability, and P-glycoprotein (Pgp) inhibition. Given a compound’s structure, the model is prompted to output a precise JSON dictionary with these properties. This design emphasizes exact-match evaluation, reflecting practical drug discovery settings where comprehensive absorption profiling from molecular information is essential for candidate selection and prioritization.

Absorption Property Prediction (C023) Images:

[Figure 194]

#### Question:

This is an exact-match question. Based on the molecular structure shown in <image>, please predict the following absorption-related properties for the molecule. For each property, provide a value or a ’yes’ / ’no’ answer where applicable. The properties are as follows: Lipophilicity, which refers to the Logarithm of the partition coefficient (logP); Solubility, which is the aqueous solubility of the molecule (mg/L); Hydration Free Energy, which is the free energy of hydration (kcal/mol); Caco-2 Permeability, which is the effective permeability of the molecule across Caco-2 cells (cm/s); Is HIA Activity, which indicates whether the molecule shows high human intestinal absorption (yes/no); Is Bioavailable, which indicates whether the molecule is bioavailable (yes/no); and Is Pgp Inhibitor, which indicates whether the molecule inhibits P-glycoprotein (yes/no). Please return your predictions in a JSON dictionary using exactly the following keys. Do not include units or additional descriptions in your output:

{ "lipophilicity": , "solubility": , "caco2_permeability": , "is_hia_activity": , "is_bioavailable": , "is_pgp_inhibitor": }

#### Answer:

{ "lipophilicity":1.08, "solubility":-4.6202577876, "caco2_permeability":-4.460681, "is_bioavailable":"yes", "is_pgp\_inhibitor":"no", }

#### A.2.7 Distribution Property Prediction (C024)

In this subtask, we focus on the prediction of pharmacokinetic distribution properties for small molecules by analyzing their chemical structures. Specifically, the challenge requires the model to infer whether a given molecule can penetrate the blood-brain barrier (is_bbb), as well as to estimate its plasma protein binding rate (ppbr, in percent) and volume of distribution at steady state (vdss, in

L/kg). The model must accurately extract salient molecular features from visualized structures and reason about their impact on distribution metrics, returning a tightly constrained JSON output for downstream evaluation. This design emphasizes precise, structure-informed reasoning and format adherence.

Distribution Property Prediction (C024) Images:

[Figure 195]

#### Question:

This is an exact-match question. Based on the molecular structure shown in <image>, please predict the following distribution-related properties for the molecule. For each property, provide a value or a ’yes’ / ’no’ answer where applicable. The properties are as follows: Is BBB, which indicates whether the molecule can cross the Blood-Brain Barrier (yes/no); PPBR, which refers to the Plasma Protein Binding Rate (unit: %); and VDss, which is the Volume of Distribution at steady state (unit: L/kg). Please return your predictions in a JSON dictionary using exactly the following keys. Do not include units or additional descriptions in your output:

{ "is_bbb": , "ppbr": , "vdss": , }

#### Answer:

{ "is_bbb":"no", "ppbr":99.6, "vdss":1, }

#### A.2.8 Metabolism Property Prediction (C025)

The Metabolism Property Prediction subtask focuses on the systematic evaluation of xenobiotic biotransformation potential by analyzing molecular structures for their interaction with cytochrome P450 isoforms. Specifically, the task requires the model to assess a given molecule’s inhibitory effects on CYP2C19, CYP2D6, CYP3A4, CYP1A2, and CYP2C9, as well as to determine substrate specificity for CYP2C9, CYP2D6, and CYP3A4. Formulated as an exact-match classification, the subtask demands precision in capturing molecular features governing P450 metabolism, and supports downstream drug design by enabling robust prediction of drug-drug interaction risks and metabolic liabilities.

Metabolism Property Prediction (C025)

[Figure 196]

#### Question:

This is an exact-match question. Based on the molecular structure shown in <image>, please predict the following metabolism-related properties for the molecule. For each property, provide a ’yes’ or ’no’ answer. The properties are as follows: whether the molecule inhibits CYP P450 enzymes including 2C19, 2D6, 3A4, 1A2, and 2C9; and whether it is a substrate for CYP2C9, CYP2D6, and CYP3A4. Please return your predictions in a JSON dictionary using exactly the following keys. Do not include any additional descriptions in your output:

{ "CYP2C19_inhibition": , "CYP2D6_inhibition": , "CYP3A4_inhibition": , "CYP1A2_inhibition": , "CYP2C9_inhibition": , "CYP2C9_substrate": , "CYP2D6_substrate": , "CYP3A4_substrate": , }

#### Answer:

{ "CYP2C19_inhibition":"no", "CYP2D6_inhibition":"no", "CYP3A4_inhibition":"no", "CYP1A2_inhibition":"no", "CYP2C9_inhibition":"no", "CYP2C9_substrate":"yes", "CYP2D6_substrate":"no", "CYP3A4_substrate":"yes" }

#### A.2.9 Excretion Property Prediction (C026)

In this subtask, the model is presented with a molecular structure image and tasked to predict key excretion-related pharmacokinetic properties, specifically half-life and clearance. The model must analyze the given molecular structure and infer the half-life (in hours) and the intrinsic clearance rate (in mL/min/kg), returning the results as precise numerical values in a standardized JSON format. This requires the model to establish a strong correspondence between molecular structural features and ADME property prediction, ensuring both robustness and reliability in the estimation of pharmacokinetic profiles from purely visual molecular information.

Excretion Property Prediction (C026)

[Figure 197]

#### Question:

This is an exact-match question. Based on the molecular structure shown in <image>, please predict the following excretion-related properties for the molecule. For each property, provide a numerical value. The properties are as follows: Half-life, which is the half-life duration of the drug (unit: hours); and Clearance, which refers to the intrinsic clearance rate of the drug (unit: mL/min/kg). Please return your predictions in a JSON dictionary using exactly the following keys. Do not include units or additional descriptions in your output:

{ "half_life": , "clearance": , }

#### Answer:

{ "half_life":18, "clearance":150, }

#### A.2.10 Toxicity Property Prediction (C027)

In this subtask, we address the challenging problem of toxicity property prediction for chemical compounds, a key concern in drug discovery and environmental safety. Given the molecular structure of a compound, the objective is to systematically assess its potential toxic effects across diverse biological dimensions, such as cellular processes, organ specificity, metabolic interactions, and environmental risks. Participants are required to analyze each molecule and select all relevant toxicity aspects from a predefined set of possible outcomes. This formulation enables a comprehensive evaluation of toxicity profiles, facilitating downstream applications and risk assessment in both pharmaceutical and ecological contexts.

Toxicity Property Prediction (C027) Images:

[Figure 198]

#### Question:

Please analyze the toxicity of this molecule. Which of the following aspects might be affected? (Multiple selections possible)

#### Options:

- (A) Potential impact on cell proliferation, apoptosis, and cell cycle.
- (B) Potential induction of oxidative stress, impact on metabolism, or mitochondrial function.
- (C) Potential effect on enzyme activity, receptor activation, or inhibition.
- (D) Potential impact on gene expression, transcription factor activation, and other gene-level regulation.
- (E) Potential influence on immune responses, cytokine release, etc.
- (F) Potential toxicity to specific organs (e.g., liver, heart, kidney).
- (G) Potential impact on development or reproduction processes.
- (H) Potential carcinogenicity or environmental toxicity to aquatic life, plants, soil, etc.
- (I) Other (please specify).

Answer: C, F

- A.2.11 QM9 Quantum Chemical Property Prediction (C028)

In this subtask, we assess a model’s ability to predict quantum chemical properties for small organic molecules from QM9, given only their 2D structural images. The task is challenging, as it requires translating visual molecular representations into accurate estimations of key quantum mechanical quantities, such as dipole moment, HOMO-LUMO energies, and atomization energies. Participants are required to output predictions for sixteen specified properties in a strict JSON format. This subtask evaluates both the holistic and fine-grained chemical understanding of deep models, bridging vision and quantum chemistry, and highlights the capacity for end-to-end property inference from purely visual cues.

QM9 Quantum Chemical Property Prediction (C028) Images:

[Figure 199]

#### Question:

This is an exact-match question. Based on the molecular structure shown in <image>, please predict the following quantum mechanical properties for the molecule. For each property, provide a numerical value. The properties are as follows: mu, which is the dipole moment (unit: Debye); alpha, the isotropic polarizability (unit: Bohr3); homo, the energy of the highest occupied molecular orbital (unit: Hartree); lumo, the energy of the lowest unoccupied molecular orbital (unit: Hartree); gap, the energy gap between HOMO and LUMO (unit: Hartree); r2, the electronic spatial extent (unit: Bohr2); zpve, the zero point vibrational energy (unit: Hartree); u0, the internal energy at 0K (unit: Hartree); u298, the internal energy at 298.15K (unit: Hartree); h298, the enthalpy at 298.15K (unit: Hartree); g298, the free energy at 298.15K (unit: Hartree); cv, the heat capacity at 298.15K (unit: cal/(mol*K)); u0_atom, the atomization energy at 0K (unit: kcal/mol); u298_atom, the atomization energy at 298.15K (unit: kcal/mol); h298_atom, the atomization enthalpy at 298.15K (unit: kcal/mol); and g298_atom, the atomization free energy at 298.15K (unit: kcal/mol). Please return your

predictions in a JSON dictionary using exactly the following keys. Do not include units or additional descriptions in your output:

{ "mu": , "alpha": , "homo": , "lumo": , "gap": , "r2": , "zpve": , "u0": , "u298": , "h298": ,

- "g298": , "cv": , "u0_atom": , "u298_atom": ,
- "h298_atom": ,

- "g298_atom": , } Answer:

{ "mu": 3.9822, "alpha": 67.58, "homo": -0.2636, "lumo": 0.0037, "gap": 0.2673, "r2": 973.0186, "zpve": 0.136173, "u0": -458.980805, "u298": -458.973242,

- "h298": -458.972298,

- "g298": -459.012433, "cv": 30.273, "u0_atom": -1696.9424682681, "u298_atom": -1707.3057794031,
- "h298_atom": -1716.788695411, "g298_atom": -1580.3933391711, }

#### A.2.12 Protein-ligand Binding Affinity Prediction (C029)

The Protein-ligand Binding Affinity Prediction subtask evaluates the capability of computational models to quantitatively infer the binding affinity between a given small molecule and a target protein, based on molecular structure images and protein FASTA sequences. Accurate prediction of binding constants (Kd, Ki, or IC50) is critical in rational drug design and virtual screening. This task presents both the molecular structure and the amino acid sequence, requiring the model to comprehensively analyze intermolecular interactions and sequence-specific features. The outcome is measured by the precision of predicted affinity values, reflecting the model’s understanding of protein-ligand recognition mechanisms.

Protein-ligand Binding Affinity Prediction (C029) Images:

[Figure 200]

Question: This question is based on the molecular structure <image> and the protein FASTA sequence of 1a30(>1A30_1|Chains A, B|HIV-1 PROTEASE|Human immunodeficiency virus 1 (11676) PQITLWKRPLVTIKIGGQLKEALLDTGADDTVIEEMSLPGRWKPKMIGGIGGFIKVRQYDQIIIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF >1A30_2|Chain C|TRIPEPTIDE GLU-ASP-LEU| EDL Please predict the binding affinity (Kd/Ki/IC50), which represents the molecule’s binding affinity to the target protein (format: Kd/Ki/IC50 = <value> <unit>).

Answer: Ki=50uM

#### A.2.13 Product SMILES Prediction (C009)

Product SMILES Prediction is designed to evaluate a model’s chemical reasoning in reaction product prediction. Each instance presents a reactant structure as an image, deliberately omitting all auxiliary components such as catalysts and solvents, thereby restricting the available information to the core reactant(s). The model is tasked to generate the canonical SMILES string corresponding to the most probable product that arises from the depicted structure. This setting challenges models to capture fundamental mechanistic knowledge and to generalize beyond rote memorization, paralleling the rigorous demand for abstraction prevalent in chemical synthesis planning.

Product SMILES Prediction (C009) Images:

[Figure 201]

Question: Generate the SMILES representation of the most likely product based on the given reactant structure <image>. The image only contains reactants, excluding catalysts, reagents, or solvents. Answer: CCOC(=O)C1=CC2=C(C=CC=C2O1)N3CCN(CC3)CC4=CC=CC=C4

#### A.2.14 Reaction Classification (C010)

The Reaction Classification subtask aims to evaluate the model’s ability to accurately identify and categorize organic reactions based on presented molecular structures. Each instance provides a reactant’s chemical diagram and a multiple-choice list of plausible reaction types, often involving subtle distinctions such as variations in leaving groups or coupling partners. The task presents significant challenges, requiring both recognition of functional groups and a nuanced understanding of common reaction mechanisms. Correctly classifying these transformations is essential for predicting chemical reactivity and synthetic planning, thereby serving as a robust benchmark for the model’s chemical reasoning and expert-level pattern recognition capabilities.

Reaction Classification (C010) Images:

[Figure 202]

Question: Based on the given reactant structure <image>, classify the most likely type of reaction.

Options:

- (A) Iodo Buchwald-Hartwig amination
- (B) Bromo N-arylation
- (C) Chloro Buchwald-Hartwig amination
- (D) Bromo Buchwald-Hartwig amination
- (E) Chloro N-arylation

Answer: C

#### A.2.15 Reactant Molecular Recognition (C012)

The Reactant Molecular Recognition subtask targets a fundamental challenge in reaction informatics: determining molecular identities from structural representations. Given an image capturing the reactant structures in a chemical reaction, the objective is to accurately extract and enumerate the molecular formula for each reactant, preserving their left-to-right order as depicted. This subtask emphasizes both precise chemical perception and the translation of visual molecular information into standard chemical formulas, bridging the gap between image-based molecular recognition and structured chemical knowledge. The task demands high accuracy and robustness against diverse molecular drawings, providing rigorous benchmarks for visual reasoning models in the chemistry domain.

Reactant Molecular Recognition (C012) Images:

[Figure 203]

Question: From the reactants structure <image>, identify the molecular formula of each reactant involved in the reaction, in left-to-right order.

Answer: C5H2Cl2IN, C8H10N2O

#### A.2.16 Reaction Condition and Catalyst Prediction (C013)

In this subtask, the objective is to predict the required catalysts, solvents, or reagents for a given organic transformation, based solely on the reactant structures. Participants are presented with a visual depiction of the reactants alongside a set of multiple-choice options, each consisting of chemical structures representing possible conditions. The task may have one or more correct answers, mirroring the ambiguity and diversity frequently encountered in chemical synthesis. This problem formulation emphasizes both chemical knowledge and reasoning, challenging models to bridge the gap between structural input and operational context, much like practical decision-making in synthetic chemistry.

Reaction Condition and Catalyst Prediction (C013) Images:

[Figure 204]

Question: This is a multiple-choice question with one or more correct answers. Based on the reactant structures <image>, select all catalysts, solvents, or reagents required for the reaction.

#### Options:

[Figure 205]

[Figure 206]

[Figure 207]

(A) (B) (C)

[Figure 208]

[Figure 209]

[Figure 210]

(D) (E) (F)

[Figure 211]

[Figure 212]

(G) (H)

Answer: C, E, F, G

#### A.2.17 Synthetic Accessibility Estimation (C006)

The Synthetic Accessibility Estimation subtask aims to evaluate a model’s capability in assessing the practical feasibility of chemical synthesis from molecular structure images. Each instance presents a single compound as a 2D structural diagram, prompting the model to predict a Synthetic Accessibility Score (SAS) within the standardized 1.0–10.0 range, where lower scores denote higher synthetic tractability. This task reflects the essential integration of cheminformatics perception and domain reasoning, requiring the system to accurately distill both structural complexity and functional group context. Performance in this subtask is critical for downstream applications in de novo molecular design and automated retrosynthetic planning.

Synthetic Accessibility Estimation (C006) Images:

[Figure 213]

Question:

Estimate the Synthetic Accessibility Score (SAS) of the compound based on the molecular structure <image>. The score should be a float value between 1.0 (very easy to synthesize) and 10.0 (very difficult to synthesize), rounded to one decimal place.

Answer: 3.8

#### A.2.18 Molecular Property Optimization (C018)

The Molecular Property Optimization (C018) subtask is designed to evaluate models’ ability to discern and select molecular structures exhibiting superior performance in annotated chemical properties.

In each instance, participants are presented with a set of diverse molecules, each paired with a specific property value or descriptor. The task requires selecting all molecules that demonstrate optimal or above-threshold performance for the given properties. This setting examines not only the recognition of molecular structures but also a nuanced understanding of structure-property relationships. Ultimately, the subtask serves as a rigorous benchmark for algorithmic reasoning in molecular optimization scenarios.

Molecular Property Optimization (C018) Question: Below are eight individual molecules, each annotated with a property. Select all molecules that exhibit superior performance in their respective property. Options:

[Figure 214]

[Figure 215]

[Figure 216]

(A) (B) (C)

[Figure 217]

[Figure 218]

[Figure 219]

(D) (E) (F)

[Figure 220]

(G)

Answer: A, B, C, F

#### A.2.19 Virtual Screening (C030)

Virtual screening is a critical subtask in computational drug discovery, aiming to efficiently identify active compounds with high binding affinity toward a specific biological target. In our benchmark, we present a series of molecular candidates alongside structural representations of the target protein. The task requires the model to discern, among a provided set of molecules, those capable of binding to the target. This subtask evaluates the model’s ability to reason about 3D molecular interactions, recognize chemical compatibilities, and generalize knowledge across diverse molecular structures, forming a key foundation for subsequent lead optimization and further experimental validation.

Virtual Screening (C030) Images:

[Figure 221]

[Figure 222]

[Figure 223]

Question: This is a multiple-choice question with one or more correct answers. Based on the target structure <image><image><image>, select all molecules that can bind to it.

#### Options:

[Figure 224]

[Figure 225]

[Figure 226]

(A) (B) (C)

[Figure 227]

[Figure 228]

[Figure 229]

(D) (E) (F)

[Figure 230]

[Figure 231]

[Figure 232]

(G) (H) (I)

[Figure 233]

[Figure 234]

[Figure 235]

(J) (K) (L)

[Figure 236]

[Figure 237]

[Figure 238]

(M) (N) (O)

#### Answer:

A, B, C, D, E

- A.3 Earth

- A.3.1 Satellite-radar matching

This subtask evaluates a model’s ability to match geostationary satellite imagery with corresponding radar-based VIL (Vertically Integrated Liquid) measurements. Given two infrared satellite channels (IR069 and IR107) capturing cloud-top properties, the objective is to identify the VIL image that best corresponds to the observed convective structures. The challenge lies in learning the non-trivial spatial and radiometric mappings between satellite brightness temperatures and radar reflectivity proxies, which vary across cloud regimes and precipitation intensities. This task benchmarks a model’s capacity for multi-modal image understanding, emphasizing physical consistency and pattern recognition across heterogeneous atmospheric sensing modalities.

Satellite-radar matching (E001) Images:

[Figure 239]

[Figure 240]

#### Question:

The first image is the IR069 image, the second image is the IR107 image, and the third to eighth images are VIL images, corresponding to options A to F. Which VIL image best matches the content shown in the IR069 and IR107 images?

#### Options:

[Figure 241]

[Figure 242]

[Figure 243]

(A) (B) (C)

[Figure 244]

[Figure 245]

[Figure 246]

(D) (E) (F)

Answer: A

- A.3.2 Perception of Extreme Precipitation Distribution

This subtask focuses on localized heavy precipitation detection from gridded satellite-based rainfall products. Given a daily accumulated precipitation map over East Asia, the goal is to identify specific geographic regions experiencing intense rainfall (greater than 50 mm). The task requires precise spatial reasoning and pattern recognition under varying rainfall intensities and distributions. Successful models must exhibit fine-grained geospatial understanding and demonstrate robust threshold-based detection under noisy meteorological conditions. By emphasizing accurate identification of rainfall extremes, this task serves as a critical benchmark for evaluating models in climate-aware perception and disaster risk assessment.

Perception of Extreme Precipitation Distribution (E004) Images:

[Figure 247]

#### Question:

This is a map showing the daily accumulated precipitation in East Asia. Please select the precipitation area that includes regions with over 50 mm of rainfall from the following options.

#### Options:

- (A) Southeast coastal region in China.
- (B) The southern waters of Japan.
- (C) Southern Philippines.
- (D) West coast of India.
- (E) Hainan Island.
- (F) Southern Thailand.
- (G) Korea.
- (H) North China region. Answer: A, B, D, F

#### A.3.3 Precipitation Event Analysis

This subtask targets causal reasoning in synoptic-scale meteorological diagnostics. Given multi-modal meteorological charts—including precipitation distribution, geopotential height fields, wind vectors, moisture flux, and vertical motion—the objective is to infer the underlying physical mechanisms driving precipitation events. The specific case focuses on eastern China and demands an accurate interpretation of trough positioning, moisture transport pathways, and dynamical lifting conditions. Successful solutions require a holistic understanding of atmospheric dynamics and their spatial interactions. This task evaluates the model’s ability to integrate heterogeneous data modalities and produce structured, domain-specific explanatory analyses—a key step toward trustworthy AI in meteorological forecasting.

Precipitation Event Analysis (E005) Images:

[Figure 248]

[Figure 249]

[Figure 250]

#### Question:

The first figure uses shading to represent precipitation. The second figure shows 500 hPa geopotential height with contour lines, 850 hPa wind field with vectors, and moisture flux with shading. The third figure displays 850 hPa geopotential height as contour lines and 500 hPa vertical velocity as shading. Please analyze the precipitation formation process in eastern China.

#### Answer:

The precipitation area is located ahead of a westerly trough, which extends southward from Gansu to Yunnan. The region is influenced by warm advection ahead of the trough and receives moisture transported from the South China Sea. The subtropical high over the northwestern Pacific is situated over the ocean east of the Philippines, steering moisture from the South China Sea toward the eastern coastal areas of China. In the precipitation zone, strong upward vertical motion leads to low-level atmospheric convergence, continuously supplying moisture.

#### A.3.4 Moisture Source Distribution

This subtask focuses on identifying the dominant moisture source regions contributing to precipitation events, based on multi-variable meteorological analysis. Given composite plots of precipitation distribution and low-level atmospheric dynamics—including wind vectors and moisture flux—the task requires reasoning over spatial patterns to infer the most plausible moisture origins. The example centers on East China, where the South China Sea is revealed as the key moisture supplier through southerly transport. This task challenges models to associate physical variables across domains and to recognize geophysical transport pathways, encouraging the development of systems that understand and reason about real-world Earth system dynamics.

Moisture Source Distribution (E006)

[Figure 251]

[Figure 252]

#### Question:

The first figure uses shading to represent precipitation. The second figure shows 500 hPa geopotential height with contour lines, 850 hPa wind field with vectors, and moisture flux with shading. What are the sources of moisture for precipitation in the East China region?

#### Options:

- (A) Bohai Sea.
- (B) Yellow Sea.
- (C) East China Sea.
- (D) South China Sea.
- (E) Bay of Bengal.

Answer: D

- A.3.5 Subtropical High Ridge Control Region

This subtask investigates the spatial extent of the Western Pacific Subtropical High Ridge through the analysis of synoptic-scale meteorological fields. The figure presents a combination of 500 hPa geopotential height, low-level wind vectors, and moisture flux shading. Participants are required to identify the geographic regions under the influence of the subtropical high, marked by anticyclonic circulation and suppressed moisture transport. The task demands spatial reasoning over multivariable meteorological patterns, reinforcing the model’s understanding of atmospheric circulation systems. It emphasizes the capability to localize semi-permanent pressure systems and their dynamic control zones, crucial for modeling large-scale weather and climate variability.

Subtropical High Ridge Control Region (E007)

[Figure 253]

#### Question:

The figure shows 500 hPa geopotential height with contour lines, 850 hPa wind field with vectors, and moisture flux with shading. Please identify regions controlled by Western Pacific Subtropical High Ridge from the following options.

#### Options:

- (A) Thailand.
- (B) Eastern China.
- (C) Southern China.
- (D) Northern Philippines.
- (E) India.
- (F) Sri Lanka.
- (G) Malaysia.
- (H) Nepal

Answer: E, F

- A.3.6 Thermocline Depth Recognition

This subtask targets the recognition of thermocline structures from vertical ocean temperature profile visualizations. Given a contour plot of ocean temperature variation with depth, models are tasked with identifying the depth range corresponding to the maximum vertical temperature gradient—an indicator of the thermocline. This requires spatial interpretation of temperature isolines and their compression zones. The task emphasizes fine-grained understanding of ocean stratification and vertical mixing processes, critical for climate dynamics and marine forecasting. By evaluating the ability to locate subtle thermal transitions, the subtask probes models’ capacity to extract physical meaning from complex geophysical patterns.

Thermocline Depth Recognition (E008) Images:

[Figure 254]

#### Question:

What is the approximate depth range of the maximum vertical temperature gradient in the figure?

#### Options:

- (A) 50-150m.
- (B) 100-200m.
- (C) 150-250m.
- (D) 200-300m.

Answer: A

- A.3.7 SAR Image Grounding

This subtask centers on object grounding in synthetic aperture radar (SAR) imagery, specifically the detection and localization of maritime vessels. Given a high-resolution SAR image, models are required to identify all visible ships and annotate their positions using horizontal bounding boxes in normalized coordinates. The challenge lies in recognizing targets with diverse scales, orientations, and backscatter intensities under varying sea surface conditions. Unlike optical imagery, SAR data demands proficiency in interpreting texture and structure under coherent illumination. This task evaluates a model’s capacity to robustly ground semantically meaningful objects in complex geospatial scenes beyond the visible spectrum.

SAR Image Grounding (E011) Images:

[Figure 255]

Question: Detect all the ships shown in the remote sensing image and describe them using a horizontal bounding box.

Answer: [0.95,0.37,1.00,0.39;0.69,0.53,0.79,0.56;0.36,0.06,0.46,0.09;0.85,0.88,0.91,0.91;0.29,0.97,0.34,1.00;0.04,0.54,0.07,0.59]

- A.3.8 Infrared Image Grounding

This subtask focuses on object grounding in thermal infrared imagery, targeting the detection of vehicles in low-light or thermally dominant environments. Given an infrared image, the model must localize all visible cars using horizontal bounding boxes in normalized coordinates. Unlike RGB images, infrared data emphasizes temperature contrasts, posing unique challenges for distinguishing objects with similar thermal signatures or partially occluded forms. Accurate detection under these conditions reflects a model’s robustness to modality shifts and its ability to exploit spatial and thermal cues. This task is a critical benchmark for grounding algorithms in non-visible spectra under real-world conditions.

Infrared Image Grounding (E012) Images:

[Figure 256]

Question: Detect all Cars shown in the image and describe using a horizontal bounding box.

Answer: [0.32,0.15,0.40,0.30]

#### A.3.9 Temperature Sequence Comparison

This subtask evaluates the model’s capacity for fine-grained temporal reasoning over climate data. Given a global annual temperature series spanning 1981 to 2000, the task requires analyzing two consecutive decades to extract key statistical characteristics—value range, extremes, and temporal trends—followed by comparative insights between periods. Successful completion demonstrates the model’s ability to interpret dense visual time series and capture nuanced variations in climate dynamics. By grounding numeric trends in specific temporal intervals, this task benchmarks a model’s capability for structured temporal analysis, a core skill for scientific understanding and decision-making in environmental monitoring and forecasting.

Temperature Sequence Comparison (E014) Images:

[Figure 257]

#### Question:

The figure below is a global annual average temperature series from 1981 to 2000, describing the temperature information (range, extreme values and trends) for the time periods 1981-1990 and 1991-2000, as well as the difference between the two time periods.

#### Answer:

The temperature range in 1981-1990 is from 278.0 to 278.5, with the minimum value occurring in 1985 and the maximum value occurring in 1990, showing an overall increasing trend. The

temperature range in 1991-2000 is from 278.1 to 278.5, with the minimum value occurring in 1993 and the maximum value occurring in 1998, showing an overall increasing trend. The minimum temperature in 1981-1990 is lower, the maximum temperature in 1981-1990 is higher, and the growth trend in 1991-2000 is greater.

#### A.3.10 Vertical Profile Comparison

This subtask investigates the model’s ability to compare and analyze vertical oceanographic profiles. The input consists of a plot showing two temperature (or other scalar) profiles across varying ocean depths. The model is required to identify the depth intervals where the discrepancy between the two profiles is the most significant. This tests the model’s proficiency in recognizing depth-wise gradient differences and isolating meaningful regions of divergence. Accurate performance in this task reflects the model’s capacity for precise visual differentiation in scientific plots, a foundational skill for downstream applications such as anomaly detection, climate diagnostics, and ocean monitoring.

Vertical Profile Comparison (E016) Images:

[Figure 258]

Question: Select the two depth ranges with the largest difference between the two data in the figure.

Options:

- (A) 0-100m.
- (B) 100-200m.
- (C) 200-300m.
- (D) 300-400m.

Answer: B, C

#### A.3.11 Outlier Analysis

This subtask evaluates the model’s capability in identifying regional outliers within a global temperature anomaly map. The input is a 2D spatial distribution of surface temperature anomalies, where the model must pinpoint the regions exhibiting the most extreme warm and cold deviations. This task necessitates nuanced spatial pattern recognition and comparative reasoning to distinguish subtle yet climatically significant anomalies. The correct identification of both hot and cold outliers demonstrates the model’s effectiveness in high-resolution geospatial anomaly detection, an essential competency for climate diagnostics, extreme event attribution, and model evaluation in Earth system science.

Outlier Analysis (E017) Images:

[Figure 259]

Question: The following figure shows the global surface temperature anomaly. Please select the two approximate areas with the hottest and coldest temperature anomalies respectively.

#### Options:

- (A) North Central Eurasia.
- (B) Qinghai-Tibet Plateau.
- (C) East Asia.
- (D) Northern North America.

Answer: A, D

- A.3.12 Differential Prediction Comparison

This subtask challenges the model to perform spatiotemporal comparison between observed and forecasted radar reflectivity sequences, focusing on Vertical Integrated Liquid (VIL) as an indicator of precipitation intensity. The model must identify key discrepancies in intensity evolution, spatial distribution, and movement patterns over time. Successfully capturing differences such as forecast underestimation of scattered echoes, intensity weakening, and deviation in convective system trajectories demonstrates the model’s ability to assess forecast accuracy and temporal dynamics. This task is critical for advancing precipitation nowcasting and improving understanding of model limitations in representing complex atmospheric convection processes.

Differential Prediction Comparison (E018) Images:

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

#### Question:

The first 4 images illustrate the observed radar reflectivity sequence, and the last 4 images present the forecasted sequence. The variable visualized in both sequences is the Vertical Integrated Liquid (VIL), which represents the total amount of liquid water contained in a vertical column of the atmosphere. value between 0 and 16 indicates high probability of a sunny weather; value between 16 and 74 indicates high probability of light rain; value between 74 and 133 indicates high probability of moderate rain; value between 133 and 160 indicates high probability of heavy rain; value between 160 and 181 indicates high probability of very heavy rain; value between 181 and 219 indicates high probability of intense rain; value above 219 indicates high probability of extreme rain .Please highlight the major discrepancies between the forecasted and observed fields.

#### Answer:

The forecast sequence exhibits a rapid weakening of intensity, with an overall trend of strengthening first and then weakening, while the observed sequence maintains a relatively steady intensity. Small scattered echoes are not captured in the forecast sequence. Additionally, the convective systems in the observed sequence show a southeastward movement, whereas the forecasted systems remain largely stationary.

#### A.3.13 Convective Weather Types Identification

This subtask requires the model to integrate multi-parameter meteorological data—such as wind shear, supercell composite indices, temperature advection, and reflectivity mosaics—to identify geographic regions at highest risk of severe convective weather. By synthesizing dynamic atmospheric instability, moisture, and storm maintenance probabilities across spatially distributed observations, the model must pinpoint the area most susceptible to severe weather impacts. This task evaluates the model’s capacity to fuse heterogeneous weather indicators for robust convective event prediction, advancing interpretability and accuracy in severe weather nowcasting and supporting timely hazard mitigation efforts.

Convective Weather Types Identification (E021)

#### Question:

The following 8 figures represent weather conditions and each figure contains multiple weather parameters. The most important parameters in each figure is provided as follow:Figure 1: 0-6km wind shear magnitude. Figure 2: Supercell Composite Parameter. Figure 3: 850 mb Temperature Advection. Figure 4: Lifting Condensation Level height. Figure 5: Geostrophic equivalent potential vorticity. Figure 6: 700-500mb lapse rate instability. Figure 7: Mesoscale Convective System maintenance probability. Figure 8: Base Reflectivity Mosaic. Choose one geographical area(s) most likely to be impacted by the severe weather event from the four options provided.

#### Options:

- (A) east Texas.
- (B) east-central Minnesota into northwest Wisconsin.
- (C) southern Indiana into Kentucky.
- (D) eastern and middle Tennessee into northwestern Alabama and northern Mississippi.

Answer: A

- A.3.14 Convective Influence regions Identification

This subtask challenges the model to assess severe weather potential by synthesizing multidimensional meteorological parameters, including wind shear, temperature advection, convective instability, and reflectivity patterns. The objective is to determine the likelihood and timing of convective watches or relevant winter weather advisories. This requires a nuanced interpretation of probabilistic indicators and dynamic atmospheric conditions to classify the scenario into distinct watch categories or winter weather events. The task evaluates the model’s ability to integrate complex weather data and uncertainty estimates, advancing predictive precision for operational forecasting and early warning decision-making in severe weather contexts.

Convective Influence regions Identification (E022)

#### Question:

The following 8 figures represent weather condition,s and each figure contains multiple weather parameters. The most important parameters in each figure is provided as follow:Figure 1: 0-6km wind shear magnitude. Figure 2: Supercell Composite Parameter. Figure 3: 850 mb Temperature Advection. Figure 4: Lifting Condensation Level height. Figure 5: Geostrophic equivalent potential vorticity. Figure 6: 700-500mb lapse rate instability. Figure 7: Mesoscale Convective System maintenance probability. Figure 8: Base Reflectivity Mosaic. Choose the most likely scenario regarding severe weather concerns from the options provided. Options include whether a convective watch has been issued, the probability of a future watch, or, in winter, potential weather phenomena related to winter storms.

#### Options:

- (A) Severe potential...Watch unlikely (confidence level on the expectation of a watch is 5 or 20%).
- (B) Severe potential...Watch possible (confidence level on the expectation of a watch is 40 or 60%).
- (C) Severe potential...Watch likely (confidence level on the expectation of a watch is 80 or 95%).
- (D) Heavy snow.
- (E) Severe potential...tornado watch likely (confidence level on the expectation of a watch is 80 or 95%)
- (F) Severe potential...severe thunderstorm watch likely (confidence level on the expectation of a watch is 80 or 95%)
- (G) Winter mixed precipitation.
- (H) Freezing rain.
- (I) Severe potential...watch needed soon (confidence level on the expectation of a watch is 95%).
- (J) Blizzard.
- (K) Snow squall.

Answer: A

- A.4 Life Science

- A.4.1 Fragment Ion Peaks Count (L001)

In this subtask, we task the model with the quantitative analysis of MS/MS spectra for peptide sequencing, focusing specifically on the identification and enumeration of fragment ion peaks. Given a peptide sequence and its associated spectrum, the model must accurately count the number of observed b ion and y ion peaks. This requires precise understanding of peptide fragmentation patterns

and careful interpretation of spectral data. The model’s output is formatted as a structured JSON object, enabling direct assessment of its peptide sequencing and mass spectrometry expertise. This subtask tests fundamental capabilities vital for automated proteomics data analysis.

Fragment Ion Peaks Count (L001) Images:

[Figure 270]

#### Question:

You are an expert in mass spectrometry and peptide sequencing. The peptide sequence is given as TGEVSDPVK. Given the MS/MS spectrum shown in <image>, please count and fill in the number of b ions and y ions in the following format:

{ "b ions": , "y ions": , }

#### Answer:

{ "b ions":3, "y ions":8 }

#### A.4.2 De Novo Peptide Sequence (L003)

For this subtask, we focus on the challenging problem of de novo peptide sequencing from tandem mass spectrometry (MS/MS) data. Unlike database search approaches, de novo sequencing requires inferring the peptide sequence directly from the observed spectrum, relying solely on fragmentation patterns and precise mass measurements. We provide spectra alongside an explicit set of amino acid and modification masses, removing ambiguity in interpretation. The task emphasizes expert-level reasoning, demanding accurate mapping from spectral peaks to sequence candidates, and evaluation of the most plausible peptide given all evidence. This subtask provides a rigorous benchmark for de novo sequencing algorithms and human expertise.

De Novo Peptide Sequence (L003) Images:

[Figure 271]

You are an expert in mass spectrometry and peptide sequencing. Given an MS/MS spectrum, your task is to predict the amino acid sequence of the peptide that generated the spectrum using your expert knowledge of peptide fragmentation patterns and *de novo* sequencing principles. The masses of amino acids and modifications are as follows: G: 57.021464, A: 71.037114, S: 87.032028, P: 97.052764, V: 99.068414, T: 101.047670, C+57.021: 160.030649, L: 113.084064, I: 113.084064, N: 114.042927, D: 115.026943, Q: 128.058578, K: 128.094963, E: 129.042593, M: 131.040485, H: 137.058912, F: 147.068414, R: 156.101111, Y: 163.063329, W: 186.079313, M+15.995: 147.035400, N+0.984: 115.026943, Q+0.984: 129.042594, +42.011: 42.010565, +43.006: 43.005814, -17.027: -17.026549, +43.006-17.027: 25.980265. Given the MS/MS spectrum shown in <image>, determine the most likely peptide sequence. Please write your answer as a single peptide sequence.

Answer: ISGEVPEEK

#### A.4.3 Atom Count Inference (L005)

The Atom Count Inference subtask is designed to evaluate a model’s capability in deducing specific elemental content—such as the number of sulfur atoms—using MS/MS spectrometric data. Each instance presents a compound’s spectrum and prompts the model to infer element counts, leveraging isotope distributions and spectral evidence. This task challenges the model’s understanding of mass spectrometry, particularly its ability to interpret subtle isotopic features, and requires integration of both quantitative peak information and qualitative spectral patterns. Through this setup, we systematically probe reasoning on elemental composition given real-world instrument data.

Atom Count Inference (L005) Images:

[Figure 272]

Question: From the compound’s MS/MS spectrum <image>, determine the number of sulfur atoms based on isotope pattern and other compound information in the spectrum.

Answer: 0

#### A.4.4 Molecular Composition Inference (L006)

The Molecular Composition Inference subtask evaluates a model’s capability to deduce the elemental makeup of a compound directly from its MS/MS spectral data. This is a crucial step in metabolomics analysis, where complex spectra encode rich information about molecular structure. By examining isotope patterns and fragmentation signals, the model is expected to identify which elements (such as C, H, O, N, S, etc.) constitute the parent compound. This subtask focuses on the essential challenge

of abstracting key chemical information from raw experimental data, providing a fundamental basis for subsequent molecular structure elucidation.

Molecular Composition Inference (L006) Images:

[Figure 273]

#### Question:

From the compound’s MS/MS spectrum <image>, infer the elemental composition of the compound based on isotope patterns and other compound information present in the spectrum.

Answer: C, H, O, S

#### A.4.5 Spectrum Matching (L007)

The Spectrum Matching subtask aims to evaluate the ability of models to interpret mass spectra and associate them with their corresponding molecular structures. Given a mass spectrum as input, the model must select the correct structure from a set of plausible candidates, each rendered as a chemical diagram. This task requires not only low-level pattern recognition in spectra but also high-level understanding of molecular fragmentation pathways. Accurate matching necessitates reasoning over both spectral peaks and chemical features, thus offering a rigorous and realistic testbed for machine understanding in the intersection of cheminformatics and machine learning.

Spectrum Matching (L007) Images:

[Figure 274]

This is a multiple-choice question with only one correct answer. Based on the given mass spectrum <image>, select the molecular structure that best matches it.

#### Options:

[Figure 275]

[Figure 276]

[Figure 277]

(A) (B) (C)

[Figure 278]

[Figure 279]

[Figure 280]

(D) (E) (F)

[Figure 281]

(G)

Answer: B

- A.4.6 Protein Chain Count (L008)

The Protein Chain Count subtask is designed to evaluate the model’s capability of understanding and interpreting molecular structural data as visualized in PDB format images. Specifically, given a protein’s structural image where hydrogen atoms are removed for clarity, the objective is to accurately enumerate the distinct protein chains present. This requires the model to correlate visual clusters or segments within the image to underlying biomolecular structures, demonstrating an integration of visual perception and biochemical knowledge. Such a task serves as a fundamental assessment of the model’s proficiency in recognizing discrete biological entities from complex visual input, echoing challenges in real-world scientific analysis.

Protein Chain Count (L008)

[Figure 282]

Question: From the molecular structure PDB image <image>, (remove hydrogen atoms), please answer the number of protein chains with integers.

Answer: 2

- A.4.7 Small Molecule Count (L009)

The Small Molecule Count (L009) subtask focuses on the quantitative analysis of biomolecular structures by requiring models to accurately discern and enumerate ligand chains present in PDB images, excluding hydrogen atoms. Given a molecular visualization, the task challenges algorithms to capture subtle structural cues indicative of ligand entities, testing both spatial understanding and chemical knowledge. This subtask is pivotal for evaluating a model’s ability to interpret complex biochemical illustrations, further advancing automated structural biology. Our formulation aligns with real-world use cases in molecular docking and drug discovery, where correct ligand identification underpins successful downstream applications.

Small Molecule Count (L009)

[Figure 283]

Question: From the molecular structure PDB image <image>, (remove hydrogen atoms), please answer the number of ligand chains with integers.

Answer: 2

- A.4.8 Specified Protein Detection (L010)

The Specified Protein Detection subtask is designed to evaluate a model’s capability in recognizing and classifying protein structures from visual representations. Given an input image depicting a protein’s tertiary or quaternary structure, the model is required to identify the functional class the protein belongs to among several biologically relevant categories, such as enzymes, structural proteins, or receptors. This task assesses both the fine-grained visual understanding and the domain-specific reasoning of the model, mirroring real-world scenarios in computational biology where structural inference is pivotal for downstream applications, such as drug discovery or pathway analysis.

Specified Protein Detection (L010)

[Figure 284]

Question: Based on the structures <image>, choose the most suitable description of the protein structure.

#### Options:

- (A) Enzymes
- (B) Storage Proteins
- (C) Structural Proteins
- (D) Transport Proteins
- (E) Signaling Proteins
- (F) Regulatory Proteins
- (G) Motor Proteins
- (H) Receptor Proteins
- (I) Defensive Proteins

Answer: A

- A.4.9 Protein Structure Feature Analysis (L019)

In this subtask, we evaluate the model’s capacity to interpret and describe high-resolution molecular structures from PDB crystal images. The task requires the model to identify the protein’s overall fold architecture, recognize salient secondary structure elements such as α-helices and β-sheets, and locate bound ligands or cofactors within the protein’s interior. By analyzing the spatial arrangement of atoms and surface features, the model is expected to provide concise, structurally grounded summaries that reflect an understanding of protein-ligand interactions. This setting emphasizes precise perception and reasoning over intricate spatial features essential in molecular biology and rational drug design.

Protein Structure Feature Analysis (L019) Images:

[Figure 285]

#### Question:

From the molecular structure PDB image <image>, which is from Crystal Structure of human FABP4 in complex with 2-(5,6,7,8,9,10-hexahydrobenzo[8]annulen-3-yl)acetic acid, please give description of this structure in one paragraph.

#### Answer:

The provided image displays the crystal structure of human Fatty Acid Binding Protein 4 (FABP4) in complex with the ligand 2-(5,6,7,8,9,10-hexahydrobenzoannulen-3-yl)acetic acid. The FABP4 protein, depicted in teal, adopts a characteristic β-barrel fold, composed of ten antiparallel β-strands that form a clam-shell like structure, creating an internal cavity. This cavity is where the ligand, shown in a ball-and-stick representation with grey/white carbons and red oxygens, is bound, indicating its role in sequestering and transporting fatty acids or related hydrophobic molecules. Short α-helical segments cap the ends of the β-barrel. Additionally, sulfate ions, likely from the crystallization buffer, are visible on the protein’s surface, represented by yellow sulfur atoms and red oxygen atoms.

#### A.4.10 Structural Domains Identification (L014)

The Structural Domains Identification subtask aims to assess the model’s capacity for fine-grained recognition of canonical and non-canonical RNA secondary structure elements from diagrammatic representations. Given a visualized RNA fold, the task requires discerning a range of features, including stems, hairpin,s and interior loops, bulges, multi-branched loops, and complex topologies such as pseudoknots. Precise identification of these motifs is crucial for understanding RNA’s functional and structural diversity. This subtask provides a rigorous benchmark for evaluating both domain-specific pattern recognition and the broader generalization ability of the model in RNA structure analysis.

Structural Domains Identification (L014)

[Figure 286]

Question: Based on the RNA secondary structure diagram <image>, which of the following structures might the RNA contain? (Multiple choices).

#### Options:

- (A) stem
- (B) Hairpin loop
- (C) Interior loop
- (D) bulge loop
- (E) Multi-branched loop
- (F) Pseudoknot

Answer: A, B, C, F

#### A.4.11 Structural Domain Count (L015)

The Structural Domain Count subtask is designed to assess the ability of models to interpret RNA secondary structure diagrams and accurately quantify the number of distinct stem regions present. Each instance presents a clear visual representation of an RNA structure, enabling the systematic evaluation of structural comprehension. This subtask emphasizes precise recognition of canonical stem features, free from confounding sequence information, and focuses on developing visual reasoning skills in a biological context. By reporting the count of stem regions, the task provides a straightforward yet challenging benchmark for assessing the model’s proficiency in structural annotation, a critical component for molecular understanding.

Structural Domain Count (L015) Images:

[Figure 287]

#### Question:

Examine the given RNA secondary structure diagram <image>. How many stem regions are present in the structure?

Answer: 3

- A.4.12 RNA Type Identification (L016)

This subtask focuses on RNA secondary structure inverse folding: given a diagram of an RNA secondary structure, the model must accurately classify the type of RNA it represents. The task requires the model to reason about spatial arrangements and base-pairing patterns from structural images, rather than relying on primary sequence information. This challenging setting tests the model’s ability to recognize and understand biologically meaningful motifs visualized within complex structures. Solving this subtask sheds light on the model’s capacity for visual pattern recognition in molecular biology, bridging the gap between image-based data and biomolecular function classification.

RNA Secondary Structure Inverse Folding (L017) Images:

[Figure 288]

Question: Examine the given RNA secondary structure diagram <image>. What type of RNA is represented by the structure?

Answer: Riboswitch

- A.4.13 RNA Secondary Structure Inverse Folding (L017)

The RNA Secondary Structure Inverse Folding subtask requires participants to infer the precise nucleotide sequence corresponding to a given RNA secondary structure image. Expert-level reading skills are essential, as the task demands meticulous extraction and transcription of the displayed bases, adhering strictly to biological conventions—notably, replacing thymine (T) with uracil (U) where applicable. Sequences are reported in the 5’ to 3’ direction, formatted as specified JSON outputs. This task rigorously tests both structural comprehension and attention to detail, providing a challenging benchmark for evaluating the capability of models in sequence-structure reasoning within biomolecular domains.

RNA Secondary Structure Inverse Folding (L017) Images:

[Figure 289]

#### Question:

You are a professional biologist, and you are skilled at reading RNA sequences. From the RNA secondary structure <image>, give the exact RNA sequence. Answer in capital letters only, from 5’ to 3’. If there exists any T base in the picture, replace them into U base. Please directly output the sequence.

Answer: GGGCCCAUAGCUCAGUGGUAGAGUGCCUCCUUUGCAAGGAGGAUGCCCUGGGUUCGAAUCCCAGUGGGUCCA

#### A.4.14 Structural Motifs and Positions Description (L020)

Structural Motifs and Positions Description (L020)

[Figure 290]

#### Question:

You are a professional biologist, andy you are skilled at reading and analyzing RNA secondary structure pictures. Please generate a natural language description for the given RNA <image>, find the length, describe the internal loops and hairpin loops of the RNA and point out their start and terminate position.

#### Answer:

The RNA is comprised of 119 bases in total, which has two internal loops from position U21 to C26, position U53 to C57; from position A71 to A78 and from position G9 to C104, and it also has two hairpin loops from position U33 to C44 and from position G85 to A90

- A.5 Materials Science

- A.5.1 Atomic Composition Description (M001)

This subtask evaluates the capability to interpret and reason about the atomic-scale structure of crystalline materials, as represented by lattice diagrams. The model is required to analyze detailed crystallographic features—including anion and cation arrangements, coordination environments, and stacking sequences—by selecting all correct statements from a set of options. The format enforces concise, decisive outputs, mirroring critical reading and matching of visual and textual information. In alignment with rigorous scientific inquiry, candidates are discouraged from speculation; only substantiated facts, as visually and structurally evidenced in the lattice, are to be identified.

Atomic Composition Description (M001)

[Figure 291]

[Figure 292]

[Figure 293]

Question: Please strictly follow the format—no additional text or explanation may be provided. You should only reply with the letter(s) of the correct option(s) (e.g., A, C) corresponding to the descriptions that match the diagram. Based on the different views of a material: front view <image>, side view <image> and top view <image>

#### Options:

- (A) The chemical formula of the lattice is AlN.
- (B) The N3− anions form a hexagonal close-packed sublattice.
- (C) Each N3− is tetrahedrally coordinated by four Al3+ cations.
- (D) Each Al3+ is octahedrally coordinated by six N3− anions.
- (E) Al3+ cations occupy exactly half of the tetrahedral voids in the N3− hcp lattice.
- (F) The conventional hexagonal unit cell contains four AlN formula units.
- (G) The anion stacking sequence along the c-axis is ABAB.

Answer: A, B, C, E, F, G

#### A.5.2 Crystal Group Identification (M002)

The Crystal Group Identification subtask aims to rigorously test the model’s understanding of symmetry in crystallographic contexts. Given a lattice diagram of a material, the model is prompted to directly infer the most likely space group symbol representative of the structure’s symmetry and lattice parameters, such as Pn3m1, without supplementary justification. This task is minimal and unambiguous, focusing the model’s attention on essential visual cues within the diagram. By constraining the output to a standard space group notation, we ensure objective evaluation, mirroring practical challenges faced by experts in structural materials analysis.

Crystal Group Identification (M002) Images:

[Figure 294]

[Figure 295]

[Figure 296]

#### Question:

Given different views of the material, determine the space-group symbol that best matches the symmetry and lattice features shown, and reply only with that symbol (e.g., Fm3m), without any additional text, explanation, or commentary. Based on the different views of a material: front view <image>, side view <image> and top view <image>, determine the most likely space group of the material.

Answer: Pn3m1

- A.5.3 Crystal Formula Determination (M003)

Crystal Formula Determination introduces a focused subtask designed to assess the model’s capability to infer chemical composition from visual lattice representations. Given a lattice-diagram of a crystalline material, the model is required to extract atomic identities, count stoichiometric ratios, and succinctly output the correct molecular formula (e.g., Al3Ir), without any extraneous explanation or content. This task evaluates whether models possess the fine-grained visual reasoning and domainspecific knowledge necessary to interpret crystal structures in a manner analogous to trained materials scientists. The subtask thus provides a controlled benchmark for assessing structural understanding and symbolic reasoning within the materials science domain.

Crystal Formula Determination (M003) Images:

[Figure 297]

[Figure 298]

[Figure 299]

Question: Given different views of the material, infer the molecular formula based solely on the lattice composition and reply only with that formula (e.g., Ag2O), without any additional text, explanation, or commentary. Based on the different views of a material: front view <image>, side view <image> and top view <image>, determine the molecular formula of the material.

Answer: Al3Ir

- A.5.4 Elemental Valence State Prediction (M004)

The Elemental Valence State Prediction subtask focuses on inferring the oxidation states of constituent elements in crystalline materials solely from their lattice diagrams. Given an image representing the atomic arrangement, the model must accurately identify and output the oxidation state for each element present. The annotation is strictly formatted as ‘Element: Oxidation state’ pairs, separated by commas, without additional explanation. This task evaluates the model’s ability to interpret crystallographic structures and deduce fundamental chemical properties, mirroring the visual reasoning capabilities essential for automated scientific discovery in materials science.

Elemental Valence State Prediction (M004) Images:

[Figure 300]

[Figure 301]

[Figure 302]

Question: Given different views of the material, identify the oxidation state of each element and reply only with those states formatted exactly as ’Element: Oxidation state’, separated by commas (e.g., Ag: +1, O: -2), without any additional text, explanation, or commentary. Based on the different views of a material: front view <image>, side view <image> and top view <image>, determine the oxidation state of each element.

Answer: B: +3, P: -3

#### A.5.5 Stability Estimation (M006)

We introduce the Stability Estimation subtask, which aims to probe a model’s capacity for extracting and reasoning over thermodynamic phase diagrams. Given a compositional phase diagram and multiple-choice options, the model must identify the composition ratio corresponding to the most thermodynamically stable phase, formulating its response as a single letter without supplementary text. This formulation encourages precise reading comprehension and domain-specific reasoning by constraining output format. The task is critical in materials science applications, where correct phase identification underpins the rational design of compounds. Our setup rigorously tests both image interpretation and scientific decision-making abilities.

Stability Estimation (M006) Images:

[Figure 303]

#### Question:

Given a phase diagram with multiple-choice answer options, select the single letter corresponding to the composition ratio at which the material exhibits its most thermodynamically

stable phase and reply only with that letter (e.g., C), without any additional text, explanation, or commentary. Based on the given phase diagram <image>, at what composition ratio does the material exhibit its most thermodynamically stable phase.

#### Options:

- (A) B (P fraction = 0.00)
- (B) B6P (P fraction ≈ 0.14)
- (C) B3P (P fraction = 0.25)
- (D) BP (P fraction = 0.50)
- (E) P (P fraction = 1.00) Answer: D

#### A.5.6 Energy Band and DOS Interpretation (M008)

This subtask focuses on interpreting electronic band structures and density of states (DOS) plots to deduce critical material properties. Given graphical data for specific materials, participants are required to ascertain whether the system exhibits metallic or semiconducting behavior, based on the presence or absence of a band gap at the Fermi level. Further, they must extract additional insights, such as identifying atomic contributions to states near the Fermi level or noting significant features in the band topology. The task demands precise visual analysis and concise summarization, resembling the interpretive rigor employed in modern condensed matter research.

Energy Band and DOS Interpretation (M008) Images:

[Figure 304]

Question: Based on the given band structure and DOS <image>, is the material metallic or semiconducting? What could be concluded more from the graph?

Answer: The material is metallic (bands cross the Fermi level). Electronic states near the Fermi level dominated by Ag atoms.

#### A.5.7 Band Gap Classification (M010)

We formulate the Band Gap Classification subtask to determine the electronic nature of a material from its computed band structure. Given a plot of energy versus k-point, the objective is to analyze the positions of the valence band maximum (VBM) and conduction band minimum (CBM). The core decision is whether the bandgap is direct, with VBM and CBM at the same high-symmetry

k-point, indirect, with these features at different k-points, or absent, indicating a metallic state. This classification provides insight into the fundamental optoelectronic properties of the material, facilitating materials discovery and characterization.

Band Gap Classification (M010) Images:

[Figure 305]

Question: Given an electronic band structure plot, analyze the positions of the valence band maximum (VBM) and conduction band minimum (CBM) to decide if the material has a direct bandgap (VBM and CBM at the same high-symmetry k-point, which you must specify), an indirect bandgap (VBM and CBM at different high-symmetry k-points, which you must specify both), or if band crossings at the Fermi level indicate a metallic state with no bandgap. Reply with a single concise narrative paragraph stating your conclusion, including the relevant high-symmetry k-points for direct or indirect cases, or for a metallic case briefly explaining that band crossings preclude any bandgap, without bullet points, lists, or additional commentary. Based on the provided band structure <image>, determine whether the material exhibits a direct or indirect bandgap.

Answer: VBM and CBM are located at same k-points, the material exhibits direct bandgap

#### A.5.8 Valence State and Electronic Orbital Analysis (M018)

This subtask focuses on rigorous interpretation of XPS spectra, requiring the analyst to accurately resolve and assign elemental signatures, oxidation states, and electronic environments present in complex material samples. The task demands a close examination of core-level binding energies, consideration of possible satellite peaks, and differentiation between multiple chemical states or phases when coexisting species are detected. By systematically associating observed peaks with established reference values, participants are expected to clearly articulate the composition and chemical identity of the sample, ensuring precise and unambiguous material characterization consistent with high standards of scientific reporting.

Valence State and Electronic Orbital Analysis (M018)

[Figure 306]

Question: Given the XPS spectrum of the sample, analyze the spectral features to determine which elements and compounds are present and assign their core-level peaks (including any satellite or oxidation-state-specific signatures), and reply with a single descriptive paragraph that mirrors the example style—no bullet points, lists, or additional commentary. Based on the provided XPS spectra <image>, please specify the element species and identify the characteristic peaks. If multiple materials are present, analyze and assign the peaks for each materials.

#### Answer:

The depicted XPS spectrum showed Ir 4f7/2 and 4f5/2 peaks, with a binding energy of 60.59 eV and 63.59 eV respectively.

#### A.5.9 Phase Identification (M020)

In this subtask, we construct a phase identification scenario that simulates a real-world X-ray diffraction (XRD) analysis workflow. The participant is presented with the XRD pattern of a composite material alongside the reference XRD patterns for a set of candidate phases. The objective is to determine, through pattern comparison, the exact combination of phases present in the mixture. This task not only assesses the ability to recognize subtle similarities in peak positions and intensities but also emphasizes robust reasoning under ambiguity as occurs in practical material characterization, which is critical for developing automated, generalizable scientific AI models.

Phase Identification (M020)

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Question: Given a combined XRD pattern and several materials’ individual XRD patterns, analyze the set of diffraction peaks to determine which crystalline phases are present and provide a concise narrative explanation of your reasoning in a single paragraph, avoiding bullet points, lists, or any additional commentary. Based on the XRD pattern of the composite material made from a mixture of three materials <image> and five XRD patterns of Ag2O <image>, SnO2 <image>, BaTe <image>, BAs <image> and Ac2CuIr <image> (orderly), identify all three phases/materials present in the composite material.

Answer: This graph contains three distinct materials: Ag2O, BaTe, and Ac2CuIr.

#### A.5.10 Lattice Constant Estimation (M021)

In this subtask, we focus on the extraction of fundamental crystallographic information from experimental X-ray diffraction (XRD) patterns. The objective is to interpret a given XRD spectrum by first identifying the material’s crystalline phase and then systematically assigning Miller indices (hkl) to the primary diffraction peaks observed. Building on the assigned peaks, precise lattice parameters are estimated through established diffraction geometry. By requiring concise, integrated responses, this subtask evaluates the ability to connect peak analysis with physical structure, encouraging rigorous yet efficient crystallographic reasoning akin to practices in real-world materials research.

Lattice Constant Estimation (M021) Images:

[Figure 313]

Question: Given an XRD pattern of a single-phase material, identify the crystalline phase, assign the obvious primary diffraction peaks to their Miller indices (hkl), calculate the lattice parameters from those peak positions, and reply with a single concise narrative paragraph integrating these findings and calculations, without bullet points, lists, or additional commentary. Based on the provided XRD pattern <image>, please identify the crystalline phase(s) present, assign Miller indices (hkl) to the distinct primary diffraction peaks (note that, you only need to specify the obvious peaks), and use those peak positions to estimate the lattice parameters.

#### Answer:

The main diffraction peaks at 2θ = 26.5°, 33.7°, 38.8°, 51.5°, 54.5°, 62.2°, 65.6°, 70.8°, 78.2° and 89.2° correspond exactly to the (110), (101), (111), (211), (220), (221), (301), (202), (321) and (312) reflections of SnO2 (Space group P42/mnm). No impurity phase is found in the XRD pattern. The calculated lattice parameters are: a = b = 4.76Å, c = 3.21Å, α = β = γ = 90°.

#### A.5.11 Crystal Grain Size Estimation (M022)

This subtask focuses on leveraging X-ray diffraction (XRD) patterns to quantitatively estimate the grain size of polycrystalline materials. By isolating the most intense diffraction peak and rigorously applying the Scherrer equation, the objective is to derive a reliable measurement of average crystallite dimensions. The setup emphasizes precision in extracting critical parameters such as peak position and full width at half maximum (FWHM), thus ensuring consistency and reproducibility. This procedure is essential in characterizing the microstructural evolution of materials, directly correlating XRD features with underlying nanoscale attributes, and ultimately bridging experimental observables with materials science fundamentals.

Crystal Grain Size Estimation (M022) Images:

[Figure 314]

Question: Given an XRD pattern of a material with a clearly defined most intense peak, apply the Scherrer equation to that peak to calculate the average crystallite size and reply with a single concise narrative paragraph stating the result and key parameters used (e.g., peak position, peak intensity), without bullet points, lists, or additional commentary. Based on the provided XRD pattern <image>, please estimate the average crystallite size of the sample by applying the Scherrer equation to the most intense XRD peak.

Answer: Applying the Scherrer formula to the most intense (110) peak, the calculated average crystallite size is 0.1nm

### B More Experimental Results

Please refer to Table 7, Table 8, Table 9, Table 10, Table 11, Table 12, Table 13, Table 14, Table 15, and Table 16 for additional experimental results.

### C Larger Maximal Number of Generated Tokens

In the main paper, we set the maximal number of generated tokens to 1024 for all models to ensure fairness and cost-effectiveness. This results in poor performance for some reasoning MLLMs like the Gemini-2.5-Pro, which rapidly consumes all 1024 tokens for reasoning. To address this, we conducted an additional experiment with a larger maximal number of generated tokens (16384) for Gemini-2.0-Flash, Gemini-2.5-Flash, and Gemini-2.5-Pro in Table 17 and Table 18.

Model A001 A003 A006 A007 A008 A009 A010 A011

- Claude-3-Opus 39.00 20.00 20.50 24.00 6.50 0.50 5.50 2.00 Claude-3.7-Sonnet 70.50 26.11 28.00 9.00 18.50 1.00 0.00 54.00 Doubao-1.5-vision-pro 58.50 27.22 18.50 18.50 27.00 11.50 2.50 63.00 Gemini-2.0-Flash 57.00 41.11 12.50 0.00 9.00 3.00 0.00 9.00

- Gemini-2.5-Flash 69.50 35.56 20.00 24.50 28.00 7.50 4.50 6.00 Gemini-2.5-Pro 39.50 0.00 0.00 0.00 0.00 0.00 0.00 1.00 GPT-4o-2024-11-20 73.00 26.11 21.50 24.50 14.00 4.50 0.00 0.00

- GPT-4.1 81.00 30.00 23.00 22.50 18.50 14.50 0.00 4.50 GPT-o1 67.50 35.00 6.00 13.50 8.50 3.00 3.50 40.50 GPT-o3 72.00 45.00 24.00 13.50 7.50 16.00 0.00 18.00 Grok-2-Vision-12-12 64.50 22.22 5.50 24.50 24.00 3.50 0.00 11.00 InternVL-3-78B 48.00 27.22 29.00 24.50 19.50 9.00 10.00 49.50 InternVL-2.5-78B 41.50 16.11 15.50 17.00 20.50 9.00 2.50 4.00

- Llama-3.2-Vision-90B 61.00 42.22 26.00 25.50 2.50 10.00 0.00 0.00 Llava-OneVision-72B 54.00 20.00 0.00 17.00 7.00 15.00 11.50 76.50 QwenVL-72B 63.00 20.00 49.00 26.50 24.50 10.00 0.00 18.00

- Table 7: Performance of different models on the Astronomy tasks (English).

Model A001 A003 A006 A007 A008 A009 A010 A011 Claude-3-Opus 40.00 32.22 10.50 16.50 24.00 0.50 2.00 4.50 Claude-3.7-Sonnet 72.00 30.00 12.00 17.00 17.50 1.00 0.00 30.00 Doubao-1.5-vision-pro 54.00 21.11 7.00 15.50 19.50 7.50 4.00 63.00 Gemini-2.0-Flash 60.50 34.44 1.00 0.00 1.00 0.00 0.00 7.50 Gemini-2.5-Flash 70.00 33.33 22.00 23.00 29.50 5.50 3.50 7.00 Gemini-2.5-Pro 42.50 5.00 0.00 0.00 0.00 0.00 0.00 1.00 GPT-4o-2024-11-20 73.00 25.00 17.00 16.50 10.50 3.00 0.00 0.00 GPT-4.1 81.00 31.11 25.00 21.00 12.50 16.00 0.00 18.00 GPT-o1 67.50 35.00 6.50 16.50 3.00 4.00 3.00 40.50 GPT-o3 76.50 40.00 21.50 16.00 11.00 4.50 0.00 22.50 Grok-2-Vision-12-12 60.00 21.67 10.50 21.00 23.50 5.00 0.00 4.50 InternVL-3-78B 59.50 41.11 13.50 23.00 13.00 3.00 5.00 40.50 InternVL-2.5-78B 49.00 37.22 11.50 14.50 21.50 5.00 0.00 0.00 Llama-3.2-Vision-90B 62.50 22.22 12.00 15.00 16.50 6.50 0.00 10.00 Llava-OneVision-72B 49.50 10.00 0.00 14.00 25.00 8.50 5.50 72.00 QwenVL-72B 58.50 25.00 39.50 25.00 9.50 5.50 0.00 4.50

- Table 8: Performance of different models on the Astronomy tasks (Chinese).

As it can be seen, with increasing the maximal number of generated tokens, the performance of Gemini-2.5-Flash improves marginally, while Gemini-2.5-Pro shows a significant performance boost across all tasks, especially in the Earth and Life tasks. This indicates that Gemini-2.5-Pro is capable of solving more complex tasks with sufficient reasoning steps, but it requires a larger number of tokens to do so.

### D Additional Evaluation Methods

For better reliability, we conduct two additional evaluations as follows:

Human evaluation. First, we randomly sample 10 questions from SFE (due to limited time) and collect the corresponding responses of six MLLMs (three close weight: GPT-o3, Claude-3.7-Sonnet, Grok-2-Vision-12-12; three open weight: InternVL-3-78B, Qwen2.5-VL-72B, Llava-OneVision72B). Then, these responses are anonymized and provided to domain experts for ranking. The resulting aggregated order is:

Claude-3-Opus 24.67 0.00 2.00 18.67 52.67 17.08 0.00 48.00 8.00 31.33

- Claude-3.7-Sonnet 52.00 18.67 38.67 52.00 20.00 22.08 8.67 86.00 47.33 32.00 Doubao-1.5-vision-pro 43.33 2.67 18.00 32.00 12.67 10.83 2.00 66.00 44.00 26.00 Gemini-2.0-Flash 62.00 15.33 28.67 39.33 14.00 23.33 0.00 81.33 46.67 28.00 Gemini-2.5-Flash 64.67 6.67 19.33 48.00 10.00 8.33 1.33 82.00 64.00 14.67 Gemini-2.5-Pro 1.33 0.00 8.67 0.00 0.00 3.75 0.00 0.00 18.00 0.00 GPT-4o-2024-11-20 48.00 0.00 23.33 36.67 0.00 21.25 0.00 86.00 29.33 20.67 GPT-4.1 50.00 1.33 16.67 42.00 7.33 4.17 6.00 86.00 38.00 30.67 GPT-o1 66.67 4.00 30.00 48.67 0.00 13.33 6.00 78.00 40.67 40.00 GPT-o3 56.67 8.00 32.67 42.00 8.67 19.58 3.33 66.00 38.00 30.67 Grok-2-Vision-12-12 36.67 1.33 27.33 30.00 5.33 16.67 0.00 68.67 12.00 23.33 InternVL-3-78B 34.67 0.00 12.00 17.33 16.00 5.42 0.00 79.33 18.67 26.00 InternVL-2.5-78B 34.00 0.00 16.67 23.33 0.00 4.17 0.00 71.33 13.33 26.67 Llama-3.2-Vision-90B 39.33 0.00 16.00 20.67 4.67 0.00 0.00 69.33 3.33 12.00 Llava-OneVision-72B 33.33 0.00 16.67 20.00 2.67 2.50 0.00 66.00 14.00 8.67 QwenVL-72B 18.67 4.00 22.67 25.33 2.00 10.83 0.00 48.00 34.67 18.00 Model C018 C023 C024 C025 C026 C027 C028 C029 C030

Claude-3-Opus 26.00 17.33 18.00 31.33 16.00 33.33 0.00 6.00 24.00 Claude-3.7-Sonnet 2.00 18.00 35.33 17.33 19.33 29.33 0.00 0.00 32.67 Doubao-1.5-vision-pro 32.67 19.33 31.33 51.33 21.33 22.67 11.33 8.67 27.33 Gemini-2.0-Flash 22.00 18.67 18.67 48.67 4.67 24.00 4.67 1.33 50.00 Gemini-2.5-Flash 0.00 14.67 37.33 49.33 13.33 14.67 8.00 2.67 0.00 Gemini-2.5-Pro 0.00 0.00 0.00 0.00 0.00 5.33 0.00 1.33 0.00 GPT-4o-2024-11-20 12.00 15.33 28.67 42.67 8.00 24.00 3.33 7.33 4.00 GPT-4.1 26.00 20.67 30.00 48.67 23.33 24.67 2.67 1.33 8.67 GPT-o1 31.33 19.33 32.67 55.33 18.00 20.00 6.00 8.00 11.33 GPT-o3 62.00 16.67 31.33 52.67 14.00 26.00 8.67 7.33 30.67 Grok-2-Vision-12-12 26.00 19.33 22.00 48.67 14.00 23.33 2.67 4.67 25.33 InternVL-3-78B 24.00 19.33 22.00 47.33 5.33 21.33 0.67 15.33 20.00 InternVL-2.5-78B 12.67 19.33 37.33 51.33 4.00 20.67 0.67 0.00 25.33 Llama-3.2-Vision-90B 27.33 13.33 22.67 40.67 20.00 21.33 0.00 1.33 20.00 Llava-OneVision-72B 18.00 18.67 32.00 15.33 10.67 17.33 0.00 3.33 20.00 QwenVL-72B 20.67 19.33 24.00 56.00 3.33 25.33 2.00 5.33 12.67

Table 9: Performance of different models on the Chemistry tasks (English).

GPT-o3 ≈ Claude-3.7-Sonnet ≈ InternVL-3-78B > Grok-2-Vision-12-12 > Llava-OneVision-72B > Qwen2.5-VL-72B

Multiple LLM judges. We re-scored all responses of the above mentioned 6 MLLMs with 3 additional judges: Claude-3.7-Sonnet, Deepseek-V3, and Grok-2. The results are demonstrated in Table 19.

As it can be seen, all judges produce similar ranking compared to human evaluation. These results show that our conclusions are robust to assessments of both human and multiple LLM judges.

Claude-3-Opus 15.33 0.00 20.67 16.67 38.67 0.00 1.33 45.33 4.00 28.00 Claude-3.7-Sonnet 57.33 16.00 27.33 53.33 14.67 13.75 8.00 86.00 40.67 28.67 Doubao-1.5-vision-pro 51.33 1.33 19.33 32.00 19.33 6.67 0.00 66.00 38.00 28.67 Gemini-2.0-Flash 42.00 10.67 15.33 30.00 26.67 10.83 0.00 76.00 51.33 32.67

- Gemini-2.5-Flash 65.33 6.67 20.00 47.33 14.00 8.33 4.00 82.00 64.00 12.67 Gemini-2.5-Pro 0.00 0.00 10.67 0.00 0.00 3.75 0.00 6.00 18.00 2.00 GPT-4o-2024-11-20 46.00 0.00 26.67 33.33 0.00 12.08 2.00 85.33 22.67 14.00 GPT-4.1 46.67 0.00 23.33 36.67 10.67 2.50 1.33 86.00 24.67 32.00 GPT-o1 62.00 3.33 33.33 50.67 6.00 18.33 0.00 66.00 38.67 46.00 GPT-o3 54.67 10.67 32.00 35.33 4.67 10.42 2.67 60.00 36.00 38.67 Grok-2-Vision-12-12 26.67 2.00 20.00 36.00 6.67 2.92 0.00 80.67 10.00 24.67 InternVL-3-78B 24.67 0.00 17.33 12.00 5.33 1.67 0.00 74.00 16.67 23.33 InternVL-2.5-78B 26.00 0.00 12.67 12.00 0.00 0.83 0.00 56.67 10.67 10.67 Llama-3.2-Vision-90B 22.67 0.00 10.67 18.00 5.33 0.00 0.00 64.67 0.00 8.67

- Llava-OneVision-72B 19.33 0.00 16.00 18.00 0.00 1.67 0.00 78.00 10.00 8.67 QwenVL-72B 10.67 3.33 22.00 26.67 2.67 0.83 0.00 30.00 26.67 26.67 Model C018 C023 C024 C025 C026 C027 C028 C029 C030

- Claude-3-Opus 23.33 15.33 17.33 32.67 18.00 29.33 0.00 8.67 27.33 Claude-3.7-Sonnet 2.67 16.67 25.33 23.33 8.00 28.00 0.00 4.67 30.00 Doubao-1.5-vision-pro 26.00 18.00 25.33 50.00 19.33 24.00 14.67 6.67 28.67 Gemini-2.0-Flash 12.00 20.00 15.33 46.00 8.67 23.33 6.67 1.33 48.67 Gemini-2.5-Flash 0.00 14.00 36.67 46.67 12.00 11.33 7.33 2.67 0.00 Gemini-2.5-Pro 0.00 0.00 0.00 0.00 0.00 2.00 0.00 0.00 0.00 GPT-4o-2024-11-20 19.33 19.33 23.33 38.00 14.67 9.33 1.33 0.00 2.67 GPT-4.1 28.00 18.00 26.67 32.00 15.33 24.00 2.67 3.33 18.00 GPT-o1 34.00 14.67 26.00 54.67 18.67 24.00 5.33 4.67 16.00 GPT-o3 52.00 17.33 22.00 54.00 27.33 25.33 8.00 9.33 40.00 Grok-2-Vision-12-12 18.67 19.33 18.67 48.00 16.67 23.33 8.67 4.00 27.33 InternVL-3-78B 15.33 18.67 20.00 44.00 6.00 5.33 0.00 15.33 18.67 InternVL-2.5-78B 16.67 18.67 17.33 48.00 6.00 24.67 0.00 10.67 33.33 Llama-3.2-Vision-90B 21.33 19.33 22.67 37.33 16.67 25.33 0.00 4.00 20.00 Llava-OneVision-72B 16.00 20.00 20.00 39.33 4.67 22.67 0.00 2.00 20.00 QwenVL-72B 19.33 20.00 17.33 48.67 6.67 19.33 0.00 2.67 15.33

- Table 10: Performance of different models on the Chemistry tasks (English).

Model E001 E004 E005 E006 E007 E008 E014 E016 E017 E018 E021 Claude-3-Opus 24.00 16.67 55.00 42.50 18.00 90.00 31.11 32.00 41.00 28.18 35.00

- Claude-3.7-Sonnet 23.33 23.33 57.50 50.83 23.00 16.00 30.00 42.00 72.00 52.73 36.50 Doubao-1.5-vision-pro 24.00 20.00 29.17 29.17 2.00 38.00 68.89 14.00 60.00 23.64 15.75

- Gemini-2.0-Flash 24.00 25.83 30.83 55.00 20.00 46.00 14.44 26.00 67.00 28.18 36.00 Gemini-2.5-Flash 56.67 18.33 27.50 27.50 10.00 28.00 14.44 33.00 48.00 23.64 38.00 Gemini-2.5-Pro 0.00 0.00 9.17 1.67 0.00 0.00 0.00 0.00 0.00 11.82 3.00 GPT-4o-2024-11-20 18.00 20.00 28.33 50.83 28.00 42.00 28.89 50.00 65.00 30.00 25.75 GPT-4.1 30.00 26.67 63.33 35.83 25.00 47.00 72.22 60.00 69.00 51.82 22.75 GPT-o1 18.00 34.17 54.17 45.83 22.00 63.00 36.67 53.00 44.00 35.45 35.25 GPT-o3 36.00 30.00 41.67 40.83 13.00 54.00 76.67 67.00 74.00 44.55 33.75 Grok-2-Vision-12-12 46.00 13.33 51.67 31.67 17.00 21.00 23.33 57.00 72.00 46.36 20.75 InternVL-3-78B 0.00 18.33 56.67 41.67 18.00 46.00 48.89 15.00 23.00 50.00 23.50 InternVL-2.5-78B 0.00 14.17 43.33 45.00 20.00 44.00 62.22 64.00 63.00 30.00 30.50 Llama-3.2-Vision-90B 28.00 21.67 33.33 21.67 22.00 15.00 23.33 26.00 69.00 28.18 23.25 Llava-OneVision-72B 24.00 12.50 40.83 11.67 6.00 27.00 40.00 15.00 60.00 36.36 13.50

- QwenVL-72B 12.00 15.83 25.00 50.83 3.00 63.00 18.89 20.00 63.00 19.09 15.75

- Table 11: Performance of different models on the Earth tasks (English).

Model E001 E004 E005 E006 E007 E008 E014 E016 E017 E018 E021 Claude-3-Opus 24.00 26.67 38.33 32.50 7.00 56.00 42.22 19.00 38.00 30.00 35.25

- Claude-3.7-Sonnet 23.33 23.33 58.33 32.50 19.00 15.00 62.22 30.00 75.00 43.64 36.25 Doubao-1.5-vision-pro 24.00 15.00 28.33 35.00 3.00 63.00 70.00 24.00 55.00 14.55 9.00

- Gemini-2.0-Flash 25.33 24.17 27.50 53.33 18.00 27.00 41.11 24.00 48.00 20.00 39.25

- Gemini-2.5-Flash 56.67 16.67 27.50 26.67 9.00 31.00 12.22 36.00 56.00 25.45 30.00 Gemini-2.5-Pro 0.00 3.33 14.17 1.67 0.00 0.00 5.56 0.00 0.00 19.09 2.25

- GPT-4o-2024-11-20 12.00 25.00 31.67 39.17 15.00 49.00 70.00 27.00 55.00 29.09 19.50 GPT-4.1 24.00 27.50 60.00 50.00 7.00 74.00 72.22 30.00 68.00 54.55 41.50 GPT-o1 30.00 32.50 54.17 40.83 3.00 63.00 70.00 48.00 56.00 33.64 21.00 GPT-o3 18.00 40.00 35.83 36.67 8.00 54.00 80.00 24.00 74.00 33.64 29.25 Grok-2-Vision-12-12 34.67 22.50 58.33 36.67 19.00 43.00 48.89 17.00 69.00 36.36 30.50 InternVL-3-78B 0.00 17.50 60.83 15.00 0.00 72.00 68.89 15.00 29.00 43.64 24.75 InternVL-2.5-78B 18.00 19.17 59.17 38.33 10.00 68.00 67.78 19.00 59.00 41.82 33.00 Llama-3.2-Vision-90B 6.00 21.67 45.83 44.17 20.00 60.00 56.67 26.00 38.00 27.27 23.25 Llava-OneVision-72B 24.00 19.17 46.67 9.17 8.00 63.00 61.11 15.00 57.00 30.00 15.75 QwenVL-72B 0.00 17.50 30.00 62.50 0.00 54.00 22.22 15.00 45.00 21.82 20.25

- Table 12: Performance of different models on the Earth tasks (Chinese).

Model L001 L003 L005 L006 L007 L008 L009 Claude-3-Opus 11.0 0.0 9.0 0.0 18.0 67.5 37.5 Claude-3.7-Sonnet 63.0 0.0 0.0 3.0 40.0 75.0 22.5 Doubao-1.5-vision-pro 62.0 0.0 54.0 11.0 18.0 82.5 45.0 Gemini-2.0-Flash 61.0 0.0 9.0 1.0 29.0 82.5 60.0 Gemini-2.5-Flash 53.0 0.0 9.0 6.0 0.0 75.0 45.0 Gemini-2.5-Pro 54.0 0.0 0.0 2.0 0.0 84.17 30.0 GPT-4o-2024-11-20 53.0 1.0 9.0 6.0 27.0 67.5 30.0 GPT-4.1 71.0 0.0 18.0 4.0 38.0 82.5 37.5 GPT-o1 76.0 0.0 9.0 4.0 27.0 82.5 30.0 GPT-o3 74.0 0.0 18.0 13.0 18.0 82.5 37.5 Grok-2-Vision-12-12 4.0 0.0 27.0 4.0 22.0 76.67 30.0 InternVL-3-78B 37.0 0.0 38.0 5.0 36.0 75.0 37.5 InternVL-2.5-78B 23.0 2.0 27.0 0.0 18.0 75.0 45.0 Llama-3.2-Vision-90B 6.0 0.0 36.0 9.0 0.0 82.5 45.0 Llava-OneVision-72B 24.0 0.0 27.0 9.0 18.0 75.0 45.0 QwenVL-72B 51.0 0.0 9.0 2.0 36.0 77.5 45.0 Model L010 L014 L015 L016 L017 L019 L020 Claude-3-Opus 0.0 42.0 27.0 18.0 1.33 56.67 24.0 Claude-3.7-Sonnet 0.0 66.0 27.0 27.0 10.0 77.5 27.0 Doubao-1.5-vision-pro 0.0 50.0 9.0 18.0 0.0 56.67 17.0 Gemini-2.0-Flash 0.0 63.0 27.0 45.0 0.0 55.0 25.0 Gemini-2.5-Flash 0.0 53.0 30.0 45.0 0.0 22.5 16.0 Gemini-2.5-Pro 0.0 4.0 9.0 18.0 0.0 62.5 3.0 GPT-4o-2024-11-20 0.0 64.0 36.0 45.0 1.33 70.0 21.0 GPT-4.1 0.0 70.0 27.0 44.0 1.33 78.33 22.0 GPT-o1 0.0 72.0 27.0 44.0 9.33 72.5 25.0 GPT-o3 1.67 74.0 18.0 36.0 4.0 69.17 28.0 Grok-2-Vision-12-12 0.0 50.0 21.0 27.0 1.33 71.67 13.0 InternVL-3-78B 0.0 56.0 21.0 36.0 1.33 74.17 17.0 InternVL-2.5-78B 0.0 60.0 21.0 18.0 0.0 76.67 9.0 Llama-3.2-Vision-90B 0.0 48.0 21.0 18.0 0.0 64.17 19.0 Llava-OneVision-72B 0.0 18.0 18.0 18.0 1.33 72.5 11.0 QwenVL-72B 0.0 64.0 15.0 18.0 0.0 25.0 15.0

- Table 13: Performance of different models on the Life tasks (English).

Model L001 L003 L005 L006 L007 L008 L009 Claude-3-Opus 29.0 0.0 9.0 0.0 18.0 60.0 30.0 Claude-3.7-Sonnet 57.0 0.0 0.0 2.0 36.0 77.5 24.2 Doubao-1.5-vision-pro 54.0 0.0 45.0 2.0 18.0 82.5 37.5 Gemini-2.0-Flash 51.0 2.0 0.0 0.0 27.0 90.0 60.0

- Gemini-2.5-Flash 57.0 0.0 9.0 4.0 0.0 76.7 45.0 Gemini-2.5-Pro 77.0 0.0 0.0 5.0 0.0 82.5 45.0 GPT-4o-2024-11-20 45.0 0.0 0.0 2.0 36.0 60.0 22.5 GPT-4.1 68.0 0.0 16.0 10.0 41.0 85.0 30.0 GPT-o1 60.0 0.0 0.0 5.0 36.0 82.5 45.0 GPT-o3 60.0 0.0 18.0 6.0 9.0 90.0 45.0 Grok-2-Vision-12-12 0.0 0.0 36.0 19.0 17.0 75.0 30.0 InternVL-3-78B 22.0 0.0 9.0 0.0 27.0 75.0 30.0 InternVL-2.5-78B 7.0 0.0 27.0 24.0 9.0 62.5 37.5 Llama-3.2-Vision-90B 6.0 0.0 36.0 4.0 15.0 77.5 22.5 Llava-OneVision-72B 24.0 0.0 45.0 7.0 18.0 75.0 37.5 QwenVL-72B 26.0 0.0 9.0 0.0 27.0 77.5 45.0 Model L010 L014 L015 L016 L017 L019 L020

Claude-3-Opus 0.0 51.0 30.0 18.0 2.0 63.33 20.0 Claude-3.7-Sonnet 0.0 62.0 9.0 36.0 9.33 75.83 22.0 Doubao-1.5-vision-pro 0.0 63.0 18.0 18.0 1.33 58.33 15.0 Gemini-2.0-Flash 0.0 53.0 27.0 27.0 2.67 23.33 19.0 Gemini-2.5-Flash 0.0 50.0 30.0 45.0 0.0 22.5 16.0 Gemini-2.5-Pro 0.0 8.0 0.0 24.0 0.0 61.67 1.0 GPT-4o-2024-11-20 0.0 67.0 24.0 54.0 2.0 61.67 18.0 GPT-4.1 0.0 76.0 27.0 36.0 1.33 67.5 20.0 GPT-o1 0.0 67.0 12.0 45.0 6.0 57.5 23.0 GPT-o3 0.0 75.0 9.0 39.0 4.0 60.0 27.0 Grok-2-Vision-12-12 0.0 59.0 21.0 27.0 0.0 65.0 14.0 InternVL-3-78B 0.0 59.0 24.0 45.0 3.33 69.17 22.0 InternVL-2.5-78B 0.0 53.0 14.0 45.0 0.0 63.33 15.0 Llama-3.2-Vision-90B 0.0 40.0 11.0 18.0 3.33 56.67 4.0 Llava-OneVision-72B 2.5 51.0 36.0 21.0 0.0 55.83 11.0 QwenVL-72B 0.0 43.0 12.0 36.0 0.0 23.33 20.0

- Table 14: Performance of different models on the Life tasks (Chinese).

Model M001 M002 M003 M004 M006 M008 M010 M018 M020 M021 M022 Claude-3-Opus 76.67 6.25 28.75 53.75 60.00 42.50 42.50 27.50 30.00 15.00 25.00 Claude-3.7-Sonnet 53.33 30.00 55.00 78.75 80.00 40.00 40.00 60.00 28.00 17.50 15.00 Doubao-1.5-vision-pro 60.00 26.25 67.50 90.00 80.00 22.50 45.00 40.00 34.00 20.00 22.50 Gemini-2.0-Flash 53.33 45.00 82.50 82.50 70.00 45.00 22.50 45.00 24.00 22.50 22.50 Gemini-2.5-Flash 66.67 45.00 90.00 85.00 70.00 60.00 17.50 52.50 28.00 22.50 25.00 Gemini-2.5-Pro 0.00 11.25 22.50 78.75 80.00 0.00 25.00 7.50 0.00 0.00 0.00 GPT-4o-2024-11-20 43.33 28.75 71.25 90.00 50.00 22.50 65.00 47.50 36.00 30.00 22.50 GPT-4.1 70.00 26.25 51.25 90.00 70.00 40.00 45.00 52.50 10.00 17.50 15.00 GPT-o1 90.00 65.00 90.00 90.00 80.00 25.00 52.50 67.50 12.00 15.00 20.00 GPT-o3 90.00 56.25 78.75 90.00 80.00 60.00 60.00 75.00 26.00 22.50 20.00 Grok-2-Vision-12-12 50.00 25.00 51.25 71.25 30.00 35.00 17.50 22.50 34.00 15.00 10.00 InternVL-3-78B 43.33 8.75 52.50 71.25 80.00 17.50 15.00 40.00 24.00 30.00 15.00 InternVL-2.5-78B 46.67 3.75 48.75 71.25 90.00 30.00 15.00 47.50 24.00 30.00 20.00 Llama-3.2-Vision-90B 53.33 15.00 50.00 75.00 30.00 15.00 5.00 30.00 18.00 27.50 5.00 Llava-OneVision-72B 10.00 7.50 58.75 71.25 70.00 30.00 5.00 40.00 18.00 30.00 5.00 QwenVL-72B 50.00 8.75 63.75 78.75 80.00 17.50 5.00 50.00 12.00 25.00 0.00

- Table 15: Performance of different models on the Materials tasks (English).

Model M001 M002 M003 M004 M006 M008 M010 M018 M020 M021 M022 Claude-3-Opus 56.67 7.50 37.50 62.50 30.00 27.50 20.00 37.50 24.00 22.50 27.50 Claude-3.7-Sonnet 20.00 21.25 43.75 78.75 80.00 40.00 42.50 55.00 40.00 20.00 22.50 Doubao-1.5-vision-pro 16.67 28.75 67.50 90.00 80.00 20.00 15.00 35.00 18.00 22.50 27.50 Gemini-2.0-Flash 60.00 47.50 82.50 78.75 80.00 25.00 0.00 32.50 28.00 25.00 10.00 Gemini-2.5-Flash 66.67 45.00 90.00 85.00 70.00 60.00 17.50 52.50 24.00 22.50 22.50 Gemini-2.5-Pro 0.00 2.50 11.25 78.75 70.00 15.00 65.00 0.00 0.00 0.00 0.00

- GPT-4o-2024-11-20 13.33 28.75 71.25 90.00 50.00 25.00 45.00 50.00 40.00 35.00 30.00 GPT-4.1 20.00 28.75 60.00 90.00 90.00 30.00 15.00 50.00 24.00 22.50 22.50 GPT-o1 76.67 63.75 90.00 90.00 90.00 47.50 15.00 55.00 26.00 22.50 20.00 GPT-o3 86.67 35.00 78.75 90.00 80.00 47.50 60.00 57.50 28.00 12.50 22.50 Grok-2-Vision-12-12 20.00 26.25 51.25 73.75 40.00 25.00 22.50 25.00 28.00 20.00 27.50 InternVL-3-78B 13.33 17.50 52.50 73.75 80.00 17.50 15.00 42.50 28.00 27.50 30.00 InternVL-2.5-78B 10.00 8.75 41.25 73.75 90.00 17.50 17.50 37.50 24.00 27.50 20.00 Llama-3.2-Vision-90B 10.00 16.25 48.75 73.75 18.89 25.00 45.00 15.00 24.00 25.00 17.50

- Llava-OneVision-72B 20.00 7.50 62.50 73.75 80.00 17.50 40.00 25.00 28.00 30.00 20.00

- QwenVL-72B 13.33 6.25 63.75 78.75 80.00 22.50 15.00 35.00 22.00 32.50 27.50

- Table 16: Performance of different models on the Materials tasks (Chinese).

Astronomy Chemistry Earth Life Material Average en zh en zh en zh en zh en zh en zh

Model

|Gemini-2.0-Flash (1024) Gemini-2.5-Flash (1024) Gemini-2.5-Pro (1024)|16.14 12.78 24.30 24.11 5.13 6.08<br><br>|27.82 24.69 23.67 23.47 2.07 2.28|34.24 32.91 31.99 30.53 2.52 3.84<br><br>|32.48 27.32 25.03 25.10 19.73 22.35<br><br>|52.79 50.49 56.39 55.90 28.69 27.70|29.49 26.33 28.03 27.63 8.04 8.96<br><br>|
|---|---|---|---|---|---|---|
|Gemini-2.0-Flash (16384) Gemini-2.5-Flash (16384) Gemini-2.5-Pro (16384)<br><br>|15.68 19.20 22.22 22.43 27.57 28.13|40.23 28.38 46.51 39.88 54.15 48.81<br><br>|31.15 27.23 36.35 32.05 40.27 39.50|34.64 26.07 44.15 37.04 47.35 38.56<br><br>|55.47 54.28 59.56 63.21 70.49 72.13|33.75 27.87 40.29 36.12 46.15 42.77<br><br>|

- Table 17: Experimental results of MLLMs on different disciplines using different maximal number of generated tokens.

Model

L1 L2 L3

Exact Match Open Question MCQ IoU LLM score Bertscore LLM score LLM score

en zh en zh en zh en zh en zh en zh en zh en zh

|Gemini-2.0-Flash (1024) Gemini-2.5-Flash (1024) Gemini-2.5-Pro (1024)|41.43 36.46 37.04 37.57 17.46 18.78<br><br>|26.86 23.70<br><br>27.89 26.98<br><br><br>6.16 6.94|22.00 21.60 14.96 15.20 1.36 2.24<br><br>|0.91 0.44 21.38 19.28 0.99 0.96 24.45 24.45<br><br>- - 8.11 8.60<br><br>|0.70 0.65 35.67 25.19 0.688 0.688 30.00 30.10 0.548 0.558 12.50 15.96|39.47 37.39 32.71 31.51 6.30 6.94<br><br>|
|---|---|---|---|---|---|---|
|Gemini-2.0-Flash (16384) Gemini-2.5-Flash (16384) Gemini-2.5-Pro (16384)<br><br>|36.53 31.63 42.92 42.62 52.48 52.48<br><br>|29.15 24.29 35.86 30.95 40.58 40.58<br><br>|25.36 17.68 27.12 22.40 27.68 27.68|1.44 0.15 23.29 19.79 3.17 2.68 30.57 29.10<br><br>16.00 10.03 37.33 35.17<br><br>|0.668 0.631 21.58 16.51 0.675 0.672 27.70 28.17 0.704 0.711 37.46 35.87<br><br>|44.98 36.73<br><br>48.68 39.47<br>49.68 44.54<br>|

- Table 18: Experimental results of MLLMs on different disciplines using different maximal number of generated tokens.

GPT-4o Claude-3.7-Sonnet Deepseek-V3 Grok-2 Average en zh en zh en zh en zh en zh

MLLM Judge

Grok-2-Vision-12-12 24.97 25.10 33.82 33.29 22.99 23.46 25.19 23.32 26.74 26.29 GPT-o3 34.08 31.60 45.70 46.25 35.82 35.33 34.05 34.28 37.41 36.86 Claude-3.7-Sonnet 31.52 29.23 35.43 31.42 32.07 25.66 30.53 28.49 32.38 28.7 Qwen2.5-VL-72B 24.17 21.51 23.51 23.08 23.23 22.61 23.11 19.92 23.50 21.78 InternVL-3-78B 26.52 24.30 25.58 23.38 26.40 24.79 26.48 23.94 26.24 24.10 Llava-OneVision-72B 22.10 23.39 22.42 22.56 22.42 22.97 22.81 22.12 22.43 22.76

Table 19: Performance comparison of different MLLMs evaluated by various LLM judges.

