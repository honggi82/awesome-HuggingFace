# arXiv:2603.28068v2[cs.CV]31Mar2026

## AIBench: Evaluating Visual-Logical Consistency in Academic Illustration Generation

Zhaohe Liao1,2∗, Kaixun Jiang3∗, Zhihang Liu1,4∗, Yujie Wei3∗, Junqiu Yu1,3∗, Quanhao Li1,3∗, Hong-Tao Yu1,5∗, Pandeng Li1†, Yuzheng Wang1, Zhen Xing1, Shiwei Zhang1, Chen-Wei Xie1, Yun Zheng1, Xihui Liu6† 1Tongyi Lab, Alibaba Group 2SJTU 3FDU 4USTC 5SEU 6HKU

Project page: https://deep-kaixun.github.io/aibench-page/

#### Abstract

Although image generation has boosted various applications via its rapid evolution, whether the state-of-the-art models are able to produce ready-to-use academic illustrations for papers is still largely unexplored. Directly comparing or evaluating the illustration with VLM is native but requires oracle multi-modal understanding ability, which is unreliable for long and complex texts and illustrations. To address this, we propose AIBench, the first benchmark using VQA for evaluating logic correctness of the academic illustrations and VLMs for assessing aesthetics. In detail, we designed four levels of questions proposed from a logic diagram summarized from the method part of the paper, which query whether the generated illustration aligns with the paper on different scales. Our VQA-based approach raises more accurate and detailed evaluations on visual-logical consistency while relying less on the ability of the judger VLM. With our high-quality AIBench, we conduct extensive experiments and conclude that the performance gap between models on this task is significantly larger than general ones, reflecting their various complex reasoning and high-density generation ability. Further, the logic and aesthetics are hard to optimize simultaneously as in handcrafted illustrations. Additional experiments further state that test-time scaling on both abilities significantly boosts the performance on this task.

#### 1 Introduction

Image generation has experienced significant growth in recent years [38, 13, 33, 4, 21]. Such advances [9, 51, 46, 7, 12, 25, 50, 49, 43, 40, 42] have enabled significantly broader applications [45, 28, 6, 49, 26, 14, 18, 47, 29, 48, 23, 22], easing the workload of graphic design. However, when focusing on academic scenarios, generating an academic illustration (i.e., the framework of a method) still heavily relies on handcrafting. Since this task faces a unique challenge, the ability of state-of-theart (SOTA) models [43, 40, 42, 49, 44] on this task remains largely unexplored. Specifically, as the model needs to take the method part of the paper as input and target generating the framework figure representing its core contribution, it needs to understand the details of the paper and conclude its main contribution, while being able to present such details and contributions in both an aesthetic and logically correct manner. Moreover, judging whether such a generated image aligns with the logic in the paper could be challenging for vision language models (VLMs).

Some works [55, 57] have made an initial attempt to evaluate such ability. However, they hide the aforementioned challenge by directly using VLMs to compare figures as a whole, assuming the oracle ability of VLMs, which could be suspicious when the underlying logic of the paper is non-trivial, and

* Equal contribution, random order. † Corresponding authors.

Preprint. Under review.

[Figure 1]

[Figure 2]

Selected Top-Tier Conference Papers Text-to-Logic Directed Graph

[Figure 3]

Paper Illustration

Text Tokenizer

K-hot Label

Text

Classification loss

Text Encoder

ImageText Data

[Figure 4]

CLIP employs two encoders: a vision encoder for images and a text encoder for text... SuperCLIP adds a lightweight linear classification head on top of the image features extracted by the CLIP vision encoder to produce tokenclassification logits..... Consider an image-text dataset D consisting of image I and caption T, CLIP learns joint image-text embeddings using paired examples... The goal is trained to maximize similarity between matching image-text pairs while minimizing similarity between non-matching pairs in a shared embedding space.

Contrastive loss

Paper Method Text

Topology

Phase Semantics

Vision Encoder

Linear Head

Image

##### AI Bench

Component

[Figure 5]

[Figure 6]

Level 1: Component Existence Level 2: Local Topology

[Figure 7]

Q: Which module directly receives the Vision Encoder output to produce classification logits? A: Linear head B: Not shown C: IDF weighting D: Loss aggregation

Q: Does the diagram depict the 'CLIP vision encoder' component? A: No B: Yes

Aesthetics

[Figure 8]

[Figure 9]

Level 3: Phase Architecture

Level 4: Global Semantics

[Figure 10]

Q: Does the diagram show the image I and caption T as parallel branches? A: No, caption is derived from the image B: Yes, shown as parallel paths C: Not enough information D: No, image starts after text

Q: What is the main objective of this pipeline?

Generated Illustration

- A: Generative image synthesis
- B: Vision–language alignment
- C: Video action recognition
- D: Not enough information

[Figure 11]

[Figure 12]

Aesthetic Score

[Figure 13]

[Figure 14]

Question-Answering-Based Logical Evaluation and Model-Based Aesthetic Assessment

- Figure 1: Overview of AIBench. We introduce a comprehensive benchmark to evaluate academic illustration generation through two primary dimensions: question-answering-based logical evaluation and model-based aesthetic assessment. To systematically assess logical accuracy, we collect top-tier conference papers, construct text-to-logic directed graphs, and manually annotate QA pairs across four hierarchical levels. Ultimately, our AIBench establishes a new evaluation standard for generated academic illustrations.

the generated figures are logically complex. Moreover, the influence of components in the illustration remains unclear, leading to an uninterpretable assessment.

Focusing on these challenges, instead of using VLMs to compare illustrations as a whole, we propose a visual question answering (VQA) style logic assessment and model style aesthetic assessment. Specifically, for logic assessment, we model the assessment as a sequence of VQA tasks, where the questions query the generated illustration at multiple levels to ensure the questions cover all necessary points of the underlying logic of the paper. Specifically, the questions are proposed according to four different levels from low-level components to high-level semantics, namely component-existencelevel, local-topology-level, phase-architecture-level, and global-semantics-level. It is worth noting that the crucial text generation ability is implicitly embedded in our QAs, as answering the questions requires correctly generated characters in the illustration. An example of these questions is provided in Fig. 1. To generate high-quality QAs for these levels, we first acquire Gemini 3 Flash [16] to generate a logic graph from the method text, then propose the questions from the generated logic graph at each corresponding level with proper prompts. A more detailed introduction to the framework is in Sec. 3.2. With this framework, we are able to comprehensively query whether the generated illustration aligns with the original paper. As for aesthetic assessment, such an objective-based evaluation suits VLMs better. Overall, a higher accuracy reflects a higher quality of the generated illustration, and the accuracy at different levels provides a more detailed evaluation. A comparison between our benchmark and existing ones is in Table 1.

Following this philosophy, we propose AIBench, a high-quality benchmark for evaluating the Academic Illustration generation ability of models. It contains 300 open-access papers accepted by top conferences with 5704 corresponding QA pairs for checking the necessary properties that an ideal illustration should have. The QAs are automatically generated via the aforementioned framework and carefully checked by multiple human experts to ensure their high quality. A more detailed dataset distribution is presented in Fig. 3.

We conduct extensive experiments on our benchmark and conclude valuable findings. First, we conclude that the gap between models on the academic illustration generation task is more profound than that on general tasks, indicating that the performance difference between models is still significant when facing long and complex text reasoning for understanding and high-density content in generation.

###### Table 1: Comparison between our AIBench and related benchmarks.

###### Benchmark Data Construction # Papers # Avg. Eval Unit Eval Method Granularity

PaperBanana [55] Auto 292 1 VLM-as-Judge Coarse AutoFigure [57] Auto 3,300 1 VLM-as-Judge Coarse AIBench (Ours) Auto + Human 300 19.01 QA-Based Fine-grained

Moreover, we conclude that aesthetics and logic are somewhat of a trade-off, which also exists in handcrafted illustrations, as more text descriptions and complex layouts may lead to better logic, while breaking aesthetics and vice versa. We further point out that both strong reasoning ability and high-density generation ability are required for this academic illustration task, and test-time scaling strategies on both sides can shatter the capability ceilings of current models. More detailed discussions are in Sec. 4. In conclusion, our contributions can be summarized as follows:

- 1. We propose AIBench, the first VQA-based benchmark for evaluating the academic illustration generation ability of models, providing high-quality QAs for evaluating both logic consistency and aesthetics.
- 2. We design a scalable QA proposing framework that raises high-quality QAs from a summarized logic diagram to evaluate the academic illustration generation ability from low-level content to high-level semantics.
- 3. We conduct experiments on SOTA open- and closed-source unified models and T2I models along with test-time scaling methods on our benchmark to give a full view of current capabilities on academic illustration generation.

#### 2 Related Work

Text-To-Image Generation Benchmarks. Along with rapid progress in text-to-image generation [49, 43, 50, 8, 44, 40, 4], diverse benchmarks have been proposed to evaluate complementary aspects of model capability. GenEval [15] primarily tests compositional prompt following, WISE [32] targets knowledge-intensive and factual consistency, and T2I-CompBench++ [20] provides a broader suite for compositional reasoning with tailored automatic metrics. Beyond general-purpose T2I evaluation, recent benchmarks move toward academic figure generation: PaperBananaBench [55] evaluates methodology diagrams with VLM-as-Judge and reference-based scoring, while FigureBench [57] targets long-context scientific illustration with VLM-based referenced scoring and pairwise comparison. Unlike VLM-judge protocols prone to evaluator instability, our AIBench builds logic-grounded multiple-choice QA pairs from paper content, enabling stable, reproducible assessment of content completeness, logical coherence, communicative clarity, and visual quality.

Automated Scientific Figure Generation. With the progress of generative models, there is growing interest in automating scientific figure creation [37, 19, 52]. Early work mainly follows extract-andlayout pipelines that rearrange existing paper content (e.g., PPSGen [19] focuses on text-only slides). Recent systems move toward agentic, code-based rendering for more controllable outputs, such as PPTAGENT [54] and Paper2Poster [36]; however, they still largely reorganize and stylize source assets rather than generating figures from scratch. Despite these advances, recent work has begun to generate scientific figures from scratch directly from paper content. PaperBanana [55] introduces an end-to-end agentic pipeline for producing academic illustrations; however, it typically conditions on limited inputs (e.g., method excerpts and captions), which can encourage style imitation while missing fine-grained technical details. AutoFigure [57] proposes a reasoned rendering paradigm that converts long-form text into a machine-readable symbolic blueprint, but at the cost of a heavier multi-stage pipeline with increased system complexity and deployment overhead.

#### 3 AIBench

Motivation. Evaluating generated academic illustrations requires a comprehensive consideration of both logical accuracy and aesthetic quality. Existing benchmarks [57, 55] predominantly adopt a VLM-as-Judge paradigm, employing a single VLM to simultaneously score logic and aesthetics. This introduces “metric ambiguity” by conflating objective logical errors with subjective aesthetic flaws,

Text-to-Logic Directed Graph

[Figure 15]

Level ④

Method Section Text

[Figure 16]

[Figure 17]

- Level ①
- Level ② Level ③

High-Quality AIBench

Multi-level Gemini

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

- Level 1: Component Existence
- Level 2: Local Topology
- Level 3: Phase Architecture
- Level 4: Global Semantics

[Figure 25]

[Figure 26]

Extraction Gemini Human Experts

[Figure 27]

Framework Illustration

- Figure 2: QA data construction pipeline of AIBench. We construct the multi-level QA pairs with the help of Gemini, followed by careful human annotation.

resulting in a lack of fine-grained interpretability. To this end, our AIBench explicitly decouples the evaluation process into two distinct dimensions: objective logic and subjective aesthetics. The logical assessment focuses on elements like core components, data flows, and text rendering, which can be objectively cross-checked against the method descriptions from multiple perspectives. Specifically, we formulate logical evaluation into a fine-grained Visual Question Answering (VQA) task using multiple-choice QA pairs. To ensure data quality, we design an automated multi-level QA construction pipeline equipped with human-in-the-loop annotation (Fig. 2). Meanwhile, the subjective aesthetic assessment is independently evaluated via a meticulously selected aesthetic model to reliably simulate human preferences. The following sections detail Data Curation, QA Construction, Data Annotation, Data Analysis, and Evaluation Protocol.

###### 3.1 Data Collection & Curation

We source our benchmark from papers in 2025 across four premier AI conferences (CVPR, ICCV, ICLR, NeurIPS). Using recent, arXiv-available papers mitigates data contamination (these papers are less likely to have been incorporated into the training corpora) and provides LaTeX sources for precise text-figure alignment. Our pipeline extracts methodology text and primary architecture figures in three steps: (i) Text extraction: LaTeX files are parsed to isolate the methodology section, aided by an LLM fallback. (ii) Figure extraction: Candidate figures are ranked via rule-based heuristics on filenames and captions, followed by a VLM-driven pairwise selection for the top-k. (iii) Quality verification: A VLM confirms the paired figure is a valid pipeline diagram. Finally, uniform sampling across conferences yields approximately 3,000 diverse text-figure pairs.

###### 3.2 QA Construction

We adopt a two-stage pipeline: (i) constructing a structured logic-directed graph from the method text, and (ii) generating multi-level QA pairs based on this graph and an academic framework illustration from the original paper.

- Stage 1: Text-to-Logic Directed Graph Construction. Methodology sections in academic papers are often lengthy and weakly structured, making direct QA generation from raw text prone to logical inconsistencies. We therefore introduce a structured intermediate representation. Since AI methods are typically described as modular pipelines or network architectures with explicit data flows, their computation naturally maps to a directed graph, which serves as an effective intermediate form. Motivated by this, we use Gemini 3 Flash to parse the methodology text and construct a logic-directed graph G = (V,E,P), where V denotes key components or data artifacts, E denotes directed data flows, and P denotes architectural phases (e.g., “pre-training”). To ensure faithfulness, we constrain the model to preserve the original terminology from the source paper. This intermediate representation converts unstructured text into an explicit structure, facilitating subsequent QA synthesis.
- Stage 2: Multi-Level QA Generation. Based on node-, edge-, phase-, and graph-level attributes of G, we generate QA pairs at four levels: component, topology, phase, and semantics. We deploy four level-specific QA generators (Gemini 3 Flash), each focusing on the corresponding structural aspects of G. To align the question abstraction with human diagram conventions, we additionally provide the original illustration as a reference image. The four levels are defined as follows:

• Level 1: Component Existence. This level verifies the presence and completeness of key nodes v ∈ V by checking whether core components specified in the directed graph also appear in the reference image (and identifying critical missing elements). Since these queries require recognizing labels and basic glyphs, they also probe text render accuracy.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

Semantics 17%

[Figure 28]

Topology 30%

[Figure 29]

[Figure 30]

16%

Total 5704 QA Level distribution

[Figure 31]

34%

Phase 21%

25%

[Figure 32]

Component 32%

25%

（a） （b） （c） （d）

- Figure 3: Statistics analysis of AIBench. (a) Papers are curated from four representative 2025 conferences (CVPR, ICCV, NeurIPS, ICLR). (b) Models are evaluated on four hierarchical QA levels: Component, Topology, Phase, and Semantics. (c) Word cloud shows lexical frequency and topic diversity of our AIBench. (d) Top research topics in 2025 research papers, covering diffusion, LLMs,

- 3D reconstruction, etc.

- • Level 2: Local Topology. This level examines local connectivity and direct transfers between

adjacent nodes via edges e ∈ E, testing whether outputs from upstream components are correctly routed to the intended downstream modules. As answering such questions depends on tracing lines and arrowheads, it also reflects the visual clarity of local layout organization.

- • Level 3: Phase Architecture. Leveraging phase annotations P, this level targets macro-

architectural organization across phases, including parallel branches, feature aggregation (macro fan-in), branching (macro fan-out), and global feedback loops. Because these patterns rely on cross-module spatial composition, the questions also indicate the coherence of overall layout.

- • Level 4: Global Semantics. By combining the method text with the whole G, this level

formulates questions about the system’s end-to-end design intent and task paradigm. Since such high-level understanding requires integrating evidence across the entire diagram, performance at this level also reflects the figure’s global visual clarity.

In conclusion, by proceeding from the intrinsic logical hierarchy of the figure, we design this pipeline in a bottom-up manner, ensuring it effectively covers all aspects of the verifiable dimensions for evaluating academic illustrations.

###### 3.3 Data Filtering & Annotation

To guarantee high quality and accuracy of our benchmark, we design a filtering pipeline that integrates automated screening with rigorous human inspection [31].

- Step 1: Accuracy-based Screening. We generate QA pairs following the pipeline in Sec. 3.2, then instruct Gemini 3 Flash to answer them based strictly on the original figures. Samples with low overall QA accuracy are discarded as their QA pairs are likely inaccurate.
- Step 2: Hallucination-based Filtering. For incorrectly answered questions, we additionally provide Gemini 3.1 Pro with the original method text. If the model still fails to answer correctly given the ground-truth context, the question is deemed a hallucination artifact and removed. These two automated steps effectively eliminate the vast majority of inaccurate or unanswerable QA pairs.
- Step 3: Human Expert Review. Human experts systematically review all surviving QA pairs for factual correctness, reasoning difficulty, and completeness. Simplistic questions are revised for greater complexity, and new challenging questions are added to ensure comprehensiveness, ultimately yielding a curated dataset of 300 samples with 5704 high-quality QA pairs.

###### 3.4 Data Analysis

Data Source. Our benchmark is constructed from papers published in 2025 at four premier conferences: CVPR (34%), ICCV (25%), NeurIPS (25%), and ICLR (16%). This selection ensures high technical quality through rigorous peer review and maintains topical freshness. By drawing from both CV and ML communities, the dataset provides balanced coverage of vision, generative modeling, and multimodal learning, effectively avoiding low-quality data source.

Topic Distribution. As illustrated in Fig 3(a), the benchmark covers major research directions in year 2025, including diffusion models, language modeling, and multimodal reasoning. While diffusion

Method Text

A “role-playing LLM” that takes a scene (plus prompts/profile-derived inputs) and generates the character’s response, and it is also queried by check_condition for semantic condition judgments......Codified functions append triggered statements from checked condition, and the role-playing LLM turns them into the final response.......We uses NLI as a signal to compare the generated response with the reference and output a label (“contradicted/neutral/entailed”)......The block encodes randomness in executable profile code to make behavior precise, tunable, and reproducible, instead of relying on LLM sampling. …

Nano Banana Pro GPT-Image-1.5 Wan 2.6 Flux 2.0

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

###### Nano Banana Pro

GPT-Image-1.5

###### Wan 2.6

###### Flux 2.0

- Level 1: Component Existence Is the ''conditioned check'' component present?
- Level 2: Local Topology When condition checking, which component uses its output to form the final statements?
- Level 3: Phase Architecture In the NLI Evaluation stage, does the diagram show data being fused?
- Level 4: Global Semantics What is the purpose of the ''Controllable Simulation'' block for character behavior?

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

AI Bench

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Aesthetic Evaluation 😍 60.0

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

😮 58.7

🤔 55.0

🫣 43.2

- Figure 4: The evaluation pipeline of AIBench. Models generate academic illustrations based on method descriptions, and are subsequently evaluated on two primary dimensions: logical accuracy, assessed via paper-specific QA pairs across four hierarchical levels (L1–L4, ranging from component existence to global semantics), and visual appeal, measured by a model-based aesthetic score.

and LLMs represent a substantial portion—reflecting current research trends, other emerging fields are consistently represented to prevent single-paradigm dominance. This diverse composition requires models to demonstrate both generative capability and structured understanding across heterogeneous semantic settings.

QA Hierarchy. To evaluate the consistency between framework diagrams and text, we designed 5,704 QA pairs across four hierarchical levels (Fig 3(d)). The Component (32%) and Topology (30%) levels verify fine-grained nodes and local connectivities. Conversely, the Phase (21%) and Semantic (17%) levels assess macro-architectural organization and global design alignment. This bottom-up structure ensures a precise, multi-granularity assessment.

###### 3.5 Evaluation Protocol

Our evaluation framework is decoupled into two complementary parts. First, we formulate the objective logic assessment as a Visual Question Answering (VQA) task based on our constructed QA pairs, explicitly covering the objective dimensions of academic illustrations. Second, for the subjective aesthetic evaluation, we employ a model-based evaluation to simulate human visual preferences. Figure 4 illustrates the complete evaluation pipeline.

Objective Logic Evaluation. The logic evaluation is formalized as a visual question answering (VQA) task. For each test sample, let V denote the generated image provided by the user (conditioned on the original method text), and Q be the corresponding set of multi-level questions constructed in Section 3.2. Given a multi-modal model solver F, the evaluation process is defined as deriving the predicted answer aˆ = F(V,q) for each question q ∈ Q. Notably, the solver strictly relies on the visual input V without any supplementary textual context. By decomposing the complex global logic into atomic QA pairs, our framework fundamentally reduces the reasoning burden on F. We adopt Qwen3-VL-235B-A22B-Instruct [2] as the default solver. Figure 6(a) shows the robustness of our evaluation across different solvers.

Specifically, logic accuracy is computed independently for each of the four hierarchical levels. Let Ql = Ni=1 Q(li) denote the complete set of level-l questions across all N samples, where Q(li) is the

- Table 2: Comparison of different aesthetic evaluation metrics and human ranking on generated academic illustrations.

Aesthetic Score

Pick Score

VLM-as -Judge

Human Rank

Method

UniPercept AesR1

Original Figure 5.193 19.423 8.92 51.11 62.39 3 Nano Banana Pro [17] 5.181 19.904 8.63 55.04 64.82 2 GPT-Image-1.5 [35] 5.210 19.717 8.87 57.50 64.71 1 Wan 2.6 [43] 5.151 19.866 7.60 51.50 59.09 4 FLUX2-dev [4] 5.150 20.127 5.53 42.40 46.48 5

subset of questions derived from the i-th sample at level l. The accuracy at level l is defined as:

1 |Ql| q∈Q

[aˆ q = a∗q], (1)

Accl =

l

where aˆq and a∗q denote the solver’s predicted option and the ground-truth answer for question q, respectively. Crucially, we adopt question-level global averaging rather than sample-level averaging. This design ensures that each visual reasoning judgment contributes equally to the final metric, naturally yielding more stable statistical estimates for architecturally complex illustrations that inherently generate more questions.

Subjective Aesthetic Evaluation. Beyond logical accuracy, the aesthetic quality of academic illustration is also crucial for an academic paper. Currently, aesthetic evaluation in image generation relies on CLIP-based metrics (e.g., Aesthetic Score [1], PickScore [24]), VLM-as-Judge paradigms (e.g., AutoFig [57]), or specialized aesthetic models (e.g., UniPercept [5], AesR1 [27]). Since no existing metric is tailored for the unique graphical characteristics of academic illustrations, we conduct a systematic preliminary study to identify the most reliable evaluator by using all 300 generated academic illustrations from our AIBench.

The quantitative results in Table 2 reveal that traditional aesthetic metrics struggle with the academic illustration generation task. CLIP-based metrics demonstrate severe misalignment and insensitivity; for instance, PickScore assigns its highest value to the human-ranked worst model, while Aesthetic Score exhibits negligible variance. General-purpose VLMs (e.g., Qwen3-VL-235b-a22b-instruct [2] evaluated in Table 2) struggle to reliably assess aesthetic quality, particularly for academic illustrations, due to a lack of aesthetic knowledge. In contrast, specialized aesthetic models, even without domainspecific fine-tuning on academic figures, are capable of evaluating their aesthetic quality, aligning more closely with human preferences. Based on these observations, we adopt UniPercept as our definitive aesthetic metric, as it offers superior evaluation efficiency. Built upon InternVL3-8B [56], UniPercept evaluates images across complementary perceptual domains, outputting continuous ratings from 0 to 100. In AIBench, we utilize its Image Aesthetics score to evaluate the holistic visual appeal, composition, and harmonic unity of the generated academic illustrations.

Overall Score. To ensure numerical consistency, we scale the logic accuracy (Accl) to a 0–100 range to match the aesthetic metric. Our evaluation comprises four objective logic dimensions plus one subjective aesthetic dimension, and the Overall Score is computed as the arithmetic mean of all five. By combining verifiable QA with human-aligned aesthetic judgment, this dual-track protocol provides a comprehensive, standardized, and interpretable benchmark for academic illustration generation.

#### 4 Experiments

###### 4.1 Experiment Settings

To provide a comprehensive analysis of the current landscape in academic illustration generation, we evaluate a diverse suite of state-of-the-art multi-modal generation models, including Closed-Source Models: Seedream 4.5 [41], Seedream 5.0 [39], Wan 2.6 [43], GPT-Image-1.5 [35], and Nano Banana Pro [17]; Open-Source T2I Models: Qwen-Image [49], Qwen-Image-2512 [49], Z-Image [44], and FLUX2-dev [4]; Open-Source Unified Models: BAGEL [12], UniWorld-V1 [25], OmniGen2 [50],

- Table 3: The evaluation results on AIBench of different state-of-the-art models. We report levelwise and overall performance across our defined five dimensions.

Methods Size Component Topology Phase Semantics Aesthetics Score Original Image - 82.65 57.98 79.57 79.13 51.11 70.09 Closed-Source Models

- Seedream 4.5 [41] - 67.89 52.42 48.14 74.47 55.48 59.68 Wan 2.6 [43] - 68.60 56.11 72.56 80.43 51.50 65.84

- Seedream 5.0 [39] - 82.93 72.81 72.10 86.53 51.78 73.23 GPT-Image-1.5 [35] - 66.23 50.87 55.95 77.55 57.50 61.62 Nano Banana Pro [17] - 87.80 74.81 82.67 88.54 55.04 77.77

Open-Source T2I Models

Qwen-Image [49] 27B 21.92 24.31 44.81 42.16 32.69 33.18 Qwen-Image-2512 [49] 27B 32.27 29.11 39.95 56.39 56.45 42.83 Z-Image [44] 6B 25.61 34.42 57.41 54.00 36.65 41.62 FLUX2-dev [4] 32B 23.40 24.84 52.55 46.52 42.40 37.94

Open-Source Unified Models

BAGEL [12] 7B 1.16 10.34 27.43 3.12 37.21 15.85 UniWorld-V1 [25] 12B 3.08 21.39 31.47 3.73 32.89 18.51 OmniGen2 [50] 7B 2.37 4.03 14.09 6.43 36.06 12.60 BLIP3o-NEXT [8] 3B 0.50 3.04 2.22 0.31 47.21 10.66 Emu3.5 [11] 34B 31.04 29.90 43.49 57.78 48.53 42.15

BLIP3o-NEXT [8], and Emu3.5 [11]. We employ Qwen3-VL-235B-A22B-Instruct [2] as the VQA solver for all generated images, and UniPercept [5] for the aesthetic evaluation. See more detailed settings in the Appendix.

###### 4.2 Main Results on AIBench

Overall Performance and Model Hierarchy. Table 3 presents a comprehensive evaluation of various state-of-the-art models on AIBench. A prominent observation is the substantial performance gap between closed-source and open-source models. While existing general generative benchmarks (e.g., GenEval [15]) often present performance saturation, where the gap between open-source and closedsource models appears marginal, our AIBench reveals a starkly different reality, which demands high information density and deep logical reasoning over long texts. Closed-source models like Nano Banana Pro maintain robust performance (77.77), demonstrating a superior capacity for deep semantic understanding and dense information rendering. In contrast, open-source models struggle significantly, highlighting that in highly challenging framework diagram scenarios, closed-source models still hold an absolute advantage in complex reasoning and structural generation.

Findings 1: The performance saturation on general generative benchmarks masks a profound capability gap between open- and closed-source models when confronting the high-density, complex reasoning required for academic illustration generation.

The Trade-off Between Logical Fidelity and Aesthetics. In the generation of complex, informationrich academic illustrations, a trade-off exists in academic illustration generation: models may struggle to optimize both rigorous logical constraints and aesthetic quality. Our evaluation demonstrates that generated results carrying high-density structural information typically achieve higher logical fidelity scores but suffer noticeable drops in aesthetic appeal. For example, GPT-Image-1.5 achieves the highest aesthetic quality among all closed-source models, yet its performance in logical dimensions remains uncompetitive. As shown in Fig. 5, the top-performing Nano Banana Pro model exhibits high information density with acceptable aesthetics. However, it still falls short of GPT; due to its lower information density, GPT achieves a cleaner and more visually appealing layout.

###### Wan-2.6: A GPT-Image-1.5: D

[Figure 59]

[Figure 60]

###### flux 2.0: A Wan-2.6: B

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Missing Components

[Figure 68]

[Figure 69]

[Figure 70]

###### Layout Error

Q: Is the final weight obtained by merging W₀ (frozen weights) with block-wise updates? A: No, the frozen weights are discarded before the final step. B: The image does not shown. C: No, the input bypasses the representation. D: Yes, it depicts a summation of the static base weights.

Q: Where does the 'condition DiT' send its 'output feature' to influence the final video generation?

A: The image does not contain sufficient information to answer. B: diffusion transformer (DiT) C: historical cross attention D: autoregressive generation strategy

(d)

(a)

###### GPT-Image-1.5: C Nano Banana Pro: A

###### Wan-2.6: A Nano Banana Pro: D

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Text Render

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Hallucination Reasoning

Q: Which component is responsible for receiving the 'scene descriptions' and producing bbox regions? A: Florence-2 B: InternVL2.5 (Global) C: SAM2 D: The image does not contain sufficient information to answer.

Q: Is the ‘Cluster’ operation exists in Temporal Puning stage?'

A: No, there is no cluster operation B: The image does not contain sufficient information. C: No, the operation exists in data selection stage. D: Yes, the cluster operation exist.

(c) (d)

- Figure 5: Qualitative examples of typical generation failure modes leading to incorrect answers: (a) missing components, (b) layout errors, (c) hallucinated reasoning/incorrect logic, and (d) unclear text rendering.

Findings 2: Models face a conflict mirroring human design challenges: logical completeness and visual aesthetics are often inherently exclusive and hard to balance.

Comparison with Original Images. Intriguingly, our evaluation reveals that the top-performing generated diagrams can surpass the Original Image baseline (70.09) in terms of the overall score. Models like Nano Banana Pro (77.77) and Seedream 5.0 (73.23) outperform human-drawn original figures primarily in Component completeness and Global Semantic alignment. This counterintuitive result may mainly come from the subjectivity of humans. Human authors often omit certain components, relying on implicit domain conventions, while advanced models can produce diagrams that explicitly and exhaustively align with the methodology text.

Typical Failure Modes in Logical Rendering. As illustrated in Fig. 5, our qualitative analysis identifies four primary failure modes that hinder current models when translating complex texts into visual layouts. First, models like FLUX 2.0 frequently suffer from missing components, entirely omitting critical functional pathways (Fig. 5a). Second, even when components are present, models may exhibit layout errors; for instance, Wan-2.6 fails to spatially arrange elements to reflect the underlying mathematical summation (Fig. 5b). Third, hallucinated reasoning is prevalent, with models like GPT-Image-1.5 fabricating disconnected data flows or misattributing functions (Fig. 5c). Finally, poor text rendering and phase confusion (e.g., Wan-2.6 in Fig. 5d) severely degrade logical usability by obscuring architectural stages. In contrast, models with superior ability (e.g., Nano Banana Pro) effectively mitigate these issues by accurately grounding textual logic into structural visual layouts. These failures underscore that academic illustration generation is a rigorous visual reasoning challenge rather than merely an aesthetic pixel arrangement for most models.

###### 4.3 Roadmap Towards Logical Visual Generation.

To further push the boundaries of academic illustration generation, we explore the potential of TestTime Scaling (TTS). The inherent difficulty of this task lies in its dual requirements: it demands not only deep comprehension and structural planning over long, complex methodology texts but also highly robust visual rendering capabilities. Consequently, we demonstrate that applying TTS to either the reasoning or generation aspect can effectively boost overall performance. We apply targeted enhancements to different models based on their distinct bottlenecks. The quantitative results of these explorations are detailed in Table 4.

Scaling the Reasoning Phase. Different from MLLMs [30], Open-source T2I models typically struggle with the long-context comprehension required to parse raw methodology texts. To mitigate

- Table 4: Exploration on the different test time scaling methods to increase the performance on AIBench. We use different models based on their distinct bottlenecks.

Methods Component Topology Phase Semantics Aesthetics Overall Rewriting

Qwen-Image-2512 [49] 32.27 29.11 39.95 56.39 56.45 42.83 Rewritten Qwen-Image-2512 56.97 45.93 57.71 71.97 59.35 58.39

AutoFigure Pipeline [57]

SVG Code (Gemini-2.5-Flash [10]) 87.51 79.66 81.90 91.98 43.12 76.83 Nano Banana Pro [17] 87.80 74.81 82.67 88.54 55.04 77.77 SVG Prompted Nano Banana Pro 87.57 76.50 78.91 92.14 55.05 78.03

Post Enhancement

Wan2.6 [43] 68.60 56.11 72.56 80.43 51.50 65.84 Wan2.6 w/ BoN 69.90 58.50 74.24 80.53 52.97 67.23 Wan2.6 w/ Edit 75.70 61.80 72.81 82.02 54.21 69.31

(a) (b)

- Figure 6: (a) shows the sensitivity analysis on different VLM solvers. (b) shows ranking correlation with human evaluation. ρ denotes Spearman’s rank correlation coefficient.

this, we introduce a Rewriting strategy [29], utilizing a powerful LLM Qwen-Max to act as a planner. The LLM pre-reasons over the method text and condenses it into a highly structured prompt. As shown in Table 4, applying this to Qwen-Image-2512 yields a massive improvement, elevating its overall score from 42.83 to 58.39, with substantial gains across all logical dimensions. Taking this a step further, we explore the AutoFigure pipeline [57], an extreme form of reasoning scaling where Gemini 2.5 Flash [10] explicitly writes SVG code to represent the framework. Evaluating this pure SVG output directly corroborates our earlier Findings 2 regarding the logic-aesthetic trade-off: the pure code-driven SVG achieves exceptional logical accuracy (e.g., 91.98 in Semantics) but suffers from severe aesthetic deficiencies (43.12). However, when this SVG is used as a structural prior to prompt a strong closed-source generator like Nano Banana Pro, we observe a synergistic effect, pushing the state-of-the-art overall score to 78.03.

Scaling the Generation Phase. For models with robust inherent reasoning but imperfect rendering, we explore Post Enhancement strategies at the generation end. We evaluate two specific pipelines on Wan2.6: 1) Best-of-N (BoN): We sample N = 4 candidate images and utilize a strong VLM judge (Qwen3-VL-235B) to select the optimal output. This search strategy effectively filters out hallucinations, boosting the overall score from 65.84 to 67.23. 2) Post-Editing: Following [29], we employ Nano Banana Pro as a dedicated editing model to explicitly correct localized visual artifacts and logical misalignments in the initial draft. This targeted refinement proves even more effective, elevating Wan2.6’s performance to 69.31. Ultimately, these explorations demonstrate that while end-to-end generation remains bottlenecked, strategically allocating compute at test-time, either to pre-plan the logical topology or post-refine the visual rendering, provides a highly effective roadmap for achieving production-ready academic illustrations.

Findings 3: Navigating the dual challenges of long, complex text reasoning and high-density rendering is crucial. By independently applying test time scaling to both stages, we can effectively shatter the capability ceilings of current models.

###### 4.4 Robustness Analysis of AIBench

Sensitivity of VLM Solvers. To investigate the robustness of our VQA-based evaluation framework, we analyze the sensitivity of the final scores to the choice of the VLM acting as the QA solver. As illustrated in Figure 6(a), we evaluate the generated diagrams alongside the original images using four different SOTA VLMs (Qwen2.5-VL-72B [3], Qwen3-VL-235B [2], Qwen3.5-397B [53], and GPT-5.1 [34]). The absolute mean scores exhibit slight fluctuations across different solvers, which is an expected phenomenon stemming from the intrinsic ability in visual reasoning and instruction following. However, the relative performance ranking of the evaluated T2I models remains remarkably consistent. This demonstrates that AIBench provides a reliable assessment, ensuring that the benchmark’s conclusions are robust and not biased by the VLM judge.

Correlation with Human Evaluation. To validate the effectiveness of our proposed metric, we conduct a system-level correlation analysis comparing our method, VLM-based judges, and human expert evaluation. For the VLM baselines, we employ Gemini 3 Flash and design two evaluation formats: 1) VLM Score: direct numerical scoring; 2) VLM Arena: a pairwise comparison approach (Original image vs five generated). To establish a human rank, domain experts performed blind pairwise comparisons of generated images, with aggregated win rates forming the final ranking. As Figure 6(b) illustrates, our method’s rank correlation with human experts (ρ = 0.886) significantly exceeds the VLM Score (ρ = 0.600) and VLM Arena (ρ = 0.714) baselines, proving its better alignment with human intuition.

#### 5 Conclusion

In this paper, we introduce AIBench, the first fine-grained, VQA-based benchmark specifically designed to evaluate the generation of complex academic illustrations. By moving beyond the unreliable paradigm of holistic VLM-as-Judge assessments, we establish a rigorous evaluation framework that explicitly decouples objective logical consistency from subjective aesthetic quality. Our extensive evaluations of state-of-the-art models reveal a profound capability gap when models are tasked with long, complex logical reasoning and high-density generation. Furthermore, we identify an inherent struggle of simultaneously optimizing rigorous structural fidelity and visual aesthetics. Finally, we demonstrate that test time scaling applied to either the reasoning or the generation phase can effectively shatter current capability ceilings. We hope AIBench serves as a critical stepping stone and roadmap for the community, driving future research toward multimodal generation frameworks that can master both deep logical planning and high-fidelity aesthetic rendering.

#### References

- [1] Improved aesthetic predictor. https://github.com/christophschuhmann/ improved-aesthetic-predictor, 2022. 7
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 6, 7, 8, 11
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923,

2025. 11

- [4] BlackForest. Flux. https://github.com/black-forest-labs/flux, 2024. 1, 3, 7, 8
- [5] Shuo Cao, Jiayang Li, Xiaohui Li, Yuandong Pu, Kaiwen Zhu, Yuanting Gao, Siqi Luo, Yi Xin, Qi Qin, Yu Zhou, et al. Unipercept: Towards unified perceptual-level image understanding across aesthetics, quality, structure, and texture. arXiv preprint arXiv:2512.21675, 2025. 7, 8
- [6] Jingye Chen, Yupan Huang, Tengchao Lv, Lei Cui, Qifeng Chen, and Furu Wei. Textdiffuser-2: Unleashing the power of language models for text rendering. In European Conference on Computer Vision, pages 386–402. Springer, 2024. 1
- [7] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025. 1
- [8] Jiuhai Chen, Le Xue, Zhiyang Xu, Xichen Pan, Shusheng Yang, Can Qin, An Yan, Honglu Zhou, Zeyuan Chen, Lifu Huang, et al. Blip3o-next: Next frontier of native image generation. arXiv preprint arXiv:2510.15857, 2025. 3, 8
- [9] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025. 1
- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 10, 3
- [11] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3.5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025. 8
- [12] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 7, 8
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 1
- [14] Yifan Gao, Zihang Lin, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, and Hongtao Xie. PosterMaker: Towards high-quality product poster generation with accurate text rendering. In CVPR, pages 8083–8093, 2025. 1
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. GenEval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Inf. Process. Syst., pages 52132–52152, 2023. 3, 8
- [16] Google. gemini3. https://blog.google/products-and-platforms/products/ gemini/gemini-3/, 2025. 2

- [17] Google. Nano banana pro. https://blog.google/innovation-and-ai/products/ nano-banana-pro/, 2025. 7, 8, 10, 3
- [18] Xiwei Hu, Haokun Chen, Zhongqi Qi, Hui Zhang, Dexiang Hong, Jie Shao, and Xinglong Wu. DreamPoster: A unified framework for image-conditioned generative poster design. arXiv preprint arXiv:2507.04218, 2025. 1
- [19] Yue Hu and Xiaojun Wan. PPSGen: Learning-based presentation slides generation for academic papers. In IJCAI, pages 1085–1097, 2014. 3
- [20] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2icompbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Trans. Pattern Anal. Mach. Intell., 47(5):3563–3579, 2025. 3
- [21] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, PhengAnn Heng, and Hongsheng Li. T2I-R1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025. 1
- [22] Kaixun Jiang, Zhaoyu Chen, Haijing Guo, Jinglun Li, Jiyuan Fu, Pinxue Guo, Hao Tang, Bo Li, and Wenqiang Zhang. Enhancing diffusion-based unrestricted adversarial attacks via adversary preferences alignment. arXiv preprint arXiv:2506.01511, 2025. 1
- [23] Kaixun Jiang, Yuzheng Wang, Junjie Zhou, Pandeng Li, Zhihang Liu, Chen-Wei Xie, Zhaoyu Chen, Yun Zheng, and Wenqiang Zhang. Genagent: Scaling text-to-image generation via agentic multimodal reasoning. arXiv preprint arXiv:2601.18543, 2026. 1
- [24] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-Pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Inf. Process. Syst., pages 36652–36663, 2023. 7
- [25] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. UniWorld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 1, 7, 8
- [26] Jinpeng Lin, Min Zhou, Ye Ma, Yifan Gao, Chenxi Fei, Yangjian Chen, Zhang Yu, and Tiezheng Ge. AutoPoster: A highly automatic and content-aware design system for advertising poster generation. In ACM MM, pages 1250–1260, 2023. 1
- [27] Boyang Liu, Yifan Hu, Senjie Jin, Shihan Dou, Gonglei Shi, Jie Shao, Tao Gui, and Xuanjing Huang. Unlocking the essence of beauty: Advanced aesthetic reasoning with relative-absolute policy optimization. arXiv preprint arXiv:2509.21871, 2025. 7
- [28] Zeyu Liu, Weicong Liang, Zhanhao Liang, Chong Luo, Ji Li, Gao Huang, and Yuhui Yuan. Glyph-ByT5: A customized text encoder for accurate visual text rendering. In European Conference on Computer Vision, pages 361–377. Springer, 2024. 1
- [29] Zhihang Liu, Xiaoyi Bao, Pandeng Li, Junjie Zhou, Zhaohe Liao, Yefei He, Kaixun Jiang, Chen-Wei Xie, Yun Zheng, and Hongtao Xie. ShowTable: Unlocking creative table visualization with collaborative reflection and refinement. arXiv preprint arXiv:2512.13303, 2025. 1, 10
- [30] Zhihang Liu, Chen-Wei Xie, Pandeng Li, Liming Zhao, Longxiang Tang, Yun Zheng, Chuanbin Liu, and Hongtao Xie. Hybrid-level instruction injection for video token compression in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8568–8578, 2025. 9
- [31] Zhihang Liu, Chen-Wei Xie, Bin Wen, Feiwu Yu, Jixuan Chen, Pandeng Li, Boqiang Zhang, Nianzu Yang, Yinglu Li, Zuan Gao, Yun Zheng, and Hongtao Xie. Capability: A comprehensive visual caption benchmark for evaluating both correctness and thoroughness, 2025. 5
- [32] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025. 3
- [33] OpenAI. Dall·e 3. https://openai.com/index/dall-e-3/, 2023. 1

- [34] OpenAI. Gpt-5. https://openai.com/index/introducing-gpt-5/, 2025. 11
- [35] OpenAI. Gpt-image-1.5. https://openai.com/index/new-chatgpt-images-is-here/,

2025. 7, 8

- [36] Wei Pang, Kevin Qinghong Lin, Xiangru Jian, Xi He, and Philip Torr. Paper2poster: Towards multimodal poster automation from scientific papers. arXiv preprint arXiv:2505.21497, 2025. 3
- [37] Yuting Qiang, Yanwei Fu, Yanwen Guo, Zhi-Hua Zhou, and Leonid Sigal. Learning to generate posters of scientific papers. In AAAI, pages 929–937, 2016. 3
- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1
- [39] ByteDance Seed. Seedream 5.0 lite. https://seed.bytedance.com/en/blog/ deeper-thinking-more-accurate-generation-introducing-seedream-5-0-lite,

2025. 7, 8

- [40] Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, Xuyan Chi, Jian Cong, Jing Cui, Qinpeng Cui, Qide Dong, Junliang Fan, Jing Fang, Zetao Fang, Chengjian Feng, Han Feng, Mingyuan Gao, Yu Gao, Dong Guo, Qiushan Guo, Boyang Hao, Qingkai Hao, Bibo He, Qian He, Tuyen Hoang, Ruoqing Hu, Xi Hu, Weilin Huang, Zhaoyang Huang, Zhongyi Huang, Donglei Ji, Siqi Jiang, Wei Jiang, Yunpu Jiang, Zhuo Jiang, Ashley Kim, Jianan Kong, Zhichao Lai, Shanshan Lao, Yichong Leng, Ai Li, Feiya Li, Gen Li, Huixia Li, JiaShi Li, Liang Li, Ming Li, Shanshan Li, Tao Li, Xian Li, Xiaojie Li, Xiaoyang Li, Xingxing Li, Yameng Li, Yifu Li, Yiying Li, Chao Liang, Han Liang, Jianzhong Liang, Ying Liang, Zhiqiang Liang, Wang Liao, Yalin Liao, Heng Lin, Kengyu Lin, Shanchuan Lin, Xi Lin, Zhijie Lin, Feng Ling, Fangfang Liu, Gaohong Liu, Jiawei Liu, Jie Liu, Jihao Liu, Shouda Liu, Shu Liu, Sichao Liu, Songwei Liu, Xin Liu, Xue Liu, Yibo Liu, Zikun Liu, Zuxi Liu, Junlin Lyu, Lecheng Lyu, Qian Lyu, Han Mu, Xiaonan Nie, Jingzhe Ning, Xitong Pan, Yanghua Peng, Lianke Qin, Xueqiong Qu, Yuxi Ren, Kai Shen, Guang Shi, Lei Shi, Yan Song, Yinglong Song, Fan Sun, Li Sun, Renfei Sun, Yan Sun, Zeyu Sun, Wenjing Tang, Yaxue Tang, Zirui Tao, Feng Wang, Furui Wang, Jinran Wang, Junkai Wang, Ke Wang, Kexin Wang, Qingyi Wang, Rui Wang, Sen Wang, Shuai Wang, Tingru Wang, Weichen Wang, Xin Wang, Yanhui Wang, Yue Wang, Yuping Wang, Yuxuan Wang, Ziyu Wang, Guoqiang Wei, Wanru Wei, Di Wu, Guohong Wu, Hanjie Wu, Jian Wu, Jie Wu, Ruolan Wu, Xinglong Wu, Yonghui Wu, Ruiqi Xia, Liang Xiang, Fei Xiao, XueFeng Xiao, Pan Xie, Shuangyi Xie, Shuang Xu, Jinlan Xue, Shen Yan, Bangbang Yang, Ceyuan Yang, Jiaqi Yang, Runkai Yang, Tao Yang, Yang Yang, Yihang Yang, ZhiXian Yang, Ziyan Yang, Songting Yao, Yifan Yao, Zilyu Ye, Bowen Yu, Jian Yu, Chujie Yuan, Linxiao Yuan, Sichun Zeng, Weihong Zeng, Xuejiao Zeng, Yan Zeng, Chuntao Zhang, Heng Zhang, Jingjie Zhang, Kuo Zhang, Liang Zhang, Liying Zhang, Manlin Zhang, Ting Zhang, Weida Zhang, Xiaohe Zhang, Xinyan Zhang, Yan Zhang, Yuan Zhang, Zixiang Zhang, Fengxuan Zhao, Huating Zhao, Yang Zhao, Hao Zheng, Jianbin Zheng, Xiaozheng Zheng, Yangyang Zheng, Yijie Zheng, Jiexin Zhou, Jiahui Zhu, Kuan Zhu, Shenhan Zhu, Wenjia Zhu, Benhui Zou, and Feilong Zuo. Seedance 1.5 pro: A native audio-visual joint generation foundation model, 2025. 1, 3
- [41] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 7, 8
- [42] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, Xiao Hu, Xiaohua Hu, Boyuan Jiang, Fangyuan Kong, Hang Li, Jie Li, Qingyu Li, Shen Li, Xiaohan Li, Yan Li, Jiajun Liang, Borui Liao, Yiqiao Liao, Weihong Lin, Quande Liu, Xiaokun Liu, Yilun Liu, Yuliang Liu, Shun Lu, Hangyu Mao, Yunyao Mao, Haodong Ouyang, Wenyu Qin, Wanqi Shi, Xiaoyu Shi, Lianghao Su, Haozhi Sun, Peiqin Sun, Pengfei Wan, Chao Wang, Chenyu Wang, Meng Wang, Qiulin Wang, Runqi Wang, Xintao Wang, Xuebo Wang, Zekun Wang, Min Wei, Tiancheng Wen, Guohao Wu, Xiaoshi Wu, Zhenhua Wu, Da Xie, Yingtong Xiong, Yulong Xu, Sile Yang, Zikang Yang, Weicai Ye, Ziyang Yuan, Shenglong Zhang, Shuaiyu Zhang, Yuanxing Zhang, Yufan Zhang, Wenzheng

Zhao, Ruiliang Zhou, Yan Zhou, Guosheng Zhu, and Yongjie Zhu. Kling-omni technical report,

2025. 1

- [43] Wan Team. Wan2.6. https://wan.video/introduction/wan2.6, 2025. 1, 3, 7, 8, 10
- [44] Z-Image Team. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 1, 3, 7, 8
- [45] Yuxiang Tuo, Yifeng Geng, and Liefeng Bo. Anytext2: Visual text generation and editing with customizable attributes. arXiv preprint arXiv:2411.15245, 2024. 1
- [46] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1
- [47] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Biao Gong, Longxiang Tang, Xiang Wang, Haonan Qiu, Hengjia Li, Shuai Tan, Yingya Zhang, et al. Dreamrelation: Relation-centric video customization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12381–12393, 2025. 1
- [48] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Yujin Han, Zhekai Chen, Jiayu Wang, Difan Zou, Xihui Liu, Yingya Zhang, Yu Liu, et al. Routing matters in moe: Scaling diffusion transformers with explicit routing guidance. arXiv preprint arXiv:2510.24711, 2025. 1
- [49] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 3, 7, 8, 10
- [50] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 1, 3, 7, 8
- [51] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528,

2024. 1

- [52] Sheng Xu and Xiaojun Wan. Posterbot: A system for generating posters of scientific papers with neural models. In AAAI, pages 13233–13235, 2022. 3
- [53] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 11
- [54] Hao Zheng, Xinyan Guan, Hao Kong, Wenkai Zhang, Jia Zheng, Weixiang Zhou, Hongyu Lin, Yaojie Lu, Xianpei Han, and Le Sun. PPTAGENT: Generating and evaluating presentations beyond text-to-slides. In EMNLP, pages 14413–14429, 2025. 3
- [55] Dawei Zhu, Rui Meng, Yale Song, Xiyu Wei, Sujian Li, Tomas Pfister, and Jinsung Yoon. PaperBanana: Automating academic illustration for ai scientists. arXiv preprint arXiv:2601.23265,

2026. 1, 3

- [56] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 7
- [57] Minjun Zhu, Zhen Lin, Yixuan Weng, Panzhong Lu, Qiujie Xie, Yifan Wei, Sifan Liu, Qiyao Sun, and Yue Zhang. AutoFigure: Generating and refining publication-ready scientific illustrations. arXiv preprint arXiv:2602.03828, 2026. 1, 3, 7, 10

### Appendix

Overview

- A More Details of AIBench 1

- A.1 QA Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.2 Human Expert Annotation Details . . . . . . . . . . . . . . . . . . . . . . . . . . 1

- B More Experiments 1

- B.1 More Experimental Settings and Details . . . . . . . . . . . . . . . . . . . . . . . 1
- B.2 More Analysis on Test-Time Scaling . . . . . . . . . . . . . . . . . . . . . . . . . 3
- B.3 The Trade-off Between Logical Fidelity and Aesthetics . . . . . . . . . . . . . . . 4
- B.4 Academic Illustrations Generated by Different Models on AIBench . . . . . . . . . 4
- B.5 Illustration of a Case in AIBench . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- C Limitation and Future Work 7
- D Copyright and Licensing 7

#### A More Details of AIBench

- A.1 QA Construction

Text-to-Logic Directed Graph. The construction of the Text-to-Logic Directed Graph proceeds in two stages: we first extract the core logic from the raw method text, and then generate the Logic Directed Graph based on the extracted execution logs. This process is implemented using the Gemini 3 Flash model, with the corresponding prompts illustrated in Figure A9. We also render several cases of directed graphs in Figure A1.

Multi-Level QA Generation. We utilize Gemini 3 Flash to generate QA pairs across four levels. The prompt designs for each level are illustrated in the figures provided: Level 1: Component Existence (Figure A10), Level 2: Local Topology (Figure A11), Level 3: Phase Architecture (Figure A12), Level 4: Global Semantics (Figure A13).

- A.2 Human Expert Annotation Details

To ensure the generation of high-quality and reliable ground truth data, a rigorous manual annotation process was conducted by human experts. All selected annotators are Ph.D. students possessing relevant domain expertise, specifically the first seven authors of this paper. In total, 300 articles were meticulously annotated. With each sample requiring an average of 20 minutes of thorough review, the entire annotation effort encompassed a substantial workload of approximately 100 hours of dedicated expert labor.

#### B More Experiments

###### B.1 More Experimental Settings and Details

Model Inference and Prompt Design. To ensure a fair and comprehensive evaluation, we separately design the inference prompts for closed- and open-source models tailored to the distinct capabilities of the evaluated models, as shown in Fig. A2. Because closed-source models generally exhibit superior instruction-following and long-text comprehension capabilities, we utilize a relatively

[Figure 83]

ID: CVPR_2025/2408.16266v2

[Figure 84]

ID: ICLR_2025/2410.06940v4

[Figure 85]

ID: ICCV_2025/2507.10225v3

Figure A1: Examples of rendered text-to-logic directed graph derived from method-text. Purple boxes of two shapes represent nodes: rectangular purple boxes denote processing modules (e.g., VAE encoder, MLLM), while elliptical purple boxes represent data entities (e.g., InD images, latent features). Black arrows denote edges, indicating the direction of data flow between nodes. Yellow boxes indicate phases, corresponding to distinct procedural steps (e.g., Image Generation, Feature Encoding).

Inference Prompt (Closed-Source Model)

prompt = f"Generate a framework figure based on the given paper method section.\n {method_text}"

Inference Prompt (Open-Source Model)

prompt = f"Generate an academic architecture diagram based on the given paper method section. Use deep learning visual elements like feature maps, network layers, blocks instead of basic flowchart boxes or only print texts. Clean vectorgraphic style.\n Method text: \n {method_text}"

Figure A2: Different inference prompts for closed- and open-source models.

concise prompt for this group. Conversely, current open-source models often struggle to grasp the specific structural requirements of academic illustration generation. Without sufficient constraints, these models frequently exhibit degenerate behaviors, such as directly rendering the raw methodology text onto the image canvas instead of synthesizing a visual framework. To mitigate this, we employ a more detailed and explicitly constrained prompt for the open-source models. Importantly, to prevent trivial outputs across the board, the prompts for all models are carefully adjusted to include negative constraints. Specifically, the models were explicitly instructed not to merely print the raw text and not to generate overly simplistic, basic flowcharts.

Resolution Settings. For the image generation process, we adhere to the default resolution settings of each respective model to reflect their standard performance. Meanwhile, to ensure that the generated academic illustrations possess sufficient clarity and high-density detail for fine-grained logic, we enforced a minimum resolution threshold of 1024×1024 pixels for all generated outputs.

QA Evaluation Protocol. During the evaluation phase, we standardize the VQA process to guarantee fair comparisons and reliable reproducibility. A major limitation of the existing VLM-as-Judge paradigm is its reliance on complex, heavily engineered, and often brittle prompts to guide the

Evaluation (QA) Prompt

system_prompt = "You are a helpful assistant." user_prompt = f"Question: {question}\nOptions:\n{option_text}\n\nBased on the given image only, answer with the option's letter from the given choices directly."

Figure A3: Prompt of QA evaluation.

Table A1: Extended evaluation of Test-Time Scaling strategies on AIBench. This table details the impact of the explicit reasoning phase ( e.g., text rewriting and SVG-based structural) priors across models with varying inherent capabilities, illustrating how the effectiveness of these enhancements heavily depends on the model’s native comprehension and generation capacity.

Methods Component Topology Phase Semantics Aesthetics Overall Rewriting

Wan2.6 w/o Prompt Enhance 35.29 42.70 64.66 79.64 42.58 52.97 Gemini Rewritten Wan2.6 47.61 49.55 66.56 79.98 58.87 60.51 Wan2.6 [43] 68.60 56.11 72.56 80.43 51.50 65.84

AutoFigure Pipeline [57]

SVG Code (Gemini-2.5-Flash [10]) 87.51 79.66 81.90 91.98 43.12 76.83 Qwen-Image-2512 [49] 32.27 29.11 39.95 56.39 56.45 42.83 SVG Based Qwen-Image-2512 4.24 2.69 4.53 6.54 43.02 12.20 Nano Banana [17] 79.07 62.89 71.00 87.23 53.86 70.81 SVG Based Nano Banana 76.90 66.99 75.88 87.30 53.09 72.03 Nano Banana Pro [17] 87.80 74.81 82.67 88.54 55.04 77.77 SVG Prompted Nano Banana Pro 87.57 76.50 78.91 92.14 55.05 78.03

judging model. In contrast, our VQA-based approach utilizes a single, straightforward QA template applied universally across all evaluated models, as illustrated in Fig. A3. This unified template directly queries the VQA solver using our automatically generated and carefully human-checked multi-level question-answer pairs. By relying on objective visual queries rather than open-ended VLM judgments, we eliminate the need for intricate evaluation prompt tuning. This ensures that the resulting performance metrics strictly reflect the true logical consistency of the generated illustrations, rather than the efficacy of the evaluation prompt itself.

###### B.2 More Analysis on Test-Time Scaling

To supplement the findings in the main paper, we provide a more granular analysis of the TestTime Scaling (TTS) strategies, specifically exploring how different models react to text rewriting and intermediate structural priors (SVG code). The extended quantitative results are presented in Table A1.

The Efficacy of Rewriting Mechanisms. While our main text demonstrates that explicit text rewriting significantly boosts the performance of models with weaker comprehension (e.g., Qwen-Image-2512), Table A1 reveals a different dynamic for state-of-the-art closed-source models. Advanced models like Wan2.6 are typically equipped with highly optimized internal prompt enhancement mechanisms. When we intentionally disable this native feature (Wan2.6 w/o Prompt Enhance), the model’s overall score experiences a severe drop from 65.84 to 52.97, particularly losing ground in logical dimensions such as Component and Topology. Furthermore, attempting to replace this native optimization with a generalized external LLM planner (Gemini Rewritten Wan2.6) only partially recovers the performance (60.51), still underperforming the model’s built-in mechanism. This confirms that explicit, external test-time rewriting is primarily crucial for open-source models lacking robust native comprehension, whereas top-tier models already possess near-optimal, deeply integrated text-processing pipelines.

Model Capacity Dictates the Success of Structural Priors. We also conduct a deeper investigation into the AutoFigure pipeline [57], which utilizes LLM-generated SVG code as an intermediate structural prior. Our findings indicate that the effectiveness of this rigid, highly structured planning format is strictly dictated by the underlying generator’s inherent capacity.

###### Seedream 5.0 Qwen-Image-2512 BAGEL

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

- Figure A4: Qualitative comparison of academic illustrations generated by Seedream 5.0, QwenImage-2512, and BAGEL on AIBench.

For models with limited structural understanding, such as Qwen-Image-2512, injecting SVG code catastrophically disrupts the generation process. The model completely fails to interpret the abstract geometric instructions, causing the overall score to collapse from 42.83 to 12.20. Conversely, for models with solid foundational capabilities, the SVG prior serves as an effective spatial anchor; for instance, Nano Banana (used in the AutoFigure paper) shows a clear performance uplift (from 70.81 to 72.03) when guided by SVG.

An interesting saturation effect occurs when scaling to exceptionally strong models like Nano Banana Pro. The performance gain becomes marginal (improving only slightly from 77.77 to 78.03). We attribute this to the fact that highly capable models already possess proficient end-to-end planning abilities. In such cases, forcing a hard intermediate translation step (Text → SVG Code → Image) may inadvertently act as an information bottleneck. The intermediate SVG generation is prone to omitting subtle methodological details or introducing structural hallucinations, thereby limiting the potential ceiling of a model that could otherwise directly render the text accurately.

###### B.3 The Trade-off Between Logical Fidelity and Aesthetics

As discussed in Section 4.2, generating complex academic illustrations often involves a trade-off between logical fidelity and aesthetic quality. In this appendix, we provide additional examples to further demonstrate this phenomenon. As shown in Fig. A4, the closed-source model Seedream 5.0 produces illustrations with clearer logical organization, where individual steps and the overall progression are represented more accurately. However, this improvement in logical fidelity is accompanied by lower aesthetic quality. By contrast, Qwen-Image-2512 produces illustrations that are more visually appealing and exhibit a stronger technological style, leading to higher aesthetic scores. Yet its logical fidelity is comparatively weaker, making it less effective at clearly conveying the methodological content of the paper. In comparison, the unified model BAGEL shows inferior performance on both logical fidelity and aesthetic quality.

###### B.4 Academic Illustrations Generated by Different Models on AIBench

In this section, we present qualitative examples of academic illustrations generated by different models from the same method text on AIBench. As shown in Fig. A5, Fig. A6, and Fig. A7, closed-source models (green boxes) generally outperform open-source models (blue boxes) and unified models (purple boxes) in logical consistency and visual quality. Among the closed-source models, Nano Banana Pro produces illustrations that are strong in both structure and aesthetics, while GPT-Image1.5 tends to generate cleaner and more concise figures, which is consistent with our quantitative findings. In contrast, open-source models still show substantial room for improvement in this task. Unified models are relatively limited in method text understanding and figure planning; for example,

###### Method Text Seedream 5.0

Seedream 4.5

[Figure 92]

[Figure 93]

Dinomaly Framework “What I cannot create, I do not understand.” — Richard Feynman. The ability to identify anomalies based on what we already know is a fundamental human capability. Inspired by this idea, we develop a reconstruction-based framework that leverages the epistemic characteristics of neural networks. Dinomaly consists of three main components: an encoder, a bottleneck, a reconstruction decoder As the encoder, we use a pretrained Vision Transformer with 12 Transformer layers. It extracts informative feature maps at different semantic levels. The bottleneck is a simple MLP, or feed-forward network, which collects feature representations from the encoder’s eight middle layers. The decoder has a ...

Qwen-Image BAGEL UniWorld-V1

[Figure 94]

[Figure 95]

[Figure 96]

- Figure A5: Qualitative comparison of academic illustrations generated by different models from the same method text on AIBench. The orange box shows the input method text, while the green, blue, and purple boxes present outputs from closed-source, open-source, and unified models, respectively.

Method Text GPT-Image-1.5

Conceptual diagram of the proposed framework network. Given a query video and a music track as multimodal input, the proposed framework produces three outputs: Relevance score: measures how well the music matches the video Temporal center: indicates the center of the detected music moment. Temporal width: indicates the duration of the detected music moment. At a high level, the goal is to build a music grounding network that can simultaneously: estimate the relevance between the video and the music track, and detect an appropriate music segment to serve as the video’s background music. The video is represented as a sequence of evenly sampled frames, while the music track is represented as a sequence of partially...

Z-Image OmniGen2 BLIP3o-NEXT

Wan 2.6

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- Figure A6: Qualitative comparison of academic illustrations generated by different models from the same method text on AIBench.

UniWorld-V1 and BLIP3o-NEXT sometimes produce semantically meaningless illustrations, which helps explain their poor performance on AIBench. We also observe that Emu3.5 often includes a large amount of method text in its generated academic illustrations, which may partially account for its relatively higher AIBench scores compared with other unified models.

###### B.5 Illustration of a Case in AIBench

To provide a concrete understanding of our dataset, we present a holistic case study in Figure A8. Each entry in AIBench is a multi-modal ensemble designed to challenge the model’s ability to cross-reference textual logic with visual structures. As illustrated, the case begins with the Method Text, which serves as the foundational description of the pipeline architecture. From this text, a corresponding Pipeline Image is generated. The core of our benchmark lies in the QA List. Unlike

###### Method Text Qwen-Image-2512

Nano Banana Pro

[Figure 102]

[Figure 103]

Logical Parsing and Reasoning Explanation-based classification can be viewed as a matching problem between the input and the explanation. Instead of relying only on similarity or entailment scores, our method aims to better exploit the logical structure contained in natural language explanations. Our model is built around a two-stage logical matching process: Stage 1: Logical parsing of the explanation. The explanation is analyzed to uncover its internal logical structure. Stage 2: Logical reasoning over the explanation and the input. The model performs reasoning between the parsed explanation and the input to produce classification scores. Unlike logic-agnostic baselines such as ExEnt and RoBERTa-sim, which ...

FLUX2-dev Emu3.5

[Figure 104]

[Figure 105]

- Figure A7: Qualitative comparison of academic illustrations generated by different models from the same method text on AIBench.

Method Text Generated Image (By Nano-Banana Pro)

[Figure 106]

- [11] question: What is the correct sequence of flow for the compression pathway in this architecture? options: A. vision tokens \mathbf{H}^{\textrm{v}} → query-based compression module → compressed vision tokens \mathbf {\hat{H}}^{\textrm{v}} B. CLIP vision encoder → modality pre-fusion module → compressed vision tokens \mathbf{\hat{H}}^ {\textrm{v}} C. compressed vision tokens \mathbf{\hat{H}}^{\textrm{v}} → Video Token Aggregation → modality pre-fusion module D. The image does not contain sufficient information to answer or none of the other options are correct.

- pred: A

[12] question: After the initial vision tokens ( )(are generated, does the diagram depict a branching architecture where thes e tokens are sent to two separate processing modules? options: A. The image does not contain sufficient information to answer or none of the other options are correct. B. Yes, they branch into the 'Compression' module and the 'Modality Pre-fusion' module C. No, the tokens are discarded after the project ion layer D. Yes, they branch into the 'Visual Encoder' and the 'Embedding' layer

- pred: B

[13] question: During the 'Modality Pre-fusion' stage, does the visual architecture show an aggregation of outputs from both the visual and linguistic processing paths?

options: A. No, it only processes text tokens ( )(B. No, it only processes vision tokens ( )(C. Yes, it acts as a fusion point fo r vision tokens ( )(and text tokens ( )(D. The image does not contain sufficient information to answer or none of the othe r options are correct.

- pred: C

[14] question: In the initial 'Visual Inputs' stage, how does the diagram visually structure the handling of standard images, hig h-resolution images, and videos? options: A. They are shown as a sequential chain where one must be processed before the next B. They are depicted as parall el, independent entry points into the visual processing pipeline C. They are shown as being merged into a single file before re aching the encoder D. The image does not contain sufficient information to answer or none of the other options are correct.

- pred: D

- [15] question: When transitioning to the final 'LLM backbone' stage, does the architecture show a convergence of different to ken types? options: A. Yes, it aggregates compressed vision tokens ( )(and fusion tokens ( )(B. No, it only receives input from the 'Co mpression' module C. The image does not contain sufficient information to answer or none of the other options are correct. D. Yes, it merges the raw 'Visual Inputs' with the 'Language Response'

- pred: A

[16] question: Based on the pipeline's input nodes, what is the primary data processing capability of this architecture? options: A. Unimodal text summarization B. Multimodal integration of visual and textual data C. Purely generative image synt hesis from noise D. The image does not contain sufficient information to answer or none of the other options are correct.

- pred: B

[17] question: What is the primary functional objective of the overall pipeline as it moves from inputs to the final stage? options: A. The image does not contain sufficient information to answer or none of the other options are correct. B. To comp ress video files for storage efficiency C. To generate a natural language response based on visual and textual prompts D. To de tect and segment objects within a 3D point cloud

- pred: C

- [18] question: Which statement best describes the system's flexibility regarding visual input formats? options: A. It can only process single, low-resolution static images. B. It is designed to handle static images, high-resolution s plits, and temporal video sequences. C. The image does not contain sufficient information to answer or none of the other opt ions are correct. D. It is restricted to processing text and cannot accept visual data. pred: C
- [19] question: What architectural component serves as the final 'brain' or reasoning engine that produces the system's outpu t? options: A. CLIP vision encoder B. Projection layer C. LLM backbone D. The image does not contain sufficient information to a nswer or none of the other options are correct. pred: C

- [1] question: Does the diagram depict the 'CLIP vision encoder & projection layer' component (this could be represente

- d by its exact name, abbreviations like Vision Encoder/Projection, or a functionally equivalent module handling visual f eature extraction)? options: A. No B. Yes pred: B

[2] question: Is the 'query-based compression module' component present (this could be represented by its exact nam

- e, an abbreviation like Compression, or a functionally equivalent module handling token reduction or cross-attention)? options: A. Yes B. No

- [4] question: Is the 'LLM backbone' component present (this could be represented by its exact name, an abbreviation li ke LLM, or a functionally equivalent module like Large Language Model)? options: A. No B. Yes

- pred: B

[5] question: Does the diagram depict the 'High-Resolution Image Splitting' component (this could be represented by it s exact name, or a functionally equivalent process handling the division of high-res images into sub-patches)? options: A. No B. Yes pred: A [6] question: Is the 'Video Token Aggregation' component present (this could be represented by its exact name, or a fu nctionally equivalent process handling the combination of tokens from multiple video frames)? options: A. No B. Yes

- pred: A

[7] question: The 'vision tokens \mathbf{H}^{\textrm{v}}' serve as a primary data source for which two subsequent mo dules in the architecture? options: A. CLIP vision encoder and LLM's embedding layer B. The image does not contain sufficient information to ans wer or none of the other options are correct. C. query-based compression module and modality pre-fusion module f(\c dot) D. LLM backbone and response generation pred: C [8] question: Which specific data representation is produced by the 'modality pre-fusion module f(\cdot)' before being passed to the LLM backbone? options: A. The image does not contain sufficient information to answer or none of the other options are correct. B. fus ion tokens \mathbf{\hat{H}}^{\textrm{q}} C. compressed vision tokens \mathbf{\hat{H}}^{\textrm{v}} D. vision tokens \ mathbf{H}^{\textrm{v}}

- pred: B

[9] question: From which component does the 'modality pre-fusion module f(\cdot)' receive its language-based input? options: A. language instruction \mathbf{X}^{\textrm{q}} B. The image does not contain sufficient information to answ er or none of the other options are correct. C. text token representations \mathbf{H}^{\textrm{q}} D. query-based com pression module

- pred: C

[10] question: In the final stage of the pipeline, the 'LLM backbone' integrates inputs from multiple sources. Which of t he following is NOT a direct topological input to the LLM backbone? options: A. The image does not contain sufficient information to answer or none of the other options are correct. B. fus ion tokens \mathbf{\hat{H}}^{\textrm{q}} C. Video Token Aggregation D. CLIP vision encoder & projection layer

- pred: D

- pred: A

[3] question: Does the diagram depict the 'modality pre-fusion module f(.)' component (this could be represented by it s exact name, an abbreviation like Pre-fusion, or a functionally equivalent module like Transformer Blocks handling the interaction between text and vision)? options: A. No B. Yes

- pred: B

QA

We introduce LLaVA-Mini, an efficient large multimodal model with minimal vision tokens. Like previous work, LLaVA-Mini uses a vision encoder to encode an image into several vision tokens. To enhance the efficiency, LLaVA-Mini significantly reduces the number of vision tokens fed into LLM backbone through a compression module. To retain visual information during compression, based on previous findings that vision tokens play a crucial role in the early layers for fusing visual information, LLaVA-Mini introduces a modality pre-fusion module before the LLM backbone to fuse the visual information into the text tokens. The details of LLaVA-Mini are as follows… The architecture of LLaVA-Mini is illustrated in Figure 6. For the visual inputs Xv, a pre-trained CLIP vision encoder (Radford et al., 2021) is employed to extract visual features from each image. These features are then mapped into the word embedding space via a projection layer, producing vision tokens Hv ∈ RN 2×dh , where N 2 is the number of vision tokens and dh is the LLM’s embedding dimension. For the language instruction Xq, LLM’s embedding layer is used to generate text token representations Hq ∈ Rlq ×dh , where lq is the number of text tokens. Vision Token Compression To enhance the efficiency of LMMs, LLaVA-Mini reduces the number of vision tokens fed into the LLM backbone by utilizing a query-based compression module. To learn compression of the vision tokens, LLaVA-Mini introduces C × C learnable compression queries Qv. These queries interact with all vision tokens Hv through cross-attention (Li et al.,2023a; Ye et al., 2024a), selectively extracting the important visual information to produce C × Ccompressed vision tokens ˆHv ∈ RC2 ×dh . To preserve the spatial information in the image during compression, we introduce a 2D sinusoidal positional encoding P E(·) (He et al., 2021) on the learnable queries and original vision tokens. Formally, the compression can be expressed as…

QA

- Figure A8: A complete AIBench case example. The case contains the Method Text (left), the generated pipeline diagram summarizing the method (middle), and an AIBench-curated QA List (right) that includes the evaluation questions and the corresponding answers grounded in the method text and pipeline.

generic visual question-answering tasks, the questions in AIBench are meticulously constructed to target deep structural reasoning. As shown in the figure, the questions require the model to identify

specific components, trace the directional flow of information, and infer the relationship between different modules. For each question, we provide a precise answer derived from the generated image, ensuring a rigorous evaluation.

#### C Limitation and Future Work

Although AIBench establishes a comprehensive and rigorous standard for evaluating academic illustration generation, it has a few limitations that pave the way for future exploration. The current benchmark is curated from premier Artificial Intelligence conferences. While this ensures highquality and dense methodological logic, it lacks representation from other scientific disciplines, such as biology, chemistry, and material sciences, which often rely on different diagrammatic conventions (e.glet@tokeneonedot, molecular structures or complex experimental apparatuses). In the future, we aim to expand the dataset to encompass a broader spectrum of academic fields to evaluate the cross-disciplinary generalization capabilities of generative models.

Furthermore, the insights derived from our experiments highlight critical pathways for the community to advance the methodological design of academic illustration generation. Future work should primarily focus on enhancing the reasoning capabilities of generative models to better handle long contexts and complex logic, alongside improving text rendering and structural generation abilities. Additionally, efforts should be made to elevate the aesthetic quality of generated outputs, extending this capability beyond academic illustrations to encompass a wider variety of diagrams, such as flowcharts. To effectively address these challenges, future research could explore architectural improvements, design more suitable test-time scaling paradigms tailored for academic illustration generation, and introduce reinforcement learning techniques to optimize both logical accuracy and aesthetic quality.

#### D Copyright and Licensing

The source papers utilized to construct this benchmark are publicly accessible via arXiv under open-access licenses (e.g., CC BY 4.0). We strictly respect the intellectual property rights of the original authors. Our utilization of the methodology descriptions and diagrams is strictly limited to non-commercial, academic research. We rely on the principles of fair use and academic exceptions for text and data mining, ensuring that our benchmark serves as a transformative evaluation tool for the research community without infringing upon the original creators’ rights.

Text-to-Logic Directed Graph

prompt_execution_log = r'''# Role

You are a Senior Technical Architect and Algorithm Analyst. Your goal is to deconstruct academic text into a clean, logical execution workflow for the purpose of generating a technical flowchart.

# Task Analyze the provided method description text and extract the **Execution Pipeline**. You must strictly separate the logical "steps" (nodes)

and "data flows" (edges) from the descriptive noise. # 🚫 STRICT Negative Constraints (What to IGNORE)

[Figure 107]

- 1. **Ignore Comparisons:** Do not mention previous versions, other baselines, or general background info.
- 2. **Ignore Qualitative Claims:** Remove subjective descriptions like "more reliable," "enhanced capability".
- 3. **Ignore Implementation Details:** Do not mention libraries (e.g., "PyTorch", "HuggingFace") unless they are distinct modules in the flow.
- 4. **NO PARAPHRASING (CRITICAL):** Do not rename components, variables, or data structures. You must use the EXACT nouns and terminologies

from the original text. # ✅ Positive Requirements (What to EXTRACT)

[Figure 108]

- 1. **Inputs:** clearly identify the starting data types (e.g., Image, Text Query).
- 2. **Modules:** Identify distinct processing units using their ORIGINAL names.
- 3. **Logic Control:**

- - **Branching:** Explicitly state "IF condition X, THEN path A; ELSE path B".
- - **Looping:** Explicitly state "Repeat steps X-Y until condition Z".

- 4. **Data Artifacts:** Identify what data is passed between steps using exact original names.

# Output Format Provide the summary as a **Structured Execution Log** using the following syntax. All bracketed variables MUST be exact quotes from the text:

**Step [N]: [Exact Component/Method Name from Text]**

- - **Action:** [What does this component do?]
- - **Input:** [Exact Data/Variable received]
- - **Output:** [Exact Data/Variable produced]
- - **Next Logic:** [Where does the flow go? If branching, specify conditions.]
- --Input Text: {input_text}'’’

prompt_graph = r'''# Role

You are a **Technical Visualizer & Knowledge Distillation Expert**. Your goal is to convert complex technical workflows (algorithms, pipelines, architectures) into a **Conceptual Graph JSON**.

# Objective The Output JSON will be used to generate a high-level **Presentation Diagram**. The terminology used in the diagram MUST remain strictly

faithful to the academic/technical text. # Abstraction Rules (CRITICAL)

- 1. **Macro over Micro:** Merge granular steps into singular functional blocks if necessary, but keep the original naming convention.
- 2. **Strict Original Terminology (CRITICAL):** You MUST NOT hallucinate, paraphrase, or invent "user-friendly" names. Node labels, phase

names, and edge labels MUST be EXACT nouns, or method names derived directly from the Input Log.

- * *Bad:* "Challenger Model" (if this phrase is not explicitly in the log)
- * *Good:* "Feature Extractor E" (if this is the exact term in the log)

- 3. **Explicit Phasing:** You MUST group nodes into logical "Steps" or "Phases" using headers or terms present in the log (e.g., "Step 1:

Pre-training", if "Pre-training" is the authentic term).

- 4. **Ignore Implementation Details:** Only keep the *primary* logical branches.
- 5. **Visual Hints:** Include an `icon_hint` field to suggest visual metaphors (e.g., "Brain", "Document", "Filter", "Database").

# Output JSON Schema Return a strict JSON object with this structure:

```json {{

"title": "Diagram Title (Use exact algorithm/method name)", "layout_direction": "Left-to-Right", // or Top-to-Bottom "phases": [

{{

"id": "step_1", "label": "Original Phase Name from Log"

}} // ... more phases

], "nodes": [

{{

"id": "unique_id", "label": "Exact Component/Data Name", // MUST be from text "phase_id": "step_1", // Must match a phase above "type": "model", // Options: "model", "data", "process", "decision" "icon_hint": "keyword" // e.g., "robot", "file", "chart"

}}

], "edges": [

{{

"source": "id_A", "target": "id_B", "label": "Exact variable/data passed" // Keep exact original name

}} ]

}}

--Input Log: {execution_log}'''

###### Figure A9: Prompt of Text-to-Logic Directed Graph during the QA construction.

Component Existence

system_prompt = r'''# Role You are a Computer Vision Auditor assessing a diagram's completeness. # Input

- - Full Logic Graph: {full_graph}
- - Reference Image (Style Guide): (Attached). Use this to understand the abstraction level, but do not treat it as the absolute ceiling of quality.

# Task Generate 6-7 "Level 1: Component Existence" questions.

# CRITICAL SELECTION STRATEGY (Balanced Dual-Track)(80%)

- **Track A: Visible Consensus (Must-Have)**

- - If a node is in BOTH the JSON and the Reference Image, you MUST ask about it.
- - **Anti-Literal Matching**: VLM evaluators often fail if they only look for the exact JSON string. You must provide 1-2 generic functional synonyms or common abbreviations to make the question robust. (e.g., If JSON is 'Teacher Model (DAM)', synonyms could be 'Depth Anything' or 'Pre-trained Guide'). Do NOT strictly force the exact visual style of the reference image.
- - **Rationale Tag**: You must start the rationale with "[Track A: Visible Consensus]".

- **Track B: Logical Necessity (High-Quality Requirement)** (20%)

- - If a node is in the JSON but **NOT** in the Reference Image, evaluate its **Architectural Significance**:
- - **KEEP IF**: It is a core model part (e.g., critical encoders/decoders). A good diagram SHOULD show these even if the reference is simplified.
- - **DISCARD IF**: It is a trivial operation (e.g., reshape, concat).
- - **Rationale Tag**: You must start the rationale with "[Track B: Logical Necessity]". # Question Phrasing
- - Ask: "Does the diagram depict the '[Node Label]' component (this could be represented by its exact name, an abbreviation like [Abbr], or a functionally equivalent module handling [Brief Function Description])?"

# Output Schema First output your step-by-step reasoning, explicitly categorizing your chosen nodes into Track A and Track B. [Your Step-by-Step Reasoning here...]

Then output the questions. ```json [ {{ "id": 1, "level": "Level 1: Component Existence", "question": "Is the '[Node Label]' component present (this could be represented by its exact name, an abbreviation like [Abbr], or a functionally equivalent module handling [Function])?", "options": {{"A": "No", "B": "Yes"}}, "correct_option": "B", "rationale": "[Track A or Track B Tag] Logic defines this node because..." }} ] ```'''

###### Figure A10: Prompt of Level 1: Component Existence during the QA construction.

Local Topology

system_prompt = r''' # Role You are a Graph Topology Expert. # Input

- - Full Logic Graph: {full_graph}
- - Reference Image: (Attached). This is SOLELY used to set the "Visual Abstraction Level". # Task Generate 5-6 "Level 2: Local Topology" questions based on the Full Logic Graph. # CRITICAL CONSTRAINTS (ANTI-OVERFITTING)
- - **DO NOT** mention the "reference image" or its specific visual details (e.g., "in the reference image", colors, specific box shapes, spatial layout) in your questions or options.
- - The generated questions will be used to evaluate **NEW** diagrams representing the same architecture. The questions MUST test the underlying structural topology, not the specific drawing style of the reference. # SELECTION STRATEGY (Dual-Track)

**Track A: Main Architectural Flow (The "Spine" )**

- - Identify connections present in BOTH the Logic Graph and the Reference Image (this tells you what level of abstraction is expected).
- - **Action**: These are mandatory. Ask about the architectural flow itself.

- *Example*: "In this architecture, which component directly receives the output from '[Source]'?" or "Where does the primary data flow from '[Source]' next?"

- *Rationale*: Ensure the main structural pathways are tested at the correct abstraction level.

- **Track B: Structural Necessity (The "Hidden Gems" )**

- - Identify important connections in the Logic Graph that might be implicit, abstracted, or messy in the Reference Image.
- - **Filter**:
- - **DISCARD IF**: It is a **Micro-Connection** inside a black box (e.g., internal layers of a ResBlock). Do NOT ask about these if the reference style is high-level.
- - **Action**: Ask questions verifying these critical architectural links.

- *Example*: "Does the architecture require a direct pathway from [A] to [B]?" or "Which component serves as the endpoint for the connection from [A]?"
- **Distractor Design**

- - Use **Logical/Plausible Neighbors**: Nodes that belong to the same parent module or are conceptually close in the Logic Graph, but are not the actual receiver. Do not rely strictly on "visual closeness" from the reference image, as new images may place them differently. # Output Schema First output your step-by-step reasoning. Explain how the reference image helped you determine the abstraction level, and why you selected these specific architectural links. [Your Step-by-Step Reasoning here...] Then output the questions.

```json [ {{ "id": 1,

- "level": "Level 2: Local Topology", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct_option": "...", "rationale": "Track A: Main flow connection at the selected abstraction level..." }} ] ``` '''

###### Figure A11: Prompt of Level 2: Local Topology during the QA construction.

Phase Architecture

system_prompt = r'''# Role You are a Systems Architect evaluating the **Phase Architectural** of a diagram. # Input

- - Full Logic Graph: {full_graph} (Focus on the intersection of `edges` and semantic `phases` to find structural patterns).
- - Reference Image: (Attached). Use this to see if phases are explicitly labeled (e.g., "Step 1") or implicitly formed by node clusters. # Task Generate 4-5 "Level 3: Phase Architecture" questions. # CRITICAL STRATEGY (Semantic Grounding & Pattern Extraction)

- **0. Semantic Grounding (Crucial pre-requisite)**: Do NOT use abstract names like "Phase 1" or "Step 2" in your questions unless they are explicitly written in large text in the Reference Image. VLM evaluators cannot see invisible boundaries in user diagrams. You MUST translate the phase into a functional description based on the nodes inside it (e.g., instead of "Phase 2", say "the knowledge distillation stage" or "the stage processing the teacher's outputs").
- **1. Phase Parallelism**:

- - Find parallel nodes (no direct edges between them)
- - *Question Phrasing*: "During the [describe the functional stage, e.g., 'data augmentation process'], does the diagram visually structure '[Node A]' and '[Node B]' as parallel, independent branches?"

- **2. Inter-Phase Fusion / Aggregation (Fan-in)**:

- - Find multiple nodes from an earlier stage feeding into a single node in a later stage.
- - *Question Phrasing*: "When transitioning from the [describe earlier functional stage] to the [describe later functional stage], does the visual flow show an aggregation/fusion where outputs converge into '[Target Node]'?"

- **3. Inter-Phase Branching (Fan-out)**:

- - Find a node splitting into multiple paths across stages.
- - *Question Phrasing*: "Does the output of '[Source Node]' visually branch out into multiple separate processing paths for the subsequent [describe later functional stage]?"

- **4. Cross-Phase Feedback Loops**:

- - Find edges flowing backward.
- - *Question Phrasing*: "Does the diagram depict a structural feedback loop returning from the later [describe later functional stage] back to the initial [describe earlier functional stage]?" # Output Schema First, output your step-by-step reasoning. You MUST explicitly map out the functional meaning of the phases first, then identify the structural patterns. <thinking_process>

- 1. Functional Phase Mapping: [e.g., phase_1 is the Teacher Training phase containing DAM and LoRA. phase_2 is the Pseudo-Labeling phase...]
- 2. Pattern Identification: [e.g., Found Inter-Phase Branching from 'Real Panoramas' to 'Color Branch' and 'Spatial Branch'...]
- 3. Question Formulation: [Drafting questions using functional descriptions instead of raw Phase IDs...] </thinking_process> Then output the questions. ```json [ {{

- "id": 1,

"level": "Level 3: Phase Architecture", "question": "During the [functional description of the stage], does the diagram visually structure '[Node A]' and '[Node B]' as parallel branches?", "options": {{"A": "No, they are shown as a sequential pipeline", "B": "Yes, they are visually depicted as parallel or concurrent paths", "C": "No, they are merged into a single block", "D": "No, they belong to visually completely different stages"}}, "correct_option": "B", "rationale": "Logic Graph defines these as parallel within the [Phase Label]. A good architectural diagram shows this structural concurrency." }}, {{

- "id": 2, "level": "Level 3: Phase Architecture", "question": "What specific node visually functions as the primary structural gateway or choke point transitioning outputs from the [describe earlier functional stage] into the [describe later functional stage]?", "options": {{"A": "'[Node X]'", "B": "'[Node Y]'", "C": "There is no single gateway; the transition happens via multiple parallel paths", "D": "The diagram does not clearly separate these two stages visually"}}, "correct_option": "A", "rationale": "According to the Logic Graph, [Node X] acts as the sole fan-in point from the previous phase before distributing data to the next phase. The options test if the VLM can identify this specific architectural bottleneck." }} ] ```'''

###### Figure A12: Prompt of Level 3: Phase Architecture during the QA construction.

Global Semantics

system_prompt = r'''# Role You are an Architectural Semantics Expert evaluating how well a pipeline's global purpose can be understood. # Input

- - raw_text: {raw_text}
- - Full Logic Graph: {full_graph}
- - Reference Image: (Attached). This is SOLELY used to identify the general abstraction level of the modalities (e.g., whether inputs/outputs are shown as abstract data blocks or concrete data types). # Task Generate 5-6 "Level 4: Global Semantics" questions based on the Full Logic Graph. # CRITICAL CONSTRAINTS (ANTI-OVERFITTING)
- - **DO NOT** ask about specific visual examples present in the reference image (e.g., if the reference image shows a "human face" or a "car" as the input, ask about the general modality like "RGB Image" or "3D Mesh", NOT the specific object).
- - **DO NOT** use spatial references from the reference image (e.g., avoid "on the left", "in the center", "at the top"). New diagrams will have different layouts.
- - The generated questions will be used to evaluate **NEW** diagrams representing the same architecture. The questions must test if the new diagram successfully conveys the global architecture semantics, regardless of its specific layout or choice of visual examples. # CRITICAL STRATEGY: The "Semantic Standalone Test"

- 1. **Input/Output Modality**:

- - Find the START/END nodes in the Logic Graph. Ask: "What is the primary data modality processed by this architecture?"
- - *Options*: Use broad modalities: [Correct: "2D RGB Image"], [Distractors: "LiDAR Point Cloud", "Text Prompt", "Audio Waveform"].

- 2. **Core Task Identification**:

- - Based on the overall flow from START to END in the Logic Graph, determine the main objective. Ask: "What is the primary task of this overall pipeline?"
- - *Options*: [Correct: "3D Object Reconstruction"], [Distractors: "Object Detection", "Image Classification", "Semantic Segmentation"].
- - *Rule*: Derive this from the architectural nodes (e.g., presence of "Mesh Renderer"), NOT from the specific visual examples in the reference image.

- 3. **Key Architectural Paradigm**:

- - Identify the most critical central bottleneck or transformation mechanism in the Logic Graph. Ask: "What is the core representation/mechanism used in this architecture?"
- - *Options*: [Correct: "Triplane Representation"], [Distractors: "Standard CNN Feature Map", "Point Cloud Voxelization", "1D Sequence Embedding"]. # NEGATIVE CONSTRAINTS (Do NOT Ask)
- - Do NOT ask about specific dataset names (e.g., "ScanNet", "COCO").
- - Do NOT ask about specific metric scores (e.g., "SOTA", "Accuracy").
- - Do NOT ask about abstract motivations (e.g., "Why did the author use Transformer?"). # Output Schema First output your step-by-step reasoning. Explain how you determined the general modalities and tasks from the Logic Graph without relying on the specific visual instances in the reference image. [Your Step-by-Step Reasoning here...] Then output the questions. ```json [ {{

- "id": 1, "level": "Level 4: Global Semantics", "question": "Looking at the pipeline's entry points, what primary data modality does this system process?", "options": {{"A": "Text Description", "B": "2D Image", "C": "Audio Waveform", "D": "3D Point Cloud"}},

- "correct_option": "B", "rationale": "The Logic Graph defines the start node as an image input. The reference image confirms modalities are visually distinguished, so we ask about the general modality (2D Image), avoiding the specific object shown." }}, {{

"id": 2, "level": "Level 4: Global Semantics", "question": "Based on the input-to-output transformation, what is the main application domain of this architecture?", "options": {{"A": "Autonomous Driving Perception", "B": "Medical CT Analysis", "C": "3D Scene/Object Reconstruction", "D": "Satellite Remote Sensing"}},

- "correct_option": "C", "rationale": "The Logic Graph ends with a 3D rendering module. The question targets the general task inferred from the nodes, ignoring specific objects used as examples in the reference drawing." }}, {{

- "id": 3, "level": "Level 4: Global Semantics", "question": "What is the core architectural mechanism driving the central representation in this pipeline?", "options": {{"A": "Standard CNN Backbone", "B": "Triplane Fusion / Representation", "C": "Vision Transformer (ViT)", "D": "Simple Element-wise Addition"}}, "correct_option": "B", "rationale": "The Logic Graph highlights 'Triplane Fusion' as the central processing unit. The question focuses on this architectural concept rather than where it is spatially drawn." }} ] ```'''

###### Figure A13: Prompt of Level 4: Global Semantics during the QA construction.

