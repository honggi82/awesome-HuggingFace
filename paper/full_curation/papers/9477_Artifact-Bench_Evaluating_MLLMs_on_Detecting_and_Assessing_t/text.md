# arXiv:2605.18984v1[cs.CV]18May2026

## Artifact-Bench: Evaluating MLLMs on Detecting and Assessing the Artifacts of AI-Generated Videos

Yuqi Tang1,3* Yang Shi2,3*† Zhuoran Zhang2* Qixun Wang2* Xuehai Bai4 Yue Ding5 Ruizhe Chen6 Bohan Zeng2 Xinlong Chen5 Xuanyu Zhu2 Bozhou Li2 Yuran Wang2 Yifan Dai7 Chengzhuo Tong2 Xinyu Liu8 Yiyan Ji9 Yujie Wei10 Yuhao Dong11 Shilin Yan10 Fengxiang Wang12 Yi-Fan Zhang5‡ Haotian Wang13‡ Yuanxing Zhang3‡ Pengfei Wan3 1HKUST(GZ) 2PKU 3Kling Team 4HDU 5CASIA 6ZJU 7SJTU 8HKUST 9NJU 10FDU 11NTU 12Shanghai AI Lab 13THU https://github.com/FrankYang-17/Artifact-Bench

### Abstract

Recent video generative models have greatly improved the realism of AI-generated videos, yet their outputs still exhibit artifacts such as temporal inconsistencies, structural distortions, and semantic incoherence. While Multimodal Large Language Models (MLLMs) show strong visual understanding capabilities, their ability to perceive and reason about such artifacts remains unclear. Existing benchmarks often lack systematic evaluation of artifact-aware perception and fine-grained diagnostic reasoning, especially across diverse AI-generated video domains beyond photorealistic content. To address this gap, we introduce Artifact-Bench, a comprehensive benchmark for evaluating MLLMs on AI-generated video artifact detection and analysis. We first establish a three-level hierarchical taxonomy of realism artifacts, covering photorealistic, animated, and CG-style videos. Based on this taxonomy, Artifact-Bench defines three complementary tasks: real vs. AI-generated video classification, pairwise realism comparison, and fine-grained artifact identification. Experiments on 19 leading MLLMs reveal substantial limitations in artifact perception and reasoning, with many models approaching random or even below-random performance in challenging settings. We further observe significant misalignment between MLLM judgments and human perceptual preferences, highlighting their limited reliability as general evaluators for AI-generated video realism.

### 1. Introduction

Recent advances in video generative models [9, 11, 13, 21, 25, 26] have significantly improved the quality of AI-

*Equal Contribution †Project Lead ‡Corresponding Author

generated videos, enabling the synthesis of visually compelling content with increasingly realistic appearance and motion. Despite this progress, most generated videos still exhibit noticeable imperfections, such as temporal inconsistencies, structural distortions, unnatural motion, and semantic incoherence. These artifacts, although sometimes subtle, fundamentally limit perceptual realism and hinder reliable deployment in real-world applications [15, 23].

Distinguishing AI-generated videos from real-world ones has therefore become increasingly important for media authenticity, content moderation, and generative model evaluation. Among various cues, generative artifacts provide particularly informative signals, as they often reflect intrinsic limitations of current generation pipelines rather than high-level semantics. Compared to purely semantic or stylebased cues, artifact-based detection offers a more principled pathway for identifying AI-generated content [15, 23], especially as generative models continue to improve in visual fidelity. Beyond binary classification, an underexplored question is whether models can identify and diagnose these artifacts, enabling more interpretable judgments and providing insights for improving generative models. In this sense, artifact analysis serves as a critical bridge between evaluation and generation, facilitating the refinement of video generation systems toward higher realism.

In parallel, Multimodal Large Language Models (MLLMs) [1, 8, 18, 19, 27–29] have emerged as powerful general-purpose models for visual reasoning. Their ability to process complex visual inputs and generate structured language outputs makes them promising candidates for scalable video evaluation. However, it remains unclear whether current MLLMs can genuinely perceive and reason about AIGC-specific artifacts. As shown in Table 1, existing benchmarks have explored authenticity detection, preference evaluation, and artifact grounding, but often in isolated settings

- Table 1. Comprehensive Comparison with Other Benchmarks. Artifact-Benchfeatures a multi-granularity progressive three-task system with difficulty levels, systematically evaluating model capabilities in AIGC video detection, realism comparison, and fine-grained artifact diagnosis across comprehensive scenarios (including non-photorealistic types such as CG and animation) and a well-established artifact taxonomy of 30 evaluation aspects.

Diff. Levels

Multi-granularity Annotation Det. Comp. Diag. Annotator Eval. Aspects

Benchmark #Tasks #Videos #Samples Scenarios

ViF-Bench [15] 1 2,995 2,995 Real ✗ ✓ ✗ ✓ Human+MLLM 23 GenBuster-Bench [32] 2 3,150 3,150 Real ✓ ✓ ✗ ✓ MLLM 3 VF-Eval [22] 4 9,740 9,740 Real & Stylized ✗ ✗ ✗ ✓ Human+MLLM 11 UVE-Bench [16] 2 1,230 4,042 Real & Stylized ✗ ✗ ✓ ✓ Human 15 AEGIS [14] 1 3,166 3,166 Real ✓ ✓ ✗ ✓ MLLM 3

###### Artifact-Bench 3 1,350 1,100 Real & Stylized ✓ ✓ ✓ ✓ Human 30

*Det., Comp., and Diag. denote AIGC Video Detection, Realism Comparison, and Artifact Diagnosis tasks, respectively.

or limited photorealistic scenarios. Moreover, most video benchmarks emphasize semantic understanding and general reasoning rather than perceptual realism and generative artifacts, making it difficult to determine whether MLLMs rely on genuine artifact-aware perception or superficial semantic priors and dataset biases.

understanding and fine-grained perceptual reasoning. We summarize our main contributions as follows:

- 1. We conduct a systematic study of artifacts in AI-generated videos and establish a three-level hierarchical taxonomy that organizes AIGC-specific artifacts from coarse visual abnormalities to fine-grained temporal and structural inconsistencies, providing a principled foundation for artifact-aware evaluation and analysis.
- 2. We introduce Artifact-Bench, a comprehensive benchmark for evaluating the ability of MLLMs to detect and analyze artifacts in AI-generated videos. Based on our artifact taxonomy, we design a multi-level evaluation framework consisting of three complementary tasks: real vs. AI-generated video classification, pairwise realism comparison, and fine-grained artifact identification. We further develop a hybrid data construction pipeline with carefully designed difficulty stratification to support reliable and in-depth evaluation.
- 3. We conduct extensive experiments across a diverse set of state-of-the-art MLLMs and reveal fundamental limitations of current models in artifact-level perception and reasoning. Our findings show that many MLLMs exhibit near-random or even below-random performance on challenging tasks and demonstrate significant misalignment with human perceptual preferences, highlighting the urgent need for future MLLMs with stronger humanaligned realism understanding capabilities.

To address this gap, we first conduct a systematic analysis of common artifacts in AI-generated videos, covering their characteristics, causes, and perceptual manifestations. Based on this analysis, we establish a three-level artifact taxonomy that organizes AIGC video artifacts from coarse visual abnormalities to fine-grained structural and temporal inconsistencies, providing a principled foundation for artifact-oriented evaluation. Building on this taxonomy, we introduce Artifact-Bench, a benchmark for evaluating MLLMs on AI-generated video artifact detection and analysis. Artifact-Bench consists of three complementary tasks: real vs. AI-generated video classification, pairwise realism comparison, and fine-grained artifact identification, which progressively probe model capabilities from coarse-grained recognition to diagnostic reasoning. To support reliable evaluation, we develop a hybrid data construction pipeline combining real-world video collection, controlled generation, and targeted artifact synthesis, together with a difficulty stratification scheme that captures varying levels of realism and artifact subtlety.

Extensive experiments on Artifact-Bench reveal fundamental limitations of current MLLMs in perceiving and understanding artifacts in AI-generated videos. Despite strong general vision-language capabilities, many models show near-random or even below-random performance on certain tasks, exposing severe weaknesses in artifact-level perception and reasoning. Moreover, model judgments often misalign with human perceptual preferences and do not consistently follow the human-defined difficulty hierarchy, suggesting reliance on superficial statistical cues or semantic priors rather than genuine artifact perception. These findings show that artifact-aware perception remains far from solved and call for future MLLMs with stronger human-aligned realism

### 2. Related Work

#### 2.1. Multimodal Large Language Model

Multimodal Large Language Models (MLLMs) [1, 8, 12, 17, 19, 31, 35, 37] have recently demonstrated remarkable proficiency in visual understanding and multimodal reasoning. Specifically, their capacity to process and interpret temporal information has enabled a diverse array of video-based applications, such as visual question answering [4, 36], video captioning [3, 19], and video-based optical character recognition (OCR) [20, 35]. Beyond basic perception, MLLMs

excel in complex visual reasoning [2, 5, 30, 38], making them increasingly viable for sophisticated real-world scenarios [6, 39]. Leveraging these robust capabilities, recent research has begun to explore MLLMs for automated AIgenerated video detection and realism assessment, as exemplified by works like BusterX++ [33] and Skyra [15].

- 2.2. Benchmarks for AI-Generated Video Detection and Assessment

As video generative models continue to advance, recent studies have explored MLLMs as general-purpose tools for detecting and assessing artifacts in AI-generated videos. Some benchmarks focus on quality assessment and diagnostic feedback. UVE-Bench [16] introduces pairwise comparison scoring across fine-grained dimensions with human preference annotations, while VF-Eval [22] formulates evaluation as a diagnostic Question-Answering (QA) task. However, preference-based scoring provides limited insight into model reasoning, and QA-style evaluation may allow models to exploit dataset biases. Other benchmarks focus on authenticity detection and artifact localization. AEGIS [14] provides multi-modality feature annotations to evaluate model reasoning chains, GenBuster-Bench [32] adopts an MLLM-as-aJudge protocol to assess authenticity prediction rationales, and ViF-Bench [15] requires spatial-temporal grounding with timestamps and bounding boxes based on a hierarchical artifact taxonomy. Despite these advances, existing benchmarks remain limited in two aspects. First, they typically evaluate models under a single paradigm, such as authenticity classification, preference scoring, or artifact grounding, lacking a unified multi-granularity evaluation framework. Second, their evaluation scenarios are often narrow, primarily focusing on photorealistic AI-generated videos. In contrast, Artifact-Bench introduces three progressively challenging tasks: real vs. AI-generated video classification, pairwise realism comparison, and fine-grained artifact identification. These tasks systematically evaluate MLLMs from coarse authenticity perception to fine-grained artifact reasoning. Moreover, Artifact-Bench covers diverse video domains, including photorealistic, anime, and CG-style videos, offering broader applicability and stronger practical relevance.

- 3. Artifact-Bench

- 3.1. Taxonomy of Realism Artifacts in AIGenerated Videos

To support fine-grained evaluation of MLLMs on AIgenerated video realism, we first establish a hierarchical taxonomy of realism artifacts. Unlike general video quality degradation or artifacts introduced by traditional rendering pipelines, artifacts in AI-generated videos often arise from the limitations of generative models in maintaining visual fidelity, object structure, temporal continuity, and semantic

consistency. These artifacts provide important evidence for distinguishing AI-generated videos from real-world ones and, more importantly, for explaining why a generated video appears unrealistic. We construct the taxonomy through an iterative human analysis process. Specifically, we examine a diverse collection of publicly accessible AIGC videos, including photorealistic videos, stylized videos, and computergenerated visuals that aim to simulate realistic appearance or motion. By repeatedly inspecting these videos, identifying recurring failure patterns, and merging semantically overlapping cases, we iteratively refine the category boundaries and ultimately establish a hierarchical taxonomy, as shown in Figure 1.

The taxonomy is designed to cover the major types of artifacts observed in AI-generated videos as comprehensively as possible, while keeping each category interpretable and actionable for human annotation and model evaluation. It is organized into three hierarchical tiers, progressing from broad artifact domains to fine-grained diagnostic labels.

At the highest tier, we divide realism artifacts into three top-level artifact domains according to the perceptual and reasoning depth required for detection. Surface Artifacts refer to low-level visual defects that can be identified primarily from local appearance cues. Structural Defects capture failures that require understanding the organization of objects and scenes. Temporal-Semantic Violations represent higher-level failures that require integrating information across frames and applying commonsense or causal reasoning.

The middle tier further decomposes each top-level domain into failure families that describe the source of the underlying defect. For instance, within Surface Artifacts, Color & Exposure, Camera & Lens, and Image Quality & Texture represent failures of distinct visual formation or rendering processes. Similarly, Structural Defects involve failure families related to identity, morphology, spatial depth, functional structure, and optical consistency, while Temporal-Semantic Violations cover failures in motion, causality, commonsense, and scene continuity. This structure allows defects with different physical, geometric, or semantic origins to be diagnosed independently.

The finest tier provides the most fine-grained artifact descriptions and serves as the operational label space for artifact-oriented evaluation. It contains 30 fine-grained artifact types, each corresponding to a concrete and visually observable failure mode, such as Texture Inconsistency, Irreversibility Violation, or Cross-Shot Coherence.

The taxonomy is diagnostic rather than strictly mutually exclusive. A single video may contain multiple co-occurring artifacts, and one visible failure may involve multiple levels of analysis, such as structural deformation and temporal inconsistency. Therefore, Artifact-Bench supports multi-label artifact annotations, enabling a more faithful evaluation of

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

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

[Figure 63]

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

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

- Figure 1. The Hierarchical Taxonomy of AI-Generated Video Artifacts. We organize AI-generated video realism artifacts into three hierarchical tiers: top-level artifact domains, mid-level failure families, and 30 fine-grained artifact types. The taxonomy spans Surface Artifacts, Structural Defects, and Temporal-Semantic (Temp-Sem) Violations, covering failures in visual appearance, object and scene structure, temporal continuity, causality, and semantic coherence. Annotated visual examples are provided for representative fine-grained artifact types.

whether MLLMs can identify the diverse causes of unrealism in AI-generated videos.

#### 3.2. Benchmark Design

To comprehensively evaluate the capability of MLLMs in recognizing and reasoning about AI-generated videos, we design 3 complementary tasks in Artifact-Bench(as illustrated in Figure 2). These tasks progressively evaluate different aspects of authenticity understanding, including (1) distinguishing AI-generated videos from real ones, (2) comparing the realism of different synthetic videos, and (3) identifying specific artifacts that reduce video realism. Together, these tasks provide a multi-level assessment of model capabilities ranging from coarse-grained recognition to fine-grained reasoning.

Task 1: Real vs. AI-Generated Video Classification (RVAC). This task evaluates the ability of MLLMs to recognize AI-generated videos. Given a single video as input, the model must determine whether the video is real or AIgenerated and output a binary answer (“Yes” or “No”) indicating whether the video is synthetic. Each real video in the

task is paired with an AI-generated counterpart that shares similar semantic content, ensuring that the task focuses on identifying realism-related artifacts rather than semantic differences. This task primarily measures whether MLLMs can detect visual inconsistencies commonly observed in generated videos, such as abnormal motion patterns, implausible physical interactions, or temporal incoherence.

- Task 2: Pairwise Video Realism Comparison (PVRC). Beyond recognizing AI-generated videos, the second task evaluates whether MLLMs can assess the relative realism of synthetic videos. Specifically, the model is given two AI-generated videos (<video A> and <video B>) and must select the one that appears more realistic by responding with either “video A” or “video B”. The two videos in each pair share similar semantic content, ensuring that the comparison focuses on differences in visual realism rather than scene semantics. Compared with binary classification, this pairwise formulation provides a more fine-grained evaluation of a model’s ability to judge the relative realism of AI-generated videos.
- Task 3: Artifact Identification (AID). This task further

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

- Figure 2. Illustration of the three proposed tasks and their evaluation workflows. From top to bottom, the figure demonstrates the input formats and expected reasoning pipelines for RVAC, PVRC, and AID. These tasks form a comprehensive hierarchy, evaluating model capabilities from coarse-grained recognition to detailed artifact identification.

evaluates the fine-grained reasoning ability of MLLMs in accurately identifying artifacts in AI-generated videos, requiring models to explain why a video appears unrealistic. Given an AI-generated video, the model is asked to determine the primary cause of its unrealism. Each example is formulated as a multi-answer multiple-choice question with

- 6 candidate options, all of which are instantiated from the 30 fine-grained artifact types in our taxonomy. The correct options correspond to the fine-grained artifact labels that are clearly observable in the video. The incorrect options are selected from semantically related or visually confusable artifact types, typically within the same or adjacent failure families. This design prevents models from solving the task through coarse category elimination and instead requires them to discriminate among fine-grained causes of unrealism. The model is required to select all valid fine-grained artifact labels from the 6 candidates. By requiring explicit identification of the underlying artifact, this task provides a deeper evaluation of whether MLLMs can analyze and reason about the causes of visual unrealism rather than merely recognizing synthetic content.

- 3.3. Benchmark Construction

synthetic videos, which enables us to balance semantic controllability, realism diversity, and artifact coverage across different tasks. Since the three tasks in Artifact-Bench target different capabilities, we adopt task-specific data construction pipelines, as shown in Figure 3. We use Gemini 3.1 Pro [8] to generate detailed captions for videos, and employ multiple video generative models to promote diversity in the generated AIGC videos, including Kling-2.5 [13], Kling-2.1 [13], Veo 3 [9], HunyuanVideo-1.5 [25], daVinciMagiHuman [21], LTX-2.3 [11], and Wan2.2 [26].

- For Task 1: Real vs. AI-Generated Video Classification (RVAC), we first collect and carefully curate real-world videos from publicly available online sources. We then caption these videos and use the captions as prompts to generate semantically aligned AI-generated counterparts with video generative models. This one-to-one construction ensures semantic alignment, thereby directing the task toward realism-related cues rather than semantic differences.
- For Task 2: Pairwise Video Realism Comparison (PVRC), we construct semantically aligned AI-generated video pairs with varying realism levels using two complementary strategies. First, we collect high-quality AI-generated videos from publicly available sources, caption them, and use the captions to generate less realistic counterparts. Second, we directly generate multiple videos from the same prompt and

Data Collection. We construct the benchmark by combining publicly available online videos with model-generated

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

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

- Figure 3. Overview of the Artifact-Bench construction pipeline. We build a hybrid dataset combining real-world and AI-generated videos for three tasks: RVAC, PVRC, and AID. Real videos are captioned to generate semantically aligned AIGC counterparts, while AIGC pairs with varying realism are constructed via re-generation and prompt-consistent sampling. For artifact coverage, we combine natural collection with targeted generation. All samples are manually annotated, verified, and stratified into three difficulty levels (L1–L3) based on realism and artifact severity.

[Figure 157]

select pairs with comparable semantics but varying levels of realism and artifact severity. Together, these strategies ensure both semantic alignment and sufficient contrast in realism and artifact severity within each pair.

we introduce a difficulty stratification scheme over all task samples. Specifically, based on the degree of visual realism, samples are grouped into 3 levels (L1–L3) with increasing difficulty. For Task 1 and Task 3, L1 corresponds to lowrealism videos with obvious artifacts, making them easy to identify, while L3 consists of highly realistic videos that are difficult to distinguish. For Task 2, L1 denotes pairs with clear differences in realism and artifact severity, whereas L3 includes pairs with highly similar realism and subtle artifact patterns, requiring fine-grained perception to differentiate. To ensure annotation reliability despite the inherent subjectivity of difficulty assessment, each sample is independently rated by 3 expert annotators. In cases of disagreement, two additional annotators are involved, and the final label is determined via discussion and majority voting. This protocol ensures consistent and high-quality annotations.

For Task 3: Artifact Identification (AID), we aim to cover a diverse set of realism-related artifacts in AI-generated videos. We first collect AIGC videos from online sources that clearly exhibit specific artifact types. However, we observe that certain artifacts are rarely present in naturally collected AIGC videos. To address this, we design prompts to intentionally expose such failure modes, generate candidate videos, and manually select qualified samples. This combination of natural collection and targeted generation improves the coverage and diversity of artifacts in the benchmark.

Annotation and Verification. Given that many AIgenerated videos are visually close to real-world videos, we adopt a fully manual annotation protocol to ensure reliability. Each AI-generated video is independently examined by 3 experienced annotators, who analyze realism-related artifacts and provide detailed annotations. A sample is accepted only if all 3 annotators reach consistent conclusions; otherwise, it undergoes a second round of review by 2 additional annotators. Finally, all accepted samples are further verified by 2 expert annotators with extensive industry experience, providing an additional layer of quality control to ensure reliability.

#### 3.4. Statistics

Through rigorous video selection and question construction, we compile a dataset of 1,350 videos, yielding 1,100 annotated samples calibrated through multiple rounds of review. As shown in Figure 4, the samples span five major categories, 20 scenarios, and diverse durations, resolutions, and subject compositions. The AI-generated videos further cover a wide range of mainstream open-source and proprietary generation systems, allowing Artifact-Bench to capture diverse artifact distributions beyond a single model family or generation pipeline.

Difficulty Stratification. To systematically evaluate model sensitivity to varying levels of realism and artifact severity,

[Figure 158]

[Figure 159]

[Figure 160]

- Figure 4. Statistics of Artifact-Bench. (a) Hierarchical breakdown of the major video categories and diverse sub-scenarios. (b) Distribution of video sources, featuring a variety of recent state-of-the-art generative models. (c)–(e) Distributions of video duration, spatial resolution, and the number of primary subjects, respectively, demonstrating the structural diversity of our dataset.

### 4. Experiments

#### 4.1. Evaluation Setup

We evaluate a total of 19 mainstream MLLMs, including 2 cutting-edge proprietary models, 14 open-source generalpurpose models, and 3 open-source specialized models designed for AI-generated videos detection. Specifically, the proprietary models include Gemini 3.1 Pro [8] and Gemini 3 Flash [10]. The open-source general-purpose models include the Qwen3-VL series [1] (8B, 30B-A3B, and 32B), the InternVL3.5 series [31] (8B, 30B-A3B, and 38B), Molmo2 8B [7], MiMo-VL 7B [34], and Keye-VL-1.5 8B [24]. The open-source specialized models include Skyra

- 7B [15], BusterX++ [33], and VideoVeritas 8B [23]. To investigate whether reasoning-enhanced MLLMs improve the detection and assessment of artifacts in AI-generated videos, we further evaluate both instruction-tuned and reasoningenhanced (i.e., thinking) variants of Qwen3-VL [1], MiMoVL [34], and Skyra [15]. For all models, we adopt a default frame sampling rate of 5 fps, with all other settings kept unchanged. Detailed experimental configurations are provided in Appendix A.

#### 4.2. Main Results

We evaluate the performance of all models on ArtifactBench and display the accuracy in Table 2. To further analyze the preference alignment and performance gap between MLLMs and humans, we additionally invite four human experts to manually evaluate the benchmark.

The experimental results reveal significant limitations of current MLLMs in artifact detection and identification scenarios. Even Gemini 3.1 Pro achieves only an overall score of 47.5 on Artifact-Bench, despite being the best-performing model. It is worth noting that RVAC and PVRC are both binary decision tasks: RVAC requires a “Yes” or “No” answer, while PVRC requires selecting either “<Video A>” or “<Video B>”. Thus, random guessing yields approximately 50% accuracy. However, most MLLMs still fail to consistently surpass this baseline, especially at higher difficulty levels, indicating their limited ability to reliably recognize and compare realism-related artifacts in AI-generated videos.

Existing MLLMs perform poorly on the AID task. AID is substantially more challenging than RVAC and PVRC: instead of making a binary decision, models must select all

- Table 2. Evaluation results on Artifact-Bench. RVAC (Real vs. AI-generated Video Classification), PVRC (Pairwise Video Realism Comparison), and AID (Artifact Identification) denote the 3 tasks in our benchmark. Each task contains 3 difficulty levels (L1–L3). Avg denotes the average accuracy across the 3 difficulty levels. Total denotes the overall score across all tasks. The highest accuracy of each task (except Human Baseline) is highlighted in green .

Task 1: RVAC Task 2: PVRC Task 3: AID

Model

Total L1 L2 L3 Avg L1 L2 L3 Avg L1 L2 L3 Avg

Proprietary Models

Gemini 3.1 Pro 68.4 76.5 77.2 74.0 45.6 52.9 47.4 48.6 19.3 6.4 3.8 9.8 47.5 Gemini 3 Flash 60.8 71.8 61.4 64.7 48.0 57.5 47.4 50.9 8.6 9.6 11.3 9.8 43.8

###### Open-Source General-Purpose Models

Qwen3-VL 8B-Instruct 48.4 63.8 36.6 49.6 51.2 49.4 36.8 45.8 11.4 3.2 1.9 5.5 36.0 Qwen3-VL 8B-Thinking 48.0 63.8 34.7 48.8 39.2 36.8 34.2 36.7 10.0 3.8 3.8 5.9 33.3 Qwen3-VL 30B-A3B-Instruct 48.0 63.1 35.6 48.9 50.4 41.4 42.1 44.6 14.3 2.5 3.8 6.9 35.5 Qwen3-VL 30B-A3B-Thinking 48.4 63.8 35.6 49.3 47.2 50.6 36.8 44.9 17.1 2.5 3.8 7.8 36.3 Qwen3-VL 32B-Instruct 54.8 63.1 42.6 53.5 53.6 54.0 44.7 50.8 15.0 5.1 1.9 7.3 39.5 Qwen3-VL 32B-Thinking 50.4 63.8 38.6 50.9 48.0 41.4 42.1 43.8 18.6 6.4 3.8 9.6 37.3 InternVL3.5 8B 47.2 61.7 35.6 48.2 48.8 46.0 44.7 46.5 7.1 2.5 1.9 3.9 34.5 InternVL3.5 30B-A3B 47.6 62.4 35.6 48.6 44.8 41.4 23.7 36.6 12.1 2.5 3.8 6.2 33.8 InternVL3.5 38B 48.0 61.1 35.6 48.2 52.8 39.1 36.8 42.9 12.1 2.5 0.0 4.9 34.7 Molmo2 8B 46.8 62.4 35.6 48.3 43.2 42.5 34.2 40.0 10.0 7.6 5.7 7.8 34.5

MiMo-VL 7B-SFT 48.8 61.7 38.6 49.7 52.0 44.8 52.6 49.8 5.7 2.5 0.0 2.8 35.4 MiMo-VL 7B-RL 50.4 61.1 38.6 50.0 42.4 48.3 50.0 46.9 12.1 2.5 3.8 6.2 35.7 Keye-VL-1.5 8B 48.8 61.7 35.6 48.7 48.8 37.9 47.4 44.7 5.0 1.3 1.9 2.7 33.8

###### Open-Source Specialized Models

Skyra 7B-SFT 47.2 63.8 36.6 49.2 19.2 23.0 21.1 21.1 10.0 3.2 3.8 5.7 29.4 Skyra 7B-RL 51.2 62.4 40.6 51.4 31.2 27.6 18.4 25.7 8.6 3.2 5.7 5.8 32.0 BusterX++ 7B 54.0 58.4 43.6 52.0 48.8 47.1 31.6 42.5 7.1 3.2 5.7 5.3 36.2

VideoVeritas 8B 62.8 72.5 69.3 68.2 60.8 56.3 42.1 53.1 16.4 3.2 3.8 7.8 46.0

###### Human Baseline

Human Expert 95.6 92.6 90.1 93.6 88.0 86.2 81.6 86.4 82.9 79.0 77.4 80.3 87.7

valid artifact categories from six candidates, with multiple correct answers possible. Almost all models exhibit a dramatic performance drop on AID, with all models achieving less than 10% average accuracy. These results suggest that although current MLLMs can partially recognize unrealistic videos at a coarse-grained level, they still struggle to explicitly analyze and explain the underlying causes of visual unrealism in AI-generated videos.

A clear performance gap exists between proprietary and open-source models. Overall, proprietary models consistently achieve stronger performance across all three tasks, indicating more robust capabilities in recognizing and reasoning about realism-related artifacts in AI-generated videos. However, despite their advantages, even the strongest proprietary models still exhibit a substantial gap compared with human experts. This result highlights the fundamental difficulty of artifact-aware video reasoning and suggests that current MLLMs remain far from reliably understanding the underlying causes of visual unrealism in AI-generated videos.

#### 4.3. Analysis and Findings

Fine-grained and temporal-spatial perception remain critical bottlenecks. Figure 5 presents two representative failure cases for MLLMs. In Figure 5 (a), the artifact appears only in a small localized region, requiring fine-grained visual

perception for accurate identification. In Figure 5 (b), the artifact is distributed across multiple frames, making temporalspatial perception necessary for detection. These failure cases reveal fundamental limitations of current MLLMs in capturing subtle perceptual inconsistencies in AI-generated videos. For localized artifacts, the abnormal region often occupies only a very small portion of the frame and may be easily suppressed during visual token compression or global feature aggregation. As a result, models tend to focus on dominant semantic content while overlooking fine-grained structural abnormalities. Meanwhile, temporal-spatial artifacts are inherently more challenging because they cannot be identified from isolated frames alone. Detecting such inconsistencies requires models to jointly reason over object dynamics, motion continuity, and cross-frame structural consistency across long temporal contexts. However, current MLLMs often rely on sparse frame sampling and coarse temporal modeling, limiting their ability to capture subtle temporal evolution patterns. These observations suggest that reliable artifact-aware evaluation not only requires stronger semantic reasoning, but also demands substantially improved fine-grained perception and temporal-spatial modeling capabilities specifically tailored for generative artifact understanding.

##### Scaling model size or enabling explicit reasoning does

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

- Figure 5. Failure cases requiring fine-grained and temporal-spatial perception. (a) The paddle penetrates the boat hull, while this artifact occupies only a small portion of the entire image; (b) the football changes from two balls to one and then back to two, requiring multi-frame object tracking.

not necessarily improve artifact detection capability. For example, InternVL3.5-38B performs comparably to its 8B counterpart, while several thinking-enabled variants even underperform their instruction-tuned counterparts. This observation suggests that artifact detection and realism evaluation require capabilities beyond general semantic understanding and chain-of-thought reasoning. Unlike conventional multimodal reasoning tasks that primarily rely on high-level semantics or world knowledge, artifact-aware evaluation demands fine-grained perceptual sensitivity to subtle spatialtemporal inconsistencies, structural distortions, and abnormal motion patterns. Merely scaling model parameters or introducing generic reasoning processes may improve linguistic coherence and abstract reasoning ability, but does not necessarily enhance the model’s ability to faithfully perceive artifacts

Existing MLLMs exhibit a substantial mismatch between their artifact perception and human perceptual preferences. As the task difficulty progressively increases from L1 to L3, human performance consistently declines across all tasks, reflecting the increasing realism and perceptual ambiguity of the AI-generated videos. In contrast, the performance of current MLLMs often fluctuates irregularly or remains nearly unchanged across difficulty levels, rather than exhibiting a monotonic degradation trend aligned with human perception. In some cases, models even achieve comparable or higher accuracy on more challenging subsets. These observations suggest that current MLLMs do not reliably base their judgments on genuine artifact-aware perception. Instead, they may overly rely on superficial semantic cues, dataset biases, or shortcut correlations that are weakly related to perceptual realism itself. This inconsistency reveals a fundamental limitation of current MLLMs in understanding video realism and generative artifacts. Although some models can partially distinguish AI-generated content under relatively easy settings, they fail to demonstrate stable human-aligned perceptual sensitivity as real-

ism increases. Such misalignment substantially limits the reliability of MLLMs as general-purpose evaluators for AIgenerated videos, particularly in applications requiring finegrained realism assessment and artifact diagnosis. More importantly, this issue may hinder the use of MLLMs as reward providers or automated judges for optimizing video generative models. Since reinforcement learning or preference optimization pipelines critically depend on stable and human-aligned reward signals, inaccurate artifact perception could encourage models to optimize toward superficial statistical patterns rather than genuinely improving perceptual realism. Our findings therefore highlight the urgent need for future MLLMs with stronger human-aligned artifact perception, temporal consistency understanding, and fine-grained realism reasoning capabilities.

### 5. Conclusion

In this paper, we introduced Artifact-Bench, a benchmark for evaluating whether MLLMs can detect and diagnose artifacts in AI-generated videos. Through a three-level artifact taxonomy and three complementary tasks, Artifact-Bench provides a systematic evaluation from coarse-grained authenticity recognition to fine-grained artifact identification. Extensive experiments show that current MLLMs still struggle with artifact-level perception and reasoning. Moreover, model judgments are not always aligned with human preferences, limiting their reliability as evaluators or reward providers for video generative models. These findings highlight the need for future MLLMs with stronger fine-grained, temporal-spatial, and human-aligned realism understanding.

### References

- [1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 1, 2, 7
- [2] Xuehai Bai, Yang Shi, Yi-Fan Zhang, Xuanyu Zhu, Yuran Wang, Yifan Dai, Xinyu Liu, Yiyan Ji, Xiaoling Gu, and Yuanxing Zhang. Edit-compass & editreward-compass: A unified benchmark for image editing and reward modeling. arXiv preprint arXiv:2605.13062, 2026. 3
- [3] Xinlong Chen, Yue Ding, Weihong Lin, Jingyun Hua, Linli Yao, Yang Shi, Bozhou Li, Yuanxing Zhang, Qiang Liu, Pengfei Wan, et al. Avocado: An audiovisual video captioner driven by temporal orchestration. arXiv preprint arXiv:2510.10395, 2025. 2
- [4] Xinlong Chen, Yuanxing Zhang, Yushuo Guan, Bohan Zeng, Yang Shi, Sihan Yang, Pengfei Wan, Qiang Liu, Liang Wang, and Tieniu Tan. Versavid-r1: A versatile video understanding and reasoning model from question answering to captioning tasks. arXiv e-prints, pages arXiv–2506, 2025. 2
- [5] Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, et al. Opengpt-4o-image: A comprehensive dataset for advanced image generation and editing. arXiv preprint arXiv:2509.24900, 2025. 3
- [6] Zhili Cheng, Yuge Tu, Ran Li, Shiqi Dai, Jinyi Hu, Shengding Hu, Jiahao Li, Yang Shi, Tianyu Yu, Weize Chen, et al. Embodiedeval: Evaluate multimodal llms as embodied agents. arXiv preprint arXiv:2501.11858, 2025. 3
- [7] Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, et al. Molmo2: Open weights and data for vision-language models with video understanding and grounding. arXiv preprint arXiv:2601.10611, 2026. 7
- [8] Google DeepMind. Gemini 3.1 pro. https://deepmind. google/models/model- cards/gemini- 3- 1pro/, 2026. 1, 2, 5, 7
- [9] Google. Veo 3. https://aistudio.google.com/ models/veo-3, 2025. 1, 5
- [10] Google DeepMind. Gemini 3 flash. https://deepmind. google/models/gemini/flash/, 2025. 7
- [11] Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. Ltx-2: Efficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233, 2025. 1, 5
- [12] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025. 2

- [13] Kuaishou AI Team. Kling ai: Video generation model. https://klingai.com/, 2024. 1, 5
- [14] Jieyu Li, Xin Zhang, and Joey Tianyi Zhou. Aegis: Authenticity evaluation benchmark for ai-generated video sequences. In Proceedings of the 33rd ACM International Conference on Multimedia, page 13346 –13353. ACM, 2025. 2, 3
- [15] Yifei Li, Wenzhao Zheng, Yanran Zhang, Runze Sun, Yu Zheng, Lei Chen, Jie Zhou, and Jiwen Lu. Skyra: Aigenerated video detection via grounded artifact reasoning,

2025. 1, 2, 3, 7

- [16] Yuanxin Liu, Rui Zhu, Shuhuai Ren, Jiacong Wang, Haoyuan Guo, Xu Sun, and Lu Jiang. Uve: Are mllms unified evaluators for ai-generated videos? arXiv preprint arXiv:2503.09949, 2025. 2, 3
- [17] OpenAI. Gpt-4o. https://openai.com/index/ hello-gpt-4o/, 2024. 2
- [18] OpenAI. Gpt-4.1. https://platform.openai.com/ docs/models/gpt-4.1, 2025. 1
- [19] Yang Shi, Jiaheng Liu, Yushuo Guan, Zhenhua Wu, Yuanxing Zhang, Zihao Wang, Weihong Lin, Jingyun Hua, Zekun Wang, Xinlong Chen, et al. Mavors: Multi-granularity video representation for multimodal large language model. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 10994–11003, 2025. 1, 2
- [20] Yang Shi, Huanqian Wang, Wulin Xie, Huanyao Zhang, Lijie Zhao, Yi-Fan Zhang, Xinfeng Li, Chaoyou Fu, Zhuoer Wen, Wenting Liu, Zhuoran Zhang, Xinlong Chen, Bohan Zeng, Sihan Yang, Yushuo Guan, Zhang Zhang, Liang Wang, Haoxuan Li, Zhouchen Lin, Yuanxing Zhang, Pengfei Wan, Haotian Wang, and Wenjing Yang. Mme-videoocr: Evaluating ocr-based capabilities of multimodal llms in video scenarios, 2025. 2
- [21] SII-GAIR and Sand.ai. Speed by simplicity: A single-stream architecture for fast audio-video generative foundation model,

2026. 1, 5

- [22] Tingyu Song, Tongyan Hu, Guo Gan, and Yilun Zhao. Vfeval: Evaluating multimodal llms for generating feedback on aigc videos. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21126–21146, 2025. 2, 3
- [23] Hao Tan, Jun Lan, Senyuan Shi, Zichang Tan, Zijian Yu, Huijia Zhu, Weiqiang Wang, Jun Wan, and Zhen Lei. Videoveritas: Ai-generated video detection via perception pretext reinforcement learning. arXiv preprint arXiv:2602.08828, 2026. 1, 7
- [24] Kwai Keye Team. Kwai keye-vl 1.5 technical report, 2025. 7
- [25] Tencent Hunyuan Foundation Model Team. Hunyuanvideo 1.5 technical report, 2025. 1, 5
- [26] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu,

- Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 5
- [27] Fengxiang Wang, Mingshuo Chen, Yueying Li, Di Wang, Haotian Wang, Zonghao Guo, Zefan Wang, Shan Boqi, Long Lan, Yulin Wang, et al. Geollava-8k: scaling remote-sensing multimodal large language models to 8k resolution. Advances in Neural Information Processing Systems, 38:159185– 159218, 2026. 1
- [28] Fengxiang Wang, Mingshuo Chen, Yueying Li, Yajie Yang, Yifan Zhang, Long Lan, Xue Yang, Hongda Sun, Yulin Wang, Di Wang, et al. Geoeyes: On-demand visual focusing for evidence-grounded understanding of ultra-high-resolution remote sensing imagery. arXiv preprint arXiv:2602.14201, 2026.
- [29] Fengxiang Wang, Mingshuo Chen, Yueying Li, Yajie Yang, Yuhao Zhou, Di Wang, Yifan Zhang, Haoyu Wang, Haiyan Zhao, Hongda Sun, et al. Text before vision: Staged knowledge injection matters for agentic rlvr in ultra-highresolution remote sensing understanding. arXiv preprint arXiv:2602.14225, 2026. 1
- [30] Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. Monet: Reasoning in latent visual space beyond images and language,

2025. 3

- [31] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 2, 7
- [32] Haiquan Wen, Yiwei He, Zhenglin Huang, Tianxiao Li, Zihan Yu, Xingru Huang, Lu Qi, Baoyuan Wu, Xiangtai Li, and Guangliang Cheng. Busterx: Mllm-powered ai-generated video forgery detection and explanation. arXiv preprint arXiv:2505.12620, 2025. 2, 3
- [33] Haiquan Wen, Tianxiao Li, Zhenglin Huang, Yiwei He, and Guangliang Cheng. Busterx++: Towards unified cross-modal ai-generated content detection and explanation with mllm. arXiv preprint arXiv:2507.14632, 2025. 3, 7
- [34] LLM-Core-Team Xiaomi. Mimo-vl technical report, 2025. 7
- [35] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025. 2
- [36] YiFan Zhang, Yang Shi, Weichen Yu, Qingsong Wen, Xue Wang, Wenjing Yang, Zhang Zhang, Liang Wang, and Rong Jin. Debiasing multimodal large language models via penalization of language priors. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4232–4241,

2025. 2

- [37] Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, Jianshu Zeng, Wulin Xie, Yang Shi, Huanyu Zhang, Junkang

- Wu, et al. Mm-rlhf: The next step forward in multimodal llm alignment. arXiv preprint arXiv:2502.10391, 2025. 2
- [38] Zhuoran Zhang, Tengyue Wang, Xilin Gong, Yang Shi, Haotian Wang, Di Wang, and Lijie Hu. When modalities conflict: How unimodal reasoning uncertainty governs preference dynamics in mllms. arXiv preprint arXiv:2511.02243, 2025. 3
- [39] Xuanyu Zhu, Yuhao Dong, Rundong Wang, Yang Shi, Zhipeng Wu, Yinlun Peng, YiFan Zhang, Yihang Lou, Yuanxing Zhang, Ziwei Liu, et al. Vtc-bench: Evaluating agentic multimodal models via compositional visual tool chaining. arXiv preprint arXiv:2603.15030, 2026. 3

### A. Experiment Details

#### A.1. Experimental Setup

For all evaluated models, we use a default video sampling rate of FPS= 5 for video input. Due to context window limitations in some models and the high resolution of certain input videos, we additionally apply frame resizing when necessary to ensure feasible inference.

For decoding-related hyperparameters such as temperature, we prioritize the officially recommended settings for each model whenever available. For example, Gemini 3.1 Pro is evaluated using the official recommended configuration with temperature = 1.0 and thinking_level = "high". Otherwise, models are evaluated using greedy decoding by default.

#### A.2. Evaluation Prompt

For reproducibility, we provide the prompt templates used for each task in Artifact-Bench below.

Prompt Template for Task 1: Real vs. AI-Generated Video Classification (RVAC)

<Video>

Determine whether the given video is generated by modern AI-based generative models (AIGC).

Videos created using traditional computer graphics (CG), animation pipelines, game engines, or professional rendering tools should NOT be classified as AIGC, even though they are not captured from the real world. This includes stylized or animated content, as well as game footage.

Such videos may exhibit visual artifacts due to artistic design, rendering limitations, compression, or real-time graphics constraints. However, these artifacts are fundamentally different from those introduced by AI generative models and should NOT be used as evidence for AIGC classification.

Your judgment should be based on the presence of artifacts that are characteristic of AIGC methods, arising from the limitations of generative models.

Typical AIGC artifacts include, but are not limited to:

- - temporal inconsistency (e.g., flickering, unstable details across frames),
- - structural distortions (e.g., warped objects, inconsistent geometry),
- - unnatural motion or dynamics,
- - semantic incoherence (e.g., objects appearing/disappearing or morphing inconsistently),
- - abnormal visual appearance or texture anomalies (e.g., overly smooth, painterly, or “oily” rendering).

Do NOT rely solely on visual style (e.g., animation, rendering style, or game graphics) when making your judgment. Instead, focus on identifying artifact patterns that are indicative of AI-based generation.

If such AIGC-specific artifacts are clearly present, classify the video as "yes". Otherwise, classify it as "no".

Respond with "yes" if the video is AIGC, and "no" otherwise.

- Prompt Template for Task 2: Pairwise Video Realism Comparison (PVRC)

- Video A: <Video A>
- Video B: <Video B>

You are given two videos, <Video A> and <Video B>. Both videos are generated by modern AI-based generative models (AIGC).

Your task is to determine which video exhibits higher perceptual realism, i.e., which video contains fewer, less noticeable, or less severe artifacts introduced by generative models.

The comparison should be based on the presence and severity of AIGC-specific artifacts, rather than overall visual style or aesthetic preference.

Typical AIGC artifacts include, but are not limited to:

- - temporal inconsistency (e.g., flickering, unstable details across frames),
- - structural distortions (e.g., warped objects, inconsistent geometry),
- - unnatural motion or dynamics,
- - semantic incoherence (e.g., objects appearing/disappearing or morphing inconsistently),
- - abnormal visual appearance or texture anomalies (e.g., overly smooth, painterly, or “oily” rendering). When making your decision:
- - Focus on the **severity, frequency, and perceptibility** of such artifacts.
- - A video with fewer and less perceptible artifacts should be considered more realistic.
- - Minor or barely noticeable artifacts are less important than severe or obvious ones. Do NOT base your decision on:
- - stylistic differences,
- - resolution or sharpness alone,
- - color grading or lighting preferences,
- - high-level semantics or scene plausibility.

Do not be biased by video length, scene complexity, or content diversity.

Even if both videos contain artifacts, you must choose the one that is relatively more realistic.

Respond with:

- - "<Video A>" if <Video A> is more realistic,
- - "<Video B>" if <Video B> is more realistic.

Prompt Template for Task 3: Artifact Identification (AID)

<Video>

You are given a video that is generated by a modern AI-based generative model (AIGC).

For the following options, select all AIGCspecific artifacts that are clearly observable in the video.

Option:

- A. [Option A]
- B. [Option B]
- C. [Option C]
- D. [Option D]
- E. [Option E]
- F. [Option F]

Respond using the corresponding letter(s), separated by commas if multiple are selected (e.g., "A", "B,D", "A,C,E").

#### A.3. Answer Extraction Prompt

We use the following prompt with Gemini 3.1 Pro to parse model responses and extract the final answers for accuracy evaluation.

Prompt Template for Task 1: Real vs. AI-Generated Video Classification (RVAC)

You are an answer extraction system.

The original task is to determine whether a video is generated by an AI-based generative model (AIGC):

- - "yes" means the video is AI-generated.
- - "no" means the video is not AI-generated.

You are given a model response that may contain lengthy reasoning, analysis, self-corrections, or <think>...</think> blocks.

Your task is to extract only the model’s final intended answer.

Model Response: [Response]

Rules:

- 1. Ignore all reasoning, explanations, and intermediate analysis.
- 2. Focus only on the final conclusion.
- 3. The only valid outputs are:

- - "yes"
- - "no"

- 4. Output exactly one valid answer without any additional text or explanation.
- 5. If the response does not contain a clear final answer indicating whether the video is AI-generated, output: "Invalid"

Prompt Template for Task 2: Pairwise Video Realism Comparison (PVRC)

You are an answer extraction system.

The original task is to compare two AI-generated videos, <Video A> and <Video B>, and determine which video has higher perceptual realism, i.e., which video contains fewer or less severe AIGCspecific artifacts.

The only valid answers are:

- - "<Video A>"
- - "<Video B>"

You are given a model response that may contain lengthy reasoning, analysis, self-corrections, or <think>...</think> blocks.

Your task is to extract only the model’s final intended answer.

Model Response: [Response]

Rules:

- 1. Ignore all reasoning, explanations, and intermediate analysis.
- 2. Focus only on the final conclusion.
- 3. Output exactly one of the following:

- - "<Video A>"
- - "<Video B>"

- 4. Do not output any additional text or explanation.
- 5. If the response does not contain a clear final answer indicating which video is more realistic, output: "Invalid"

- Prompt Template for Task 3: Artifact Identification (AID) You are an answer extraction system.

The original task is to identify all AIGC-specific artifacts that are clearly observable in a given AI-generated video.

The candidate options are multiple-choice options labeled with letters (e.g., A-F). Multiple options may be correct.

You are given a model response that may contain lengthy reasoning, analysis, self-corrections, or <think>...</think> blocks.

Your task is to extract only the model’s final intended answer.

Model Response: [Response]

Rules:

- 1. Ignore all reasoning, explanations, and intermediate analysis.
- 2. Focus only on the final conclusion.
- 3. Extract only the selected option letters.
- 4. If multiple options are selected, output them separated by commas (e.g., "A,C,E").
- 5. Do not output any additional text or explanation.
- 6. Only output valid option letters that appear in the final answer.
- 7. If the response does not contain a clear final answer, output: "Invalid"

### B. Benchmark Details

- B.1. Representative Examples from Artifact-Bench

In order to comprehensively convey the characteristics of tasks in Artifact-Bench, two representative examples are presented for each task, as illustrated in Figures 6, 7, and 8.

- B.2. Task Distribution

Table 3 shows the number of QA pairs of each difficulty level of each task in Artifact-Bench.

### C. Limitations

Despite our efforts, Artifact-Bench still has limitations. Due to resource constraints, the number of human experts and the dataset scale can be further expanded. Future work will enlarge the benchmark with more diverse video sources,

Table 3. Number of QA Pairs per task in RealVideo-Bench.

Task Category Difficulty Level

#QA

- Task 1: Real vs. AI-Generated Video Classification (RVAC)

L1 250 L2 149 L3 101 Total 500

- Task 2: Pairwise Video Realism Comparison (PVRC)

L1 125 L2 87 L3 38 Total 250

- Task 3: Artifact Identification (AID)

L1 140 L2 157 L3 53 Total 350

Total - 1,100

artifact types, and expert annotations, enabling more comprehensive and reliable evaluation of artifact-aware video understanding.

### D. Compute Resources

All experiments were conducted on a distributed setup consisting of four identical machines, each equipped with 8 NVIDIA H800 GPUs and 1000 GiB of system memory. No additional compute beyond the reported experiments (excluding preliminary runs) is required to reproduce the main results.

### E. Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

###### Figure 6. Representative examples for Task 1: Real vs. AI-Generated Video Classification (RVAC).

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

###### Figure 7. Representative examples for Task 2: Pairwise Video Realism Comparison (PVRC).

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

###### Figure 8. Representative examples for Task 3: Artifact Identification (AID).

