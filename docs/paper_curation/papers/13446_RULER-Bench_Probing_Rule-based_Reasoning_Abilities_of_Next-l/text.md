## RULER-Bench: Probing Rule-based Reasoning Abilities of Next-level Video Generation Models for Vision Foundation Intelligence

#### Xuming He1* Zehao Fan1* Hengjia Li1*† Fan Zhuo1 Hankun Xu1 Senlin Cheng2 Di Weng1 Haifeng Liu1 Can Ye2 Boxi Wu1 1Zhejiang University 2Ant Group

[Figure 1]

###### https://hexmseeu.github.io/RULER-Bench-proj/

# arXiv:2512.02622v1[cs.CV]2Dec2025

[Figure 2]

###### Human Alignment

Prompt Suite

###### Drop a small, freshly cut piece of metallic sodium into the beaker containing a clear mixture of water and phenolphthalein solution.

grok-4

68.2

[Figure 3]

Consistency

Gemini2.5

72.08

GPT-4o

72.82

Seed-1.6

76.32

Upon contact with water, sodium reacts vigorously, producing hydrogen gas and forming sodium hydroxide. The metal rapidly moves across the water surface, melting into a bright, silvery sphere that darts around before disappearing. The reaction generates heat and releases ...

[Figure 4]

GPT-5

79.95

GPT-o3

85.12

[Figure 5]

[Figure 6]

Evaluation Protocol

[Figure 7]

[Figure 8]

Immediately after the sodium contacts the liquid, are bubbles produced at the contact area on the water surface?

Generation Models

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Medium.

[Figure 15]

[Figure 16]

Do the beaker, and background remain geometrically consistent throughout the video without sudden scene changes?

Seedance 1.0-pro

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Good.

Veo3.1

[Figure 25]

Is the video sharp enough to clearly distinguish small bubbles without blur, heavy noise, or ghosting artifacts?

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Good.

[Figure 33]

Wan2.5

[Figure 34]

Is only one piece of sodium added to the beaker without introducing any other reagents or solids?

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Sora2

Good.

Figure 1. Overview of RULER-Bench. We propose RULER-Bench, a comprehensive benchmark designed to evaluate the rule-based reasoning abilities of video generation models. Left: Grounded in three fundamental domains, we formulate rule-based reasoning ability into six categories: Science, Vision, Hypothesis, Game, Semantics, and Humanity. These categories are further subdivided into 40 tasks. Center: Using the collected samples, we evaluate 10 video models based on the corresponding checklist across four metrics. Each checklist question is scored by GPT-o3 with discrete labels. To validate the reliability of the evaluator, we conduct a human alignment study, in which GPT-o3 achieves 85% agreement with human judgments. Right: Extensive experiments demonstrate that Veo3.1 achieves the best performance. However, all models exhibit limited reasoning ability across different rule categories.

#### Abstract

video generation models from the perspective of cognitive rules. Built upon two fundamental paradigms: text-to-video and image-to-video, RULER-Bench covers 40 representative tasks spanning six rule categories with 622 high-quality annotated instances. For the evaluation of each generated video, we construct a checklist covering four metrics and leverage GPT-o3 to assign scores to each question, achieving 85% alignment with human judgements. Extensive experiments show that the state-of-the-art model achieves only 48.87% on the rule coherence metric, highlighting significant room for improvement in the reasoning capability of next-level video models. We expect that the insight obtained from RULER-Bench will facilitate further development of reasoning-aware video generation, advancing video generation models toward vision foundation intelligence.

Recent advances in video generation have enabled the synthesis of videos with strong temporal consistency and impressive visual quality, marking a crucial step toward vision foundation models. To evaluate these video generation models, existing benchmarks primarily focus on factors related to visual perception and understanding, like visual aesthetics, instruction adherence, and temporal coherence. However, the rule-based reasoning capabilities of video generation models remain largely unexplored. Although recent studies have carried out preliminary explorations into whether video models can serve as zero-shot learners, they still lack a fine-grained decomposition of reasoning capabilities and a comprehensive evaluation protocol. To address this gap, we introduce RULER-Bench, a benchmark designed to evaluate the reasoning ability of

#### 1. Introduction

Video generation focuses on producing video clips that exhibit strong temporal consistency and high visual quality,

*equal contributions. †project lead.

serving as a fundamental technique for downstream applications such as customization [43, 44, 75, 85] and world modeling [1, 17, 74]. Empowered by advances in generative frameworks such as diffusion models [8, 23, 32, 59, 61, 88] and autoregressive [40, 66, 67] approaches, recent video generation systems have achieved remarkable progress on perceptual and understanding abilities, such as aesthetic quality and instruction adherence. However, with the rapid scaling of high-quality datasets and model parameters, state-of-the-art models such as Sora2 [58], Veo3 [26], and Wan2.5 [2] have nearly saturated on these dimensions.

Mirroring the evolution of foundation language models [5, 7, 25, 27, 68, 83] in natural language processing, video models are expected to gradually advance from perception and understanding to reasoning, thereby paving the way for ultimately becoming the foundation model for vision. Fig. 1 illustrates the current stage of this evolution. Given the instruction “drop a piece of sodium into the phenolphthalein solution” and an input image, all four generation models produce visually coherent video clips. However, their reasoning abilities differ significantly. For instance, Veo3.1 [26] accurately infers that sodium reacts violently with phenolphthalein solution, releasing gas and forming an alkaline product. Therefore, the generated video depicts vigorous bubbling and the distinctive red coloration. In contrast, Seedance1.0-pro [24], Wan2.5, and Sora2 exhibit limited reasoning ability, capturing only a subset of the expected reaction phenomena. This capability gap reveals a fundamental challenge: How to assess whether nextlevel video generation models possess the reasoning abilities necessary to achieve vision foundation intelligence?

Recently, Wiedemer et al. [76] have conducted an initial exploration into whether Veo3 could serve as a vision foundation model. They constructed instances spanning four aspects: perception, modeling, manipulation, and reasoning, and then measured Veo3’s success rates on each dimension. However, their exploration lacks a fine-grained decomposition of reasoning characteristics and a systematic evaluation framework. Additionally, VBench-2.0 [94] and PhyGenbench [53] target intrinsic faithfulness and physical laws, but remain limited to physics and commonsense reasoning, overlooking the diversity of reasoning dimensions.

To bridge this gap, we conceptualize reasoning as cognitive rule-based prediction, the ability of a generative model to infer rules from inputs and apply them to predict plausible phenomena in videos. Building on this formulation, we organize reasoning scenarios into three fundamental domains: Nature, Society, and Virtuality, which correspond to real-world, human-centered, and virtual environments, respectively. Within these domains, we further define six categories of cognitive rules: Vision, Science, Semantics, Hypothesis, Game, and Humanity, which collectively span diverse reasoning scenarios in video generation.

Based on these categories, we propose RULER-Bench (RULE-based Reasoning Benchmark for video generation), a comprehensive benchmark designed to evaluate the reasoning capabilities of next-level video generation models in two typical generation scenarios: text-to-video and image-to-video. RULER-Bench adopts a hierarchical paradigm, organizing 40 tasks within six rule categories. To ensure evaluation reliability, RULER-Bench comprises 622 high-quality instances, evenly distributed across tasks.

Furthermore, leveraging multimodal large language models (MLLMs), we introduce a comprehensive evaluation protocol that assesses generated videos across four metrics: Instruction Following, Visual Consistency, Visual Fidelity, and Rule Coherence. Unlike subjective continuous scoring, we construct a checklist of objective questions for each instance based on these four metrics. Each question is then rated on a discrete three-point scale: good, medium, or bad. To validate the reliability of our evaluation protocol, we manually annotate 813 checklist questions based on generated videos and verify the consistency between MLLM responses and human judgments. As shown in Fig. 1, our evaluation protocol achieves alignment with human annotation on 85% of the checklist questions, which confirms its effectiveness. Building upon RULER-Bench, we further conduct a systematic evaluation of 10 state-of-the-art video generation models. Extensive experiments show that all models consistently exhibit limitations in rule coherence, demonstrating substantial potential for enhancing the reasoning capability of next-level video models.

The main contributions of this work are:

- • We conceptualize reasoning in video generation as cognitive rule-based prediction and formulate the first taxonomy that organizes reasoning into six rule categories.
- • We introduce RULER-Bench, a comprehensive benchmark specifically designed to evaluate the reasoning abilities of video generation models, and systematically covering 40 tasks and 622 high-quality instances.
- • We conduct extensive experiments on 10 state-of-the-art video generation models, revealing substantial limitations across all models across different rule types.

#### 2. Related Work

Video Generation Models. Recent advancements in diffusion models [18, 32–34, 55, 62, 63, 88, 91, 92] and autoregressive approaches [10, 35, 40, 46, 67, 80] have led to rapid progress in video generation [8, 28, 30, 31, 39, 47, 48, 52, 73, 77]. By leveraging large-scale, high-quality training data and expanding model capacity, recent systems such as Sora2, Veo3, and Wan have achieved remarkable performance in the dimensions of perception and understanding, including visual consistency, aesthetic quality, and instruction adherence. As visual fidelity continues to improve, research attention has begun to shift focus towards exploring

reasoning capabilities [1, 12, 22, 49, 54, 72, 81, 82, 89, 90], such as understanding physical laws and performing logical inference. NewtonGen [86] introduces Neural Newtonian Dynamics to model and predict Newtonian motions, while V-Chain [37] incorporates a chain-of-visual-thought mechanism to inject visual reasoning signals into generative processes. However, existing benchmarks lack a systematic framework to assess the reasoning abilities of video models. To address this challenge, RULER-Bench provides a comprehensive benchmark for evaluating the emerging capabilities of video generation models for rule-based reasoning.

Benchmarks for Video Generation Methods. To effectively evaluate the capabilities of video generation models, a variety of benchmarks have been proposed. VBench [36], EvalCrafter [51], and FetV [50] primarily focus on Textto-Video (T2V) tasks, assessing fundamental technical attributes. In contrast, T2V-CompBench [64], PhyGenbench [53], and VBench-2.0 [94] focus on deeper principles such as compositionality, physical law, and intrinsic faithfulness. For Image-to-Video (I2V) tasks, benchmarks such as AIGC-Bench [19] and AnimateBench [19] evaluate the video generation models along perceptual dimensions, including instruction following and visual consistency, while UI2V-Bench [87] evaluates models from a general understanding perspective. Regarding Video-to-Video (V2V) generation, VE-Bench [65] and EditBoard [13] cover general editing scenarios, including subject, style, and attribute editing, whereas IVE-Bench [14] integrates traditional metrics with large language model-based assessments across multiple editing categories. However, as shown in Tab. 1, these benchmarks are typically constrained to a single task type and primarily concentrate on fundamental abilities or specific emerging capabilities. Systematic evaluation of reasoning dimensions remains largely unexplored, with insufficient evaluation data [29] and a lack of formal rulebased frameworks [76]. RULER-Bench unifies two task paradigms: T2V and I2V, and provides a comprehensive evaluation of rule-based reasoning in video models, addressing limitations in existing evaluation practices and advancing the field toward vision foundation models.

#### 3. RULER-Bench

###### 3.1. Rule-Based Task Formulation

Inspired by KRIS-Bench [78], a benchmark that evaluates image editing models through the lens of knowledge, we conceptualize reasoning in video generation as cognitive rule-based prediction. Specifically, video models are required to infer implicit rules from the given input, predict possible outcomes, and present them through the generated video. Grounded in the three fundamental domains of the world: Nature, Society, and Virtuality, we define six rule categories that collectively characterize diverse and com-

Table 1. Comparison of open-source video generation benchmarks. ✓✗ represents insufficient reasoning dimensions.

Benchmark Size Categories Tasks Type Reason Rule

FetV [50] 619 3 22 T2V ✗ ✗ EvalCrafter [51] 700 4 12 T2V ✗ ✗ TC-Bench [20] 270 - 3 T/I2V ✗ ✗ VBench-2.0 [94] 1260 5 18 T2V ✓✗ ✗ VideoPhy [6] 688 3 6 T2V ✓✗ ✗ UI2V-Bench [87] 500 4 21 I2V ✓✗ ✗ MME-CoF [29] 59 - 12 T/I2V ✓ ✗

RULER-Bench 622 6 40 T/I2V ✓ ✓

plex reasoning scenarios in video generation.

Natural Domain requires video generation models to perform reasoning grounded in real-world rules, thereby generating videos with plausible visual phenomena. The rule categories in this domain include:

- • Vision. The ability of video models to reason about realworld visual compositions, such as appearance variations, spatial arrangements, and dynamic transformations.
- • Science. The ability of models to reason about scientific phenomena by leveraging the underlying laws of nature.

Social Domain challenges video models to reason grounded in rules derived from human society, generating videos exhibiting socially consistent behaviors and interactions. The rule categories in this domain encompass:

- • Humanity. The ability of video models to infer human behaviors by leveraging principles of social dynamics, such as sports rules, traditional customs, and festivals.
- • Semantics. The ability of video models to reason about authentic expressions through semantic regularities.

Virtual Domain requires video models to perform reasoning grounded in rules of virtual environments, generating videos that follow the internal logic of these environments. The rule categories within this domain include:

- • Game. The ability of video models to perform rational actions according to game rules to win matches or complete levels, for example, executing a checkmate in chess.
- • Hypothesis. The ability of video models to reason about and predict phenomena based on hypothetical premises.

For each rule category, we carefully design distinct tasks to enable a finer-grained decomposition. As shown in Fig. 1, RULER-Bench encompasses a total of 40 tasks, with detailed descriptions provided in the Appendix.

###### 3.2. Data Construction

RULER-Bench encompasses two task paradigms: Textto-Video (T2V) and Image-to-Video (I2V). To construct a benchmark with high-quality and diverse coverage, we curate the data through a hybrid process that integrates human annotation with GPT-5 [56] generation. In this section, we provide a detailed elaboration on the data construction procedures for each task paradigm. An overview of the data construction pipeline is illustrated in Part B of Fig. 2.

A. Task Formulation B1. Dataset Collection B2. Scenario-to-image Pipeline

Reason Oriented

[Figure 42]

Please generate task-specific image based on the prompt.

Task Categories

[Figure 43]

###### Curated Scenarios

[Figure 44]

[Figure 45]

[Figure 46]

A small metallic cart rests at the ...

Rule Based

Seed Sample Collection

A transparent container is filled ...

[Figure 47]

Generation model

A clear conical flask sits on a ...

Task Diversity

[Figure 48]

T2V I2V

A street intersection with a ...

###### Task Requirements

Task Validation

Human Collection

MLLMs

Web Data

B3. Image-to-scenario Pipeline

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Please generate task-specific prompt based on the image.

B3

[Figure 56]

Vision

Semantics Science

B2

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

MLLMs

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Rule-based Samples

Hypothesis Humanity

Game

Formatted structure

A live housefly is released in front of the chameleon.

Prompts Implicit Explanations Initial Scenarios

Rule Categories

C. Checklist Construction D1. Dataset Quality Control D2. Checklist Quality Control

Instruction Following Rule Coherence Visual Consistency Visual Fidelity

[Figure 70]

[Figure 71]

###### Full Checklist Validation Rule Document Drafting Small Batch Testing Researchers Annotators

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

We need rigorous quality control !

###### Embeddng

###### SmartySefConsstency

Other Prompts

Prvacy&Ethcs

[Figure 77]

###### Samples Checklist

[Figure 78]

Task Prompt

Annotation Rule

Propose

###### Add several drops of the iodine solution into the starch suspension.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

###### Please generate quality analysis questions based on the provided 4 criterias.

[Figure 83]

[Figure 84]

[Figure 85]

Human Refinement

[Figure 86]

[Figure 87]

[Figure 88]

Iodine fits into the central cavity of the amylose helix... The mixture immediately develops a deep blue-black coloration, indicating the presence of starch ...

[Figure 89]

[Figure 90]

Task Diversity Rule Correctness Typical Scenario

[Figure 91]

[Figure 92]

[Figure 93]

###### +

General Checklist

[Figure 94]

Correctness Irrelevant Removal

Ambiguity Elimination

[Figure 95]

[Figure 96]

Pa d

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

ding

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

and only support fixed aspect ratio. If an image does not match the fixed aspect ratio, it will be cropped or stretched.

[Figure 106]

[Figure 107]

[Figure 108]

We currently do not support uploads of images containing photorealistic people.

Curated Checklist

Positive Framing

Key-value Consistency

Completeness

Sample-Specific Checklist

16:9

Figure 2. Overview of dataset construction and validation. First, we formulate our tasks based on the six rule categories. Second, we design task-specific data construction pipelines for T2V and I2V tasks. Third, we leverage MLLM to construct checklist questions across four evaluation metrics. Finally, we conduct quality control and data refinement for the constructed dataset and checklists.

Seed sample collection. We first manually curate seed samples for each task to provide GPT-5 with in-context guidance, which enables it to better comprehend task requirements and thereby generate more reliable samples.

prompt and implicit explanation based on the image content and task requirements. However, for samples in the Game category, MLLMs often struggle to generate implicit explanations that correctly reflect the underlying game strategies. To address this limitation, we collect ground truth images representing the final frame of the expected outcome, and manually create the corresponding implicit explanations to accurately capture the underlying rules.

T2V data collection. For each T2V instance, we first define a tuple (prompt, implicit explanation), where the prompt serves as the sole input to the video generation models, and the implicit explanation specifies the expected outcome along with the underlying reasoning principles. Second, we provide GPT-5 with detailed task requirements and a set of seed samples, and instruct it to generate new samples that strictly adhere to the defined criteria, thereby ensuring both diversity and consistency in the dataset.

For samples containing generated images, we define a triplet (initial scenario, prompt, implicit explanation), with the initial scenario serving as the input to the text-to-image (T2I) model. To ensure consistency, we instruct GPT-5 to leverage the initial scenario as the starting conditions when generating the prompt. Additionally, to maintain high visual quality, GPT-5 is guided to design initial scenarios that prevent the T2I model from generating images containing textual content, which remains challenging in current T2I systems. We employ the powerful nano banana [21] for image generation, providing both the generated image and the corresponding prompt as inputs to the I2V tasks.

I2V data collection. Traditional image collection for generative model benchmarks relies on web-sourced images [78, 95]. However, these images are largely confined to generic scenes, such as animals or cultural landscapes, while providing minimal coverage of domain-specific content. With the recent advances in image generation, stateof-the-art models [11, 16, 42] can generate richly detailed images based on highly customizable prompts. Therefore, we collect I2V data by integrating web-sourced and generated images based on the characteristics of each task.

###### 3.3. Quality Control and Data Refinement

To ensure high data quality, we design a meticulous quality control pipeline, illustrated in Part D1 of Fig. 2. First, we use a robust embedding model to encode all prompts into text embeddings. Second, we compute cosine similarity be-

For instances containing web images, we adopt the same procedure as in T2V by defining a tuple (prompt, implicit explanation). For each image, GPT-5 is used to design the

tween prompts in the feature space. Highly similar prompts are deduplicated. To further ensure content fidelity, we instruct MLLM to verify consistency among the prompt, implicit explanation, and the corresponding input image. We then perform human refinement on instances that are deduplicated and self-consistent, with each sample carefully validated for rule correctness, task diversity, and scenario typicality. Finally, we pre-generate the verified samples using Sora2 to filter out instances subject to privacy or ethical constraints. During generation, we observe that Sora2 generates videos with fixed aspect ratios (16:9 or 9:16), which may cause stretching or cropping. To prevent information loss, we pad each input to match the target aspect ratio while preserving all critical visual content.

Does the video depict a resident holding a banana peel and dropping it into a bin as instructed by the prompt?

Good.

Instruction Following

Does the banana peel visibly leave the resident's hand and land inside a bin during the clip?

Good.

Do the bins and the resident maintain consistent positions and appearance throughout the video without sudden changes ?

Medium.

GPTo3Generaton

Visual Consistency

Is the banana peel's trajectory continuous from the resident's hand into the bin without teleportation, or abrupt motion changes?

[Figure 109]

[Figure 110]

[Figure 111]

Good.

Scores

Is the video clear enough to distinguish the banana peel, the resident, and the bin, with stable colors and no noticeable visual artifacts?

Good.

Visual Fidelity

Good.

Does the video show noticeable changes over time?

Is the scene and structure in the video reasonable?

Good.

Does the resident choose the wet waste bin for the banana peel, consistent with the waste-sorting rule ?

Rule Coherence

Good.

Figure 3. Evaluation pipeline of RULER-Bench.

for each sample. As shown in Fig. 3, each checklist contains multiple questions derived from the four evaluation metrics, and each question is answered using one of the three options: good, medium, or bad. Compared to traditional scoring methods [64, 78], the checklist protocol provides greater interpretability, allowing evaluators to make more consistent and human-aligned judgements. Given the recent advances in MLLMs of video understanding, we employ o3 [57] as the evaluator to assess the rule-based reasoning ability of video generation models.

#### 4. Evaluation Protocol

###### 4.1. Evaluation Metrics

To comprehensively evaluate the performance of state-ofthe-art video models on RULER-Bench, we consider 3 commonly used metrics: Instruction Following, Visual Consistency, and Visual Fidelity [29, 78, 93]. Additionally, we introduce a novel rule-based dimension, termed Rule Coherence, which assesses whether the generated video adheres to the implicit rules underlying the given prompt. The detailed definitions of the 4 metrics are presented below.

###### 4.2. Checklist Construction and Quality Control

As shown in Fig. 2, we leverage MLLMs to generate evaluation checklists. For each sample, we instruct GPT-5 to generate multiple questions based on the input image, prompt, and implicit explanation, covering the four evaluation metrics. Each question is designed to evaluate the generated video from a distinct perspective, ensuring that diverse aspects of video content are systematically assessed. Detailed instruction prompts are provided in the Appendix.

Instruction Following evaluates whether video models generate videos that faithfully follow the user’s instructions. It focuses on the semantic alignment with the input prompt and is independent of implicit rules. For instance, given the prompt “place a small wooden sphere and an iron sphere into water”, the Instruction Following metric examines whether the balls are placed into water, without considering the subsequent phenomena or physical principles.

To ensure the quality of the checklists, we perform a careful manual verification for each sample. We instruct human annotators to review each checklist question and make revisions based on accuracy, clarity, consistency, and completeness. After manual verification, we finalize 6,500 checklist questions for 622 samples, ensuring comprehensive coverage across all four evaluation metrics.

Visual Consistency evaluates whether video generation models preserve identities of elements that are expected to remain unchanged throughout the generated video. For example, in sports scenes, the Visual Consistency dimension focuses on whether visual attributes, such as stadium color, remain consistent across the entire video.

Visual Fidelity assesses the overall visual quality of the generated video, including whether it is visually clear, stable, and free from noise, artifacts, or distortions.

#### 5. Experiments and Evaluations

General Settings. We evaluate the rule-based reasoning capabilities of 10 state-of-the-art video models on RULERBench, including six closed-source models: Veo3.1 [26], Veo2 [69], Sora2 [58], PixelVerse-V5 [60], Wan2.5 [2], and Seedance1.0-pro [24], and four open-source models: HunyuanVideo [41], CogVideoX1.5-5B [84], Wan2.1-14B [71], and Wan2.2-A14B [71]. Notably, the four open-source models provide separate T2V and I2V versions, whose results are reported as a unified entry in our experiments.

Rule Coherence assesses whether the generated video adheres to scene-specific rules. It requires video models to leverage implicit rules to predict or infer visual phenomena from the given instructions. For example, given the prompt “add extra water to the left arm of a transparent U-shaped tube and observe the water levels in both arms”, generation models are expected to apply the principle of communicating vessels to infer that the final water levels remain equal and depict this phenomenon accurately in the video.

We use the default implementations of these video models, with detailed configurations provided in the Appendix.

Inspired by [45], we design a checklist-based evaluation

Table 2. Evaluation result across different rule categories and metrics, including Instruction Following (IF), Visual Consistency (VC), Visual Fidelity (VF), and Rule Coherence (RC). All models exhibit limited rule-based reasoning ability. The performance of closed-source models and open-source models is separately marked with the best results in bold, and the second best underlined.

Closed-Source Models Open-Source Models Veo3.1 Veo2 Sora2

Rule Categories Metric

Pixel Verse-V5

Hunyuan Video

CogVideoX 1.5 5B

Wan 2.2 A14B

Wan 2.1 14B

Seedance 1.0-pro

Wan 2.5

IF 65.05 42.17 66.00 57.13 57.38 58.86 24.92 27.97 37.15 35.80 VC 83.18 73.3 88.01 80.76 80.48 80.99 48.46 48.84 68.52 65.74 VF 91.37 82.33 89.49 89.74 85.35 87.69 71.29 70.96 80.37 81.93 RC 50.97 22.16 47.09 41.41 33.64 31.96 12.64 13.70 17.16 15.90 Avg 72.64 54.99 72.65 67.26 64.21 64.87 39.33 40.37 50.80 49.84

Science Rule

IF 39.75 24.25 39.19 30.10 26.59 24.26 14.75 22.75 16.29 19.30 VC 51.45 36.33 72.33 67.09 72.71 68.79 40.07 55.29 64.56 37.52 VF 77.95 59.15 88.18 80.59 86.28 88.39 59.13 69.45 80.13 72.20 RC 17.70 8.17 19.97 13.06 15.45 15.61 6.98 7.56 14.12 10.48 Avg 46.71 31.98 54.92 47.71 50.26 49.26 30.23 38.76 43.77 34.88

Game Rule

IF 71.83 56.44 68.12 65.08 59.91 61.28 38.51 46.06 48.77 46.27 VC 92.65 91.18 90.85 91.18 87.33 87.67 80.39 75.82 82.35 80.72 VF 91.62 82.50 83.43 89.02 82.19 84.55 79.17 70.69 82.70 83.09 RC 67.57 44.13 53.69 56.80 49.95 49.42 32.01 37.34 37.73 38.40 Avg 80.92 68.56 74.02 75.52 69.84 70.73 57.52 57.48 62.89 62.12

Semantics Rule

IF 86.97 58.55 72.44 80.13 71.93 64.32 44.44 41.45 61.11 61.75 VC 85.90 64.32 77.35 81.62 66.45 67.74 51.92 50.43 64.74 55.56 VF 92.20 81.54 82.50 85.73 76.86 79.66 73.89 63.8 77.03 75.17 RC 46.79 12.50 41.35 46.69 18.31 28.31 9.62 11.00 12.93 17.84 Avg 77.96 54.23 68.41 73.54 58.39 60.01 44.97 41.67 53.95 52.58

Hypothesis Rule

IF 79.90 53.46 80.04 72.87 63.28 68.93 46.56 42.32 49.76 52.28 VC 87.37 73.10 88.06 84.25 79.83 83.13 70.60 54.23 72.34 70.47 VF 94.49 84.38 88.08 89.65 83.90 88.52 80.94 67.76 83.15 82.32 RC 61.23 35.23 56.78 50.63 33.41 38.75 27.78 20.60 30.21 29.21 Avg 80.75 61.54 78.24 74.35 65.10 69.83 56.47 46.23 58.86 58.57

Humanity Rule

VC 59.53 46.19 57.77 56.14 70.04 61.86 43.49 24.79 59.03 51.26 VF 72.67 57.63 57.77 71.61 68.32 76.06 52.94 29.41 65.55 49.58 RC 48.94 30.58 28.50 40.47 42.24 41.74 18.91 14.78 29.34 23.25 Avg 60.38 44.80 48.02 56.07 60.20 59.89 38.45 22.99 51.31 41.36

Vision Rule

IF 68.70 46.97 65.16 61.06 55.82 55.53 33.84 36.11 42.62 43.08 VC 76.68 64.07 79.06 76.84 76.14 75.03 55.82 51.57 68.59 60.21 VF 86.72 74.59 81.58 84.39 80.48 84.14 69.56 62.01 78.15 74.05 RC 48.87 25.46 41.23 41.51 32.17 34.30 17.99 17.50 23.58 22.51 Avg 70.24 52.77 66.76 65.95 61.15 62.25 44.30 41.80 53.24 49.96

Average

Win Rate 0.397 0.186 0.340 0.300 0.257 0.267 0.151 0.151 0.193 0.162

Metrics. To enable straightforward comparison, we normalized all scores to a 100-point scale. In particular, we map good, medium, and bad ratings to 100, 50, and 0, respectively. For each checklist, the score of each metric is computed as the average rating across all associated questions. To obtain the overall score, we assign equal weights to the four metrics and compute their mean score.

###### 5.1. Main Results

We evaluate the performance of 10 state-of-the-art video generation models across six rule categories under four evaluation metrics. Our quantitative results in Tab. 2 demonstrate that RULER-Bench effectively reveals finegrained differences across reasoning capabilities and eval-

Table 3. Human alignment comparison across different MLLMs.

Model KRCC SRCC PLCC ACC

Claude-haiku-4-5 0.2941 0.5128 0.5199 0.6728 Claude-sonnet-4-5 0.3143 0.5356 0.5442 0.7036

Doubao-seed-1-6 0.1892 0.4827 0.5157 0.7632 Gemini-2.5-flash 0.3436 0.5976 0.6023 0.7368

Gemini-2.5-pro 0.3586 0.6073 0.6112 0.7208 Grok-4 0.3522 0.5504 0.5612 0.6820

- GPT-4o 0.3044 0.5564 0.5615 0.7282

- GPT-5 0.4204 0.7295 0.7289 0.7995 GPT-o3 0.4622 0.8011 0.8042 0.8512

uation metrics. We summarize the key observations below.

Observation 1. Video models exhibit limited performance on the Rule Coherence metric compared to other

[Figure 112]

[Figure 113]

###### Semantics Vision Science Game Humanity Hypothesis

- Figure 4. Average performance of video generation models across different tasks on RULER-Bench. Video models generally perform best on tasks in Humanity and Hypothesis, while showing lower performance on Vision and Game categories.

Table 4. Effect of Prompt Enhancement (PE) on Rule Coherence. (Game category is excluded since most samples rely heavily on ground truth images, thus less affected by PE).

evaluation metrics. As shown in Tab. 2, all video models achieve their lowest performance on the Rule Coherence metric, highlighting the significant challenge of rule-based reasoning for current generation models. Moreover, we observe a consistent decline in scores as the metrics reflect more complex cognitive skills. For instance, Veo3.1 obtains an average score of approximately 80 on the perceptionbased metrics, including Visual Consistency and Visual Fidelity. The score drops to 68.7 on the Instruction Following metric, which evaluates visual understanding, and further decreases to 48.87 on the Rule Coherence metric. In addition, open-source models generally exhibit lower reasoning capabilities compared to closed-source models, suggesting that higher-quality training data and larger model capacity can enhance rule-based reasoning ability.

Method Science Vision Semantics Hypothesis Humanity

Veo3.1 50.97 48.94 67.57 46.79 61.23 Veo3.1+PE 62.62 52.61 72.19 58.12 67.40

△ +9.65 +3.67 +4.62 +11.33 +6.17 Sora2 47.09 28.50 53.69 41.35 56.78

Sora2+PE 55.45 36.06 63.37 63.35 65.15 △ +8.36 +7.56 +9.68 +22.0 +8.37

based reasoning, the models also exhibit insufficient reasoning of image content, which further constrains their capability to generate high-quality videos. To obtain deeper insight into this limitation, we analyze the result across task types and find that video generation models achieve an average score of 65.55 on T2V tasks, significantly higher than 48.56 on I2V tasks, indicating that an increase in input modalities leads to a sharp decline in reasoning performance.

###### Observation 2. Video generation models exhibit di-

verse performance patterns across different categories. As illustrated in Tab. 2, video models achieve the highest scores on the Humanity and Semantics categories, with Veo3.1 reaching an average of 80. This indicates that current models exhibit strong generative competence in the social domain, demonstrating a solid understanding of human customs, language, and behaviors. In contrast, all models achieve an average Rule Coherence score below 15 on the Game category, suggesting limited generalization ability in customized, strategy-driven scenarios. As shown in Fig. 4, we further analyze the average performance of different video models across tasks. We observe that models also exhibit considerable intra-category variations in reasoning ability. For example, under the Humanity category, video models generally achieve higher scores on the dress task than on other tasks, reaching an average of 84.9, which can be attributed to the prevalence of clothing-related cues in human-centric scenes, thus enabling video models to generalize dress-related rules through training.

###### 5.2. Analysis

Impact on Prompt Enhancement. To analyze the impact of prompt enhancement at test times, we use GPT-o3 to generate an enhanced prompt based on the prompt and implicit explanation of each sample, explicitly incorporating the expected outcomes. As shown in Tab. 4, prompt enhancement substantially improves the performance of Veo3.1 and Sora2 on Rule Coherence. However, these results also suggest that even with explicit guidance on the expected phenomena, current video models still struggle to generate ruleconsistent video clips based on their reasoning abilities, revealing a remaining gap toward zero-shot reasoners.

Case study. Fig. 5 presents qualitative case studies of video models’ performance on RULER-Bench across different rule categories. In row #1, all models misinterpret the metaphor “planting seeds for the future” literally, depicting the sowing and watering scenes. In row #2, Veo3.1 fails to infer the correct placement of AED electrodes, while the other two models show limited understanding of the device’s function. In row #3, Veo3.1 and Sora2 demonstrate

###### Observation 3. Video models struggle with visual un-

derstanding and reasoning. As shown in Tab. 2 and Fig. 4, generative models perform worst on the Game and Vision categories, both of which consist entirely of I2V instances. This indicates that, beyond their limited capabilities of rule-

Category

Prompt Veo3.1 Sora2 PixelVerse-v5

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

A startup founder faces repeated rejections but keeps planting seeds for the future.

Semantics

[Figure 123]

[Figure 124]

[Figure 125]

Continues metaphorical “seed planting” through sustained efforts. Persists in metaphorical actions after rejection. Actions align with the “planting” metaphor, not literal planting.

Continues metaphorical “seed planting” through sustained efforts. Persists in metaphorical actions after rejection. Actions align with the “planting” metaphor, not literal planting.

Continues metaphorical “seed planting” through sustained efforts. Persists in metaphorical actions after rejection. Actions align with the “planting” metaphor, not literal planting.

[Figure 126]

[Figure 127]

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

A person is using a portable AED (automated external defibrillator) at a public place for the victim.

Humanity

[Figure 141]

[Figure 142]

[Figure 143]

Shock delivered only after AED instruction and button press. Pads placed as indicated on AED diagrams. Defibrillation delivered promptly after button press.

Shock delivered only after AED instruction and button press. Pads placed as indicated on AED diagrams. Defibrillation delivered promptly after button press.

Shock delivered only after AED instruction and button press. Pads placed as indicated on AED diagrams. Defibrillation delivered promptly after button press.

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

[Figure 157]

[Figure 158]

Ignite the burner to gently heat the hydrated copper(II) sulfate crystals.

Science

[Figure 159]

[Figure 160]

[Figure 161]

Blue crystals turn white as heating dehydrates them. Droplets form on cooler upper tube surface during heating. Color change starts near flame and moves upward.

Blue crystals turn white as heating dehydrates them. Droplets form on cooler upper tube surface during heating. Color change starts near flame and moves upward.

Blue crystals turn white as heating dehydrates them. Droplets form on cooler upper tube surface during heating. Color change starts near flame and moves upward.

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

Assume winter and summer are swapped. In a normally cold winter city, people walk outside.

Hypothesis

[Figure 177]

[Figure 178]

[Figure 179]

People wear summer clothes despite winter setting. Clothes placed correctly on body parts.

People wear summer clothes despite winter setting. Clothes placed correctly on body parts.

People wear summer clothes despite winter setting. Clothes placed correctly on body parts.

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

You are a Go player holding the black stones. Please make a move to prevent your stones from being captured.

Game

[Figure 192]

[Figure 193]

[Figure 194]

Black stone placed on empty spot; no invalid removals. Black moves to the only free top intersection. Black group remains safe with at least one liberty.

Black stone placed on empty spot; no invalid removals. Black moves to the only free top intersection. Black group remains safe with at least one liberty.

Black stone placed on empty spot; no invalid removals. Black moves to the only free top intersection. Black group remains safe with at least one liberty.

[Figure 195]

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

Please correct the anomaly in the animal's appearance so that it conforms to natural laws.

Vision

[Figure 210]

[Figure 211]

[Figure 212]

Transitions from blue to realistic gray skin. Transitions from blue to realistic gray skin. Transitions from blue to realistic gray skin.

- Figure 5. Case studies on three closed-source models across six rule categories. Each sample is provided with the Rule Coherence aspects derived from the checklist questions. The three video models exhibit varying performance across different instances.

scientific reasoning abilities, with Veo3.1 exhibiting finergrained details. Meanwhile, in row #4, all models correctly infer the summer setting, yet their abilities to reason about human clothing vary. In row #5, Sora2 successfully performs game-strategy reasoning, whereas the other models fail to comprehend the task instruction. Finally, in row #6, Veo3.1 and PixelVerse-v5 effectively detect and rectify visual anomalies, while Sora2 struggles to reason over the visual context. These results show notable differences in reasoning competence across models, reflecting their varying abilities to integrate conceptual cues.

ing Kendall Rank Correlation Coefficient (KRCC), Spearman Rank Correlation Coefficient (SRCC), Pearson Linear Correlation Coefficient (PLCC), and the overall Accuracy (ACC). As shown in Tab. 3, all models demonstrate strong video understanding capabilities. Among them, GPT-o3 achieves the highest accuracy of 0.8512 and outperforms other MLLMs across all three correlation metrics, demonstrating the reliability of using GPT-o3 as the evaluator.

#### 6. Conclusion

In this paper, we introduce RULER-Bench, a comprehensive benchmark designed to evaluate the rule-based reasoning abilities of video generation models. RULER-Bench comprises 622 high-quality instances spanning 40 tasks across six rule categories, addressing the critical need for fine-grained evaluation of reasoning capabilities in video models. To ensure objective evaluation, we design a checklist-based protocol from four evaluation metrics. Finally, we conduct extensive experiments on 10 state-of-the-

Human alignment validation. To evaluate the reliability of our evaluation protocols, we conduct a human preference alignment experiment. First, we randomly select 80 generated videos along with the corresponding 813 checklist questions. Second, human annotators are invited to answer each question based on the video content. Third, we evaluate the consistency between outputs of different closed-source MLLMs and human annotations, report-

art video generation models and provide detailed analyses based on the results. We hope our work will provide valuable insights for advancing reasoning-aware video generation towards vision foundation intelligence.

#### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 2, 3
- [2] Alibaba Cloud (Wan Series). Wan 2.5, 2025. 2, 5
- [3] Anthropic. Introducing claude haiku 4.5, 2025. 17
- [4] Anthropic. System card: Claude sonnet 4.5. Technical report, Anthropic, 2025. Technical report, Anthropic, September 2025. 17
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2
- [6] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, KaiWei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024. 3
- [7] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek LLM: Scaling opensource language models with longtermism. arXiv preprint arXiv:2401.02954, 2024. 2
- [8] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In CVPR, 2023. 2
- [9] ByteDance Seed. Introducing to seed1.6 series, 2025. 17
- [10] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 2

- [11] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 4

- [12] Yunuo Chen, Junli Cao, Anil Kag, Vidit Goel, Sergei Korolev, Chenfanfu Jiang, Sergey Tulyakov, and Jian Ren. Towards physical understanding in video generation: A 3d point regularization approach. arXiv preprint arXiv:2502.03639, 2025. 3
- [13] Yupeng Chen, Penglin Chen, Xiaoyu Zhang, Yixian Huang, and Qian Xie. Editboard: Towards a comprehensive evaluation benchmark for text-based video editing models. In AAAI, 2025. 3
- [14] Yinan Chen, Jiangning Zhang, Teng Hu, Yuxiang Zeng, Zhucun Xue, Qingdong He, Chengjie Wang, Yong Liu, Xiaobin

- Hu, and Shuicheng Yan. Ivebench: Modern benchmark suite for instruction-guided video editing assessment. arXiv preprint arXiv:2510.11647, 2025. 3
- [15] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 17
- [16] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 4
- [17] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983, 2025. 2
- [18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2
- [19] Fanda Fan, Chunjie Luo, Wanling Gao, and Jianfeng Zhan. Aigcbench: Comprehensive evaluation of image-to-video content generated by ai. BenchCouncil Transactions on Benchmarks, Standards and Evaluations, 2023. 3
- [20] Weixi Feng, Jiachen Li, Michael Saxon, Tsu-jui Fu, Wenhu Chen, and William Yang Wang. Tc-bench: Benchmarking temporal compositionality in text-to-video and image-tovideo generation. arXiv preprint arXiv:2406.08656, 2024. 3
- [21] Alisa Fortin, Guillaume Vernade, Kat Kampf, and Ammaar Reshi. Introducing gemini 2.5 flash image: our state-of-theart image model, 2025. Google Developers Blog, May 2025. 4
- [22] Chongkai Gao, Haozhuo Zhang, Zhixuan Xu, Zhehao Cai, and Lin Shao. Flip: Flow-centric generative planning as general-purpose manipulation world model. arXiv preprint arXiv:2412.08261, 2024. 3
- [23] Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024. 2
- [24] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 2, 5

- [25] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. ChatGLM: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024. 2
- [26] Google DeepMind. Veo-3 technical report. Technical report, Google DeepMind, 2025. 2, 5
- [27] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma,

- Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 2025. 2
- [28] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [29] Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025. 3, 5
- [30] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jos´e Lezama. Photorealistic video generation with diffusion models. In ECCV, 2024. 2
- [31] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In CVPR, 2025. 2
- [32] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [33] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.
- [34] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 2
- [35] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2
- [36] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 3
- [37] Ziqi Huang, Ning Yu, Gordon Chen, Haonan Qiu, Paul Debevec, and Ziwei Liu. Vchain: Chain-of-visualthought for reasoning in video generation. arXiv preprint arXiv:2510.05094, 2025. 3
- [38] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 17
- [39] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In CVPR, 2024. 2
- [40] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 2

- [41] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang,

- et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 5
- [42] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 4

- [43] Hengjia Li, Lifan Jiang, Xi Xiao, Tianyang Wang, Hongwei Yi, Boxi Wu, and Deng Cai. Magicid: Hybrid preference optimization for id-consistent and dynamic-preserved video customization. arXiv preprint arXiv:2503.12689, 2025. 2
- [44] Hengjia Li, Haonan Qiu, Shiwei Zhang, Xiang Wang, Yujie Wei, Zekun Li, Yingya Zhang, Boxi Wu, and Deng Cai. Personalvideo: High id-fidelity video customization without dynamic and semantic degradation. In ICCV, 2025. 2
- [45] Ouxiang Li, Yuan Wang, Xinting Hu, Huijuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, Xiaojuan Qi, and Fuli Feng. Easier painting than thinking: Can text-to-image models set the stage, but not direct the play? arXiv preprint arXiv:2509.03516, 2025. 5
- [46] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Zhe Lin, Rita Singh, and Bhiksha Raj. Controlvar: Exploring controllable visual autoregressive modeling. arXiv preprint arXiv:2406.09750, 2024. 2
- [47] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 2
- [48] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025. 2
- [49] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded imageto-video generation. In ECCV, 2024. 3
- [50] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. Fetv: A benchmark for fine-grained evaluation of open-domain text-tovideo generation. NeurIPS, 2023. 3
- [51] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In CVPR, 2024. 3
- [52] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2
- [53] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024. 2, 3
- [54] Antonio Montanaro, Luca Savant Aira, Emanuele Aiello, Diego Valsesia, and Enrico Magli. Motioncraft: Physicsbased zero-shot video generation. In NeurIPS, 2024. 3

- [55] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 2
- [56] OpenAI. Gpt-5 system card. Technical report, OpenAI,

2025. 3, 17

- [57] OpenAI. Openai o3 and o4-mini system card. Technical report, OpenAI, 2025. 5, 17
- [58] OpenAI. Sora 2 system card. Technical report, OpenAI,

2025. 2, 5

- [59] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2
- [60] PixVerse Team. Pixverse v5, 2025. 5
- [61] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2
- [62] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [63] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2
- [64] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In CVPR, 2025. 3, 5
- [65] Shangkun Sun, Xiaoyu Liang, Songlin Fan, Wenxu Gao, and Wei Gao. Ve-bench: Subjective-aligned benchmark suite for text-driven video editing quality assessment. In AAAI, 2025. 3
- [66] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 2
- [67] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024. 2
- [68] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2
- [69] A¨aron van den Oord, Elias Roman, Olivier Lacombe, Alisa Fortin, and Guillaume Vernade. Veo 2, 2024. Google Research Blog, December 2024. 5
- [70] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models, 2022. 15
- [71] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang,

- Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 5
- [72] Jing Wang, Ao Ma, Ke Cao, Jun Zheng, Zhanjie Zhang, Jiasong Feng, Shanyuan Liu, Yuhang Ma, Bo Cheng, Dawei Leng, et al. Wisa: World simulator assistant for physics-aware text-to-video generation. arXiv preprint arXiv:2503.08153, 2025. 3
- [73] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. IJCV, 2025. 2
- [74] Yi Wang, Xinhao Li, Ziang Yan, Yinan He, Jiashuo Yu, Xiangyu Zeng, Chenting Wang, Changlian Ma, Haian Huang, Jianfei Gao, et al. Internvideo2. 5: Empowering video mllms with long and rich context modeling. arXiv preprint arXiv:2501.12386, 2025. 2
- [75] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Biao Gong, Longxiang Tang, Xiang Wang, Haonan Qiu, Hengjia Li, Shuai Tan, Yingya Zhang, et al. Dreamrelation: Relation-centric video customization. arXiv preprint arXiv:2503.07602, 2025. 2
- [76] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2, 3
- [77] Tao Wu, Yong Zhang, Xintao Wang, Xianpan Zhou, Guangcong Zheng, Zhongang Qi, Ying Shan, and Xi Li. Customcrafter: Customized video generation with preserving motion and concept composition abilities. In AAAI, 2025. 2
- [78] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, MingHsuan Yang, and Xu Yang. Kris-bench: Benchmarking next-level intelligent image editing models. arXiv preprint arXiv:2505.16707, 2025. 3, 4, 5
- [79] xAI. Grok 4 model card. Technical report, xAI, 2025. 17
- [80] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 2
- [81] Tianyi Xie, Yiwei Zhao, Ying Jiang, and Chenfanfu Jiang. Physanimator: Physics-guided generative cartoon animation. In CVPR, 2025. 3
- [82] Qiyao Xue, Xiangyu Yin, Boyuan Yang, and Wei Gao. Phyt2v: Llm-guided iterative self-refinement for physicsgrounded text-to-video generation. In CVPR, 2025. 3
- [83] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024. 2

- [84] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 5
- [85] Zixuan Ye, Huijuan Huang, Xintao Wang, Pengfei Wan, Di Zhang, and Wenhan Luo. Stylemaster: Stylize your video with artistic generation and translation. In CVPR, 2025. 2
- [86] Yu Yuan, Xijun Wang, Tharindu Wickremasinghe, Zeeshan Nadir, Bole Ma, and Stanley H Chan. Newtongen: Physicsconsistent and controllable text-to-video generation via neural newtonian dynamics. arXiv preprint arXiv:2509.21309,

2025. 3

- [87] Ailing Zhang, Lina Lei, Dehong Kong, Zhixin Wang, Jiaqi Xu, Fenglong Song, Chun-Le Guo, Chang Liu, Fan Li, and Jie Chen. Ui2v-bench: An understanding-based image-to-video generation benchmark. arXiv preprint arXiv:2509.24427, 2025. 3
- [88] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In CVPR, 2023. 2
- [89] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In ECCV, 2024. 3
- [90] Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. Videorepa: Learning physics for video generation through relational alignment with foundation models. arXiv preprint arXiv:2505.23656, 2025. 3
- [91] Yu Zhang, Wenxiang Guo, Changhao Pan, Dongyu Yao, Zhiyuan Zhu, Ziyue Jiang, Yuhan Wang, Tao Jin, and Zhou Zhao. Tcsinger 2: Customizable multilingual zero-shot singing voice synthesis. arXiv preprint arXiv:2505.14910,

2025. 2

- [92] Yu Zhang, Wenxiang Guo, Changhao Pan, Zhiyuan Zhu, Ruiqi Li, Jingyu Lu, Rongjie Huang, Ruiyuan Zhang, Zhiqing Hong, Ziyue Jiang, et al. Versatile framework for song generation with prompt-based control. arXiv preprint arXiv:2504.19062, 2025. 2
- [93] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025. 5
- [94] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, WeiShi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025. 2, 3
- [95] Bojia Zi, Penghui Ruan, Marco Chen, Xianbiao Qi, Shaozhe Hao, Shihao Zhao, Youze Huang, Bin Liang, Rong Xiao, and Kam-Fai Wong. Se\˜ norita-2m: A high-quality instructionbased dataset for general video editing by video specialists. arXiv preprint arXiv:2502.06734, 2025. 4

## RULER-Bench: Probing Rule-based Reasoning Abilities of Next-level Video Generation Models for Vision Foundation Intelligence

### Supplementary Material

#### Appendix Contents

The Appendix of RULER-Bench is structured as follows:

- • Sec. A: Detailed task explanations.
- • Sec. B: Implementation settings and configurations of different video generation models.
- • Sec. C: Explorations on Video-to-Video tasks.
- • Sec. D: More detailed configuration and experimental results of human annotation and user study.
- • Sec. E: Prompts used for evaluation and generation.
- • Sec. F: Experimental results across tasks.
- • Sec. G: More visualization results.
- • Sec. H: Limitations and future works.
- • Sec. I: Potential social impact of RULER-Bench.

#### A. Detailed Tasks Explanations

RULER-Bench evaluates the reasoning abilities of the stateof-the-art video generation models across six rule categories: Vision, Science, Semantics, Hypothesis, Game, and Humanity. To enable a more fine-grained assessment, we design a suite of 40 tasks distributed over these categories, spanning a broad spectrum of scenarios across the nature, society, and virtuality domains. In the following section, we comprehensively define each task.

###### A.1. Vision Rule

The Vision category comprises 10 distinct tasks, covering physical, spatial, and temporal visual attributes. Tasks in this category are designed to evaluate whether video generation models can accurately infer the visual composition of the real world from a given image and generate videos that adhere to fundamental visual principles.

Anomaly task focuses on visual plausibility reasoning. Given an image containing a clear visual anomaly, generative models are required to infer the correct appearance of the object based on implicit visual rules and generate videos that restore visual coherence. For example, when the input image depicts a cat with four tails, the model should reason that a cat normally has only one tail and accordingly correct the anomaly in the generated video clip.

Color task focuses on color reasoning and manipulation. Given an input image, generative models are required to reason about the color attributes of specific objects and modify them while preserving spatial-temporal coherence.

Size task targets object-scale adjustment and proportional reasoning. Given an instruction to resize a particular object, the model should modify its scale appropriately

while preserving scene geometry and physical plausibility.

Count task evaluates numerical reasoning and controllability. Models are required to adjust the number of instances of a given object according to the instruction while preserving spatial realism and temporal coherence.

Direction task focuses on directional reasoning and motion control. Given an input image and a directional instruction, generative models are required to adjusting the object’s facing and orientation, ensuring temporal coherence.

Shape task evaluates geometric reasoning and controllability. Video models are required to reason about the specified object’s geometry and modify its structure accordingly.

Position task evaluates the reasoning ability of video models on spatial attributes. The model needs to reason about the object’s location and relocate it accordingly.

View task focuses on viewpoint reasoning and perspective adaptation. Given a target view specification, video models should generate videos from the new viewpoint.

Motion task targets dynamic prediction and temporal reasoning. Models are required to infer plausible motion trajectories from a given static or partial motion input and predict temporally coherent frames. For instance, when observing a basketball player suspended in mid-air, the model should predict and generate the subsequent motion that reflects a realistic landing and continuation of the action.

Style task focuses on appearance reasoning and stylistic transformation. Given an input image and a style transformation instruction, video models are required to apply the target style while preserving object identity.

###### A.2. Science Rule

The Science category consists of 10 tasks spanning multiple disciplines within the natural sciences. These tasks are designed to evaluate whether video generation models have acquired a fundamental understanding of natural laws and scientific principles, enabling them to accurately simulate experimental phenomena, biological behaviors, and other science-driven processes in the real world.

Physics task focuses on scenes that involve fundamental physical principles, covering multiple domains such as mechanics, optics, and electromagnetism. Video models are required to predict physically consistent phenomena based on the implicit laws embedded within each sample.

Chemistry task focuses on fundamental chemical reaction principles and their observational experimental manifestations. Video generation models are required to infer plausible reaction processes and products by reasoning over

the chemical rules, given the reactants and conditions.

Biology task focuses on fundamental biological principles. Video models are required to infer plausible experimental outcomes based on organism-specific characteristics, covering a variety of scenarios such as controlled biological experiments, genetics, and conditioned reflexes.

Earth task focuses on geographical and environmental phenomena driven by factors such as location, climate variation, and seasonal cycles. Video generation models are required to infer long-term scene evolution according to underlying geographical principles and reflect these changes coherently within the generated video clip.

Math task focuses on the application of fundamental mathematical principles. Given a mathematical problem, video generation models are required to apply the relevant principles, derive the solution step by step, and visually illustrate the reasoning process clearly and coherently.

Medicine task focuses on the correct application and demonstration of medical protocols. Given a clinical scenario, video generation models are required to follow the specified procedural standards, such as venipuncture guidelines, the seven-step handwashing protocol, or basic surgical procedures, and accurately depict these procedures.

Life task focuses on fundamental principles of animal behavior. Video generation models are required to infer plausible behavioral responses when experimental disturbances occur, reasoning according to species-specific behavioral patterns and ecological logic.

###### A.3. Hypothesis Rule

The Hypothesis category includes two tasks that encompass diverse scenarios, designed to evaluate whether video generation models can reason according to newly defined rules and assumptions beyond real-world rules. Unlike other categories, the governing rules here are explicitly provided in the prompt, requiring the model to perform zero-shot reasoning under newly introduced conditions.

Subjective Scenario task focuses on human-defined hypothetical worlds. These scenarios violate real-world logic, requiring video generation models to reason about state changes according to the provided subjective rules. For example, given the rule “lying causes the nose to grow”, the model must recognize a lying event in the scene and apply the specified rule to generate a video in which the character’s nose lengthens accordingly.

Objective Law task focuses on modifying objective real-world principles. Unlike subjective scenarios, Objective Law tasks remain grounded in real physical principles but introduce deliberate modifications to them. Video models are required to abandon the original rule, adopt the newly specified one, and perform reasoning under the modified physical dynamics. For example, given the instruction “assume the density order is reversed,” the model must rea-

son according to the new law and generate outcomes that contradict the behavior dictated by real-world physics.

###### A.4. Game Rule

The Game category comprises 10 tasks that cover a wide range of logical reasoning and game scenarios. These tasks are designed to evaluate whether video generation models can reason according to game strategies, execute rational moves, and ultimately achieve victory. To ensure controllable complexity, each sample is constrained to the minimal number of required moves and simplified difficulty settings.

Chinese Chess task focuses on one-move checkmate scenarios. Models are required to generate a single, legally valid move that adheres to the piece-specific movement rules and delivers a decisive checkmate against the opponent’s General, ensuring that the resulting board state is both legally valid and tactically forced.

Gomoku task focuses on fundamental offensive and defensive patterns such as open-three formations, doublethree configurations, or immediate winning opportunities. Models must infer the optimal placement to either complete a five-in-a-row or block the opponent’s winning threat.

Go task focuses on one-move capture scenarios. Models are required to reason over the current board configuration and place a single stone that removes one or more opponent groups that are in atari with only one remaining liberty, thereby executing a valid and tactically justified capture.

Sudoku task evaluates logical deduction under strict numerical constraints. Given a partially filled grid, models must infer the correct number placement that simultaneously satisfies row, column, and subgrid consistency rules.

Chess task targets tactical positions where a checkmate can be achieved in the next move. Models must analyze the board configuration, identify forced-mate patterns, and generate the correct move that results in a checkmate.

Minesweeper task focuses on number-based spatial constraints. Models must analyze the revealed mine counts and infer which unrevealed tiles must contain mines with logical certainty, subsequently producing a move that marks all deterministically identifiable mine locations.

Maze task evaluates pathfinding and spatial reasoning under structural constraints. Models are required to generate a sequence of movements that leads the agent from the starting point to the goal while avoiding dead ends.

Puzzle task focuses on identifying spatial correspondences and assembling fragmented pieces into a unified structure. Models must infer the correct arrangement of puzzle pieces by reasoning about shape compatibility and boundary alignment to form a coherent final pattern.

Number Sliding task evaluates sequential planning under movement constraints. Models must generate a series of sliding operations that reposition numbered tiles toward the target configuration while preserving the adjacency rules.

Matchsticks task focuses on geometric and arithmetic reasoning under limited move constraints. Models must determine which matchsticks to reposition in order to form a valid equation or geometric configuration.

###### A.5. Semantics Rule

The Semantics category comprises 3 tasks, designed to evaluate whether video generation models can infer high-level semantic information from contextual cues and generate videos that accurately reflect the intended visual meaning.

Definition task focuses on the accurate illustration of a provided term or concept. Models are required to infer the essential semantic attributes and contextual nuances of the term and produce visual content that aligns with its correct definition rather than surface-level associations.

Metaphor task focuses on the interpretation of metaphorical expressions Given a metaphorical phrase, the models are required to generate a video that represents the underlying conceptual meaning, rather than a literal depiction of the words, demonstrating understanding of figurative language and abstract semantic mapping.

Idiom task focuses on culturally conventional idiomatic expressions. Models need to avoid literal interpretations and instead generate scenes that correctly express the idiom’s figurative semantic content, reflecting understanding of linguistic conventions and contextualized meaning.

###### A.6. Humanity Rule

The Humanity category includes 8 tasks that encompass diverse social scenarios and human conventions. These tasks are designed to evaluate whether video generation models can reason about societal norms, cultural practices, and customs to generate plausible human behaviors.

Dress task focuses on culturally appropriate clothing choices and outfit consistency. Models are required to generate videos where characters wear attire suitable for the given context, season, or activity.

Food task examines dietary norms and context-aware food selection. Models must generate scenes where the food type, preparation style, or eating behavior aligns with common culinary conventions and the specified scenario.

Emotion task evaluates the understanding of human affect and expressive behavior. Models are required to generate facial expressions, gestures, and body movements that reflect the specified emotional state in a natural manner.

Festival task focuses on cultural celebrations and associated symbolic practices. Models must incorporate appropriate festival elements, such as decorations, attire, or rituals, corresponding to the designated cultural event.

Safety task assesses whether models can adhere to basic safety norms in daily life. Given specific situations, models must avoid unsafe behaviors and instead generate actions aligned with conventional safety practices.

Table A1. Details of the configurations of generated videos.

Model Resolution Frames FPS Durations (s) Closed-Source Model

Veo3.1 1280×720 192 24 8.00 Veo2 1280×720 192 24 8.00 Sora2 1280×704 300 30 10.00

PixelVerse-V5 1280×720 121 24 5.04 Seedance1.0-pro 1248×704 121 24 5.04

Wan2.5 1280×720 121 24 5.04 Open-Source Model

CogVideoX1.5 5B 1360×768 161 16 10.06 HuyuanVideo 832×480 81 15 5.40

- Wan2.1 14B 832×480 81 16 5.06
- Wan2.2 A14B 832×480 81 15 5.40

Social task focuses on everyday social conventions. Models are required to generate actions that adhere to common social norms, engaging in other routine prosocial behaviors that reflect cultural expectations.

Sports task targets sport-specific conventions. Models must generate videos where human movements, equipment usage, and gameplay dynamics align with the rules and common practices of the designated sport.

Transportation task focuses on concrete road-traffic rules. Models are required to generate plausible interactions with real-world traffic systems, such as obeying traffic lights and road signs, or navigating multi-lane roads according to standard driving conventions.

#### B. Implementation Detail

We follow the official configurations of different video generation models. As summarized in Tab. A1, the videos have a duration of 5∼10 seconds and a resolution of either 720P or 480P. Specifically, for open-source models, we implement them using the Diffusers framework [70].

#### C. Explorations on Video-to-Video tasks

To provide a more systematic and comprehensive evaluation, we further explore the rule-based reasoning abilities of video generation models on the Video-to-Video (V2V) tasks. Following the same data collection pipeline as in the I2V task, we manually curate web videos and use GPT-5 for captioning. This web video set covers eight tasks from the Vision category and consists of 81 high-quality samples. We leverage Wan2.1 14B to perform rule-based reasoning based on these V2V instances. We find that in most cases, Wan2.1 14B struggles to perform plausible reasoning based on the prompt and input video, exhibiting relatively poor generative performance on V2V tasks, as illustrated in Tab. A3 Surprisingly, the model achieves substantially higher scores on the Visual Consistency and Visual Fidelity metrics compared to I2V instances in the same task. We attribute this to the presence of the input video, which

Table A2. Human alignment comparison across MLLMs. KRCC, SRCC, PLCC, and ACC are reported across the four evaluation metrics.

KRCC SRCC PLCC ACC

Model

IF RC VC VF IF RC VC VF IF RC VC VF IF RC VC VF

Claude-haiku-4-5 .2655 .3913 .2912 .1410 .4186 .5805 .5139 .3647 .4384 .5789 .5028 .3514 .6456 .6188 .6312 .7474 Claude-sonnet-4-5 .2998 .3913 .3375 .1414 .4852 .5832 .5809 .3397 .4909 .5794 .5640 .3630 .6519 .6733 .7188 .7440

Grok-4 .3656 .4129 .3874 .2184 .5680 .6200 .5990 .4097 .5798 .6150 .6042 .4251 .6603 .7050 .7006 .6678 doubao-seed-1-6 .2540 .3785 .1387 -.0031 .5473 .5521 .3904 -.0385 .5569 .5521 .4143 -.0385 .6471 .6875 .6875 .9259 Gemini-2.5-flash .3725 .4032 .3238 .1874 .6021 .6091 .5647 .4791 .6113 .6063 .5649 .4805 .7025 .7129 .7250 .7782

Gemini-2.5-pro .4012 .4426 .3224 .1708 .6639 .6926 .5478 .3843 .6559 .6921 .5431 .3974 .7152 .7525 .6937 .7167

- GPT-4o .3336 .3670 .3064 .1535 .5869 .5512 .5661 .4209 .5777 .5534 .5516 .4037 .6962 .6584 .7125 .8020

- GPT-5 .4598 .4923 .4152 .1831 .7495 .7751 .6898 .5301 .7309 .7750 .6869 .5284 .8038 .7871 .7625 .8259 GPT-o3 .5281 .5218 .4564 .2419 .8770 .8188 .8010 .6315 .8765 .8171 .8015 .5996 .8924 .8218 .8500 .8498

Table A3. Quantitative results of Wan2.1 14B for V2V tasks.

[Figure 213]

Metrics Color Count Direction Position Shape Size Style View

RC 3.57 18.18 10.00 12.50 18.18 14.58 7.5 5.00 VC 42.86 22.73 55.00 71.67 31.82 62.5 52.5 42.50 VF 89.29 64.39 82.50 77.50 59.09 72.92 50.83 51.67 Avg 45.24 35.10 49.17 53.89 36.36 50.00 36.94 33.06

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

###### OutputInput

Figure A2. Annotation interface for human alignment evaluation.

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

question. For each sample, we provide detailed explanations of the associated metadata, such as the prompt and implicit explanation, and standardize the annotation requirements to ensure reliable judgments. Each annotators are required to make revisions based on the following criteria:

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

identify the two snowy owls and replace them with exactly three snowy owls. Replace the central cluster of purple spheres in with a cluster of purple cubes

Figure A1. Qualitative results of Wan2.1 14B for V2V tasks.

- • Accuracy evaluates whether the question accurately reflects the expected video content based on specific rules.
- • Clarity considers whether the checklist question is stated clearly and without ambiguity.
- • Relevance measures whether the question is relevant to the scenario and prompt of the given sample.
- • Consistency assesses whether the question type aligns with the corresponding evaluation dimension.
- • Completeness ensures whether the checklist covers all necessary aspects for thorough evaluation.

likely serves as a strong visual reference and encourages the model to align both visual quality and temporal consistency closely with the provided input. However, in a few instances, as shown in Fig. A1, Wan2.1 14B is able to produce results that meet expectations, although this capability is observed only on sporadic samples.

#### D. Details of Human Annotation

We conduct large-scale human annotation for two aspects: checklist validation and human alignment experiments. In the checklist validation stage, human annotators are employed to assess the soundness of the proposed checklist and refine it when necessary. In the human alignment experiment, annotations are collected to support a user study designed to evaluate the alignment between the output of GPT-

###### D.2. User Study

To verify the effectiveness of our GPT-o3-based automated evaluation, we conduct a human preference alignment experiment. First, we randomly sample 80 generated videos along with their corresponding checklist questions. Second, we develop an annotation interface to improve labeling efficiency. As shown in Fig. A2, annotators are provided with the generated video, the corresponding prompt, and the checklist questions. Third, annotators need to complete each checklist question based on the visual content of the video. After annotation, we aggregate all responses and compare human annotation with the output from state-

- o3 and human preferences. The overall annotation process is collaboratively conducted by the authors. In this section, we present the detailed annotation procedures and results.

###### D.1. Checklist Validation

To validate the effectiveness of the constructed checklist, we perform quality control for each individual checklist

- of-the-art MLLMs, including Claude Haiku 4.5 [3], Claude Sonnet 4.5 [4], Doubao Seed 1.6 [9], Gemini 2.5 flash [15], Gemini 2.5 pro [15], Grok-4 [79], GPT-4o [38], GPT-5 [56], and GPT-o3 [57]. KRCC, SRCC, PLCC, and ACC are reported.

As shown in Tab. A2, GPT-o3 demonstrates a significant performance advantage across all four evaluation metrics. Its alignment with human preferences exceeds 85%, indicating its effectiveness as an evaluator.

#### E. Prompt for Generation and Evaluation

Fig. A11 illustrates the prompt used to evaluate the performance across four evaluation metrics of MLLMs. Fig. A14 presents the prompt for generating a checklist based on the provided metadata of the specific instance. Fig. A12 demonstrates the system prompt used for our Image-toscenario pipeline. GPT-5 is required to generate the corresponding prompt and implicit explanation based on the instruction and provided image. Finally, we use the prompt presented in Fig. A13 to generate initial scenario, prompt, and implicit explanation according to the task property.

#### F. Experimental Results per Task

We summarize the performance of different video generation models across the evaluation tasks. As shown in Tab. A4, Veo3.1 demonstrates impressive generative capability on multiple tasks, and closed-source models consistently outperform open-source models. Moreover, we find that although prompt enhancement leads to observable performance gains, current video models still fall short of fully achieving reliable rule-based reasoning.

##### G. More Visualization Results Additional qualitative results are presented as follows:

- • Fig. A3: Qualitative results of the Anomaly task.
- • Fig. A4: Qualitative results of the Safety task.
- • Fig. A5: Qualitative results of the Earth task.
- • Fig. A6: Qualitative results of the Life task.
- • Fig. A7: Qualitative results of the Physics task.
- • Fig. A8: Qualitative results of the Chemistry task.
- • Fig. A9: Qualitative results of the Emotion task.
- • Fig. A10: Qualitative results of the Color task.

#### H. Limitations and Future Works

While RULER-Bench provides a comprehensive evaluation for the rule-based reasoning ability of video generation models, there are still many challenges:

• Lack of a unified evaluation that jointly assesses T2V, I2V, and V2V tasks. Since most existing video models support either T2V and I2V jointly or V2V exclusively, we focus our evaluation on the T2V and V2V settings for

consistency. Future work may extend this framework to all three task paradigms, enabling a unified assessment for the next-level video models that natively support T2V, I2V, and V2V in a comprehensive manner.

• Potential for both depth and breadth expansion. While RULER-Bench covers a broad spectrum of rule-based reasoning scenarios, its breadth can be further extended to tasks such as GUI interaction, classical algorithms, and table reasoning, which are left for future work.

#### I. Social Impacts

This work may positively influence the development of more reliable video generation by encouraging consistent rule-based reasoning, thereby supporting better downstream applications. However, stronger generative abilities may also increase the risk of misuse that obeys ethical norms and legal requirements. These risks highlight the importance of ethical deployment practices and appropriate oversight to ensure that progress in video generation technology benefits society while maintaining responsible use.

Table A4. Evaluation result across tasks in six rule categories.

Closed-Source Models Open-Source Models Prompt Enhancement Veo3.1 Veo2 Sora2

Category Task

Pixel Verse-V5

Seedance 1.0-pro

CogVideoX 1.5 5B

Hunyuan Video

Wan2.1 14B

Wan2.2 A14B

Wan2.5

Veo3.1 PE Sora2 PE

transportation 80.00 54.50 71.56 73.44 68.19 67.53 37.15 48.72 53.24 50.67 78.50 69.38 sport 78.08 62.80 73.48 71.66 65.87 74.95 44.98 58.74 56.06 59.66 82.36 79.65 social 74.88 51.09 77.77 65.23 64.24 61.18 36.54 49.49 51.42 51.72 75.08 75.22 safety 82.79 48.18 80.13 75.13 65.79 57.40 45.46 47.79 52.93 59.36 85.49 87.88

Humanity

festival 80.83 60.43 84.35 80.03 78.65 61.54 48.28 61.81 61.71 68.03 90.03 89.74 dress 90.87 87.18 89.82 86.86 88.02 85.10 72.76 81.97 81.25 85.74 94.15 93.67 food 90.94 78.25 88.86 78.65 72.90 67.67 57.68 66.03 65.61 60.97 93.36 85.68

emotion 69.21 56.67 63.60 66.34 59.85 45.46 33.68 42.89 52.07 40.12 59.54 55.57

chemistry 81.13 71.20 85.78 75.21 66.61 65.73 45.90 36.87 46.25 51.62 86.59 81.63 physics 79.09 56.99 76.41 72.32 73.44 76.34 34.78 36.18 54.30 54.17 74.49 82.25 biology 83.31 52.48 84.14 73.63 75.68 74.97 47.30 40.95 55.81 52.44 81.15 76.38

Science

earth 57.27 35.46 52.92 52.02 47.81 49.12 39.42 39.12 50.27 48.94 64.35 51.12 math 61.52 55.34 70.09 57.86 57.31 65.76 32.78 39.48 42.68 43.33 61.28 72.92

medicine 63.38 53.53 64.64 64.18 61.11 56.09 36.18 44.91 46.72 47.84 68.14 80.62 life 74.51 51.50 65.86 67.54 65.28 55.44 42.77 37.21 52.43 55.62 83.68 73.50

chess 20.94 16.25 45.94 28.12 36.84 28.91 28.28 31.09 21.88 33.28 23.44 50.78

puzzle 64.43 41.56 44.95 58.39 52.63 59.90 27.03 24.06 38.65 36.67 53.54 49.95 gomoku 39.43 41.67 56.98 41.20 51.72 46.51 46.61 32.24 36.98 44.90 46.82 71.72 sudoku 52.36 30.69 71.81 64.31 51.76 51.11 46.81 34.72 43.33 53.61 51.67 67.50

maze 56.60 39.17 54.44 45.28 52.64 57.57 28.89 26.67 38.89 44.51 59.31 51.25 minesweeper 70.00 42.50 71.41 68.75 68.44 73.44 47.66 47.81 50.78 50.47 57.66 78.12 number puzzle 50.83 30.83 46.11 32.50 41.67 47.78 26.39 22.22 33.06 46.11 32.22 52.22 sticks 39.53 21.98 54.17 54.01 43.54 55.31 38.96 23.44 33.39 47.55 40.99 57.24 xiangqi 34.27 19.41 47.64 39.09 44.08 32.95 51.37 25.82 27.47 34.10 31.36 48.68

Game

Go 41.44 38.10 58.26 43.53 46.15 49.40 44.42 33.11 22.20 51.64 32.74 59.30 Hypothesis

subjective 74.06 46.08 62.73 58.87 49.91 49.42 35.56 38.50 46.24 44.90 75.36 73.02 objective 81.31 61.21 73.27 86.12 68.66 65.64 46.91 50.52 58.02 61.72 81.91 84.32

idioms 76.53 60.89 72.14 72.11 65.56 60.00 51.13 52.21 59.05 57.83 83.82 75.13 metaphors 81.36 78.75 70.37 80.56 74.04 72.12 67.98 68.93 71.52 70.54 79.73 82.07 definition 87.25 71.50 80.08 76.38 76.15 82.61 58.10 55.78 58.68 64.01 86.62 79.01

Semantics

anomaly 42.19 34.69 32.81 33.12 37.19 36.25 8.12 24.06 21.56 28.44 38.44 35.62 color 63.89 26.39 22.92 36.81 51.39 54.17 22.22 29.17 30.56 33.33 62.50 38.19 count 20.14 18.75 34.03 26.56 38.89 42.36 8.33 23.61 22.22 38.89 26.39 38.19

direction 32.50 30.00 30.00 35.00 48.75 32.81 16.88 17.50 30.00 44.38 36.88 28.75 position 37.50 43.12 41.88 44.38 37.50 45.62 21.88 28.13 32.50 36.88 28.12 42.50

Vision

shape 46.88 27.34 39.58 47.22 54.17 36.11 18.06 27.78 31.94 37.50 50.78 40.28 size 47.92 30.73 38.02 43.75 43.75 51.14 20.83 42.71 36.98 41.67 46.35 31.25 style 47.50 23.12 43.75 36.88 31.25 43.75 7.50 20.62 25.62 32.50 47.50 46.88 view 51.25 29.38 31.25 50.62 45.62 53.75 19.38 42.50 31.88 38.12 51.25 32.50

motion 55.31 51.04 41.77 56.88 56.88 53.12 27.29 31.25 42.71 50.21 57.92 53.65

[Figure 244]

###### Figure A3. Qualitative results of the Anomaly task across 10 video models. Prompt: Please correct the bicycle’s wheel shape anomaly so that the vehicle can function properly.

[Figure 245]

###### Figure A4. Qualitative results of the Safety task across 10 video models. Prompt: In an office, thick smoke and small flames are rising from a pile of paper near a power socket. Several employees notice the fire spreading quickly across nearby papers and office supplies. They look around the room, searching for a way to extinguish it.

[Figure 246]

###### Figure A5. Qualitative results of the Earth task across 10 video models. Prompt: Observe the lava flow over several years under continuous volcanic activity, noting changes along its path.

[Figure 247]

###### Figure A6. Qualitative results of the Life task across 10 video models. Prompt: The process of a tadpole transforming into a frog.

[Figure 248]

###### Figure A7. Qualitative results of the Physics task across 10 video models. Prompt: Fill a U-shaped transparent tube with colored water so that both arms have the same initial height. Add extra colored water to the left arm and observe the water levels in both arms.

[Figure 249]

###### Figure A8. Qualitative results of the Chemistry task across 10 video models. Prompt: First, add 0.5 mL of 0.1 M dilute hydrochloric acid and observe the litmus color change; then add 1 mL of 0.1 M dilute sodium hydroxide, recording the color change.

[Figure 250]

###### Figure A9. Qualitative results of the Emotion task across 10 video models. Prompt: A student has just learned that their exam results are excellent, exceeding expectations.

[Figure 251]

###### Figure A10. Qualitative results of the Count task across 10 video models. Prompt: Increase the number of pink paperclips in the input image so that there are exactly five partially overlapping paperclips, arranged in a visually natural stacked formation where each clip slightly overlaps the next.

System Prompt for Evaluation You are a video quality assessment expert. Respond using the following format exactly:

<think></think> <answer></answer>

In <think></think>, first provide a concise, concrete description of the task. Next, give an objective understanding/summary of the input video (what happens in the video, salient visual / content elements). For each checklist question, give a brief factual justification (one or two short sentences) explaining why you rated that question as you did. Do not reveal private chain-of-thought; provide only a concise, professional justification for each answer.

In <answer></answer>, write the answers to all checklist questions as a JSON-style list of labels in order, for example: ["Good", "Medium", "Poor", ...] The list length must exactly match the number of checklist questions. Each item must be one of “Good”, “Medium”, or “Poor”. Do not include any other text, commentary, or explanation in <answer>. You will be provided:

- • a video
- • the generation prompt
- • an optional implicit explanation
- • an optional input image
- • an optional ground-truth image
- • a set of checklist questions Definitions
- • generation prompt: the single textual instruction given to the video-generation model.
- • implicit explanation: an optional detailed explanation of the intended outcome.
- • input image: optional visual input provided to the model.
- • ground truth image: the expected final frame or expected motion trajectory.
- • checklist questions: specific quality or content questions to assess. Notes
- • “Good” → The video clearly and fully satisfies the question.
- • “Medium” → The video partially satisfies the question or has minor issues.
- • “Poor” → The video fails to satisfy the question or has major issues. Rules

- 1. If the premise of a question is not met by the video (e.g., the question asks about an element that never appears), rate that question “Poor”.
- 2. Evaluate each question independently; do not let other answers influence a question’s rating.
- 3. Provide concise justifications in <think> and only the ordered list of labels in <answer>.
- 4. The response must strictly follow the <think></think><answer></answer> format and include nothing outside those tags.

Figure A11. System prompt template for the evaluation of generated videos.

System Prompt for Image-to-scenario Pipeline

For each sample, you must output two keys: prompt and implicit explanation, based on the provided image. Follow the detailed rules below precisely to ensure clarity and consistency.

###### 1. Prompt:

- • Describe the action or external event applied to the provided image — the only textual input to the model.
- • Requirements:

- – Begin with a single sentence briefly describing the visual content of the initial scenario (e.g., “In the image, the beaker on the left contains hydrochloric acid, and the beaker on the right contains a sodium hydroxide solution.”)

- – Must contain exactly one main action or event.
- – Exclude results, explanations, or causes.
- – Keep it precise and maintain the same style and perspective as the initial scene.
- – Use unambiguous, temporally clear language.
- – Must not leak any information from implicit explanation.

- – Ensure standalone completeness: the description must remain logically self-contained even when no initial scenario is provided.
- – Clearly specify all references in each sentence; avoid vague terms such as “some”, “certain”, “a kind of”, or “unknown” — always use explicit, concrete nouns.

###### 2. Implicit Explanation:

- • Explain the hidden reasoning behind the prompt: the mechanism that will unfold after the prompt’s action.
- • Requirements:

- – Describe factual mechanisms and expected consequences logically.
- – Include a clear cause → process → result sequence.
- – Use precise, domain-correct terminology (no metaphors or emotional tone).
- – Maintain full consistency with both initial scenario and prompt.

###### Cautions:

- • All samples must maintain high-school level reasoning difficulty: complex enough to require logical understanding but not specialized technical expertise.
- • Scenarios must remain within the real-world physical scale: avoid microscopic (e.g., atomic, cellular) and macroscopic (e.g., planetary, galactic) contexts.
- • Ensure internal logical coherence: objects, actions, and outcomes must follow physically plausible relationships.
- • All components (initial scenario, prompt, implicit explanation) must align stylistically and factually.

- • Avoid fantasy or science-fiction elements unless explicitly grounded in real-world analogs.
- • Maintain visual realism and a neutral tone — no symbolic, emotional, or metaphorical descriptions.
- • Each generated sample should strictly correspond to the specific task requirements and concepts, with no ambiguity in objects, actions, or reasoning related to the task.

Figure A12. System prompt template for Image-to-scenario pipeline designed for I2V tasks.

System Prompt for Scenario-to-image Pipeline

For each sample, you must output 3 keys: initial scenario, prompt, and implicit explanation. Follow the detailed rules below precisely to ensure clarity and consistency.

###### 1. initial scenario:

- • Describe the static starting state of the scene in precise, visual, and objective language. This defines what the model initially “sees” before any change occurs.
- • Requirements:

- – The description should be detailed and concise, including visual objects, spatial arrangement, lighting, composition, and style, etc.
- – No actions, no temporal hints, no implied future events.
- – Should logically support the descriptions in prompt.
- – Do not include any visible text, letters, characters, or numbers in the scene description (e.g., avoid “a book with the title ‘History’ on the cover” or “a clock showing the number 12”).
- – Clearly specify all references in each sentence; avoid vague terms such as “some”, “certain”, or “unknown”

— always use explicit, concrete nouns.

###### 2. prompt:

- • Describe the action or external event applied to the initial scene — the only textual input to the generative model.
- • Requirements:

- – Begin with a single sentence briefly describing the visual content of the initial scenario (e.g., “In the image, the beaker on the left contains hydrochloric acid, and the beaker on the right contains a sodium hydroxide solution.”)

- – Must contain exactly one main action or event.
- – No results, explanations, or causes.
- – Maintain the same style and perspective as the initial scene.
- – Must not leak any information from implicit explanation.

- – Standalone completeness: The description must be logically self-contained even when no initial scenario is provided.
- – Ensure continuity between actions and the initial scene.
- – Clearly specify all references in each sentence; avoid vague terms such as “some”, “certain”, “a kind of”, “unknown” — always use explicit, concrete nouns.

###### 3. implicit explanation:

- • Explain the hidden reasoning behind the prompt: the mechanism that will unfold after the prompt’s action.
- • Requirements:

- – Describe factual mechanisms and expected consequences logically.
- – Include cause → process → result sequence.
- – Use precise, domain-correct terminology (no metaphors or emotional tone).
- – Must stay fully consistent with initial scenario and prompt.

###### Cautions

- • Scenarios must remain strictly within the real-world physical scale: avoid microscopic (e.g., atomic, cellular) and macroscopic (e.g., planetary, galactic) contexts.
- • Ensure internal logical coherence: objects, actions, and outcomes must follow physically plausible relationships.
- • All components (initial scenario, prompt, implicit explanation) must align stylistically and factually with one another.

- • Each generated sample should strictly correspond to the specific task requirements and concepts, with no ambiguity in objects, actions, or reasoning related to the task.

Figure A13. System prompt template for Scenario-to-image pipeline. Initial scenarios are instructed to T2I systems to generate images.

System Prompt for Checklist Generation

You are an expert evaluator specialized in assessing video generation models. Your task is to create a checklist composed of questions that will be used by a multimodal model to judge the quality of a generated video.

###### Inputs

- • prompt: the only textual instruction used by the video generation model.
- • input image or video (optional): the only visual input provided to the model.
- • gt image (optional): the expected final frame or the expected motion trajectory.

- • implicit explanation (optional): a detailed explanation of the intended outcome. Evaluation Dimensions

- • Instruction Following – Whether the video correctly follows the prompt’s instruction.
- • Visual Consistency – Whether objects, actors, and camera motions remain coherent across space and time including alignment, motion realism, and smooth scene transitions.
- • Rule Coherence – Whether the video logically reflects causal, physical, or rule-based reasoning implied by the prompt.
- • Visual Fidelity – Whether the video maintains high visual quality — including clarity, color stability, and absence of artifacts or distortions.

Checklist Design Rules

- 1. Include at least five questions.
- 2. Each question must refer to specific objects, actions, or outcomes described in the inputs.
- 3. Include at least one question that explicitly tests the model’s reasoning ability.
- 4. Each reasoning-related question must focus on only one reasoning aspect (e.g., causal reasoning, physical reasoning, spatial reasoning) — do not combine multiple reasoning types in a single question.
- 5. Phrase every question so that answering “Yes” means the generated video meets the expected criteria, and “No” means it does not.
- 6. Use only information that appears in the prompt, input image / video, gt image, and explicit explanation.

- 7. If part of the explicit explanation describes something that is not a necessary result of the prompt, you may ignore it.

- 8. If multiple reasoning aspects are involved, design separate questions for each reasoning aspect.
- 9. The final checklist must include at least one question corresponding to each of the four dimensions: Instruction Following, Visual Consistency, Rule Coherence, and Visual Fidelity.

Output Format

Return the checklist as a valid Python list of JSON objects, where each JSON object uses the dimension name as the key and the corresponding Yes/No question as the value.

Figure A14. System prompt template for checklist generation across 4 evaluation metrics.

