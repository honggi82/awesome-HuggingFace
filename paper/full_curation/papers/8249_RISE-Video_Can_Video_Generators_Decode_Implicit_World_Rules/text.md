# arXiv:2602.05986v1[cs.CV]5Feb2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

## RISE-Video: Can Video Generators Decode Implicit World Rules?

Mingxin Liu1,2,∗, Shuran Ma3,2,∗, Shibei Meng4,∗, Xiangyu Zhao1,∗‡, Zicheng Zhang1, Shaofeng Zhang1, Zhihang Zhong1, Peixian Chen2, Haoyu Cao2, Xing Sun2, Haodong Duan5, Xue Yang1,†

1Shanghai Jiao Tong University, 2Tencent Youtu Lab, 3Xidian University, 4Beijing Normal University, 5The Chinese University of Hong Kong

∗Equal contribution, ‡Project Lead, †Corresponding Author

###### Abstract

While generative video models have achieved remarkable visual fidelity, their capacity to internalize and reason over implicit world rules remains a critical yet under-explored frontier. To bridge this gap, we present RISE-Video, a pioneering reasoning-oriented benchmark for Text-Image-to-Video (TI2V) synthesis that shifts the evaluative focus from surface-level aesthetics to deep cognitive reasoning. RISE-Video comprises 467 meticulously human-annotated samples spanning eight rigorous categories, providing a structured testbed for probing model intelligence across diverse dimensions, ranging from commonsense and spatial dynamics to specialized subject domains. Our framework introduces a multi-dimensional evaluation protocol consisting of four metrics: Reasoning Alignment, Temporal Consistency, Physical Rationality, and Visual Quality. To further support scalable evaluation, we propose an automated pipeline leveraging Large Multimodal Models (LMMs) to emulate human-centric assessment. Extensive experiments on 11 state-of-the-art TI2V models reveal pervasive deficiencies in simulating complex scenarios under implicit constraints, offering critical insights for the advancement of future world-simulating generative models.

Date: February 6, 2026 Code: https://github.com/VisionXLab/RISE-Video Hugging Face: https://huggingface.co/datasets/VisionXLab/RISE-Video

### 1 Introduction

Recent years have witnessed rapid progress in video generation, driven largely by advances in large-scale generative models.

Increasingly realistic Text-to-Video (T2V) [29, 36, 39, 43] and Text-Image-to-Video (TI2V) generation [13, 21] models have demonstrated remarkable success in enhancing visual fidelity and structural controllability. Despite these strides, a critical question remains unanswered: can contemporary TI2V models reliably internalize and reason over implicit world rules that extend beyond explicit textual instructions? While general-purpose frameworks like VBench [14] provide comprehensive evaluations, and various task-oriented benchmarks [18, 41] have emerged, most existing metrics predominantly emphasize perceptual quality and temporal coherence. Consequently, there is a notable scarcity of evaluation protocols focused on implicit

[Figure 8]

[Figure 9]

[Figure 10]

Reasoning Question Has the person removed the

Instruction

[Figure 11]

[Figure 12]

Show the process of the man drinking from the bottle.

Input Image

bottle cap from the drink bottle by twisting it open?

[Figure 13]

Veo 3.1

Hailuo 2.3

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

Sora 2

[Figure 25]

Kling 2.6

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

HunyuanVideo 1.5

[Figure 37]

[Figure 38]

CogVideoX1.5

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

- Figure 1 An example from the Experiential Knowledge dimension of RISE-Video, revealing limitations in experiencebased reasoning of current TI2V models.

reasoning, particularly within the TI2V paradigm. These exigencies necessitate the development of a dedicated, reasoning-oriented diagnostic framework.

To bridge the gap in rule-aware evaluation for TI2V models, we introduce RISE-Video, a benchmark explicitly engineered to prioritize implicit reasoning over superficial generative quality. At the foundational level, the benchmark is organized into eight distinct reasoning dimensions: experiential, commonsense, temporal, societal, perceptual, spatial, subject-specific, and logical reasoning. This taxonomy enables a comprehensive coverage of the reasoning landscape in video synthesis, spanning from low-level perceptual cues to high-level abstract inferences. RISE-Video comprises 467 meticulously curated samples, each subject to rigorous human expert annotation to ensure ground-truth reliability. Building upon this structured data foundation, we define four evaluation metrics to provide a holistic appraisal: Reasoning Alignment, Temporal Consistency, Physical Rationality, and Visual Quality. This multi-dimensional approach ensures that generated videos are not only visually plausible but also strictly adhere to the underlying cognitive and physical constraints mandated by the input instructions.

To enable scalable evaluation, we further develop an automated LMM (Large Multimodal Models)-based judging pipeline guided by manually designed, reasoning-aware questions and prompts. Using this pipeline, we evaluate 11 representative TI2V models, and the results reveal clear reasoning limitations across current systems. Moreover, we validate that the proposed evaluation pipeline exhibits a high degree of alignment with human judgments, indicating that LMM-based evaluation can serve as a reliable and cost-effective alternative to large-scale human assessment.

Overall, our contributions are as follows:

- 1. We introduce RISE-Video, a pioneering benchmark designed to evaluate the capacity of TI2V models to internalize and execute implicit world rules. It encompasses 467 meticulously human-annotated samples across eight distinct reasoning domains, providing comprehensive coverage of diverse scenarios.
- 2. We propose four complementary evaluation dimensions to assess reasoning correctness beyond perceptual fidelity and develop an automated LMM-based evaluation pipeline, enabling scalable evaluation while maintaining strong alignment with human judgments.

[Figure 49]

[Figure 50]

[Figure 51]

Puzzle Solving ~17 Geometric Reasoning ~14 Game Actions ~11

Contextual Knowledge ~23 Intention Causality ~23 Identity Reasoning ~14 Procedural Knowledge ~16

Position ~13 Size ~10

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

Color ~10 Count ~7 Occlusion ~4

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

c

[Figure 66]

[Figure 67]

[Figure 68]

Instruction:

[Figure 69]

Instruction:

Instruction:

[Figure 70]

The process of Mario continuing to move to the right.

Generate a reasonable continuation that shows what naturally happens in the very next moment after the given scene.

As the vehicle continues moving forward, it becomes fully visible.

- 6 5 4

- 7

- 8

2

##### 1

[Figure 71]

[Figure 72]

Physics ~23 Geography ~22 Chemistry ~20 Sports ~13

Medium ~19 Short ~15 Long ~14 Reverse ~13

[Figure 73]

[Figure 74]

~76 Experiential Knowledge

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Logi~4cal2

[Figure 79]

[Figure 80]

C ap

ual

~44 Percept

[Figure 81]

[Figure 82]

ablity

Knowledge

[Figure 83]

[Figure 84]

Instruction:

Instruction:

The piece of marble is placed into the beaker containing dilute hydrochloric acid.

Show the changes in the scene over one hour at room temperature.

~78 Subject Knowledge

RISE-Video

~61 Temporal Knowledge

~467

Instruction:

[Figure 85]

[Figure 86]

The athlete stands in front of the basketball hoop, with his knees slightly bent and holding the basketball in both hands, then he completes the shooting motion.

~5 Spatial0 Kno wledge

Instruction:

3 Societal Knowledge

~3

Show the process of time reversing over five seconds.

~83 Commonsense Knowledge

3

[Figure 87]

[Figure 88]

Cultural Customs ~15 Social Rules ~9 Emotions ~9

Viewpoint ~20 Object Arrangement ~19 Structural Inference ~11

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Physical Commonsense ~38 Life Commonsense ~33 HealthCare ~12

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

c

[Figure 101]

Instruction:

[Figure 102]

[Figure 103]

c

This is the healthy alveolar structure and how it changes if a person has emphysema?

[Figure 104]

Instruction:

Instruction:

[Figure 105]

[Figure 106]

The camera viewpoint gradually shifts from the original angle to an overhead view of the provided mug.

Place the single most iconic traditional Chinese New Year main food on the plate.

- Figure 2 Task distribution of the RISE-Video benchmark, which comprises eight major task categories: Experiential Knowledge, Perceptual Knowledge, Temporal Knowledge, Spatial Knowledge, Commonsense Knowledge, Societal Knowledge, Subject Knowledge, and Logical Capability. Each category further contains comprehensive sub-categories and diverse data samples.
- 3. We conduct a comprehensive evaluation on 11 representative TI2V models, revealing systematic reasoning limitations and providing insights into current model capabilities. 2 Related Work

#### 2.1 Video Generation Models

Video generation [4, 25, 32, 43] research has advanced primarily through diffusion models [5–7, 11, 12, 30]. Early works [3, 9] integrated motion priors into image generators by adding temporal modules to latent diffusion models, enabling text-to-video and image-to-video synthesis. Beyond training recipes, architectural advances further strengthen long-range temporal coherence; Lumiere [2] adopts a space–time U-Net to generate entire clips in a single pass, while CogVideoX [40] scales diffusion-transformer designs with a 3D VAE to support longer, higher-resolution, and better text-aligned videos. In parallel, a complementary line explores unified multimodal generation and editing instead of single-condition synthesis: VideoPoet [16] reformulates video generation as autoregressive multimodal token prediction, and Movie Gen [24] extends this paradigm toward high-resolution generation with instruction-based editing and audio alignment. At the frontier, large-scale closed and production systems [23, 25, 32, 37] further push video duration, realism, and controllability, highlighting the need for systematic evaluation across text-to-video and image-to-video settings.

- 2.2 Evaluation of Video Generation Models

Video generation benchmarks have progressively evolved from coarse perceptual metrics toward more structured and semantically grounded evaluation protocols. Early evaluations [10, 26, 34] rely on frame- or video-level metrics, which measure overall realism but fail to capture motion coherence. To go beyond coarse metrics and capture the diverse capabilities of modern video generation, VBench [14] provides a unified framework with 8 data categories and 16 evaluation dimensions. In parallel, specific research [1, 18, 20, 27]focuses on evaluating whether generated content adheres to basic physical commonsense, and additional benchmarks [8, 15, 17, 41] further emphasize video dynamics. Recent benchmarks [22, 31, 38, 42, 45] move toward comprehensive, human-aligned evaluation, increasingly using LMM as judger. However, existing benchmarks for video generation mainly assess perceptual quality and temporal coherence, yet fall short in evaluating higher-level reasoning abilities.

- 3 Method

- 3.1 Data Construction

With a primary focus on reasoning capabilities, we partition the dataset based on the types of reasoning knowledge involved. Specifically, as shown in Fig. 2, we define eight categories of reasoning knowledge, each category targeting a specific aspect of reasoning required for understanding or generating videos under structured constraints.

Commonsense Knowledge. This dimension evaluates whether video generation models encode and apply commonsense knowledge about everyday physics and human life, comprising three sub-aspects: 1) Physical commonsense assesses understanding of basic cause–effect relations, such as footprints left on snow or a vase breaking when hit; 2) Life commonsense examines knowledge of everyday biological responses, including skin swelling after a mosquito bite; 3) Healthcare commonsense evaluates familiarity with basic health practices, such as dental decay formation and mouthwash use. These sub-aspects together measure the model’s ability to reflect widely shared commonsense knowledge during video generation.

Subject Knowledge. Subject knowledge refers to structured, discipline-specific knowledge that extends beyond everyday experience and general common sense. It is organized into four sub-domains: 1) Physics evaluates understanding of fundamental physical principles across multiple subfields, including electricity, mechanics, and optics; 2) Chemistry examines knowledge of common chemical phenomena and reactions; 3) Geography involves a diverse range of topics such as celestial systems, river formations, and weather-related processes; 4) Sports focuses on generating subject-specific movements like soccer shooting, volleyball bump, and the iron cross in gymnastics.

Perceptual Knowledge. Accurate perception of basic visual attributes is a prerequisite for complex video generation. This dimension thus evaluates models’ capacity to capture and manipulate low-level perceptual semantics, including 1) size, 2) color, 3) count, and 4) position. In addition to these core perceptual attributes, we further introduce a more challenging sub-aspect, 5)occlusion, which assesses whether models can correctly infer and reconstruct objects that are partially occluded in the scene. These aspects probe the robustness of perceptual grounding required for reliable video generation.

Societal Knowledge.To generate videos that adhere to real-world social norms, we propose the societal knowledge dimension for assessing models’ understanding of social and cultural contexts. This dimension consists of three sub-aspects: 1) Emotion recognition infers emotional states from visual cues such as facial expressions; 2) Social rules captures commonly accepted behavioral norms, like disposing of trash properly or stopping at red lights; 3) Cultural customs reflects practices rooted in different societies, including dietary traditions and festival-related activities.

Logical Capability. Logical capability requires models to apply explicit rules systematically over visual elements, representing a challenging integrative aspect of reasoning due to the demand for structured, constraint-based inference. We divide this dimension into three sub-aspects: 1) Game actions evaluate whether models can follow the rules of classic game scenarios, such as Super Mario, and generate valid actions; 2) Puzzle solving

focuses on logic-driven scenarios such as mazes, board games (e.g., Gomoku), and word-linking puzzles, where correct generation depends on satisfying well-defined logical constraints; 3) Geometric Reasoning evaluates whether models can systematically reason under geometric rules and generate outputs that strictly adhere to the given structural constraints.

Experiential Knowledge. This dimension evaluates whether video generation models capture human-like experience-based knowledge for interpreting intentions, identities, procedures, and context. It comprises four sub-aspects: 1) Intention causality – inferring goals from intention cues (e.g., spoon near mouth implies eating);

- 2) Identity reasoning – identifying and tracking a specified individual among multiple entities; 3) Procedural knowledge – understanding correct action sequences (e.g., peeling before eating an orange); 4) Contextual knowledge – applying experiential knowledge based on textual scenario descriptions during generation.

Spatial Knowledge. This dimension evaluates a model’s ability to understand spatial relationships and to manipulate objects within a three-dimensional environment. Inspired by RISEBench [44], we decompose spatial knowledge into three aspects: 1) Viewpoint assesses whether models can perform viewpoint transformations by following a specified camera trajectory, as camera positioning and motion are critical factors in video generation; 2) Object arrangement examines whether multiple objects can be organized according to spatial attributes like relative size and shape; 3) Structural inference tests the capacity to integrate incomplete components into a spatially consistent structure.

Temporal Knowledge. This dimension evaluates temporal reasoning in video generation across different time spans and ordering patterns. We categorize temporal knowledge into four types: 1) short-term temporal reasoning, which involves events occurring within a few seconds, such as changes in traffic signal states within a five-second interval; 2) medium-term temporal reasoning, covering durations from minutes to several months;

- 3) long-term temporal reasoning, which requires understanding changes over periods exceeding one year; and 4) reverse temporal reasoning, where events are presented in reverse order to increase task difficulty, for example, an adult elephant gradually transforming as time rewinds over ten years.

Following the established categories, RISE-Video comprises 467 samples, each meticulously curated and annotated by human experts to ensure a diverse and representative coverage of reasoning scenarios.

#### 3.2 Evaluation Metrics

Evaluation metrics fundamentally determine both the capabilities assessed by a benchmark and the interpretation of model performance. As shown in Fig. 3, we evaluate model performance along four complementary dimensions, as demonstrated below:

Reasoning Alignment. This metric assesses whether the generated video demonstrates correct knowledge-based reasoning by evaluating the accuracy of inferred relationships, changes, and outcomes. To improve the accuracy and specificity of LMM-based evaluation under this metric, we adopt a targeted assessment strategy in which, for each sample, a set of manually designed, knowledge-aware questions is constructed according to the reasoning type being evaluated. The questions are answered by the LMM judge in a binary (Yes/No) manner, based on which each sample receives a 0–1 score for Reasoning Alignment. We further employ reasoning-aware frame sampling strategies to better support judgment under different temporal requirements: samples that require evaluating the full progression of an event are uniformly sampled at 2 fps, while scenarios that primarily focus on the final state (e.g., assessing whether a kitten becomes an adult cat after one year) adopt a lower sampling rate to emphasize terminal outcomes. This approach reduces redundant visual input and evaluation cost while preserving the information required for reliable judgment.

Within Logical Capability, we categorize tasks with abstract visual primitives including maze navigation, symmetry generation, and board games as Schematic Puzzles. These are ill-suited for standard LMM-as-a-Judge due to their rigid geometry and the difficulty of describing ground-truth (GT) states linguistically. We implement specialized strategies as shown in Fig. 4. For Maze Navigation, we bypass linguistic judging by tracking agent trajectories across all frames via color matching to verify two constraints: (1) no wall-crossing and (2) reach the target. The count of satisfied constraints {0,1,2} maps to scores {0,0.5,1}. For Symmetry Generation, to decouple Reasoning Alignment from Temporal Consistency, correctness is assessed via grid-level positional alignment between the last frame and the GT reference, disregarding specific color matches. Cells

[Figure 107]

Schematic Puzzles Maze Navigation

[Figure 108]

###### Reasoning Alignment

[Figure 109]

[Figure 110]

[Figure 111]

Questions: · Is the yellow lemon taken away? · Are the apples still staying in

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

No wall-crossing Reach target

Yes. No.

[Figure 116]

⋯

[Figure 117]

[Figure 118]

|[Figure 119]|
|---|

[Figure 120]

NYes NQuestions

[0, 0.5, 1] Num of Cond. Met Score

≤ 2FPS the same position? Score =

[Figure 121]

Traj

[0, 1, 2]

[Figure 122]

Temporal Consistency

[Figure 123]

Dst

[Figure 124]

[Figure 125]

Prompt: … Evaluate object consistency throughout the video using the … Instruction: A hand takes away the …

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

###### Symmetry Generation

[Figure 131]

Score 1~5

[Figure 132]

⋯

[Figure 133]

[Figure 134]

FP+FN Ncells

Acc=1−

Uniform 16 frames

[Figure 135]

[Figure 136]

[Figure 137]

[0, 0.5, 1] Score

Visual Quality

[Figure 138]

[0, 0.85), [0.85,1), 1

Prompt: … Your task is to evaluate the overall visual fidelity and technical quality of the batch of …

Last Ref.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Score 1~3

[Figure 144]

[Figure 145]

[Figure 146]

###### Board Games

[Figure 147]

Prompt: …Score similarity for the aspect/ region/attribute specified by …

Uniform 4 frames

[Figure 148]

[Figure 149]

[Figure 150]

Physical Rationality

[Figure 151]

[Figure 152]

[Figure 153]

1～5 Score

Prompt: … Evaluate the physical correctness of the provided video frames …

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Score 1~5

[Figure 159]

⋯

[Figure 160]

Last Ref.

[Figure 161]

Samples ∉ Logical Capability

Uniform 16 frames

[Figure 162]

[Figure 163]

[Figure 164]

Frame Extraction Metric Computation Score Mapping

[Figure 165]

[Figure 166]

###### Frame Extraction

LMM-As-Judge

Figure 3 Evaluation pipeline of the RISE-Video benchmark.

Figure 4 Specialized evaluation pipeline for reasoning alignment in Schematic Puzzle tasks, which are not well-suited for standard LMM-as-a-Judge evaluation, including trajectory-based constraint checking, grid-level structural alignment, and reference-assisted LMM comparison, enabling accurate and interpretable scoring of structured visual reasoning outcomes.

It covers four metrics: Reasoning Alignment, Temporal Consistency, Visual Quality, and Physical Rationality, with dimension-specific frame extraction strategies. Carefully designed prompts guide GPT-5 as the primary judge (GPT5-mini for Visual Quality only), ensuring fair and objective evaluation.

identified via HSV are used to determine False Positives (FP, misplaced) and False Negatives (FN, missing). Accuracy is then calculated as 1 − (FP + FN)/N, where N denotes the total cell count in the grid. This value is subsequently discretized into scores {0,0.5,1} based on the intervals [0,0.85), [0.85,1), and {1}; the

- 0.85 threshold reflects the human-perceptual boundary for structural trends. For Board Games, where rules are difficult to articulate linguistically, we provide the LMM judge with the last frame alongside an auxiliary GT reference image. This dual-input approach provides essential visual grounding, enabling the judge to perform a precise structural comparison between the generated output and the target state.

Temporal Consistency. Temporal Consistency evaluates whether the generated video exhibits only the changes explicitly or implicitly required by the instruction, while preserving all other aspects that are irrelevant to the instruction, such as object attributes, scene layout, and character identity. This metric emphasizes isolating instruction-induced changes from unintended variations. In practice, the instruction is provided to the LMM judge to explicitly identify and exclude the changes specified by the instruction, and to assess the consistency of all remaining elements in the generated video. To support this assessment, we apply uniform frame sampling to provide a representative and temporally distributed view of the content, balancing temporal coverage and evaluation efficiency. The judgment is reported on a 1–5 scale, reflecting the degree to which non-instructed components remain stable throughout the video.

Physical Rationality. Physical Rationality evaluates whether the generated video adheres to fundamental laws of physics and real-world logic, encompassing aspects such as gravity, object permanence, collision dynamics, and fluid motion. This metric emphasizes the plausibility of dynamic interactions and adherence to physical laws, ensuring objects maintain structural integrity and interact naturally. Besides, this metric is applicable strictly to physically grounded environments. Abstract tasks of planar logical puzzles or symbolic reasoning, which do not rely on real-world physical constraints, are excluded from this assessment. In practice, the LMM judge is instructed to verify the physical accuracy and confirm the logical coherence of movements

Table 1 Performance comparison of different models across four evaluation metrics and overall scores. Metrics include RA (Reasoning Alignment), TC (Temporal Consistency), PR (Physical Rationality), and VQ (Visual Quality). Overall performance is reported using W.Score (Weighted Score) and Accuracy.

Overall

Models RA TC PR VQ

W.Score Accuracy

▼ Closed-source Model

Hailuo2.3 [19] 76.6% 87.2% 71.0% 92.0% 79.4% 22.5% Veo3.1 [37] 64.9% 86.0% 78.9% 91.9% 76.4% 22.3% Sora-2 [23] 64.0% 92.2% 76.3% 92.2% 77.0% 21.3% Wan2.6 [35] 70.0% 88.8% 72.5% 94.5% 77.8% 21.3% Kling2.6 [32] 53.7% 86.4% 78.0% 95.1% 72.1% 19.5% Seedance1.5-pro [28] 61.2% 81.1% 70.7% 96.2% 72.0% 17.6% ▼ Open-source Model

Wan2.2-I2V-A14B [35] 39.5% 79.2% 75.4% 94.0% 63.9% 11.4% HunyuanVideo-1.5-720P-I2V [33] 38.1% 75.0% 68.4% 92.6% 60.4% 8.6% HunyuanVideo-1.5-720P-I2V-cfg-distill 38.9% 74.0% 65.8% 92.9% 59.9% 7.3% Wan2.2-TI2V-5B [35] 32.6% 70.5% 72.8% 89.7% 57.8% 5.4% CogVideoX1.5-5B [39] 30.7% 62.3% 56.7% 74.5% 49.5% 1.9%

and environmental reactions within the scene. To support the assessment, we apply uniform frame sampling to provide a representative and temporally distributed view of the motion dynamics, balancing temporal coverage and evaluation efficiency. The judgment is reported on a 1–5 scale, reflecting the degree to which the video maintains physical realism and temporal coherence without distortions.

Visual Quality. Visual Quality evaluates the perceptual fidelity and technical integrity of the generated video, focusing on subject sharpness, texture preservation, and lighting consistency. Notably, we manually apply super-resolution to low-clarity images prior to evaluation. This operation prevents the LMM judge from misinterpreting low native resolution as technical blur, allowing for a fairer assessment of the actual generative artifacts. In practice, the LMM judge assesses a batch of sampled frames to verify that the main subject remains crisp and structurally coherent throughout the sequence. To support this assessment, we uniformly sample 6 frames from the entire video and exclude the first and last frames to mitigate boundary instability. The judgment is reported on a 1–3 scale, classifying results from severe technical failure to professional-standard clarity.

Building on the above metrics, we introduce two types of overall scores to aggregate evaluation results. The first is Weighted Score, computed by assigning weights of 0.4, 0.25, 0.25, and 0.1 to Reasoning Alignment, Temporal Consistency, Physical Rationality, and Visual Quality, respectively. The second is Accuracy, where a case is counted as correct only if all four dimensions achieve full scores, and the resulting accuracy is normalized to a 100-point scale.

### 4 Experiments

To evaluate the reasoning capabilities of current TI2V models, we conduct experiments on 11 representative models, covering both closed-source and open-source systems. The closed-source models include Hailuo 2.3 [19], Wan2.6 [35], Sora 2 [23], Veo 3.1 [37], Kling 2.6 [32], and Seedance 1.5-pro [28], which typically demonstrate strong visual quality and reflect the current upper bound of deployed TI2V performance. In parallel, the open-source models span multiple architectures and training strategies, including Wan2.2-I2V [35], HunyuanVideo-1.5-720P-I2V [33] and its distilled variant, and CogVideoX1.5-5B [39]. For evaluation, GPT-5 is used as the judge for Reasoning Alignment, Temporal Consistency, and Physical Rationality, while GPT-5-mini is used for Visual Quality.

Table 2 Overall performance of different models across eight reasoning categories

Commonsense Subject Perceptual Societal Logical Experiential Spatial Temporal

Models

WS Acc. WS Acc. WS Acc. WS Acc. WS Acc. WS Acc. WS Acc. WS Acc.

▼ Closed-source Model

Hailuo 2.3 85.5 26.5 82.8 28.2 86.7 31.8 78.9 18.2 61.7 14.3 85.4 23.7 70.0 14.0 73.9 16.4 Veo 3.1 82.2 27.6 77.0 17.9 86.2 34.9 80.9 15.0 68.8 25.0 81.5 21.3 64.6 6.0 72.6 20.4 Sora 2 81.4 25.3 78.2 20.5 85.0 39.5 80.3 21.2 55.6 11.9 78.7 22.4 68.9 12.0 72.8 24.1 Wan 2.6 83.9 22.0 76.6 10.3 90.7 38.6 83.0 24.2 64.9 24.4 79.5 25.0 69.5 20.0 72.2 14.8 Kling 2.6 77.2 18.1 75.1 15.4 84.2 43.2 78.3 24.2 57.1 16.7 74.2 19.7 57.3 8.0 67.1 18.0 Seedance 1.5pro 76.0 18.0 74.0 11.5 85.1 36.4 77.9 21.2 46.1 9.8 78.3 21.1 62.6 18.0 66.7 9.8

▼ Open-source Model

Wan2.2-I2V-A14B 66.0 12.1 68.1 14.1 74.7 22.7 68.6 6.1 40.1 7.1 66.5 7.9 56.3 14.0 60.4 6.6 HunyuanVideo-1.5 64.7 6.0 62.5 3.9 68.4 18.2 58.2 3.0 45.1 12.2 65.3 14.5 51.0 10.0 57.0 3.3 HunyuanVideo-1.5(d) 66.1 7.2 62.4 5.1 65.3 13.6 58.9 12.1 45.5 2.4 61.7 9.2 52.8 8.0 56.7 3.3 Wan2.2-TI2V-5B 60.3 9.6 57.2 3.9 65.8 11.4 65.2 12.1 41.8 4.8 58.8 1.3 48.3 4.0 60.2 0.0 CogVideoX1.5-5B 54.9 1.2 54.1 1.3 62.1 6.8 54.9 3.0 29.7 0.0 48.2 1.3 36.2 0.0 47.2 3.3

#### 4.1 Main Results

- Tab. 1 summarizes the performance of all evaluated models across the four evaluation metrics. In general, open-source models consistently underperform closed-source models in both reasoning capability and visual quality. In particular, as shown in Fig. 1, models such as CogVideoX1.5 frequently exhibit visual artifacts, including frame-level blurring, ghosting effects, and degraded spatial sharpness, which lead to low Visual Quality scores and hinder reliable reasoning assessment. From the perspective of accuracy, all evaluated models achieve relatively low scores, indicating that reasoning remains a significant challenge for current TI2V systems. Even the best-performing model, Hailuo 2.3, attains an accuracy of only 22.5%. The secondand third-ranked models are Veo 3.1 and Sora 2, achieving accuracies of 22.3% and 21.3%, respectively. This highlights limitations in existing models’ ability to satisfy reasoning-oriented requirements. Among all evaluated models, Hailuo 2.3 demonstrates a particularly notable advantage in Reasoning Alignment, where it exceeds the second-ranked model Wan 2.6 by 6.6%. Notably, Sora 2 exhibits a clear strength in Temporal Consistency, suggesting that it is more effective at preserving non-instructed elements and maintaining stable generation behavior across videos.
- Tab. 2 reports the weighted scores and accuracies of all evaluated models across different reasoning categories. Overall, current TI2V models perform notably better on Perceptual Knowledge than on other reasoning types, indicating that models are relatively strong at perceiving low-level visual attributes such as color, size, and count. In contrast, performance on Logical Capability, which requires the integration of perceptual evidence with abstract reasoning, is consistently low across all models, suggesting that such tasks constitute a major bottleneck for current TI2V systems. As illustrated in Fig. 5, in the Gold Miner game scenario, where the hook is extended as shown and the model is required to generate the most likely grabbing process, none of the evaluated models successfully capture the stone along the current hook trajectory. Veo 3.1 exhibits consistency issues, with noticeable changes in the hook’s shape, while Kling 2.6 incorrectly moves the object without physical contact between the hook and the gold. This highlights the difficulty of rule-based decision-making in such game-like reasoning settings. Notably, in the Experiential category, Hailuo 2.3 and Veo 3.1 demonstrate clear advantages. As illustrated in Fig. 1, when generating a scenario in which a person drinks water from a bottle, only Veo 3.1 and Hailuo 2.3 are able to infer the necessary action of unscrewing the bottle cap, whereas other models fail to exhibit this reasoning behavior.

In terms of dynamic behavior, several models show limited responsiveness to instructions. For example, Kling 2.6 often produces videos with minimal motion or near-static content. As illustrated in Fig. 5 for the chameleon camouflage and capillary action of a rose tasks, Kling 2.6 tends to preserve the original appearance without performing the required commonsense transformation, resulting in both limited visual dynamics and poor alignment with the underlying reasoning requirement. While Wan 2.6 and Hailuo 2.3 demonstrate stronger instruction following and more dynamic generation behavior, Veo 3.1 and Sora 2 show relatively weaker responsiveness to dynamic instructions. In some cases, these models partially follow the instruction without fully realizing the intended transformation, or produce little effective change. For instance, as shown

###### Veo 3.1 Kling 2.6

###### Hailuo 2.3

[Figure 167]

[Figure 168]

[Figure 169]

A rose gradually undergoing capillary action.

[Figure 170]

[Figure 171]

[Figure 172]

It is gradually camouflaging itself.

[Figure 173]

[Figure 174]

[Figure 175]

Generate the result of connecting the circuit with a copper rod.

[Figure 176]

[Figure 177]

[Figure 178]

In the Gold Miner game, the hook is extended as shown, generate the most likely grab process.

Figure 5 Representative generation results of leading models. We show the examples generated by Hailuo 2.3, Veo 3.1, and Kling 2.6.

in Fig. 5, Veo 3.1 fails to correctly model the camouflage behavior, as the color of the chameleon does not sufficiently adapt to match the surrounding branch. In addition, Sora 2 and Veo 3.1 exhibit noticeable temporal discontinuities, characterized by abrupt changes between consecutive frames. Such discontinuities disrupt temporal smoothness and adversely affect overall video quality. Additional qualitative visualizations are provided in the supplementary material.

Table 3 Comparison of MAE and STD between LMM-as-Judge and human evaluations across different judge models.

RA TC PR VQ MAE (↓) STD (↓) MAE (↓) STD (↓) MAE (↓) STD (↓) MAE (↓) STD (↓)

Judge Model

Qwen3-VL-235B 0.12 0.25 0.42 0.80 0.82 0.86 0.25 0.41 Gemini-3-Flash 0.13 0.26 1.08 1.15 1.52 0.96 0.84 0.63 GPT-5†(Ours) 0.11 0.23 0.51 0.85 0.80 0.76 0.22 0.36

Note: † Specifically for the Visual Quality dimension in the GPT-5 row, we utilize gpt-5-mini to balance cost and performance.

#### 4.2 Ablation Study

In this section, we conduct human evaluations and calculate the Mean Absolute Error (MAE) and Standard Deviation (STD) between automatic metric scores and human ratings to identify the best metric for each category.

We evaluate each dimension on its native scale to preserve the granularity of specific tasks. Specifically, the value ranges are defined as follows: RA is scored in [0,1], Cons. and PR are assessed on a [1 − 5] scale, and VQ operates on a [1 − 3] scale. For the human ground truth, we employed five independent expert annotators and calculated the average of their ratings for each sample. Consequently, the MAE is computed as the mean absolute difference between the model’s predicted score and this aggregated human consensus, while the STD denotes the standard deviation of these absolute errors.

As illustrated in Tab. 3, GPT-5 demonstrates the most robust alignment with human preference across the majority of metrics. While Qwen3-VL-235B demonstrates a lower error rate on Temporal Consistency, a closer inspection suggests that this model exhibits a tendency towards higher acceptance rates (i.e., looser constraints on consistency). This high-score bias is accurate for perfect samples, but it compromises the

model’s ability to differentiate between truly high-quality outputs and those with severe defects. We show more cases in Fig A.3. Besides, our ablation on the Visual Quality dimension reveals that the cost-effective gpt-5-mini is highly capable of perceptual assessment; it achieves tighter alignment with human ratings than both Gemini-3-Flash and Qwen3-VL-235B.

### 5 Conclusion

In this work, we present RISE-Video, a reasoning-centric benchmark for TI2V models that systematically evaluates their ability to generate videos consistent with diverse reasoning requirements. By organizing data into eight complementary reasoning categories and conducting a comprehensive evaluation across four dimensions, our benchmark enables a holistic assessment of models beyond perceptual fidelity. We further introduce an automated LMM-based judging pipeline that supports scalable and fine-grained evaluation while maintaining a high degree of alignment with human judgments. Extensive evaluation on 11 representative TI2V models reveals that, despite strong perceptual quality, current systems continue to struggle with higher-level and implicit reasoning. These findings underscore the gap between visual realism and rule-consistent reasoning in current TI2V models. We hope that RISE-Video will facilitate more rigorous evaluation and inspire future research toward reasoning-aware TI2V model design and training.

### References

- [1] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.

- [2] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.

- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

- [4] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024.
- [5] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

- [6] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34:19822–19835, 2021.

- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Mu¨ller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [8] Weixi Feng, Jiachen Li, Michael Saxon, Tsu-jui Fu, Wenhu Chen, and William Yang Wang. Tc-bench: Benchmarking temporal compositionality in text-to-video and image-to-video generation. arXiv preprint arXiv:2406.08656, 2024.

- [9] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.

- [10] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [12] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

- [13] Haoyang Huang, Guoqing Ma, Nan Duan, Xing Chen, Changyi Wan, Ranchen Ming, Tianyu Wang, Bo Wang, Zhiying Lu, Aojie Li, et al. Step-video-ti2v technical report: A state-of-the-art text-driven image-to-video generation model. arXiv preprint arXiv:2503.11251, 2025.

- [14] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [15] Pengliang Ji, Chuyang Xiao, Huilin Tai, and Mingxiao Huo. T2vbench: Benchmarking temporal dynamics for text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5325–5335, 2024.

- [16] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose´ Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.

- [17] Mingxiang Liao, Qixiang Ye, Wangmeng Zuo, Fang Wan, Tianyu Wang, Yuzhong Zhao, Jingdong Wang, Xinyu Zhang, et al. Evaluation of text-to-video generation models: A dynamics perspective. Advances in Neural Information Processing Systems, 37:109790–109816, 2024.

- [18] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.

- [19] MiniMax. Hailuo 2.3, 2025. URL https://hailuoai.video/zh-Intl.
- [20] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? arXiv preprint arXiv:2501.09038, 2025.

- [21] Haomiao Ni, Bernhard Egger, Suhas Lohit, Anoop Cherian, Ye Wang, Toshiaki Koike-Akino, Sharon X Huang, and Tim K Marks. Ti2v-zero: Zero-shot image conditioning for text-to-video diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9015–9025, 2024.

- [22] Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, and Li Yuan. Video-bench: A comprehensive benchmark and toolkit for evaluating video-based large language models. Computational Visual Media, 2025.

- [23] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, Yuhui Wang, Anbang Ye, Gang Ren, Qianran Ma, Wanying Liang, Xiang Lian, Xiwen Wu, Yuting Zhong, Zhuangyan Li, Chaoyu Gong, Guojun Lei, Leijun Cheng, Limin Zhang, Minghao Li, Ruijie Zhang, Silan Hu, Shijie Huang, Xiaokang Wang, Yuanheng Zhao, Yuqi Wang, Ziang Wei, and Yang You. Open-sora 2.0: Training a commercial-level video generation model in 200k, 2025.
- [24] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

- [25] Runway. Runway-gen4.5, 2025. URL https://runwayml.com/research/introducing-runway-gen-4.5.
- [26] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

- [27] Enes Sanli, Baris Sarper Tezcan, Aykut Erdem, and Erkut Erdem. Can your model separate yolks with a water bottle? benchmarking physical commonsense understanding in video generation models. arXiv preprint arXiv:2507.15824, 2025.

- [28] Team Seedance, Heyi Chen, Siyan Chen, Xin Chen, Yanfei Chen, Ying Chen, Zhuo Chen, Feng Cheng, Tianheng Cheng, Xinqi Cheng, Xuyan Chi, Jian Cong, Jing Cui, Qinpeng Cui, Qide Dong, Junliang Fan, Jing Fang, Zetao Fang, Chengjian Feng, Han Feng, Mingyuan Gao, Yu Gao, Dong Guo, Qiushan Guo, Boyang Hao, Qingkai Hao, Bibo He, Qian He, Tuyen Hoang, Ruoqing Hu, Xi Hu, Weilin Huang, Zhaoyang Huang, Zhongyi Huang, Donglei Ji, Siqi Jiang, Wei Jiang, Yunpu Jiang, Zhuo Jiang, Ashley Kim, Jianan Kong, Zhichao Lai, Shanshan Lao,

- Yichong Leng, Ai Li, Feiya Li, Gen Li, Huixia Li, JiaShi Li, Liang Li, Ming Li, Shanshan Li, Tao Li, Xian Li, Xiaojie Li, Xiaoyang Li, Xingxing Li, Yameng Li, Yifu Li, Yiying Li, Chao Liang, Han Liang, Jianzhong Liang, Ying Liang, Zhiqiang Liang, Wang Liao, Yalin Liao, Heng Lin, Kengyu Lin, Shanchuan Lin, Xi Lin, Zhijie Lin, Feng Ling, Fangfang Liu, Gaohong Liu, Jiawei Liu, Jie Liu, Jihao Liu, Shouda Liu, Shu Liu, Sichao Liu, Songwei Liu, Xin Liu, Xue Liu, Yibo Liu, Zikun Liu, Zuxi Liu, Junlin Lyu, Lecheng Lyu, Qian Lyu, Han Mu, Xiaonan Nie, Jingzhe Ning, Xitong Pan, Yanghua Peng, Lianke Qin, Xueqiong Qu, Yuxi Ren, Kai Shen, Guang Shi, Lei Shi, Yan Song, Yinglong Song, Fan Sun, Li Sun, Renfei Sun, Yan Sun, Zeyu Sun, Wenjing Tang, Yaxue Tang, Zirui Tao, Feng Wang, Furui Wang, Jinran Wang, Junkai Wang, Ke Wang, Kexin Wang, Qingyi Wang, Rui Wang, Sen Wang, Shuai Wang, Tingru Wang, Weichen Wang, Xin Wang, Yanhui Wang, Yue Wang, Yuping Wang, Yuxuan Wang, Ziyu Wang, Guoqiang Wei, Wanru Wei, Di Wu, Guohong Wu, Hanjie Wu, Jian Wu, Jie Wu, Ruolan Wu, Xinglong Wu, Yonghui Wu, Ruiqi Xia, Liang Xiang, Fei Xiao, XueFeng Xiao, Pan Xie, Shuangyi Xie, Shuang Xu, Jinlan Xue, Shen Yan, Bangbang Yang, Ceyuan Yang, Jiaqi Yang, Runkai Yang, Tao Yang, Yang Yang, Yihang Yang, ZhiXian Yang, Ziyan Yang, Songting Yao, Yifan Yao, Zilyu Ye, Bowen Yu, Jian Yu, Chujie Yuan, Linxiao Yuan, Sichun Zeng, Weihong Zeng, Xuejiao Zeng, Yan Zeng, Chuntao Zhang, Heng Zhang, Jingjie Zhang, Kuo Zhang, Liang Zhang, Liying Zhang, Manlin Zhang, Ting Zhang, Weida Zhang, Xiaohe Zhang, Xinyan Zhang, Yan Zhang, Yuan Zhang, Zixiang Zhang, Fengxuan Zhao, Huating Zhao, Yang Zhao, Hao Zheng, Jianbin Zheng, Xiaozheng Zheng, Yangyang Zheng, Yijie Zheng, Jiexin Zhou, Jiahui Zhu, Kuan Zhu, Shenhan Zhu, Wenjia Zhu, Benhui Zou, and Feilong Zuo. Seedance 1.5 pro: A native audio-visual joint generation foundation model, 2025. URL https://arxiv.org/abs/2512.13507.
- [29] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

- [30] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

- [31] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8406–8416, 2025.

- [32] Kling Team, Jialu Chen, Yuanzheng Ci, Xiangyu Du, Zipeng Feng, Kun Gai, Sainan Guo, Feng Han, Jingbin He, Kang He, Xiao Hu, Xiaohua Hu, Boyuan Jiang, Fangyuan Kong, Hang Li, Jie Li, Qingyu Li, Shen Li, Xiaohan Li, Yan Li, Jiajun Liang, Borui Liao, Yiqiao Liao, Weihong Lin, Quande Liu, Xiaokun Liu, Yilun Liu, Yuliang Liu, Shun Lu, Hangyu Mao, Yunyao Mao, Haodong Ouyang, Wenyu Qin, Wanqi Shi, Xiaoyu Shi, Lianghao Su, Haozhi Sun, Peiqin Sun, Pengfei Wan, Chao Wang, Chenyu Wang, Meng Wang, Qiulin Wang, Runqi Wang, Xintao Wang, Xuebo Wang, Zekun Wang, Min Wei, Tiancheng Wen, Guohao Wu, Xiaoshi Wu, Zhenhua Wu, Da Xie, Yingtong Xiong, Yulong Xu, Sile Yang, Zikang Yang, Weicai Ye, Ziyang Yuan, Shenglong Zhang, Shuaiyu Zhang, Yuanxing Zhang, Yufan Zhang, Wenzheng Zhao, Ruiliang Zhou, Yan Zhou, Guosheng Zhu, and Yongjie Zhu. Kling-omni technical report, 2025. URL https://arxiv.org/abs/2512.16776.
- [33] Tencent Hunyuan Foundation Model Team. Hunyuanvideo 1.5 technical report, 2025. URL https://arxiv.org/ abs/2511.18870.
- [34] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019.
- [35] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025.
- [36] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.

- [37] Thadda¨us Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

- [38] Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang, Yixiao Ge, Xiaodong Cun, David Junhao Zhang, Jia-Wei Liu, Yuchao Gu, Rui Zhao, Weisi Lin, Wynne Hsu, Ying Shan, and Mike Zheng Shou. Towards a better metric for text-to-video generation, 2024. URL https://arxiv.org/abs/2401.07781.
- [39] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

- [40] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer, 2025.
- [41] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024.

- [42] Ailing Zhang, Lina Lei, Dehong Kong, Zhixin Wang, Jiaqi Xu, Fenglong Song, Chun-Le Guo, Chang Liu, Fan Li, and Jie Chen. Ui2v-bench: An understanding-based image-to-video generation benchmark. arXiv preprint arXiv:2509.24427, 2025.

- [43] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation, 2023.
- [44] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.

- [45] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

## Appendix

- A.1 Data Source Input images for our benchmark are primarily sourced from the following categories:

- 1. Images generated by high-quality image generation models, selected to ensure sufficient visual fidelity and diversity for downstream TI2V tasks.
- 2. Images obtained from websites with permissive licenses, collected in accordance with their respective usage terms.
- 3. Images manually curated from the RISEBench dataset, where suitable samples are identified and adapted for transfer to TI2V reasoning tasks.

A.1.1 Privacy-Preserving Image Stylization

Our dataset includes tasks involving human activities, some of which contain images of real individuals. To mitigate potential privacy concerns, we apply image stylization to tasks where preserving the original appearance of real persons is not essential for reasoning evaluation. This processing removes identifiable visual details while retaining the structural and semantic information required by the task, ensuring that evaluation remains valid without exposing sensitive personal information.

- A.2 Prompt for Judgement Prompt for evaluating Reasoning Alignment without reference

You are a video understanding assistant. Answer the user's questions and explain the reasons based ONLY on the provided video frames. Do NOT guess or hallucinate. For each queation, answer strictly in JSON Format: [{"question": "repeat the question", "answer":"Yes or No", "reason":"the reason"}] For each input video, if there are multiple questions, you MUST return the answers as a JSON list of dictionaries. Example output: [ {

"question": "repeat the question", "answer": "Yes or No", "reason": "the reason"

}, {

"question": "repeat the question", "answer": "Yes or No", "reason": "the reason"

} ] Do NOT wrap the JSON output in markdown code blocks (no ```json, no ```). Return only a valid JSON array.

###### Prompt for evaluating Reasoning Alignment with reference

# Video Evaluation Instruction You are a strict **visual judge**. You will receive two images:

- - **First image**: the generated video’s final frame
- - **Second GT image**: the reference label image
- - **FocusQuestion**: a short question that **specifies which part/region/attribute** of the LastFrame must match GT (e.g., "Is the top-right button label the same as GT?") Judge **only** what is visible in these two images. Do NOT guess or hallucinate.

## Scoring (concise 1-5) Score similarity for the aspect/region/attribute specified by **FocusQuestion**:

- - **5** - Perfect match: all key objects, attributes, and layout align; no extra/missing elements.
- - **4** - Mostly match: only minor deviations; structure intact; no extra/missing

**main** elements.

- - **3** - Partial match: several noticeable differences; core objects/layout still largely present.
- - **2** - Major mismatch: missing/extra main elements, wrong layout/relations, or multiple attribute errors.
- - **1** - Unrelated/unjudgeable: object types don't match, layout invalid, or images unreadable.

## Output (strict JSON) Example output: {

"Question": "repeat the question", "Score": 1-5, "Reason": "the reason"

} Do NOT wrap the JSON output in markdown code blocks (no ```json, no ```). Return only a valid JSON dictionary.

###### Prompt for evaluating Temporal Consistency

# Video Object Consistency Evaluation Instruction You are a highly skilled **video evaluator**. You will receive a video clip and a specific **instruction**. The video may depict an evolving scene, but your task is

**ONLY** to evaluate whether the **objects remain visually and semantically consistent across frames**, **except** for changes that are explicitly required or implied by the instruction.

## Task Evaluate **object consistency throughout the video** using the following 1-5 scale:

- - **5 (Perfect Consistency)** Apart from changes required by the instruction (e.g., motion, action, time progression), all other details-object identity, personal features, colors, shapes, background, and spatial layout-remain completely stable across all frames.
- - **4 (Minor Differences)** Mostly consistent with only one minor temporal discrepancy not implied by the instruction (e.g., brief lighting flicker, a momentarily missing accessory).
- - **3 (Noticeable Differences)**

- One **noticeable inconsistency** across frames (e.g., attribute shifts briefly, background element jumps).
- - **2 (Significant Differences)**

**Two or more** inconsistencies (e.g., appearance changes and environment changes, an object identity briefly swaps/disappears, or appearance of unexpected new objects).

- - **1 (Severe Differences)** Visual/semantic continuity repeatedly breaks. Key identities or scene attributes (e.g., major appearance features, background layout) change drastically, clearly deviating from intended continuity.

## Example

**Instruction:** Two women-one in a black dress and one in a white dress-are sitting on a bench. The woman in the black dress stands up.

- - **Score 5 | Perfect Consistency** Both women's clothing, hairstyles, skin tones, and body shapes remain stable; the bench texture and background stay unchanged; only the black-dress woman smoothly transitions from sitting to standing with no flicker or jumps.
- - **Score 4 | Minor Differences** Overall consistent; the black-dress woman stands normally. There is a single brief exposure flicker (or the white-dress woman's earring is momentarily occluded for one frame) that immediately returns to normal, without affecting identity or layout stability.
- - **Score 3 | Noticeable Differences** The stand-up motion is correct, but during a segment the black dress shifts slightly toward gray for a few frames and then reverts; identities and scene layout remain stable, with only this one brief, localized inconsistency.
- - **Score 2 | Significant Differences** Two issues or more prolonged issue: the black-dress woman's hair length repeatedly shortens and returns over many frames, and the bench wood grain changes at several moments; identities are still recognizable and the scene is not fundamentally reconfigured.
- - **Score 1 | Severe Differences** Identity- or scene-level failures: the black-dress woman morphs into a different person or swaps dress colors with the white-dress woman, the white-dress woman disappears or teleports, and the background jumps from a park bench to an indoor hallway-continuity is clearly broken. ## Notes
- - **Ignore** changes explicitly stated or implied by the **instruction**.
- - Focus on unintended issues: identity drift, texture flicker, background jump, spatial discontinuity, or attribute change (e,g, color, size, count and so on).
- - **DO NOT** judge whether the video follows the instructions. Only evaluate based on object consistency for scoring.

## Input

**Instruction:** {instruct} ## Output Format (**strict JSON**) {{

"Instruction": "Repeat the instruction you received", "Final Score": 1-5,

"Reason": "A concise 1-2 sentence analysis to support your score"

}} Do NOT wrap the JSON output in markdown code blocks (no ```json, no ```). Return only a valid **JSON dictionary**.

###### Prompt for evaluating Physical Rationality

**Role:** You are a rigorous physics and visual effects analyst.

**Objective:** Evaluate the physical correctness of the provided video frames. ### Evaluation Rubric (Amplitude-Aware)

- **1 (Scene Broken):** Scene jumps to unrelated content. Common-sense continuity of both the main subject and the background is lost.
- **2 (Severe & Large-Amplitude Errors):** Persistent, large-amplitude physical failures in the main subject or core interaction (e.g., deep clipping, structural break, rigid bodies melting, sudden appearing/vanishing). Immediately breaks immersion.
- **3 (Noticeable & Medium/Large Amplitude):** Medium to large-amplitude physical violations in the main subject **or background** (e.g., clear distortion, unnatural fluid, objects popping in/out, abrupt trajectory/velocity change). Semantics still understandable, realism reduced.
- **4 (Minor & Small Amplitude, Needs Review):** Small-amplitude physical artifacts in the main subject **or background** (e.g., slight texture shimmering/flicker, minor liquid jitter). Does not block understanding, often requires replay to confirm.
- **5 (Physically Seamless):** No perceivable physical errors. Motion, contact, fluidity, object permanence, and material state transitions feel naturally continuous. ### Requirements

- Respond with **one valid JSON object**:

**Example Output:** {{

"score": 2, "justification": "The object clipped deeply through the surface and cast no shadow."

}}

###### Prompt for evaluating Visual Quality

**Role:** You are a meticulous Image Quality Analyst.

**Objective:** Your task is to evaluate the **overall visual fidelity** and technical quality of the **batch of {num_frames} image frames** provided. These frames are sampled from a single video clip.

**CRITICAL RULES:**

- 1. **Ignore Artistic Blur:** Do NOT penalize background bokeh/depth-of-field.
- 2. **Ignore Occlusion:** Do NOT penalize if the subject is partially blocked.

### Core Evaluation Criteria Critically assess these aspects across all provided frames to determine your

**average score**.

- 1. **Subject Sharpness & Clarity:**

- * Are the **visible portions** of the **Main Subject** crisp and defined (on average)?
- * Are fine details preserved?
- * Are the frames free from global "softness" or low-resolution haziness?

- 2. **Artifacts & Distortion:**

- - **AI Artifacts:** Are there "melting" textures, distorted faces/hands?
- - **Compression:** Are there visible blocks, banding, or ringing artifacts?
- - **Noise:** Is there unintended grain that degrades the details?

- 3. **Lighting & Visual Integrity:**

- - Is the exposure balanced (subject is visible)?
- - Are colors natural and consistent? ### EVALUATION RUBRIC (Strict 1-3 Scale)
- - **1 (Reject / Unusable):**

**Severe Technical Failure.** The main subject is unrecognizable, heavily blurred (technical blur), or suffers from gross AI distortions (melted faces/limbs). The image is broken.

- - **2 (Passable / Average):**

**Noticeable Imperfections.** The subject is structurally correct but lacks fine detail. Looks "soft," "waxy," or has visible noise/artifacts. Usable, but clearly digital or low-res.

- - **3 (Excellent / High Quality):**

**Professional Standard.** The main subject is **razor-sharp** with rich textures (hair/skin visible). No visible noise, compression, or AI artifacts. Looks like high-end photography.

### Output Format Return a single JSON object with the integer score (1, 2, or 3).

**Example:** {{

"score": 3, "justification": "Subject is razor-sharp. No artifacts."

}}

#### A.3 Analysis on the Judge Models

Instruction: Generate the most natural continuation of the action, illustrating what would reasonably occur after the moment captured in the initial frame.

[Figure 179]

Temporal Consistency Score (1~5 Scale)

Qwen3-VL-235B Score:5 Reason: All cubes maintain consistent color, shape, and identity throughout; their stacking sequence follows the instruction without any unintended visual or spatial inconsistencies.

Gemini-3-Flash

- Score:2 Reason: There are multiple significant inconsistencies where the cubes change shape and size during movement, such as the green cube momentarily extending into a longer rectangular brick and the white cube becoming tall and thin before reverting to its original form.

GPT-5

- Score:3 Reason: Objects are mostly consistent in color and identity while moving, but the green cube briefly turns into a longer rectangular brick with more studs before reverting. Aside from this noticeable shape change, the scene remains stable.

Figure 6 Qualitative comparison across different judge models on the temporal consistency.

Instruction: Generate a reasonable continuation that shows what naturally happens in the very next moment after the given scene.

[Figure 180]

Physical Rationality Score (1~5 Scale)

Qwen3-VL-235B Score:5 Reason: All cubes maintain consistent color, shape, and identity throughout; their stacking sequence follows the instruction without any unintended visual or spatial inconsistencies.

Gemini-3-Flash Score:2 Reason: The video exhibits severe physical errors during the core interaction. As the thumb presses down, both the thumb and the lighter's plastic casing lose their structural integrity, appearing to 'melt' and clip into each other. The purple plastic deforms in a non-mechanical, fluid-like way that contradicts the properties of a rigid object. Additionally, the flame appears detached from the lighter's nozzle before stabilizing, further breaking the physical realism of the scene.

GPT-5 Score:3 Reason: Hand grip and lighter mechanism open smoothly with consistent motion. The main issue is the flame: it briefly appears as a large orange blob not anchored to the nozzle and then snaps back to a normal blue/orange jet, which is a noticeable physical violation. There is also a stray spark-like artifact away from the lighter. Aside from these, continuity is maintained.

Figure 7 Qualitative comparison across different judge models on the physical rationality.

Instruction: The experimental process of generating the knee-jerk reflex.

[Figure 181]

###### Image Quality Score (1~3 Scale)

Qwen3-VL-235B Score:3 Reason: Subject is razor-sharp. No artifacts.

Gemini-3-Flash

- Score:1 Reason: The frames exhibit severe AI artifacts. In the first frame, the subject's arms are partially missing or 'melting' into the background. The second frame shows distorted, blob-like hands and a 'ghost' artifact where the ball should be. Overall, the subject lacks fine detail and sharpness.

GPT-5

- Score:2 Reason: The subject's torso, legs and footwear are generally well defined with preserved texture in clothing, but several frames show noticeable motion blur (hands/ball) and mild overall softness compared with professional razor-sharp imagery. No obvious AI melting or severe distortion; noise is low and compression artifacts are minor. Exposure and color are consistent and natural. Usable but not studio-perfect.

Figure 8 Qualitative comparison across different judge models on the visual quality.

#### A.4 More Vsualizations

[Figure 182]

Perceptual Knowledge: As the vehicle continues moving forward, it becomes fully visible.

(a) Frame0 and Instruction of TI2V.

[Figure 183]

- (b) Hailuo 2.3

[Figure 184]

- (c) Wan 2.6

[Figure 185]

- (d) Veo 3.1

[Figure 186]

- (e) Sora 2

[Figure 187]

(f) Seedance 1.5pro

[Figure 188]

(g) Kling 2.6

[Figure 189]

- (h) Wan 2.2-I2V-A14B

[Figure 190]

- (i) Wan 2.2-TI2V-5B

[Figure 191]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 192]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 193]

(l) CogVideoX1.5-5B

- Figure 9 Qualitative examples of generation results from different TI2V models on Perceptual Knowledge tasks.

[Figure 194]

Commonsense Knowledge: This is a normal person's toe bone. Generate a video showing the changes in this person's toe joints under long-term high uric acid conditions.

(a) Frame0 and Instruction of TI2V.

[Figure 195]

(b) Hailuo 2.3

[Figure 196]

- (c) Wan 2.6

[Figure 197]

- (d) Veo 3.1

[Figure 198]

- (e) Sora 2

[Figure 199]

(f) Seedance 1.5pro

[Figure 200]

(g) Kling 2.6

[Figure 201]

(h) Wan 2.2-I2V-A14B

[Figure 202]

(i) Wan 2.2-TI2V-5B

[Figure 203]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 204]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 205]

(l) CogVideoX1.5-5B

- Figure 10 Qualitative examples of generation results from different TI2V models on Commonsense Knowledge tasks.

[Figure 206]

Temporal Knowledge: Show the process of time reversing over five seconds.

(a) Frame0 and Instruction of TI2V.

[Figure 207]

- (b) Hailuo 2.3

[Figure 208]

- (c) Wan 2.6

[Figure 209]

- (d) Veo 3.1

[Figure 210]

- (e) Sora 2

[Figure 211]

(f) Seedance 1.5pro

[Figure 212]

(g) Kling 2.6

[Figure 213]

- (h) Wan 2.2-I2V-A14B

[Figure 214]

- (i) Wan 2.2-TI2V-5B

[Figure 215]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 216]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 217]

(l) CogVideoX1.5-5B

- Figure 11 Qualitative examples of generation results from different TI2V models on Temporal Knowledge tasks.

[Figure 218]

Experiential Knowledge: Show the process of the hands taking a letter out of it.

(a) Frame0 and Instruction of TI2V.

[Figure 219]

(b) Hailuo 2.3

[Figure 220]

- (c) Wan 2.6

[Figure 221]

- (d) Veo 3.1

[Figure 222]

- (e) Sora 2

[Figure 223]

(f) Seedance 1.5pro

[Figure 224]

(g) Kling 2.6

[Figure 225]

(h) Wan 2.2-I2V-A14B

[Figure 226]

(i) Wan 2.2-TI2V-5B

[Figure 227]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 228]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 229]

(l) CogVideoX1.5-5B

- Figure 12 Qualitative examples of generation results from different TI2V models on Experiential Knowledge tasks.

[Figure 230]

Logical Capability: The faucet turns on, and the water flows into all the correct containers.

(a) Frame0 and Instruction of TI2V.

[Figure 231]

(b) Hailuo 2.3

[Figure 232]

- (c) Wan 2.6

[Figure 233]

- (d) Veo 3.1

[Figure 234]

- (e) Sora 2

[Figure 235]

(f) Seedance 1.5pro

[Figure 236]

(g) Kling 2.6

[Figure 237]

(h) Wan 2.2-I2V-A14B

[Figure 238]

(i) Wan 2.2-TI2V-5B

[Figure 239]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 240]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 241]

(l) CogVideoX1.5-5B

- Figure 13 Qualitative examples of generation results from different TI2V models on Logical Capability tasks.

[Figure 242]

Societal Knowledge: Place the single most iconic traditional American Thanksgiving main food on the plate.

(a) Frame0 and Instruction of TI2V.

[Figure 243]

(b) Hailuo 2.3

[Figure 244]

- (c) Wan 2.6

[Figure 245]

- (d) Veo 3.1

[Figure 246]

- (e) Sora 2

[Figure 247]

(f) Seedance 1.5pro

[Figure 248]

(g) Kling 2.6

[Figure 249]

(h) Wan 2.2-I2V-A14B

[Figure 250]

(i) Wan 2.2-TI2V-5B

[Figure 251]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 252]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 253]

(l) CogVideoX1.5-5B

- Figure 14 Qualitative examples of generation results from different TI2V models on Societal Knowledge tasks.

[Figure 254]

Spatial Knowledge: The given cookies are arranged from left to right in the following shape order: circle, square, star, heart.

(a) Frame0 and Instruction of TI2V.

[Figure 255]

(b) Hailuo 2.3

[Figure 256]

- (c) Wan 2.6

[Figure 257]

- (d) Veo 3.1

[Figure 258]

- (e) Sora 2

[Figure 259]

(f) Seedance 1.5pro

[Figure 260]

(g) Kling 2.6

[Figure 261]

(h) Wan 2.2-I2V-A14B

[Figure 262]

(i) Wan 2.2-TI2V-5B

[Figure 263]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 264]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 265]

(l) CogVideoX1.5-5B

- Figure 15 Qualitative examples of generation results from different TI2V models on Spatial Knowledge tasks.

[Figure 266]

Subject Knowledge: Show the dynamic process after the phenolphthalein solution from the dropper is added into the beaker containing sodium hydroxide solution.

(a) Frame0 and Instruction of TI2V.

[Figure 267]

(b) Hailuo 2.3

[Figure 268]

- (c) Wan 2.6

[Figure 269]

- (d) Veo 3.1

[Figure 270]

- (e) Sora 2

[Figure 271]

(f) Seedance 1.5pro

[Figure 272]

(g) Kling 2.6

[Figure 273]

(h) Wan 2.2-I2V-A14B

[Figure 274]

(i) Wan 2.2-TI2V-5B

[Figure 275]

(j) HunyuanVideo-1.5-720P-I2V

[Figure 276]

(k) HunyuanVideo-1.5-720P-I2V-cfg-distill

[Figure 277]

(l) CogVideoX1.5-5B

- Figure 16 Qualitative examples of generation results from different TI2V models on Subject Knowledge tasks.

