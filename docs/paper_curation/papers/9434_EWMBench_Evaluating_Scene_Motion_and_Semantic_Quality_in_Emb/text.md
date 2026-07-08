arXiv:2505.09694v2[cs.RO]18May2025

# EWMBENCH: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models

Yue Hu1,4,∗ Siyuan Huang2,∗ Yue Liao3 Shengcong Chen1 Pengfei Zhou1

Liliang Chen1,† Maoqing Yao1,⋄ Guanghui Ren1,⋄

## Abstract

Recent advances in creative AI have enabled the synthesis of high-fidelity images and videos conditioned on language instructions. Building on these developments, text-to-video diffusion models have evolved into embodied world models (EWMs) capable of generating physically plausible scenes from language commands, effectively bridging vision and action in embodied AI applications. This work addresses the critical challenge of evaluating EWMs beyond general perceptual metrics to ensure the generation of physically grounded and action-consistent behaviors. We propose the Embodied World Model Benchmark (EWMBENCH), a dedicated framework designed to evaluate EWMs based on three key aspects: visual scene consistency, motion correctness, and semantic alignment. Our approach leverages a meticulously curated dataset encompassing diverse scenes and motion patterns, alongside a comprehensive multi-dimensional evaluation toolkit, to assess and compare candidate models. The proposed benchmark not only identifies the limitations of existing video generation models in meeting the unique requirements of embodied tasks but also provides valuable insights to guide future advancements in the field. The dataset and evaluation tools are publicly available at https://github.com/AgibotTech/EWMBench.

## 1 Introduction

General Generation Video

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Embodied Generation Video

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 1: Comparison between general video generation and embodied video generation. Unlike general videos, embodied videos typically feature more structured scenes, consistent motion patterns, and clearer task logic.

∗*Equal contribution. †Project leader. ⋄Corresponding authors. 1†AgiBot. 2SJTU. 3MMLab-CUHK. 4HIT.

Preprint.

Creative AI has advanced rapidly in recent years, propelled by innovations in model architectures—such as variational autoencoders (VAEs) and diffusion models—increased parameter scaling, and the availability of large-scale, high-quality datasets. These developments have empowered generative models to synthesize images and videos conditioned on language instructions with unprecedented fidelity and controllability. Building on the momentum of text-to-video diffusion models, recent efforts have expanded their scope from generating high-fidelity, high-resolution videos to serving as embodied world models (EWMs) capable of synthesizing physically actionable scenes from language instructions (e.g., “move the robot arm approaching the cup”) or physical action instructions, i.e., an action policy sequence. This emerging capability establishes a critical link between vision and action in embodied AI, facilitating applications such as robotic manipulation, where instruction-conditioned trajectories must conform to physical and kinematic constraints.

Despite advancements in EWMs, a fundamental question remains unresolved: “How can we determine whether a video generation model qualifies as a good embodied world model, beyond merely serving as a general-purpose video generator?” Addressing this question is essential for guiding model development and assessing a model’s ability to produce physically grounded, action-consistent behaviors. While existing video generation benchmarks [18] focus on perceptual metrics like visual fidelity, language alignment, and human preference, these criteria are insufficient for evaluating EWMs. Embodied generation tasks have unique requirements, such as coherence in embodiment motion and plausibility in action execution, as illustrated in Figure 1. For instance, in robotic manipulation scenarios, the background, object configuration, and embodiment structure (e.g., robot morphology) are expected to remain static, while only the robot’s pose and interactions evolve according to instructions. This structured realism sets EWMs apart from general video generators and demands evaluation beyond conventional criteria.

In this work, we introduce a dedicated benchmark, Embodied World Model Benchmark (EWMBENCH), to systematically assess embodiment motion fidelity and spatiotemporal consistency in robotic manipulation. We first formalize the benchmarking setup for EWMs. Given an initial video segment that specifies the embodiment (e.g., a robotic arm) and the environment, along with a manipulation instruction, the candidate EWM is tasked with autoregressively generating future frames depicting the embodiment’s motion until the instruction is completed. We design an evaluation protocol based on three key aspects: (1) Visual Scene Consistency, ensuring static elements like the background, objects, and embodiment structure remain unchanged during motion; (2) Motion Correctness, requiring the generated embodiment trajectory to be coherent and aligned with the task objective; and (3) Semantic Alignment and Diversity, assessing the model’s alignment with linguistic instructions and its ability to generalize across diverse tasks. For these aspects, we develop systematic evaluation tools, including prompt engineering with video-based MLLMs.

To benchmark EWMs under our proposed criteria, we construct a comprehensive colosseum consisting of a curated benchmark dataset and open-source evaluation tools. The dataset is built on AgibotWorld [2], the largest real-world robotic manipulation dataset, featuring diverse tasks at scale. We select 30 candidate samples across ten tasks with clear action-ordering constraints, where correct execution requires understanding logical dependencies and affordances—posing significant challenges for embodied video generation. For each sample, static initial frames are clipped to ensure subsequent frames strictly reflect annotated language instructions without redundant movements. To reflect task diversity, we account for cases where multiple trajectories achieve the same goal and incorporate voxelized scoring to encourage variation. With this dataset, initial frames and language instructions are fed into different video generators, and the generated videos are compared against ground truth (GT) using various metrics.

Contributions: We summarize our contributions as follows: (1) We propose the first world generation benchmark tailored for embodied tasks, EWMBENCH. (2)We curate a high-quality, diverse dataset for our benchmark evaluation. (3) We introduce and open-source the systematical evaluation metrics, which covers key aspects in the embodied world model generation. (4) we provide insights into the performance of existing video models on embodied generation tasks.

## 2 Related Works

- 2.1 Video Generative Models

Diffusion-based video generation models have made significant advances in recent years, particularly in text-to-image (T2I) generation [5, 16, 33]and text-to-video (T2V) [41, 8, 30, 12, 38] generation. Recent works [21, 42, 4] have explored the usage of Diffusion Transformers in the denoising process. Some researchers [7] in robot learning aim to leverage this knowledge to address the challenge of data scarcity in robotic data collection. In some works [6, 39]video models are employed to predict and simulate future states of dynamic systems. In robotics, video models have been utilized to predict future frames based on textual and visual inputs [43, 17, 10].

- 2.2 Evaluation of Video Generative Models

Motion Scene Semantic

Benchmark

TrajD TrajP SceneC DivM TempC InterL ImgC SemS

TC-Bench [11] ✗ ✗ ✓ ✓ ✓ ✓ ✓ ✗ Physics-IQ [26] ✓ ✓ ✗ ✗ ✓ ✓ ✓ ✗ VBench [18] ✗ ✗ ✓ ✗ ✗ ✗ ✗ ✓ VBench++ [19] ✗ ✗ ✓ ✗ ✗ ✗ ✓ ✓ PhyGenBench [25] ✗ ✗ ✗ ✗ ✗ ✗ ✗ ✓ T2V-CompBench [34] ✗ ✓ ✗ ✓ ✗ ✓ ✗ ✓ VMBench [23] ✓ ✗ ✓ ✗ ✗ ✗ ✓ ✗ EvalCrafter [24] ✗ ✗ ✓ ✓ ✓ ✓ ✗ ✓ T2VBench [20] ✗ ✗ ✓ ✓ ✓ ✓ ✗ ✗ EVA [10] ✗ ✓ ✓ ✗ ✓ ✓ ✓ ✓ Ours ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

Table 1: Comparison of video generation benchmarks on 8 key evaluation dimensions: trajectory dynamics (TrajD), trajectory plausibility (TrajP), scene consistency (SceneC), diversity (DivM), temporal causality (TempC), interaction logic (InterL), image quality (ImgC), and semantic similarity (SemS).

With the rapid development of video generation models and their broad applications across various domains, evaluating these models has become increasingly important. Earlier approaches primarily relied on conventional metrics such as Fréchet Inception Distance (FID)[15], Inception Score (IS)[31], and Fréchet Video Distance (FVD) [35]. However, these metrics primarily focus on visual appearance and offer limited insights into the diverse and complex capabilities of modern video generation models. Recent evaluation frameworks [18, 19] introduced a more structured approach that considers multiple capability dimensions. Nevertheless, these benchmarks still emphasize visual quality, including aesthetic appeal and motion smoothness. To address these limitations, specialized benchmarks have emerged. For instance, PhyGenBench [25] evaluates a model’s understanding of physical laws using Vision-Language Models (VLMs), while T2V-CompBench [34] assesses compositionality, covering motion, actions, spatial relationships, and attributes.

Although these efforts have significantly expanded evaluation dimensions, they remain focused on general video generation. In the context of world models, particularly in the embodied domain, video generation requires enhanced controllability, physical plausibility, and robust object interactions. However, previous works have generally overlooked critical factors such as action plausibility, object interactions, and manipulation. To bridge this gap, we propose EWMBENCH, a systematic framework for evaluating embodied world models. A detailed comparison with existing benchmarks is provided in Table 1.

###### Evaluation Metrics

###### World Initialization

###### Generative Models

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

- - Image-Text to Video Models
- - Action Conditional Video Models

Init. Scene Image(s)

Semantics

Scene

Motion

Diver. / Logics …

Consisteny / …

Align. / Dyna. …

###### Generated Video Frames

Task Instructions

Hang the shower head… Action Trajectory

[Figure 21]

…

[Figure 22]

L. R.

- Figure 2: Overview of the EWMBENCH benchmark design. The framework begins with unified world initialization, where generative models are instructed to produce predictive video frames based on initial scene images, task instructions, and optional action trajectories. These generated video frames are subsequently evaluated using multi-dimensional metrics, focusing on scene consistency, motion dynamics, and semantic alignment.
- 3 The EWMBENCH Benchmark

### 3.1 EWMBENCH Overview

Overview The primary objective of this work is to establish a comprehensive evaluation benchmark for embodied world model generation. EWMBENCH introduces three core components: (1) a unified world initialization, (2) a meticulously curated dataset for embodied manipulation tasks, and (3) systematic evaluation metrics. The complete pipeline is illustrated in Figure 2. The world is initialized using initial scene images and corresponding task instructions, with the sampled action trajectory being optional for generative models. Leveraging these unified input modalities, various generative models are instructed to produce video frames, while some contextual modalities may remain optional depending on the model. The outputs generated are evaluated using EWMBENCH metrics, which focus on three critical factors: scene, motion, and semantics. These factors collectively form the foundation for evaluating robotic tasks. In this work, we primarily focus on robotic manipulation tasks, which are the most dominant and representative within the domain of embodied tasks. We plan to extend this framework to broader tasks in future work.

Evaluation Task Formulation An embodied world model generates a video as expressed in Equation 1, where I, L, and T represent the input context image, language, and trajectory, respectively. We provide this unified information, including up to four initial images. The action trajectory, for-

matted as a sequence of 6D poses, is optional for generation model inference. The function fproc represents model-specific preprocessing. To ensure fairness, we apply vnorm to normalize the raw generated video frames before evaluation. The evaluation framework comprises a multi-modal LLM for high-level semantic analysis, a trajectory detector for low-level trajectory-based evaluation, and several visual foundation models for visual feature processing.

### 3.2 Dataset Construction

V = vnorm(gworld(fproc(I,L,T ))) (1)

We developed our evaluation dataset using the open-source Agibot-World dataset. Ten tasks were carefully selected based on their clear operational goals and sequential dependencies, covering both household and industrial contexts. These tasks emphasize action-ordering constraints that require reasoning about affordances and procedural contexts. To ensure diverse motion patterns, action trajectories were encoded into voxel grids, and a greedy algorithm was employed to select the most diverse trajectories for each task. An overview of the constructed dataset is in Figure 3. Details on task descriptions can be found in the Appendix.

For fine-grained evaluation, we adopted a task-oriented decomposition strategy. Each high-level task was broken down into a sequence of 4 to 10 atomic sub-actions, with each sub-action paired with a step-level caption. This approach guarantees a one-to-one alignment between video segments, sub-action labels, and their corresponding linguistic descriptions.

[Figure 23]

###### Diverse Task Scenes

###### Diverse Action Motions Diverse Semantics

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Articulated

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Rigid

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### Fluid

Slide Deformable

Figure 3: Overview of the constructed dataset. Left: Task scenes spanning household, commercial, and industrial environments. Middle: Diverse task-specific trajectory variations within each scene. Right: Broad semantic coverage across various manipulation contexts.

### 3.3 Evaluation Metrics

EWMBENCH systematically evaluates three dimensions to ensure the generated outputs are visually realistic, action plausible, and semantically meaningful. We leave the metric definition details in the Appendix.

- A. Scene. We introduce the Scene Consistency metric, which examines visual layouts, object permanence, and viewpoint coherence. DINOv2, fine-tuned on an embodied dataset, extracts patchlevel frame representations. Cosine similarity between patch embeddings of consecutive and initial frames quantifies frame-to-frame consistency. Higher scores indicate stable scene structures and coherent viewpoints throughout the video.

- B. Action Motion. The quality of generated motions is evaluated through a Trajectory-Based Evaluation, which compares generated trajectories with ground truth trajectories. The trajectories capture physical consistency, task logic, and interaction constraints. In our setup, we use the endeffector’s (EEF) trajectory as the evaluation target and EEF is detected with our finetuned detector. Symmetric Hausdorff Distance (HSD) measures spatial alignment by calculating the maximum deviation between points on generated and GT trajectories. Normalized Dynamic Time Warping (NDTW) captures spatial-temporal alignment, ensuring correct sequence and timing of motions. Dynamic Consistency (DYN) evaluates motion dynamics, such as velocity and acceleration, using Wasserstein distance with motion normalization. To ensure fairness, generative models are required to produce three candidate trajectories for each task. The best trajectory is selected based on Hausdorff distance.

- C. Semantics. Semantic evaluation focuses on (1) alignment between task instructions and generated videos and (2) diversity within the task space. For semantic alignment, we use the generated video’s language caption as an intermediate representation, comparing them to ground truth annotations to compute an alignment score. Captions are extracted at three levels, with details on the prompt design provided in Section 3.4. For semantic diversity, we use CLIP model, global video features are extracted, and the diversity score is computed as 1 − similarity. This reflects the model’s ability to generalize and produce varied outputs.

### 3.4 MLLM Prompt Suite Design

EWMBENCH Prompt suite is designed to be compact yet representative. We perform this evaluation across three levels of language analysis. Full prompts are in the project page.

Global Video Caption Representation: At the global level, a video MLLM generates a compact caption summarizing the entire video. This caption is compared with the raw task instruction to evaluate overall alignment between the task goal and the generated video’s content using BLEU score.

[Figure 39]

|Rank|Model|Score|
|---|---|---|
| |EnerVerse_FT|4.7010|
| |LTX_FT|4.5493|
| |Kling|3.8698|
| |COSMOS|3.4125|
| |Hailuo|3.2872|
| |LTX|3.1392|
| |OpenSora|2.9676|

(A) Evaluation Results of Video Generative Models (B) Comprehensive Ranking of Tested Models

Figure 4: Evaluation Results of Video Generative Models.

Key Steps Description: Robot tasks often involve multiple key steps that may be lost in global representations. To address this, the video MLLM produces a detailed, step-by-step description of the task’s key steps. These descriptions are compared with GT step descriptions generated by the MLLM using CLIP score.

Logical Error Punishment: Logical errors, such as hallucinations or spatial inconsistencies, are critical in robotic applications as they can lead to unsafe outcomes. The MLLM evaluates generated videos for commonsense violations, explicitly penalizing errors like hallucinated object manipulations or illogical spatial relationships. These penalties ensure that the model prioritizes realistic and coherent task execution.

## 4 Experiments

Models We evaluate EWMBENCH across seven video generation models categorized as opensource, commercial, and domain-adapted. The open-source models include OpenSora 2.0[29], which demonstrates strong performance on VBench with low training costs; LTX[13], capable of realtime generation for interactive tasks; and COSMOS-7B[1], pre-trained for digital twin applications. Commercial models optimized for zero-shot generation include Kling-1.6[22] and Hailuo I2V-01live[14], both of which rank highly in recent benchmarks. Domain-adapted models fine-tuned for embodied scene understanding and action prediction include LTX_FT, a fine-tuned version of LTX, and EnerVerse, which is specifically designed for embodied scenarios. At present, we primarily focus on the Image-Text-to-Video setting, as no open-source action-conditioned video generation models are currently available. The evaluation of such models is left for future work. Nonetheless, with our unified input format and visual-space evaluation operations, EWMBENCH could also support this evaluation in the future. For the current evaluation, we tested 10 tasks, each consisting of 10 ground-truth episodes. Three videos were generated per model per episode, and the best prediction was selected using a best-of-three strategy, resulting in a total of 2,100 videos.

### 4.1 Evaluation Results

We evaluate models across dimensions using normalized scores between 0 and 1, where higher values indicate better performance. Results in Table 2 show that domain-adapted models, such as EnerVerse and LTX_FT, consistently outperform commercial models (e.g., Kling, Hailuo) and open-source models (e.g., COSMOS, OpenSora, LTX). This highlights the effectiveness of domain-specific finetuning in capturing motion dynamics and task semantics. Notably, EnerVerse and Kling demonstrate strong semantic alignment, reflecting a solid understanding of task logic.

To validate the reliability of our evaluation, Figure 5 provides representative examples. Low scene consistency is marked by changes in spatial layout and object presence, while high scene consistency preserves both. Poor trajectory consistency features mismatched end-effector motion and task failure, whereas good cases exhibit motion patterns closely aligned with the ground truth, ensuring task success. Importantly, scene and trajectory consistency are complementary: visually plausible but static videos may score high in scene consistency while lacking meaningful motion. This emphasizes the need for a systematic, multi-dimensional evaluation approach.

Scene Motion Semantics

Type Model

Overall SceneC HSD Dyn nDTW Avg. Diversity BLEU CLIP Logics Avg.

EnerVerse_FT[17] 0.9427 0.5356 0.5363 0.5957 1.6676 0.0691 0.1800 0.8638 0.9778 2.0907 4.7010 LTX_FT 0.9436 0.4758 0.6197 0.5208 1.6163 0.0162 0.1740 0.8548 0.9444 1.9894 4.5493

Dom.

Kling[22] 0.8888 0.3231 0.3047 0.3162 0.9440 0.0493 0.1675 0.8535 0.9667 2.0370 3.8698 Hailuo[14] 0.8577 0.2229 0.1344 0.1789 0.5362 0.0370 0.1848 0.8857 0.9111 2.0186 3.4125

Comm.

COSMOS[1] 0.7963 0.2500 0.2052 0.2533 0.7085 0.0803 0.1230 0.8458 0.7333 1.7824 3.2872 OpenSora[29] 0.9210 0.1548 0.0474 0.1420 0.3442 0.0415 0.1598 0.8505 0.8222 1.8739 3.1392 LTX[13] 0.9156 0.1575 0.1002 0.1425 0.4002 0.0174 0.0687 0.8324 0.7333 1.6518 2.9676

Open.

Table 2: Evaluation results categorized into task scene, action motion, and semantics.

###### Ground Truth Bad Example Good Example

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Task： "Pick up the rubik's cube on the table with the right arm."

SceneC : 0.954 SceneC : 0.956

###### SceneC : 0.637

[Figure 45]

[Figure 46]

[Figure 47]

Task： "Place the toast on the plate."

HSD : 0.115 DYN : 0.064 nDTW : 0.064 HSD : 0.556 DYN : 0.591 nDTW : 0.764

[Figure 48]

[Figure 49]

[Figure 50]

Task： "Pass the held showerhead to right arm with left arm."

SceneC : 0.890

SceneC : 0.994 HSD : 0.178 DYN : 0.005 nDTW : 0.123 SceneC : 0.878 HSD : 0.520 DYN : 0.614 nDTW : 0.558

- Figure 5: Typical examples. EWMBENCH scores align well with scene and motion accuracy, demonstrating the interpretability and robustness of the proposed metrics.

### 4.2 Human Evaluation

To evaluate the alignment between automated metrics and human judgment, we conducted a human evaluation on videos generated by four representative models: LTX_FT, Kling-1.6, Hailuo I2V-01live, and OpenSora-2.0. Annotators ranked the predictions based on overall quality, assigning 3 points to the best, 2 to the second-best, and 0 to the worst. Final rankings were derived by aggregating scores across all annotators and samples, with multiple review rounds ensuring annotation reliability.

We then compared this aggregated human rankings (Figure 6 (A)) with those produced by EWMBENCH and VBench [18], one of the most popular video generation evaluation benchmarks. As shown in Figure 6 (B), EWMBENCH’s rankings align more closely with human judgments than VBench rankings, indicating stronger consistency with human perception.

### 4.3 Complementarity of Trajectory Metrics

To validate the necessity of employing all three trajectory consistency metrics—HSD, nDTW, and DYN—we conducted controlled experiments involving sequence reversal, outlier insertion, and frame repetition. As shown in Figure 6 (C), each metric responds uniquely, demonstrating its specific strengths. In the sequence reversal test, only nDTW showed a significant drop due to its sensitivity to temporal order, underscoring its role in detecting alignment errors. In the outlier test, HSD and DYN experienced substantial declines, reflecting their focus on spatial accuracy and motion integrity—both essential for safe and precise embodied execution. In the frame repetition test, nDTW increased due to repeated alignment, while DYN decreased, highlighting its sensitivity to motion smoothness. These findings confirm the complementary roles of the three metrics in providing a comprehensive evaluation of trajectory quality.

### 4.4 Further Analysis and Discussions

Characteristics of SOTA Models We present key findings from EWMBENCH ’s evaluation, focusing on trade-offs, model characteristics, and the impact of domain adaptation in embodied video generation. Qualitative examples illustrating these failure modes are provided in Appendix A.4.

[Figure 51]

[Figure 52]

[Figure 53]

LTX_FT Kling

Hailuo OpenSora

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

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

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

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

(A) Human Rank Result (B) Benchmark Score Result

(1) Reverse Sequence Experiment (2) Outlier Experiment (3) Insert Still Frame Experiment

[Figure 714]

[Figure 715]

[Figure 716]

0.005

(C) Experimental results of trajectory consistency complementarity

- Figure 6: (A) Aggregated human rankings of model predictions. (B) Comparison of rankings produced by EWMBENCH and VBench, highlighting EWMBENCH’s closer alignment with human judgments.(C) Complementarity of trajectory metrics.

Domain-adapted models show the best overall performance, particularly in semantic and dynamic dimensions, demonstrating that targeted fine-tuning significantly enhances task understanding and motion alignment. However, they occasionally exhibit empty grasping behaviors, revealing limitations in fine-grained action grounding.

Kling achieves the best overall performance among general commercial and open-source video models, demonstrating strong and robust capabilities.

Hailuo performs reasonably well in zero-shot embodied scenarios, but its generated scenes often appear cartoon-like, limiting visual realism.

COSMOS and LTX display a bias toward human hand representations and frequently fail to adapt semantic understanding to robotic contexts. LTX, in particular, suffers from abrupt scene transitions, inconsistent task execution, and and a tendency to generate stationary states in action sequences. In contrast, COSMOS struggles to maintain consistent viewpoints, highlighting inadequate control of camera parameters.

OpenSora shows partial understanding of task scenes, action motions, and semantic alignment in manipulation tasks. However, it suffers from jittery robotic arm movements and frequently generates static videos.

Comparison with VBench Metrics Our experiments reveal that VBench struggles to separate foreground and background features, limiting the effectiveness of its subject-level metrics. In contrast, our Scene Consistency metric, leveraging fine-tuned DINOv2, excels at capturing layout structure and is more sensitive to viewpoint changes. This heightened sensitivity enables the detection of visual instability, a critical factor in embodied video generation. Additional details and feature map visualizations are provided in the Appendix.

## 5 Conclusions and Limitations

In this work, we propose EWMBENCH, a comprehensive benchmark suite for evaluating embodied world generation models. With its multi-dimensional, human-aligned metric design and a motiondiverse, multi-scene dataset, EWMBENCH serves as a valuable tool for measuring progress in embodied world model development.

Limitations and Future Work. First, our method currently focuses on the trajectory of the robotic arm’s end-effector, but future work will incorporate the state and configuration of the entire arm. Second, the current evaluation is conducted in fixed-viewpoint scenes; future research will explore flexible viewpoints, such as dynamic camera setups. Lastly, we aim to extend the scope of embodied tasks—from the current manipulation tasks to more diverse domains, including navigation and mobile manipulation.

## References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] AgiBot. Agibot world. https://agibot-world.com, 2024.
- [3] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.
- [4] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [6] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. 2024. URL https://openai. com/research/video-generation-models-as-world-simulators, 3:1, 2024.
- [7] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.
- [8] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023.
- [9] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Realtime open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16901–16911, 2024.
- [10] Xiaowei Chi, Hengyuan Zhang, Chun-Kai Fan, Xingqun Qi, Rongyu Zhang, Anthony Chen, Chi-min Chan, Wei Xue, Wenhan Luo, Shanghang Zhang, et al. Eva: An embodied world model for future video anticipation. arXiv preprint arXiv:2410.15461, 2024.
- [11] Weixi Feng, Jiachen Li, Michael Saxon, Tsu-jui Fu, Wenhu Chen, and William Yang Wang. Tc-bench: Benchmarking temporal compositionality in text-to-video and image-to-video generation. arXiv preprint arXiv:2406.08656, 2024.
- [12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023.
- [13] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.
- [14] Hailuo. Hailuoai. https://hailuoai.video/, 2025.
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in neural information processing systems, 2017.
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [17] Siyuan Huang, Liliang Chen, Pengfei Zhou, Shengcong Chen, Zhengkai Jiang, Yue Hu, Yue Liao, Peng Gao, Hongsheng Li, Maoqing Yao, et al. Enerverse: Envisioning embodied future space for robotics manipulation. arXiv preprint arXiv:2501.01895, 2025.
- [18] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

- [19] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024.
- [20] Pengliang Ji, Chuyang Xiao, Huilin Tai, and Mingxiao Huo. T2vbench: Benchmarking temporal dynamics for text-to-video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5325–5335, 2024.
- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [22] Kuaishou. Kling. https://app.klingai.com/cn/, 2025.
- [23] Xinrang Ling, Chen Zhu, Meiqi Wu, Hangyu Li, Xiaokun Feng, Cundian Yang, Aiming Hao, Jiashu Zhu, Jiahong Wu, and Xiangxiang Chu. Vmbench: A benchmark for perception-aligned video motion generation. arXiv preprint arXiv:2503.10076, 2025.
- [24] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [25] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.
- [26] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models learn physical principles from watching videos? arXiv preprint arXiv:2501.09038, 2025.
- [27] Meinard Müller. Dynamic time warping. Information retrieval for music and motion, pages 69–84, 2007.
- [28] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [29] Xiangyu Peng, Zangwei Zheng, Chenhui Shen, Tom Young, Xinying Guo, Binluo Wang, Hang Xu, Hongxin Liu, Mingyan Jiang, Wenjun Li, et al. Open-sora 2.0: Training a commercial-level video generation model in 200k. arXiv preprint arXiv:2503.09642, 2025.
- [30] Weiming Ren, Harry Yang, Ge Zhang, Cong Wei, Xinrun Du, Stephen Huang, and Wenhu Chen. Consisti2v: Enhancing visual consistency for image-to-video generation. arXiv preprint arXiv:2402.04324, 2024.
- [31] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, and Xi Chen. Improved techniques for training gans. In Advances in neural information processing systems, 2016.
- [32] Jean Serra. Hausdorff distances and interpolations. Computational Imaging and Vision, 12:107–114, 1998.
- [33] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [34] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. arXiv preprint arXiv:2407.14505, 2024.
- [35] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation. In ICLRW, 2019.
- [36] Cédric Villani and Cédric Villani. The wasserstein distances. Optimal transport: old and new, pages 93–111, 2009.
- [37] Andrew Vince. A framework for the greedy algorithm. Discrete Applied Mathematics, 121(1-3):247–260, 2002.
- [38] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2025.
- [39] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 1(2):6, 2023.

- [40] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M Ni, and Heung-Yeung Shum. Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605, 2022.
- [41] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qing, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models, 2023.
- [42] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, March 2024. URL https://github.com/hpcaitech/Open-Sora.
- [43] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024.

## A Appendix

- A.1 Additional Details on World Specification

We provide implementation details for the pre-processing module. These components ensure consistency across different models and support metric-specific evaluation.

Pre-Processing We resize all reference images I to a fixed resolution of 640 × 480. For the task prompt L, we use the aligned step-level caption corresponding to the current sub-action. To ensure viewpoint and temporal consistency, we append the following constraint to the prompt:

“Keep the first-person view of the robot unchanged. Keep the first frame of this video unchanged.”

Video Normalization All generated videos are resized to 640 × 480 and resampled to 30 FPS. The normalized video is used to evaluate task scene consistency and semantic alignment.

Trajectory Extraction We extract 2D trajectories of both end-effectors using a consistent detection pipeline. Specifically, we apply a fine-tuned YOLO-World [9] model for per-frame detection and use BoT-SORT [3]for temporal association. To ensure fair evaluation, we compute the convex hull of each hand’s trajectory and select the hand with the largest spatial extent—measured by the maximum Euclidean distance between any two points on the hull—as the primary trajectory for motion evaluation.

YOLO-World Training Details. We fine-tune yolov8s-worldv2 on 1451 manually annotated frames from Agibot-World. Two tasks (Freezer Restocking and Factory Packing) are held out for validation to construct a hard-sample set. The model is trained for 100 epochs and achieves a final performance of Recall: 0.91667, Precision: 1.0.

- A.2 Additional Details on Dataset Curation

- A.2.1 Task Selection and Scene Diversity

To support evaluation across a wide range of manipulation scenarios, we select 10 representative tasks from the Agibot-World [2] dataset. The selection prioritizes tasks with clear operational goals, sequential dependencies, and diverse object interactions. These tasks span both household and industrial contexts and are designed to challenge models across spatial reasoning, tool-use, and action ordering.

- • Retrieving toast from a toaster (Toaster)
- • Pouring water (Pour Water)
- • Setting cutlery (Place Cutlery)
- • Restocking a freezer (Restock Freezer)
- • Producing ice (Produce Ice)
- • Packing laundry detergent (Factory Packing)
- • Cleaning bottles (Brush Bottle)
- • Heating food in a microwave (Heat Food)
- • Installing a showerhead (Hang Showerhead)
- • Storing objects in a drawer (Store in Drawer)

These tasks exhibit substantial variation in manipulated object types (e.g., rigid, deformable, articulated), spatial layouts, and interaction complexity. Figure 7 visualizes the task scenes, their corresponding trajectory overlays, and object property distributions.

- A.2.2 Action Motion Sampling Strategy

To construct a comprehensive and diverse evaluation set, we uniformly sample 100 video instances per task. Each task is decomposed into fine-grained primitive actions, ensuring one-to-one correspondence between video segments, sub-action labels, and textual descriptions.

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

Take Toast Brush Bootle Restock Freezer Hang Shower Head Heat Food

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

Place Cutlery Factory Packing Pour Water Store in Drawer Produce Ice

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

Slide Fluid Articulated Multi-Subject Deformable/Rigid

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

- Figure 7: Dataset overview. The first two rows display selected task scenarios and their associated action motion trajectories. The third row categorizes object properties (e.g., fluid, articulated, rigid, deformable, multi-body) using color-coded legends and representative examples.

For trajectory analysis, we extract the left and right end-effector positions and convert them into voxel grid representations. A pairwise similarity matrix is computed using 3D Intersection over Union (IoU):

min(ViL,VjL) + min(ViR,VjR) max(ViL,VjL) + max(ViR,VjR) + ϵ

IoUi,j =

(2)

where V L and V R denote the voxel grids for the left and right end-effectors, respectively. To promote trajectory diversity in the final evaluation subset, we adopt a greedy selection algorithm [37]:

- • Start by selecting the trajectory with the lowest average IoU relative to all others.
- • Iteratively select the trajectory with the lowest average IoU to the already selected set.
- • Repeat until 10 representative trajectories are selected for each task.

This sampling strategy ensures that the evaluation set includes both common and atypical motion patterns, providing a broad spectrum of behavioral diversity to test the generalization ability of generative models. The resulting distribution is reflected in Figure. 7.

- A.3 Additional Details on Metrics

- A.3.1 Scene Consistency: DINOv2 vs. VBench Analysis

Fine-tuning DINOv2 for Embodied Scenes. To enhance the extraction of task-relevant visual features in embodied operation scenarios, we fine-tune DINOv2 [28] on the Agibot-World dataset using 20,000 iterations of unsupervised training. We use the dinov2-vitb14-reg4 checkpoint as

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

Input VBench DINOv2 WMBM Input VBench DINOv2 WMBM

- Figure 8: Feature map comparison across models. DINOv2 fine-tuned on embodied data captures agents and tools with sharper spatial coherence, enabling more reliable scene stability evaluation.

[Figure 820]

Scene Consistency: 0.6927 Background Consistency: 0.8845

- Figure 9: Scene Consistency failure cases. Despite camera movement and background drift, VBench assigns high scores (0.88–0.91). Our metric, however, detects the instability through decreased cosine similarity.

the initialization. We visualize feature maps from three model variants: (1) the original DINO [40] (ViT-B/16, used in VBench), (2) pre-trained DINOv2, and (3) our fine-tuned DINOv2.

As shown in Figure. 8, only the fine-tuned DINOv2 consistently focuses on task-relevant agents and manipulated tools, while the others fail to highlight key foreground regions or exhibit backgroundforeground entanglement. This justifies the necessity of adapting foundation models to the embodied task domain.

Failure Cases of VBench Background Consistency. We further visualize representative cases in which VBench assigns high background consistency scores despite significant viewpoint changes or layout shifts. As shown in Figure. 9, the cosine similarity of our metric drops significantly in these cases, whereas VBench remains insensitive. This highlights the limitation of VBench’s DINO-ViT-B/16 features in separating foreground and background cues.

- A.3.2 Action Motion Metrics

Symmetric Hausdorff Distance Consistency(HSD Consistency)

The Symmetric Hausdorff Distance[32] (SymH) measures the maximum spatial deviation between the generated trajectory and the corresponding ground truth trajectory. This distance represents the greatest of the minimum distances between points from both trajectories. HSD is particularly useful for evaluating the spatial alignment of the generated trajectory with the true trajectory, ensuring that the generated path adheres to expected movement patterns and does not deviate significantly from the true action trajectory.

To ensure that the score is positively correlated with consistency, we take the reciprocal of this value:

1 dsymH(G,P)

HSDscore =

(3)

Where G represents the ground truth trajectory, and P represents the generated trajectory. Normalized Dynamic Time Warping Distance Consistency(NDTW Consistency)

NDTW[27] is used to evaluate the overall shape similarity and temporal alignment of trajectories. While HSD focuses on spatial deviations, NDTW evaluates the overall trajectory shape and how well the generated actions align with the timing and sequence of the true actions. This metric is particularly useful for capturing the temporal causality and task sequencing that the model should learn from the true trajectory.

By aligning both the spatial and temporal dimensions of the trajectories, NDTW assesses whether the generated sequence matches the timing and order of the ground truth actions. The similarity score is then calculated as the reciprocal of this value:

#### 1 NDTW(G,P)

NDTWscore =

(4)

### Dynamic Consistency (DYN)

Velocity and acceleration are critical components in robotic control, directly affecting the physical feasibility of the generated actions. To evaluate how well the predicted trajectories align with the real-world motion characteristics, we extract the 2D velocity and acceleration time series from both predicted and ground-truth trajectories. We then compute the Wasserstein distance[36] (Earth Mover’s Distance, EMD) between these distributions to quantify their differences.

The Wasserstein distance captures the global distributional alignment between sequences, it allows for soft matching and does not require strict temporal alignment, which makes it more robust and better suited to capturing continuous motion trends across the entire trajectory.

To enhance the robustness of this metric across different motion amplitudes, inspired by the IoU calculation method, we introduce an amplitude normalization factor that uses the difference between the maximum and minimum velocity/acceleration to construct the following ratio:

min max(vgt) − min(vgt), max(vpred) − min(vpred) + ϵ max(max(vgt) − min(vgt), max(vpred) − min(vpred)) + ϵ

VR =

(5)

min max(agt) − min(agt), max(apred) − min(apred) + ϵ max(max(agt) − min(agt), max(apred) − min(apred)) + ϵ

AR =

(6)

where ϵ = 1 × 10−8. To account for variations in motion amplitudes, we introduce amplitude normalization factors. The final dynamic consistency score is defined as:

1 W(v)

1 W(a)

DYNscore = α · VR ·

+ β · AR ·

, α = 0.007, β = 0.003 (7)

Here, W(·) represents the Wasserstein distance, and VR, AR are the amplitude normalization factors for velocity and acceleration, respectively. These corrections ensure that low-amplitude trajectories do not introduce numerical amplification, thereby preserving the accuracy of dynamic consistency assessment.

### A.4 Visual Examples of Model Generation Results

To complement the discussion in Section 5, we provide qualitative examples illustrating the observed characteristics of recent SOTA models evaluated under WMBM.

Domain-Adapted Models. Despite strong semantic alignment, domain-adapted models sometimes fail in action grounding. As shown in Figure.10, the generated video depicts a precise scene and goal instruction, but the agent executes an empty grasp motion without interacting with the object.

[Figure 821]

- Figure 10: Domain-adapted model failure case. The robot hand moves toward the correct region but fails to close the gripper on the object, resulting in empty grasping.

LTX and COSMOS. Figure. 12 shows LTX generating abrupt scene transitions and failing to maintain object continuity. LTX and COSMOS, as in Figure. 13, frequently renders human hands

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

##### Figure 11: Despite being given explicit camera viewpoint control instructions, COSMOS fails to maintain a consistent viewpoint throughout the video. This indicates a limitation in its ability to follow spatial constraints, leading to unstable or drifting perspectives.

Init Frame LTX Good Example

[Figure 827]

[Figure 828]

[Figure 829]

Task:

“Place the toast on the plate.”

[Figure 830]

[Figure 831]

[Figure 832]

Task:

“Pick up the blue kettle with right arm.”

##### Figure 12: Examples illustrating the poor task understanding and temporal instability of the LTX model. The middle column shows LTX-generated frames, which often exhibit abrupt visual changes and scene inconsistencies. In contrast, the rightmost column presents examples with better scene preservation, highlighting the gap in temporal coherence.

instead of robot arms, revealing a semantic adaptation failure.A failure of viewpoint control in COSMOS is illustrated in Figure. 11.

OpenSora.As discussed in the main text, OpenSora shows partial understanding of task semantics but struggles with motion control. Figure. 14 presents a representative example highlighting this issue: while the generated scene correctly reflects the intended manipulation context, the robotic arm undergoes significant jitter and fails to execute smooth movements. This supports our observation that OpenSora’s motion instability remains a key limitation in embodied video generation.

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

##### Figure 13: Generated videos from COSMOS and LTX models often depict human hands instead of robotic arms, indicating a bias toward human hand representations in their training data. This bias hinders the models’ ability to correctly generalize to robotic manipulation tasks, where accurate mapping to robotic arms is essential.

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

##### Figure 14: An example from OpenSora illustrating unstable robotic arm motion. Although the scene is semantically aligned with the manipulation task, the arm exhibits visible jitter and lacks smooth trajectory control, consistent with the motion instability discussed in the main text.

