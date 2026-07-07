[Figure 1]

## Reasoning via Video: The First Evaluation of Video Models’ Reasoning Abilities through Maze-Solving Tasks

Cheng Yang1∗ Haiyuan Wan2,3∗ Yiran Peng1∗ Xin Cheng4 Zhaoyang Yu1 Jiayi Zhang1,8 Junchi Yu5† Xinlei Yu6 Xiawu Zheng7 Dongzhan Zhou3 Chenglin Wu1† 1DeepWisdom 2Tsinghua University 3Shanghai Artificial Intelligence Laboratory 4Renmin University of China 5University of Oxford 6National University of Singapore 7Xiamen University 8Hong Kong University of Science and Technology (GuangZhou)

# arXiv:2511.15065v2[cs.CV]24Nov2025

https://imyangc7.github.io/VRBench Web

∗ Core contributors † Corresponding authors

（A) Regular Maze Irregular Maze 3D Maze Trapfield Sokoban

（B)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

c

Chain of Frame

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

（C)

[Figure 24]

###### （D) Difficulty Generalization Texture Generalization Maze Type Generalization Test Time Scaling

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

Figure 1. Overview of VR-Bench. (A) Maze Types. VR-Bench comprises five maze types—Regular Maze, Irregular Maze, 3D Maze, Trapfield, and Sokoban—covering both 2D and 3D settings as well as diverse task structures, yielding a broad range of spatial reasoning scenarios. (B) Reasoning via Video Paradigm. VR-Bench adopts a chain-of-frame reasoning paradigm [45], requiring models to produce frame-by-frame inferences that capture sequential visual reasoning. (C) Benchmark Performance. Leading VLMs and video models are evaluated on four core metrics across all maze types, revealing clear differences in spatial reasoning capability. (D) Additional Analysis. VR-Bench also supports evaluations on difficulty generalization, texture generalization, maze-type generalization, and test-time scaling, enabling a comprehensive assessment of model robustness and generalization.

#### Abstract

Video Models have achieved remarkable success in highfidelity video generation with coherent motion dynamics. Analogous to the development from text generation to textbased reasoning in language modeling, the development of video models motivates us to ask: Can video models reason via video generation? Compared with the discrete text corpus, video grounds reasoning in explicit spatial layouts and temporal continuity, which serves as an ideal substrate for spatial reasoning. In this work, we explore the reasoning via video paradigm and introduce VR-Bench—a comprehensive benchmark designed to systematically evaluate video models’ reasoning capabilities. Grounded in mazesolving tasks that inherently require spatial planning and multi-step reasoning, VR-Bench contains 7,920 procedurally generated videos across five maze types and diverse visual styles. Our empirical analysis demonstrates that SFT can efficiently elicit the reasoning ability of video model. Video models exhibit stronger spatial perception during reasoning, outperforming leading VLMs and generalizing well across diverse scenarios, tasks, and levels of complexity. We further discover a test-time scaling effect, where diverse sampling during inference improves reasoning reliability by 10–20%. These findings highlight the unique potential and scalability of reasoning via video for spatial reasoning tasks.

#### 1. Introduction

With the rapid development of diffusion-based and autoregressive-based generative architectures, video models have witnessed tremendous success in high-fidelity video generation. Previous works, such as Stable Video Diffusion [3] and Imagen Video [15], showcase the capability of video models to generate physically realistic and temporally consistent videos conditioned on their input instructions. Recent studies further reveal that advanced video models are capable of performing a diverse range of visual tasks beyond generation itself, including perception, understanding, and even reasoning. These findings suggest that video models are evolving from pure generative models into generalpurpose visual intelligence models. Analogous to the evolution of language models from text generation to text-based reasoning, the development of video models leads to a question: “Can video models reason via video generation?”

Crucially, the spatiotemporal nature of video modality offers a new perspective on reasoning. The traditional paradigm, which we term reasoning via text, uses language as the medium for expressing intermediate reasoning steps. Representative work, such as Chain-of-Thought prompting [39, 44, 49–52, 54], achieves this by eliciting large language models (LLMs) to generate a coherent textual reasoning chain. Recently, this reasoning via text paradigm has been

introduced to visual domains, including multimodal question answering and video understanding. However, even in these multimodal settings, current paradigms still express reasoning through textual continuation instead of visual or physical dynamics. In contrast, video represents reasoning as a process of visual continuation over time. Each frame in a video builds upon its previous ones, capturing the dynamics of motion, spatial consistency, and temporal causality within 2D and 3D space. The continuous and structured nature of frames makes video an ideal substrate for multimodal reasoning. Building on this insight, we propose reasoning via video, where reasoning emerges through nextframe generation rather than next-token prediction.

However, a comprehensive testbed for reasoning via video is lacking. To this end, we introduce VR-Bench, a dedicated benchmark designed to systematically assess the reasoning capabilities of video generation models. As shown in Figure 1, we ground our benchmark in the mazesolving task, a natural fit for visual reasoning due to its open-ended solution space and rich trajectory-based supervision. Each instance inherently demands spatial planning, dynamic tracking, and multi-step reasoning, making it an ideal testbed for evaluating model inference quality over time. Our dataset comprises 7,920 procedurally generated maze-centric videos, each paired with a corresponding Trace Reasoning Task that requires models to infer the optimal path. To ensure broad generalizability and challenge model robustness, VR-Bench spans five distinct maze types—Regular Maze, Irregular Maze, 3D Maze, Sokoban, and Trapfield—covering a wide spectrum of spatial structures and decision patterns. Additionally, each maze is rendered in diverse visual styles across more than a dozen themes, enabling fine-grained analysis of how well models generalize across varied visual domains and increasing the realism and complexity of the reasoning tasks.

Building upon the proposed VR-Bench, we conduct a systematic study of the reasoning via video paradigm. We construct instruction-following datasets derived from VRBench to elicit the reasoning capability of open-source video models. After supervised fine-tuning (SFT), these models exhibit significant performance gain across all reasoning tasks in VR-Bench. Moreover, SFT endows video models with strong out-of-domain generalization under diverse distribution shifts, including task difficulty, background style, and task type. Compared with vision–language models (VLMs) [2, 5, 6, 19, 22] that reason via text, video models consistently outperform their counterparts on high-complexity reasoning tasks, showing greater stability and even superior performance as task difficulty increases, across diverse scenarios and tasks. This finding confirms that videos serve as a more expressive substrate for spatial reasoning, which facilitates video models to leverage temporal continuity and dynamic visual context.

Interestingly, we further observe that video models exhibit a test-time scaling effect analogous to that of LLMs. As the inference budget increases, their performance improves substantially. By employing diverse sampling strategies at test time, video models effectively explore multiple reasoning trajectories, reducing uncertainty and achieving an average performance gain of 10–20%. These empirical results highlight the unique potential and scalability of the reasoning via video paradigm.

Our contributions are summarized as follows:

- • We make an early and systematic exploration of the reasoning via video paradigm, where reasoning emerges from sequential frame generation rather than token prediction. Compared with text-based approaches, this paradigm naturally captures temporal continuity and spatial causality, offering a more expressive and scalable substrate for solving spatial reasoning tasks.
- • We construct VR-Bench, a comprehensive benchmark grounded in maze-solving tasks with diverse spatial structures, difficulty levels, and texture styles. It provides finegrained trajectory-level supervision and supports evaluations on path accuracy, rule compliance, generalization.
- • Through extensive experiments, we demonstrate that video-based reasoning outperforms text-based reasoning (e.g., VLMs) on complex tasks, especially under distribution shifts in maze type, visual style, and difficulty. Finetuned video models exhibit stronger performance, lower path redundancy, and higher structural fidelity.
- • We reveal a test-time scaling effect for video models, where performance consistently improves with larger inference budgets. Similar to that in LLMs, diverse sampling unlocks multi-path exploration and yields up to 20% performance gains across metrics and difficulty levels.

#### 2. Related Works

##### 2.1. Video Generation

Video models have advanced rapidly in both understanding and generation. Early understanding methods, such as MViT [10], Video Swin Transformer [23], and VideoMAE [37], focused on learning robust video representations for downstream tasks. With LLMs [1, 3, 29], recent approaches tokenize videos and leverage language backbones for captioning [36], event localization [33], and reasoning [16]. On the generation side, Sora-2 [4] achieved controllable, physically consistent outputs with synchronized dialogue and sound. Proprietary systems such as Runway’s Gen-3 [31], Pika Labs [30], Luma AI [24], and Google DeepMind’s Veo series [7, 8] further enhance video quality and realism but remain closed-source. In contrast, open-source frameworks such as Stable Video Diffusion [3], OpenSora[55], HunyanVideo [21], and the Wan series [40] democratize access, offering efficient architectures and scalable training for state-

of-the-art video synthesis.

##### 2.2. Evolution of Reasoning Paradigms

Chain-of-Thought (CoT) prompting has significantly enhanced the reasoning abilities of language models [12, 41, 44]. Reinforcement learning further integrates CoT-style reasoning into model training, enabling models to internalize multi-step thought processes. More recently, such paradigms have been extended to vision-language models (VLMs). Systems like o3 and o4-mini [27] introduce the ”Think with Image” framework, where reasoning is grounded in visual operations such as zooming and cropping. This allows the model to dynamically interact with image regions as part of the CoT process, thereby improving multimodal reasoning [32, 56, 57]. In parallel, the rise of unified models for both generation and understanding has given birth to a new reasoning paradigm centered on interleaved vision-language outputs. Instead of purely textual reasoning traces, these models generate coherent sequences that alternate between textual and visual elements [9, 39, 46, 47], providing a more grounded and expressive format for complex multimodal reasoning.

##### 2.3. Evaluation of Video Generation Reasoning

Previous benchmarks for video generation models have predominantly focused on assessing visual quality, temporal coherence, and alignment with human preferences [14, 17, 18, 53]. However, these evaluations largely neglect the reasoning capabilities of video models. Recent works have begun to explore reasoning via video generation—the ability of models to solve reasoning tasks through the generation process itself [13, 34, 45]. For instance, models like Veo 3 demonstrate zero-shot competence in tasks such as maze navigation and symmetry recognition. These tasks require perceiving, modeling, and manipulating the visual world, indicating that video generation can inherently support spatial-temporal reasoning. Despite these promising directions, current benchmarks for video reasoning still suffer from several limitations: (1) Lack of fine-grained and objective evaluation: Current evaluations rely heavily on manual inspection or coarse metrics, without capturing the reasoning trajectory embedded in the video; (2) Absence of modality comparisons: There is a lack of systematic comparison with think with text or think with image paradigms, making it unclear whether video generation truly provides unique advantages for reasoning; (3) Neglect of tuning and scaling analysis: Unlike language or multimodal models, video reasoning benchmarks seldom explore whether supervised fine-tuning (SFT) or test-time scaling can improve performance. These gaps call for a new benchmark that evaluates not only generation quality but also the reasoning process in videos, using rigorous metrics, multimodal comparisons, and extensible settings.

#### 3. VR-Bench

##### 3.1. Dataset Construction

The VR-Bench dataset is a Visual Trace Reasoning (VTR) dataset that constructs various Maze Puzzles into visual reasoning tasks. Its construction process comprises two steps: Maze Generation and Video Generation.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Figure 2. Variations of difficulty level and maze texture

Maze Generation. The maze generation process in our work encompasses five distinct types, with all 7,920 maze instances in the dataset generated programmatically through custom code [35]. Each type is tailored to assess specific visual reasoning capabilities, as elaborated below:

- 1. Regular Maze. We generate mazes with a grid-based layout to focus on the model’s ability to perceive basic maze structures and its path-finding and problemsolving competence, serving as a fundamental testbed for maze reasoning tasks.
- 2. Trapfield. This type transforms the ”walls” of traditional mazes into grid-shaped trap regions, reversing the logic from ”finding feasible paths” to ”avoiding traps”. Beyond altering the problem-solving logic, the more flexible movement space also challenges the model’s ability to plan and find optimal paths.
- 3. Irregular Maze. Moving away from regular blockshaped paths, we adopt curve-based path designs. This design prevents the model from relying on coordinatebased position encoding, thereby rigorously evaluating its pure visual perception of maze layouts. It also explicitly decouples visual reasoning from text-based reasoning, focusing on the video model’s capability to reason via video itself.
- 4. Sokoban. We modify the underlying rules of traditional mazes by introducing the ”Sokoban” task mechanism. Models need to comprehend and apply Sokoban logic on top of path finding, increasing task complexity and emphasizing the model’s ability to internalize and apply logic.
- 5. 3D Maze. By extending the maze to a 3D space, we employ a stereoscopic structural design to test the model’s

spatial perception ability in 3D environments and its cross-dimensional path reasoning capability.

Maze Variations. To evaluate the generalization ability on the VTR task and enhance robustness in adapting to diverse maze scenarios, we introduce variations across two key dimensions: (1) Difficulty Level: We define three difficulty grades (Easy, Medium, and Hard) by adjusting the maze size (e.g., expanding from 5×5 to 7×7), modifying the number of maze branches, and adding obstacles; (2) Maze Texture: We vary the textures of maze obstacles, paths, and other components using textures generated via procedural methods and generative models, as shown in Figure 6, which exposes the policies to a broad visual distribution and mitigates overfitting to clean, synthetic environments.

Video Generation. To generate solution videos from maze images, we use a Breadth-First Search solver to compute the optimal path for each maze. These paths are rendered into videos at 24 fps and standardized to 192 frames (8 seconds) by adjusting playback speed, producing consistent image–video pairs for training and evaluation.

##### 3.2. Metric Design

We selected two different evaluation paradigms to comprehensively assess our task, as detailed below.

Path Matching. To objectively and comprehensively evaluate the VTR task, we perform target tracking across each frame of the model-generated videos to record the motion trajectory of the target. By comparing and analyzing these trajectories against the optimal path for each task [48], we propose the following four evaluation metrics.

###### 1. Exact Match (EM)

Defined as EMi = nj=1i I(ˆvij = vij). This metric measures whether the model successfully generates the com-

plete and correct trajectory that aligns with the shortest optimal valid path. One step of deviation from the optimal solution is considered incorrect.

###### 2. Success Rate (SR)

Defined as SRi = I p(endgen) ∈ Bgoal . SR measures whether the generated trajectory successfully reaches the designated goal region. It reflects the model’s capability to complete the task by arriving at the target position, with a value of 1 indicating successful goal attainment and 0 indicating failure to reach the goal.

###### 3. Precision Rate (PR)

ni j=1

- j
- k=1 I(ˆvik = vik) . PR

Defined as PRi = n1

i

quantifies the proportion of consecutively correct steps along the optimal path. It offers a softer metric than EM, reflecting the model’s ability to make steady, meaningful progress toward the complete correct trajectory.

EM (↑) SR (↑) PR (↑) SD (↓) Base Irreg Trap 3D Soko Base Irreg Trap 3D Soko Base Irreg Trap 3D Soko Base Irreg Trap 3D Soko

Method

Gemini-2.5-pro 2.8 36.1 13.9 2.8 25.0 4.2 37.5 13.9 2.8 31.9 9.5 47.9 57.6 19.4 33.8 25.0 1.9 0.0 0.0 1.1 Gpt-5 high 13.9 31.9 18.1 0.0 23.6 69.4 33.3 27.8 1.4 34.7 11.8 43.6 53.7 23.7 35.6 31.0 2.1 6.9 20.0 0.5 Qwen2.5-VL-7B♡ 0.0 1.3 0.0 0.0 1.4 1.4 6.9 1.4 1.4 2.8 6.5 12.4 14.3 11.3 7.8 300.0 26.7 66.7 80.0 1150.0

VLM

Qwen2.5-VL-7B-SFT 12.5 29.2 22.2 31.9 29.8 52.8 34.7 52.8 36.1 37.5 32.5 45.1 71.6 59.3 43.0 52.6 2.3 11.3 4.5 3.0

∆↑ +12.5 +27.9 +22.2 +31.9 +28.4 +51.4 +27.8 +51.4 +34.7 +34.7 +26.0 +32.7 +57.3 +48.0 +35.2 -247.4 -24.4 -55.4 -75.5 -1147.0

Closed-Source

Veo-3.1-fast 0.0 0.0 0.0 0.0 2.8 40.3 36.1 38.9 48.6 43.1 20.2 24.8 28.2 13.4 21.7 195.3 111.5 80.7 33.5 112.3 Veo-3.1-pro 0.0 4.2 1.4 0.0 0.0 47.2 36.1 59.7 50.0 37.5 24.6 33.9 39.1 18.0 21.4 140.7 94.5 85.4 40.1 141.8 Sora-2 1.4 5.6 0.0 0.0 4.2 75.0 72.2 83.0 37.5 43.1 45.1 45.7 46.6 19.3 27.4 302.9 187.0 145.1 92.4 138.7 kling-v1 0.0 0.0 0.0 0.0 0.0 2.8 0.0 1.4 27.8 12.5 6.3 8.8 10.4 11.7 9.0 25.2 – – 69.7 356.1 Seedance-1.0-pro 0.0 2.8 2.8 0.0 0.0 75.0 45.8 59.7 77.8 13.9 12.8 35.8 42.7 23.6 17.1 162.3 143.4 99.1 84.4 241.9 MiniMax-Hailuo-2.3 0.0 1.4 2.8 0.0 0.0 68.1 40.3 70.8 55.6 45.8 23.2 24.2 30.3 20.3 15.5 464.0 170.0 90.9 50.1 165.5

GeneralVideoModel

###### Open-Source

Wan2.5-i2v-preview 0.0 2.8 4.2 0.0 0.0 58.3 26.4 77.8 24.5 22.4 14.3 21.8 34.4 24.5 17.1 378.4 281.8 73.2 119.9 278.0 Wan2.2-TI2V-5B♢ 0.0 0.0 0.0 0.0 0.0 6.9 12.5 0.0 31.9 11.1 6.6 9.1 7.1 12.8 9.2 388.7 66.1 – 5.4 176.6

Wan-R1 33.3 56.9 38.9 65.3 4.2 76.4 69.4 100.0 100.0 69.4 60.6 71.6 79.1 93.5 44.3 10.3 2.4 3.9 3.9 10.2

Ours

∆↑ +33.3 +56.9 +38.9 +65.3 +4.2 +69.5 +56.9 +100.0 +68.1 +58.3 +54.0 +62.5 +72.0 +80.7 +35.1 -12.8 -25.1 – -7.8 -100.1

- Table 1. The five tasks of VR-Bench correspond to Base (Regular Maze), Irrg (Irregular Maze), Trap (TrapField), 3D (3D Maze), and Soko (Sokoban). The best and second-best results in each column are bolded and underlined, respectively. “–” indicates that the model produced no successful cases for the corresponding task, making SD undefined. ♢ denotes the base model of Wan-R1, for comparisons.

###### 4. Step Deviation (SD)

ines five key dimensions: (1) Motion Continuity of the main subject, (2) Temporal Consistency of the subject, (3) Trajectory Rationality of the main subject, (4) Structural Consistency of the maze, and (5) Interactional Rationality of subject–maze interactions. Each generated video is interpreted by a VLM, which identifies and scores potential violations of these rules. Specifically, each dimension is assigned a binary score—0 if the behavior is deemed unreasonable and 1 if it is reasonable. The scores from all five dimensions are then aggregated to compute a unified VLM-score, ranging from 0 to 5, which provides a quantitative measure of the overall rule compliance in the generated videos.

(gen) i

Defined as SDi = L

− 1. SD quantifies the rela-

L(igt)

tive path-length redundancy of the generated trajectory, representing how much longer the model’s path is compared to the optimal one. A smaller SD indicates higher efficiency and closer adherence to the optimal solution.

[Figure 45]

[Figure 46]

[Figure 47]

Motion Continuity

[Figure 48]

Temporal Consistency

[Figure 49]

Ball deformation

Since Structural Consistency offers only a binary judgment of whether the maze layout has changed, we introduce Maze Fidelity (MF) to quantitatively measure the degree of structural consistency across frames, which defined as

[Figure 50]

[Figure 51]

[Figure 52]

Trajectory Rationality

[Figure 53]

[Figure 54]

Ball Wall Traversal

MF = M1 Mi=1 1 − |p:|I

0(p)−Ii(p)|>τ|

Structural Consistency

Ni .

[Figure 55]

[Figure 56]

Here, M is the number of sampled frames; I0 and Ii denote the background regions of the first and the i-th frames; τ is the pixel-difference threshold; and Ni is the number of valid overlapping pixels. MF quantifies background stability across frames, with higher values indicating better preservation of the static maze layout.

[Figure 57]

Interactional Rationality

Maze Layout Change

Figure 3. Bad case visualization and VLM-as-judge schematic

#### 4. Experiment

Rule Compliance. Not all generated videos faithfully depict the ball following the maze path to complete its task. As shown in Figure 6, we observed numerous failure cases during testing, including the ball “breaking through maze walls,” “disappearing and reappearing,” and “inconsistent maze layouts across frames.” To consistently evaluate the model’s adherence to spatial and physical rules [18], we designed a prompt-based assessment protocol that exam-

To comprehensively evaluate the reasoning capability of video models, we conduct experiments on our proposed VR-Bench. We evaluate both state-of-the-art proprietary and open-source video models on this benchmark. To highlight the advantages of video models over traditional multimodal approaches, we also include representative VLMs in our evaluation. In addition, we fine-tune the open-source

MF (↑) VLM-Score (↑)

Method

Base Irreg Trap 3D Soko Base Irreg Trap 3D Soko

Closed-Source

Veo-3.1-fast 43.2 86.3 22.5 69.1 63.5 0.8 2.5 1.1 1.7 2.0 Veo-3.1-pro 80.5 89.3 82.2 73.4 95.8 2.5 2.8 1.5 2.0 2.5 Sora-2 96.5 97.2 97.2 95.4 95.2 3.9 4.1 3.9 3.3 3.2 kling-v1 55.5 73.8 54.4 84.9 72.9 1.4 2.5 1.9 2.8 2.8 Seedance-1.0-pro 87.7 97.2 64.3 86.3 83.0 3.0 3.8 2.4 2.7 3.2 MiniMax-Hailuo-2.3 92.4 93.4 91.4 94.9 93.3 3.5 3.4 3.2 3.7 3.6

GeneralVideoModel

###### Open-Source

Wan2.5-i2v-preview 69.2 74.8 70.6 82.7 90.4 1.2 1.8 2.6 2.6 2.9 Wan2.2-TI2V-5B♢ 85.5 97.4 83.7 94.7 93.5 2.8 3.1 1.5 3.3 3.1

Wan-R1 91.2 98.1 93.3 95.7 94.1 4.2 4.3 4.4 4.0 4.1

Ours

-Wan2.2-TI2V-5B♢ +5.7 +0.7 +9.6 +1.0 +0.6 +0.3 +0.2 +0.5 +0.7 +0.5

- Table 2. MF and VLM-Score denote Maze Fidelity and the rulecompliance score evaluated by a VLM. The best and second-best results in each column are bolded and underlined.

video model Wan2.2-TI2V-5B using VR-Bench to investigate their generalization ability on reasoning tasks. This allows us to assess whether such reasoning capabilities can emerge via supervised fine-tuning, and whether they can generalize across different settings.

##### 4.1. Training Configurations

To investigate how well open-source video models can acquire and generalize reasoning abilities through fine-tuning, we trained Wan-R1 based on the proposed dataset.

Specifically, we used the first scene from each of the five game types in our benchmark. For each game, we created two training settings: one using only easy samples, and another using a mixture of easy, medium, and hard samples. In each case, the data was split into 80% for training and 20% for validation.

All models were fine-tuned using the Accelerate framework on A100 GPUs. We adopted a LoRA-based training strategy on the Wan2.2-TI2V-5B architecture, with a learning rate of 1e-4, image resolution of 512×512, and video length of 193 frames. We applied LoRA (rank 32) to key attention and feedforward modules (q, k, v, o, ffn.0, ffn.2) of the Dit backbone. Each model was trained for 5 epochs, with a dataset repetition factor of 100. Other training parameters such as batch size and GPU days are in Appendix.

##### 4.2. Baseline Model

We compare the Wan-R1 against a wide range of baselines: 1) 6 Closed-source video models: Veo-3.1-fast, Veo-3.1pro [8], Sora-2 [28], Kling-v1 [20], Seedance-1.0-pro [11], MiniMax-Hailuo-2.3 [25]. 2) 2 Open-source video models: Wan2.5-i2v-preview [38], Wan2.2-TI2V-5B [40]. 3) 3 VLMs: Gemini-2.5-pro [6], Gpt-5 high [26], Qwen2.5-VL7B [2]. The settings of the baseline are in the appendix.

Task EM SR PR SD

Easy Med. Hard Easy Med. Hard Easy Med. Hard Easy Med. Hard Base

0.0 0.0 0.0 8.3 8.3 4.2 13.2 3.8 2.7 154.8 – –

0.0

4.2

0.0

83.3

41.7

58.3

40.1

26.1

6.0

28.2

10.1

–

(+0.0)

(+4.2)

(+0.0)

(+75.0)

(+33.4)

(+54.1)

(+26.9)

(+22.3)

(+3.3)

(-126.6)

(–)

(–)

0.0 0.0 0.0 29.2 4.2 4.2 13.4 8.1 5.8 48.3 – 39.3

Irrg

83.3

66.7

54.2

95.8

87.5

62.5

88.0

74.8

68.4

3.5

7.8

3.1

(+83.3)

(+66.7)

(+54.2)

(+66.6)

(+83.3)

(+58.3)

(+74.6)

(+66.7)

(+62.6)

(-44.8)

(–)

(-36.2)

0.0 0.0 0.0 0.0 0.0 0.0 6.0 6.4 8.9 – – –

Trap

62.5

0.0

12.5

100.0

95.8

62.5

86.7

43.7

56.7

2.1

5.9

3.5

(+62.5)

(+0.0)

(+12.5)

(+100.0)

(+95.8)

(+62.5)

(+80.7)

(+37.3)

(+47.8)

(–)

(–)

(–)

0.0 0.0 0.0 54.2 16.7 25.0 7.6 9.8 15.2 53.0 87.9 33.6

3D

41.7

0.0

4.2

100.0

79.2

83.3

78.7

47.4

58.7

4.6

10.2

13.8

(+41.7)

(+0.0)

(+4.2)

(+45.8)

(+62.5)

(+58.3)

(+71.1)

(+37.6)

(+43.5)

(-48.4)

(-77.7)

(-19.8)

0.0 0.0 0.0 20.8 8.3 4.2 14.6 8.3 5.6 354.2 61.4 –

Soko

4.2

0.0

0.0

83.3

45.8

33.3

62.2

12.5

10.4

18.8

84.5

16.1

(+4.2)

(+0.0)

(+0.0)

(+62.5)

(+37.5)

(+29.1)

(+47.6)

(+4.2)

(+4.8)

(-335.4)

(+23.1)

(–)

Table 3. Difficulty generalization of Wan-R1 on VR-Bench. Each task block compares the baseline (Wan2.2-TI2V-5B) and the finetuned model (trained only on Easy level) across difficulty levels (Easy, Medium, Hard) and four metrics (EM, SR, PR, SD). Green indicates improvements, red indicates degradation, gray denotes no change or undefined cases.

#### 5. Insights and Discussions

In this section, we discuss the observations and insights we draw from our comprehensive evaluation experiments.

· Wan-R1 Outperforms Prior Models on VR-Bench. As shown in Table 1, our method consistently achieves top performance across nearly all tasks and evaluation metrics, demonstrating both high accuracy and rollout efficiency. Notably, Wan-R1 attains a perfect SR of 100.0 on the Trap and 3D maze tasks, highlighting its robust success capabilities even in complex environments. Compared to its base model Wan2.2-TI2V-5B♢, Wan-R1 achieves a remarkable EM improvement of +65.3 on 3D, and reduces SD by 100.1 on Soko. These gains underscore the effectiveness of our fine-tuning strategy in enhancing both correctness and trajectory quality across diverse reasoning settings.

· Success Alone Does Not Guarantee Efficient Reasoning. Some models manage to complete tasks, but their rollouts remain highly inefficient. For instance, Sora-2 and MiniMax-Hailuo-2.3 achieve strong SR of 75.0 and 68.1 on the Base task, yet their corresponding SD values reach 302.9 and 464.0, revealing substantial path redundancy. Even more striking, the open-source VLM Qwen2.5-VL-7B produces a very low SR of 1.4, but still yields a high SD of 300.0, indicating unstable or erratic generation. As shown in Table 1, Wan-R1 achieves a comparable or better SR while reducing SD to just 10.3, demonstrating its ability to generate correct and efficient trajectories consistently.

· Reasoning via Video Outperforms Reasoning via Text. Under the same training data and settings, we fine-tune both the vision-language model (Qwen2.5-VL-7B) and the video model (Wan2.2-TI2V-5B). As shown in Table 1, the videobased model (Wan-R1) yields significantly larger gains across all metrics and tasks, especially in challenging settings like Trap and 3D. In contrast, Qwen2.5-VL-7B-SFT

###### Task Model EM SR PR SD

Base Irreg Trap 3D Soko Base Irreg Trap 3D Soko Base Irreg Trap 3D Soko Base Irreg Trap 3D Soko Wan2.2-TI2V-5B Baseline 0.0 0.0 0.0 0.0 0.0 6.9 12.5 0.0 31.9 11.1 6.6 9.1 7.1 12.8 9.2 388.7 66.1 – 5.4 176.6 Regular Maze Fine-tuned

33.3

5.6

1.4

0.0

0.0

76.4

8.3

88.9

69.4

30.6

60.6

22.7

25.2

13.7

19.0

10.3

51.7

3.8

12.7

49.3

(+33.3)

(+5.6)

(+1.4)

(+0.0)

(+0.0)

(+69.5)

(-4.2)

(+88.9)

(+37.5)

(+19.5)

(+54.0)

(+13.6)

(+18.1)

(+0.9)

(+9.8)

(-378.4)

(-14.4)

(–)

(+7.3)

(-127.3)

0.0

56.9

0.0

0.0

0.0

11.1

69.4

52.8

79.2

12.5

16.6

71.6

18.1

16.8

15.5

35.8

2.4

6.9

9.0

40.7

Irregular Maze Fine-tuned

(+0.0)

(+56.9)

(+0.0)

(+0.0)

(+0.0)

(+4.2)

(+56.9)

(+52.8)

(+47.3)

(+1.4)

(+10.0)

(+62.5)

(+11.0)

(+4.0)

(+6.3)

(-352.9)

(-63.7)

(–)

(+3.6)

(-135.9)

0.0

5.6

0.0

65.3

0.0

38.9

31.9

30.6

100.0

20.8

6.8

20.6

6.7

93.5

15.0

108.2

10.9

8.8

3.9

80.6

3D Maze Fine-tuned

(+0.0)

(+5.6)

(+0.0)

(+65.3)

(+0.0)

(+32.0)

(+19.4)

(+30.6)

(+68.1)

(+9.7)

(+0.2)

(+11.5)

(-0.4)

(+80.7)

(+5.8)

(-280.5)

(-55.2)

(–)

(-1.5)

(-96.0)

0.0

1.4

1.4

0.0

4.2

0.5

22.2

18.1

22.2

69.4

15.7

23.7

36.2

15.7

44.3

46.3

34.4

39.9

20.1

10.2

Sokoban Fine-tuned

(+0.0)

(+1.4)

(+1.4)

(+0.0)

(+4.2)

(-6.4)

(+9.7)

(+18.1)

(-9.7)

(+58.3)

(+9.1)

(+14.6)

(+29.1)

(+2.9)

(+35.1)

(-342.4)

(-31.7)

(–)

(+14.7)

(-166.4)

0.0

0.0

38.9

0.0

0.0

93.1

40.3

100.0

79.2

6.9

10.9

12.9

79.1

14.7

10.0

57.5

16.8

3.9

11.4

57.8

Trapfield Fine-tuned

(+0.0)

(+0.0)

(+38.9)

(+0.0)

(+0.0)

(+86.2)

(+27.8)

(+100.0)

(+47.3)

(-4.2)

(+4.3)

(+3.8)

(+72.0)

(+1.9)

(+0.8)

(-331.2)

(-49.3)

(–)

(+6.0)

(-118.8)

Table 4. Comparison between the baseline (Wan2.2-TI2V-5B) and task-specific fine-tuned models across five game types (Base, Irreg, Trap, 3D, Soko). Each cell reports absolute performance and relative change over the baseline on four metrics: EM, SR, PR, and SD. Green indicates improvement, red indicates degradation, and gray denotes no change or undefined difference.

shows only moderate improvements. This highlights the advantage of reasoning via video in learning temporal reasoning and efficient path planning over static VLMs.

· Rule Compliance and Structural Fidelity. As shown in Table 2, our model Wan-R1 consistently achieves the highest VLM-Score across all maze types, and performs competitively in MF, ranking top-2 in most categories: 1) it consistently attains the highest VLM-Score across all maze types (≥ 4.0), indicating superior rule-following behavior in motion continuity, physical plausibility, and environmental interactions; 2) it ranks among the top performers in MF, especially excelling on structurally complex mazes like Irreg and 3D. 3) it shows consistent improvements over its base model Wan2.2-TI2V-5B, confirming the effectiveness of our training paradigm in enhancing both visual stability and behavioral correctness.

###### · Reasoning via Video Scales Better. 1) As illustrated in

- Figure 4, model performance consistently declines with increasing maze difficulty on the Trap and Irre Maze tasks. In Easy settings, VLMs often match or even surpass state-ofthe-art video models. However, as the maze complexity escalates, VLMs experience a sharper performance drop compared to video models. On large-scale hard mazes, even toptier VLMs such as Gemini-2.5-Pro and GPT-5 are outperformed by leading video models like Sora-2 and Seedance1.0-pro. 2) We attribute this trend to fundamental differences in the reasoning paradigms of the two model families. VLMs rely on encoding static visual observations into textual tokens and performing reasoning within a languagedominant latent space. As maze size increases, the number of visual tokens grows substantially, leading to contextlength saturation and degraded long-horizon reasoning. In contrast, video models reason via visual dynamics, constructing a temporally grounded CoF that maintains spatial continuity across time. This video-centric reasoning mechanism preserves efficiency as scene complexity increases, since the number of visual tokens per frame remains stable, with visual tokens carrying significantly higher infor-

mation density than textual tokens, a finding validated by DeepSeek OCR [43] through optical context compression. Remarkably, models such as Sora-2 even exhibit improved performance in the Irreg Maze under higher difficulty levels, particularly in the SR metric. 3) These observations suggest that reasoning via video constitutes a more native and scalable paradigm for visual reasoning, enabling temporal–spatial problem solving that remains robust under increasing environmental complexity.

Gemini-2.5-pro

Qwen2.5-VL-7B

Veo

VLM avg

Gpt-5 high

Seedance-1.0-pro

Veo-3.1-pro

Video avg

MiniMax-Hailuo-2.3

Sora-2

kling-v1

###### Irregular Maze PR

###### Irregular Maze SR

0.8

0.8

0.6

0.6

0.4

PR

SR

0.4

0.2

0.2

0.0

0.0

Easy Medium Hard

Easy Medium Hard

###### Trapfield PR

Trapfield SR

0.8

0.8

0.6

0.6

PR

SR

0.4

0.4

0.2

0.2

0.0

Easy Medium Hard

Easy Medium Hard

Figure 4. Model performance (PR and SR) on Irregular Maze and Trapfield across difficulty levels. Each curve represents a baseline, while the dashed and dotted lines indicate VLM and Video Model averages. Results for other maze types are in the Appendix.

· Test-Time Scaling for Video Models. 1) Test-time scaling (TTS), exemplified by self-consistency [42], has shown strong effectiveness in text-based reasoning tasks. Its key intuition is that complex reasoning problems often admit multiple valid solution trajectories, and sampling diverse paths increases the likelihood of converging to a correct answer. Maze-solving naturally shares this property: the solution space is open-ended, and multiple routes can lead to the goal. Motivated by this, we apply TTS to video models

by perturbing the sampling noise to generate diverse rollouts, and evaluate performance using Pass@K, which selects the best solution among K independent attempts. 2)

- Figure 5 shows the scaling behavior of Wan-R1 on the Irregular Maze benchmark. As K increases from 1 to 16, the model achieves steady gains across all difficulty levels, with improvements of roughly 10–20% depending on metric and difficulty. On Easy mazes, performance rises almost monotonically and nears saturation at higher K. Medium difficulty shows clear early gains, often improving by 5– 10% from K = 1 to K = 4, followed by continued smaller increases. Even on Hard mazes, where Pass@1 scores are lower, TTS still yields consistent upward trends, recovering solutions that single attempts often miss. 3) Overall, these results demonstrate that TTS significantly enhances videomodel reasoning in VTR tasks. By initializing generation from different noise conditions, the model explores multiple solution pathways within the maze’s open-ended search space. This multi-path exploration effectively unlocks additional reasoning capacity, enabling video models to produce trajectories that are more accurate, more reliable, and consistently closer to the desired solution than those obtained under standard single-sample inference.

· Difficulty Generalization The results in Table 3 demonstrate the strong difficulty generalization capability of WanR1 across all five maze tasks. Although the model is finetuned only on the Easy level, it consistently delivers substantial improvements over the Wan2.2-TI2V-5B baseline on Medium and Hard mazes as well. This indicates that WanR1 does not simply memorize small or low-complexity layouts; rather, it internalizes a more principled and transferable reasoning procedure. The gains observed across unseen difficulty tiers show that fine-tuning on small mazes induces broad generalization: the model acquires a mazesolving strategy that scales to larger, more intricate structures without additional supervision. Such behavior reflects a deeper structural understanding of the environment and verifies that Wan-R1’s improvements stem from reasoningpattern internalization rather than task-specific overfitting.

· Maze Type Generalization. As shown in Table 4, finetuning on a single game (e.g., Regular Maze, Trapfield, or 3D Maze) not only improves in-domain performance but also yields substantial gains on unseen games across all metrics (EM, SR, PR, SD). This highlights the emergence of transferable video reasoning capabilities. Notably, models fine-tuned on 3D Maze exhibit strong generalization, this overall transfer pattern underscores that training on complex 3D structures fosters general reasoning skills applicable to other maze types.

· Texture Generalization. Although fine-tuned only on the Raw skin of each game type, the model shows consistent and often substantial gains on unseen textures (Skin2 and Skin3). As shown in Table 5, this pattern holds across

Task EM SR PR SD

Raw Skin2 Skin3 Raw Skin2 Skin3 Raw Skin2 Skin3 Raw Skin2 Skin3

0.0 0.0 0.0 6.9 4.2 2.8 6.6 4.6 9.4 388.7 11.7 14.9

Base

33.3 1.4 23.6 76.4 38.9 68.1 60.6 12.3 51.0 10.3 19.0 9.0

(+33.3) (+1.4) (+23.6) (+69.5) (+34.7) (+65.3) (+54.0) (+7.7) (+41.6) (-378.4) (+7.3) (-5.9)

0.0 0.0 0.0 12.5 4.2 2.1 9.1 12.0 47.8 66.1 39.5 42.0

Irreg

56.9 22.2 15.3 69.4 15.3 23.6 71.6 36.5 26.2 2.4 5.1 7.7

(+56.9) (+22.2) (+15.3) (+56.9) (+11.1) (+21.5) (+62.5) (+24.5) (-21.6) (-63.7) (-34.4) (-34.3)

0.0 0.0 0.0 0.0 4.2 4.2 7.1 8.9 8.3 – 73.6 59.3

3D

65.3 43.1 50 100.0 97.2 100 93.5 83.8 86.8 3.9 6.4 5.9

(+65.3) (+43.1) (+50.0) (+100.0) (+93.0) (+95.8) (+86.4) (+74.9) (+78.5) (–) (-67.2) (-53.4)

0.0 0.0 0.0 31.9 2.8 7.1 12.8 9.3 8.1 5.4 – –

Soko

4.2 0.0 1.4 69.4 34.7 58.3 44.3 21.0 14.1 10.2 58.6 82.6

(+4.2) (+0.0) (+1.4) (+37.5) (+31.9) (+51.2) (+31.5) (+11.7) (+6.0) (+4.8) (–) (–)

0.0 0.0 0.0 11.1 0.0 0.0 9.2 9.0 7.5 176.6 – – 38.9 9.7 0.0 100.0 38.9 1.4 79.1 29.0 9.9 3.9 8.7 18.5

Trap

(+38.9) (+9.7) (+0.0) (+88.9) (+38.9) (+1.4) (+69.9) (+20.0) (+2.4) (-172.7) (–) (–)

Table 5. Texture generalization under different skin. For each task, the table reports the baseline performance, the results after finetuning on the Raw texture, and the relative change across three texture conditions (Raw, Skin2, Skin3) on EM, SR, PR, and SD. Green denotes improvement, red denotes degradation, and gray indicates no change or undefined differences.

all five task domains. For example, in the Base task, Skin3—never encountered during training—improves by +23.6 in EM and +41.6 in PR. In the more challenging 3D Maze, the generalization is even stronger, reaching +50.0 EM and +78.5 PR on Skin3. These results demonstrate the model’s strong texture-level generalization and indicate that the learned spatio-temporal reasoning transfers robustly to new visual styles.

###### Irregular Maze (Wan2.2-TI2V-5B)

###### Exact Match (EM)

###### Success Rate (SR)

1.0

0.9

0.9

| |
|---|

Pass@KScore

Pass@KScore

0.8

| |
|---|

0.8

| |
|---|

0.7

0.7

| |
|---|

0.6

0.6

0.5

0.5

| |
|---|

| |
|---|

1 4 8 12 16

1 4 8 12 16

K

K

###### Precision Rate (PR)

Step Deviation (SD)

0.04

1.0

| |
|---|

0.03

| |
|---|

| |
|---|

0.9

Pass@KScore

Pass@KScore

0.02

0.01

| |
|---|

0.8

0.00

0.7

- -0.02

- -0.01

0.6

1 4 8 12 16

1 4 8 12 16

K

K

Easy Medium Hard

Figure 5. Performance on Irregular Maze using Wan-R1 under test-time scaling. Results are shown across different sampling numbers (K ∈ 1, 4, 8, 12, 16) and difficulty levels (Easy, Medium, Hard). Results for other maze types are in the Appendix.

#### 6. Conclusion

In this work, we take a step forward in evaluating whether video models can reason via video generation. We propose VR-Bench, a comprehensive benchmark grounded in

maze-solving tasks to assess the spatial reasoning ability of video models. Our experiments demonstrate that fine-tuned video models exhibit strong spatial reasoning and consistently outperform leading vision-language models. Moreover, our analysis reveals a test-time scaling effect akin to self-consistency in language models, underscoring the scalable potential of video-based reasoning.

Limitations and Future Work. While VR-Bench provides a focused and rigorous testbed for spatial reasoning, it currently emphasizes maze-centric tasks. Future iterations of VR-Bench will explore broader and more challenging reasoning scenarios. For instance, we plan to incorporate Olympiad-level problem-solving tasks, such as solving complex physics or mathematics competition problems via video-based visual reasoning. In addition, we aim to support embodied reasoning settings where models are required to predict or simulate coherent action sequences within interactive environments.

#### References

- [1] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023. 3
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 6
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024. 3
- [5] Xinyan Chen, Renrui Zhang, Dongzhi Jiang, Aojun Zhou, Shilin Yan, Weifeng Lin, and Hongsheng Li. Mint-cot: Enabling interleaved visual tokens in mathematical chain-ofthought reasoning. arXiv preprint arXiv:2506.05331, 2025. 2
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2, 6
- [7] Google DeepMind. Veo 2, 2024. Accessed: 2024. 3
- [8] Google DeepMind. Veo-3 technical report. Technical report, Google DeepMind, 2025. 3, 6

- [9] Chengqi Duan, Rongyao Fang, Yuqing Wang, Kun Wang, Linjiang Huang, Xingyu Zeng, Hongsheng Li, and Xihui Liu. Got-r1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning. arXiv preprint arXiv:2505.17022, 2025. 3
- [10] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6824–6835, 2021. 3
- [11] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 6

- [12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3
- [13] Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025. 3
- [14] Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Yufan Deng, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, et al. Video-bench: Human-aligned video generation benchmark. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18858–18868, 2025. 3
- [15] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [16] Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025. 3
- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 3
- [18] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024. 3, 5
- [19] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, et al. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621, 2025. 2
- [20] Kling AI. Kling Video Generation Platform, 2025. Accessed on November 25, 2025. 6

- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [22] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 2
- [23] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3202–3211, 2022. 3
- [24] LumaLabs. Dream machine, 2024. Accessed: 2024. 3
- [25] MiniMax. MiniMax Hailuo 2.3, 2024. Accessed on November 25, 2025. 6
- [26] OpenAI. GPT-5, 2025. Accessed on November 25, 2025. 6
- [27] OpenAI. OpenAI o3 and o4-mini System Card. Technical report, OpenAI, 2025. Accessed: 2025-11-01. 3
- [28] OpenAI. Sora 2 system card. Technical report, OpenAI,

2025. 6

- [29] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022. 3
- [30] PikaLabs. Pika 1.5, 2024. Accessed: 2024. 3
- [31] Runway. Introducing gen-3 alpha: A new frontier for video generation, 2024. Accessed: 2024. 3
- [32] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025. 3
- [33] Yapeng Tian, Jing Shi, Bochen Li, Zhiyao Duan, and Chenliang Xu. Audio-visual event localization in unconstrained videos. In Proceedings of the European conference on computer vision (ECCV), pages 247–263, 2018. 3
- [34] Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang, Xiaomeng Hu, Yining Zheng, et al. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025. 3
- [35] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, et al. Code2logic: Game-code-driven data synthesis for enhancing vlms general reasoning. arXiv preprint arXiv:2505.13886, 2025. 4
- [36] Tony Cheng Tong, Sirui He, Zhiwen Shao, and Dit-Yan Yeung. G-veval: A versatile metric for evaluating image and video captions using gpt-4o. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7419– 7427, 2025. 3
- [37] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022. 3

- [38] Wan. Wan2.5-i2v-preview, 2025. Accessed on November 25, 2025. 6
- [39] Haiyuan Wan, Chen Yang, Junchi Yu, Meiqi Tu, Jiaxuan Lu, Di Yu, Jianbao Cao, Ben Gao, Jiaqing Xie, Aoran Wang, et al. Deepresearch arena: The first exam of llms’ research abilities via seminar-grounded tasks. arXiv preprint arXiv:2509.01396, 2025. 2, 3
- [40] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3, 6
- [41] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171,

2022. 3

- [42] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171,

2022. 7

- [43] Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseekocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025. 7
- [44] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 2, 3
- [45] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 1, 3
- [46] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 3
- [47] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 3
- [48] Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, and Ivan Vuli´c. Visual planning: Let’s think only with images. arXiv preprint arXiv:2505.11409, 2025. 4
- [49] Cheng Yang, Jiaxuan Lu, Haiyuan Wan, Junchi Yu, and Feiwei Qin. From what to why: A multi-agent system for evidence-based chemical reaction condition reasoning. arXiv preprint arXiv:2509.23768, 2025. 2
- [50] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.
- [51] Junchi Yu, Ran He, and Zhitao Ying. Thought propagation: An analogical approach to complex reasoning with large language models. In The Twelfth International Conference on Learning Representations, 2024.

- [52] Zhaoyang Yu, Jiayi Zhang, Huixue Su, Yufan Zhao, Yifan Wu, Mingyi Deng, Jinyu Xiang, Yizhang Lin, Lingxiao Tang, Yingchao Li, et al. Recode: Unify plan and action for universal granularity control. arXiv preprint arXiv:2510.23564, 2025. 2
- [53] Shenghai Yuan, Jinfa Huang, Yongqi Xu, Yaoyang Liu, Shaofeng Zhang, Yujun Shi, Rui-Jie Zhu, Xinhua Cheng, Jiebo Luo, and Li Yuan. Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems, 37:21236–21270, 2024. 3
- [54] Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, et al. Aflow: Automating agentic workflow generation. arXiv preprint arXiv:2410.10762,

2024. 2

- [55] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 3
- [56] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025. 3
- [57] Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, et al. Active-o3: Empowering multimodal large language models with active perception via grpo. arXiv preprint arXiv:2505.21457, 2025. 3

## Reasoning via Video: The First Evaluation of Video Models’ Reasoning Abilities through Maze-Solving Tasks

### Supplementary Material

#### 7. Experiment Details

Parameter Value

##### 7.1. Implementation Details of Baselines

Frame resolution 512 × 512 Number of frames 193 Dataset repeat factor 100 Base model Wan2.2-TI2V-5B LoRA rank 32 Learning rate 1 × 10−4 Epochs 5

Video Models. To ensure fair comparison across different video models, we standardized input preprocessing and output postprocessing. Since most maze images are square (1 : 1), we applied black-border padding to satisfy modelspecific resolution or aspect ratio requirements, followed by center cropping to restore the maze region. Generated videos were temporally aligned according to each model’s fixed or adjustable duration settings.

Table 6. Key training parameters used in our fine-tuning setup.

Veo-3.1 and Veo-3.1-Pro generate fixed 8-second videos in a 9 : 16 aspect ratio, with black-border padding and postcropping applied. Doubao outputs 10-second, 1 : 1 videos, padding and cropping inputs smaller than 300×300 pixels. Kling produces 10-second videos without further adjustments. MiniMax also yields 10-second videos at its default 768p resolution, using the same padding–cropping scheme for small inputs. Sora-2 generates 10-second, 9 : 16 videos, followed by black-border padding and cropping to maintain maze integrity.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

VLM. Given an initial observation image I0, the VLM predicts an action sequence apred = [a1,...,aT], representing its intended movements in the environment. The actions are sequentially executed in the simulator to verify trajectory validity against the optimal reference aopt.

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

For each task type, we define its corresponding action space. In TrapField, Sokoban, and Regular Maze, actions correspond to four-directional moves {up, down, left, right}, and the VLM outputs sequences such as [“up”, “right”, “down”]. In the Irregular Maze, actions are defined over irregular graph nodes (A, B, ...), where the VLM outputs node transition sequences like [“A”, “C”, “E”]; the model is evaluated on whether these transitions correspond to valid path connections. For the 3D Maze, actions include six directional movements covering both horizontal and vertical axes, and the VLM outputs sequences such as [“forward left”, “up”, “forward right”].

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Figure 6. Dynamic visualization of trajectory tracking across five maze types. For each task, the green polyline denotes the groundtruth trajectory, while the blue polyline represents the trajectory tracked from the generated video. Each column shows a temporally ordered frame sampled from the tracking process, illustrating how the agent’s motion evolves over time.

##### 7.2. Training Details

We fine-tuned one model for each of the five maze categories in our benchmark. For each category, the fine-tuning set consists of 80% of the data from the first skin, covering the easy, medium, and hard difficulty levels. All models were trained using the DiffSynth-Studio framework. The training hyperparameters are summarized in Table 6.

#### 8. Evaluation Details

##### 8.1. Path Matching

Setup For each generated video clip we load its UnifiedState description and enumerate all candidate ground-truth (GT) videos sharing the same scene identifier. The generated clip’s spatial resolution and frame rate define a unified evaluation specification; both generated and GT videos are resized and temporally resampled to this specification before comparison.

Trajectory Extraction The controllable agent’s initial bounding box is read from the state and scaled to the evaluation resolution. An object tracker (priority order: CSRT, with fallbacks KCF/MOSSE) is initialized on the first frame. Tracking proceeds frame by frame; sampling follows a fixed temporal interval derived from the unified frame rate. When tracking fails on a sampling frame, the last valid center is reused to preserve temporal alignment. The output is a pixel-space polyline of agent centers across time, paired with the first frame for later visualization.

Normalization and Resampling Extracted pixel trajectories are normalized to the unit square using the evaluated video’s width and height. Physical-distance resampling is applied: the GT trajectory is resampled into a fixed number of equidistant points along arc length, and the generated trajectory is interpolated at the same cumulative distances (clipped to its total length). This produces speed-invariant, sparsity-controlled trajectory pairs.

GT Selection and Outputs Each generated clip is evaluated against all candidate GT trajectories using a score based solely on trajectory length consistency; the GT with the highest score is selected as the match. For every generated clip, we record the matched GT identity and per-clip summary statistics, and we produce visual diagnostics that overlay GT (green) and generated (blue) resampled trajectories on the first frame, along with start (yellow) and goal (red) bounding boxes. Batch evaluation yields per-difficulty summaries and overall totals, enabling direct comparison of trajectory performance across difficulty levels.

##### 8.2. Rule Compliance

Human Alignment To verify that our VLM-as-Judge evaluation faithfully reflects human judgment across all five diagnostic dimensions, we conducted a large-scale human preference study over a diverse set of generated videos. The annotation protocol follows standard practices used in prior video-evaluation benchmarks, with controlled instructions, representative examples, and multiple rounds of cross-checking to ensure consistent and reliable human annotations.

In line with established evaluation frameworks, we computed the correlation between the VLM-as-Judge scores and human preference outcomes. Figure 7 presents the alignment plots, showing the relationship between human win ratios and our VLM-based scores for each dimension, including Motion Continuity, Temporal Consistency, Trajectory Rationality, Structural Consistency, and Interactional Rationality. A linear regression is fitted for visualization, and the Pearson correlation coefficient (ρ) is reported for each dimension. Across all five dimensions, VLM-score aligns closely with human judgment, demonstrating that our proposed evaluation protocol provides a reliable and humanconsistent assessment of video reasoning quality.

#### 9. Additional Analysis

Test-Time Scaling We evaluate how increasing the number of test-time trajectory samples K ∈ {1,4,8,12,16} influences model performance across different maze types and difficulty levels. As illustrated in Figure 9, larger K values generally lead to improved performance across four evaluation metrics: Exact Match (EM), Success Rate (SR), Precision Rate (PR), and Step Deviation (SD).

For Regular Maze, performance improvements are consistent and significant across difficulty levels. All metrics—especially EM and PR—demonstrate strong upward trends with increasing K, particularly for hard instances, indicating enhanced trajectory fidelity and goal alignment.

In Maze3D, higher K yields substantial gains on EM and PR across all difficulty levels. Notably, SR remains saturated (close to 1.0), suggesting that while reaching the goal is easy, fine-grained precision benefits from more diverse samples. SD decreases with K, highlighting trajectory smoothness gains.

Sokoban exhibits the most challenging dynamics: absolute scores remain low, especially on EM and PR. However, performance steadily improves as K increases, most noticeably on the easy setting. This reflects the high complexity introduced by object manipulation and interaction constraints.

For Trapfield, metrics show moderate but consistent gains, with a gradual rise in EM and PR as K increases. SR is already saturated at lower K values, and SD maintains stability, implying that while high-level goals are achievable, detailed path planning benefits from greater sample diversity.

Overall, the results suggest that test-time scaling is an effective strategy for enhancing model robustness, particularly in structurally complex or interaction-heavy environments. The diminishing returns observed on easy levels also motivate future directions such as difficulty-adaptive sampling strategies.

###### Motion Continuity

###### Temporal Consistency

###### Trajectory Rationality

###### Structural Consistency

###### Interactional Rationality

1.0

= 0.989

= 0.974

= 0.991

= 0.969

= 0.992

0.8

ModelScore

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Human

Human

Human

Human

Human

Fitting

Veo-3.1-pro

kling-v1

MiniMax-Hailuo-2.3

Wan2.2-TI2V-5B

| |
|---|

Sora-2

Wan2.5-i2v-preview

Wan-R1

Veo-3.1-fast

Seedance-1.0-pro

- Figure 7. Human Alignment of VLM-as-Judge Evaluation. Each plot corresponds to Motion Continuity, Temporal Consistency, Trajectory Rationality, Structural Consistency, or Interactional Rationality. Dots show the human preference win ratio (horizontal axis) and the VLM-score (vertical axis) for nine video generation models. A linear fit visualizes the correlation, and Pearson’s correlation coefficient (ρ) is reported for each dimension, indicating close alignment between VLM-as-Judge scores and human judgment across all five dimensions.

[Figure 78]

- Figure 8. Model performance across all five maze types and difficulty levels. Each subplot corresponds to a specific maze type (Base, Irregular, Trapfield, 3D, Sokoban) and evaluation metric (Exact Match, Success Rate, Precision Rate, or Step Deviation). Each curve represents a baseline model, while the dashed and dotted lines denote the VLM and Video Model averages. Full metric trends across difficulty levels illustrate the degradation patterns and performance differences among models.

Model Performance Across Difficulty Levels To better understand model behavior under varying spatial structures and difficulty levels, we conduct a comprehensive analysis over all five maze types using all four evaluation metrics (EM, SR, PR, SD). As shown in Fig. 8, this section highlights global performance trends, examines cross-model and cross-metric differences, and discusses how structural characteristics of each maze family amplify specific failure modes. Together, these observations provide a detailed view of the strengths and limitations of both video models

and VLMs in long-horizon trajectory reasoning tasks.

1) Overall Difficulty Trends. Performance consistently declines with increasing difficulty across all maze types and metrics. Across all five maze types (Base, Irregular, Trapfield, 3D, Sokoban), model performance decreases steadily from Easy to Hard difficulty, and this pattern is consistent across EM, SR, PR, and SD. The decline is most pronounced in Irregular and Trapfield, where complex geometry and branching paths increase long-horizon planning difficulty. On Easy mazes, several VLMs perform

###### Regular Maze (Wan2.2-TI2V-5B)

###### Exact Match (EM)

###### Success Rate (SR)

1.0

| |
|---|

| |
|---|

0.6

0.9

| |
|---|

| |
|---|

0.8

Pass@KScore

Pass@KScore

0.5

| |
|---|

0.7

0.4

0.6

0.3

0.5

0.2

0.4

0.3

0.1

1 4 8 12 16

1 4 8 12 16

K

K

###### Precision Rate (PR)

Step Deviation (SD)

0.8

0.20

| |
|---|

Pass@KScore

Pass@KScore

0.7

0.15

0.6

0.10

0.05

0.5

0.00

1 4 8 12 16

1 4 8 12 16

K

K

Easy Medium Hard

###### Sokoban (Wan2.2-TI2V-5B)

###### Exact Match (EM)

###### Success Rate (SR)

0.35

0.08

0.30

0.25

0.06

Pass@KScore

Pass@KScore

0.20

0.04

0.15

0.10

0.02

0.05

0.00

0.00

1 4 8 12 16

1 4 8 12 16

K

K

###### Precision Rate (PR)

Step Deviation (SD)

0.7

0.04

0.6

Pass@KScore

Pass@KScore

0.02

0.00

0.5

- -0.04

- -0.02

0.4

0.3

1 4 8 12 16

1 4 8 12 16

K

K

Easy Medium Hard

###### Maze3D (Wan2.2-TI2V-5B)

###### Exact Match (EM)

###### Success Rate (SR)

| |
|---|

1.04

0.4

| |
|---|

Pass@KScore

Pass@KScore

1.02

| |
|---|

0.3

| |
|---|

| |
|---|

1.00

0.2

0.98

0.1

0.96

0.0

1 4 8 12 16

1 4 8 12 16

K

K

###### Precision Rate (PR)

Step Deviation (SD)

0.90

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.075

| |
|---|

0.88

0.070

| |
|---|

Pass@KScore

Pass@KScore

0.86

0.065

0.84

0.060

0.82

| |
|---|

0.055

0.80

0.050

1 4 8 12 16

1 4 8 12 16

K

K

Easy Medium Hard

###### Trapfield (Wan2.2-TI2V-5B)

###### Exact Match (EM)

###### Success Rate (SR)

0.8

1.04

0.6

Pass@KScore

Pass@KScore

1.02

1.00

0.4

0.98

0.2

0.96

| |
|---|

| |
|---|

0.0

1 4 8 12 16

1 4 8 12 16

K

K

###### Precision Rate (PR)

Step Deviation (SD)

| |
|---|

0.06

0.95

| |
|---|

0.90

0.05

Pass@KScore

Pass@KScore

0.85

0.04

0.80

0.75

0.03

0.70

0.02

0.65

| |
|---|

| |
|---|

1 4 8 12 16

1 4 8 12 16

K

K

Easy Medium Hard

- Figure 9. Performance across four maze types using Wan-R1 under test-time scaling. Results are shown across different sampling numbers (K ∈ {1, 4, 8, 12, 16}) and difficulty levels (Easy, Medium, Hard). Each subplot corresponds to one maze type (Regular Maze, Irregular Maze, Trapfield, and Sokoban) and one evaluation metric (Exact Match, Success Rate, Precision Rate, or Step Deviation), illustrating how test-time scaling influences trajectory accuracy and stability across diverse maze structures.

comparably to video models, but their performance drops much faster as maze complexity increases. On Hard mazes, even strong VLMs such as Gemini-2.5-pro and GPT-5-high are frequently surpassed by video models like Sora-2 and Seedance-1.0-pro. In contrast, Base and Sokoban show milder degradation: Base Maze remains structurally simple, allowing more stable PR and SR, while Sokoban’s gridbased layout reduces positional ambiguity despite strict EM constraints.

2) Cross-Metric and Cross-Model Patterns. Video models display stronger robustness, while VLMs show high variance across maze structures. Across metrics, video models such as Sora-2, Seedance-1.0-pro, and Veo-3.1pro consistently rank highest, especially in PR and SR, which rely on spatial alignment and correct goal attain-

ment. VLMs including Gemini-2.5-pro, Qwen2.5-VL-7B, and GPT-5-high exhibit large performance variance: they perform well on simpler layouts like Base Maze but degrade sharply on Trapfield and 3D Maze. The difference becomes particularly apparent in SD, where VLM trajectories drift substantially as maze complexity grows. EM remains the strictest metric, and most models fail to achieve meaningful EM on Hard variants of Irregular, Trapfield, 3D Maze, and Sokoban, reflecting the difficulty of producing fully correct long-horizon trajectories.

3) Structural Effects Across Maze Families. Different maze structures stress models in distinct ways, highlighting complementary challenges. Irregular Maze and Trapfield serve as the strongest discriminators of model capability: only a few video models maintain usable SR and PR at

Hard difficulty, while most VLMs collapse almost entirely. 3D Maze introduces challenges related to depth perception and occlusion, resulting in instability in PR and SD even at Medium difficulty. Sokoban emphasizes rule consistency: minor violations lead directly to failure, causing sharp drops in EM and SR on Hard difficulty.

Overall, these results show that video models maintain stronger spatial-temporal consistency as difficulty increases, whereas VLMs perform well primarily when maze structures are simple or when short-horizon reasoning dominates. The diversity of maze structures provides a comprehensive stress test for evaluating trajectory fidelity.

#### 10. Failure cases

To illustrate typical failure cases, we adopt a three-frame visualization format for each example:

- • Left: The initial frame of the reasoning video, showing the static scene before any agent action.
- • Middle: A representative intermediate frame capturing the failure behavior during the reasoning process.
- • Right: A trajectory visualization comparing the predicted and ground-truth reasoning paths. The green line denotes the ground-truth trajectory, the blue line represents the model’s predicted path, the yellow square indicates the starting point, and the red square marks the final destination.

These visualization conventions are applied consistently in all subsequent failure case illustrations (see Figures 10– 20).

#### 11. Prompt Template

As shown in Tables, we design unified prompt templates for all five maze types (Base, Irregular, Trapfield, 3D, and Sokoban), each provided in two variants tailored for video models and VLMs. The video model prompts are presented in gray, reflecting their focus on motion-generation constraints, while the VLM prompts are shown in light blue, highlighting their emphasis on trajectory interpretation and rule compliance.

Each maze type follows a consistent prompt structure composed of a system-level prompt and task-specific instructions. For video models, the system prompt specifies the maze layout, agent behavior rules, and rendering constraints, followed by an execution prompt that defines: (1) the valid movement space; (2) expected trajectory behavior; and (3) visual requirements for temporal and spatial consistency across frames. For VLMs, the system prompt defines the evaluation context, and the task prompt formulates: (1) the extracted agent trajectory; (2) rule-checking dimensions; and (3) the expected reasoning format for judging motion validity.

Across the five maze families, these templates provide video models with precise generative instructions, while giving VLMs structured reasoning prompts for consistent evaluation.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

- Figure 10. Failure case: incorrect reasoning leads to unreachable path. Although the model attempts to reach the target, its predicted trajectory (in blue) passes through an infeasible region, resulting in failure to arrive at the goal (red). The ground-truth trajectory is shown in green.

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

[Figure 97]

[Figure 100]

- Figure 11. Failure case: duplicated object during reasoning. During inference, the reasoning object unexpectedly appears in two locations simultaneously, indicating a visual inconsistency likely caused by faulty temporal coherence. This distracts the model and leads to an invalid reasoning path (in blue), which fails to reach the correct goal (red).

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

[Figure 115]

[Figure 118]

- Figure 12. Failure case: suboptimal path to target. The model starts at the green point and eventually reaches the red goal, but takes a significantly longer path than the shortest possible route. This suggests limitations in global path planning and trajectory efficiency.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

- Figure 13. Failure case: invalid path through obstacles. The model starts from the blue point and attempts to reach the green goal, but its predicted trajectory crosses red obstacle regions, violating the environment’s physical constraints.

[Figure 141]

[Figure 144]

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

- Figure 14. Failure case: spurious object motion and goal inactivity. During reasoning, the model introduces a moving object unrelated to the task, while the actual target object remains static. This reflects a failure to correctly ground the reasoning process on the intended object and leads to an incorrect trajectory.

[Figure 159]

[Figure 162]

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

- Figure 15. Failure case: target object replaced by incorrect appearance. The target object is initially a carrot, but during reasoning it is incorrectly replaced by a red dot. This severe visual inconsistency indicates the model’s failure to maintain object identity, undermining its ability to reason about the correct target.

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

- Figure 16. Failure case: incorrect understanding of object interaction. The model misinterprets the intended interaction between the character and the box. Instead of pushing the box toward the target area, the model attempts to approach the goal by pulling the box, which violates the correct physical reasoning for this task.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

- Figure 17. Failure case: moving object disappears midsequence. The model begins reasoning from the blue object and generates a few initial frames correctly. However, the moving object then vanishes unexpectedly, leaving only the path trace without a visible agent completing the trajectory.

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

- Figure 18. Failure case: reversed motion direction and object deformation. The task requires the puck to move toward the goal hole, but the model instead makes the hole move toward the puck. Moreover, the goal object is mistakenly deformed into a green sphere, indicating a failure in both physical role understanding and object appearance consistency.

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

- Figure 19. Failure case: incorrect goal identification. The model starts correctly from the purple ball but mistakenly identifies the wrong goal position at the top-left green region. It generates a valid path—yet toward the incorrect endpoint—resulting in a complete failure in target localization.

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

- Figure 20. Failure case: misunderstanding of reasoning target roles. The model misinterprets the intended interaction: instead of moving the green ball toward the red target, it moves both objects simultaneously. This reveals a failure in understanding asymmetric goal-directed behavior and role assignment.

###### Regular Maze

Create a 2D animation based on the provided image of a maze. The red circle slides smoothly along the white path, stopping perfectly on the green square. The red circle never slides or crosses into the blue areas of the maze. The camera is a static, top-down view showing the entire maze.

###### Maze:

- • The maze paths are white, the walls are blue.
- • The red circle moves to the goal position, represented by a green square.
- • The red circle slides smoothly along the white path.
- • The red circle never slides or crosses into the blue areas of the maze.
- • The red circle stops perfectly on the green square.

###### Scene:

- • No change in scene composition.
- • No change in the layout of the maze.
- • The red circle travels along the white path without speeding up or slowing down.

###### Camera:

- • Static camera.
- • No zoom.
- • No pan.
- • No glitches, noise, or artifacts.

###### Irregular Maze

- Create a 2D animation based on the provided image of a maze. The green circle slides smoothly along the white path, stopping perfectly on the red circle. The green circle never slides or crosses into the black areas of the maze. The camera is a static, top-down view showing the entire maze. Maze:

- • The maze paths are white, the walls are black.
- • The green circle moves to the goal position, represented by a red circle.
- • The green circle slides smoothly along the white path.
- • The red circle never slides or crosses into the black areas of the maze.
- • The green circle stops perfectly on the red circle.

###### Scene:

- • No change in scene composition.
- • No change in the layout of the maze.
- • The green circle travels along the white path without speeding up or slowing down.

###### Camera:

- • Static camera.
- • No zoom.
- • No pan.
- • No glitches, noise, or artifacts.

###### 3D Maze

- Create a 3D animation based on the provided image of a cube maze.A yellow ball slides smoothly along the gray cube pathway, climbs up the vertical ladders step by step, and finally stops perfectly on the red cube at the top.The yellow ball never touches or passes through the blue cube or any non-gray areas of the maze.The camera remains static in an isometric, top-down angle showing the entire structure. Maze:

- • The maze consists of stacked transparent gray cubes forming a 3D pathway.
- • The red cube represents the goal position. The blue cube marks the starting platform where the yellow ball begins.
- • The yellow ball moves upward along the gray path, climbing vertically via the ladders.
- • The ball slides smoothly without sudden changes in direction or speed.
- • The ball stops exactly on top of the red cube at the end.

###### Scene:

- • No structural or color changes during animation.
- • The maze layout and cube arrangement remain unchanged.
- • The yellow ball moves continuously at a constant speed along the 3D path.

###### Camera:

- • Static camera.
- • No zoom.
- • No pan.
- • No glitches, noise, or artifacts.

###### Trapfield

Create a 2D animation based on the provided image of a maze. The blue circle slides smoothly along the gray path, stopping perfectly on the green circle. The blue circle never slides into or crosses the red areas with crosses (trap areas). The camera is a static, top-down view showing the entire maze.

###### Maze:

- • The maze paths are gray, and the trap areas are red with crosses.
- • The blue circle moves to the goal position, represented by the green circle.
- • The blue circle slides smoothly along the gray path.
- • The blue circle never slides into or crosses the red areas with crosses of the maze.
- • The blue circle stops perfectly on the green circle.

###### Scene:

- • No change in scene composition.
- • No change in the layout of the maze.
- • The blue circle travels along the gray path without speeding up or slowing down.

###### Camera:

- • Static camera.
- • No zoom.
- • No pan.
- • No glitches, noise, or artifacts.

###### Sokoban

Create a 2D animation based on the provided image of a grid puzzle. The blue ball moves into position behind the yellow square and smoothly pushes it toward the red goal square. The yellow square only slides when pushed from behind by the blue ball and moves in a straight line along the white floor tiles. When the direction of the yellow square’s movement needs to change, the blue ball must reposition itself to a new side of the yellow square.The square never crosses or overlaps any gray walls.

###### Maze:

- • The floor area is white, and the walls are gray.
- • The yellow square can only move when pushed by the blue ball from behind.
- • The blue ball cannot pull the square or move through walls.
- • The yellow square slides smoothly in one direction until it reaches the red goal square.
- • The animation stops perfectly when the yellow square aligns with the red goal square.

###### Scene:

- • No change in scene composition.
- • No change in the layout of the maze.
- • The movement is smooth, with no speed variation.

###### Camera:

- • Static camera.
- • No zoom.
- • No pan.
- • No glitches, noise, or artifacts.

Sokoban You are given an image of a grid-based Sokoban puzzle. Gray tiles represent walls and cannot be crossed. White tiles represent open floor tiles that can be moved through. The blue ball represents the player or agent. The yellow square represents the box that needs to be pushed. The red square represents the goal destination for the box.

Task: Infer the complete movement sequence required for the blue ball to push the yellow square onto the red goal square. The blue ball moves in four directions: up, down, left, right. When the blue ball moves into a box, it automatically pushes the box if there is space behind it. The box and the blue ball cannot cross or overlap any gray walls. Diagonal movement is not allowed, and the camera remains fixed from a top-down view.

Output Format: Return the entire movement sequence as a JSON array of directional actions, where each element is one of “up”, “down”, “left”, or “right”. Do not include any explanations or additional text.

Example of expected output: { ”actions”: [”right”, ”right”, ”down”, ”left”, ”down”] }

Regular Maze You are given an image of a grid-based maze. Black tiles represent walls and cannot be crossed. White tiles represent open paths that can be moved through. The green circle represents the starting point of the path. The red circle represents the goal or destination.

Task: Infer the shortest valid path from the green starting point to the red goal circle. Movement can only occur between adjacent open tiles — up, down, left, or right. Diagonal movement is not allowed, and the path must not cross or touch any black walls.

Output Format: Return the entire movement sequence of the green circle as a JSON array of directions, where each element is one of “up”, “down”, “left”, or “right”. Do not include any explanations or additional text.

Example of expected output: { ”path”: [”up”, ”up”, ”left”, ”down”, ”right”, ”right”] }

Irregular Maze You are given an image of a pathfinding puzzle. The image shows a network of curved paths connecting various waypoints. Each waypoint (intersection or junction) is labeled with a letter or letter combination (A, B, C, ..., Z, AA, AB, etc.). The green circle represents the starting point. The red circle represents the goal or destination.

Task: Find the shortest valid path from the green starting point to the red goal. The path must follow the visible roads/paths in the image. You can only move along the connected paths shown in the image.

Output Format: You MUST return a JSON object with a “path” field containing an array of waypoint labels. The array should start with the label closest to the starting point and end with the label closest to the goal. Do not include any explanations or additional text. Important: The “path” field MUST be an array of strings, not a single string.

Example of expected output: { ”path”: [”A”, ”B”, ”C”, ”D”, ”E”] }

Maze3D You are given an image of a 3D maze composed of gray cubes that represent walkable platforms suspended in space. Each cube represents a solid tile that the ball can stand on or move across. The yellow sphere represents the starting point. The blue cubes represent the initial platform where the ball begins. The red cube represents the goal or destination.

Task: Infer the shortest valid 3D path for the yellow sphere to move from its starting position to the red goal cube.

Movement Rules: Horizontal movements (forward left, forward right, backward left, backward right): each move spans 2 grid units horizontally. Vertical movements (up, down): each move spans 3 grid units vertically via a ladder; a ladder must be present at the starting position. The sphere cannot move through empty space or overlap any cube structure. All movements must follow valid cube surfaces and ladder connections.

The six valid directions of movement are: “forward left” – move diagonally forward and to the left (2 units) within the same layer. “forward right” – move diagonally forward and to the right (2 units) within the same layer. “backward left” – move diagonally backward and to the left (2 units) within the same layer. “backward right” – move diagonally backward and to the right (2 units) within the same layer. “up” – move vertically upward (3 units) via a ladder. “down” – move vertically downward (3 units) via a ladder.

Output Format: Return the full sequence of movement directions as a JSON array, where each step is one of the six valid directions. Do not include any explanations, reasoning, or extra text.

Example of expected output: { ”path”: [”up”, ”forward right”, ”forward left”, ”up”, ”forward right”] }

Trapfield You are given an image of a grid-based maze. Red tiles marked with an “X” represent trap zones that must be avoided. White tiles represent open paths that can be moved through. The blue circle represents the starting point of the path. The green circle represents the goal or destination.

Task: Infer the shortest valid path for the blue circle to reach the green circle. Movement can only occur between adjacent open tiles – up, down, left, or right. Diagonal movement is not allowed. The path must not cross or touch any red trap tiles.

Output Format: Return the full movement sequence of the blue circle as a JSON array of directions, where each element is one of “up”, “down”, “left”, or “right”. Do not include any explanations, reasoning, or extra text.

Example of expected output: { ”path”: [”left”, ”left”, ”down”, ”down”] } 11

