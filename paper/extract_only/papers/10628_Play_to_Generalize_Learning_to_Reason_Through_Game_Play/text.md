# arXiv:2506.08011v4[cs.CV]9Oct2025

## Play to Generalize: Learning to Reason Through Game Play

Yunfei Xie1, Yinsong Ma2, Shiyi Lan3, Alan Yuille2, Junfei Xiao2*, Chen Wei1† 1Rice University, 2Johns Hopkins University, 3NVIDIA

Website https://yunfeixie233.github.io/ViGaL

Code & Model & Data https://github.com/yunfeixie233/ViGaL

### Abstract

Developing reasoning capabilities in multimodal large language models (MLLMs) remains challenging. Motivated by literature suggesting that gameplay promotes transferable reasoning skills, we propose a novel post-training method, Visual Game Learning (ViGaL), where MLLMs develop generalizable reasoning skills through playing arcade-like games. Specifically, we show that training a 7B-parameter MLLM via reinforcement learning (RL) on simple games like Snake significantly enhances the downstream performance on multimodal math benchmarks like MathVista, on multi-discipline questions like MMMU and on 3D spatial reasoning benchmarks like VSI-Bench, without seeing any worked solutions, equations, or diagrams during RL. Remarkably, our model outperforms specialist models post-trained on benchmark-oriented multimodal reasoning data, while preserving the model’s performance on general visual benchmarks, a challenge where specialist models often fall short. Our findings suggest that multimodal reasoning can emerge from gameplay, pointing to a promising strategy of designing surrogate tasks for RL post-training.

Average Accuracy Increase on Mathverse, Mathvision and Mathvista

Post-Training RL on Visual Games

Evaluation Downstream Tasks

RL on game +2.9% RL on math +2.4%

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Emergent Ability…

Math Data Required for RL

[Figure 5]

[Figure 6]

RL on game 0 RL on math 12K

Math Reasoning

MultiDiscipline

Snake Game

Figure 1 | Overview of ViGaL. Left: We propose a novel post-training method where MLLMs are finetuned via RL to play arcade-style games such as Snake [34]. We demonstrate that gameplay post-training enables MLLMs to achieve out-of-domain generalization, enhancing their performance on downstream multimodal reasoning tasks requiring math, spatial and multi-discipline reasoning, without using math or multi-displine data during RL. Right: Our ViGaL (RL on game) achieves higher average accuracy increase than MM-Eureka [50] (RL on math) across three multimodal math benchmarks. This is notable because MM-Eureka trains on large-scale, curated math datasets, while ViGaL only uses game data. Details are in Tab 2.

*Project Lead; †Corresponding Author

#### 1. Introduction

Games, beyond their entertainment value, provide rich and diverse structured environments for developing and studying general reasoning and problem-solving abilities. Humans from early childhood acquire foundational cognitive skills through diverse game-like activities such as arranging objects, navigating spaces, and manipulating tools. These experiences foster essential building blocks of abstract thinking, including pattern recognition, spactial reasoning, and causal inference [9, 10]. In cognitive science, games are used as experimental platforms to reveal the inductive biases of the human mind [3, 4], such as planning depth in the game Four-in-a-Row [67], or the cognitive basis of tool use through the game Virtual Tools [5].

AI agents, too, have benefited from games resembling aspects of human play. These environments encourage exploration, robustness to sparse rewards, and learning from multimodal inputs. For example, emergent tool use has been observed in agents trained via hide-andseek [8], and Atari gameplay has been incorporated into training generalist agents [55]. By learning in these environments, AI systems develop robust and transferable reasoning capabilities.

Recent work has shown that post-training with Reinforcement Learning (RL) can unlock reasoning behaviors from their base models [18, 51]. These RL-trained models can “think before they speak”, generating internal chain-of-thought traces before outputting a final answer. More importantly, growing evidence suggests that RL often generalizes more robustly to out-ofdistribution samples than supervised fine-tuning (SFT). For example, models trained with RL on mathematics transfer their reasoning skills to physics [50], and navigation agents adapt to novel environments beyond their training domains [16]. Motivated by these findings, we ask: since games already serve as a natural medium through which humans acquire reasoning strategies, can post-training multimodal LLMs on gameplay similarly enhances their ability to reason across diverse tasks?

The results are striking (Fig. 1). We show that post-training a 7B-parameter multimodal LLM, Qwen2.5-VL-7B [7], to play simple arcade-style games like Snake [34] yields two surprising outcomes: (1) the model generalizes to previously unseen games (Sec. 2.3); and (2) it exhibits strong reasoning abilities on multimodal math benchmarks like MathVista [46], and multi-domain QA like MMMU [78]. Despite never observing worked solutions, equations, or diagrams during RL post-training, the model achieves competitive results not only against large-scale industrial systems like GPT-4o [33], but also against specialist models post-trained on math datasets (Tabs. 2 and 3). Furthermore, it improves on reasoning benchmarks without degrading general visual understanding, a common limitation of domain-specialist training (Tab. 6). Overall, gameplay emerges as an effective surrogate task for incentivizing reasoning in multimodal LLMs.

Why does it work? Our ablation studies suggest that reasoning skills incentivized by gameplay can be helpful to other multimodal reasoning tasks. For example, Snake, a game set on a 2D grid where the player maneuvers the “snake” to avoid collisions and collect apples, significantly improves performance on math problems involving 2D coordinates. In contrast, Rotation, a puzzle requiring recognition of 3D object rotation angles, more strongly boosts performance on geometry questions involving angles and lengths (Fig. 3). Furthermore, jointly training on both games yields consistently stronger results on downstream benchmarks than training on either game alone, suggesting the compositionality of the acquired skills (Tab. 2).

All these results point to a new post-training strategy: rather than relying solely on domainspecific datasets, we can design scalable and controllable surrogate tasks for post-training, such as games, that unlock reasoning behaviors transferable to downstream applications. Synthetic

Use reasoning instructions…

MLLM Response

[Figure 7]

RotationGameSnakeGame

- • Obstacle Avoidance
- • Shortest Path
- • Collision Check
- • Movement Prediction Action Space

<think>To determine the best and worst moves, we need to evaluate

- ◼ Snake 1
- ◼ Snake 2
- ◼ Apples

the possible moves and their

consequences: 1.Current Position: My head is at position...</think>

[Figure 8]

<best_answer>RIGHT</best_answer>

[Figure 9]

<worst_answer>DOWN</worst_answer>

[Figure 10]

###### Reflection

Action

Observation

MLLM Response

Use reasoning instructions…

In-Context Example

Game Task

- • Angle Estimation
- • Axis Alignment
- • Triangle Reasoning
- • Polar Analysis Action Space

<think>In the transformation from Image 3 to Image 4, the character has been rotated 90 degrees clockwise. The symmetry axes have

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

180° ?°

shifted 90 degrees...， indicating

[Figure 15]

[Figure 16]

a 90-degree rotation.</think> <answer>90</answer>

[Figure 17]

- Figure 2 | Post-training MLLMs to reason through RL with games. We propose post-training MLLMs via RL by playing visual games. We demonstrate this with two games: the classic arcade game Snake [34], and Rotation, a self-designed task to investigate spatial reasoning. In each game, the model receives multimodal inputs and follows reasoning instructions, e.g., path planning in Snake, angle estimation in Rotation. It reflects to choose an action, outputs its chain-of-thoughts and decision, e.g., best/worst move or predicted angle, and receives a reward. Through gameplay, the model obtains reasoning abilities that transfer to downstream multimodal reasoning tasks such as math and multi-discipline question answering.

game environments offer structured, rule-based rewards and fine-grained controllability, while also scaling far more easily than human-annotated data. This promising paradigm of posttraining with surrogate tasks reminisces self-supervised pre-training in vision and language [20, 30, 54], where carefully designed pretext tasks produce broad generalization.

#### 2. Reinforcement Learning on Visual Games

We introduce ViGaL, a novel post-training paradigm designed to enhance generalization capabilities.

##### 2.1. Game Environment

As show in Fig. 2, under our ViGaL paradigm, the model is trained in a game environment where it receives states from game environment, outputs next actions, and obtains rewards as feedback from the environment. Formally, each task, given an instruction 𝐼, can be formulated as a partially observable Markov decision process (POMDP): (S, A, O,𝑇, 𝑅, Ω), where S is the set of possible environment states, O is the set of observations available to the model, and A represents actions model can do in this game environment. 𝑇 : S × A → S is the state transition function, while 𝑅 is a binary reward from the environment representing the correctness of action. Due to partial observability, the agent perceives only observations 𝑜 = Ω(𝑠).

Snake and Rotation Games. We design two complementary games, Snake and Rotation, to study the proposed paradigm (Fig. 2), each focusing on different MLLM capabilities. The Snake game, inspired by prior work showing that competition can enhance reasoning in MLLMs [22], emphasizes strategic decision-making. We set up a dual-snake game based on SnakeBench [34], where each model independently controls one snake. The objective is to reach

Model Wins (/10) ViGaL vs.

Model Acc. (%)

ViGaL 71.9 Qwen2.5-VL-7B 47.4 Qwen2.5-VL-72B 52.1 Llama-4-Maverick 66.2 Gemini-2.5-Pro 51.0 Claude-3.7-Sonnet 65.6 GPT-4o 61.5 o4-mini 70.8

Qwen2.5-VL-7B 9 Qwen2.5-VL-72B 7 Llama-4-Maverick 7 Gemini-2.5-Pro 8 Claude-3.7-Sonnet 6 GPT-4o 8 o4-mini 6

Game ViGaL Qwen2.5-VL-7B

Space Invaders 280.0 85.0 Ms. Pacman 1370.0 670.0 Seaquest 80.0 60.0 Alien 540.0 450.0 Frogger 7.0 5.0 Breakout 0.0 9.0 Pong -26.0 -26.0

Cumulative Reward 2251.0 1253.0

(a) Snake game.

(b) Rotation game.

(c) Atari game.

- Table 1 | Game Performance. (a) ViGaL gets high win rates (6-9 wins out of 10 matches) on Snake playing against advanced proprietary models. (b) ViGaL shows best performance on Rotation. (c) ViGaL trained on Snake and Rotation shows zero-shot generalization to unseen Atari games, achieving a nearly doubled cumulative reward compared to its base model (Qwen2.5-VL-7B).

apples, score points, and outcompete the opponent. At time 𝑡, the environment state 𝑠𝑡 includes the coordinates of both snakes (𝑥𝑠𝑡𝑖, 𝑦𝑠𝑡𝑖) for 𝑖 ∈ {1,2}, the apple location (𝑥𝑎𝑡, 𝑦𝑎𝑡), and the previous actions 𝐴𝑡−1

𝑖 . All elements are placed on a 10 × 10 board. Each snake then selects its next action

𝐴𝑡𝑖 ∈ {up,down,left,right}. A snake dies if it collides with itself, the other snake, or the board boundary; the survivor wins, or in the case of simultaneous death, the higher score decides.

Unlike SnakeBench, which uses only text to represent states, we provide both images of the game board and textual descriptions as observations 𝑜𝑡 = Ω(𝑠𝑡) for richer input. The Rotation game, inspired by rotation-angle prediction as a pre-text task in self-supervised learning [29], evaluates visual perception and spatial reasoning. The model is presented with two views of the same 3D object: an initial view 𝐼init and a rotated view 𝐼rot, obtained by rotating the object 90◦ or 180◦ around the 𝑧-axis (pointing toward the viewer). The task is to identify which rotation angle transforms 𝐼init into 𝐼rot. To guide reasoning, we include an in-context example with a known rotation. As in the Snake game, observations combine images and text. Together, these two games allow systematic exploration of reasoning and perception, two fundamental aspects of MLLM abilities.

##### 2.2. Rule-Based Reinforcement Learning

We apply rule-based RL to directly post-train MLLMs for visual games, without relying on supervised learning as a warm up. The algorithm is described as follows:

Reward design. We use a simple rule-based reward function to avoid reward hacking [28] and help the model learn how to play the games effectively. This reward function has two components: an accuracy reward and a format reward. The total reward 𝑟 is computed as the sum of an accuracy reward and a format reward 𝑟 = 𝑟accuracy + 𝑟format. The accuracy reward 𝑟accuracy is 1 if the answer is correct, and 0 otherwise. Details of reward for each game are in Appendix Sec. A.3.

Advantage estimation and policy update. We employ REINFORCE Leave-One-Out (RLOO) algorithm [2, 37] in our RL training phase. Following Group Policy Gradient [17], we omit KL divergence regularization. Without KL constraints limiting policy changes, the model explores the solution space more freely, potentially discovering better reasoning strategies. This enables more flexible adaptation during RL training.

##### 2.3. Implementation and Evaluation on Games

Implementation details. We employ Qwen2.5-VL-7B-Instruct [7] as our base model. We follow DeepSeek-R1 [18], using a combination of rule-based format rewards and accuracy rewards,

with RLOO [2, 37] as the core RL algorithm. We implement our training within a multimodal input RL framework based on OpenRLHF [31]. For hyperparameters, we adopt the default settings from MM-Eureka [50], including a global batch size of 128, a rollout batch size of 128, a rollout temperature of 1.0, and a learning rate of 1𝑒−6. Training uses 6 A100-80G GPUs.

Game training data. We build game environments to collect training data for our experiments. For Snake, we leverage SnakeBench [34] as our data engine. For Rotation, we utilize Hunyuan3D [62], which generates 3D meshes from images or text instructions. We render each mesh into 2D images from different orientations, creating image pairs with associated rotation angles as ground truth labels for RL training. Our comprehensive data generation pipeline enables producing training samples at any desired scale with fully customized settings. For experiments, we synthesize 36K samples per game, sufficient for convergence. Details are in Appendix Sec. A.1.

Competing with leading models on Snake and Rotation. To evaluate the game capabilities of ViGaL models, we initialize environments in diverse states unseen during training. For Snake (Tab. 1a), we randomly initialize games 10 times with two models competing directly, measuring win counts. For Rotation (Tab. 1b), we measure rotation angle prediction accuracy on comprehensive validation sets with 3D object meshes unseen during training. Our 7B-parameter model consistently outperforms proprietary models in both games. Results confirm that RL effectively unlocks small 7B models’ ability to excel in visual games requiring environmental understanding, reasoning, planning, and interactive decision-making.

Out-of-distribution generalization to Atari games. We then test ViGaL on Atari-GPT [70], a benchmark for evaluating MLLMs as decision-making agents in Atari video games such as in Fig. 5. The benchmark consists of seven different Atari games, with detailed settings in Appendix Sec. B.1. We follow most settings and prompts from Atari-GPT, with a small modification to ensure format correctness for all models. Following Atari-GPT [70], we report cumulative reward over 1K steps as the evaluation metric, where higher rewards indicate better performance. As shown in Tab. 1c, ViGaL shows significant cumulative reward improvement on Atari games despite being trained only on Snake and Rotation games. This is particularly notable because Atari games differ substantially from our training games in both visual appearance and gameplay strategies. These results suggest that our rule-based RL training approach enables strong generalization to previously unseen game environments.

#### 3. Visual Reasoning Generalization

Evaluation collection. Following prior studies [40, 65], we systematically divide existing benchmarks into two broad categories: (i) reasoning-oriented benchmarks requiring multi-step or mathematical reasoning, and (ii) general-purpose perception benchmarks assessing visual understanding and perception abilities.

For reasoning-oriented evaluation, we test on four key areas: Math (MathVista [46], MathVerse [83], MathVision [69]), Geometry (GeoMath [27, 58], Geometry3K [47]), CLEVR+ (CLEVRMath [43], Super-CLEVR [41]), and Multi-Discipline (MMMU [78], MMMU-Pro [79]). For general perception, we evaluate across three categories: General (MuirBench [68], CRPE [35]), VisionCentric (MMVP [66], RealWorldQA [73], MMStar [13], MME [25], BLINK [26]), and OCR & Chart (AI2D [36], SEED-Bench-2-Plus [38], DocVQA [49], OCRBench [45]). More detailed descriptions of each benchmark are provided in Appendix B.5.

Math Geometry Model

Avg. Avg. MathVista MathVerse MathVision Avg. GeoMath Geo3K

Proprietary Model

GPT-4o [33] 47.5 47.3 61.4 50.2 30.4 46.8 50.2 43.5 Gemini-2.0-Flash [60] 55.4 56.4 73.4 54.6 41.3 54.4 55.3 53.5

Multimodal Reasoning Model Post-Trained on Qwen2.5-VL-7B [7]

|Base Model (Qwen2.5-VL-7B)|46.3|47.7 68.0 49.0 26.0|44.8 44.0 45.6<br><br>|
|---|---|---|---|
|R1-Onevision-7B [77] R1-VL-7B [12] MM-Eureka-Qwen-7B [50] Reason-RFT-Zero-7B [59] VLAA-Thinker-7B [11] OpenVLThinker-7B [19]<br><br>|40.9 40.9 39.3 46.5 51.3 52.1<br><br>|46.8 64.1 46.4 29.9 42.7 63.5 40.0 24.7 50.1 73.0 50.3 26.9 38.1 60.7 35.3 18.3 48.7 68.0 51.7 26.4<br><br>47.8 70.2 47.9 25.3<br>|35.0 45.4 24.5 39.0 42.0 36.1 28.4 53.1 3.8<br><br>54.9 55.0 54.8 53.9 51.1 56.6 56.4 49.2 63.5<br><br>|
|ViGaL Snake ViGaL Rotation ViGaL Snake + Rotation|51.6 52.8 53.9<br><br>|49.4 70.7 51.1 26.5<br><br>49.3 71.2 50.4 26.3<br><br>50.6 71.9 52.4 27.5<br>|55.0 49.9 60.0 57.9 51.7 64.1 57.1 51.0 63.3<br><br>|

±0.3 ±0.3 ±0.4 ±0.2 ±0.3 ±0.5 ±0.3 ±0.4

- Table 2 | Results on multimodal mathematical benchmarks. We compare to other multimodal reasoning models. Results post-trained on the same subject as the evaluation are de-emphasized, while our ViGaL models only use games for post-training. Bold numbers are the best in each Avg. column. We include standard deviations of three independent runs for ViGaL Snake + Rotation.

CLEVR+ Multi-Discipline Model

Avg. Avg. CLEVR-M S-CLEVR Avg. MMMUval MMMU-Prooverall

Proprietary Model

GPT-4o [33] 55.9 51.2 68.1 34.3 60.5 69.1 51.9 Gemini-2.0-Flash [60] – 46.3 64.9 27.6 – 71.9 –

Multimodal Reasoning Model Post-Trained on Qwen2.5-VL-7B [7] Base Model: Qwen2.5-VL-7B 50.3 54.9 74.6 35.2 45.7 54.3 37.0

|R1-Onevision-7B [77]|53.7|65.1 75.5 54.7|42.3 51.9 32.6|
|---|---|---|---|
|R1-VL-7B [12]|53.9|68.0 87.4 48.6|39.7 50.0 29.4|
|MM-Eureka-Qwen-7B [50]|62.8|79.3 98.4 60.1|46.4 55.8 36.9|
|Reason-RFT-Zero-7B [59]|58.6|76.2 99.4 53.0|40.9 51.2 30.6|
|VLAA-Thinker-7B [11]|61.7|83.4 94.7 72.1|40.1 48.2 31.9|
|OpenVLThinker-7B [19]|60.4|82.4 93.8 71.0|38.5 54.8 22.1|
|ViGaL Snake|64.4|82.6 92.6 72.6|46.2 55.8 36.6|
|ViGaL Rotation|63.3|80.7 93.0 68.3|45.9 54.1 37.7|
|ViGaL Snake + Rotation|64.7|81.7 91.9 71.4|47.7 58.0 37.4|

- Table 3 | Results on multimodal spatial and multi-discipline reasoning benchmarks. CLEVRM denotes CLEVR-Math [43], and S-CLEVR stands for Super-CLEVR [41]. Results post-trained on the same subject as the evaluation are de-emphasized, while ViGaL is exclusively posttrained using games. Bold numbers are the best in each Avg. column.

Obj.Count Abs.Dist.Obj.Size RoomSizeRel.Dist.Rel.Dir.RoutePlanAppr.Order

Model Avg.

Proprietary Model

GPT-4o [33] 34.0 46.2 5.3 43.8 38.2 37.0 41.3 31.5 28.5 Gemini-1.5-Flash [61] 42.1 49.8 30.8 53.5 54.4 37.7 41.0 31.5 37.8 Gemini-1.5-Pro [61] 45.4 56.2 30.9 64.1 43.6 51.3 46.3 36.0 34.6

Open-source Models

InternVL2-8B [15] 34.6 23.1 28.7 48.2 39.8 36.7 30.7 29.9 39.6 InternVL2-40B [15] 36.0 34.9 26.9 46.5 31.8 42.1 32.2 34.0 39.6 LongVILA-8B [14] 21.6 29.1 9.1 16.7 0.0 29.6 30.7 32.5 25.5 VILA-1.5-40B [42] 31.2 22.4 24.8 48.7 22.7 40.5 25.7 31.5 32.9 LongVA-7B [82] 29.2 38.0 16.6 38.9 22.2 33.1 43.3 25.4 15.7

Multimodal Reasoning Model Post-Trained on Qwen2.5-VL-7B [7] Base Model (Qwen2.5-VL-7B) 36.7 41.9 21.4 50.4 36.8 38.5 40.9 29.9 34.1

Video-R1 [24] 35.1 – – – – – – – – Visual Jigsaw [72] 38.5 – – – – – – – – ViGaL 38.7 42.9 22.6 46.7 45.8 38.3 41.7 35.1 36.1

- Table 4 | Results on VSI-Bench 3D spatial reasoning. We compare proprietary models, opensource multimodal models, and multimodal reasoning models post-trained on Qwen2.5-VL-7B. 3.1. Main Results

Zero-shot generalization from gameplay to multimodal reasoning. Our approach consistently shows remarkable generalization capabilities on mathematical and other reasoning tasks, despite having no direct exposure to in-domain training data during RL post-training. As shown in Tab. 2, our method notably outperforms models specifically RL-trained on mathematical tasks. For instance, ViGaL Snake + Rotation achieves 0.5% higher accuracy than MM-EurekaQwen-7B [50] 28.7% on Geometry, even though MM-Eureka-Qwen-7B was explicitly trained on high-quality mathematical and geometry datasets.

This strong generalization extends beyond mathematics. Tab. 3 shows that ViGaL Snake + Rotation outperforms R1-OneVision-7B [77] by 5.4% on average across MMMU series benchmarks, which test multi-disciplinary reasoning. This is particularly notable since R1-OneVision-7B was trained on a carefully curated comprehensive dataset spanning multiple subjects.

Furthermore, Tab. 4 demonstrates that post-training on two games using synthesized 2D images successfully generalizes to 3D spatial reasoning tasks. On VSI-Bench [76], ViGaL improves by 2.0% after game-based posttraining, slightly surpassing Visual Jigsaw [72]. This is notable because Visual Jigsaw is posttrained on 3D video data, whereas our approach uses only 2D game data.

These empirical results suggest that gameplay-based post-training develops fundamental reasoning capabilities that transfer more effectively than direct RL training on diverse taskspecific datasets. Moreover, the gameplay environment appears to encourage general problemsolving strategies that consistently generalize well to out-of-domain tasks.

Blending multiple games enhances generalization. As shown in Tab. 2, post-training on Snake achieves best performance on the CLEVR+ benchmark, while training on Rotation yields stronger results on geometry reasoning. Furthermore, training on both Snake and Rotation enables learning complementary skills, improving the overall benchmark average to 63.1%. These findings indicate that combining game environments drives meaningful performance gains, demonstrating Visual Gaming Learning’s potential as a promising paradigm for enhancing generalizable reasoning without large-scale domain-specific data. Expanding the

| |Applied<br><br>Area<br><br>+6.25 +6.25 +6.16<br><br>+3.38<br><br>+1.96<br><br>+1.55<br><br>ViGAL-Snake > Baseline ViGAL-Snake < Baseline<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |
| |CoordinateExpression LengthProperty Volume Angle<br><br>-0.18<br><br>-0.66|
| | |
| | |

10

AccuracyDifferences(%)

8

6

4

- 2

0

2

4

(a) Snake vs. Baseline

| |PropertyExpression<br><br>+8.75<br><br>+4.62<br><br>+2.35<br><br>+0.74 +0.62<br><br>+0.00<br><br>ViGAL-Rotation > Baseline ViGAL-Rotation < Baseline<br><br>|
|---|---|
| | |
| | |
| | |
| | |
| | |
| |Angle LengthVolumeAppliedCoordinate Area<br><br>-0.84<br><br>-3.75|
| | |
| | |

10

AccuracyDifferences(%)

8

6

ion

4

2

0

2

4

(b) Rotation vs. Baseline

- Figure 3 | Per-category gains on MathVerse are not uniform. The eight math categories follow MathVerse [83]. (a) Snake yields the largest gains on Coordinates and Expressions, consistent with its 2D grid structure. (b) Rotation boosts Angle and Length questions but reduces Expression accuracy, suggesting its training primarily incentivizes orientation recognition.

<think> …Current Position:… - Snake 1‘s head is now at (0, 8). Body positions are (0, 7), (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (2, 2). Possible Moves: - UP (0, 9): Move to… - RIGHT (1, 8): Moves towards … - LEFT: Not possible, wall is present. Strategy and Objective: If move RIGHT to (1, 8), distance to apple becomes |3 − 1| + |1 − 8| = 2 + 7 = 9… Decision: - Move RIGHT to (1, 8) to maintain central board position and access more strategic options. </think> …

[Figure 18]

[Figure 19]

To determine the inequality that describes the region in the number plane,….1. Identify the boundary line: The graph shows a straight line passing through the points (0, -10) and (5, 10). We can find the equation of this line using the slope-intercept form (y = mx + b). Calculate the slope (m): m = frac{10 - (-10)}{5 - 0} = frac{20}{5} = 4 - Use the point (0, -10) to find the y-intercept (b): -10 = 4(0) + b implies b = -10 ……Therefore, the inequality that describes this region is: y leq 4x - 10 The final value is (boxed{y leq 4x - 10}).

[Figure 20]

[Figure 21]

<think> The panda statue in Image 3 is shown from the front, and in Image 4, it is rotated such that its back is facing forward, which would typically indicate a 180-degree rotation. The orientation change suggests the panda has flipped around the vertical axis, while also facing in the opposite direction. The symmetry and the way the legs appear supports this as a 180-degree rotation rather than a 90-degree rotation. </think> <answer>180</answer>

To solve the problem, we need to analyze the given information and use geometric properties to find the value of angle DAO + angle DCO. …Consider the symmetry and properties of the angles: - Since angle ABC = angle ADC = 65°, we can infer that the quadrilateral might be a kite or a specific type of quadrilateral where the angles are symmetric… The sum of the angles around point O (which is the intersection of the diagonals) is 360°…By analyzing the symmetry and the properties of the quadrilateral, we can deduce that the sum of angles DAO and DCO is 110°.Therefore, the correct answer is \\boxed{B}.

Snake

Rotation Angle

Expression

[Figure 22]

- Figure 4 | Reasoning trace of different games and math questions. Top: Algebraic functions and coordinate-level interpretations that emerge from playing the Snake game help solving Expression questions. Bottom: Spatial reasoning skills incentivized by playing the Rotation game appear when solving Angle-related problems.

types of games during training consistently scales performance across different visual reasoning tasks.

Different games benefit distinct math subfields. To study which types of problems in the math benchmarks benefit from game play, we analyze accuracy differences across MathVerse [83] subcategories between ViGaL models trained with Snake or Rotation, as shown in Fig. 3. We find that training on the Snake game significantly improves performance on the subcategories like Expressions and Coordinates, while training on Rotation notably enhances performance on questions about angles and lengths. To understand why different games help with different types of math, we compare the reasoning processes required for playing games versus solving math problems. As shown in Fig. 4, solving Expressions questions involves algebraic functions and coordinate-level interpretations of graphical representations, which closely align with the spatial reasoning process in Snake. Similarly, solving angle-related questions is consistent with requirement of playing Rotation game to reason about rotational angles of 3D objects. These results suggest that playing different games develops fundamental skills like spatial modeling and algebraic calculation that transfer to visual math questions. The experiment on quantitatively analyzing the correlation between math and game is in Sec. B.7 in the Appendix. Furthermore, joint training on both games leads to improvements across all reasoning categories (see Appendix Sec. B.3). We also include qualitative analyses on improvements in math reasoning after RL in Appendix Sec. C.

###### Model Avg. MathVista MathVerse MathVision

Base Model: Qwen2.5-VL-7B 47.7 68.0 49.0 26.0 MM-Eureka-Qwen-7B 50.1 73.0 50.3 26.9 ViGaL Snake + Rotation 50.6 71.9 52.4 27.5 ViGaL Snake + Rotation + Math Data 51.8 72.3 54.5 27.7

- Table 5 | Gameplay complements math data. Adding math data MMK12 on top of ViGaL yields further gains in math performance. With access to the same amount of math data, ViGaL outperforms MM-Eureka [50] on average of the three math benchmarks.

General Vision-Centric OCR & Chart Model

Avg. Avg.

MuirBench

CRPErel. Avg. MMVP

RealWorldQA

MMStar BLINKval MMEp Avg.

AI2D

w.M.

SEEDBench-2+

DocVQA val

OCRBench

Proprietary Model GPT-4o [33] 74.8 72.3 68.0 76.6 69.4 – 75.4 64.7 68.0 1614 82.6 84.6 72.0 91.1 736 General Multimodal Language Model Qwen2.5-VL-7B [7] 72.4 68.0 59.6 76.4 65.8 74.3 68.5 63.9 56.4 1698 83.3 83.9 70.4 95.7 864 Multimodal Reasoning Model Post-Trained on Qwen2.5-VL-7B

|R1-Onevision-7B [77] R1-VL-7B [12] MM-Eureka-Qwen-7B [50] Reason-RFT-Zero-7B [59] VLAA-Thinker-7B [11] OpenVLThinker-7B [19]|– 67.4 71.8 68.4 69.7 –<br><br>|66.8 46.3 87.3<br><br>63.3 54.1 72.4 68.9 61.1 76.7<br><br>66.9 58.5 75.2 65.9 57.1 74.6<br><br>64.3 52.8 75.8<br><br><br>|56.5 61.3 58.0 57.8 48.7 59.6 70.3 61.4 55.6 51.0 65.1 74.3 66.1 65.9 54.0 58.5 58.0 65.3 59.1 51.6 62.6 71.6 65.4 60.4 53.0 50.4 32.3 60.2 59.1 49.9<br><br>|1504 1657 1626 1653 1593 1513<br><br>|– – – – – 79.2 81.7 66.4 89.4 81.0 81.5 84.3 68.2 92.0 87.0<br><br>79.8 83.3 68.0 88.1 82.0<br><br>80.6 83.4 67.4 90.9 84.5<br><br><br>– – – – –<br>|
|---|---|---|---|---|---|
|ViGaL Snake + Rotation|72.2|68.6 60.5 76.7|65.7 74.6 67.3 65.4 55.6|1685|82.2 84.8 69.1 92.7 86.6|

- Table 6 | Main results on multimodal language benchmarks targeting more general and comprehensive visual ability. We compare with models post-trained on Qwen2.5-VL-7B [7].

Best category averages are highlighted in bold. Note that MMEp is excluded from vision-centric category average accuracy due to scale differences.

Gameplay complements math data. We explore the complementary benefits of adding math data to the gameplay training pipeline, for which we implement a two-stage training process. Stage 1 equals to ViGaL setup, training the model on Snake and Rotation games. In stage 2, we further finetune the stage 1 model on MMK12 [50], a multimodal mathematical reasoning dataset containing approximately 12k examples. Stage 2 training uses the identical data and settings as MM-Eureka-Qwen-7B [50]. As shown in Tab. 5, the integration of mathematical data in stage 2 yields a continuous improvement of 1.2% on average across three mathematical benchmarks. This demonstrates the complementary relationship between our visual game learning approach and mathematical data post-training. Moreover, ViGaL with math data significantly outperforms MM-Eureka-Qwen-7B by 1.7% on mathematical benchmarks on average, using the same math data. These results suggest that visual game learning can serve as an effective surrogate task together with domain-specific data to improve performance on target tasks.

Preserving general visual capabilities while reasoning enhancement. To examine whether generalization on reasoning tasks leads to degradation in general visual capabilities, we evaluate ViGaL Snake + Rotation on a broader set of MLLM benchmarks. As shown in Tab. 6, compared to Qwen2.5-VL-7B prior to RL tuning, our model maintains comparable general visual performance while achieving stronger math reasoning results. In contrast, other models that improve math performance through RL post-training often exhibit substantial drops in general visual capabilities. These results demonstrate that our gameplay-based approach enables math generalization without compromising other visual abilities.

(a) Text prompt design.

prompt Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 w/o reasoning instr. 59.5 48.0 80.4 50.1 w/ reasoning instr. 62.3 49.4 82.6 55.0

(b) Reward design.

reward Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 best moves 59.6 48.2 80.4 50.2 best & worst moves 62.3 49.4 82.6 55.0

w/ random label 49.4 47.5 55.4 47.5

(c) Difficulty control.

difficulty control Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 w/o difficulty control 60.6 48.8 81.4 51.8 w/ difficulty control 62.3 49.4 82.6 55.0

(d) Data scalability.

(e) Input modality.

(f) SFT vs. RL.

training samples Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 16K 60.1 48.9 81.2 50.3 36K 62.3 49.4 82.6 55.0

input modality Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 text 59.6 48.5 80.1 50.3 vision & text 62.3 49.4 82.6 55.0

post-training Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 SFT 47.2 38.0 71.5 32.1 RL 62.3 49.4 82.6 55.0

- Table 7 | Ablation study. We ablate different aspects of ViGaL with Snake and evaluate on downstream benchmarks. The similar evaluation with Rotation is in Sec. B.2. Each benchmark consists of several subtasks (Tab. 2 and Tab. 3), and we report their averages. The base model is Qwen2.5-VL-7B, whose results are in gray. The default settings in Tab. 2 and Tab. 3 are highlighted in blue .

##### 3.2. Ablation Study

We ablate key design choices in the Snake environment, evaluate each variant on downstream benchmarks, and report the results in Tab. 7. The corresponding ablation for the Rotation environment is provided in Appendix Sec. B.2.

Reasoning instructions in the text prompt help. We use reasoning instructions, such as “finding the nearest apple by calculating Manhattan distances”, in the text prompts to guide the model thinking chains. The complete text prompts are in Appendix Sec. A.2. In Tab. 7a, we demonstrate that reasoning instructions brings significant improvement of 1.9%, from 59.5% to 61.4%, for Snake in average accuracy over the three out-of-domain benchmarks.

Reward design of pre-text game matters for downstream tasks. We show that reward design of RL for games plays a crucial role for the downstream tasks. As shown in Tab. 7b, we first ask the model to predict only the best next move, defined as the action that moves toward the closest apple while avoiding death. In our improved reward design, we task the model with simultaneously predicting both the best and worst next moves, where the worst move leads directly to losing the game. More importantly, it leads to improvements across all downstream tasks, bringing an average increase of 1.8%. These results suggest that proper reward design in pre-text game can improve not only gameplay capabilities but also generalization to downstream tasks.

Furthermore, inspired by several prior works that improve model performance without labeled rewards [84] or with random labels [57], we also provide a random reward ablation, where we still ask the model to predict both best and worst moves but use random moves as the labels. We report the results in the last row in Tab. 7b. In our gameplay setting, RL with random labels reports 49.4% on averagne and does no provide significant gains over the base model, different from the conclusions in prior works [57]. Potential explanations lie in the difference in data domains and base models, where other works applied random labels to text-only mathematical data while our work applies random labels to visual game data.

Controlling game difficulty for better reasoning. Gameplay for RL post-training offers unique opportunities to easily control task difficulty. We present an ablation study on difficulty control importance. We define difficulty based on snake length, where longer snakes represent higher difficulty. For controlled difficulty, we collect training data using states where snake length falls within a moderate range of 1-5. Details are in Sec. A.1. As shown in Tab. 7c, difficulty control achieves 61.4% overall accuracy compared to 60.6% without control. This suggests our game engine can easily generate appropriately difficult data, helping prevent model sub-optimization during RL training.

RL on games shows data scalability. Thanks to using game engine, we can generate data at any scale with high flexibility. To show data scalability on RL of visual games, we conduct experiments using 16k and 32k snake game samples, respectively. As in Tab. 7d, scaling data from 16k to 32k brings a performance improvement of 1.3% on average across all domains. This suggests the potential of the proposed ViGaL paradigm to improve downstream performance by easily scaling training data, which contrasts with the data scaling challenges of domain-specific human annotated data, requiring extensive manual effort.

Both text and vision contribute to better visual reasoning. To isolate the contributions of text and vision modalities, we conduct an ablation study with a text-only setting. In this setup, we represent game states—including snake positions, apple locations, and boundary constraints—using only textual descriptions during RL training. The model trained with textonly inputs on the Snake game demonstrates substantial improvements across all multimodal benchmarks, with average performance increasing from 49.1% to 59.6%. Incorporating visual inputs yields an additional 1.8% performance gain. These results demonstrate that multimodal RL enhances visual reasoning capabilities, with complementary contributions from both text and vision modalities.

RL generalizes better than SFT from games to math. To evaluate the out-of-domain generalization of ViGaL, we compare it with supervised fine-tuning (SFT) using identical visual game data. Tab. 7f shows that SFT with Snake game data degrades the base model’s performance on both mathematical reasoning and geometry tasks by a notable 9.7% and 12.7%, respectively. While SFT produces modest improvements on CLEVR+, these gains are substantially smaller than those achieved by RL. Overall, RL improves performance by 12.3%, whereas SFT decreases performance by 1.9%. This stark contrast demonstrates that RL better preserves and extends the model’s reasoning capabilities to new domains.

#### 4. Related Work

Reinforcement Learning in MLLMs. Reinforcement Learning (RL) increasingly enhances reasoning in Large Language Models (LLMs) beyond Supervised Fine-Tuning (SFT). Text-only models like DeepSeek-R1 [18] demonstrate RL’s efficacy, especially with rule-based rewards, for complex reasoning. This paradigm is now being extended to Multimodal LLMs (MLLMs). Recent MLLM research explores RL for improved visual reasoning, drawing from LLM successes. Various works [12, 32, 53] investigate multi-stage training, trace supervision, or rule-based RL for specific visual subdomains like geometry and counting. Others focus on different RL algorithms like Process Reward Models (PRMs) [48, 74], often moving beyond SFT-based Chainof-Thought generation [21, 63]. Many efforts favor simpler rule-based rewards [32, 85] over complex reward models prone to hacking [23]. Unlike approaches training on costly, domainspecific reasoning datasets, our ViGaL paradigm extends rule-based RL to simple, synthetic visual games, demonstrating these serve as scalable, cost-effective pre-text tasks.

Generalization in MLLMs. Achieving robust generalization to novel tasks, distributions, and domains is central to MLLM development. RL shows promise for better out-of-distribution (OOD) generalization compared to SFT [12, 50], and developing multi-step reasoning like CoT [71] is itself generalization. This is often pursued through training on large, diverse instruction-following datasets [15, 39, 44] or explicitly training general reasoning capabilities [32, 77]. While these methods advance OOD generalization, they typically operate within the same broad domain of complex visual reasoning as training data.The work most closely related to ours is Game-RL [64], which demonstrates how constructing diverse game-based tasks for RL can enhance the general reasoning abilities of vision-language models (VLMs). Our ViGaL paradigm investigates stronger out-of-domain generalization, showing fundamental skills

learned from simple synthetic games transfer zero-shot to enhance performance on entirely different, complex domains like visual mathematics, multi-discipline questions, and spatial reasoning, without domain-specific data exposure.

#### 5. Conclusion

We introduced Visual Game Learning (ViGaL), a novel post-training paradigm where MLLMs learn transferable reasoning by playing simple arcade-style games. Our core finding is that RL on games like Snake and Rotation, without any in-domain math data, significantly boosts MLLM performance on mathematical and multi-discipline benchmarks, surpassing specialized models and even large proprietary systems. Ablations confirm the importance of game design, reward structure, and that RL outperforms SFT, while distinct games unlock different skills. We posit that games instill fundamental reasoning skills, suggesting a new avenue for using scalable, controllable synthetic games as powerful pre-text tasks to unlock generalizable reasoning. This work opens doors to exploring a broader range of game-based learning for generalizable AI.

#### References

- [1] Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.
- [2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In ACL, 2024.
- [3] Fahad Alhasoun and Sarah Alneghiemish. Probabilistic programming bots in intuitive physics game play. In AAAI, 2021.
- [4] Kelsey Allen, Franziska Brändle, Matthew Botvinick, Judith E. Fan, Samuel J. Gershman, Alison Gopnik, et al. Using games to understand the mind. Nature Human Behaviour, 2024.
- [5] Kelsey R Allen, Kevin A Smith, and Joshua B Tenenbaum. Rapid trial-and-error learning with simulation supports flexible tool use and physical reasoning. PNAS, 2020.
- [6] Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, and Donghyun Kwak. Online difficulty filtering for reasoning-oriented reinforcement learning. arXiv:2504.03380, 2025.
- [7] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [8] Bowen Baker, Ingmar Kanitscheider, Todor Markov, Yi Wu, Glenn Powell, Bob McGrew, and Igor Mordatch. Emergent tool use from multi-agent autocurricula. In ICLR, 2019.
- [9] Lara Bertram. Digital learning games for mathematics and computer science education: The need for preregistered rcts, standardized methodology, and advanced technology. Frontiers in Psychology, 2020.
- [10] Franziska Brändle, Kelsey R Allen, Josh Tenenbaum, and Eric Schulz. Using games to understand intelligence. In CogSci, 2021.

- [11] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, et al. SFT or RL? an early investigation into training R1-Like reasoning large vision-language models. arXiv:2504.11468, 2025.
- [12] Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-V: Reinforcing super generalization ability in vision-language models with less than $3. https://github.com/Dee p-Agent/R1-V, 2025.
- [13] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv:2403.20330, 2024.
- [14] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. In The Thirteenth International Conference on Learning Representations.
- [15] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, et al. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024.
- [16] Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, et al. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. arXiv:2501.17161, 2025.
- [17] Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, and Yong Wang. GPG: A simple and strong reinforcement learning baseline for model reasoning. arXiv:2504.02546, 2025.
- [18] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv:2501.12948, 2025.
- [19] Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. OpenVLThinker: An early exploration to complex vision-language reasoning via iterative self-improvement. arXiv:2503.17352, 2025.
- [20] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.
- [21] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-V: Exploring long-chain visual reasoning with multimodal large language models. arXiv:2411.14432, 2024.
- [22] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multi-agent debate. In ICML, 2023.
- [23] Jacob Eisenstein, Chirag Nagpal, Alekh Agarwal, Ahmad Beirami, Alex D’Amour, DJ Dvijotham, et al. Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. arXiv:2312.09244, 2023.
- [24] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [25] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv:2306.13394, 2023.

- [26] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.
- [27] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, et al. G-LLaVA: Solving geometric problem with multi-modal large language model. arXiv:2312.11370, 2023.
- [28] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. arXiv:2210.10760, 2022.
- [29] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv:1803.07728, 2018.
- [30] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.
- [31] Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. OpenRLHF: An easy-to-use, scalable and high-performance RLHF framework. arXiv:2405.11143,

- 2024.

[32] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, et al. Vision-R1: Incentivizing reasoning capability in multimodal large language models. arXiv:2503.06749,

- 2025.

- [33] Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, et al. GPT-4o system card. arXiv:2410.21276, 2024.
- [34] Greg Kamradt. Snake Bench: Competitive snake game simulation with LLMs. https: //github.com/gkamradt/SnakeBench, 2025.
- [35] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In EMNLP, 2014.
- [36] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, 2016.
- [37] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free! In ICLR Workshop on Deep Reinforcement Learning Meets Structured Prediction, 2019.
- [38] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024.
- [39] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, et al. LLaVA-NeXT-Interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv:2407.07895, 2024.
- [40] Jian Li, Weiheng Lu, Hao Fei, Meng Luo, Ming Dai, Min Xia, Yizhang Jin, Zhenye Gan, Ding Qi, Chaoyou Fu, et al. A survey on benchmarks of multimodal large language models. arXiv preprint arXiv:2408.08632, 2024.
- [41] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L. Yuille. Super-CLEVR: A virtual benchmark to diagnose domain robustness in visual reasoning. In CVPR, 2023.

- [42] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: on pre-training for visual language models. In CVPR, 2024.
- [43] Adam Dahlgren Lindström and Savitha Sam Abraham. Clevr-Math: A dataset for compositional language, visual and mathematical reasoning. In IJCLR, 2022.
- [44] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge. https://llav a-vl.github.io/blog/2024-01-30-llava-next/, 2024.
- [45] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. arXiv:2305.07895, 2023.
- [46] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, et al. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv:2310.02255, 2024.
- [47] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and SongChun Zhu. Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning. arXiv:2105.04165, 2021.
- [48] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Yiyao Yu, Xinzhe Ni, Zicheng Lin, et al. URSA: Understanding and verifying chain-of-thought reasoning in multimodal mathematics. arXiv:2501.04686, 2025.
- [49] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In WACV, 2021.
- [50] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, et al. MM-Eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv:2503.07365, 2025.
- [51] OpenAI. Introducing OpenAI o1. https://openai.com/o1/, 2024.
- [52] OpenAI. Gpt-5 system card. openai.com/index/gpt-5-system-card, 2025.
- [53] YingZhe Peng, Gongrui Zhang, Xin Geng, and Xu Yang. LMM-R1. https://github.c om/TideDra/lmm-r1, 2025.
- [54] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. OpenAI, 2018.
- [55] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv:2205.06175, 2022.
- [56] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.
- [57] Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. Spurious rewards: Rethinking training signals in rlvr. https://rethink-rlvr.notion.site/Spurious-Rewards-Rethi nking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f, 2025. Notion Blog.

- [58] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-LLaVA: Bootstrapping mathematical reasoning for multimodal large language models. arXiv:2406.17294, 2024.
- [59] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Minglan Lin, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-RFT: Reinforcement fine-tuning for visual reasoning. arXiv:2503.20752, 2025.
- [60] Gemini Team. Gemini: A family of highly capable multimodal models. arXiv:2312.11805, 2023.
- [61] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530, 2024.
- [62] Tencent Hunyuan3D Team. Hunyuan3D 2.0: Scaling diffusion models for high-resolution textured 3d assets generation. arXiv:2501.12202, 2025.
- [63] Omkar Thawakar, Dinura Dissanayake, Ketan More, Ritesh Thawkar, Ahmed Heakl, Noor Ahsan, et al. LLaMAV-o1: Rethinking step-by-step visual reasoning in LLMs. arXiv:2501.06186, 2025.
- [64] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, et al. Code2logic: Game-code-driven data synthesis for enhancing vlms general reasoning. arXiv preprint arXiv:2505.13886, 2025.
- [65] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.
- [66] Shengbang Tong, Zhuang Liu, Yuexiang Zhu, Xingjian Chen, Ruoyu Zhang, Bo Li, et al. Eyes wide shut? exploring the visual shortcomings of multimodal llms. arXiv preprint arXiv:2401.06209, 2024.
- [67] Bas Van Opheusden, Ionatan Kuperwajs, Gianni Galbiati, Zahy Bnaya, Yunqi Li, and Wei Ji Ma. Expertise increases planning depth in human gameplay. Nature, 2023.
- [68] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024.
- [69] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with Math-Vision dataset. In NeurIPS, 2024.
- [70] Nicholas R. Waytowich, Devin White, M.D. Sunbeam, and Vinicius G. Goecks. Atari-GPT: Investigating the capabilities of multimodal large language models as low-level policies for atari games. arXiv:2408.15950, 2024.
- [71] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.
- [72] Penghao Wu, Yushan Zhang, Haiwen Diao, Bo Li, Lewei Lu, and Ziwei Liu. Visual jigsaw post-training improves mllms. arXiv preprint arXiv:2509.25190, 2025.

- [73] X.AI. Grok-1.5 vision preview. https://x.ai/blog/grok-1.5v, 2024.
- [74] Kun Xiang, Zhili Liu, Zihao Jiang, Yunshuang Nie, Runhui Huang, Haoxiang Fan, et al. AtomThink: A slow thinking framework for multimodal mathematical reasoning. arXiv:2411.11930, 2024.
- [75] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.
- [76] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025.
- [77] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, et al. R1-OneVision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv:2503.10615, 2025.
- [78] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [79] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.
- [80] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? arXiv:2504.13837, 2025.
- [81] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 881–916, 2025.
- [82] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.
- [83] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, et al. MathVerse: Does your multi-modal LLM truly see the diagrams in visual math problems? arXiv:2403.14624, 2024.
- [84] Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.
- [85] Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1-Zero’s “aha moment” in visual reasoning on a 2b non-sft model. arXiv:2503.05132, 2025.

#### Appendix

##### Content

- A. Data. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .19

- A.1. Training Data Synthesis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .19
- A.2. Training Prompt in Visual Game Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- A.3. Detail of Format Reward . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- B. Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23

- B.1. Evaluation Detail of Atari Game . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- B.2. Ablation On Rotation Game . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .24
- B.3. Synergistic Effects of Multi-Game Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- B.4. Reasoning Ability Boundary via Pass@𝑘 Evaluation. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .26
- B.5. Detail of Evaluation Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .27
- B.6. Inference Length Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- B.7. Reasoning Correlation Analysis Between Game and Math . . . . . . . . . . . . . . . . . . . . . . . . . 28

- C. Case Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .30

#### A. Data

##### A.1. Training Data Synthesis

Thanks to using the synthetic game data engine, we can flexibly generate large-scale training data with precisely controlled difficulty levels. This completely eliminates the need for extensive data filtering strategies used in previous rule-based RL work training on domain-specific data like math [6, 50], where difficulty is hard to define and filtering can significantly reduce dataset size.

For the Snake game, the environment consists of a 10 × 10 grid game board with two snakes of 1-grid initial length, where at each time step 𝑡, each snake receives one action to move, resulting in a new game state 𝑠𝑡+1. We define difficulty based on snake length—longer snakes create more complex game situations and more constrained movement options, closely aligning with how humans perceive difficulty when playing Snake. To generate meaningful moves that accomplish the objective of collecting more apples while remaining alive, we implement a policy network based on Proximal Policy Optimization (PPO) [56]. The observation space is represented as a 10 × 10 grid with distinct values indicating empty cells (0), apples (1), the agent’s own body (2), and other agents’ bodies (3), stacked across 4 time steps to incorporate temporal information, resulting in an input tensor X ∈ R10×10×4. The policy network architecture consists of two convolutional layers with 3 × 3 kernels (𝐶1 = 16 and 𝐶2 = 32 output channels), followed by fully connected layers that output action logits for the four possible movements, transformed into a probability distribution 𝝅(𝑎|𝑠) using softmax. To prevent suicidal moves, we incorporate action priors by masking logits for dangerous actions. The model employs the standard PPO objective with entropy regularization coefficient 𝛽 = 0.01, value function coefficient 𝜆 = 0.5, and clipping parameter 𝜀 = 0.2. Agents receive rewards of 𝑟 = +1 for collecting apples and penalties of 𝑟 = −1 for dying, enabling them to learn complex behaviors such as obstacle avoidance, apple pursuit, and multi-step trajectory planning.

For the Rotation game, training data comprises synthetically generated visual puzzles focused on 3D spatial reasoning, utilizing 540 unique 3D object meshes (408 from Hunyuan3D 2.0 [62] and 132 from Hunyuan3D 2.5). Our custom pipeline produces pairs of images (𝐼init, 𝐼rot) representing objects before and after defined rotations. Difficulty in Rotation is determined by the rotation angle between two images, where smaller angle differences present greater perceptual challenges. Each pair is generated through a precise sequence: establishing diverse initial viewpoints through compound transformations (base orientation plus additional zaxis rotation from {0◦,30◦, . . . ,330◦} to prevent trivial pattern learning), then applying target rotations of either 90◦ or 180◦ exclusively around the z-axis. All objects are rendered at 512 × 512 pixel resolution using a consistent perspective camera under standardized lighting conditions, resulting in approximately 32k unique pairs.

Based on empirical results, we established optimal difficulty parameters for RL training across both games, which we ablate in Tab. 7c. This controlled progression of difficulty, made possible by our synthetic data generation approach, enables more effective learning trajectories compared to traditional data collection methods.

##### A.2. Training Prompt in Visual Game Learning

Prompt for Snake Game Your role is to guide a snake within a Snake game featuring multiple apples. This game is played on a board of size 10 by 10. The board uses a standard Cartesian coordinate system, where (0,0) represents the bottom-left position and (9,9) is the top-rightmost coordinate. Apples at: {apple_position} Direction of Your Last Action: {last_action} Rules:

- 1) If you move onto an apple, you grow and gain 1 point.
- 2) If your head moves to a position where its coordinates (x, y) are outside the board boundaries (meaning x < 0, x > 9, y < 0, or y > 9), or into a space occupied by another snake’s body, or into a space occupied by your own body, you die. That’s the worst move.
- 3) The goal is to prioritize snake not die, then efficiently collecting apples. First avoid the worst move, then for each apple, find the nearest apple by calculating Manhattan distances. But only choose best next move to get closer the nearest apple if you can confirm best next move will not run outside the range of the listed coordinates, run into the position of another snake, or yourself. Otherwise it will be the worst move.

Your snake with the ID {snake_id} in {snake_color} has its head now positioned at {snake_position}, and its body extends to {body_position} You should avoid your next move into your own snake’s position.

Enemy snakes in {enemy_color} positions: {enemy_position}.

- Decreasing your x coordinate is to the LEFT, increasing your x coordinate is to the RIGHT.
- Decreasing your y coordinate is DOWN, increasing your y coordinate is UP.

Read out another snake’s position and apple position. Try to predict another snake’s next move and avoid colliding with it.

Best answer is one of next move that is the closest to the apple and not lead to your death. Worst answer is all of next moves 1. makes your head’s coordinates (x, y) are outside the board boundaries, meaning x < 0, x > 9, y < 0, or y > 9. 2. moves into a position occupied by another snake’s body. 3. moves into a position occupied by body of yourself.

Check all the next moves to list out all the worst moves in <worst_answer> tag. If no worst answer, return None for worst answer, e.g., "<worst_answer>None</worst_answer>"

The best answer and the worst answer are mutually exclusive and different. You need first to give your reasoning process then to choose one of best next move and worst next move from [’UP’, ’DOWN’, ’LEFT’, ’RIGHT’].

The reasoning process and answer are enclosed within <think> </think>, <best_answer> </best_answer> and <worst_answer> </worst_answer> tags, respectively, i.e., "<think> reasoning process here </think><best_answer> one best move here </best_answer><worst_answer> all worst moves here </worst_answer>"

Prompt for Rotation Game

I’m showing you 4 images. Images 1-2 are an example pair, and Images 3-4 are the test pair. In each pair, the first image shows the initial orientation, and the second shows the object after rotation.

### EXAMPLE OF ROTATION ### Example: Image 1 shows the initial view and Image 2 shows the object after a 180 degree rotation. ### YOUR TASK ### Now, considering the transformation from Image 3 (initial) to Image 4 (rotated)

. Determine the angle of rotation from Image 3 to Image 4 on the plane Analyze the rotation carefully using the example pair (Images 1-2) as a reference.

- 1. Coordinate System Transformation:

- - Draw an x-y coordinate system on both original and rotated images with origin at center
- - Identify a distinct feature point and note its coordinates in both images
- - Apply rotation matrix equations to verify the transformation

Example: A star icon at coordinates (3,1) in the original image appears at (-1,3) in the rotated image. Testing with the 90° clockwise rotation matrix [cos(90°), sin(90°); -sin(90°), cos(90°)] confirms the transformation from (3,1) to (-1,3), verifying a 90° clockwise rotation.

- 2. Angular Displacement Measurement:

- - Mark the image center as the origin in both images
- - Draw a straight line from center to a distinctive feature in both images
- - Measure the angle between these two lines using counterclockwise as positive

Example: A line from center to a red dot makes a 30° angle with horizontal in the original image. In the rotated image, this line makes a 210° angle with horizontal. The difference (180°) indicates a clockwise 180° rotation.

- 3. Symmetry Axis Tracking:

- - Identify major symmetry axes in the original image
- - Locate the same symmetry axes in the rotated image
- - Calculate the angular displacement between original and rotated axes

Example: A rectangular logo has vertical and horizontal symmetry axes. After rotation, the vertical axis now points right and horizontal points down. This 90° shift of both axes confirms a clockwise 90° rotation.

- 4. Triangle Configuration Analysis:

- - Select three non-collinear distinct points forming a triangle in both images
- - Compare the orientation of this triangle in both images using vector cross products
- - Determine rotation angle from the triangle’s orientation change

Example: Three points form a right triangle with vertices clockwise arranged. After rotation, the same triangle has its vertices arranged in counterclockwise order while maintaining the same shape. This inversion indicates a clockwise 180° rotation.

- 5. Polar Coordinate Comparison:

- - Convert key points to polar coordinates (r,𝜃) relative to image center
- - Compare 𝜃 values of the same features in original and rotated images
- - Calculate consistent angular difference across multiple points

Example: A feature at polar angle 45° in the original image appears at 135° in the rotated image. Another feature shifts from 10° to 100°. Both show a +90° shift in polar angle, confirming a clockwise 90° rotation.

Choose the rotation angle from this list: [’counter clockwise 90’, ’180’]

21

The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think><answer> answer here </answer>"

While the model takes images as input to understand the current state of the game, we design a structural text prompt framework to also provide game guidance. Our game prompts consist of two parts: (1) game settings and (2) reasoning instructions. (1) To help the model understand the game environment, we describe the background, current game state, rules, goals, action space, etc. in text besides the input image. (2) In the reasoning instruction part, we provide specific thinking guidance since games can be approached with various thinking chains. To encourage broader thinking, we implement different types of reasoning instructions to guide decision-making process. Specifically, we used GPT-4o [33] to synthesize mathematical thinking instructions for Snake, such as “finding the nearest apple by calculating Manhattan distances”, and spatial thinking instructions for Rotation, for example, “identify major symmetry axes in the original image”. With reasoning instructions for games, the obtained reasoning abilities generalize to downstream evaluation on visual math questions (Tab. 7a). Bold text indicates reasoning instructions synthesized by GPT-4o [33].

- A.3. Detail of Format Reward The format reward 𝑟format validates whether the response follows the task-specific format:

𝑟format =

0.1, if the response follows the required format 0, otherwise

(1)

For Snake game, the desired format is: <think>...</think><best_answer>...</best_answer><worst_answer>...</worst_answer>.

As suggested by the format, we encourage the model to predict both a positive move that moves toward the apple and a negative move that leads to failure. This reward encourages contrastive decision-making, which not only improves the model’s gameplay abilities but also boosts downstream reasoning performance on visual math benchmarks. We ablate the effect in Tab. 7b. For the rotation task, the required format is simply <think>...</think><answer>...</answer>.

|Alien|
|---|

###### Ms. Pacman

###### Frogger

Breakout

[Figure 23]

Goal: Navigate through a maze,

Goal: Navigate through maze-like environments while shooting alien enemies and collecting items.

eating all dots while avoiding ghosts

Goal: Use a paddle to bounce a ball

Goal: Guide frogs across a busy road

- B. Evaluation B.1. Evaluation Detail of Atari Game

or eating them when powered up. Example response:

to break bricks at the top of the screen. Example response: Reasoning: The ball is moving right. I

and river to reach their homes safely.

Example response: Reasoning: There's a car coming from

Example response:

Reasoning: Ms. Pacman is slightly

Reasoning: The alien is right above,

below the ghost now, but moving up

need to move right further to

the left. Moving up will help to avoid

continue firing to try and take it

should still allow her to catch it and gain points. Action: 1

intercept it.

it.

d

|down<br><br>Action:1 Alien|
|---|

###### Ms. Pacman

Action: 2

Action: 1

###### Frogger

###### Breakout

A

[Figure 24]

[Figure 25]

[Figure 26]

|Pong|
|---|

###### Frogger

###### Alien

[Figure 27]

[Figure 28]

Goal: Navigate through

Goal: Guide frogs across a busy road and river to

maze-like environments while shooting alien

reach their homes safely.

enemies and collecting

items.

Goal: Navigate through a maze,

Goal: Navigate through maze-like environments while shooting alien enemies and collecting items.

eating all dots while avoiding ghosts

Goal: Use a paddle to bounce a ball

Goal: Guide frogs across a busy road

###### Example response:

or eating them when powered up. Example response:

to break bricks at the top of the screen. Example response: Reasoning: The ball is moving right. I

and river to reach their homes safely.

###### Example response:

Reasoning: There's a car

Reasoning: The alien is

coming from the left. Moving up will help to

Example response: Reasoning: There's a car coming from

Example response:

Reasoning: Ms. Pacman is slightly

right above, continue firing to try and take it

Reasoning: The alien is right above,

Goal: Navigate through maze-like

below the ghost now, but moving up

need to move right further to

the left. Moving up will help to avoid

continue firing to try and take it

environments while shooting alien

should still allow her to catch it and gain points. Action: 1

avoid it.

intercept it.

it.

down

enemies and collecting items. E

down

Action: 2

Action: 1

|xa<br><br>e<br><br>ti|mple response:<br><br>asoning: The alien is right above,<br><br>on:1 Alien| |
|---|---|---|
|continue firing to try and take it<br><br>down Pong| | |

Ac

Action: 1

###### Ms. Pacman

###### Frogger

###### Breakout

Action: 1

[Figure 29]

R

[Figure 30]

[Figure 31]

c

###### Ms. Pacman

###### Pong

d

Action: 1

[Figure 32]

[Figure 33]

Goal: Navigate through a maze, eating all dots

Goal: Use your paddle to

hit the ball past your opponent's paddle to

while avoiding ghosts or

```json { "reasoning": "Ms. Pacman is now directly above the ghost. Moving down should allow

eating them when powered up.

score points.

her to eat it and gain points.", "action": 4 } ```

Goal: Navigate through a maze,

Goal: Navigate through maze-like environments while shooting alien enemies and collecting items.

###### Example response:

Example response: Reasoning: The ball is

eating all dots while avoiding ghosts

Goal: Use a paddle to bounce a ball

Goal: Guide frogs across a busy road

Reasoning: Ms. Pacman

or eating them when powered up. Example response:

to break bricks at the top of the screen. Example response: Reasoning: The ball is moving right. I

and river to reach their homes safely.

is now directly above the ghost. Moving down

moving towards our

Example response: Reasoning: There's a car coming from

Example response:

Reasoning: Ms. Pacman is slightly

paddle, we must move the paddle down to

Reasoning: The alien is right above,

Goal: Navigate through maze-like

below the ghost now, but moving up

need to move right further to

the left. Moving up will help to avoid

should allow her to eat

continue firing to try and take it

environments while shooting alien

should still allow her to catch it and gain points. Action: 1

intercept it.

it.

down

enemies and collecting items. Example response:

|Alien|
|---|

it and gain points. Action: 4

intercept it.

###### Ms. Pacman

Action: 2

Action: 1

###### Frogger

Breakout

Action: 1

Action: 3

Reasoning: The alien is right above,

|continue firingtoPongtry andtakeit|
|---|

cont

###### Space Invaders

###### Seaquest

down

Action: 1

[Figure 34]

[Figure 35]

Goal: Shoot waves of

Goal: Control a

descending alien invaders while avoiding

submarine to rescue divers while fighting sea

```json { "reasoning": "Ms. Pacman is now directly above the ghost. Moving down should allow

their attacks.

creatures and managing

her to eat it and gain points.", "action": 4 } ```

oxygen.

Goal: Navigate through a maze,

Goal: Navigate through maze-like environments while shooting alien enemies and collecting items.

eating all dots while avoiding ghosts

Goal: Use a paddle to bounce a ball

Goal: Guide frogs across a busy road

or eating them when powered up. Example response:

###### Example response:

to break bricks at the top of the screen. Example response: Reasoning: The ball is moving right. I

and river to reach their homes safely.

###### Example response:

Reasoning: The invaders

Reasoning: The invaders

Example response: Reasoning: There's a car coming from

Example response:

Reasoning: Ms. Pacman is slightly

are at the top of the screen. Firing is the best

are at the top of the screen. Firing is the best

Reasoning: The alien is right above,

Goal: Navigate through maze-like

below the ghost now, but moving up

need to move right further to

the left. Moving up will help to avoid

continue firing to try and take it

environments while shooting alien

should still allow her to catch it and gain points. Action: 1

intercept it.

it.

down

enemies and collecting items. Example response:

option.

option.

Action: 2

Action: 1

Action: 1

|isright above,Pong|
|---|

Action: 1

Action: 1

Reasoning: The alien

i

continue firing to try and take it

###### Frogger

###### Breakout

down

Action: 1

[Figure 36]

[Figure 37]

Goal: Use a paddle to

Goal: Guide frogs across a busy road and river to

bounce a ball to break bricks at the top of the

reach their homes safely.

```json { "reasoning": "Ms. Pacman is now directly above the ghost. Moving down should allow

screen.

her to eat it and gain points.", "action": 4 } ```

Example response: Reasoning: The ball is

###### Example response:

Reasoning: There's a car

moving right. I need to

coming from the left. Moving up will help to

move right to intercept it, but I'm nearing the

avoid it.

right side of the screen.

Action: 1

Action: 1

##### Figure 5 | Goal and example response from model of Atari games used for evaluation. We implement 7 kinds of Atari games from Atari-GPT [70].

```json { "reasoning": "Ms. Pacman is now directly above the ghost. Moving down should allow

her to eat it and gain points.", "action": 4 } ```

To evaluate out-of-distribution generalization, we test ViGaL on Atari-GPT [70], a benchmark for evaluating MLLMs as decision-making agents in Atari video games, as shown in Fig. 5. The benchmark consists of seven different Atari games: Alien, Frogger, Pong, Ms. Pacman, Seaquest,

Space Invaders and Breakout. These games present diverse visual environments which is different from Snake game and Rotation game, and require different strategic approaches to finish the goal, making them an ideal test bed for ViGaL evaluating out-of-distribution generalization capabilities.

For evaluation, we input game frames as pixel observations to our model, following the established protocol in Atari-GPT. Specifically, each game frame is resized from 210 × 160 × 3 to 512 × 512 × 3, then provided to our model along with game-specific action information. We maintain a context buffer containing the two previous frames and responses together with the current frame to enable temporal reasoning. Following Atari-GPT, we implement frame skipping of 8 frames, which extends the standard 4-frame skipping in ALE to reduce computational intensity while preserving gameplay continuity.

We evaluate our method through four independent rollouts of 1,000 timesteps each and report the average cumulative reward, with results presented in Tab. 1c.

##### B.2. Ablation On Rotation Game

- Table 8 | Ablation study. Similar to the evaluation in Tab. 7, we analyze how different aspects of our post-training strategy within the Rotation game affect downstream generalization benchmarks. The base model is Qwen2.5-VL-7B, with results shown in gray. The default settings from Tab. 2 and Tab. 3 are highlighted in blue . We observe the same improvement trends for each strategy as reported in Tab. 7.

(a) Prompt design.

(b) SFT vs. RL.

prompt Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 w/o Reasoning Instruction

post-training Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 SFT 55.6 44.0 75.4 47.5 RL 62.6 49.3 80.7 57.9

- 61.4 48.9 80.4 54.8

w/ Reasoning Instruction

- 62.6 49.3 80.7 57.9

(c) Difficulty control.

difficulty control Avg. Math CLEVR+ Geo. base model 49.1 47.7 54.9 44.8 w/o difficulty control 61.0 48.0 80.2 54.8 w/ difficulty control 62.6 49.3 80.7 57.9

As shown in Tab. 8, we conduct a similar ablation study to Tab. 7, but replace the Snake game environment with the Rotation game. Our results demonstrate the same consistent improvement trends on downstream generalization benchmarks for each strategy employed.

Specifically, we control the task difficulty by varying the rotation angles between two images. In the uncontrolled difficulty setting, the rotation angle between images can be clockwise 90°, counter-clockwise 90°, or 180°. However, we found that explicitly requiring the model to distinguish between clockwise and counter-clockwise rotations leads to training difficulties. Therefore, we remove it and only retain option of clockwise 90° and 180° rotations.

Unlike the Snake game, we cannot conduct the ablations shown in Tab. 7e because the Rotation game is inherently vision-dependent and requires visual input. Similarly, we cannot perform the ablations in Tab. 7b because the Rotation game provides only binary answer options, making it impossible to meaningfully designate both "best" and "worst" answers simultaneously.

8

| |+6.25<br><br>+5.72<br><br>+3.73<br><br>+3.12 +2.82<br><br>+1.86<br><br>+1.10 +0.78 +0.63|
|---|---|
| | |
| | |
| | |
| | |
| |CoordinateLength AngleAppliedPropertyAnalytic AreaVolumeExpression|
| | |
| | |

AccuracyDifferences(%)

6

4

2

0

2

4

6

- Figure 6 | Accuracy differences between ViGaL-Snake+Rotation and base model without RL training across mathematical subfields in Mathverse. The synergistic effects of jointly training on two games observed suggest that complementary games can enhance overall mathematical reasoning capabilities.

##### B.3. Synergistic Effects of Multi-Game Training

As discussed in Sec. 3.1, our analysis reveals that each game develops distinct reasoning abilities in the model. To investigate potential combined benefits, we conducted experiments where models were trained simultaneously on both the Snake and Rotation games. Fig. 6 shows that joint training effectively combines the strengths of each individual game, improving performance across the mathematical areas where each game shows particular effectiveness, resulting in greater overall gains on Mathverse. These results suggest that strategically combining games with complementary strengths offers a simple yet effective approach to enhance model generalization abilities.

##### B.4. Reasoning Ability Boundary via Pass@𝑘 Evaluation

MathVista Ours

0.85

MM-Eureka-Qwen-7B-Instruct

Qwen2.5-VL-7B-Instruct

0.80

Coverage(pass@k)

0.75

0.70

0.65

1 2 4 8 16

Number of Sample K

- Figure 7 | Pass@𝑘 performance curves on MathVista comparing base models with their zero-RL counterparts trained on mathematical data and game data, respectively.

We explore the reasoning ability boundary of models trained with different RL approaches by evaluating the pass@𝑘 metric. This metric measures the probability that at least one of 𝑘 independent model samples solves a given problem, indicating the true scope or boundary of a model’s reasoning capability - essentially what problems the model can potentially solve given enough sampling attempts.

We evaluate the pass@𝑘 performance of three models: the Base Model without RL training, MM-Eureka-Qwen-7B-Instruct, and our ViGaL. As shown in Fig. 7, our ViGaL consistently demonstrates increasing pass@𝑘 scores on Mathverse as 𝑘 increases. This finding suggests that our approach can effectively solve complex problems when allowed multiple reasoning attempts, uncovering capabilities not apparent in single-sample evaluations.

Moreover, compared to the other RL-trained model, MM-Eureka-Qwen-7B-Instruct, our model achieves a steeper improvement in pass@𝑘 as 𝑘 increases. This indicates that ViGaL possesses a broader reasoning boundary and stronger reasoning abilities, enabling it to solve a wider range of problems when given sufficient opportunities to explore different solution paths.

Finally, our results demonstrate that as 𝑘 increases, base models without RL training eventually outperform RL-trained models. This aligns with the findings in [80] that highlight a fundamental limitation of reinforcement learning with verifiable rewards (RLVR): while RL training significantly improves performance at small 𝑘 values (e.g., pass@1), base models possess a wider coverage of solvable problems. This suggests a trade-off where RL optimization focuses on solving high-probability problems at the expense of broader solution coverage. Future work should explore RLVR algorithms that can improve pass@𝑘 performance across all values of 𝑘, effectively extending the reasoning boundary beyond that of the base model.

| |MathMath| |
|---|---|---|

###### Atari

###### CLEVR+ Geometry

###### Multi-Discipline

###### Atari

###### Geometry

###### CLEVR+

Breakout

Mathvista CLEVR-Math GeoMath

Breakout

MMMU & MMMU-Pro Comprehensive Disciplines

Mathvista

GeoMath

CLEVR-Math

|Re Th th pa po Ac|[Figure 38]<br><br>Reasoning: The ball is about to hit the paddle. The paddle is already wellpositioned. Action: 0<br><br>[Figure 39]<br><br>asoning: e ball is about to hit e paddle. The ddle is already wellsitioned. tion: 0| |
|---|---|---|

|Qu fu sm x=|Question: Which function has the smallest value at x=5?<br><br>[Figure 40]<br><br>estion: Which nction has the<br><br>allest value at 5?<br><br>[Figure 41]| | |Q<br><br>th an th AB|Question: Subtract<br><br>all blue blocks. Subtract all brown spheres. How<br><br>many blocks are<br><br>left?<br><br>uestion: What is<br><br>e measure of gle B if angle D in e parallelogram<br><br>CD is 60 degrees?<br><br>[Figure 42]|[Figure 43]| |Q<br><br>al Su sp<br><br>m<br><br>le|Question: What is<br><br>the measure of angle B if angle D in the parallelogram ABCD is 60 degrees?<br><br>[Figure 44]<br><br>[Figure 45]<br><br>uestion: Subtract<br><br>l blue blocks. btract all brown heres. How<br><br>any blocks are<br><br>ft?| |
|---|---|---|---|---|---|---|---|---|---|---|

[Figure 46]

Ms. Pacman

Mathvision Super-CLEVR Geometry3K

Ms. Pacman

Geometry3K

Mathvision

Super-CLEVR

|Q<br><br>I A (A|[Figure 47]<br><br>Question:<br><br>Is <image 1> an Alternating group? (A) Yes (B) No<br><br>[Figure 48]<br><br>uestion: s <image 1> an lternating group?<br><br>) Yes (B) No| | |Q D m<br>R ( (<br>|Question: Add 1 small bikes. How many<br><br>objects are left?<br><br>[Figure 49]<br><br>uestion: etermine the easure of angle<br><br>. Choices:<br><br>[Figure 50]<br><br>A) 5 (B) 40 C) 78 (D) 122| | |Q A H<br><br>ob|Question: Determine the measure of angle<br><br>R. Choices:<br><br>(A) 5 (B) 40 (C) 78 (D) 122<br><br>uestion: dd 1 small bikes. ow many<br><br>jects are left?<br><br>[Figure 51]|[Figure 52]|
|---|---|---|---|---|---|---|---|---|---|---|

[Figure 53]

[Figure 54]

Reasoning:

Reasoning:

Moving up collects

Moving up collects

a pellet and keeps a safe distance from ghosts. Action: 1

a pellet and keeps a safe distance from ghosts. Action: 1

(a) Atari games.

(b) Out-of-domain tasks.

- Figure 8 | Samples from our generalization reasoning benchmarks. We evaluate the proposed ViGaL with two types of generalization: (a) out-of-distribution generalization, where models trained on our visual games are tested on unseen Atari games [70]; and (b) out-of-domain generalization, where models trained only on game tasks are evaluated on diverse multimodal reasoning tasks including mathematical reasoning, geometric problem-solving, 3D understanding on CLEVR+ and multi-discipline reasoning on MMMU series.

##### B.5. Detail of Evaluation Benchmarks

To obtain a clearer picture of the various facets of MLLM performance, we follow prior studies [40, 65] and systematically and carefully divide existing benchmarks into two broad groups: (i) reasoning-oriented benchmarks, which require multi-step or mathematical reasoning to solve the problems, and (ii) general-purpose perception benchmarks, which primarily assess broad visual understanding and perception abilities.

For reasoning-oriented benchmarks, we comprehensively evaluate the visual reasoning generalization capabilities of RL through gaming on a diverse collection of tasks that specifically demand advanced visual reasoning skills, including math-focused tasks like Math and Geometry, and other comprehensive reasoning benchmarks beyond math, like CLEVR+ and Multi-Discipline. Fig. 8b illustrates specific examples from each benchmark.

- • Math evaluates multimodal math reasoning with widely-used datasets: MathVista (testmini) [46], MathVerse (testmini) [83], and MathVision (test) [69]. MathVista offers diverse problems spanning VQA, logic, algebra, and geometry; MathVerse emphasizes algebraic and geometric image comprehension; MathVision tests abstract visual reasoning.
- • Geometry evaluates structural interpretation skills across mathematical diagrams, medical images, charts, and architectural layouts. It uses datasets GeoMath (Geo170K [27], Math360K [58]) and Geometry3K [47], featuring both choice and non-choice questions. Following Reason-RFT [59], we test with 820 GeoMath and 800 Geometry3K samples.
- • CLEVR+ evaluates the integration of mathematical and spatial reasoning skills through challenging arithmetic problems in complex 3D block-based scenes, including sub-tasks on CLEVR-Math [43] and Super-CLEVR [41]. Following Reason-RFT [59], we use 1K test samples from each of CLEVR-Math and Super-CLEVR.
- • Multi-Discipline evaluates college-level expert knowledge across six disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. We follow the evaluation setting of MMMU [78] val set (900 questions) and MMMU-Pro [79] overall score (average of standard 10-option and vision-only settings).

For general-purpose perception benchmarks, we systematically evaluate comprehensive visual capabilities. Following previous work, these benchmarks are categorized into three distinct types: General, Vision-Centric, and OCR & Chart.

- • General benchmarks assess fundamental visual understanding capabilities. We evaluate MuirBench [68] for multi-image understanding and CRPE [35] for relation understanding.
- • Vision-Centric benchmarks thoroughly evaluate perception, real-world understanding, and multi-modal capabilities. We assess MMVP [66], RealWorldQA [73], MMStar [13], MME [25], and BLINK [26].
- • OCR & Chart understanding benchmarks focus on text-rich visual content. We specifically use AI2D [36] for diagram understanding, SEED-Bench-2-Plus [38] for text-rich visual comprehension, DocVQA [49] for document understanding, and OCRBench [45] for comprehensive OCR evaluation.

For 3D spatial reasoning, we evaluate on VSI-Bench [76]. We run the benchmark through the LMMs-Eval [81] harness with the standard VSI Bench configuration, use greedy decoding for inference, and uniformly sample 32 frames per video across the full duration.

##### B.6. Inference Length Analysis

Recent reinforcement learning studies [1, 75] have raised questions about whether performance improvements stem from genuinely enhanced reasoning capabilities or merely from models generating longer responses. To address this concern, we analyze the relationship between response length and performance for models trained with our game-based approach.

Model Response Length Math Avg.

Qwen2.5-VL-7B (baseline) 250 47.7 ViGaL (ours, RL on games) 268 50.6

- Table 9 | Response length and performance on visual math benchmarks. Our game-based RL approach achieves significant performance gains while maintaining comparable inference costs.

Table 9 demonstrates that our performance improvements are not simply due to increased verbosity. Our ViGaL model achieves substantial performance gains (50.6% vs. 47.7%) while maintaining nearly identical inference costs—the response length increases by only 7% (268 vs. 250 tokens). This minimal increase in response length, coupled with the significant accuracy improvement, indicates that the model has learned transferable skills rather than merely generating longer outputs.

These results suggest that game-based RL training enables effective knowledge transfer from game environments to mathematical problem-solving. For example, spatial reasoning skills acquired from the Rotation game and coordinate recognition abilities developed through Snake gameplay transfer effectively to visual math tasks. The model thus learns genuine problem-solving strategies while maintaining inference costs.

##### B.7. Reasoning Correlation Analysis Between Game and Math

To understand the mathematical reasoning patterns in snake game playing, we developed a systematic approach to extract and analyze reasoning steps from multiple gameplay traces. Our methodology uses GPT-5 [52] as an analytical tool in a two-stage process.

- In Stage A, we collect multiple snake game "thinking traces", which are detailed reasoning

sequences generated during gameplay, and distill them into a generalized set of 8 core reasoning steps. These steps abstract away specific details like exact coordinates or particular board configurations to capture fundamental cognitive operations. The operations include parsing board state, enumerating moves, safety screening, path metric selection, distance computation,

target identification, enemy anticipation, and move ranking. This summarization ensures our analysis focuses on transferable reasoning patterns rather than game-specific instances.

- In Stage B, we quantify how mathematical each reasoning step is by evaluating its correlation

with nine distinct mathematical aspects. We use a simple 3-level scoring system where 0 means no correlation, 1 means low correlation, and 2 means high correlation. GPT-5 analyzes how strongly each step relates to mathematical concepts such as coordinate manipulation, distance metrics, analytical reasoning, and geometric properties.

The resulting correlation matrix in Tab. 10 reveals clear patterns. Coordinate-based reasoning dominates steps that involve spatial parsing and movement planning, particularly Steps 1 through 3, Step 5, and Step 7. Meanwhile, analytical and length-based reasoning become prominent in optimization steps like target identification and move ranking, seen in Steps 6 and 8. Steps 4 and 5, which involve path metrics and distance computation, show high correlation with both coordinate systems and length calculations. This confirms the geometric nature of pathfinding in grid-based environments. Our systematic analysis demonstrates that even seemingly simple game-playing behaviors require sophisticated integration of multiple mathematical reasoning capabilities.

Prompt Template for Reasoning Step Extraction and Correlation Analysis

- Stage A - Step Extraction: Given multiple snake game thinking traces, extract N general reasoning steps (6-9 steps) that capture the core operations. Abstract away instance-specific details and output:

- • Short, action-oriented step names with one-line descriptions
- • General patterns covering: state parsing, move generation, safety screening, target selection via distance, opponent awareness, scoring/tie-breaks, decision, reporting

- Stage B - Mathematical Aspect Correlation: For each extracted step, assign correlation levels (0/1/2) to these mathematical aspects:

- • Expression: Formatting/structuring outputs
- • Coordinate: Reading/writing positions, mapping moves to (x,y)
- • Area: Board regions/bounds as areas
- • Volume: 3D spatial reasoning (if applicable)
- • Applied: Goal-directed task execution
- • Property: Rules/invariants (bounds, occupancy, collision)
- • Angle: Angle-based path reasoning
- • Analytic: Selection/optimization, tie-break logic
- • Length: Distance metrics (Manhattan/L1, grid paths)

Output as structured table with integer scores only (0 = no correlation, 1 = low, 2 = high).

###### .

- Table 10 | Correlation Matrix of each step reasoning trace of playing snake game with solving math questions. (0=No Correlation, 1=Low Correlation, 2=High Correlation)

Coordinate

Expression

Property

Analytic

Applied

Volume

Length

Angle

Area

Step Operation

- 1 Parse the board state

- 2 Enumerate candidate moves

- 3 Safety screening (worst-move test)

- 4 Choose a path metric

- 5 Compute distances to apples

- 6 Identify nearest target

- 7 Anticipate enemy motion (when present)

- 8 Rank safe moves

##### Correlation Legend:

= No Correlation = Low Correlation = High Correlation

#### C. Case Study

Rotation

Question: As shown in the figure, the straight line a and the straight line b are intercepted by

[Figure 55]

the straight line c. Line b is perpendicular to c, and the foot of the perpendicular is

point A. ∠1 = 70°. If line b is parallel to line a, how many degrees clockwise can line b be rotated around point A?

Choices: A: 70° B: 50° C: 30° D: 20°

Ours

Qwen2.5-VL-7B-Instruct

[Figure 56]

- 1. Identify the given information: Line b is perpendicular to line c . The foot of the perpendicular is point A . ∠ 1 = 70°. Line b is parallel to line a.
- 2. Analyze the geometric relationships: Since line b is perpendicular to line c , ∠BAC = 90°. Line b is parallel to line a, so the corresponding angles are equal. The angle ∠ 1 is given as 70°.
- 3. Determine the rotation angle: - The angle ∠ 1 represents the angle between line b and the vertical line c . Since line b is perpendicular to line c , the angle between line b and line a (which is parallel to line b ) is 90° - 70° = 20° .

[Figure 57]

- 1. Identify the given information: Line b is perpendicular to line c. The foot of the perpendicular is point A. Angle 1 is 70°. Line b is parallel to line a.
- 2. Understand the geometric relationships: Since line b is perpendicular to line c, angle 1 is the angle between line a and line c. Since line b is parallel to line a, the angle between line b and line c is the same as the angle between line a and line c, which is 70°.
- 3. Determine the angle of rotation: The angle of rotation is the difference between the initial angle 70°. and the final angle 0°, since line b is perpendicular to line c.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Therefore, the line b can be rotated 20° clockwise around the point A .The

Therefore, the angle of rotation is 70°.Thus, the line b can be rotated 70°clockwise around the point A. The correct answer is: A

correct answer is D.

- (a) A case study from Mathverse. Base model misinterpreted the geometric configuration and rotation direction, while our model correctly identified the perpendicular relationship and calculated the proper angle.

Question: As shown in the figure, the straight line a and the straight line b are intercepted by

the straight line c. Line b is perpendicular to c, and the foot of the perpendicular is

point A. ∠1 = 70°. If line b is parallel to line a, how many degrees clockwise can line b be rotated around point A?

Choices: A: 70° B: 50° C: 30° D: 20°

[Figure 64]

- 1. Identify the given information: Line b is perpendicular to line c. The foot of the perpendicular is point A. Angle 1 is 70°. Line b is parallel to line a.
- 2. Understand the geometric relationships: Since line b is perpendicular to line c, angle 1 is the angle between line a and line c. Since line b is parallel to line a, the angle between line b and line c is the same as the angle between line a and line c, which is 70°.
- 3. Determine the angle of rotation: The angle of rotation is the difference between the initial angle 70°. and the final angle 0°, since line b is perpendicular to line c.

Therefore, the angle of rotation is 70°.Thus, the line b can be rotated 70°clockwise around the point A. The correct answer is: A

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

- 1. Identify the given information: Line b is perpendicular to line c . The foot of the perpendicular is point A . ∠ 1 = 70°. Line b is parallel to line a.
- 2. Analyze the geometric relationships: Since line b is perpendicular to line c , ∠BAC = 90°. Line b is parallel to line a, so the corresponding angles are equal. The angle ∠ 1 is given as 70°.
- 3. Determine the rotation angle: - The angle ∠ 1 represents the angle between line b and the vertical line c . Since line b is perpendicular to line c , the angle between line b and line a (which is parallel to line b ) is 90° - 70° = 20° .

Therefore, the line b can be rotated 20° clockwise around the point A .The

correct answer is D.

Question:

Determine whether the graphed function is even, odd, or neither.

Let's analyze the graph:

- 1.Symmetry with respect to the y-axis: If the graph were symmetric with respect to the y-axis, then for every point (x, y) on the graph, the point (-x, y) would also be on the graph. However, the graph in the image does not appear to be symmetric with respect to the y-axis. For example, the point (3, y) does not have a corresponding point (-3, y) on the graph.
- 2. Symmetry with respect to the origin: If the graph were symmetric with respect to the origin, then for every point (x, y) on the graph, the point (-x,

-y) would also be on the graph. However, the graph in the image does not

appear to be symmetric with respect to the origin. For example, the

point(3, y) does not have a corresponding point (-3, -y) on the graph.

Since the graph does not exhibit symmetry with respect to either the y-

axis or the origin, the function is neither even nor odd.The final answer is: neither.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Let's examine the graph of the function. The graph shows a parabola that opens upwards and is symmetric about the y-

axis. This symmetry indicates that the function is even.

To confirm, we can check a few points:

For x = 2 , the function value is f(2). For x = -2 , the function

value is f(-2) .Since the graph is symmetric about the y-axis, f(-2) = f(2) .

This confirms that f(-x) = f(x) , which is the definition of an

even function.

Therefore, the function is even.The final answer is: even.

[Figure 79]

Qwen2.5-VL-7B-Instruct

Qwen2.5-VL-7B-Instruct Ours

Coordinate

Rotation

Ours

- (b) A case study from Mathverse. Base model misperceived critical visual information like symmetry and coordinates in graphs, while our model demonstrated accurate visual perception for mathematical elements.

Coordinate

[Figure 80]

Question:

Determine whether the graphed function is even, odd, or neither.

Qwen2.5-VL-7B-Instruct Ours

Let's analyze the graph:

Let's examine the graph of the function. The graph shows a parabola that opens upwards and is symmetric about the y-

- 1.Symmetry with respect to the y-axis: If the graph were symmetric with respect to the y-axis, then for every point (x, y) on the graph, the point (-x, y) would also be on the graph. However, the graph in the image does not appear to be symmetric with respect to the y-axis. For example, the point (3, y) does not have a corresponding point (-3, y) on the graph.
- 2. Symmetry with respect to the origin: If the graph were symmetric with respect to the origin, then for every point (x, y) on the graph, the point (-x,

[Figure 81]

[Figure 82]

axis. This symmetry indicates that the function is even.

###### To confirm, we can check a few points:

For x = 2 , the function value is f(2). For x = -2 , the function

[Figure 83]

value is f(-2) .Since the graph is symmetric about the y-axis, f(-2) = f(2) .

-y) would also be on the graph. However, the graph in the image does not

[Figure 84]

appear to be symmetric with respect to the origin. For example, the

This confirms that f(-x) = f(x) , which is the definition of an

point(3, y) does not have a corresponding point (-3, -y) on the graph.

even function.

Since the graph does not exhibit symmetry with respect to either the y-

[Figure 85]

Therefore, the function is even.The final answer is: even.

[Figure 86]

axis or the origin, the function is neither even nor odd.The final answer is: neither.

30

- Figure 9 | Comparison of base model and our model after rule-based RL training, showing improved visual-mathematical reasoning on geometric and coordinate problems.

We provide quantitative comparison examples below to demonstrate reasoning improvements on mathematical problems after RL training. In Fig. 9a, when solving a geometric angle problem, the base model fails to correctly interpret the critical relationship between perpendicular lines and corresponding angles. It makes contradictory assumptions about angle measures, leading to an incorrect calculation of the required rotation. In contrast, our ViGaL precisely tracks the geometric constraints and properly calculates the angle difference between initial and target positions. In Fig. 9b, when analyzing function properties from a graph, the base model incorrectly claims the function lacks symmetry despite clear visual evidence. It fails to recognize the fundamental y-axis symmetry of the parabola shown in the image. Our model immediately identifies this critical symmetrical pattern and correctly applies the appropriate mathematical definition of an even function, demonstrating enhanced visual perception of mathematical structures.

