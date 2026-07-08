## Where to Look: Can Foundation Models Reach a Target Viewpoint Through Active Exploration?

### Liyang Li*, Muzhi Zhu*, Zhiyue Zhao, Hengyu Zhao, Ke Liu Linhao Zhong, Hao Chen, Chunhua Shen† Zhejiang University

###### *Equal contribution †Corresponding author

Target Viewpoint Reproduction

Reasoning: The current view shows a corner with a side table, a statue, and part of a gray couch. The target view shows a different angle with a striped armchair, a laptop, and curtains. To match the target, I need to move forward and rotate right to shift the viewpoint towards the window area and the striped armchair.

[Figure 1]

[Figure 2]

# arXiv:2606.01247v1[cs.CV]31May2026

[Figure 3]

###### Action: Move Ahead

Active Head Rotation Exploration

[Figure 4]

Target View Current View Spatial Memory

|[Figure 5]|
|---|

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

- Step 1: RotateRight
- Step 2: RotateRight
- Step3:Move…Ahead

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Body Move Direction

Figure 1: Target Viewpoint Reproduction (TVR). Can a foundation model actively reproduce a target viewpoint in 3D, closing the perception–reasoning–action loop through body translation and head rotation?

### Abstract

offline Single-turn GRPO, and on-policy Multiturn GRPO from live simulator rollouts. Visualaction SFT supplies the main gain, raising a 9B open-source model to 50.8% success; Multiturn GRPO provides targeted multi-room refinement and reaches 51.4% overall, while CoT supervision and Single-turn GRPO degrade closed-loop performance. These results establish TVRBench as a testbed for measuring and training foundation models that actively perceive and act in 3D environments. Our code, data, and models are available at https://github.com/aim-uofa/TVRBench.

Humans can reproduce the viewpoint specified by a target image through active head and body motion, yet spatial intelligence in foundation models has largely been studied as passive understanding of pre-collected observations. We introduce Target Viewpoint Reproduction (TVR)—an active task where an agent adjusts its viewpoint in a 3D environment until its observation matches a given target imageand TVRBench, an indoor-simulation benchmark spanning scene scale and target-view visual richness. TVR is far from solved: on the evaluation split, the strongest open-source and closed-source models reach only 7.8% and 12.0% success. Fine-grained analysis identifies two consistent bottlenecks: off-the-shelf models struggle with multi-turn visual history, and performance drops sharply when viewpoint reproduction requires body translation rather than in-place rotation, exposing a gap in mapping spatial discrepancies to embodied movement. To study reducing this gap, we build a unified TVR post-training framework covering experttrajectory SFT, rationale-supervised CoT-SFT,

### 1 Introduction

Reproducing the viewpoint from a single target image is a basic form of active spatial intelligence. The agent must compare the target with its egocentric view, infer the viewpoint gap, map it to body translation, rotation, and head motion, update spatial belief from new observations, and decide when the match is accurate enough to stop. Humans do this naturally: instead of passively matching static content, we move in 3D, gather

visual evidence, and refine actions through a closed perception–action loop.

Recent spatial-intelligence research on foundation models, especially MLLMs, has introduced diverse tasks and benchmarks for relative position, directional relations, 3D layouts, and cross-view reasoning (Chen et al., 2024; Cheng et al., 2024; Hong et al., 2023; Yang et al., 2025b). Yet most assume visual observations are given in advance, as a static image, multi-view inputs, or a prerecorded video, and thus ask only what is where, not where should I move and look next. Active-exploration tasks such as ImageNav (Zhu et al., 2017; Krantz et al., 2022) move closer to embodied spatial intelligence, but typically evaluate whether agents reach a target region rather than whether their final egocentric observation reproduces the goal image.

This raises a central question: can foundation models infer current-to-target spatial relations, map them to embodied actions, and reproduce target views through active exploration? We introduce Target Viewpoint Reproduction (TVR), where an agent receives a target image and initial observation in a 3D environment, then acts until its observation matches the target. TVR is active, gathering new observations in a closed perception– action loop, and evaluates explicit viewpoint control: the agent must reproduce the target viewpoint rather than merely reach a region. We instantiate TVR in indoor simulation as TVRBench, covering single-room and multi-room scenes with diagnostics for exploration efficiency, spatial memory, and perception-to-action mapping.

Across the open- and closed-source MLLMs we evaluate, TVRBench shows TVR remains far from solved: the strongest open-source model reaches 7.8% success and the strongest closedsource model 12.0%, versus 93% human performance on a 100-task subset. Fine-grained analysis finds two bottlenecks. First, off-the-shelf models struggle with multi-turn visual history: every opensource model performs better with an action-only recap than full visual-action memory (mean gap +3.8pp). Second, performance drops when viewpoint reproduction requires body translation rather than in-place rotation, suggesting the main difficulty is mapping spatial discrepancies to embodied movement, beyond static visual recognition.

We build a unified TVR post-training framework to target these bottlenecks, covering experttrajectory SFT (Kim et al., 2024), CoT-SFT, offline Single-turn GRPO (Liao et al., 2025),

and on-policy Multi-turn GRPO from live rollouts (Zeng et al., 2024). It compares models/training paradigms for closed-loop active perception. Visual-action SFT gives the main gain, raising Qwen3.5-9B to 50.8% without CoT. Multi-turn GRPO refines VA-SFT to 51.4%, mainly on multiroom tasks where SFT is weakest. In contrast, CoT supervision and Single-turn GRPO reduce success, suggesting per-step rationales or action matching may not transfer to embodied multi-step control.

Our main contributions are as follows:

- • We introduce Target Viewpoint Reproduction (TVR), a closed-loop target-viewpoint reproduction task, and TVRBench, an indoorsimulation benchmark with protocols diagnosing exploration efficiency, spatial memory, and perception-to-action mapping.
- • We benchmark open- and closed-source foundation models on TVRBench and identify two consistent bottlenecks: exploiting multi-turn visual history and mapping spatial discrepancies to body translation.
- • We develop a unified TVR post-training framework for comparing expert-trajectory SFT, CoT-SFT, and single-/multi-turn GRPO in closed-loop environments.
- • Using this framework, we show that visualaction SFT supplies the main improvement (50.8%) and Multi-turn GRPO provides targeted multi-room refinement (51.4% overall), while CoT supervision and Single-turn GRPO degrade closed-loop performance.

### 2 Related Work

#### 2.1 Spatial Intelligence

Early foundation-model work on spatial intelligence addressed static inputs: from text-image pairs or single visual observations, models answer questions about relative position, orientation, directional relations, topology, or 3D layout (Johnson et al., 2017; Liu et al., 2023; Wang et al., 2024; Chen et al., 2024; Cheng et al., 2024; Li et al., 2025). Later work extended this to multi-view settings for cross-view matching, spatial-relation inference, and local-to-global scene-structure understanding (Hong et al., 2023; Yeh et al., 2026; Yin et al., 2025; Xu et al., 2025; Yang et al., 2025b), and videos, where continuous observations enable spatial updating and temporal reasoning (Yang et al.,

2025a; Zhou et al., 2025). Another line grounds spatial reasoning in embodied agents via embodied question answering and affordance prediction (Ma et al., 2022; Majumdar et al., 2024; Zhou et al., 2024; Yuan et al., 2024).

Across settings, however, visual observations are typically pre-collected, not acquired through exploration: the model is asked only “what is where,” not “where should I look next.”

#### 2.2 Active Embodied Reasoning

Visual navigation dominates active embodied tasks. Goals are specified by an object class (ObjectNav (Batra et al., 2020)), a goal image (ImageNav (Zhu et al., 2017; Krantz et al., 2022)), or a natural-language instruction (VLN (Anderson et al., 2018; Zhou et al., 2024)). Across settings, success measures whether the agent’s position reaches the target region or fulfills the instruction, rather than whether its final observation reproduces a target viewpoint. Even ImageNav uses a goal image but scores proximity, not exact visual match.

Recent work investigates active spatial reasoning with foundation models (Yang et al., 2025a; Yin et al., 2025; Zhu et al., 2025b; Zhang et al., 2026; Zhu et al., 2025a), often using simplified action spaces such as teleportation or restricted agent positions. Concurrent humanoid visual search work (Yu et al., 2025) studies head-rotation-only object and path search over 360◦ panoramas, while visually grounded active view selection (Koo et al., 2025) selects informative next views without reproducing a specific target. Hong et al. (2026) introduce ESI-Bench, a broad embodied-spatial-intelligence benchmark with ten OmniGibson task categories, and Sakamoto et al. (2026) propose E3VS-Bench for active VQA in 3DGS scenes.

TVR differs from these settings along two axes: (i) success is defined by an explicit viewpoint match—the agent’s observation must reproduce the viewpoint of a given target image—rather than reaching a position, identifying an object, or completing an instruction; and (ii) the action space spans both body movement and head rotation, without teleportation, fixed positions, or restriction to a single action modality, demanding coordination.

#### 2.3 Post-Training for Vision-Language and Embodied Tasks

Recent work applies post-training to spatial reasoning vision-language models, achieving substantial gains on static spatial benchmarks. Large-scale

SFT on simulator-generated spatial QA establishes the supervised paradigm (Ray et al., 2024), while pure GRPO lifts a small VLM past proprietary baselines on video spatial QA (Liao et al., 2025), and a two-stage SFT-then-GRPO has emerged as a dominant recipe (Wu et al., 2026). These methods, however, target static spatial QA from pre-collected inputs rather than closed-loop active perception.

Vision-Language-Action (VLA) models extend pretrained VLMs to robotic control through supervised demonstrations on robot data (Kim et al., 2024; Black et al., 2024), framing embodied control as action-token sequence prediction. Concurrent work on transformer-based on-policy reinforcement learning shows scaling RL produces strong embodied navigators (Zeng et al., 2024).

Our experiments suggest a mismatch between this per-step paradigm and TVR’s active multi-step structure: per-step Single-turn GRPO regresses below its SFT initialisation, while trajectory-level Multi-turn GRPO over live rollouts is required to refine rather than overwrite the supervised priors.

### 3 Target Viewpoint Reproduction and TVRBench

Active spatial intelligence involves more than recognizing what is visible; it also requires choosing where to look next and moving to obtain that view. We study this ability through a task in which success depends on viewpoint recovery, without language grounding as a confounding factor.

#### 3.1 The TVR Task

In Target Viewpoint Reproduction (TVR), an agent operates in a 3D indoor environment and is given a single target image I⋆ rendered from a viewpoint in the same scene. At each timestep, the agent observes the current first-person image It and selects one action. The episode ends when the agent selects the Stop action or reaches the step limit for the task. The agent succeeds only if its final pose exactly matches the viewpoint of I⋆.

State and action space. The agent state st = (xt,zt,θt,ϕt) comprises the ground-plane position (xt,zt), body yaw θt, and camera horizon (head pitch) ϕt. At each step, the agent selects one of nine discrete agent-centric actions. MoveAhead, MoveBack, MoveLeft, and MoveRight translate the body by 0.25m. RotateLeft and RotateRight rotate the body by 45◦; LookUp and LookDown

[Figure 13]

###### Single-room / Easy

Single-room / Hard

[Figure 14]

[Figure 15]

[Figure 16]

Top-down

start

Top-down

start

[Figure 17]

[Figure 18]

target

target

[Figure 19]

[Figure 20]

Multi-room / Easy Multi-room / Hard

Top-down

Top-down

[Figure 21]

[Figure 22]

start

start

[Figure 23]

[Figure 24]

target

target

- Figure 2: TVRBench Task Structure A 2×2 task design crossing scene scale and target-view visual richness. Each category shows one representative task: an orthographic top-down with start (yellow) and target (red) poses, and first-person views at both poses. We label the four categories Single-easy, Single-hard, Multi-easy, Multi-hard.

shift the camera horizon by 30◦. Stop signals task completion and ends the episode.

Observation and termination. At each step, the agent observes only the first-person RGB image It rendered from st, with no privileged access to its pose, the target pose, or a scene map. An episode ends when the agent issues Stop or reaches the task step limit. The limit is 30 steps for single-room and 40 steps for multi-room tasks, as multi-room tasks typically require longer routes.

Success criterion. Because action steps and target poses share the same discrete pose grid, the agent can reach the target pose exactly. An episode succeeds if and only if the agent issues Stop and its final pose sT is identical to the target pose s⋆:

##### sT = s⋆.

Thus, the final observation must exactly match the viewpoint of I⋆, not merely approximately. Success is evaluated on the same 0.25m pose grid used by the action space. At this resolution, adjacent poses produce distinguishable observations, so exact matching appropriately tests viewpoint identity.

#### 3.2 The TVRBench Benchmark

Design rationale. TVRBench separates two difficulty sources in viewpoint reproduction: scene

scale and target-view visual evidence. Scene scale tests whether agents move beyond local adjustment, as multi-room cases require traversing rooms to reach target area. Target-view evidence determines how images disambiguate the viewpoint: objectrich views provide landmarks and geometric cues, whereas sparse views offer fewer anchors. We stratify by scene scale and target-view visual richness, with easy/hard tiers for each. The four equal-sized categories, Single-easy, Single-hard, Multi-easy, and Multi-hard, support analysis of movement difficulty and target-view evidence.

Scene sources and sampling. TVRBench uses two scene sources: single-room tasks use iTHOR (Kolve et al., 2017), with 120 kitchens, living rooms, bedrooms, and bathrooms, while multi-room tasks use ProcTHOR-10k (Deitke et al., 2022), with two- or three-room homes separated by physical walls. We split the 240 scenes, 120 per source, into disjoint SFT, evaluation, and RLtraining sets at a 1:2:3 ratio, excluding evaluation scenes from training. Per scene, we uniformly sample (start, target) pose pairs from the reachable grid and filter by visible-object count, the number of non-structural objects1 visible from the target view, and shortest start-to-target action-path length. Easy tasks require at least 9 target-visible objects, while

1Walls, floor, ceiling, and the agent itself are excluded from the count.

- Table 1: Foundation model evaluation on TVRBench. Success rate (%) and diagnostics on the test split (S-e/S-h: single-room easy/hard; M-e/M-h: multi-room easy/hard); top-3 per column: red , green , blue .

Model Overall↑ S-e↑ S-h↑ M-e↑ M-h↑ Steps F-stop↓ |∆p| ↓ |∆θ| ↓ |∆ϕ| ↓

Qwen3.5-9B (VA) 0.0 0.0 0.0 0.0 0.0 34.3 100.0 2.05 86.8 15.8 Qwen3.5-9B (AO) 2.8 6.4 4.8 0.0 0.0 29.9 89.6 1.94 67.8 12.8

- Qwen3.5-27B (VA) 3.2 5.6 4.0 0.8 2.4 31.2 82.0 1.71 52.6 9.8

- Qwen3.5-27B (AO) 7.8 14.4 13.6 0.0 3.2 28.3 75.9 1.57 49.2 6.6

- Qwen3.5-35B-A3B (VA) 0.8 2.4 0.8 0.0 0.0 33.8 84.0 2.03 80.4 10.6

- Qwen3.5-35B-A3B (AO) 3.8 8.0 5.6 0.0 1.6 33.1 67.2 2.00 76.0 9.8
- Qwen3.6-27B (VA) 3.2 8.8 3.2 0.8 0.0 30.9 82.6 1.94 57.0 12.9

- Qwen3.6-27B (AO) 7.0 12.8 9.6 1.6 4.0 27.2 80.8 1.66 53.7 7.6

- Qwen3.6-35B-A3B (VA) 0.2 0.0 0.8 0.0 0.0 34.1 94.7 2.10 74.6 14.7

- Qwen3.6-35B-A3B (AO) 4.8 9.6 8.0 0.8 0.8 28.8 84.5 1.80 70.6 9.3

- GPT-4o (VA) 2.8 5.6 4.0 1.6 0.0 31.3 87.3 1.68 61.1 11.9

- GPT-4o (AO) 5.2 13.6 5.6 0.8 0.8 33.6 49.0 1.60 52.3 9.2

- GPT-5 (VA) 8.0 15.2 8.0 3.2 5.6 33.7 27.3 1.30 43.2 4.2

- GPT-5 (AO) 8.0 22.4 6.4 2.4 0.8 34.0 0.0 1.54 56.6 7.0 Gemini-3.1-Pro (AO) 12.0 21.6 21.6 0.8 4.0 10.3 87.1 1.47 41.9 8.2 Human 93.0 100.0 88.0 88.0 96.0 22.8 2.1 0.05 0.9 0.0

hard tasks allow only 3–6. Shortest paths span 2–8 action steps in single-room scenes and 10–20 in multi-room scenes. The benchmark contains 125 tasks per category and 500 evaluation tasks total. Representative examples appear in Figure 2.

Memory representations. The agent needs a trajectory record to avoid revisits and judge progress toward I⋆, making past-step representation an important design choice. We use two memory representations throughout experiments. In action-only memory (AO), the model receives the current observation It, target I⋆, and a brief summary of previous actions. In visual-action memory (VA), the full past observation-action sequence remains available in a multi-turn multimodal context. These representations emphasize trade-offs. VA tests whether a model can effectively use trajectory visual history, whereas AO reduces the number of images sent per call. AO makes rate/context-limited closed-source evaluation cheaper/faster.

Evaluation metrics. Beyond the binary success criterion, we report three diagnostic metrics describing how an agent fails. The final pose errors |∆p|,|∆θ|,|∆ϕ| between sT and s⋆ quantify remaining distance in failed episodes. The stop rate, the fraction of episodes terminating with Stop, and the false-stop rate, the fraction of Stop actions taken at non-target poses, separate cases where the agent never stops from cases where it stops at an incorrect pose. We report the mean number of steps to termination, which measures exploration efficiency.

### 4 Can Foundation Models Reproduce Target Viewpoints?

Models and protocol. We benchmark five opensource baselines: dense Qwen3.5-9B (Qwen Team, 2026a), Qwen3.5-27B, Qwen3.6-27B (Qwen Team, 2026b), and MoE Qwen3.5-35B-A3B and Qwen3.6-35B-A3B (Qwen Team, 2026c). We also evaluate three closed-source models: GPT4o (Hurst et al., 2024), GPT-5 (Singh et al., 2025), and Gemini-3.1-Pro (Google DeepMind, 2026). All are evaluated on the held-out 500-task split with step budgets and VA/AO memory settings defined in Section 3.2. Gemini-3.1-Pro is evaluated under AO only, because VA multi-image inference over the full split is prohibitively slow. Open-source models use greedy decoding; closed-source models use the lowest API-supported temperature. For reference, we report human performance from five participants on a balanced 100-task subset, using the same resolution, action space, step budget, and success criterion.

Main results. Table 1 reports success rates by task category across 15 model-memory configurations. The best configuration reaches only 12.0% overall success (Gemini-3.1-Pro, AO), and no model exceeds 13%. By contrast, humans achieve 93.0% on a balanced 100-task subset.

Scaling brings small gains. Dense Qwen3.5 improves from 2.8% at 9B to 7.8% at 27B, while the best closed-source models remain at the 12% ceiling. Results show a consistent pattern. Every open-source model performs better under AO

###### Memoryless revisits

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Step 2

###### Step 3

Step 1

Step 2

target

target

target

target

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Action: Look up Action: Look Down ❌

Action: Move Left

Action: Move Right

[Figure 33]

❌

[Figure 34]

[Figure 35]

Walks in circles Looks in circles

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

###### Step 38 Step 39

Step 39 Step 40

target

target

target

target

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Action: Move Left Action: Move Right

[Figure 44]

[Figure 45]

❌ ❌

Action: Look up ❌ Action: Look Down ❌

[Figure 46]

[Figure 47]

###### Action Distribution Movement bottleneck

Thinking gap

[Figure 48]

[Figure 49]

[Figure 50]

RotateLeft

- 24.9%
- 25.9%

80.5%

Qwen3.5 – 27B Qwen3.5 – 27B

RotateRight

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | |10.0%| | |
| | | | | |

LookUp

13.1%

7.8%

7.0%

LookDown

10.0%

MoveBack

- 7.1%
- 8.5%

MoveRight

3.0%

2.8%

MoveAhead

- 5.2%
- 5.3%

MoveLeft

Stop

0.1%

w/o Thinking w Thinking

Rotate / Look Move only

0% 5% 10% 15% 20% 25% 30%

- Figure 3: Why an untrained 9B fails at TVR. Top: Qwen3.5-9B visits only 3.5 distinct grid positions per episode and revisits 83% of poses, producing two stable failure modes—walks in circles (left) and looks in loops (right). Bottom-left: action selection distribution (rotation 50.8%, body translation 26.1%, Stop 0.1%). Bottom-middle: enabling chain-of-thought multiplies tokens per response by ∼10× without changing success rate. Bottom-right: removing body translation lifts the model to 80.5%; restricting to it keeps the model at 10.0%.

than VA, with a mean gap of +3.8pp, suggesting past observations in context can hurt foundation models not trained for this setting. When a model invokes Stop, it usually does so at the wrong pose: F-stop exceeds 75% for 11 of 15 configurations and reaches 100% for Qwen3.5-9B (VA). GPT-5 is the exception (0% AO, 27.3% VA): when it commits to Stop, it is usually already at the target pose. Models rarely terminate on their own: for 14 of 15 rows, mean episode length is close to the pertask step budget, so most episodes hit the step limit rather than end with Stop.

Controlled ablation: body translation is a dominant bottleneck. To locate failures, we run two single-room ablations with 200 tasks each under restricted action spaces. In rotate/look, start/target states share position and differ only in yaw and head pitch. In move-only, they share yaw and pitch and differ only in position. Removing bodytranslation actions raises Qwen3.5-9B from 2.8% baseline to 80.5%, whereas allowing only body translation keeps it at 10.0% (Figure 3, bottomright). Results suggest body-translation control is a dominant failure mode in TVR, rather than viewpoint appearance matching alone.

Failure behavior patterns. The full benchmark shows three recurring behavioral patterns consistent with the controlled-ablation result. Per episode, Qwen3.5-9B chooses 34.3 actions on average, yet visits only 3.5 distinct grid positions and returns to 83% of its own poses. Failed trajectories mainly follow two stable patterns: the agent walks in circles, moving back and forth between adjacent cells, or looks in loops, alternating head pitch while staying put. Examples appear at the top of Figure 3.

The action distribution points to the same issue. Among 17,159 actions across the benchmark, rotations account for 50.8%, body translations only 26.1%, and Stop just 0.1% (Figure 3, bottom-left). In practice, the model rotates too often and rarely moves forward or ends the episode.

Enabling Qwen3.5’s native thinking mode does not resolve this behavior. It increases response tokens by roughly an order of magnitude, but success remains unchanged (Figure 3, bottom-middle).

Across model scales, strategies, and memory formats, TVRBench reveals a consistent bottleneck: models struggle to map spatial discrepancies to reliable body translation, not visual matching.

SFT Single Turn Reinforcement Learning

Multi Turn Reinforcement Learning

[Figure 51]

|Prompt|
|---|

TVR RL Dataset

TVR SFT Task

[Figure 52]

Multimodal LLM 🔥

[Figure 53]

[Figure 54]

[Figure 55]

|[Figure 56]<br><br>current|
|---|

|[Figure 57]<br><br>target|
|---|

[Figure 58]

Group sampling

[Figure 59]

Rule-Based Data Annotation

[Figure 60]

[Figure 61]

[Figure 62]

Multimodal LLM 🔥 Environment

[Figure 63]

Rule-based reward

Format reward Accuracy reward

[Figure 64]

COT Annotation

###### Action

[Figure 65]

[Figure 66]

Multiple Trajectories

[Figure 67]

Trajectory Reward ❄Multimodal LLM

[Figure 68]

❄Multimodal LLM KL

Cross Entropy loss

[Figure 69]

[Figure 70]

KL

Group Relative Policy Optimization loss

[Figure 71]

Multimodal LLM 🔥

[Figure 72]

Group Relative Policy Optimization loss

- Figure 4: Post-training pipelines on TVRBench. SFT: supervised fine-tuning on rule-based expert trajectories

(optionally with CoT). Single-turn RL: GRPO on fixed (It,I⋆,a∗t) prompts. Multi-turn RL: GRPO on on-policy rollouts in TVRBench with dense per-step plus terminal reward.

- 5 Can Post-Training Improve Active Viewpoint Control?

SFT Learns Action Mappings from VisualAction History, Not CoT Rationales. Section 4 identified a central bottleneck in TVR: models often fail to map spatial discrepancies to reliable embodied actions, especially body translation. Supervised fine-tuning on expert trajectories substantially improves this discrepancy-to-action mapping across memory formats, with visual-action memory yielding the strongest results.

Setup. We use Qwen3.5-9B as the backbone for post-training. For supervised fine-tuning (SFT), we vary the memory representation defined in Section 3.2, using either action-only (AO) or visualaction (VA) memory, and also vary whether the supervision contains intermediate Chain-of-Thought (CoT) rationales. Training trajectories are produced by a rule-based expert in simulation. For the CoT variants, MiMo-V2.5 (Xiaomi MiMo Team, 2026) provides the intermediate rationales through its API. Appendix C describes the annotation pipeline and dataset statistics.

The best SFT setting, VA-SFT without CoT, reaches 50.8% overall success on TVRBench (Table 2), far above both the untrained Qwen3.5-9B baseline and the strongest closed-source baseline. Performance is especially strong on single-room tasks (Single-easy 82.4%, Single-hard 68.8%), while multi-room performance remains lower (Multi-easy 27.2%, Multi-hard 24.8%), leaving the main room for further improvement.

We further apply Group Relative Policy Optimisation (GRPO) to the SFT checkpoints under two training setups (Figure 4). Single-turn GRPO uses curated single-step prompts and an action-matching reward, while Multi-turn GRPO uses live TVRBench rollouts and an episode-level heuristic reward. We use action-only memory for Single-turn GRPO and visual-action memory for Multi-turn GRPO, because visual-action memory is needed to retain the observation-action history required for trajectory-level optimisation. Appendix D gives the RL data construction procedure, reward definitions, and hyperparameters. All post-training checkpoints are evaluated on the TVRBench test split under the same step budgets as Section 3.2. Table 2 reports per-category success rates and diagnostic metrics.

The SFT ablations show two consistent trends: visual-action memory improves SFT performance, while our CoT rationales do not help. Without CoT, switching from action-only to visual-action memory raises overall success from 44.2% to 50.8%; with CoT, the same switch raises success from 24.8% to 35.6%. Conversely, adding CoT reduces success under both memory formats, from 44.2% to 24.8% with action-only memory and from 50.8% to 35.6% with visual-action memory. Stop calibration follows the same direction: both visual-action variants have F-stop = 0%, whereas

- Table 2: Post-training results on TVRBench. Success rate (%) and diagnostics on the test split; Init names the SFT checkpoint each RL policy starts from; other columns follow Table 1; top-3 per column (excluding untrained baselines): red , green , blue .

Method Mem CoT Init Overall↑ S-e↑ S-h↑ M-e↑ M-h↑ Steps Stop F-stop↓

Qwen3.5-9B AO – – 2.8 6.4 4.8 0.0 0.0 29.9 27.0 89.6 Qwen3.5-9B VA – – 0.0 0.0 0.0 0.0 0.0 34.3 2.4 100.0

AO-CoT-SFT AO yes Qwen3.5-9B 24.8 40.8 38.4 8.8 11.2 30.5 25.4 2.4 AO-SFT AO no Qwen3.5-9B 44.2 61.6 60.8 32.0 22.4 25.9 48.0 7.9 VA-CoT-SFT VA yes Qwen3.5-9B 35.6 68.0 51.2 12.8 10.4 28.9 35.6 0.0 VA-SFT VA no Qwen3.5-9B 50.8 82.4 68.8 27.2 24.8 25.3 50.8 0.0

Single-turn GRPO AO no AO-SFT 26.2 48.8 35.2 12.0 8.8 28.2 33.0 20.6 Multi-turn GRPO (ours) VA no VA-SFT 51.4 81.6 64.0 34.4 25.6 24.9 51.4 0.0

action-only variants still make false Stop decisions in 2.4–7.9% of Stop invocations. Thus, the SFT results identify visual-action memory as the more reliable ingredient for TVR, while CoT supervision is not beneficial under our current annotation scheme.

The degradation suggests that these rationales do not provide useful supervision for this control policy, and may interfere with action learning under the current annotation scheme. Whether CoT supervision tailored specifically to active viewpoint control can help remains an open question.

Trajectory-level GRPO selectively improves multi-room exploration, whereas Single-turn GRPO regresses. Although the aggregate gain over VA-SFT is modest (+0.6pp), the split-level results are more informative. Multi-turn GRPO improves the long-distance multi-room splits, where SFT remains weakest: Multi-easy rises from 27.2 to 34.4 (+7.2pp), and Multi-hard from 24.8 to 25.6 (+0.8pp). The single-room splits remain close to the SFT checkpoint, with Single-easy changing from 82.4 to 81.6 and Single-hard from 68.8 to 64.0. Thus, the benefit of Multi-turn GRPO is selective rather than uniform: it is most visible on the harder multi-room settings, while the stronger single-room performance is not substantially degraded. The final model also keeps F-stop at 0%, suggesting that the multi-room gains do not come at the cost of worse stop calibration.

By contrast, casting the same RL data into single-step action-matching prompts consistently degrades the SFT policy. Starting from AOSFT, Single-turn GRPO reduces overall success by 18.0pp (44.2 → 26.2, Table 2). Starting from AOCoT-SFT, it also reduces success by 9.8 to 15.4pp across KL coefficients β ∈ {0.01,0.05}. Stop calibration deteriorates as well: F-stop rises from 7.9%

at the AO-SFT initialization to 20.6% after Singleturn GRPO. Together, these results suggest that TVR-style active tasks benefit from RL only when the optimization objective matches their closedloop, multi-step structure. Per-step action matching is insufficient here and can degrade the supervised policy, whereas trajectory-level Multi-turn GRPO gives its clearest gains on the long-distance multiroom settings.

Most post-training gains on TVRBench come from visual-action SFT. Multi-turn GRPO adds a smaller, targeted refinement, with its clearest benefit on the long-distance multiroom splits.

### 6 Conclusion

We introduced Target Viewpoint Reproduction (TVR), a closed-loop task for reproducing a target image through embodied movement/reorientation, and TVRBench, spanning scene scale and targetview visual richness. TVRBench exposes a large model–human gap: best closed/open-source models reach 12.0%/7.8% success versus 93.0% humans, with failures mainly in mapping viewpoint discrepancies to reliable body movement. We further build a unified TVR post-training framework covering expert-trajectory SFT, CoT-SFT, Singleturn GRPO, and trajectory-level Multi-turn GRPO. Visual-action SFT plus Multi-turn GRPO lifts a 9B model from 2.8% to 51.4%, while CoT and Single-turn GRPO hurt closed-loop performance. Together, TVR, TVRBench, and the post-training framework provide a compact testbed for improving foundation models that actively perceive and act in 3D.

### Limitations

TVRBench is built entirely in simulation (AI2THOR and ProcTHOR-10k) with a discrete pose grid and an exact-pose success criterion. These choices keep task difficulty controllable and the success signal unambiguous, but our results therefore characterize this setting rather than continuous, tolerance-based viewpoint control in the physical world. Our post-training conclusions also rest on a single 9B open-source backbone, and we have not established how broadly they hold across model families, scales, and other active-perception tasks.

### Ethical Considerations

TVRBench builds on the AI2-THOR and ProcTHOR simulators and on MiMo-V2.5, GPT-4o, GPT-5, and Gemini-3.1-Pro (accessed via API), all used within their stated terms; human performance was collected from five volunteers. We will release TVRBench, the trajectory pipeline, our posttraining checkpoints, and supporting code under permissive open-source licenses. TVR is evaluated entirely in indoor simulation; agents that actively control viewpoints could in principle enable intrusive uses, so any real-world deployment should be paired with domain-specific safety evaluation.

### References

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. 2018. Visionand-language navigation: Interpreting visuallygrounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3674– 3683.

Dhruv Batra, Aaron Gokaslan, Aniruddha Kembhavi, Oleksandr Maksymets, Roozbeh Mottaghi, Manolis Savva, Alexander Toshev, and Erik Wijmans. 2020. Objectnav revisited: On evaluation of embodied agents navigating to objects. arXiv preprint arXiv:2006.13171.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, and 1 others. 2024. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. 2024. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. 2024. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093.

Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Jordi Salvador, Kiana Ehsani, Winson Han, Eric Kolve, Ali Farhadi, Aniruddha Kembhavi, and Roozbeh Mottaghi. 2022. ProcTHOR: Large-Scale Embodied AI Using Procedural Generation. In NeurIPS. Outstanding Paper Award.

Google DeepMind. 2026. Gemini 3.1 Pro. Model Card. Yining Hong, Chunru Lin, Yilun Du, Zhenfang Chen,

Joshua B Tenenbaum, and Chuang Gan. 2023. 3d concept learning and reasoning from multi-view images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9202–9212.

Yining Hong, Jiageng Liu, Han Yin, Manling Li, Leonidas Guibas, Li Fei-Fei, Jiajun Wu, and Yejin Choi. 2026. Esi-bench: Towards embodied spatial intelligence that closes the perception-action loop. Preprint, arXiv:2605.18746.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. 2017. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, and 1 others. 2024. Openvla: An opensource vision-language-action model. arXiv preprint arXiv:2406.09246.

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. 2017. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv.

Juil Koo, Daehyeon Choi, Sangwoo Youn, Phillip Y Lee, and Minhyuk Sung. 2025. Toward ambulatory vision: Learning visually-grounded active view selection. arXiv preprint arXiv:2512.13250.

Jacob Krantz, Stefan Lee, Jitendra Malik, Dhruv Batra, and Devendra Singh Chaplot. 2022. Instancespecific image goal navigation: Training embodied agents to find object instances. arXiv preprint arXiv:2211.15876.

Rong Li, Shijie Li, Lingdong Kong, Xulei Yang, and Junwei Liang. 2025. Seeground: See and ground for zero-shot open-vocabulary 3d visual grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3707–3717.

Zhenyi Liao, Qingsong Xie, Yanhao Zhang, Zijian Kong, Haonan Lu, Zhenyu Yang, and Zhijie Deng. 2025. Improved visual-spatial reasoning via r1-zerolike training. arXiv preprint arXiv:2504.00883.

Fangyu Liu, Guy Emerson, and Nigel Collier. 2023. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 11:635–651.

Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. 2022. Sqa3d: Situated question answering in 3d scenes. arXiv preprint arXiv:2210.07474.

Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, and 1 others. 2024. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16488–16498.

- Qwen Team. 2026a. Qwen3.5: Towards native multimodal agents.
- Qwen Team. 2026b. Qwen3.6-27B: Flagship-level coding in a 27B dense model.
- Qwen Team. 2026c. Qwen3.6-35B-A3B: Agentic coding power, now open to all.

Arijit Ray, Jiafei Duan, Ellis Brown, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, and 1 others. 2024. Sat: Dynamic spatial aptitude training for multimodal language models. arXiv preprint arXiv:2412.07755.

Koya Sakamoto, Taiki Miyanishi, Daichi Azuma, Shuhei Kurita, Shu Morikuni, Naoya Chiba, Motoaki Kawanabe, Yusuke Iwasawa, and Yutaka Matsuo. 2026. E3vs-bench: A benchmark for viewpointdependent active perception in 3d gaussian splatting scenes. arXiv preprint arXiv:2604.17969.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 others. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, Yixuan Li, and Neel Joshi. 2024. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. Advances in Neural Information Processing Systems, 37:75392– 75421.

Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. 2026. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. Advances in neural information processing systems, 38:13569– 13597.

Xiaomi MiMo Team. 2026. Mimo-v2.5-pro. https://huggingface.co/collections/ XiaomiMiMo/mimo-v25.

Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. 2025. Multi-spatialmllm: Multiframe spatial understanding with multi-modal large language models. arXiv preprint arXiv:2505.17015.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. 2025a. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, and 1 others. 2025b. Mmsibench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764.

Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, TaYing Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, and Yi Ma. 2026. Seeing from another perspective: Evaluating multiview understanding in mllms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 12000–12008.

Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, and 1 others. 2025. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25.

Heyang Yu, Yinan Han, Xiangyu Zhang, Baiqiao Yin, Bowen Chang, Xiangyu Han, Xinhao Liu, Jing Zhang, Marco Pavone, Chen Feng, and 1 others. 2025. Thinking in 360 {\deg}: Humanoid visual search in the wild. arXiv preprint arXiv:2511.20351.

Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. 2024. Robopoint: A vision-language model for spatial affordance prediction for robotics. arXiv preprint arXiv:2406.10721.

Kuo-Hao Zeng, Zichen Zhang, Kiana Ehsani, Rose Hendrix, Jordi Salvador, Alvaro Herrasti, Ross Girshick, Aniruddha Kembhavi, and Luca Weihs. 2024. Poliformer: Scaling on-policy rl with transformers results in masterful navigators. arXiv preprint arXiv:2406.20083.

Pingyue Zhang, Zihan Huang, Yue Wang, Jieyu Zhang, Letian Xue, Zihan Wang, Qineng Wang, Keshigeyan Chandrasegaran, Ruohan Zhang, Yejin Choi, and 1 others. 2026. Theory of space: Can foundation models construct spatial beliefs through active exploration? In The Fourteenth International Conference on Learning Representations.

Gengze Zhou, Yicong Hong, and Qi Wu. 2024. Navgpt: Explicit reasoning in vision-and-language navigation with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 7641–7649.

Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, and Achuta Kadambi. 2025. Vlm4d: Towards spatiotemporal awareness in vision language models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8600–8612.

Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, and 1 others. 2025a. Active-o3: Empowering multimodal large language models with active perception via grpo. arXiv preprint arXiv:2505.21457.

Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. 2017. Target-driven visual navigation in indoor scenes using deep reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA), pages 3357–3364. ieee.

Ziyu Zhu, Xilin Wang, Yixuan Li, Zhuofan Zhang, Xiaojian Ma, Yixin Chen, Baoxiong Jia, Wei Liang, Qian Yu, Zhidong Deng, and 1 others. 2025b. Move to understand a 3d scene: Bridging visual grounding and exploration for efficient and versatile embodied navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8120–8132.

### A Appendix Overview

The appendix extends the main paper along five axes:

- • Appendix B: TVRBench construction. The 1:2:3 scene split into SFT, evaluation, and RL pools, the per-category task generation procedure, the nine-action space, the four diagnostic metrics used throughout the paper, and the human evaluation protocol behind the human reference row.
- • Appendix C: SFT data pipeline. Rule-based expert trajectories in simulation, Chain-ofThought annotation with MiMo-V2.5, and the two memory formats—action-only and visualaction—that the SFT ablation crosses.
- • Appendix D: Post-training configuration. Hyperparameters and data construction for supervised fine-tuning, Single-turn GRPO, and Multi-turn GRPO, including the heuristic reward and action mask used in the multi-turn rollouts.
- • Appendix E: Additional quantitative results. A KL ablation for Single-turn GRPO, an RL-from-base bootstrap experiment, and a comparison between per-step matching accuracy and closed-loop episode success.
- • Appendix F: Qualitative examples. Four representative trajectories on TVRBenchtwo failure modes of the untrained 9B (a rotation loop and a walking loop) and two successes of our VA-SFT + Multi-turn GRPO policy (one single-room iTHOR, one multiroom ProcTHOR).

### B TVRBench Construction Details

#### B.1 Scene splits

TVRBench uses 240 distinct indoor scenes split into post-training (SFT), evaluation, and reinforcement-learning (RL) pools at a 1:2:3 ratio applied independently to each scene family to preserve the same family balance across pools. The single-room half draws 120 scenes from AI2THOR, with 30 each from its four scripted room categories—kitchen, living room, bedroom, and bathroom—yielding 20 SFT / 40 evaluation / 60 RL scenes. The multi-room half draws 120 scenes uniformly at random from the training split of ProcTHOR-10k (each a procedurally generated 2–3

room layout with physical wall separation between rooms), partitioned under the same 1:2:3 split. No scene is shared across the three pools, ensuring that held-out evaluation tasks are drawn from environments unseen during both SFT and RL training, so reported results reflect genuine generalisation rather than scene memorisation.

#### B.2 Task generation

Each TVRBench task is a (start, target) pose pair sampled within a single scene and characterised by two independent dimensions: (i) the shortestpath length between start and target on the agent’s discrete pose graph—the minimum number of unit actions a rule-based expert needs to navigate from one to the other, which proxies the spatial extent of the navigation required—and (ii) the segment count seg at the target viewpoint, computed as the number of visible objects excluding structural geometry, the agent itself, and meshes flagged by an internal exclusion list, a value that proxies the visual richness of the target viewpoint. Crossing the two dimensions yields the four task categories used throughout the paper: single-room easy (seg ≥ 9, path length 2–8) and single-room hard (seg ∈ [3,6], path length 2–8), both drawn from AI2-THOR scenes; multi-room easy (seg ≥ 9, path length 10– 20) and multi-room hard (seg ∈ [3,6], path length 10–20), both drawn from ProcTHOR-10k scenes. The intermediate band seg ∈ [7,8] is held out as a gap to keep the easy and hard tiers clearly separated. The total number of generated tasks is 1,600 for SFT (40 per scene over the 40 SFT scenes), 500 for evaluation (125 per category), and 4,800 for RL (40 per scene over the 120 RL scenes).

#### B.3 Action space

At each step the agent selects one of nine discrete actions on AI2-THOR’s discrete pose grid: four agent-frame translations (MoveAhead, MoveBack, MoveLeft, MoveRight) by 0.25m, two body rotations (RotateLeft, RotateRight) by ±45◦ about the vertical axis (refined from the simulator’s 90◦ default for finer viewpoint control), two head pitches (LookUp, LookDown) by ±30◦ within the simulator’s [−30◦,+30◦] horizon range, and a single termination action (Stop). The simulator rejects any action that would result in a collision with scene geometry; in such cases the pose is unchanged but the step still counts against the per-task budget, which discourages blind movement into obstacles. Episodes terminate either when the agent

issues Stop (success requires that this happens at a target-matching pose) or when the step budget is exhausted, the latter counted as a failure.

#### B.4 Diagnostic metrics

Let E = {ei}Ni=1 be a set of N evaluation episodes, with per-episode quantities Si ∈ {0,1} (success per the criterion in Section B.3), stopi ∈ {0,1} (whether Stop was issued), the pose-match indicator mi ∈ {0,1} (whether sT = s⋆, so that Si = stopi mi), Ti (number of actions taken), and the final-step pose errors |∆p|i, |∆θ|i, |∆φ|i defined in Section 3.2. We report:

1 N i

SR =

Si,

1 N i

Steps =

Ti,

stopi (1 − mi) i stopi

F-stop = i

,

1 N i |∆x|i, x ∈ {p,θ,φ}.

|∆x| =

The per-category rates S-e, S-h, M-e, M-h are obtained by restricting the sum to episodes in the respective category (each has N = 125 in the evaluation split), which isolates performance on each difficulty tier. F-stop, the false-stop rate, is conditioned on the episodes that stop: it measures how often Stop is issued at a non-target pose among them, so a model that rarely stops can still have a high F-stop if those few stops are wrong, and a low value should be read together with the Stop rate.

#### B.5 Human evaluation protocol

To establish a human reference point, five volunteers each completed a balanced 100-task subset of the evaluation split, with 25 tasks drawn from each of the four categories so that scene scale and visual richness are equally represented. Participants drove the agent through a single-user web interface that displays the current first-person observation and the target image side by side, and issued the same nine discrete actions available to the models through a fixed keyboard mapping: W/S/A/D for the four translations, Q/E for body rotation, R/F for head pitch, and the space bar for Stop. They were instructed to reproduce the target viewpoint as closely as possible and then press Stop to declare completion. Every run used exactly the same image resolution,

action space, per-task step budget (30 actions for single-room iTHOR tasks and 40 for multi-room ProcTHOR tasks), and pose-matching success criterion (|∆p| ≤ 0.01m, |∆θ| ≤ 1◦, |∆φ| ≤ 1◦) as the model evaluation, so the human and model rows are directly comparable. Participation was voluntary and unpaid, and participants were informed that their anonymous task performance would be used solely as the human reference reported in this paper. The task involves only navigating an indoor simulator and poses no foreseeable risk to participants, so no risk disclaimers were required.

### C SFT Data Pipeline

#### C.1 Rule-based trajectory generation

Expert trajectories for the SFT pool are produced offline by an oracle planner with privileged access to simulator-internal state: the agent’s exact pose, the precomputed reachable-position graph of each scene, and the target pose. This information is unavailable to any of the learned models we evaluate. For each task (s0,s⋆) the planner emits a three-phase action sequence:

- 1. View alignment. Rotate the body and adjust

head pitch from (θ0,φ0) to (θ⋆,φ⋆) using the minimum number of RotateLeft/Right and LookUp/Down actions.

- 2. Navigation. Run Dijkstra’s shortestpath algorithm on the discrete state

space (x,z,θ) from (p0,θ⋆) to (p⋆,θ⋆), where each of the six body-motion actions (MoveAhead/Back/Left/Right, RotateLeft/Right) is a unit-cost edge. The agent is permitted to rotate away from θ⋆ during navigation but must end at θ⋆, which avoids inefficient zig-zag motion toward off-axis targets.

- 3. Termination. Issue Stop.

The planner is deterministic and produces exactly one minimum-action-count trajectory per task, for a total of 1,600 SFT trajectories whose lengths equal the shortest-path lengths used to define task difficulty (Section B.2), so every demonstration is action-optimal by construction. The trajectories serve solely as the supervision target for SFT; the learned policy operates strictly from first-person observations and never receives the privileged state used by the planner.

#### CoT Annotation Prompt

The ground-truth action is: {a*_t}. This action is CORRECT. Your task is to JUSTIFY it—not propose a different one. Even if you would have picked another action, your rationale MUST logically support ‘{a*_t}‘.

Write a SHORT observation-grounded rationale (1–3 sentences):

- • briefly note 1–2 key objects/landmarks visible in the CURRENT view,
- • compare with the TARGET (what’s misaligned: heading, distance, or position),
- • conclude why ‘{a*_t}‘ reduces that gap (do NOT name any other action verb).

Only mention objects you actually see (do not invent or guess).

Output STRICT JSON: {"reasoning": "<your rationale here>"}—no markdown, no prefix like Action: or Rationale:; the value must be a plain prose string.

- Figure 5: Instructions appended to the per-step SFT user message at step t (containing the current observation It, target image I⋆, and action history) when querying MiMo-V2.5 for a chain-of-thought rationale. {a*_t} is the expert action returned by the rule-based planner.

#### C.2 CoT annotation with MiMo-V2.5

For the CoT variants (AO-CoT-SFT and VA-CoTSFT), we augment the rule-based trajectories with intermediate chain-of-thought rationales. For each (current observation It, target image I⋆, expert action a∗t) triple produced by the planner, we prompt the MiMo-V2.5 model (accessed via API) to write a short, observation-grounded justification of why a∗t is correct, keeping every rationale consistent with the optimal action label. The prompt (Figure 5) instructs the model to (i) reference 1–2 visible landmarks in the current view, (ii) compare them against the target and identify the misalignment dimension (heading, distance, or position), and (iii) explain how the given action reduces that gap, without naming any alternative action. The 1–3 sentence cap is deliberately tight: SFT trajectories preserve the full multi-turn history of observations and reasoning across up to 30–40 steps, so any per-step rationale length is multiplied by the trajectory length when accumulated in context. We accept the returned rationale only if it parses as the requested JSON object, discarding any malformed response.

#### C.3 Two memory formats

The two memory representations produce structurally different SFT samples. Under action-only memory each trajectory step becomes an indepen-

dent single-turn sample whose user message contains the current observation It, the target image I⋆, and a short action-history text; under visualaction memory the entire trajectory is packed into a single multi-turn sample whose turns accumulate end to end, exposing every past observation in context at every step, so action-only keeps sequences short while visual-action preserves the full visual memory. For the CoT variants, each model response is optionally prefixed by a chain-of-thought rationale wrapped in <think>. . . </think> tags. Figures 6 and 7 give concrete schematics of each format.

### D Post-Training Configuration

Compute budget. A single supervised finetuning run uses 4 NVIDIA H100 GPUs for roughly 6 hours; Multi-turn (online) GRPO uses 8 NVIDIA H200 GPUs for roughly 10 hours; and Single-turn (offline) GRPO uses 8 H200 GPUs for roughly 4 hours.

#### D.1 Supervised fine-tuning

All four SFT variants fine-tune Qwen3.5-9B with full-parameter updates and a frozen vision encoder that preserves its pretrained visual representations. We use AdamW with bf16 precision, learning rate 1 × 10−5 under a cosine schedule with 10% linear warmup, image resolution capped at 262144

#### Action-Only Memory Sample (per-step) SYSTEM

You are a navigation agent in an indoor environment. Your task is to navigate and adjust your viewpoint to PRECISELY match a target image. You must match the exact position, orientation, and camera angle—the goal is for your observation to look identical to the target.

Available actions:

- - MoveAhead: Move forward 0.25m
- - MoveBack: Move backward 0.25m
- - MoveLeft: Move left 0.25m
- - MoveRight: Move right 0.25m
- - RotateRight: Rotate clockwise 45°
- - RotateLeft: Rotate counter-clockwise 45°
- - LookUp: Tilt camera up 30°
- - LookDown: Tilt camera down 30°
- - Stop: Declare that you have reached the target viewpoint

You will receive: (1) your current observation; (2) your recent action history (if available); (3) the target viewpoint you need to match. Use your action history to avoid repeating ineffective actions. You MUST respond in exactly this format: “Action: <action name>”.

USER Your CURRENT observation: [image: It] Your recent actions (5 steps):

- - Step 5: MoveAhead
- - Step 6: MoveAhead
- - Step 7: RotateRight
- - Step 8: MoveAhead
- - Step 9: MoveAhead

TARGET viewpoint you must match: [image: I⋆] Valid actions at this step: MoveAhead, MoveBack, RotateLeft, RotateRight, LookUp, LookDown, Stop.

GPT <think> The target view shows the kitchen counter centred and close, while my current view has it to the right of centre. Rotating right will align my heading with the target. </think> Action: RotateRight.

- Figure 6: SFT sample format under action-only memory. Each trajectory step becomes one independent single-turn sample with a textual recent-action history; no past observations remain in context. The <think>. . . </think> prefix appears only in CoT variants. “Valid actions at this step:” lists the actions the simulator allows at the current pose. The 1,600 SFT trajectories expand to ≈ 20,700 such per-step samples.

pixels, per-device batch size 1 with gradient accumulation 8 across 4 GPUs (effective batch 32), and DeepSpeed ZeRO-2 with gradient checkpointing. Training runs for 3 epochs on the AO variants (AOSFT, AO-CoT-SFT) and 5 epochs on the VA variants (VA-SFT, VA-CoT-SFT), whose far smaller sample count warrants the additional passes.

#### D.2 Single-turn GRPO

Single-turn GRPO optimises a per-step action policy on a parquet dataset of (It,I⋆,a∗t) prompts, built by flattening the SFT trajectories into independent (state, expert-action) tuples, so each action is trained in isolation from its trajectory context. The policy is initialised from the AO-SFT check-

Visual-Action Memory Sample (multi-turn, single trajectory) SYSTEM (identical to the prompt above)

- USER (turn 0)

Your CURRENT observation: [image: I0] TARGET viewpoint you must match: [image: I⋆] Valid actions at this step: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, Stop.

- GPT (turn 0) <think> The target view shows a kitchen counter on the left; my current view shows it diagonally ahead. I should turn left first to align my heading. </think> Action: RotateLeft.

USER (turn 1) Your CURRENT observation: [image: I1] TARGET viewpoint you must match: [image: I⋆] Valid actions at this step: MoveAhead, MoveBack, RotateLeft, RotateRight, LookUp, LookDown, Stop.

- GPT (turn 1) <think> Counter now sits straight ahead but the target view is closer. I need to move forward. </think> Action: MoveAhead.

USER (turn 2) Your CURRENT observation: [image: I2] TARGET viewpoint you must match: [image: I⋆] Valid actions at this step: MoveAhead, MoveBack, RotateLeft, RotateRight, LookUp, LookDown, Stop.

- GPT (turn 2) <think> Closer, but I am still further than the target view; one more step ahead. </think> Action: MoveAhead.

USER (turn 3) Your CURRENT observation: [image: I3] TARGET viewpoint you must match: [image: I⋆] Valid actions at this step: MoveBack, RotateLeft, RotateRight, LookUp, LookDown, Stop.

- GPT (turn 3) <think> The view now matches the target in both heading and distance. Terminate. </think> Action: Stop.

- Figure 7: SFT sample format under visual-action memory. The entire trajectory is packed into a single multi-turn sample; all past observations remain in context at every step (the SYSTEM prompt is identical to Figure 6). The <think>. . . </think> prefix appears only in CoT variants. The trajectory shown is a 4-step example; in TVRBench, trajectories range from a few up to roughly 30–40 steps. This yields exactly 1,600 multi-turn samples.

point, and we inherit the GRPO implementation from verl (Shao et al., 2024).

GRPO objective. For each prompt q, we sample a group of G responses {oi}Gi=1 from the current

rollout policy πold and score each with a scalar reward ri. The group-relative advantage centres ri on the group mean and normalises by the group

standard deviation,

ri − r¯ σr

Ai =

,

r¯ = G1 j rj, σr = std({rj}), and is broadcast to every token of oi. The GRPO objective adopts the PPO-style clipped surrogate over this advantage, together with a KL anchor against the SFT reference πref,

J (θ) = Ei,t min ρtAi, clip(ρt,1−ϵ,1+ϵ)Ai −β DKL[πθ ∥πref],

where ρt = πθ(oi,t | q,oi,<t)/πold(oi,t | q,oi,<t) and DKL is estimated with the unbiased lowvariance K3 form.

Reward. The per-response reward ri is a gated combination of format validity and action correctness,

ri = 1[format(oi)] · 0.1 + 0.9 · 1[a(oi) = a∗t] , so a response that drops the required “Action: <name>” format receives 0, a correctlyformatted but wrong action receives 0.1, and a correctly-formatted matching action receives 1.0; the floor still rewards valid formatting even when the chosen action is wrong.

Hyperparameters. Group size G = 8 rollouts at temperature 0.9 and top-p 0.95. AdamW with learning rate 1 × 10−6, gradient clip 1.0, no entropy bonus. GRPO clip threshold ϵ = 0.2 (verl default); the KL coefficient is β ∈ {0.01, 0.05}, with β = 0.01 reported as the default row in Table 2 and β = 0.05 included in the KL ablation (Appendix E.1).

#### D.3 Multi-turn GRPO

Multi-turn GRPO optimises an episode-level policy by rolling out trajectories in the live TVRBench simulator, learning on-policy from closed-loop interaction. For each task (a (start,target) pose pair drawn from a dedicated 4,800-task RL split), the policy is rolled out G times against the simulator; each rollout produces a trajectory

##### τ(i) = (obs0, a1,r1, ..., aTi,rTi,obsTi),

with per-step rewards rt given below. We initialise from the VA-SFT checkpoint, inherit the GRPO core from verl (Shao et al., 2024), and use a custom agent loop that interleaves model-generated actions with simulator observations.

Per-step reward. The reward at step t decomposes additively into four components,

rt = −cstep + rfmt(t) + rprog(t) + rterm(t) ,

with: (i) a constant step penalty cstep = 0.01 to encourage efficiency; (ii) a format term rfmt(t) = +0.005 if the model output parses to a valid action, −0.01 otherwise; (iii) an asymmetric progress term that only rewards strict improvements in the running minimum pose distance,

rprog(t) = max 0, d(mint−1) − dt ,

where d(mint−1) = mins≤t−1 ds tracks the best distance seen so far, so backtracking toward already-

visited poses earns no reward; and (iv) a terminal term rterm(t) = +1.0 when the agent issues Stop at the target pose, −0.5 when it issues Stop at a non-target pose, and 0 otherwise, so a premature or mistaken Stop is actively penalised rather than merely left unrewarded. The pose distance is a weighted geodesic

dt = ∥pt − p⋆∥2 + 0.25nrott + 0.25nhort ,

where nrott = min(|∆θt|,360◦−|∆θt|)/45◦ and nhort = |∆φt|/30◦ are the integer numbers of rotation and head-pitch actions needed to align with the target, weighted so that one such action contributes the same as one 0.25m translation step.

Trajectory-level advantage. The scalar reward attributed to each rollout is the sum of its per-step rewards,

Ti

rt(i),

R(i) =

t=1

and the group-relative advantage is computed at the trajectory level and broadcast to every assistant token of τ(i),

R(i) − R¯ σR

A(i) =

,

R¯ = G1 j R(j), σR = std({R(j)}).

Token-masked objective. Because each trajectory interleaves environment observations with model-generated actions, only assistant tokens carry gradients, so the policy is never trained to predict the simulator’s observations. We mask the

GRPO loss accordingly,

J (θ) = E mt · min ρtA(i),

clip(ρt,1−ϵ,1+ϵ)A(i) −β DKL(πθ ∥πref),

where mt = 1[tokent ∈ assistant] and πref is the VA-SFT initialisation.

Hyperparameters. Group size G = 8 trajectories per task, maximum trajectory length Tmax = 30 turns, with 8 parallel environment instances per rollout worker. AdamW with learning rate 1 × 10−7—an order of magnitude smaller than the Single-turn case to preserve the stronger VASFT initialisation—and gradient clip 1.0. GRPO clip ϵ = 0.2 and KL coefficient β = 0.01, both inherited from the Single-turn configuration (Appendix D.2).

### E Additional Quantitative Results

#### E.1 KL ablation for Single-turn GRPO

We expand the claim from Section 5 that “even the most KL-conservative setting still regresses below its SFT initialisation” with the per-category breakdown in Table 3. Starting from AO-CoT-SFT, Single-turn GRPO drops by −9.8pp at β = 0.05 and by −15.4pp at the more permissive β = 0.01. Both settings also degrade the stop calibration: F-stop rises from 2.4% at the SFT init to 10.9% (β = 0.05) and 23.5% (β = 0.01), so both success rate and stop calibration worsen monotonically as the KL leash is loosened.

Table 3: KL ablation for Single-turn GRPO initialised from AO-CoT-SFT. Both β settings regress below the SFT init and worsen F-stop calibration.

Config Overall↑ S-e↑ S-h↑ M-e↑ M-h↑ AO-CoT-SFT (init) 24.8 40.8 38.4 8.8 11.2

+β = 0.05 15.0 32.8 19.2 4.8 3.2 +β = 0.01 9.4 20.8 12.8 1.6 2.4

#### E.2 RL from a base initialisation

We also evaluated both GRPO variants without any SFT warm-up, starting directly from the untrained Qwen3.5-9B (Table 4). Single-turn GRPO improves the AO baseline only marginally (2.8 → 3.6, +0.8pp); Multi-turn GRPO, by contrast, lifts the VA baseline from 0% to 26.2% overall and achieves perfect stop calibration (F-stop = 0%).

Trajectory-level on-policy RL alone produces a workable policy from scratch, whereas per-step RL does not, because the shaped progress reward supplies a learning signal even to a near-random initial policy.

Table 4: GRPO from a base (no SFT) initialisation. Multi-turn GRPO bootstraps a workable policy from the untrained Qwen3.5-9B (VA); Single-turn GRPO does not.

Config Overall↑ S-e↑ S-h↑ M-e↑ M-h↑ Qwen3.5-9B base (AO) 2.8 6.4 4.8 0.0 0.0

+ Single-turn GRPO 3.6 11.2 3.2 0.0 0.0 Qwen3.5-9B base (VA) 0.0 0.0 0.0 0.0 0.0 + Multi-turn GRPO 26.2 52.8 31.2 15.2 5.6

#### E.3 Per-step versus closed-loop accuracy

We check whether the Single-turn GRPO closedloop regression is masked by per-step gains. Replaying the validation split of the per-step prompt dataset (500 prompts × 8 rollouts) through the AOCoT-SFT + Single-turn GRPO checkpoint (β = 0.01, step 100) yields a per-step action-matching accuracy of 72.1%, with format validity 99.98% and an average per-step reward of 0.749; a parallel run at β = 0.001 produces a comparable per-step accuracy of 0.78. The same β = 0.01 checkpoint, however, reaches only 9.4% on the closed-loop benchmark (Table 2), a gap of over 60 points between per-step matching and episode success. The gap reflects compounding error: small per-step inaccuracies accumulate over the roughly 30 decisions per episode, and the policy never learns recovery, as it is trained only on expert-conditioned states, not the off-expert states it visits at test time. A per-step matching objective therefore does not translate into end-to-end trajectory success.

### F Qualitative Examples

We complement the aggregate numbers in Sections 4–5 with four end-to-end TVRBench traces: two failures of the untrained Qwen3.5-9B (one rotating in place, one walking in a short loop) and two successes of our VA-SFT + Multi-turn GRPO policy (single-room iTHOR and multi-room ProcTHOR). Each trace shows an orthographic floor plan with start (yellow), target (red), and final pose (blue), the full path, and first-person frames sampled along the trajectory.

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

target

Top Down

start final

[Figure 77]

[Figure 78]

internal target

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Step 1

Step 2

Step 3

Step 4

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Action: Rotate Right

Action: Rotate Right

Action: Move Ahead

Action: Rotate Left

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Step 5

Step 6

Step 7

Step 8

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Action: Move Ahead

Action: Rotate Right

Action: Rotate Right

Action: Move Ahead

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Step 9

Step 10

Step 11

Step 12

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Action: Rotate Left

Action: Rotate Right

Action: Rotate Left

Action: Rotate Right

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Step 13

Step 14

Step 15

Step 16

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Action: Rotate Left

Action: Rotate Left

Action: Rotate Right

Action: Rotate Right

Action: Rotate Right

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Step 17

Step 18

Step 19

Step 20

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Action: Rotate Right

Action: Rotate Left

Action: Rotate Right

Action: Rotate Right

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Step 21

Step 22

Step 23

Step 24

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Action: Rotate Left

Action: Rotate Right

Action: Rotate Right

Action: Rotate Right

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Step 25

Step 26

Step 27

Step 28

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Action: Rotate Left

Action: Rotate Right

Action: Rotate Right

Action: Rotate Right

- Figure 8: A failure case from the untrained Qwen3.5-9B. With action-only memory, the agent advances twice in its first four steps and then issues 35 consecutive Rotate actions at the same position until the 40-step budget runs out. The action history alone cannot tell the policy it has already tried—and rejected—each yaw, so the same micro-decision repeats indefinitely. For space, the panels show only the first 28 of 40 steps, which continue the same in-place rotation.

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

target

Top Down

start final

[Figure 140]

[Figure 141]

internal target

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Step 1

Step 2

Step 3

Step 4

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Action: Rotate Right

Action: Rotate Right

Action: Look Down

Action: Rotate Right

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Step 5

Step 6

Step 7

Step 8

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Action: Rotate Left

Action: Rotate Right

Action: Rotate Left

Action: Rotate Right

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Step 9

Step 10

Step 11

Step 12

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Action: Rotate Left

Action: Move Right

Action: Move Ahead

Action: Rotate Right

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Step 13

Step 14

Step 15

Step 16

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Action : Move Right

Action: Move Ahead

Action: Rotate Right

Action: Rotate Right

Action: Rotate Left

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Step 17

Step 18

Step 19

Step 20

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Action: Move Right

Action: Rotate Left

Action: Rotate Right

Action: Rotate Right

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Step 21

Step 22

Step 23

Step 24

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Action: Move Right

Action: Rotate Right

Action: Rotate Right

Action: Rotate Right

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Step 25

Step 26

Step 27

Step 28

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Action: Rotate Left

Action: Move Back

Action: Move Ahead

Action: Move Ahead

- Figure 9: A failure case from the untrained Qwen3.5-9B. On a single-room iTHOR scene the agent shuttles between a handful of cells—issuing 12 Move actions among only four distinct positions—without ever closing the gap to the target. The action history alone cannot register that these cells have already been visited, so the same short walking loop repeats. For space, the panels show only the first 28 of 30 steps.

###### A single-room success from VA-SFT + Multi-turn GRPO

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

target start final

Top Down

[Figure 203]

[Figure 204]

internal target

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Step 1

Step 2

Step 3

Step 4

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Action: Rotate Right

Action: Rotate Right

Action: Rotate Right

Action: Move Back

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Step 5

Step 6

Step 7

Step 8

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Action: Move Back

Action: Move Back

Action: Move Left

Action: Move Back

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Step 9

Step 10

Step 11

target

[Figure 225]

[Figure 226]

[Figure 227]

Action: Rotate Left

Action: Move Back

Action: Move Left

- Figure 10: A single-room success from VA-SFT + Multi-turn GRPO. On a single-room iTHOR scene, the policy translates and rotates to align with the target view within a handful of steps and terminates with Stop at the correct pose. Visual-action memory lets each step condition on the actual observation history, so the model no longer revisits previously tried yaws.

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

A Multi-room success from VA-SFT + Multi-turn GRPO

target start final

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

internal target

Action: Rotate Right

[Figure 245]

Step 1

Action: Move Ahead

[Figure 246]

Step 2

Action: Move Ahead

[Figure 247]

Step 3

Action: Move Left

[Figure 248]

Step 4

Action: Move Left

[Figure 249]

Step 5

Action: Move Ahead

[Figure 250]

Step 6

Action: Move Left

[Figure 251]

Step 7

Action: Move Ahead

[Figure 252]

Step 8

Action: Move Ahead

[Figure 253]

Step 9

Action: Move Ahead

[Figure 254]

Step 10

Action: Move Ahead

[Figure 255]

Top Down

target

- Figure 11: A multi-room success from VA-SFT + Multi-turn GRPO. On a multi-room ProcTHOR scene, the policy traverses the layout across rooms and aligns with the target view before issuing Stop. Multi-room tasks are where the SFT initialisation alone is weakest (Table 2); traces like this illustrate where Multi-turn GRPO adds its largest gain over VA-SFT.

