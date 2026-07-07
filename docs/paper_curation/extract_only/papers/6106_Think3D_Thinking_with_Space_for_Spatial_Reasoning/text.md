# arXiv:2601.13029v3[cs.CV]17Mar2026

## Think3D: Thinking with Space for Spatial Reasoning

Zaibin Zhang1∗, Yuhan Wu1∗, Lianjie Jia1∗, Yifan Wang1, Zhongbo Zhang1, Yijiang Li2†, Binghao Ran1, Fuxi Zhang1, Zhuohan Sun1, Zhenfei Yin3, Lijun Wang1, Huchuan Lu1

1 Dalian University of Technology, 2 University of California San Diego, 3 University of Oxford dlutzzb@gmail.com, {tracy1252684562,jialianjie}@mail.dlut.edu.cn, ljwang@dlut.edu.cn ∗ Equal contribution † Project leader

Abstract. While contemporary Vision-Language Models (VLMs) excel at 2D visual understanding, they remain constrained by a passive, 2Dcentric paradigm that severely limits genuine 3D spatial reasoning. To bridge this gap, we introduce Think3D, a novel framework that equips VLM agents with interactive, 3D chain-of-thought reasoning capabilities. By integrating a suite of 3D manipulation tools, Think3D transforms passive perception into active spatial exploration, closely mirroring human geometric reasoning. We demonstrate that Think3D acts as a highly effective zero-shot plug-in for state-of-the-art closed-source models (e.g., GPT-4.1, Gemini 2.5 Pro), yielding absolute performance gains of +7.8% on BLINK Multi-view and MindCube, and +4.7% on VSI-Bench. Furthermore, to optimize tool-use in smaller open-weight models, we propose Think3D-RL, a reinforcement learning paradigm designed to autonomously learn spatial exploration strategies. When applied to Qwen3-VL-4B, Think3D-RL amplifies the performance gain from a marginal +0.7% to a substantial +10.7%. Notably, this RL formulation induces an exploration policy that qualitatively aligns with the sophisticated behavior of much larger models, entirely circumventing the need for costly operation-trajectory annotations. Ultimately, Think3D establishes tool-augmented active exploration as an effective paradigm for unlocking human-like 3D reasoning in multimodal agents. Code, models, and data are available at https://github.com/zhangzaibin/spagent.

Keywords: Vision Language Model · Spatial Intelligence · 3D

### 1 Introduction

Understanding and interacting with the physical world has long been a fundamental objective of vision-language models (VLMs) [4, 14, 21]. Achieving this objective necessitates spatial intelligence—the ability to reason about geometry, viewpoint, and spatial relationships [18,69,76].

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

| |
|---|

[Figure 5]

[Figure 6]

[Figure 7]

| |
|---|

Corp Image [120, 30,….]

Corp Image [120, 30,….]

[Figure 8]

2D Manipulation Yes, the clock is to the

[Figure 9]

2D Manipulation

VLM

left of the laptop.

Multi Rounds

Think with Image

[Figure 10]

[Figure 11]

|[Figure 12]|
|---|

[Figure 13]

|[Figure 14]<br><br>|
|---|

Select Camera 3

Select Camera 4

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Multi-view images

3D Reconstruction

Drag Global View (A:10, E: 30)

Drag Global View (A:0, E: 60)

[Figure 21]

[Figure 22]

From the 3D top-down

3D Manipulation

3D Manipulation

VLM

view, the answer is ……

Multi Rounds

Think with 3D Space

- Fig. 1: Comparison between prior “think with image” [46,84] and our “think with 3D space”. While the former reasons over 2D content by manipulating images, our method operates directly within 3D point cloud space for spatial understanding.

Despite remarkable progress in visual understanding, current VLMs remain powerful yet fundamentally 2D analyzers. Their performance drops sharply on tasks that require spatial reasoning, such as multi-view understanding and route planning. For instance, although recent models achieve near human-level performance on comprehensive benchmarks like MMMU [77], they still lag far behind humans on tasks that demand genuine 3D reasoning [69,75].

Several main research directions have emerged to bridge this gap. The first direction aims to internalize spatial knowledge by training on large-scale and spatially diverse datasets [8, 17, 22, 52, 86]. This approach requires enormous computation and can degrade general reasoning ability. The second direction, often referred to as think with image [45, 46, 81, 84, 88], enables models to call external tools (e.g., zoom [84], crop [81], depth estimation [88]) to enhance perception. However, these 2.5 operations primarily provide shallow spatial cues, such as relative depth, object ordering, or counting, and they do not support deeper reasoning across multiple views or 3D geometry [36,65]. By comparison, humans naturally manipulate consistent 3D representations through operations such as dragging and rotation to support comprehensive spatial reasoning. Inspired by this cognitive process, we ask: Can VLMs “think” with 3D space as humans do?

Recent advances in 3D reconstruction [23,56,59] make this possible. These models can estimate camera poses and reconstruct 3D point clouds from videos or multi-view images, providing a geometric foundation for explicit spatial reasoning. Building on this foundation, we propose Think3D—a framework that

enables VLMs to actively interact with reconstructed 3D point clouds and reason in a spatial manner through thinking with 3D space.

Effective 3D reasoning requires a consistent reference frame. We use estimated camera poses as anchors so the model can interpret rotations and directions consistently, avoiding ambiguous spatial manipulations. With this design, the agent can choose a camera, select a rotation, and decide where to explore next, while switching between a global view (overall layout) and a local view (fine-grained details) to capture both coarse and fine cues. The process is inherently iterative: the model repeatedly interacts with the reconstructed 3D scene, observes new views, and refines its understanding step by step. Through this iterative reasoning process, Think3D develops a coherent spatial representation, mirroring how humans explore 3D space.

We find that the effectiveness of spatial exploration is strongly correlated with the intrinsic reasoning capability of VLMs. Frontier models such as GPT-4.1 [2] and Gemini-2.5-Pro [14] tend to generate diverse and semantically meaningful viewpoints, whereas less capable models often drift toward redundant or even misleading camera poses, which ultimately limits spatial understanding. To narrow this gap, we propose a reinforcement learning approach, Think3D-RL, that enables models to autonomously discover effective exploration policies. Importantly, Think3D-RL relies only on final task rewards, without supervision over how the model should manipulate the 3D scene. During training, the model conducts multi-round spatial exploration, and the reward signal reinforces trajectories that lead to stronger downstream performance. Through this reward-driven process, the model progressively learns when and how to interact with the 3D environment, converging to substantially more informative viewpoint manipulation strategies. As a result, models exhibit increasingly consistent exploration behaviors that more closely match those of frontier VLMs, which leads to substantial improvements across diverse spatial reasoning benchmarks.

We evaluate Think3D on three challenging benchmarks (BLINK Multi-view [19],

MindCube [75], and VSI-Bench [69]) and observe consistent improvements across all tasks. For closed-source models such as GPT-4.1 and Gemini-2.5-Pro, we apply Think3D in a training-free manner, yielding an average +7.8% gain on BLINK Multi-view and MindCube and an additional +4.7% improvement on VSI-Bench. For open-source models, we introduce Think3D-RL; on Qwen3-VL4B [3], the benefit of tool usage rises from +0.7% before RL to +10.7% after RL, demonstrating that learned exploration strategies strengthen the model’s ability to extract informative 3D viewpoints and improve reasoning performance.

Our main contributions can be summarized as follows:

- – A new perspective on spatial reasoning. We frame spatial reasoning as an active 3D exploration process, referred to as Think with 3D Space, rather than conventional passive 2D perception.
- – A framework for explicit 3D interaction. We design Think3D, allowing the VLM-based agent to manipulate point clouds through camera-based reference actions and iterative spatial reasoning chains.

- – Reinforcement learning for spatial exploration. We formulate the model’s acquisition of viewpoint and action selection as an RL process, enabling it to develop efficient 3D exploration strategies that enhance reasoning performance across spatial benchmarks.

### 2 Related Work

- 2.1 VLMs for Spatial Reasoning

Recent advances in Vision Language Models (VLMs) have substantially improved spatial reasoning, which is a key capability for understanding and interacting with the physical world. This progress is driven by more capable models [12,25,29,42,54,73] and comprehensive benchmarks [6,12,13,35,65,69]. Methods such as VLM-3R [17], SpatialRGPT [12], and SpatialVLM [8] incorporate 3D reconstruction, depth cues [39], and large-scale 3D spatial VQA data [5,78] to improve quantitative spatial reasoning. Recent works further strengthen the coupling between perception and reasoning through spatial prompting [25,29,36, 48,49,78], mental simulation [11,25], visual chain-of-thought or RL-based reasoning [16,42,60,61], and explicit visual grounding [66]. In robotics, systems such

- as RoboBrain [22,51], Gemini Robotics [1,52], and RoboRefer [85] extend these capabilities to embodied interaction and precise 3D spatial grounding, and evaluation often uses standardized spatial benchmarks such as VSI-Bench [69] and MindCube [13,65,75]. In navigation, a growing body of work studies how VLMs perceive, plan, and act in 3D environments [80,87], often by coupling visual understanding with mapping, route planning, and embodied decision making. More recent works [10,32] have also explore code-driven use of 3D model outputs to improve spatial intelligence with an emphasis on task decomposition rather than direct spatial reasoning. Another concurrent work related to ours is [71], which uses video generative models to imagine the 3D spatial space. Our Think3D differs from [71] in two aspects. First, [71] selects the exploration trajectory with beam search, whereas Think3D empowers the VLMs with the ability to actively plan when and how to explore the 3D space with 3D manipulation tools. Second, Think3D performs exploration on reconstructed 3D point clouds, thereby avoiding the hallucinations introduced by video generative models. Overall, Think3D offers a more faithful paradigm that allows models to reason about 3D space in a manner more aligned with human geometric reasoning.

- 2.2 VLM tool calling

The efficacy of VLMs is further enhanced by tool calling, where the model leverages external tools via prompting or code generation, as in HuggingGPT [44] and related systems [47,63,73,81]. For long-horizon or high-complexity problems, agent-based systems have been applied to long-video understanding [7,48,72,79], high-resolution image analysis [24,70,89], and medical diagnosis [30,33]. OpenThinkImage [45] provides a unified platform for tool-augmented vision-language

models, while others [20,28,31,50,55,68,89] train VLMs to use specific toolsets through fine-tuning. Reinforcement learning (RL) has become a central paradigm for tool-use and reasoning policies [9,15,45,67,83,88]. In particular, DeepEyes [84] promotes “think with images”, enabling models to leverage internal visual reasoning without external tools and directly inspiring our design.

##### 2.3 3D Reconstruction

In the parallel field of computer vision, 3D reconstruction from 2D images has seen significant breakthroughs, largely driven by transformer-based architectures [40]. DUSt3R [58] introduces a novel paradigm for multi-view 3D reconstruction that does not require predefined camera poses. Building on this, MASt3R [26] enhances the process by regressing dense local feature maps to produce metric-scale reconstructions. VGGT [56], a feed-forward neural network, is capable of directly inferring a comprehensive set of 3D scene attributes—including camera parameters, depth maps, and point tracks—from multiple views in a single forward pass. Methods like CUT3R [57], MapAnything [23], and Pi3 [59] further support continual reconstruction, multi-task metric 3D geometry, and permutation-equivariant visual geometry, providing versatile backbones for our 3D spatial reasoning framework.

### 3 Think3D for Spatial Reasoning

Response 3DManipulationToolkit Response 3DManipulation

Response Answer GT

[Figure 23]

[Figure 24]

Toolkit

Reward Manager

Vision-Language-Model

| | | | |
|---|---|---|---|
|𝑅𝑎𝑛|𝑠 𝑅| |𝑓𝑚𝑡|
| | | | |

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

If I am standing

- at image4, what is behind me?

GRPO

[Figure 29]

[Figure 30]

question Multi Image

Rendered Image Rendered Image

Response

[Figure 31]

3D ManipulationToolkit

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

3D Reconstruction

3D Manipulation

Camera id

| |
|---|

[Figure 37]

|Angle (∆𝛼,∆𝛽)|
|---|

[Figure 38]

Global Mode

[Figure 39]

[Figure 40]

Global/Ego

[Figure 41]

Ego Mode

[Figure 42]

[Figure 43]

Rendered Image

Multi Image

- Fig. 2: The Think3D pipeline. The VLM interacts with the 3D scene through iterative calls to the 3D Manipulation Toolkit, issuing viewpoint-manipulation actions that control camera pose and rendering parameters. Each rendered image is appended to the agent’s memory and informs the next reasoning step, forming a repeated cycle of observe → manipulate → reflect.

##### 3.1 Framework Overview

As illustrated in Figure 2, Think3D equips a VLM with the ability to explore and reason directly in 3D via a multi-turn observe → manipulate → reflect loop. Given multi-view images {It}Tt=1 and a query q, the VLM autonomously decides whether to invoke the 3D reconstruction tool to obtain a 3D point cloud and camera poses. During the subsequent 3D interaction process, the VLM is able to iteratively manipulate the point cloud and observe the 3D environment from novel views. By progressively accumulating complementary geometric observations, the VLMs form an explicit 3D chain of thought, facilitating structured spatial exploration that cannot be achieved using static 2D inputs alone. We run the loop for at most K iterations (default K=3), with at most one reconstruction call per query and at most K rendering calls. The agent can terminate early by issuing a STOP action when it deems the evidence sufficient. The above 3D interaction process is powered by the following three key components of Think3D. We present the details in the subsequent sections.

- – 3D Manipulation Toolkit integrates a suite of callable 3D tools, providing the agent with flexible and expressive control for exploring the 3D environment.
- – Spatial Reasoning Agent performs 3D interactions by calling 3D manipulation tools and reasoning over the geometric observations.
- – Think3D-RL Reinforcement Learning Module optimizes multi-step 3D exploration policy through tool calling, trained with Group Relative Policy Optimization (GRPO) [43].

##### 3.2 3D Manipulation Toolkit

Under the Think3D framework, a suite of callable 3D tools enables flexible agentic 3D manipulation and exploration, featuring three core functionalities: 3D reconstruction, 3D transformation, and novel-view rendering.

###### 3D Reconstruction: Given multi-view images {It}Tt=1, a 3D point cloud and the corresponding camera poses can be estimated using Pi3 [59]. Each camera is represented as

Ct = (Kt,Rt,tt), (1)

where Kt ∈ R3×3 denotes the intrinsic matrix, Rt ∈ SO(3) denotes the rotation matrix, and tt ∈ R3 represents the camera center in world coordinates. Here, t indexes the input views. Depth and confidence predictions are fused across views to obtain a clean colored point cloud:

X = {(xn,cn)}Nn=1, (2) where xn is the 3D location and cn is the RGB color.

- 3D Transformation To enable flexible 3D exploration, the agent manipulates the reconstructed 3D point cloud to select informative viewpoints. At each step, it predicts: (i) a discrete camera index i ∈ {1,...,T}, (ii) a pair of rotation angles (∆α,∆β) specifying horizontal (azimuth) and vertical (elevation) rotations, and (iii) a binary transformation mode m ∈ {global,ego} indicating whether to use a global or an ego-centric view. For m = global, we define a global center c as the centroid of X and apply a scene-centric similarity transform, i.e., we rotate (and optionally scale) the scene around c while keeping the camera fixed:

x′n = s∆R(∆α,∆β)(xn − c) + c, (3) where ∆R(∆α,∆β) ∈ SO(3) is induced by the predicted angles and s controls the zoom level of the global view (we set s=1 by default). This yields an updated point cloud Xg = {(x′n,cn)}Nn=1. We then render Xg using the selected camera Ci to provide a consistent global overview. For m = ego, we keep the point cloud fixed and apply a camera-centric rotation around the anchor camera center. Given the selected anchor camera Ci, we construct a virtual camera:

Cnew = (Ki, ∆R(∆α,∆β)Ri, ti), (4)

where ∆R(∆α,∆β) ∈ SO(3) is an incremental rotation applied to the camera orientation, while the camera center remains fixed at ti. When ∆R = I, the virtual camera Cnew reduces to the original camera Ci.

Novel View Rendering In the global mode, we render the transformed point cloud Xg with the anchor camera Ci to obtain a global view of the 3D scene. In the ego-centric mode, we emulate a first-person perspective by restricting X to a wide field-of-view cone aligned with the forward direction of the virtual camera Cnew, yielding Xe. A lightweight, point-based renderer then produces the synthesized view:

Iˆk = Render X(m),C(m),m , (5)

where X(m) = Xg and C(m) = Ci for the global mode, and X(m) = Xe and C(m) = Cnew for the ego mode.

##### 3.3 VLM-based Spatial Reasoning Agent

As shown in Figure 2(a), the VLM-based agent iteratively interacts with the 3D scene via the manipulation toolkit and accumulates rendered observations for spatial reasoning.

In the k-th iteration, given the history Hk−1, the VLM acts as a multimodal policy:

ok = πθ q,{It}Tt=1,Hk−1 , (6)

where q and {It}Tt=1 denote the input query and the original multi-view images, respectively. The output ok is parsed into a textual response and an optional tool call.

We optionally invoke 3D reconstruction once at the beginning with a binary decision r ∈ {0,1}. When r = 1, Pi3 [59] reconstructs the point cloud X and estimates camera poses {Ct}Tt=1 from the multi-view inputs. For viewpoint manipulation and rendering, the tool-call parameters at iteration k are:

ak = (ik,mk,∆αk,∆βk), (7) where ik ∈ {1,...,T} selects the anchor camera Ci

, mk ∈ {global,ego} specifies the view mode, and ∆αk,∆βk denote the azimuth and elevation angles.

k

Given ak, the toolkit applies the transformation associated with mk. If mk = ego, it constructs a virtual camera via Eq. (4):

Cnew(k) = Ki

, ∆R(∆αk,∆βk)Ri

, ti

k

k

k

. (8)

If mk = global, it applies the global transform in Eq. (3) to obtain Xg while keeping the anchor camera Ci

fixed. The renderer then synthesizes the novel view Iˆk according to Eq. (5), which is appended to the history:

k

Hk = Hk−1 ∥ (Iˆk,ak) . (9)

Thus, Think3D implements an iterative observe → manipulate → reflect loop, where the VLM maintains an explicit 3D-aware reasoning trace over the rendered views. The detailed prompts are provided in the supplementary material.

##### 3.4 Think3D-RL for Multi-Step Exploration

While the reasoning loop allows the model to explore 3D space, its effectiveness depends on learning which viewpoints provide informative observations and when such exploration should be conducted. We therefore optimize the exploration policy using reinforcement learning.

Trajectory Formulation & Training-time Sampling. We represent an agentic reasoning episode as the following trajectory:

τ = {(s1,o1),(s2,o2)...,(sK,oK),yˆ}, (10)

where sk = (q,{It},Hk−1) represents an input to the VLM agent at the k-th iteration; yˆ denotes the final answer generated by the agent; and K denotes the total number of exploration steps determined by the agent.

To avoid repeatedly reconstructing 3D geometry during optimization, we precompute a point cloud for each sample beforehand. At the k-th exploration step, the agent selects a camera pose, and we render the corresponding view from the precomputed point cloud as the observation ok. This design keeps the reasoning loop fully interactive while making training and evaluation efficient.

Trajectory-level reward. Rewards are assigned only at the end of each trajectory:

R(τ) = Rans(ˆy) + Rfmt(ˆy), (11)

where Rans(ˆy) measures answer correctness by matching yˆ to the multiple-choice ground-truth option and Rfmt applies a small formatting bonus. This trajectorylevel reward jointly reinforces all preceding viewpoint decisions, thereby promoting more efficient multi-step spatial exploration.

Optimization. We train the policy with Group Relative Policy Optimization (GRPO) [43] for stable multi-turn reasoning. Following [84], we use a token-wise mask to stop gradients on observation tokens (rendered images encoded as text), optimizing only model-generated action and answer tokens.

### 4 Experiment

##### 4.1 Experiment Setup

Setting and Dataset Our reinforcement learning (RL) training framework is based on SWIFT [82]. We fine-tune the VLM using the GRPO training strategy with 8 rollouts per step to estimate advantages. The model is trained for one epoch on 8 H200 GPUs with a batch size of 8 and gradient accumulation of 4, using a cosine learning rate schedule with 5% warmup and a base learning rate of 1×10−6. The maximum completion length is set to 1024 tokens. During training, the language model is fully fine-tuned while the vision encoder is frozen. The training set contains 977 samples randomly selected from the MindCube dataset, with no overlap with the test set. During inference, we deploy a Pi3 tool on a RTX 3090 GPU to perform inference. We provide full implementation details of Think3D, along with the prompts used in supplementary material.

Benchmarks We evaluate our method on 3 challenging spatial reasoning benchmarks: BLINK (Multi-view) [19], MindCube [75], and the video-based VSIBench [69]. BLINK (Multi-view) uses all the multi-view data from the BLINK dataset and focuses on multi-view geometric understanding, particularly assessing a model’s ability to infer relative camera motion across views. MindCube contains 3 canonical camera-motion types—rotation, around, and among. We sample 40 questions from each category, resulting in 120 questions in total for evaluation. VSI-Bench assesses visual–spatial intelligence in dynamic egocentric videos across four tasks: route planning, object relative direction prediction, appearance order reasoning, and relative distance. We adopt the VSI-Bench-tiny split and sample 7 frames from each video for evaluation. All models are evaluated on the same sample sets for fair comparison.

- Table 1: Results on VSI-Bench-tiny (%). Think3D uses up to two exploration iterations for proprietary baselines and up to three for Qwen-VL-4B. Qwen3-VL-4BT3RL is trained with Think3D-RL, and Qwen3-VL-4BGRPO with standard GRPO.

Model Route Plan Rel. Dir. Rel. Dist. App. Order Avg Proprietary models

GLM-4.5V [53] 34.69 41.03 40.00 79.16 48.72 Doubao-1.5 [41] 42.86 18.00 40.00 71.40 43.07

###### Specialized Spatial Models

RoBoBrain [22] 28.57 36.00 16.00 12.24 23.20 Spatial-MLLM [64] 38.30 44.00 40.00 65.31 46.94 VLM-3R [17] 46.94 64.27 38.00 55.10 51.08 REVPT [88] 28.57 40.00 40.00 51.02 39.90

GPT-4.1 [37] 40.80 40.63 43.30 68.00 48.18 Think3D (GPT-4.1) 45.26 (+5.18) 45.30 (+4.67) 46.00 (+2.70) 68.00 (+0.00) 51.14 (+2.96) Gemini-2.5-Pro [14] 45.58 28.67 50.67 55.73 45.16 Think3D (Gemini-2.5-Pro) 46.93 (+1.35) 37.30 (+8.63) 54.00 (+3.33) 68.24 (+12.51) 51.61 (+6.45)

Qwen3-VL-4B [38] 34.69 40.67 35.33 42.44 38.28 Think3D (Qwen3-VL-4B) 30.61 (-4.08) 44.00 (+3.33) 29.33 (-6.00) 52.38 (+9.94) 39.08 (+0.80)

Qwen3-VL-4BGRPO 28.57 38.00 36.00 30.61 33.30 Qwen3-VL-4BT3RL 27.89 30.67 32.00 42.86 33.36 Think3D (Qwen3-VL-4BT3RL) 36.73 (+8.84) 39.00 (+8.33) 44.67 (+12.67) 61.22 (+18.36) 45.41 (+12.05)

Baseline Models For leading closed-source state-of-the-art models, we evaluate GLM-4.5V [53], Doubao-1.5 [41], GPT-4.1 [37], and Gemini-2.5-Pro [14]. In addition, we compare against specialized models fine-tuned on spatial reasoning datasets, including RoboBrain [22], Spatial-MLLM [64], and VLM-3R [17], as well as REVPT [88], a tool-augmented fine-tuning method. For Qwen3-VL-4B, we use the standard GRPO algorithm and denote the resulting model as Qwen3VL-4BGRPO. Training uses an accuracy-based reward and a format reward, and the training setting is aligned with Think3D-RL. More model experiments are provided in the supplementary material.

##### 4.2 Main Results

Results on the multi-view reasoning benchmark (Table 2) show that Think3D substantially improves proprietary models such as GPT-4.1 and Gemini-2.5-Pro, yielding 11.57% and 4.00% relative gains, respectively, in a zero-shot setting. In contrast, for smaller models such as Qwen3-VL-4B, the gain is marginal (0.61%), suggesting limited spatial reasoning capacity constrains the benefit of exploration. However, once Qwen3-VL-4B is fine-tuned with Think3D-RL (Qwen3VL-4BT3RL), it improves by 9.32% with Think3D. This provides evidence that RL strengthens viewpoint selection and spatial exploration. We further analyze how RL-trained models achieve these gains in Section 5.4. On VSI-Bench (Table 1), results further support Think3D, yielding a 2.96% improvement on GPT-

- 4.1 and a 6.45% improvement on Gemini-2.5-Pro. These gains indicate that Think3D also improves video-based spatial reasoning. Moreover, our RL-finetuned model achieves larger gains with Think3D, rising from 0.8% to 12.05%,

- Table 2: Results on BLINK (Multi-view) and the MindCube subset (%). Think3D uses up to three exploration iterations. Qwen3-VL-4BT3RL is trained with Think3DRL, and Qwen3-VL-4BGRPO with standard GRPO.

Model BLINK (MV) MC (Rotation) MC (Among) MC (Around) Avg Proprietary models

GLM-4.5V [53] 39.85 45.00 45.00 22.50 38.09 Doubao-1.5 [41] 50.93 72.50 40.00 35.00 49.61

###### Specialized Spatial Models

RoBoBrain [22] 55.64 32.50 57.50 52.50 49.54 Spatial-MLLM [64] 56.06 32.50 47.50 35.00 42.77 VLM-3R [17] 41.35 25.00 47.50 37.50 37.84 REVPT [88] 51.89 30.00 55.00 50.50 47.35

GPT-4.1 [37] 36.82 60.00 46.67 55.00 49.62 Think3D (GPT-4.1) 63.91 (+27.09) 63.33 (+3.33) 57.50 (+5.00) 60.83 (+14.16) 61.19 (+11.57) Gemini-2.5-Pro [14] 44.86 85.00 49.17 58.33 59.34 Think3D (Gemini-2.5-Pro) 52.88 (+8.02) 86.67 (+1.67) 54.17 (+5.00) 60.83 (+2.50) 63.34 (+4.00)

Qwen3-VL-4B [38] 47.87 34.17 20.00 41.67 35.92 Think3D (Qwen3-VL-4B) 48.62 (+0.75) 35.83 (+1.66) 28.33 (+8.33) 33.33 (-8.34) 36.53 (+0.61)

Qwen3-VL-4BGRPO 52.38 35.00 21.67 28.33 34.34 Qwen3-VL-4BT3RL 46.11 30.83 25.83 35.83 34.65 Think3D (Qwen3-VL-4BT3RL) 53.39 (+7.28) 42.50 (+11.67) 37.47 (+11.64) 42.50 (+6.67) 43.97 (+9.32)

highlighting that RL enables more effective 3D spatial exploration. We also provide a qualitative example in Figure 3.

[Figure 44]

- Fig. 3: Spatial exploration behavior of Think3D. The agent autonomously selects viewpoints and switches between global and ego-centric views; after RL training, it explores angles more systematically than the untuned baseline.

- Table 3: Ablation on different 3D reasoning components. All results are reported

- as accuracy (%). 3D Rec. denotes reasoning with reconstructed 3D geometry; Cam. Anchor uses the camera pose as the manipulation anchor; Cam. Cho. enables camera selection; and Ego-view indicates whether the model may request ego-centric views. We report results on BLINK (multi-view) and MindCube.

3D Rec.

Cam. Anchor

Cam. Cho.

Ego View

BLINK MindCube

36.82 55.83 ✓ 41.17 54.59 ✓ ✓ 55.46 57.22 ✓ ✓ ✓ 61.65 58.89 ✓ ✓ ✓ ✓ 63.91 63.33

### 5 Ablation Study

##### 5.1 Ablation of Components

As shown in Table 3, we ablate key Think3D components. Compared to the GPT-4.1 baseline (first row) that never calls the 3D tool, directly using 3D reconstruction space without an anchor camera pose to guide point cloud manipulation causes a mild performance drop. This suggests raw 3D input alone is insufficient, as the model must actively explore multiple viewpoints to reach the correct answer. Adding anchor camera selection and ego-view configuration greatly improves performance. These components help the model process 3D point clouds more efficiently.

##### 5.2 Ablation of Space Exploration Strategy

- As shown in Figure 4a, we analyze the spatial exploration strategies of VLMs across multiple task types—including multi-view reasoning, route planning, and object-orientation estimation—and across models with different base capabilities. Visualizing GPT-4.1’s exploration behavior reveals clear task-dependent patterns. For instance, in route planning and appearance-order tasks, GPT-4.1 predominantly uses top-down viewpoints to capture global spatial structure. In contrast, for tasks such as MindCube and object-orientation estimation, the model relies more on rotational viewpoints that support orientation inference.

5.3 Ablation of Reinforcement Learning Dynamics

- As shown in Figure 5, we visualize RL training dynamics of one checkpoint by tracking the evolution of the accuracy reward and the number of turns per trajectory. During the first 50 training steps, the model tends to reduce turns to increase reward. However, this reduction causes a noticeable drop in accuracy: with fewer turns, the model invokes spatial tools less often and thus obtains fewer

- 3D viewpoints. After about 50 training steps, the model gradually increases its spatial tool usage to render point-cloud images, resulting in steady improvement in accuracy.

Task-Specific Angle Preference

100

RoutePlanning

[Figure 45]

[Figure 46]

80 6 8

RelativeDistance

UsagePercentage(%)

80

- 38 9 19 7

37 24 7 7

62 5

- 39 19 12 7

ObjectDirection

60

Task

AppearanceOrder

40

BLINK

20

MindCube

36 17 10 15

0

(0,60) (-45,0) (45,0) (90,0) (0,0) (-90,0)(90,60)(45,60)

Angle Combination (Azimuth, Elevation)

(a) Task-level patterns.

Model Comparison: Angle Distribution

100

[Figure 47]

[Figure 48]

Qwen3-VL-4B

48 29 17 5

UsagePercentage(%)

80

Qwen3-VL-4B-RL

62 5 20 9

60

Model

GPT-4.1-mini

15 10 5 4 13 8

40

GPT-4.1

74 11 3

20

Gemini2.5-pro

68 7 12

0

(0,60)(-45,0) (45,0) (0,0)(45,60)(90,60)(45,30) (90,0)

Angle Combination (Azimuth, Elevation)

(b) Model-level patterns.

- Fig. 4: Spatial exploration patterns in viewpoint selection. Strong models concentrate on informative angles (e.g., oblique and top-down views); after RL fine-tuning, Qwen3VL-4BT3RL shifts toward a similar distribution. Across tasks, exploration varies substantially (e.g., route planning prefers top-down views around (0, 60)).

[Figure 49]

[Figure 50]

- Fig. 5: Reinforcement Learning Dynamics. As RL fine-tuning progresses, the model learns when extra 3D tool calls are worthwhile, shifting from shorter but less accurate trajectories to more informative explorations with higher reward.

##### 5.4 Ablation on What the Model Learns through RL

To better understand what the model learns from reinforcement learning, we analyze spatial exploration behavior before and after RL fine-tuning. We visualize viewpoint distributions of strong models such as GPT-4.1 and Gemini-2.5-Pro, whose robust strategies correlate with substantial gains under Think3D. We then compare these behaviors with those of a smaller model, Qwen3-VL-4B, and its RL-enhanced variant, Qwen3-VL-4BT3RL. As shown in Figure 4b, Qwen3-VL4BT3RL adopts viewpoint patterns closer to the stronger models, for example selecting top-down perspectives more often to capture global spatial structure. These results indicate that RL improves informed 3D exploration.

##### 5.5 Ablation of Exploration Rounds

We further analyze how the number of exploration iterations affects model performance. As shown in Figure 6, for models without RL training, increasing the number of interaction turns does not yield a clear performance gain. After RL training, Qwen3-VL-4BT3RL begins to follow the same trend as the stronger models: its accuracy steadily increases as the number of exploration turns grows, indicating improved returns from additional visual evidence. These results suggest that RL enables the model to learn deeper and more effective spatial exploration strategies, which supports a more reliable and efficient utilization of Think3D.

Performance across Exploration Rounds

GPT-4.1

65

Gemini-2.5-pro

Qwen3-VL-4B

Qwen3-VL-4B-T3RL

60

AverageAccuracy(%)

55

50

45

40

35

1 2 3 4

Spatial Exploration Rounds

Fig. 6: The ablation of turns.

##### 5.6 Efficiency vs. Multi-round Prompting Baseline

A natural question is whether Think3D’s gain simply comes from running more rounds of reasoning. We compare Think3D to a strong multi-round prompting baseline that matches the same number of rounds (3) using SelfRefine [34] without any 3D tools. As shown in Figure 7, multi-round self-critiques bring only marginal improvements, while incurring comparable or higher token/time cost. This indicates that Think3D’s advantage primarily stems from explicit 3D interaction rather than multi-round prompting.

| | | | |
|---|---|---|---|
| | | | |
| | |Thin (6.7k|k3D tokens)|
|GPT|-4.1<br><br>Self-R (7.7k|efine tokens)| |
|(1.3k|tokens)| | |
| | | | |

70

AvgAcc(%)

60

50

40

30

GPT-4.1 Self-Refine Think3D

Fig. 7: Efficiency ablation on BLINK (accuracy vs. token usage).

##### 5.7 Robustness of the 3D Reconstruction Tool

To test whether Think3D depends on a specific 3D reconstructor, we replace Pi3 with VGGT while keeping the rest of the pipeline unchanged, including prompting, tool-calling, and the reasoning budget. As shown in Table 4, Think3D remains effective: VGGT still delivers clear gains over the no-tool baseline on both BLINK and MindCube, and retains most of the improvement achieved with Pi3. This indicates that Think3D is largely reconstructor-agnostic and can benefit from off-the-shelf 3D tools with different accuracy profiles.

- Table 4: Comparison of 3D reconstruction tools on spatial benchmarks (Acc.%).

###### Tool Conference BLINK MindCube

w/o Tool – 36.82 53.89 VGGT [56] CVPR 2025 59.65 59.59 Pi3 [59] ICLR 2026 63.91 60.55

### 6 Conclusion

We introduce Think3D, a framework that lets VLM agents actively reason in 3D rather than rely on passive 2D perception. By iteratively exploring reconstructed point clouds with a 3D manipulation toolkit, Think3D achieves deeper and more consistent spatial understanding. Its RL-enhanced variant (Qwen-4BVLT3RL) learns efficient exploration, enabling smaller VLMs to approach the behavior and performance of large proprietary models. Experiments on BLINK, MindCube, and VSI-Bench-Tiny show strong gains and cross-benchmark generalization. Overall, Think3D suggests explicit 3D interaction as a promising route to genuine spatial reasoning in VLMs.

### A Prompts and Implementation Details

Training-free Workflow Prompt. Following [27,74], the prompt design in Think3D is structured into three parts: a system prompt (Fig. 8), a tool prompt that describes the 3D tools and their usage rules (Fig. 9), and a continual prompt that updates the context at the beginning of each reasoning round (Fig. 10). We adopt this modular design to improve prompt clarity and make multi-round tool-augmented reasoning more stable.

The system prompt defines the model’s role, the overall reasoning workflow, and the required output format. This helps reduce invalid tool calls and keeps the reasoning process executable and consistent. The tool prompt specifies how the model should use 3D reconstruction tools, including which new viewpoints are worth exploring and how to avoid redundant views. In particular, we explicitly state that the input image already corresponds to the default (0◦,0◦) view, and provide recommended alternative viewpoints such as left, right, top, back, and diagonal views. This design encourages the model to request complementary observations that are more informative for spatial reasoning.

The continual prompt is used to maintain reasoning progress across multiple rounds. At each round, it reminds the model of the task goal, the current context, and the need to decide whether additional tool use is still necessary. This helps the model stay focused on unresolved spatial uncertainty rather than repeatedly generating redundant analysis. Overall, this three-part prompt design improves format reliability, viewpoint efficiency, and multi-round reasoning stability in the training-free workflow.

RL Training Prompt. During RL training, directly invoking external tools online

- at every rollout step is prohibitively time-consuming, which would significantly reduce training efficiency and make large-scale optimization impractical. To address this issue, we pre-generate point clouds for all training scenes in advance, thereby eliminating the need to run Pi3 inference during RL training. This design substantially reduces the per-sample processing overhead and enables more efficient policy optimization while keeping the spatial input representation consistent across training iterations.

In addition, we observe that smaller open-source models generally exhibit weaker instruction-following and prompt-utilization capabilities than larger proprietary models. A single fixed prompt is therefore often insufficient for stable multi-round RL optimization, especially when the model needs to progressively refine its reasoning and action prediction over repeated iterations. To improve prompt efficiency, we further divide the continual training prompts according to the current iteration stage. Specifically, different prompts are used for the initial training round, intermediate continuation rounds, and the final refinement stage. This stage-aware prompt design allows the model to receive instructions that are better aligned with its current optimization status, improving both prompt utilization and training stability.

You are a helpful assistant that can analyze images and answer questions. Tools You have access to the following tools to assist with user queries: <tools>{tools_json}</tools> How to call a tool When you need to use a tool, return a JSON object with the function name and arguments within <tool_call></tool_call> XML tags: <tool_call>{{"name": "<function-name>", "arguments": {{"param1": "value1", "param2": "value2"}}}}</tool_call> You can call multiple tools if needed by using multiple <tool_call> blocks. Multi-Step Workflow You can perform MULTIPLE rounds of tool calls and analysis. When using 3D reconstruction tools (Pi3), autonomously explore viewpoints:

IMPORTANT: The input image(s) already show the scene at (azimuth=0°, elevation=0°) viewpoint. DO NOT call Pi3 tools with (0°, 0°) as it will just return the same view you already have! The camera is visualized as a pyramid frustum, where the apex represents the camera's position and viewing direction.

Recommended NEW viewing angles to explore: · Left views: azimuth=-45° or -90° (see scenes from right view) · Right views: azimuth=45° or 90° (see scenes from left view) · Top views: elevation=30° to 60° (see scenes from top view, better capture the object relation and relatifve position of cam and objects.) · Back views: azimuth=180° or ±135° (see scenes from back view) · Diagonal views: combine azimuth and elevation (e.g., 45°, 30°)

###### Workflow:

- 1. Analyze the current view(s) you have
- 2. Decide which NEW angles (NOT 0°,0°!) would help answer the question
- 3. Call tools with specific angles that are DIFFERENT from (0°,0°)
- 4. If you have multiple input images: Try different rotation_reference_camera values (1, 2, 3, etc.) to see the scene from different camera positions base on your analysis on the question.
- 5. Consider using camera_view=true to get first-person perspective from specific camera positions, especially useful for understanding spatial relationships and what each camera can actually see
- 6. After each round, analyze whether additional angles, camera positions, or perspective modes would reduce uncertainty

- 8. Continue until additional views no longer change your conclusion
- 9. Only put number (like 1,2,3) or Options in <answer></answer> tags, do not put any other text.

Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame. You can examine the image to understand what is around cam1. The 3D reconstruction provides relative positional information, so you should reason interactively and complementarily between the 2D image and the 3D reconstruction to form a complete understanding. You need to analyze deeply the camera, its orientation, and the content captured in the frame.

TIPS: For questions related to orientation or relative positioning, it is recommended to choose top view. Please analyze the following image(s): Images to analyze: {images_info}

Question: {question}

Think step by step to analyze the question and provide a detailed answer. Important Notes: · You can call tools MULTIPLE times with different parameters to gather comprehensive information · After each tool execution, you'll see the results and can decide if you need more information · Only provide your final <answer></answer> when you have gathered sufficient information

You MUST output your thinking process in <think></think> and tool choices in <tool_call></tool_call>. When you have enough information, output your final choice in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text.

- Fig. 8: The system prompt. Instruction prompt detailing tool invocation rules and the multi-step workflow for iterative 3D viewpoint exploration, including tool-call format, recommended angles, and guidelines for reasoning with reconstructed camera poses.

The corresponding prompts used in different stages are shown in Fig. 11, Fig. 12, and Fig. 13.

This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis. It performs 3D reconstruction from images to generate point clouds and visualizations from CUSTOM viewing angles.

Important Note: The 0° azimuth angle and 0° elevation angle corresponds to the first input image viewpoint (cam1). Do not use this angle.

Angle Parameters: · azimuth_angle (-180° to 180°, integer only): Controls left-right rotation. · elevation_angle (-90° to 90°, integer only): Controls up-down rotation. By convention, (azimuth=0, elevation=0)

corresponds EXACTLY to the first input image viewpoint (cam1). All rotations are defined in the INPUT CAMERA coordinate frame: azimuth rotates left/right around the camera's vertical axis; elevation rotates up/down around the camera's right axis.

· rotation_reference_camera (must be output, 1-based):This parameter is used to rotate around a specific input image's camera. By picking an image you pick its camera (e.g., set rotation_reference_camera=3 for the third image's viewpoint; defaults to 1).

camera_view (must be output, boolean): This parameter is used to generate first-person perspective from the selected camera position (as if standing at that camera looking at the scene), instead of the default global bird's-eye view. This is especially useful for understanding what each camera can see and analyzing spatial relationships from specific viewpoints. Combine with rotation_reference_camera to experience the scene from different camera positions.

Note that default camera_view is false. You must output camera_view = true if you want to set ego-view. If you want to set global-view, you must output camera_view = false.

Usage Strategy: You can call this tool MULTIPLE times with DIFFERENT angles and different camera views to analyze the 3D structure comprehensively. The MLLM is encouraged to autonomously explore angles (coarse-to-fine) until sufficient evidence is gathered. The generated visualization uses cone-shaped markers to indicate camera positions, numbered from 1 (cam1, cam2, etc.).

- Fig. 9: The Pi3 Tool Prompt. The prompt specifies the tool’s capabilities, key control parameters, and multi-angle query usage strategies to support comprehensive spatial understanding.

Prompt for evaluation without tools. When no tool is available, we adopt standard chain-of-thought (CoT) [62] reasoning. The corresponding prompt is shown in Fig 14.

Prompt for self-refine experiment. The corresponding prompt is shown in Fig 15. Algorithm 1 Self-Refine Inference

Require: Question q, iterations T

- 1: a0 ← LLM(q)
- 2: for t = 0 to T − 1 do
- 3: ct ← LLMcritique(q, at)
- 4: at+1 ← LLMrefine(q, at, ct)
- 5: end for
- 6: return aT

=== Multi-Step Analysis: Iteration {current_iteration}/{max_iterations} === Original Question: {question} Your Previous Response: {last_response} Tool Execution Summary: {tool_summary_text} {angle_info_text} Original Images: {original_images_info} Generated Images Available for Analysis: {additional_images_info}

=== Next Steps === You have {remaining} more iteration(s) available. You can:

- 1. Continue investigating - Call tools with DIFFERENT parameters: · IMPORTANT: Your original input images are already at (azimuth=0°, elevation=0°). DO NOT call Pi3 tools with (0°, 0°)

again! · For Pi3 tools: Try NEW viewing angles to understand the 3D structure better · Recommended NEW angles (NOT 0°,0°!):

· Left: (-45°, 0°) or (-90°, 0°) · Right: (45°, 0°) or (90°, 0°) · Top: (0°, 45°) or (0°, 60°) · Bottom: (0°, -45°) · Back: (180°, 0°) or (±135°, 0°) · Diagonal: (45°, 30°) or (-45°, 30°)

· Each NEW angle reveals different aspects of the 3D structure Advanced Pi3 Parameters: · rotation_reference_camera (integer, 1-based): When you have multiple input images, try DIFFERENT camera positions

as rotation centers · Default is 1 (first camera), Set to 2, 3, etc. to rotate around different camera positions · Example: rotation_reference_camera=2 rotates around the second camera's viewpoint · Useful for analyzing different parts of the scene from various perspectives

· camera_view (boolean): Control the visualization perspective · False (default): Global bird's-eye view showing the entire scene · True: First-person camera view - see the scene from the selected camera's perspective (as if standing at that camera) · Combine with rotation_reference_camera to experience different camera viewpoints · Example: camera_view=True with rotation_reference_camera=2 shows first-person view from camera 2 · Useful for understanding what each camera can see and spatial relationships

- 2. Provide final answer - If you have sufficient information from current viewpoints: · Output your comprehensive analysis in <think></think> tags · Reference the specific viewpoints that helped you understand the structure

Instructions: · Think: Do you need to see the object from another NEW angle (NOT 0°,0°!) to answer the question better? · If YES: Use <tool_call></tool_call> to request a DIFFERENT viewing angle (avoid 0°,0° as you already have it!) · If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text.

Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame. You can examine the image to understand what is around cam1. The 3D reconstruction provides relative positional information, so you should reason interactively and complementarily between the 2D image and the 3D reconstruction to form a complete understanding.

Please continue:

###### Fig. 10: Multi-step prompt for iterative 3D viewpoint exploration. Including angle selection, camera rotation controls, tool invocation rules to refine spatial reasoning.

You are a helpful assistant that can analyze images and answer questions. Tools You have access to the following tools to assist with user queries: <tools>{tools_json}</tools> How to call a tool When you need to use a tool, return a JSON object with the function name and arguments within <tool_call> </tool_call> XML tags: <tool_call>{{"name": "<function-name>", "arguments": {{"param1": "value1", "param2": "value2"}}}}</tool_call> You can call multiple tools if needed by using multiple <tool_call> blocks. Multi-Step Workflow You can perform MULTIPLE rounds of tool calls and analysis. When using 3D reconstruction tools (Pi3), autonomously explore viewpoints:

IMPORTANT: The input image(s) already show the scene at (azimuth=0°, elevation=0°) viewpoint. DO NOT call Pi3 tools with (0°, 0°) as it will just return the same view you already have! The camera is visualized as a pyramid frustum, where the apex represents the camera's position and viewing direction. You only have 8 viewing angles to choose: · Left views (see scenes from right view): (azimuth_angle, elevation_angle) = (-45°, 0°)/(-90°, 0°) · Right views (see scenes from left view): (azimuth_angle, elevation_angle) = (45°, 0°)/(90°, 0°) · Top views (see scenes from top view, better capture the object relation and relatifve position of cam and objects.): (azimuth_angle, elevation_angle) = (0°, 60°) · Opposite views: (azimuth_angle, elevation_angle) = (45°, 60°)/(45°, 30°)/(-45°, 30°)

Workflow:

- 1. Analyze the current view(s) you have
- 2. Decide which NEW angles (NOT 0°,0°!) would help answer the question
- 3. Call tools with specific angles that are DIFFERENT from (0°,0°)
- 4. After each round, analyze whether additional angles or perspective modes would reduce uncertainty
- 5. Continue until additional views no longer change your conclusion
- 6. Only put letters of options (like A,B,C) in <answer></answer> tags, do not put any other text. TIPS: For questions related to orientation or relative positioning, it is recommended to choose top view. Please analyze the following image(s):

Images to analyze: {images_info}

Question: {question}

Think step by step to analyze the question and provide a detailed answer Important Notes: · You can call tools MULTIPLE times with different parameters to gather comprehensive information · After each tool execution, you'll see the results and can decide if you need more information · Only provide your final <answer></answer> when you have gathered sufficient information

You MUST output your thinking process in <think></think> and tool choices in <tool_call></tool_call>. When you have enough information, output your final choice in <answer></answer>. Only put Options in <answer> </answer> tags, do not put any other text.

###### Fig. 11: The RL system prompt. Instruction prompt defining the constrained 3-view 3D analysis workflow, including tool-call format, angle selection rules (left, right, top), and iterative reasoning steps for viewpoint-guided spatial understanding.

=== Multi-Step Analysis: Iteration {current_iteration}/{max_iterations} === Original Question: {question} Your Previous Response: {last_response} Tool Execution Summary: {tool_summary_text} {angle_info_text} Original Images: {original_images_info} Generated Images Available for Analysis: {additional_images_info}

=== Next Steps === You have {remaining} more turn(s) available. You can: Continue investigating - Call tools with DIFFERENT parameters:

· IMPORTANT: Your original input images are already at (azimuth=0°, elevation=0°). DO NOT call Pi3 tools with (0°, 0°) again!

· For Pi3 tools: Try NEW viewing angles to understand the 3D structure better

Instructions: · Think: Do you need to see the object from another NEW angle (NOT 0°,0°!) to answer the question better? · If YES: Use <tool_call></tool_call> to request a DIFFERENT viewing angle (avoid 0°,0° as you already have

it!)

· If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Only put

letters of option in <answer> </answer> tags, do not put any other text. · You only have 8 viewing angles to **choose**: · Left views (see scenes from right view): (azimuth_angle, elevation_angle) = (-45°, 0°)/(-90°, 0°) · Right views (see scenes from left view): (azimuth_angle, elevation_angle) = (45°, 0°)/(90°, 0°) · Top views (see scenes from top view, better capture the object relation and relatifve position of cam and

objects.): (azimuth_angle, elevation_angle) = (0°, 60°) · Opposite views: (azimuth_angle, elevation_angle) = (45°, 60°)/(45°, 30°)/(-45°, 30°) · Do not request the same angle as before.

Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1

represents the first frame. You can examine the image to understand what is around cam1. The 3D reconstruction provides relative positional information, so you should reason interactively and

complementarily between the 2D image and the 3D reconstruction to form a complete understanding. IMPORTANT: You MUST start your response with <think>...</think> tags to explain your reasoning! Provide final answer - If you have sufficient information from current viewpoints:

· Output your comprehensive analysis in <think></think> tags · Reference the specific viewpoints that helped you understand the structure · if you want to provide final answer, the reasoning process and answer are enclosed within <think> </think>

and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>

Please continue:

###### Fig. 12: The RL continuation prompt used during non-final turns. Iterative-step instruction prompt outlining allowed viewpoint choices (left/right/top), tool-call rules, and the decision process for progressing or concluding 3D spatial analysis.

⚠ WARNING: This is the FINAL turn Original Question: {question} Original Images: {original_images_info} The 3D Reconstruction image. {additional_images_info} Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame. You can examine the image to understand what is around cam1. The 3D reconstruction provides relative positional information, so you should reason interactively and complementarily between the 2D image and the 3D reconstruction to form a complete understanding. All available tools have been used up, and you can no longer call any additional tools. You have no remaining steps for tool invocation. You can now see the different perspectives generated by the previous tool calls, as well as the original image. Please use the provided content to answer the original question. You MUST NOT call any tools. You MUST NOT output <tool_call>. You MUST directly reason and answer in this round. All reasoning MUST be written explicitly within <think> </think> tags. The final answer MUST be written within <answer> </answer> tags. Format strictly as follows: <think>[Your reasoning process here — show step-by-step thinking, explanations, or derivations]</think><answer> [Your final answer here — only put your choice here]</answer> Example: <think>First, analyze the question carefully. Then, derive the solution using logical reasoning.</think> <answer>A</answer>

###### Fig. 13: The RL continuation prompt used in the final turn. Final-turn instruction prompt specifying the no-tool phase, requiring explicit reasoning and a final answer based solely on previously generated 3D views and the original image.

You are a helpful assistant that can analyze images and answer questions. Please analyze the following image(s): Images to analyze: {images_info}

Question: {question}

Think step by step to analyze the question and provide a detailed answer. You MUST output your thinking process in <think></think> and your final answer in <answer></answer>. Only put Options in <answer></answer> tags, do not put any other text.

###### Fig. 14: The prompt without tools. Base instruction prompt for direct image-question analysis, requiring explicit reasoning and final answer formatting without tool interactions.

=== Self-Refine: Critique Round {critique_round + 1}/{max_iterations} === You are a critical reviewer. Your task is to carefully analyze the following answer and provide constructive feedback. Original Question: {question} Answer to Critique: {last_answer} Your Task: Please critically evaluate the above answer. Consider:

- 1. Correctness: Is the answer factually correct? Are there any logical errors?
- 2. Completeness: Does the answer fully address all aspects of the question?
- 3. Reasoning: Is the reasoning clear and well-supported by visual evidence?
- 4. Confidence: How confident should we be in this answer? Are there alternative interpretations? Provide your critique in <critique></critique> tags. Be specific about:

- - What aspects are correct and well-reasoned
- - What aspects might be wrong or need improvement
- - Specific suggestions for how to improve the answer

Format: <critique> [Your detailed critique here] </critique>"""

=== Self-Refine: Refinement Round {refine_round}/{max_iterations} === Based on the critique below, please refine your answer to the original question. Original Question: {question} Your Previous Answer: {last_answer} Critique of Your Answer: {critique} Your Task: Carefully consider the critique and improve your answer. Address the issues raised and strengthen your reasoning.

Please provide an improved answer based on the critique. You MUST output your thinking in <think></think> tags and your refined answer in <answer></answer> tags."""

Format your response as: <think> [Your refined reasoning, addressing the critique] </think> <answer> [Your refined answer] </answer>"""

###### Fig. 15: The critique prompt and refinement prompt used in the self-refine experiment.

### B Further Experiment Analysis

Ego view analysis As shown in Fig 16, we visualize the proportion of ego-view versus global-view usage by GPT-4.1 across different tasks. We find that tasks requiring fine-grained local understanding—such as MindCube and Object Direction—exhibit a much higher reliance on ego-view. In contrast, tasks like Route Planning, which demand broader global context, show minimal use of ego-view and favor global-view instead.

###### Task Comparison: Ego View vs Local View Usage

Ego View

100

Local View

80

###### Percentage(%)

76.5%

60

85.4%

99.5%

100.0% 100.0% 100.0%

40

20

23.5%

14.6%

0

BLINK MindCube ObjDirection RelDistance RoutePlan APOrder

Task Type

###### Fig. 16: Ego view usage ratio across different tasks. Distribution of GPT-4.1’s reliance on ego-view versus global-view across tasks. Fine-grained tasks emphasize ego-centric information, whereas tasks requiring broad context predominantly utilize global-view.

Tool calling iteration analysis As shown in Fig 17, we also visualize the proportion of tool calls across different tasks. We find that for route planning, GPT-4.1 uses the tools much less frequently. For the other tasks, GPT-4.1 often performs multiple rounds of tool calls to obtain richer spatial information.

Tool Iteration Distribution by Task (Percentage)

100

- Iteration 1

| |
|---|

- Iteration 2

15.4% (6)

21.7% (15)

22.4% (46)

23.9% (39)

36.0% (32)

80

Percentage(%)

60

100.0% (51)

84.6% (33)

78.3% (54)

77.6% (159)

40

76.1% (124)

64.0% (57)

20

0

MindCube BLINK AppearanceOrder ObjectDirection RelativeDistance RoutePlanning

Task

###### Fig. 17: Tool calling iteration ratio across different tasks. GPT-4.1 rarely uses tools for route planning, while conducting multiple rounds of tool calls for other tasks to acquire richer spatial information.

- Table 5: Training and evaluation parameters used in both the RL optimization process and subsequent evaluation.

Parameter Setting Foundation model Qwen3-4B-Instruct Number of trained agents 1 Number of solution rounds 3 Number of evaluation rounds 3 Horizon for discussion history 1 Token limit for prompts 180000 Token limit for responses 1024 Training temperature 0.6 Evaluation temperature 1.0 Clipping epsilon 0.2 Weight of KL penalty 0.05 Number of training epochs 1 Training batch size 32(8*4accu) Rollout batch size 64 Optimizer name AdamW Learning rate 1e-6 Weight decay 0.1 Gradient norm 0.5 Gradient clipping False Gradient checkpoint True Flash Attention True Mixed precision True Enable vLLM False Enable DeepSpeed True

### C Think3D-RL Training And Evaluation Setting

As shown in Tab 5, we provide the parameters used for both RL training and evaluation. For all experiments, including the main results and ablation studies, we run each setting three times and report the average performance to ensure a fair comparison.

### D Angle choose

Random samples two views from eight candidate angles, while Heuristic uses fixed views (−45,30) and (45,30); both use the original Qwen3-VL-4B. We also test the RL-trained backbone with random exploration (RLrandom). All methods run three rounds. As shown in Table 6, RLrandom performs similarly to Random, while the learned RL policy significantly improves performance, indicating that gains mainly come from spatial policy learning.

###### Table 6: Effect of exploration strategies.

Angle BLINK MindCube Avg (%)

Random 45.86 27.50 36.68 Heuristic 47.37 30.00 38.69 RLrandom 45.11 30.00 37.56 RL 53.39 40.82 47.11

### E Results on More Models

We provide additional evaluation results on more vision-language models in this section. The results on BLINK(Multi-view) and MindCube subset are reported in Table 7. These results provide a broader comparison of current VLMs on spatial reasoning benchmarks.

Table 7: More results on BLINK(Multi-view) and the MindCube subbset(%).

Model BLINK (MV) MC (Rotation) MC (Among) MC (Around) Avg

- InternVL3.5-1B 51.13 37.50 35.00 25.00 36.66
- InternVL3.5-2B 49.62 37.50 32.50 42.50 38.46 InternVL3.5-4B 53.38 50.00 30.00 25.00 36.67 Llava-onevision-qwen2-0.5b 43.61 32.50 22.50 40.00 34.65 Llava-onevision-qwen2-7b 45.86 37.50 25.00 35.00 35.84 Qwen3-VL-8B 47.36 45.00 30.00 47.50 42.47 Qwen3-VL-30B-a3b-instruct 49.62 37.50 37.50 52.50 44.28 GPT-4.1-mini 49.62 54.17 39.17 45.83 47.20 Think3D(GPT-4.1-mini) 52.38 53.33 47.5 41.67 48.72

#### F Interaction Visualization We provide additional visualization examples, as illustrated in the figures below.

Question: If I am standing at the same spot and facing the same direction as shown in image 2, then I turn 180 degrees around, what is to my right? A. Several pictures B. Window and brown cabinet C. Green cabinet

Assistant: <think>Let's break down the question and images step by step:

- - Image 1 shows a green cabinet, a door, and a picture frame.
- - Image 2 shows a window and a brown cabinet.
- - Image 3 shows several pictures on the wall and a green cabinet (partially visible). Let's visualize:
- - Facing image 2 direction, turning 180° means facing the opposite direction (image 3's direction).
- - "To my right" after turning 180° is the same as "to my left" before turning (relative to the original facing).

To be sure, I need to confirm the spatial arrangement of the cabinets, window, and pictures. The best way is to use a top-down (elevation=60°) view to see the relative positions of all objects and cameras.

I will call the Pi3 tool with a top-down view (azimuth=0°, elevation=60°) using camera 1 as reference to get the spatial layout. </think> <tool_call>{"name": "pi3_tool", "arguments": {"image_path": ["dataset/mindcube/data/other_all_image/rotation/arkitscenes_43896310/view_1_1.jpg", "dataset/mindcube/data/other_all_image/rotation/arkitscenes_43896310/view_2_7.jpg", "dataset/mindcube/data/other_all_image/rotation/arkitscenes_43896310/view_3_14.jpg"], "azimuth_angle": 0, "elevation_angle": 60, "rotation_reference_camera": 1, "camera_view": false}}</tool_call>

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Assistant: <think>Let's reason using the images and the top-down 3D reconstruction: Step-by-step:

- 1. **Image 2 (cam2)**: Facing the window and brown cabinet.
- 2. **Turn 180°**: Now facing the opposite direction (as in image 3, cam3).
- 3. **What is to my right?**: When facing the direction of image 3, "to my right" is the direction that was "to my left" when facing image 2. From the top-down view:

- - The cameras are arranged in a triangle, each 90° apart.
- - Cam2 (image 2) faces the window/brown cabinet.
- - Cam3 (image 3) faces the wall with several pictures.
- - Cam1 (image 1) faces the green cabinet and door.

When facing cam3's direction (after 180° turn from cam2), the green cabinet (from cam1) is to the right of cam3's viewpoint, and the wall with several pictures is directly ahead.

**Mapping options:**

- A. Several pictures — these are directly ahead when facing cam3.
- B. Window and brown cabinet — these are behind you (opposite direction).
- C. Green cabinet — this is to the right when facing cam3.

**Conclusion:** After turning 180° from image 2's direction, the green cabinet is to your right. </think><answer>C. Green cabinet</answer>

###### Fig. 18: The Mindcube example.

Question: If I am standing at the same spot and facing the same direction as shown in image 1, what is to my behind? A. Wardrobes B. Door C. Window

Assistant: <think> <think> Let's analyze the spatial arrangement step by step. Image 1 shows a door on the left and what appears to be wardrobes on the right.

- - Image 2 is said to be a 90° right turn from image 1 and shows wardrobes directly ahead, with a door visible to the left and

background room elements on the right.

- - Image 3 is another 90° right turn (so 180° from image 1), showing a window and likely facing the direction opposite image 1. Given the camera rotation pattern:
- - If I'm standing in the same spot facing the door and wardrobes (as in image 1), then turning right 90° takes me to face the wardrobes directly (image 2), and another 90°, the window (image 3).
- - The question asks what is behind me when facing the image 1 direction.

To confirm, the spatial layout should show what's on the opposite side of image 1's direction, which is image 3's viewpoint. In image 3, the window is directly in front, so that would be behind me if I started facing the door and wardrobes.

For further confirmation of object positions and spatial relations, a top-view 3D reconstruction can clarify relative placements (especially the window's location in relation to cam1 and cam2). I will call the tool for a top view (azimuth=180°, elevation=60°) to see relationships from above. I will call the tool to generate a top-down view. <tool_call> {"name": "pi3_tool", "arguments": {"image_path": ["dataset/mindcube/data/other_all_image/rotation/arkitscenes_41126518/view_1_8.jpg",

- "dataset/mindcube/data/other_all_image/rotation/arkitscenes_41126518/view_2_27.jpg",
- "dataset/mindcube/data/other_all_image/rotation/arkitscenes_41126518/view_3_59.jpg"], "azimuth_angle": 180, "elevation_angle": 60, "camera_view": false}} </tool_call>

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Assistant: <think> Let’s reason through the answer using the provided images and the 3D top-down reconstruction.

- - Image 1 (cam1) shows a door on the left and wardrobes on the right.
- - We know from the prompt that for image 2 (cam2), the camera was rotated 90° right (clockwise), and for image 3 (cam3),

another 90° right (so image 3's view is directly opposite to image 1's view).

- The 3D reconstruction (top-down, azimuth=180°, elevation=60°) confirms the spatial relations:

- - The camera markers’ field of view confirms that cam1 faces towards the top edge of the box (where the door and wardrobes are).
- - Cam3, 180° from cam1, is facing towards the bottom edge, where the large window and sofa are located. If you are standing at cam1's spot, facing the same way as image 1 (towards the door/wardrobes), your back is towards the opposite direction — i.e., towards the window and sofa, as seen in cam3 and in the 3D model. Thus, the object behind you while facing image 1's direction is the window. </think> <answer>C. Window</answer>

###### Fig. 19: The Mindcube example.

###### Question:

Is the camera moving left or right when shooting the video? Select from the following options. (A) left (B) right

Assistant: <think>To determine whether the camera is moving left (clockwise) or right (counter-clockwise) around the object, I need to analyze the change in perspective between the two images. The key object is the mug with the bicycle and Chinese text. In the first image, the handle of the mug is on the right side, and in the second image, the handle is more visible and appears to have rotated further to the right, suggesting the camera has moved around the mug. To confirm the direction, I should examine the relative movement of background objects and the mug's orientation. A top-down view from 3D reconstruction will help clarify the camera's path around the mug and the direction of movement. </think> <tool_call>{"name": "pi3_tool", "arguments": {"image_path": ["dataset/BLINK_images/Multiview_Reasoning_val_000101_img1.jpg", "dataset/BLINK_images/Multi-view_Reasoning_val_000101_img2.jpg"], "azimuth_angle": 0, "elevation_angle": 60, "rotation_reference_camera": 1, "camera_view": false}}</tool_call>

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

(0, 60) (90, 0)

Assistant: <think>From the current 3D reconstruction (azimuth=60°, elevation=60°), I can see the relative positions of cam1 and cam2. Both cameras are positioned to the left of the scene (from the viewer's perspective), with cam2 slightly to the right of cam1. The mug's handle is more visible in the second image (cam2), indicating the camera moved to reveal more of the right side of the mug. The background objects (bottles and basket) also shift slightly rightward in the second image, consistent with a rightward (counter-clockwise) camera movement around the object. However, to be absolutely certain of the direction, I want to see the scene from a more horizontal viewpoint, which will clarify the arc of the camera movement. I will request a new viewpoint from azimuth=90°, elevation=0°, which gives a direct side view of the camera positions and their movement path.</think> Assistant: <think>Let's synthesize all the evidence:

- 1. **2D Image Comparison**:

- - In the first image, the mug's handle is barely visible, mostly hidden behind the mug.
- - In the second image, the handle is clearly visible on the right side of the mug.
- - The background objects (bottles, basket) shift slightly to the right in the second image, indicating the camera moved to reveal

more of the right side of the scene.

- 2. **3D Reconstruction (azimuth=90°, elevation=0°)**:

- - The top-down view shows cam1 (first image) is positioned slightly to the left of cam2 (second image).
- - The camera movement path is from cam1 (left) to cam2 (right), both facing the scene.
- - The arc of movement is consistent with a rightward (counter-clockwise) motion around the object.

- 3. **Spatial Reasoning**:

- - If the camera moves rightward around the mug, the handle (which is on the right side of the mug from the initial viewpoint) becomes more visible.
- - The 3D visualization confirms the camera's trajectory is from left to right.

**Conclusion**: The camera is moving to the right (counter-clockwise) around the object, as evidenced by both the change in handle visibility and the 3D camera positions.

</think> <answer>(B) right</answer>

###### Question:

Is the camera moving left or right when shooting the video? Select from the following options.

- (A) left (B) right

###### Assistant: <think>

- Step 1: Analyze both images for camera position and orientation changes.

- - Both images show a bowl and instant noodle containers on a shelf in a store.
- - In the first image, the camera is positioned to the right of the bowl and shelf, showing the aisle and products on the right side.
- - In the second image, the camera is more to the left of the bowl and shelf, facing more directly toward the front of the shelf, with the aisle less visible and the price tag more prominent.

- Step 2: Infer camera movement direction.

- - The camera starts from a position showing more of the aisle (right side of the shelf) and moves to a position showing more of the front of the shelf (left side).
- - This suggests the camera is moving left (clockwise around the object).

- Step 3: Use 3D reconstruction for confirmation.

- - To confirm, I will generate a top-down view (elevation=60°) to clearly see the relative camera positions and movement direction. </think> <tool_call> {"name": "pi3_tool", "arguments": {"image_path": ["dataset/BLINK_images/Multi-view_Reasoning_val_000119_img1.jpg", "dataset/BLINK_images/Multi-view_Reasoning_val_000119_img2.jpg"], "azimuth_angle": 0, "elevation_angle": 60, "rotation_reference_camera": 1, "camera_view": false}} </tool_call>

Assistant: <think> Let's analyze the 3D reconstruction and the original images to clarify the camera movement direction.

- - In the top-down 3D view (azimuth=0°, elevation=60°), Cam1 (red) and Cam2 (blue) are positioned along the X axis, both facing toward the shelf with the bowls and instant noodles.
- - Cam1 is further to the left (lower X value), Cam2 is to the right (higher X value).
- - Comparing with the 2D images:
- - Image 1 (Cam1): Shows more of the aisle on the right side of the shelf, with the bowl in the foreground and the instant

[Figure 63]

[Figure 64]

[Figure 65]

noodles stacked to the left.

- Image 2 (Cam2): Shows more of the front of the shelf, with less aisle visible, and the price tag in front.

From the 3D view, the camera moves from Cam1 (left side of the shelf, showing the aisle) to Cam2 (right side, facing the front of the shelf). This means the camera is moving to the right (counter-clockwise around the shelf).

The 3D reconstruction confirms that the camera started on the left and ended on the right, moving counter-clockwise (right) around the object. </think> <answer>(B) right</answer>

Question: Measuring from the closest point of each object, which of these objects (heater, plant, chair, ceiling light) is the closest to the pillow?

- A. heater
- B. plant
- C. chair
- D. ceiling light

Assistant: <think> To answer which object (heater, plant, chair, ceiling light) is closest to the pillow (measuring from the closest point of each object), I first need to identify all four objects and the pillow in the frames. The pillow is clearly visible in frames 2, 3, 6, and 7. The plant (a green bush in a brown box) is visible in frame 6, right behind the pillow. The chair is visible in frames 2, 6, and 7, positioned near the foot/side of the bed and wall. The heater (radiator under the window) is visible in frame 1. The ceiling light (track light) is seen at the top of frame 1 but is physically far from the pillow, as it is on the ceiling. To determine proximity, I want a top-down view that can capture all objects’ spatial relationship. A top view (elevation=60°) will best reveal their placements relative to the pillow, especially between heater, plant, and chair. </think> <tool_call> {"name": "pi3x_tool", "arguments": {"image_path": ["temp_frames/scannetpp_bcd2436daf_frame_0.jpg", "temp_frames/scannetpp_bcd2436daf_frame_1.jpg", "temp_frames/scannetpp_bcd2436daf_frame_2.jpg", "temp_frames/scannetpp_bcd2436daf_frame_3.jpg", "temp_frames/scannetpp_bcd2436daf_frame_4.jpg", "temp_frames/scannetpp_bcd2436daf_frame_5.jpg", "temp_frames/scannetpp_bcd2436daf_frame_6.jpg"], "azimuth_angle": 0, "elevation_angle": 60, "rotation_reference_camera": 1, "camera_view": false}} </tool_call>

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Assistant: <think> The top-down (azimuth=0°, elevation=60°) point cloud image gives a clear layout of the room, helping to assess distances from each object to the pillow. Here’s what is visible:

- - The pillow is at the left side of the bed.
- - The heater is positioned below the window, on the far wall relative to the pillow.
- - The plant (green foliage in brown box) and chair are situated along the wall at the head of the bed, nearer to the pillow, with the plant directly above the pillow’s head-board.
- - The ceiling light is overhead, but from top view it appears not particularly close horizontally. From the 3D view and 2D frames:
- - Plant: Frame 6 shows the plant is immediately above or behind the headboard next to the pillow, with virtually no gap. In the point cloud, it is directly above the pillow’s closest edge.
- - Chair: It is beside the plant, but spatially further—frames 2 and 6 confirm it is not as close as the plant and separated by a gap.
- - Heater: Clearly further, at the opposite wall relative to pillow.
- - Ceiling light: On the ceiling, not horizontally adjacent.

The top-down point cloud confirms the plant is nearest to the pillow (measured from closest point of each object). I do not need another angle to make the determination. </think> <answer>

- (B) B. plant </answer>

###### Fig. 22: The VSI-Bench example.

Question: Measuring from the closest point of each object, which of these objects (heater, plant, chair, ceiling light) is the closest to the pillow? A. heater B. plant C. chair D. ceiling light

Assistant: <think> To answer the question, I need to:

- 1. Identify the window and the clock in the room.
- 2. Determine the position where a person would stand "by the window and facing the clock."
- 3. Locate the trash bin relative to this position and orientation.
- 4. Use the Cartesian quadrant analogy to determine if the trash bin is front-left, front-right, back-left, or back-right. From the images:

- - The window is visible in frames 0 and 1 (right side of the room, above the desk).
- - The clock is visible in frames 5 and 6 (on the wall above the desk area).
- - The trash bin is visible in frame 0 (under the desk with the computer, next to the blue chair).

To confirm the spatial relationships, I will use a top-down (bird's-eye) view to see the layout of the room and the relative positions of the window, clock, and trash bin. </think> <tool_call> {"name": "pi3x_tool", "arguments": {"image_path": ["temp_frames/scannet_scene0378_01_frame_0.jpg", "temp_frames/scannet_scene0378_01_frame_1.jpg", "temp_frames/scannet_scene0378_01_frame_2.jpg", "temp_frames/scannet_scene0378_01_frame_3.jpg", "temp_frames/scannet_scene0378_01_frame_4.jpg", "temp_frames/scannet_scene0378_01_frame_5.jpg", "temp_frames/scannet_scene0378_01_frame_6.jpg"], "azimuth_angle": 0, "elevation_angle": 60, "rotation_reference_camera": 1, "camera_view": false}} </tool_call>

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Assistant: <think> The top-down 3D reconstruction (azimuth=0°, elevation=60°) gives a clear layout of the room. The window is along the upper right wall, and the clock is on the wall above the desk area (visible in frames 5 and 6). The trash bin is under the desk with the computer, next to the blue chair (visible in frame 0).

To answer the question, I need to confirm the spatial relationship from another angle, ideally from the left or right, to see the depth and relative positions more clearly. This will help verify if the trash bin is in front-left, front-right, back-left, or back-right when standing by the window and facing the clock.

I will request a Pi3X visualization from the left side (azimuth = -90°, elevation = 45°) to better understand the spatial arrangement. </think> <tool_call> {"name": "pi3x_tool",

"arguments": {

"image_path": ["..."], "azimuth_angle": -90, "elevation_angle": 45, "rotation_reference_camera": 1, "camera_view": false}} </tool_call>

Assistant: <think> Let's synthesize the spatial information:

- - The window is visible in frames 0, 1, and 2, and is along the right wall in the top-down 3D reconstructions.
- - The clock is above the desk with the computer (frames 5, 6, and 0), and is on the wall opposite the window.
- - The trash bin is under the desk with the computer (frame 0), which is adjacent to the window.

If you are standing by the window and facing the clock, you are facing away from the window toward the opposite wall. The desk with the computer (and the trash bin) is directly to your right, as seen in the 3D reconstructions (especially the azimuth = -90°, elevation = 45° view).

In Cartesian terms:

- - Facing the clock (positive y-axis), the window is behind you.
- - The trash bin (under the computer desk) is to your right and slightly forward.

Therefore, the trash bin is in the front-right quadrant relative to your position. </think> <answer>B. front-right</answer>

###### Fig. 23: The VSI-Bench example.

### References

- 1. Abdolmaleki, A., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Balakrishna, A., Batchelor, N., Bewley, A., Bingham, J., Bloesch, M., et al.: Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342 (2025)
- 2. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023)
- 3. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631

(2025)

- 4. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923

(2025)

- 5. Balazadeh, V., Ataei, M., Cheong, H., Hosein Khasahmadi, A., Krishnan, R.G.: Synthetic vision: Training vision-language models to understand physics. arXiv e-prints pp. arXiv–2412 (2024)
- 6. Cai, W., Ponomarenko, I., Yuan, J., Li, X., Yang, W., Dong, H., Zhao, B.: Spatialbot: Precise spatial understanding with vision language models. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 9490–9498. IEEE (2025)
- 7. Chen, B., Yue, Z., Chen, S., Wang, Z., Liu, Y., Li, P., Wang, Y.: Lvagent: Long video understanding by multi-round dynamical collaboration of mllm agents. arXiv preprint arXiv:2503.10200 (2025)
- 8. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14455–14465 (2024)
- 9. Chen, Y., Shen, Y., Huang, W., Zhou, S., Lin, Q., Cai, X., Yu, Z., Bu, J., Shi, B., Qiao, Y.: Learning only with images: Visual reinforcement learning with reasoning, rendering, and visual feedback. arXiv preprint arXiv:2507.20766 (2025)
- 10. Chen, Z., Lu, X., Zheng, Z., Li, P., He, L., Zhou, Y., Shao, J., Zhuang, B., Sheng, L.: Geometrically-constrained agent for spatial reasoning. arXiv preprint arXiv:2511.22659 (2025)
- 11. Chen, Z., Zhang, M., Yu, X., Luo, X., Sun, M., Pan, Z., Feng, Y., Pei, P., Cai, X., Huang, R.: Think with 3d: Geometric imagination grounded spatial reasoning from limited views. arXiv preprint arXiv:2510.18632 (2025)
- 12. Cheng, A.C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., Liu, S.: Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems 37, 135062–135093 (2024)
- 13. Chow, W., Mao, J., Li, B., Seita, D., Guizilini, V., Wang, Y.: Physbench: Benchmarking and enhancing vision-language models for physical world understanding. arXiv preprint arXiv:2501.16411 (2025)
- 14. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon,

I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

- 15. Dong, G., Mao, H., Ma, K., Bao, L., Chen, Y., Wang, Z., Chen, Z., Du, J., Wang, H., Zhang, F., et al.: Agentic reinforced policy optimization. arXiv preprint arXiv:2507.19849 (2025)

- 16. Fan, Y., He, X., Yang, D., Zheng, K., Kuo, C.C., Zheng, Y., Narayanaraju, S.J., Guan, X., Wang, X.E.: Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879 (2025)
- 17. Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang, K., Qu, H., Wang, D., Yan, Z., et al.: Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279 (2025)
- 18. Feng, J., Zeng, J., Long, Q., Chen, H., Zhao, J., Xi, Y., Zhou, Z., Yuan, Y., Wang, S., Zeng, Q., et al.: A survey of large language model-powered spatial intelligence across scales: Advances in embodied agents, smart cities, and earth science. arXiv preprint arXiv:2504.09848 (2025)
- 19. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive. In: European Conference on Computer Vision. pp. 148–166. Springer (2024)
- 20. Han, Y., Chi, C., Zhou, E., Rong, S., An, J., Wang, P., Wang, Z., Sheng, L., Zhang, S.: Tiger: Tool-integrated geometric reasoning in vision-language models for robotics. arXiv preprint arXiv:2510.07181 (2025)
- 21. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)
- 22. Ji, Y., Tan, H., Shi, J., Hao, X., Zhang, Y., Zhang, H., Wang, P., Zhao, M., Mu, Y., An, P., et al.: Robobrain: A unified brain model for robotic manipulation from abstract to concrete. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1724–1734 (2025)
- 23. Keetha, N., Müller, N., Schönberger, J., Porzi, L., Zhang, Y., Fischer, T., Knapitsch, A., Zauss, D., Weber, E., Antunes, N., et al.: Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414 (2025)
- 24. Lee, J., Choi, Y., Choi, H., Kim, H., Kim, S.: A training-free, task-agnostic framework for enhancing mllm performance on high-resolution images. arXiv preprint arXiv:2507.10202 (2025)
- 25. Lee, P.Y., Je, J., Park, C., Uy, M.A., Guibas, L., Sung, M.: Perspective-aware reasoning in vision-language models via mental imagery simulation. arXiv preprint arXiv:2504.17207 (2025)
- 26. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r. In: European Conference on Computer Vision. pp. 71–91. Springer (2024)
- 27. Li, G., Hammoud, H., Itani, H., Khizbullin, D., Ghanem, B.: Camel: Communicative agents for" mind" exploration of large language model society. Advances in neural information processing systems 36, 51991–52008 (2023)
- 28. Lin, Y., Li, Y., Chen, D., Xu, W., Clark, R., Torr, P.: Olympus: A universal task router for computer vision tasks. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 14235–14246 (2025)
- 29. Liu, B., Dong, Y., Wang, Y., Ma, Z., Tang, Y., Tang, L., Rao, Y., Ma, W.C., Krishna, R.: Coarse correspondences boost spatial-temporal reasoning in multimodal language model. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 3783–3792 (2025)
- 30. Liu, J., Wang, H., Zhang, Y., Luo, X., Hu, J., Liu, Z., Xie, M.: Insightx agent: An lmm-based agentic framework with integrated tools for reliable x-ray ndt analysis. arXiv preprint arXiv:2507.14899 (2025)
- 31. Liu, S., Cheng, H., Liu, H., Zhang, H., Li, F., Ren, T., Zou, X., Yang, J., Su, H., Zhu, J., et al.: Llava-plus: Learning to use tools for creating multimodal agents. In: European conference on computer vision. pp. 126–142. Springer (2024)

- 32. Luo, Z., Zhang, C., Yong, S., Dai, C., Wang, Q., Ran, H., Shi, G., Sycara, K., Xie, Y.: pyspatial: Generating 3d visual programs for zero-shot spatial reasoning. In: The Fourteenth International Conference on Learning Representations (2026), https://openreview.net/forum?id=yv15C8ql24
- 33. Lyu, X., Liang, Y., Chen, W., Ding, M., Yang, J., Huang, G., Zhang, D., He, X., Shen, L.: Wsi-agents: A collaborative multi-agent system for multi-modal whole slide image analysis. arXiv preprint arXiv:2507.14680 (2025)
- 34. Madaan, A., et al.: Self-refine: Iterative refinement with self-feedback. NeurIPS

(2023)

- 35. Majumdar, A., Ajay, A., Zhang, X., Putta, P., Yenamandra, S., Henaff, M., Silwal, S., Mcvay, P., Maksymets, O., Arnaud, S., et al.: Openeqa: Embodied question answering in the era of foundation models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 16488–16498 (2024)
- 36. Marsili, D., Agrawal, R., Yue, Y., Gkioxari, G.: Visual agentic ai for spatial reasoning with a dynamic api. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 19446–19455 (2025)
- 37. OpenAI: Introducing gpt-4.1 in the api (2025), https://openai.com/index/gpt4-1
- 38. QwenTeam: Qwen3-vl: Sharper vision, deeper thought, broader action (2025), https : / / qwen . ai / blog ? id = 99f0335c4ad9ff6153e517418d48535ab6d8afef & from=research.latest-advancements-list
- 39. Roy, R., Das, D., Banerjee, A., Bhattacharjee, A., Dasgupta, K., Tripathi, S.: Bydeway: Boost your multimodal llm with depth prompting in a training-free way. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6058–6064 (2025)
- 40. Schonberger, J.L., Frahm, J.M.: Structure-from-motion revisited. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4104–4113

(2016)

- 41. Seed, B., Chen, J., Fan, T., Liu, X., Liu, L., Lin, Z., Wang, M., Wang, C., Wei, X., Xu, W., et al.: Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914 (2025)
- 42. Shao, H., Qian, S., Xiao, H., Song, G., Zong, Z., Wang, L., Liu, Y., Li, H.: Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems 37, 8612–8642 (2024)
- 43. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024)
- 44. Shen, Y., Song, K., Tan, X., Li, D., Lu, W., Zhuang, Y.: Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems 36, 38154–38180 (2023)
- 45. Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., et al.: Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617 (2025)
- 46. Su, Z., Xia, P., Guo, H., Liu, Z., Ma, Y., Qu, X., Liu, J., Li, Y., Zeng, K., Yang, Z., et al.: Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918 (2025)
- 47. Surís, D., Menon, S., Vondrick, C.: Vipergpt: Visual inference via python execution for reasoning. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11888–11898 (2023)

- 48. Taguchi, S., Deguchi, H., Hamazaki, T., Sakai, H.: Spatialprompting: Keyframedriven zero-shot spatial reasoning with off-the-shelf multimodal large language models. arXiv preprint arXiv:2505.04911 (2025)
- 49. Tang, H., Cao, M., Liu, R., Liang, X., Li, L., Li, G., Liang, X.: Video spatial reasoning with object-centric 3d rollout. arXiv preprint arXiv:2511.13190 (2025)
- 50. Tang, Z., Wang, S., Cho, J., Yoo, J., Sun, C.: How can objects help video-language understanding? arXiv preprint arXiv:2504.07454 (2025)
- 51. Team, B.R., Cao, M., Tan, H., Ji, Y., Chen, X., Lin, M., Li, Z., Cao, Z., Wang, P., Zhou, E., et al.: Robobrain 2.0 technical report. arXiv preprint arXiv:2507.02029

(2025)

- 52. Team, G.R., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Armstrong, T., Balakrishna, A., Baruch, R., Bauza, M., Blokzijl, M., et al.: Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020 (2025)
- 53. Team, V., Hong, W., Yu, W., Gu, X., Wang, G., Gan, G., Tang, H., Cheng, J., Qi, J., Ji, J., et al.: Glm-4.5 v and glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025. URL https://arxiv. org/abs/2507.01006
- 54. Wake, N., Kanehira, A., Sasabuchi, K., Takamatsu, J., Ikeuchi, K.: Gpt-4v (ision) for robotics: Multimodal task planning from human demonstration. IEEE Robotics and Automation Letters (2024)
- 55. Wang, C., Luo, W., Dong, S., Xuan, X., Li, Z., Ma, L., Gao, S.: Mllm-tool: A multimodal large language model for tool agent learning. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 6678–6687. IEEE (2025)
- 56. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)
- 57. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10510–10522 (2025)
- 58. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 20697–20709 (2024)
- 59. Wang, Y., Zhou, J., Zhu, H., Chang, W., Zhou, Y., Li, Z., Chen, J., Pang, J., Shen, C., He, T.: pi3: Scalable permutation-equivariant visual geometry learning. arXiv e-prints pp. arXiv–2507 (2025)
- 60. Wang, Y., Wang, S., Cheng, Q., Fei, Z., Ding, L., Guo, Q., Tao, D., Qiu, X.: Visuothink: Empowering lvlm reasoning with multimodal tree search. arXiv preprint arXiv:2504.09130 (2025)
- 61. Wang, Z., Guo, X., Stoica, S., Xu, H., Wang, H., Ha, H., Chen, X., Chen, Y., Yan, M., Huang, F., et al.: Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448 (2025)
- 62. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837 (2022)
- 63. Wu, C., Yin, S., Qi, W., Wang, X., Tang, Z., Duan, N.: Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671 (2023)
- 64. Wu, D., Liu, F., Hung, Y.H., Duan, Y.: Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747 (2025)

- 65. Wu, H., Huang, X., Chen, Y., Zhang, Y., Wang, Y., Xie, W.: Spatialscore: Towards unified evaluation for multimodal spatial understanding. arXiv preprint arXiv:2505.17012 (2025)
- 66. Wu, J., Guan, J., Feng, K., Liu, Q., Wu, S., Wang, L., Wu, W., Tan, T.: Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing. arXiv preprint arXiv:2506.09965 (2025)
- 67. Wu, M., Yang, J., Jiang, J., Li, M., Yan, K., Yu, H., Zhang, M., Zhai, C., Nahrstedt, K.: Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255 (2025)
- 68. Wu, Y., Wang, Y., Tang, S., Wu, W., He, T., Ouyang, W., Torr, P., Wu, J.: Dettoolchain: A new prompting paradigm to unleash detection ability of mllm. In: European Conference on Computer Vision. pp. 164–182. Springer (2024)
- 69. Yang, J., Yang, S., Gupta, A.W., Han, R., Fei-Fei, L., Xie, S.: Thinking in space: How multimodal large language models see, remember, and recall spaces. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10632– 10643 (2025)
- 70. Yang, S., Li, J., Lai, X., Yu, B., Zhao, H., Jia, J.: Visionthink: Smart and efficient vision language model via reinforcement learning. arXiv preprint arXiv:2507.13348

(2025)

- 71. Yang, Y., Liu, J., Zhang, Z., Zhou, S., Tan, R., Yang, J., Du, Y., Gan, C.: Mindjourney: Test-time scaling with world models for spatial reasoning. arXiv preprint arXiv:2507.12508 (2025)
- 72. Yang, Z., Chen, D., Yu, X., Shen, M., Gan, C.: Vca: Video curious agent for long video understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20168–20179 (2025)
- 73. Yang, Z., Li, L., Wang, J., Lin, K., Azarnasab, E., Ahmed, F., Liu, Z., Liu, C., Zeng, M., Wang, L.: Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381 (2023)
- 74. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K.R., Cao, Y.: React: Synergizing reasoning and acting in language models. In: The eleventh international conference on learning representations (2022)
- 75. Yin, B., Wang, Q., Zhang, P., Zhang, J., Wang, K., Wang, Z., Zhang, J., Chandrasegaran, K., Liu, H., Krishna, R., et al.: Spatial mental modeling from limited views. In: Structural Priors for Vision Workshop at ICCV’25 (2025)
- 76. Yu, S., Chen, Y., Ju, H., Jia, L., Zhang, F., Huang, S., Wu, Y., Cui, R., Ran, B., Zhang, Z., et al.: How far are vlms from visual spatial intelligence? a benchmarkdriven perspective. arXiv preprint arXiv:2509.18905 (2025)
- 77. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9556– 9567 (2024)
- 78. Zhang, H., Liu, M., Li, Z., Wen, H., Guan, W., Wang, Y., Nie, L.: Spatial understanding from videos: Structured prompts meet simulation data. arXiv preprint arXiv:2506.03642 (2025)
- 79. Zhang, X., Jia, Z., Guo, Z., Li, J., Li, B., Li, H., Lu, Y.: Deep video discovery: Agentic search with tool use for long-form video understanding. arXiv preprint arXiv:2505.18079 (2025)
- 80. Zhao, H., Liu, A., Zhang, Z., Wang, W., Chen, F., Zhu, R., Haffari, G., Zhuang, B.: Cov: Chain-of-view prompting for spatial reasoning. arXiv preprint arXiv:2601.05172 (2026)

- 81. Zhao, S., Zhang, H., Lin, S., Li, M., Wu, Q., Zhang, K., Wei, C.: Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998 (2025)
- 82. Zhao, Y., Huang, J., Hu, J., Wang, X., Mao, Y., Zhang, D., Jiang, Z., Wu, Z., Ai, B., Wang, A., Zhou, W., Chen, Y.: Swift:a scalable lightweight infrastructure for fine-tuning (2024), https://arxiv.org/abs/2408.05517
- 83. Zheng, W., Mao, X., Ye, N., Li, P., Zhan, K., Lang, X., Zhao, H.: Driveagent-r1: Advancing vlm-based autonomous driving with hybrid thinking and active perception. arXiv e-prints pp. arXiv–2507 (2025)
- 84. Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., Yu, X.: Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362 (2025)
- 85. Zhou, E., An, J., Chi, C., Han, Y., Rong, S., Zhang, C., Wang, P., Wang, Z., Huang, T., Sheng, L., et al.: Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308 (2025)
- 86. Zhou, E., Chi, C., Li, Y., An, J., Zhang, J., Rong, S., Han, Y., Ji, Y., Liu, M., Wang, P., et al.: Robotracer: Mastering spatial trace with reasoning in visionlanguage models for robotics. arXiv preprint arXiv:2512.13660 (2025)
- 87. Zhou, G., Hong, Y., Wu, Q.: Navgpt: Explicit reasoning in vision-and-language navigation with large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 7641–7649 (2024)
- 88. Zhou, Z., Chen, D., Ma, Z., Hu, Z., Fu, M., Wang, S., Wan, Y., Zhao, Z., Krishna, R.: Reinforced visual perception with tools. arXiv preprint arXiv:2509.01656 (2025)
- 89. Zhu, M., Tian, Y., Chen, H., Zhou, C., Guo, Q., Liu, Y., Yang, M., Shen, C.: Segagent: Exploring pixel understanding capabilities in mllms by imitating human annotator trajectories. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 3686–3696 (2025)

