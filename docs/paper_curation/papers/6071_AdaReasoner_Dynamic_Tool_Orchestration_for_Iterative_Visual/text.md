# arXiv:2601.18631v2[cs.AI]28Jan2026

[Figure 1]

## DAREASONER: DYNAMIC TOOL ORCHESTRATION FOR ITERATIVE VISUAL REASONING

Mingyang Song∗,1, Haoyu Sun∗,2, Jiawei Gu∗,3, Linjie Li∗,4, Luxin Xu5, Ranjay Krishna4,†, Yu Cheng6,†

1Fudan University, 2Tongji University, 3National University of Singapore, 4University of Washington, 5University of Electronic Science and Technology of China, 6The Chinese University of Hong Kong

Homepage: https://adareasoner.github.io Code: https://github.com/ssmisya/AdaReasoner Models and Data: https://huggingface.co/AdaReasoner

[Figure 2]

[Figure 3]

[Figure 4]

ABSTRACT

When humans face problems beyond their immediate capabilities, they rely on tools, providing a promising paradigm for improving visual reasoning in multimodal large language models (MLLMs). Effective reasoning, therefore, hinges on knowing which tools to use, when to invoke them, and how to compose them over multiple steps, even when faced with new tools or new tasks. We introduce AdaReasoner, a family of multimodal models that learn tool use as a general reasoning skill rather than as tool-specific or explicitly supervised behavior. AdaReasoner is enabled by (i) a scalable data curation pipeline exposing models to long-horizon, multi-step tool interactions; (ii) Tool-GRPO, a reinforcement learning algorithm that optimizes tool selection and sequencing based on end-task success; and (iii) an adaptive learning mechanism that dynamically regulates tool usage. Together, these components allow models to infer tool utility from task context and intermediate outcomes, enabling coordination of multiple tools and generalization to unseen tools. Empirically, AdaReasoner exhibits strong tool-adaptive and generalization behaviors: it autonomously adopts beneficial tools, suppresses irrelevant ones, and adjusts tool usage frequency based on task demands, despite never being explicitly trained to do so. These capabilities translate into state-of-the-art performance across challenging benchmarks, improving the 7B base model by +24.9% on average and surpassing strong proprietary systems such as GPT-5 on multiple tasks, including VSP and Jigsaw.

1 INTRODUCTION

Humans often rely on external tools to solve the complex reasoning problems that go beyond what they can handle through internal thinking alone (Clark & Chalmers, 1998). This offers a valuable perspective for improving the visual reasoning capabilities of Multimodal Large Language Models (MLLMs). By offloading perceptual and intermediate computation to external tools, the model can more effectively handle fine-grained visual perception (Tong et al., 2024) and multi-step reasoning (Hao et al., 2025), where precise intermediate verification and long-horizon planning are crucial. In this setting, adaptive tool usage becomes particularly important. MLLMs should learn not only how to use tools, but also when to use them and how to coordinate multiple tools over several steps, including tools it has never seen before.

However, unlike humans who can flexibly decide when to resort to tools and which tools to use, current models still struggle to master adaptive tool usage. Early SFT- and prompt-based approaches (Ma et al., 2024; Hu et al., 2024) explored multi-tool usage, but largely relied on rigid,

∗Equal contribution. †Equal Advisory Contribution

[Figure 5]

Adaptive Adopt and Discard Tools

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Generalize to New Tools

[Figure 12]

[Figure 13]

[Figure 14]

<think>… Player at (4,1). Goal at (1,2). Pathfinding... 1.Up, 2.Up, 3.Right... </think> (No tools called) <response> … The path will be returned as a sequence of moves: L (left), R (right), U (up), D (down). \boxed{R,R,U,L,D,D} <response>

Q: Guide the player from start to goal, while avoiding holes.

[Figure 15]

[Figure 16]

[Figure 17]

<think>… I will use the `Point` tool again to locate the Gift in the image.

[Figure 18]

[Figure 19]

… I will use the

`AStarWithPixelCoordinate`

tool to find the shortest path. … I have received the path 'U,U,U,R' from the `AStarWith PixelCoordinate` tool. …,

[Figure 20]

[Figure 21]

NEW Tools in Context

[Figure 22]

[Figure 23]

[Figure 24]

I will use the `Draw2DPath` tool. I will draw this path on the original image.

[Figure 25]

[Figure 26]

Figure 1: AdaReasoner performs adaptive and generalized tool-using.

pre-defined invocation patterns rather than autonomous and adaptive planning. More recent RLbased methods, such as DeepEyes (Zheng et al., 2025) and Pixel-Reasoner (Su et al., 2025b), have improved perceptual reasoning through crop-based search strategies. However, they are typically constrained to single-tool trajectories or fixed interaction loops. As a result, a critical gap remains: existing methods lack the ability to adaptively plan and coordinate diverse tools in a flexible and task-aware manner. They do not treat the decision of what tools to use, when to use them, and how to combine them as a core component of multimodal reasoning. Moreover, because these approaches are not explicitly designed for adaptability, their tool-use policies tend to be brittle, exhibiting limited generalization to unseen tools or novel task distributions beyond their training scope.

We present AdaReasoner, a tool-aware reasoning model designed to overcome the limitations of rigid, single-tool paradigms and poor generalization by enabling adaptive, multi-turn tool planning. Our framework is built upon three key innovations. First, to establish a robust foundation, we introduce a new data curation pipeline that automatically synthesizes high-quality, multi-turn trajectories. Second, we further refine this policy using a tailored Tool GRPO (TG) paradigm optimized for long-horizon strategic planning. This training strategy enables the model to reason over extended interaction sequences and make more coherent tool-use decisions. Finally, integrated within both the TC and TG stages is our proposed adaptive learning method (ADL), which is specifically engineered to decouple tool-use logic from specific tasks, thereby significantly enhancing the model’s generalizability to unseen domains. As illustrated in Figure 1, this active reasoning cycle enables the agent to not only extract visual evidence but to dynamically transform it, yielding a deeper and more robust multimodal understanding.

Through adaptive tool interaction, AdaReasoner achieves substantial performance gains across diverse benchmarks. In particular, the 7B variant attains an average improvement of +24.9%, while also surpassing strong proprietary models – outperforming Claude Sonnet 4 and GPT-5 on multiple tasks. Beyond accuracy, as shown in Figure 1, AdaReasoner also exhibits self-adaptive tool-use behaviors. It learns to select effective tools, discard irrelevant ones, and regulate their use according to task demands and feedback, revealing strong flexibility and generalization. Moreover, AdaReasoner is able to generalize its tool-planning capability to unseen tasks and new tool definitions, maintaining both a high frequency and strong accuracy of tool usage even when encountering new tasks. These findings offer a compelling answer to the long-standing question of which tools should be included and how models should learn to use them, suggesting that with proper training, MLLMs can autonomously curate tool-use strategies from a broad candidate set and extend their visual reasoning capacity in a goal-directed manner. In summary, our main contributions are as follows:

- • We propose a comprehensive method for developing tool-augmented models, built upon three core innovations: a data curation recipe for multi-turn tool planning, an RL framework for multi-turn tool interaction, and an adaptive learning method that can enhance model’s generalizability.
- • Based on our method, we introduce AdaReasoner, a new family of state-of-the-art models for complex tool planning. AdaReasoner exhibits self-adaptive behaviors, learning to autonomously adopt beneficial tools, discard irrelevant ones, and modulate its usage frequency, while maintaining strong generalization to unseen tool definitions and novel tasks.
- • Our AdaReasoner achieves significant gains over their base counterparts and delivers performance that is competitive with, or superior to, leading proprietary models like GPT-5 and Claude Sonnet 4 on structured-reasoning tasks. This establishes that our methodology can elevate smaller, opensource models to the state-of-the-art.

- 2 METHOD

- 2.1 PRELIMINARY

Problem Formulation As shown in figure 2, we formalize tool-augmented multimodal reasoning as a sequential decision-making process. An MLLM represented as a policy πθ parameterized by weights θ, is tasked with solving a problem by generating a reasoning trajectory τ. The policy is equipped with access to a predefined set of visual tools T = {t1,...,tn}.

A trajectory τ is a sequence of state-action-observation tuples that represent the model’s step-by-step reasoning process:

τ = {(s0, a0, o0), (s1, a1, o1), . . . , (sT, aT, oT)} (1)

Here, st denotes the problem state, at ∈ T is a tool-calling action encapsulated by special tokens, and ot is the resulting observation from the tool’s execution. Each action at induces a transition from state st to st+1 based on the new information in ot:

s0 −→a0 s1 −→a1 s2 −→a2 . . . −−→aT sT+1 (2)

Visual Tools Our AdaReasoner framework is built upon a diverse and powerful suite of visual tools, which it executes and integrates directly into the reasoning process. This toolset is intentionally designed to cover three core reasoning functions: perception (e.g., POINT, OCR), manipulation (e.g., DRAWLINE, INSERTIMAGE), and calculation (e.g., ASTAR). Furthermore, this suite seamlessly integrates both lightweight, offline tools for immediate execution and computationally intensive, expert-model-based online tools. These foundational capabilities are summarized in Table 1, with detailed specifications for each tool provided in Appendix A.1.

Tool Description Arguments Tool Output

POINT Point to a target object Image + Description Point coordinates DRAW2DPATH Draw a path using directional commands Image + Start + Directions Image with a line ASTAR Use A* to find the shortest obstacle-free path Start + Goal + Obstacle Shortest path DETECTBLACKAREA Detect pure black areas in an image Image Bounding boxes of black areas INSERTIMAGE Insert image into base at bounding box position Image + Coordinates + Insert Combined image OCR Extracts and localizes text from the image Image Text with their bounding box CROP Crop a region and augment it Image + Coordinates Cropped Image

- Table 1: Visual tools integrated within AdaReasoner. We illustrate their arguments, outputs, and core functions description. More detailed descriptions of our tools are presented in Appendix A.1.

- 2.2 HIGH-QUALITY TRAJECTORY DATA CURATION

As illustrated in Figure 2a, our data curation follows a unified, three-stage process designed to generate high-fidelity, human-like reasoning trajectories.

Abstract Trajectory Design First, for each task, we manually design an abstract, optimal problemsolving blueprint. For example, the VSP trajectory follows a perception-planning-verification logic, Jigsaw mimics an iterative trial-and-error process, and GUIQA involves a focus-then-extract strategy. However, to ensure the model develops true robustness beyond simply following these “perfect” paths, we deliberately incorporate two critical types of complex scenarios:

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- (a) TC: Tool Cold Start
- (b) TG: Multi-TurnTool GRPO

(c) Adaptive Learning

[Figure 31]

<think> My previous paths are incorrect ... </think>

<think>I need to call Point to locate the start point.</think>

<think> I devise an initial path… I will use the Draw2DPath tool to visualize </think>

<think> …re-plan a path from (3,3) to (1,2) </think>

Tool Cold Start Tool GRPO

CoT Data

Multi-turn Trajectory

[Figure 32]

[Figure 33]

[Figure 34]

###### T A O T A O T A O

T A O

… … …

Randomized Setting

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

|Name: DrawLine Desc: Draw a continuous path on an image by connecting ... Parameters:<br><br>Start: [int, int] Goal: [int, int] Color: str<br><br>Tool Definition<br><br>Func_X7a2<br><br>Given the coordinat es<br><br>...<br><br>Para_9a7y Para_0c9u Para_4t6p<br><br>|
|---|

[Figure 39]

[Figure 40]

: Think : Action : Observation : Turn

Tool Edited Image

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Overall Reward

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

… …

Advantage

……

[Figure 52]

Policy Model

Group

Zero-shot Adaption

[Figure 53]

- Figure 2: An overview of our AdaReasoner framework. The pipeline consists of two main stages: (a) the Tool Cold Start (TC) phase, where trajectories are carefully constructed to support multi-turn reasoning; and (b) the Tool GRPO (TG) phase, where the policy is further refined via reinforcement learning guided by our adaptive, multi-turn reward. In addition, the Adaptive Learning method (c) can be applied throughout both the TC and TG stages, enabling improved generalization across tasks and tool configurations.

- • Reflection and Backtracking: We include trajectories designed to encourage a process of trial and verification. These feature explicit self-correction steps where the model must reflect on a sub-optimal outcome and backtrack, teaching it to actively validate its own hypotheses and learn from intermediate failures.
- • Explicit Tool Failure: To prevent over-reliance on external tools, we introduce cases where tools fail or return erroneous results. In these scenarios, after recognizing that a tool is not providing a useful output, the trajectory prompts the model to fall back on its own intrinsic capabilities to generate a “best-effort” answer, ensuring it develops a resilient, dual-strategy approach.

Tool Calling supplements Subsequently, we ground these abstract blueprints by programmatically executing the tool calls to populate them with concrete, real-world inputs and outputs.

CoT Data Generation Finally, we leverage a powerful LLM to generate the corresponding Chainof-Thought (CoT) reasoning that connects each step. This process yields a final dataset of rich, toolaugmented trajectories that teach the model not just what tools to call, but why and how to reason between them. Details for our trajectory data curation can be found in Appendix A.2.

- 2.3 MULTI-TURN TOOL GRPO

To train our model for complex multi-turn tool-planning scenarios, we extend the GRPO framework to effectively handle multi-turn tool-calling reasoning trajectories. Concretely, we use Multi-turn Reward Accumulation and Adaptive Tool Reward to ensure the efficacy of the RL procedure.

Multi-turn Reward Accumulation Our total reward, Rtotal is formulated as Rtotal = Rformat ·(λtool · Rtool + λacc · Racc), with each component adapted for multi-turn trajectories τ = {τ0,...,τT}.

- • Format Reward Rformat = ni=1 Rformat(τi) Correct formatting is mandatory at every step. Therefore, the overall format reward for a trajectory is set to 1 if and only if every individual step

within it is correctly formatted. A single format error at any turn results in Rformat = 0, nullifying the entire reward for the trajectory. This enforces strict adherence to the reasoning structure.

- • Tool Reward The overall tool reward is the average of the fine-grained scores from all toolcalling turns (from τ0 to τT−1). It is calculated as Rtool = T1 Tt=0−1 Rtool(τt). Each individual tool call, Rtool(τt), is evaluated using a hierarchical score of 0-4 based on four criteria (Structure, Name, Parameter Name, and Parameter Content).

- • Accuracy Reward This reward is granted only based on the final turn, τT. If the final answer is correct, Racc = 1; otherwise, it is 0.

### Adaptive Reward for Encouraging Tool Use.

To encourage reliable tool usage under uncertainty, we design an adaptive reward with an asymmetric structure. When the final prediction is correct, the model receives a full reward regardless of tool usage, encouraging concise reasoning without unnecessary tool calls. When the prediction is incorrect, the reward depends on the quality of tool usage, granting partial credit to informative tool-based reasoning while penalizing unsupported guessing. This design treats tools as a fallback mechanism under uncertainty rather than a mandatory step, leading to more robust and adaptive decision-making. Details are provided in Appendix A.4.

- 2.4 ADAPTIVE LEARNING FOR IMPROVED GENERALIZATION

To bolster the model’s generalization capabilities when encountering unseen tools and new tasks, we introduce an Adaptive Learning strategy. Concretely, we randomize tool names as well as their parameter names and descriptions, preventing the model from overfitting to fixed identifiers and encouraging it to rely on the semantic understanding of tool functionalities and parameter descriptions when selecting appropriate tools.

Token-Level Randomization for Identifiers. We hypothesize that a robust tool planner should not depend on semantically meaningful identifiers (e.g., relying on the word “Calculator” to know it performs math). To test and enforce this, we apply a replacement policy to all functional identifiers, including tool names and argument names. Specifically, these identifiers are replaced with completely random alphanumeric strings (e.g., replacing GetWeather with Func X7a2). This process strips away the linguistic cues from the identifiers, compelling the model to infer functionality solely from the provided descriptions and context.

Semantic-Level Paraphrasing for Descriptions. While identifiers are randomized, it remains crucial to maintain sufficient diversity in the descriptions while preserving their semantic integrity to ensure learnability. Therefore, for the descriptions of both tools and arguments, we employ a semantic rephrasing strategy. We leverage Gemini 2.5 Flash to rephrase the original descriptions. The objective is to alter the syntactic structure and lexical choices while strictly maintaining the original functional meaning. This creates a diverse set of equivalent tool definitions, preventing the model from overfitting to specific phrasing and enhancing its robustness to the variations in tool documentation encountered in real-world scenarios.

- 3 EXPERIMENTS

In this section, we describe our experimental setup in detail, analyze the impact of tool augmentation on agent planning, investigate the adaptability and generalization of learned tool-using behaviors, and finally evaluate the resulting model across diverse multimodal reasoning tasks.

Our core experiments are based on the Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct models (Bai et al., 2025). We evaluate our approach across four diverse tasks that probe complementary aspects of multimodal reasoning. Specifically, we consider Visual Spatial Planning, which assesses multi-step planning and perceptual grounding, evaluated on our custom out-of-distribution benchmark (VSPO) as well as the standard VSP benchmark (Wu et al., 2024). We also include Jigsaw, which focuses on visual compositionality, evaluated on both our Jigsaw-COCO dataset and the Jigsaw subset of BLINK (Fu et al., 2024). To assess fine-grained visual understanding of GUI scenarios, we further evaluate on GUIQA, using GUIChat (Chen et al., 2024) and WebQA from the WebMMU benchmark (Awal et al., 2025). To better evaluate the effectiveness of our visual tools in GUI-based scenarios, we focus on the agent acting subset of the English split of WebMMU. Detailed settings and implementation details for all tasks are provided in Appendix B.1.

- 3.1 VISUAL TOOLS HELP BRIDGING THE CAPABILITY GAP

We first fine-tune the model on each individual task and systematically evaluate the contribution of TC and TG to the overall performance. We include Direct SFT and Direct GRPO as strong baseline settings. Specifically, Direct SFT refers to supervised fine-tuning of the base model on the training set of each individual task, while Direct GRPO applies rule-based GRPO to the base

model to enhance its reasoning capability following prior work (Zhou et al., 2025). The results are summarized in Table 2.

Model VSPO VSP Jigsaw BLINK-J GUIChat WebMMU†

Qwen2.5-VL 3B 26.53 26.73 39.80 48.67 45.11 55.89 + Direct SFT 38.15 38.82 42.60 53.33 55.51 61.38 + Direct GRPO 24.55 32.73 42.70 52.67 52.49 56.30 + TC (cold-start) 47.01 51.09 66.00 70.00 45.32 44.72 + TG (tool-GRPO) 29.10 35.09 43.00 47.33 89.60 72.15

- + TC + TG 84.73 94.73 94.80 88.67 85.45 81.71

- ∆ vs. base +58.20 +68.00 +55.00 +40.00 +40.34 +25.82 Qwen2.5-VL 7B 25.39 28.09 45.70 52.67 59.46 67.48

+ Direct SFT 42.18 46.64 86.40 88.00 62.68 65.65 + Direct GRPO 28.38 30.18 64.90 80.00 67.67 83.54 + TC (cold-start) 61.69 64.91 84.20 83.33 61.85 64.63 + TG (tool-GRPO) 59.70 73.18 72.30 80.67 92.52 88.62 + TC + TG 85.09 97.64 96.60 96.00 88.57 82.32

- ∆ vs. base +59.70 +69.55 +50.90 +43.33 +29.11 +14.84

- Table 2: Comparison of the Effects of Tool Cold-Start (TC) and Tool-GRPO (TG) under a Single-Task FineTuning Setting. †WebMMU reports the Agentic Action (Act.) score. Best is bold, second-best is underlined. Detailed sub-task breakdown is provided in Appendix Table 11.

Visual Tools Bring Stable Improvements As shown in Table 2, our TC + TG recipe consistently improves the performance of base models, demonstrating an average gain of +38.66% on the 7B model. This tool-augmented approach transforms tasks like VSP from an under-optimized baseline (∼31.64%) to near-perfect execution (97.64%). This performance significantly surpasses traditional optimization methods such as task-specific SFT (46.64%) and Direct GRPO (30.18%). Furthermore, our TC + TG recipe enables the 7B model to achieve the state-of-the-art results that are competitive with, or superior to the best proprietary models. For instance, on VSP and Jigsaw, our model outperforms GPT-5 (OpenAI, 2025) (96.60% vs. 80.10%). This confirms that our method is a highly effective strategy for unlocking advanced reasoning capabilities in open-source models.

Visual Tools Help Overcome Scale-Based Limitations The results also reveal that tool augmentation can redefine the performance ceiling of MLLMs by overcoming scale-based limitations. As illustrated in Table 2 and Figure 10, while the baseline performance of 3B and 7B models is disparate and low, our tool-augmented versions both achieve near-perfect accuracy (94.7% and 97.6%). This suggests that the primary performance bottleneck has shifted from the model’s scale to the intrinsic quality of the tools it employs.

Why Visual Tools Help Our empirical analysis reveals that the effectiveness of visual tools stems from their complementary roles in enhancing perception, verification, and planning. First, expert perception tools compensate for the intrinsic visual limitations of MLLMs by providing precise, structured observations, substantially improving downstream reasoning even in zero-shot settings. Second, manipulation tools enable models to externalize intermediate hypotheses and verify them through explicit visual operations, transforming abstract reasoning into concrete, checkable decisions. Finally, high-quality tool-augmented trajectories further teach models when and how to apply these tools, aligning planning behavior with task structure. Together, these components shift the bottleneck from internal reasoning accuracy to effective tool planning, explaining the consistent and large performance gains observed across tasks. Detailed analyses are provided in Appendix C.2.

- 3.2 ADAREASONER CAN LEARN ADAPTIVE TOOL-USING

To investigate whether MLLMs can effectively learn to select tools and adaptively regulate their usage frequency, we carried out a systematic study to build an adaptive tool planning model, which is the main characteristic of our AdaReasoner. We choose VSP as the primary testbed for this adaptive study, as it comprises multiple interdependent subtasks (e.g., navigation and verification) and requires more complex tool planning decisions involving diverse tool types.

AdaReasoner Can Use New Tools during Inference Time During inference, the model exhibits a notable ability to adapt to previously unseen but powerful tools. To systematically examine this capability, we evaluate whether the model can effectively leverage a strong planning tool, A*, which is deliberately excluded during the Tool Cold Start (TC) phase. As shown in Table 3, when the A*

VSP VSPO A* Statistics Nav Verify Overall Nav Verify Overall CPS Succ.

Stage Reflection A*

TC + TG ✗ RL 96.33 99.20 97.64 73.44 98.70 85.09 0.56 100.0 TC + TG ✓ - 84.33 99.80 91.36 63.89 99.61 80.36 0.00 0.0 TC + TG ✓ Inf 55.17 84.60 68.55 57.22 99.61 76.77 0.68 16.9 TC + TG ✗ Inf 62.33 80.00 70.36 43.78 88.70 64.49 0.52 94.5 TC Only ✗ - 41.00 93.60 64.91 31.58 94.01 61.69 0.00 0.0 TC + TG ✗ - 44.83 94.20 67.27 27.67 94.81 58.62 0.00 0.0 TC Only ✗ Inf 46.00 79.40 61.18 32.11 81.43 54.85 0.49 85.2

- Table 3: Ablation study on adaptive tool usage. Stage compares Tool Cold Start (TC) + Tool GRPO (TG) against TC alone. Reflection indicates training with (✓) or without (✗) reflection data. A* specifies availability: during RL training, at Inference (Inf), or unavailable (-). A* Statistics: CPS = calls per sample, Succ. = success rate (%). Best is bold, second-best is underlined.

tool is introduced solely at inference time (Inf), it provides a significant performance boost to the relevant task. For our standard TC+TG model (without reflection), this elevates the VSP navigation score from 44.83 to 62.33. The A* Statistics corroborate this adaptive behavior, showing a high invocation success rate of 94.53%, which indicates the model is not just guessing but is correctly learning the tool’s syntax and purpose in a zero-shot setting.

However, although the model exhibits adaptive tool invocation behavior at inference time, this adaptability remains unstable and inconsistent. For example, the ASTAR tool does not contribute to verification and instead acts as a distractor in this task. When the tool is made available, the model nevertheless invokes it, leading to a substantial performance drop from 94.20 to 80.00. Similar unstable tool invocation behaviors become more pronounced when reflection is enabled. Therefore, the adaptability to new tools during inference remains unstable and still requires RL to stabilize it.

1.2

3.8

- 0.8

- 1.0

- 1.2

| |
|---|

1.1

3.4

| |
|---|

Callspersample

Callspersample

Callspersample

1.0

3.0

| |
|---|

| |
|---|

| |
|---|

0.9

2.6

Qwen 2.5 VL 3B - Nav Qwen 2.5 VL 7B - Nav

Qwen 2.5 VL 3B - Nav Qwen 2.5 VL 7B - Nav

| |
|---|

0.6

0.8

2.2

Qwen 2.5 VL 3B - Ver Qwen 2.5 VL 7B - Ver

Qwen 2.5 VL 3B - Ver Qwen 2.5 VL 7B - Ver

| |
|---|

0.7

1.8

0.4

| |
|---|

Qwen 2.5 VL 3B - Nav Qwen 2.5 VL 7B - Nav

0.6

1.4

0.2

Qwen 2.5 VL 3B - Ver Qwen 2.5 VL 7B - Ver

0.5

1.0

0.0

0 20 40 60 80 100 120 140 160 180 200

0 20 40 60 80 100 120 140 160 180 200

0 40 80 120 160 200 240 280 320 360 400

Training Steps

Training Steps

Training Steps

(a) ASTAR

(b) POINT

(c) DRAW2DPATH

- Figure 3: Trend for tool calling frequencies for ASTAR, POINT, and DRAW2DPATH during RL. The model is optimized on VSP Verification (cool-color) and VSP Navigation (warm-color) tasks.

Learning to Adopt and Master Beneficial Tools through RL To improve the stability of adaptive tool invocation, we incorporate the A* tool into the tool set during the TG stage. Specifically, we initialize training from the same SFT checkpoint that has never been exposed to A*, and then introduce A* as an available tool during the TG process. As shown in Table 3, incorporating A* during RL leads to more stable tool usage patterns and yields consistently improved performance. In addition, we also observe three notable adaptive behaviors emerging during the TG stage.

- (1) Learning to Adopt Beneficial Tools. As illustrated in Figure 3a, for the Path Navigation task (warm-colored curves), the model’s invocation frequency for ASTAR progressively increases, stabilizing at a high rate of over 1.0 call per sample. This upward trajectory indicates that the model, guided by task-completion rewards, correctly identifies ASTAR as a highly beneficial tool for pathfinding and actively incorporates it into its problem-solving strategy. As a result, this mastery translates to a dramatic performance increase, with the VSP navigation score soaring to 96.33, which achieves the best performance.

- (2) Learning to Discard Irrelevant Tools. Critically, AdaReasoner learns to discard the tool when it is irrelevant. Figure 3a (cool-colored curves) shows the inverse trend for the Verification task. The model initially explores using the A* tool but, receiving no reward for doing so, gradually learns to suppress its usage, with the invocation frequency decaying towards zero. This adaptive pruning prevents the negative degradation observed in the zero-shot inference scenario, allowing the Verification performance to remain at a near-perfect 99.20.

###### Model VSPO VSP Jigsaw BLINK-J GUIChat WebMMU†

Qwen2.5-VL 7B 25.39 28.09 45.70 52.67 68.09 67.48 + TC 27.66 29.36 83.50 82.00 36.59 41.87 + Rnd TC 26.71 30.36 77.90 80.00 35.55 42.89 + TG 24.01 30.73 66.10 82.00 82.54 67.68 + TC + TG 24.43 28.00 59.20 78.00 80.35 59.76 + Rnd TC + TG 57.78 47.27 90.60 91.33 82.74 69.11 + Rnd TC + Rnd TG (Ours) 69.94 78.91 83.50 80.67 80.87 70.93

∆ vs. base +44.55 +50.82 +37.80 +28.00 +12.79 +3.46

- Table 4: Impact of Different Experimental Settings on Model Generalization Performance. Rnd TC and Rnd TG denote the randomized Tool Cold Start and Tool GRPO settings trained using our Adaptive Learning Method. †WebMMU reports the Agentic Action (Act.) performance. Best is bold, second-best is underlined. Full sub-task breakdown in Appendix Table 12.

- (3) Learning to Modulate Tool-Use Frequency Beyond the binary choice of adopting or discarding a tool, the model exhibits a more nuanced adaptive behavior: dynamically modulating the invocation frequency of continuously useful tools to find an optimal balance. For instance, as shown in Figures 3b and 3c, the model learns that the Point tool is significantly more critical for navigation, maintaining a high and stable call frequency (∼ 3.2 calls/sample), while keeping its usage minimal for verification (∼ 1.0 call/sample).

- 3.3 ADAREASONER CAN LEARN GENERALIZED TOOL-USING

While adaptability to new tools within a known context is valuable, a more fundamental challenge lies in cross-tool and cross-task generalization. In practice, simply applying our TC + TG training recipe is not sufficient to achieve robust generalization: models may still fail when confronted with previously unseen tools or novel task distributions. To overcome this limitation, we propose Adaptive Learning method that randomizes tool definitions at both the token level and the semantic level. This strategy can be integrated into both the TC and TG stages. We evaluate its effectiveness from two complementary perspectives:

Generalize to New Tasks To validate whether the tool-planning capability can generalize to new tasks, we conduct TC only using the Jigsaw data, while withholding all training data related to the VSP and WebQA tasks. We then apply various training recipes to investigate whether tool-planning capabilities acquired from Jigsaw trajectories could be effectively transferred to these unseen domains. All data from the three tasks is used during the TG stage.

Generalize to New Tools To validate whether the tool-planning capability can generalize to new tools, the toolset definition during evaluation is completely different from the one used during training, while preserving the same underlying tool functionalities. This design allows us to assess the model’s ability to generalize to unseen tools that provide equivalent functionality, thereby evaluating its robustness and generalizability beyond memorizing tool-specific interfaces.

The results, as shown in Table 4, reveal the potential for generalized tool-using, demonstrating that tool-planning skills learned from a single task can generalize to unseen domains. Specifically, the model trained with Randomized TC + Randomized TG achieves the most significant improvement over the base model. On the unseen VSP task, it boosts the overall score from 28.09 to 78.91, and also demonstrates strong generalization when averaged across all three tasks (75.81 vs. 46.50). In contrast, other training settings fail to improve performance on unseen tasks and, in some cases, even degrade it. These results indicate that our method enables the model to learn generalizable tool usage, rather than overfitting to task-specific patterns.

- 3.4 MAIN RESULTS

Following our proposed training recipe and detailed analysis, we select the data from VSP, Jigsaw, and WebQA as our cold start data and VSP, Jigsaw, WebQA, and Visual Search as our RL data. The training details can be found in Appendix B.3.4. To enhance the model’s generalizable tool-use capabilities, we applied the Adaptive learning method (i.e., Randomized TC + Randomized TG) strategy on Qwen 2.5VL 7B. We evaluate our method on a diverse set of benchmarks. In addition to the benchmarks introduced earlier, we further include visual search benchmarks to assess the

Model VSPO VSP Jigsaw BLINK-J GUIChat WebMMU HRBench V* Avg. Closed-Source Models

Gemini 2.5 flash 40.12 53.55 67.20 65.33 83.05 66.26 78.25 68.59 65.54 GPT 5 34.25 55.64 80.10 73.33 71.41 80.49 74.38 74.87 68.56 Claude 4 sonnet 51.56 56.27 58.60 65.33 93.14 83.54 60.62 59.69 66.09

###### Open-Source Models

Qwen 2.5 VL 3B 26.53 26.73 39.80 48.67 46.26 54.47 53.00 43.98 42.93 Qwen 2.5 VL 7B 25.39 28.09 45.70 52.67 68.09 67.48 63.62 63.35 51.80 Qwen 2.5 VL 32B 28.56 33.91 59.50 64.67 85.21 85.98 70.12 72.25 62.53 Qwen 2.5 VL 72B 33.41 39.09 70.10 71.33 88.01 91.06 73.00 80.10 68.76 InternVL3 78B 28.14 35.09 52.80 60.00 79.83 71.34 75.12 81.15 60.93

###### Tool-Planning Models

Qwen 2.5 VL 7B + Tools 28.98 30.45 45.00 56.00 56.76 69.51 63.75 54.97 50.93 Qwen 2.5 VL 72B + Tools 39.64 45.00 61.50 65.33 77.13 76.83 67.12 65.97 62.82 GPT 5 + Tools 52.75 71.36 84.50 76.00 76.51 88.82 78.50 70.16 74.83 DeepEyes∗ 10.18 12.18 44.80 50.67 65.90 72.76 67.00 67.54 48.88 PixelReasoner∗ 21.38 24.55 52.20 59.33 72.45 69.51 52.12 53.40 50.87 AdaReasoner 7B 71.08 78.36 88.60 88.00 73.91 72.15 69.12 70.68 76.49 ∆ vs. base +45.69 +50.27 +42.90 +35.33 +5.82 +4.67 +5.50 +7.33 +24.72

- Table 5: Main results on visual reasoning, WebQA, and general VQA tasks. †WebMMU reports the Agentic Action (Act.) score. Avg denotes the average score across all benchmarks. Bold and underline indicate the best and second-best results within tool-planning models. Detailed results are provided in Appendix B.4.3 and Appendix Table 13. ∗All evaluations are conducted under a unified evaluation framework.

Model

Jigsaw Vstar

# Turns CPS Succ(%) Acc. # Turns CPS Succ(%) Acc. Orig. Acc.

DeepEyes 1.51 1.09 22.71 44.80 1.01 0.02 25.00 67.54 90.10 PixelReasoner 1.13 0.77 2.09 52.20 1.19 0.26 32.00 53.40 84.30 GPT5 + Tools 1.05 0.00 0.00 84.50 1.24 0.24 89.13 70.16 – Qwen 2.5 VL 7B + Tools 3.40 2.06 64.41 45.00 1.91 0.18 94.29 54.97 – Qwen 2.5 VL 72B + Tools 4.24 0.57 90.40 61.50 1.72 0.27 100.00 65.97 – AdaReasoner 7B 4.49 3.54 98.50 88.60 2.35 1.47 90.04 70.68 –

- Table 6: Tool usage statistics on Jigsaw and Vstar. #Turns denotes the number of interaction rounds. CPS (calls per sample) represents the average number of tool invocations per sample. Succ denotes the tool execution success rate. Acc is taken from the corresponding benchmark results in Table 5. Orig. Acc. denotes the performance under the model’s original tool definition.

generalizability of our approach beyond structured visual reasoning tasks, specifically V*(Wu & Xie, 2024) and HRBench(Wang et al., 2025b). The comparative results are summarized in Table 5.

Overall Performance Across Tasks As shown in the table, our model achieves consistent and substantial performance improvements across both visual reasoning tasks (e.g., VSP and Jigsaw) and general multimodal tasks (e.g., WebQA and Visual Search). In visual reasoning, our method yields large and stable gains (∆ > +40%), enabling the 7B model to outperform several competitive closed-source models and effectively narrowing the capability gap between smaller open-source models and larger counterparts. For general tasks, while tool integration cannot fully offset the performance advantages brought by model scaling due to their open-ended nature and the lack of deterministic tools, it still provides meaningful and robust performance boosts, demonstrating the broad effectiveness of visual tool augmentation.

Generalizable Tool Using. A defining characteristic of our approach is its robustness against domain shifts in both tool definitions and task contexts. During evaluation, the tool definitions differ from those used in training, forming a strict zero-shot tool adaptation scenario for all models. As shown in Table 6, while prior tool-planning methods exhibit limited tool engagement or low execution reliability under this shift, AdaReasoner consistently demonstrates the highest level of effective tool usage. In particular, it achieves the largest number of calls per sample (CPS) while maintaining near-perfect execution success (e.g., 3.54 CPS with 98.50% success on Jigsaw), indicating that the model has learned transferable tool-planning principles rather than overfitting to specific tool APIs.

[Figure 54]

Q: Please guide a player from the start to the goal.

[Figure 55]

Q: How can the user access the buying option from the webpage?

GPT 5: \boxed{D,D,D,D,D,L ,L,L,L,U,U,U,U}

GPT5: \boxed{Click the green “Buy Now >>” button. You can use either of the two on the page:- Top button: [560, 214, 654, 248]

Ours:

Ours:

[Figure 56]

<think>The first step is to locate the start point I will use the `Point` tool.</think> <tool_call> Point </tool_call> <think>After located the start point, I need to locate the target... </think> <tool_call> Point </tool_call> <think>After located the start point and target, I need to locate the ice holes ... </think> <tool_call> Point </tool_call> <think>I will call Astar to find the correct path... </think> <tool_call> Astar </tool_call>

<think>I will use the `DetectBlackArea` tool to find the bounding box of this missing region </think> <tool_call> DetectBlackArea </tool_call>

Ours:

<think> I can see a distinct section on the page that features a book cover and a prominent green button... To accurately describe how a user would access this, I should focus on this specific area of the page.. </think> <tool_call> Crop </tool_call>

[Figure 57]

[Figure 58]

<think>I will now attempt to insert the first candidate image... </think> <tool_call> InsertImage </tool_call>

[Figure 59]

<think>To understand how a user would access this buying option, I need to analyze the text ... Now, I will use the OCR tool...</think>

- `img_2` (Option A) provides a more plausible and natural completion of the original image.</think> <response> Based on my visual inspection, the best fit is option A. \boxed{A} </response>

BBOX: [210, 136, 426, 284]

Q: which one is the missing part?

[Figure 60]

[Figure 61]

[Figure 62]

A. B.

[Figure 63]

[Figure 64]

<think>does not align seamlessly

... I will now attempt to insert

- `img_3` (Option B) into ..</think> <tool_call> InsertImage </tool_call>

Path: LDLD...

{'label': 'Buy', 'label':'Now'}

[Figure 65]

<think>I have now inserted both images into the missing region. Upon visual inspection of the resulting images, neither insertion perfectly resolves the jigsaw puzzle,

<tool_call> OCR </tool_call>

<think>I will call Draw2DPath to verify the path </think> <tool_call> Draw2DPath </tool_call>

<think>I have successfully used the OCR tool on the cropped imae. The button is situated directly beneath the book's title and description, making it a direct and intuitive way for a user to access the purchasing interface.</think> <response>

as the tool's output ... However, I must still determine the most suitable option based on visual evidence ...

<think>Based on observations through tools, I can confirm that the path is correct </think> <response> \boxed{L,D,L,D,D,L,L,L,L,U,U,U} </response>

To access this buying option, a user would likely need to be logged in ... Once logged in, the user would click on the "Buy Now" button ... </response>

(a) VSP-Navigation (b) Jigsaw (b) GUIQA

- Figure 4: Our AdaReasoner-7B demonstrates advanced capabilities for multi-turn, tool-assisted reasoning and reflection, enabling it to achieve performance that is on par with, or even superior to, state-of-the-art closedsource models.

Crucially, this generalizable tool usage translates into clear performance gains. On Jigsaw, AdaReasoner attains the best accuracy (88.60), substantially outperforming prior methods, which exhibit limited effectiveness when confronted with previously unseen tasks and tool definitions. On VStar, which evaluates real-world general VQA scenarios and does not provide explicit tool-calling supervision during training, our model still actively invokes tools (1.47 CPS) and achieves the highest accuracy (70.68). In contrast, several existing approaches rely heavily on their original tool definitions and suffer notable performance degradation when evaluated under the new setting, as reflected by the gap between their original and adapted accuracies. Together, these results demonstrate that our method effectively decouples tool-use logic from task-specific supervision, empowering the model to autonomously formulate tool-augmented strategies for new problems.

Moreover, beyond the quantitative improvements reported on benchmark metrics, as shown in Figure 4, AdaReasoner-7B exhibits robust qualitative behaviors, including multi-turn tool planning, reflective correction of imperfect tool outputs, and synergistic tool composition, enabling it to outperform strong baselines such as GPT-5.

- 4 RELATED WORK

- 4.1 REINFORCEMENT LEARNING FOR MULTIMODAL REASONING

The recent success of DeepSeek-R1 (Guo et al., 2025), which demonstrated that rule-based Group Relative Policy Optimization (GRPO) can effectively induce strong reasoning behaviors in LLMs, has spurred a wave of research aimed at replicating this paradigm in the multimodal domain. Several studies have successfully extended this approach, with Zhou et al. (2025) reproducing the emergent “aha” moment in MLLM reasoning, R1-OneVision (Yang et al., 2025) introducing a cross-modal formalization pipeline, and works like Feng et al. (2025) and Li et al. (2025) improving temporal reasoning in videos. A collection of other strong works have also leveraged R1-style methods to achieve impressive results in general MLLM reasoning (Huang et al., 2025; Shen et al., 2025; Lu et al., 2025). However, a key limitation of the R1-style, rule-based reward structure is that it primarily targets the reasoning process and does not directly improve the model’s underlying perceptual abilities. Since accurate perception is the foundation for sound reasoning, error accumulation from faulty perception can still lead to hallucinations and degrade performance. AdaReasoner directly addresses this shortcoming. By leveraging the precise perceptual capabilities of external expert models and specialized tools, our framework ensures a high-fidelity understanding of the visual input, thereby improving the reliability of the entire reasoning pipeline.

- 4.2 TOOL-AUGMENTED MULTIMODAL REASONING

There is a growing interest in enhancing MLLMs with tool-use capabilities. Early efforts focused on foundational aspects such as infrastructure and data. LLaVA-Plus (Liu et al., 2024a), for example, introduced a dedicated tool server to provide services for MLLMs. On the data front, CogCoM (Qi et al., 2024) identified six key manipulation strategies and trained models on synthetic Chainof-Manipulation (CoM) data, while TACO (Liu et al., 2024b) contributed a large-scale dataset of reasoning traces derived from 15 visual tools. Subsequent research has explored different paradigms for tool interaction. One prominent line of work enhances visual reasoning by training models to generate code (Zhang et al., 2025; Zhao et al.). While powerful, these code-based environments are ill-suited for integrating computationally intensive capabilities, such as invoking large expert models. Another line of research leverages simpler, atomic visual tools like zoom-in functions to augment model perception (Wang et al., 2025a; Zheng et al., 2025; Su et al., 2025a; Zhu et al., 2025b; Su et al., 2025c). However, these approaches typically focus on single-step actions and have not explored the more complex challenges of multi-turn planning or dynamic tool composition. Our work, AdaReasoner, is designed to bridge these gaps, providing a framework that enables models to perform multi-turn planning and reasoning while adaptively selecting from a diverse suite of tools.

- 5 CONCLUSION

We introduce a comprehensive framework that integrates high-quality trajectory curation, ToolGRPO, and an adaptive learning mechanism to enable effective multi-turn tool planning. Based on this framework, we develop AdaReasoner, a family of tool-planning models. Unlike approaches that rely on rigid, task-specific patterns, AdaReasoner learns tool usage as a generalizable reasoning capability, allowing it to coordinate complex tool sequences and adapt zero-shot to unseen tool definitions. Extensive experiments demonstrate that this paradigm achieves state-of-the-art performance, with models exhibiting autonomous adaptive behaviors, selectively adopting beneficial tools, suppressing irrelevant ones, and dynamically modulating tool usage frequency according to task demands. More fundamentally, our findings show that effective tool orchestration shifts the primary performance bottleneck from intrinsic model scale to tool utility, enabling a 7B model to surpass strong proprietary systems such as GPT-5 on challenging visual reasoning tasks.

REFERENCES

Anthropic. Claude 4 Sonnet Model Card. Technical report, Anthropic, May 2025. URL https:// www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47. pdf. Accessed: 2025-09-17.

Rabiul Awal, Mahsa Massoud, Aarash Feizi, Zichao Li, Suyuchen Wang, Christopher Pal, Aishwarya Agrawal, David Vazquez, Siva Reddy, Juan A Rodriguez, et al. Webmmu: A benchmark for multimodal multilingual website understanding and code generation. arXiv preprint arXiv:2508.16763, 2025.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317, 2024.

Andy Clark and David Chalmers. The extended mind. Analysis, 58(1):7–19, 1998. ISSN 00032638,

14678284. URL http://www.jstor.org/stable/3328150.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. In European conference on computer vision, pp. 126–142. Springer, 2024a.

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, and Li Yi. Taco: Benchmarking generalizable bimanual tool-action-object understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21740–21751, 2024b.

Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025.

Zixian Ma, Jianguo Zhang, Zhiwei Liu, Jieyu Zhang, Juntao Tan, Manli Shu, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Caiming Xiong, et al. Taco: Learning multi-modal action models with synthetic chains-of-thought-and-action. arXiv preprint arXiv:2412.05479, 2024.

Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European conference on computer vision, pp. 69–84. Springer, 2016.

OpenAI. GPT-5 System Card. Technical report, OpenAI, August 2025. URL https://cdn. openai.com/gpt-5-system-card.pdf. Accessed: 2025-09-17.

Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. Cogcom: A visual language model with chain-of-manipulations reasoning. arXiv preprint arXiv:2402.04236, 2024.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model, 2025. URL https://arxiv. org/abs/2504.07615, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning, 2025a. URL https://arxiv.org/abs/2505.15966.

Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025b.

Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025c.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024.

Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goul˜ao, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, et al. Vgr: Visual grounded reasoning. arXiv preprint arXiv:2506.11991, 2025a.

Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7907–7915, 2025b.

Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094, 2024.

Qiucheng Wu, Handong Zhao, Michael Saxon, Trung Bui, William Yang Wang, Yang Zhang, and Shiyu Chang. Vsp: Assessing the dual challenges of perception and reasoning in spatial planning tasks for vlms. arXiv preprint arXiv:2407.01863, 2024.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025.

Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, and Chen Wei. Pyvision: Agentic vision with dynamic tooling, 2025. URL https://arxiv. org/abs/2507.07998.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing ”thinking with images” via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.14362.

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. R1zero’s” aha moment” in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025a.

Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, et al. Active-o3: Empowering multimodal large language models with active perception via grpo. arXiv preprint arXiv:2505.21457, 2025b.

A METHOD DETAILS

- A.1 BASIC SETTINGS

We first formalize multimodal reasoning with tools as an agentic planning process, enabling a systematic description of how models decompose and solve complex tasks.

Problem Formulation We denote an MLLM with tool-using capability as a policy πθ, parameterized by model weights θ. At initialization, πθ is equipped with access to a pool of tools T = t1,t2,··· ,tn, where n denotes the number of available tools. Given a task description g and the original multimodal input x = text,image, the system begins from an initial state s0. Building on this setup, the planning framework is formalized through three essential components. State st represents the current problem status: the initial state s0 corresponds to the original input, while intermediate states capture textual reasoning steps conditioned on accumulated observations, until a special token triggers an action. Action at denotes a one-step tool invocation, delimited by the special symbols <tool call> and </tool call>, which executes a selected tool. Observation ot is the execution result returned by the invoked tool and is incorporated into the subsequent state.

A typical tool-integrated reasoning trajectory τ involves multiple tool invocations over several reasoning steps, which can be represented as a sequence of rounds:

τ = {τ0,τ1,τ2,...,τT}

where each round is defined as τi = {si,ai,oi}, and the sequence proceeds as follows:

s0 −→a0 s1 −→a1 s2 −→a2 ... −−→aT sT+1

To enable the model to autonomously generate reasoning traces and tool calls, we utilize a system prompt as shown in B.2 during rollout. The tool list placeholder denotes the tool set T , which contains all tools available for invocation.

Tool Definition and Usage This section provides a detailed description of the visual tools integrated within our AdaReasoner framework. For each tool, we outline its core functionality, input arguments, output format, and its specific role in addressing the challenges of our evaluation tasks.

### • POINT

- – Functionality: A perceptual tool designed for precise object localization. Given an image and a natural language description of a target (e.g., “the start point”, “red cars”), it returns a list of pixel coordinates (x, y) of the objects’ center.
- – Role in VSP: This tool is fundamental for grounding the model in the spatial environment. In both Navigation and Verification, it is the first step to accurately identify the locations of the start, goal, and hazardous ice holes, converting the visual grid into a structured representation that can be used for planning.

### • DRAW2DPATH

- – Functionality: A visualization and verification tool. It takes a starting coordinate and a sequence of directional commands (e.g., [‘U’, ‘U’, ‘R’]) and overlays the corresponding path onto the input image.
- – Role in VSP: This tool externalizes the model’s internal plan into a visual artifact. In Verification, it renders the given path for the model to judge. In Navigation, it serves as a final check, allowing the model to visually confirm that its generated path is correct and safe before outputting the final answer.

### • ASTAR

- – Functionality: A classic planning algorithm encapsulated as a tool. It computes the shortest obstacle-free path between a start and a goal coordinate, given the locations of obstacles.
- – Role in VSP: This tool offloads the complex pathfinding computation. After the POINT tool identifies all key locations, ASTAR can be invoked to generate an optimal, logically sound path, freeing the MLLM to focus on higher-level task management and verification.

### • DETECTBLACKAREA

- – Functionality: A specialized perception tool for the Jigsaw task. It analyzes an image and returns the bounding box coordinates of any completely black, rectangular regions, which correspond to the missing puzzle piece.
- – Role in Jigsaw: This tool provides a deterministic way to identify the “problem space”. It is the critical first step in the solution trajectory, telling the model precisely where the candidate patches need to be inserted.

### • INSERTIMAGE

- – Functionality: A visual manipulation tool. It takes a base image, a patch image, and a set of coordinates, and returns a new image where the patch has been inserted at the specified location.
- – Role in Jigsaw: This tool enables iterative hypothesis testing. The model uses it to physically place each candidate patch into the missing slot identified by DETECTBLACKAREA. The resulting composite image is then fed back to the model, allowing it to visually assess the quality of the fit.

### • CROP

- – Functionality: An attentional tool. It takes an image and bounding box coordinates and returns a new, smaller image containing only the specified region.
- – Role in GuiQA: This tool mimics the human ability to focus on a specific part of a dense interface. By cropping a region of interest (e.g., a button or a text block), the model can reduce noise and ambiguity, creating a cleaner input for subsequent analysis by other tools or by the model itself.

### • OCR

- – Functionality: An information extraction tool. It performs Optical Character Recognition on an input image, returning all detected text along with its corresponding bounding box coordinates.
- – Role in GUIQA: This tool grounds the model’s understanding in the literal text present in the GUI. It is often used synergistically with CROP. After isolating a key UI element, the model can use OCR to reliably read its label (e.g., “Buy Now”), preventing the visual-only hallucinations common in MLLMs.

- A.2 HIGH-QUALITY COLD START TRAJECTORY DATA CURATION

For our structured reasoning tasks, we developed customized data generation and trajectory creation pipelines to ensure high quality and diversity. Some detailed data samples are shown in appendix

- A.3

VSP The VSP benchmark environments were procedurally generated using the Gymnasium (Towers et al., 2024) framework. To ensure variety, we systematically controlled the distribution of start points, end points, and ice holes. We synthesized environments of sizes 4 × 4, 6 × 6, and 8 × 8 for the training set, while reserving larger 5 × 5, 7 × 7, and 9 × 9 grids for testing. The Tool Cold Start (TC, SFT) trajectories were designed to mimic an optimal problem-solving process. For Navigation tasks, the trajectory consists of: (1) invoking the POINT tool to localize the start, end, and all ice holes; (2) performing textual reasoning based on these coordinates; and (3) calling DRAW2DPATH for final verification. Crucially, we also incorporated reflection and backtracking data derived from failure cases. For Verification tasks, the trajectory involves: (1) using POINT to locate the start, (2) employing DRAW2DPATH to render the proposed path, and (3) prompting the model to judge if this path intersects any ice holes.

Jigsaw The Jigsaw dataset was constructed using images from the COCO-2017 (Lin et al., 2014) training set. Each instance was created by first dividing an image into a 3 × 3 grid. A 2 × 2 sub-grid was then selected as the base image, from which one patch (e.g., top-right) was removed to create the problem. The removed patch served as the correct answer, while one of the remaining five patches from the original 3 × 3 grid was chosen as a distractor. The TC trajectory instructs the model to: (1)

|[Figure 66]|
|---|

Step4: Verify Solution

User: You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes. You can use the following tools to help you finish this task: {Point}, {DrawPath}

Assistant: I have successfully identified the Elf, the Gift , and all the Ice Holes. My goal is to find a path from the Elf to the Gift, avoiding all ice holes. Given the grid structure and the relative positions, I'll test the path 'R,D,R,D,R,D,R,D' as a first attempt to see if it reaches the goal safely. I need to visualize this path to confirm its validity.</think> <tool_call>{"name": "Draw2DPath","parameters": {"image": "img_1","start_point": [352.0,160.0],"directions": "R,D,R,D,R,D,R,D","pixel_coordinate": true}}</tool_call>

- Step1: Determine the start

- Step2: Determine the end

- Step3: Determine the Holes

|[Figure 67]|
|---|

Assistant: <think> I need to identify the starting point of the maze. I will use the `Point` tool to locate the coordinates of the Elf on the provided image.</think> <tool_call>{"name": "Point", "parameters": {"image": "img_1", "description": "Elf"}}</tool_call>

User: {"tool_response_from": "Draw2DPath", "status": "success", "messages": "Path drawn successfully." "error_code": 0} <image>

|[Figure 68]|
|---|

User: {"tool_response_from": "Point", "status": "success", "points": [{"x": 352.0, "y": 160.0}], "image_dimensions_pixels": {"width": 512,"height": 512}, "error_code": 0}

- Step5: Correct Solution and Verify

- Step6: Analysis and Response

Assistant: The Draw2DPath tool has returned an image showing the path 'R,D,R,D,R,D,R,D'. Upon reviewing the drawn path, I can clearly see that it does not reach the

destination (Gift) and potentially crosses over an ice hole or gets stuck. I need to re-evaluate the maze……I will try the sequence 'L,U,L,L,L,D,D,D,D,R,R,D,R,R,U,R'. This path attempts to navigate around the clusters of ice holes and find a clear passage.</think> <tool_call>{"name": "Draw2DPath","parameters": {"image": "img_1","start_point": [352.0,160.0],"directions": "L,U,L,L,L,D,D,D,D,R,R,D,R,R,U,R",……}}</tool_call>

Assistant: <think>Now that I have successfully located the Elf, my next step is to find the destination. I will use the Point again to identify the coordinates of the Gift.</think> <tool_call>{"name": "Point", "parameters": {"image": "img_1", "description": "Gift"}}</tool_call>

|[Figure 69]|
|---|

|[Figure 70]|
|---|

User: {"tool_response_from": "Draw2DPath", "status": "success", "messages": "Path drawn successfully." "error_code": 0} <image>

User: {"tool_response_from": "Point", "status": "success", "points": [{"x": 416.0, "y": 352.0}], "image_dimensions_pixels": {"width":512, "height":512}, "error_code": 0}

Assistant: <think>I have successfully identified the starting point (Elf) and the destination (Gift). The crucial next step is to locate all the "Ice Holes" on the map, which represent obstacles that must be avoided.</think> <tool_call>{"name": "Point", "parameters": {"image": "img_1", "description": "Ice Holes"}}</tool_call>

Assistant: The Draw2DPath tool has successfully drawn the new path. Upon visual inspection of the image, the drawn path clearly starts at the Elf's position, safely navigates through the open ice, avoids all the identified ice holes, and successfully reaches the Gift. This path is correct and achieves the objective of guiding the player from start to finish while avoiding obstacles. I now have the final answer. </think> <response>\\boxed{L,U,L,L,L,D,D,D,D,R,R,D,R,R,U,R} </response>

|[Figure 71]|
|---|

User: {"tool_response_from": "Point", "status": "success", "points": [{"x": 32.0, "y": 32.0}], ………… "points": [{"x": 160.0, "y": 480.0}], "image_dimensions_pixels": {"width":512, "height":512}, "error_code": 0}

Figure 5: An example of a multi-turn cold-start trajectory for the VSP task.

call DETECTBLACKAREA to identify the coordinates of the missing section, and (2) iteratively call INSERTIMAGE for each candidate patch until the puzzle is solved. To enhance robustness and diversity, we introduced several key variations: ➊ The order of patch insertion attempts was randomized to ensure a uniform distribution of options. ➋ Scenarios involving tool failures (e.g., detection errors) were included, prompting the model to fall back on its intrinsic knowledge after several failed attempts. ➌ A proportion of samples were designed to be solvable directly by the model without tool use, encouraging adaptive tool invocation.

GUIQA The process begins with 44k single-turn instances from the Guichat dataset. (Chen et al.,

- 2024) To identify challenging cases that necessitate tool use, we first prompted a powerful visionlanguage model, Qwen-VL-2.5-72B (Bai et al., 2025), to answer the questions. We retained only the instances where the model failed, resulting in a subset of 7,100 “hard” questions. Next, for these 7,100 instances, we rendered the ground-truth answer coordinates as bounding boxes on the

|[Figure 72]|
|---|

|[Figure 73]|
|---|

User: Given the first image with one part missing, can you tell which one of the second, third or the fourth images is the missing part (A or B or C) ? You can use the following tools to help you finish this task: {AstartWithPixelCoordinate}, {Draw2DPath}, {TurnCoordinateIntoTextMap}, {DetectBlackArea}, {InsertImage}

User: {"status": "success", "message": "Image inserted successfully", "error_code": 0, "tool_response_from": "InsertImage"} <image>

###### Step3: Analyze and Attempt Choice C

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

Assistant: <think>I have inserted `img_2` (Option A) into the missing region. Upon reviewing the resulting image, it's clear that `img_2` does not fit seamlessly; the water pattern and the boat's structure do not align with the surrounding parts of the original image. Therefore, I will now proceed to test the next option. And through analysis I feel that option C is more like the missing part, I will try C first.</think> <tool_call>{"name": "InsertImage","parameters": {"base_image": "img_1","image_to_insert": "img_4","coordinates": "[0, 142, 213, 284]"}}</tool_call>

###### A. B. C.

Step1: Identify Missing Region

|[Figure 77]|
|---|

Assistant: <think>The problem requires me to find the correct candidate image to fill the missing black area in the main picture. My first step is to accurately identify the location of this missing region. I will use the DetectBlackArea tool to find the bounding box coordinates of the black area in img_1. </think> <tool_call>{"name": "DetectBlackArea", "parameters": {"image": "img_1"}}</tool_call>

User: {"status": "success", "message": "Image inserted successfully", "error_code": 0, "tool_response_from": "InsertImage"} <image>

|[Figure 78]|
|---|

User: {"status": "success", "bounding_boxes": ["[0, 142, 213, 284]"], "message": "Found 1 black areas", "error_code": 0, "tool_response_from": "DetectBlackArea"}

###### Step4: Analyze and Response

Assistant: <think>I have now inserted `img_4` (Option C) into the missing region. Observing the result, `img_4` clearly provides a much better fit than `img_2`. The lines of the boat, the water, and the overall context align perfectly, indicating a seamless integration with the original image. This confirms that `img_4` is the correct missing piece.</think>\n\n <response>After trying both options, I can see that option A doesn't fit well - the edges don't align properly and the content doesn't flow naturally with the rest of the image.\n\nHowever, option C fits perfectly into the missing region. The edges align seamlessly, and the content flows naturally when this piece is inserted.\n\n\\boxed{C}</response>

Step2: Attempt Choice A

Assistant: <think>The DetectBlackArea tool has successfully identified the coordinates of the missing region as [0, 142, 213, 284]. According to the problem trajectory, I need to insert each candidate image into this region until the correct answer is found. I will start by inserting the second image (`img_2`), corresponding to option A, into the detected black area to evaluate its fit.</think> <tool_call>{"name": "InsertImage","parameters": {"base_image": "img_1","image_to_insert": "img_2","coordinates": "[0, 142, 213, 284]"}}</tool_call>

- Figure 6: An example of a multi-turn cold-start trajectory for the Jigsaw task. This trajectory showcases an iterative trial-and-error process. The agent first uses DETECTBLACKAREA to identify the missing region. It then sequentially attempts to InsertImage with each candidate patch, analyzing the visual result of each attempt before arriving at the correct solution.

|[Figure 79]|
|---|

User: You are a visual assistant capable of solving visual reasoning problems. Question: Can I download the thesis on 'Music in the Science Classroom' from this webpage? You can use the following tools to help you finish this task: {Point}, {DrawPath}

Step1: Analyze and Crop

Assistant: <think>The user is asking whether a specific thesis, \"Music in the Science Classroom,\" can be downloaded from the webpage. To answer this, I need to find information on the page that indicates download availability, such as a link or a download button. I can see the thesis title displayed prominently. Below the \"McFadden, Tom\" author name, to the right of the thumbnail image, there is a section providing citation details. Within this section, I observe the phrase \"Permanent link to OUR Archive version:\" followed by a URL. This phrase strongly suggests that the thesis is available for retrieval and download. To confirm this, I need to focus on this specific area. I have identified the coordinates of the \"Permanent link\" text as [321,446,522,462].</think> <tool_call>{"name": "Crop", "parameters": {"image": "img_1", "coordinates": " [321,446,522,462]"}}</tool_call> User: {"tool_response_from": "Crop", "status": "success", "message": "Image cropped successfully using absolute coordinates.", "image_dimensions_pixels": {"width": 201,"height": 32}, "error_code": 0} <image>

|[Figure 80]|
|---|

Step2: OCR

Assistant: <think>I have successfully cropped the specific area containing the "Permanent link to OUR Archive version:", creating `img_2`. Now, I need to use OCR on this cropped image to extract the actual text of the link and determine if it confirms the ability to download the thesis.</think> <tool_call>{"name": "OCR", "parameters": {"image": "img_2"}}</tool_call>

User: {"tool_response_from": "OCR", "status": "success", "detections": [{"label": "http://hdl.handle.net/10523/4203","confidence": 0.96,"pixel_bbox": {"x_min": 0,"y_min": 10,"x_max": 200,"y_max": 23}}], "image_dimensions_pixels": {"width": 201,"height": 32}, "error_code": 0} <image>

Step3: Analyze and Response

Assistant: <think>The OCR tool successfully extracted the text "http://hdl.handle.net/10523/4203" from the cropped image. This confirms that there is a permanent link to the archive version of the thesis available on the webpage. The presence of such a link indicates that the thesis can be downloaded.</think> <response>Yes, you can download the thesis on 'Music in the Science Classroom' from this webpage. A permanent link (http://hdl.handle.net/10523/4203) to the thesis is provided.</response

- Figure 7: An example of a multi-turn cold-start trajectory for a GUI-QA task. This sample illustrates a focusthen-extract strategy. The agent first uses the CROP tool to isolate a specific, relevant section of the webpage. It then applies the OCR tool to this cropped, unambiguous input to perform precise information extraction.

images. We then performed a manual visual inspection to ensure these boxes contained meaningful and relevant information, which filtered the set down to 1,800 valid data points. To generate highfidelity tool-use trajectories for these cases, we provided the ground-truth answer and coordinates to gemini-2.5-flash Comanici et al. (2025), prompting it to produce the chain-of-thought reasoning and tool invocation sequence required to solve the problem. Finally, all generated trajectories were validated against our predefined format, and only those that strictly conformed were retained. This final curation step yielded a high-quality dataset of 1,139 instances for our TC stage.

After defining the abstract trajectory structure for all tasks, we followed a unified, two-stage process to create the final training data. First, we executed these trajectories programmatically to populate them with real tool inputs and outputs. Subsequently, we leveraged Gemini 2.5 Flash (Comanici

et al., 2025) to generate the corresponding chain-of-thought (CoT) reasoning for each step. This process resulted in a final dataset of high-fidelity, tool-augmented trajectories complete with explicit reasoning chains, ready for our cold-start training.

- A.3 DATA SAMPLES

To provide a more concrete understanding of our cold-start data, we present representative multi-turn trajectory samples for each of our core tasks in Figures 5, 6, and 7. These examples are designed to showcase the sophisticated, human-like reasoning patterns we aim to instill in the model during the supervised fine-tuning phase.

The VSP sample (Figure 5) illustrates a methodical, multi-stage problem-solving process that includes perception, verification, and analysis. The Jigsaw sample (Figure 6) demonstrates an iterative trial-and-error strategy, where the agent actively evaluates the outcome of each tool call. Finally, the GUIQA sample (Figure 7) highlights a synergistic tool-use pattern, where one tool (‘Crop’) is used to create optimal conditions for another (‘OCR’). Across all examples, the interplay between the model’s internal thoughts (‘<think>’), tool calls, and observations from the environment is clearly demonstrated.

- A.4 TOOL GRPO

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that evaluates policy performance by directly comparing a group of candidate reasoning trajectories. The process of Tool GRPO in AdaReasonerbegins with the initial state s0 = ⟨g,text,image⟩, for which the policy πθ samples a set of N complete trajectories as candidate responses, {τ1,τ2,...,τN}. Each trajectory is evaluated by a reward function, yielding rewards ri = R(τi). GRPO then calculates a group-relative advantage Ai for each trajectory by normalizing its reward against the statistics of the entire group:

ri − mean{r1, r2, . . . , rN} std{r1, r2, . . . , rN}

Ai =

. (3)

The policy is then updated to favor trajectories with higher relative advantages by maximizing a clipped surrogate objective function. This objective is designed to ensure stable training by preventing excessively large policy updates. The full objective is:

JGRPO(θ) = Eq∼P(Q),{τi}Gi=1∼πθold(·|q)

|τi|

(4)

G

1 G|τi|

min mijAi, clip sij, 1 − ε, 1 + ε Ai − β DKL πθ πref .

i=1

j=1

- i
- j−si|si)

Here, mij = πθ(τ

πθold(τji−si|si) is the importance sampling ratio that measures the change between the new policy πθ and the old policy πθ

used to generate the samples. The Kullback-Leibler (KL) divergence penalty, DKL(πθ||πref) regularizes the policy update by penalizing large deviations from a reference policy πref.

old

Reward Design Our reward function is designed to evaluate both the structural syntax and the semantic correctness of the model’s output. The total reward, Rtotal, is a composite score defined as:

Rtotal = Rformat · (λtool · Rtool + λacc · Racc) (5)

Here, Rformat acts as a binary gate, ensuring that rewards for tool usage (Rtool) and final answer accuracy (Raccuracy) are granted only if the output adheres to the required structure. This design incentivizes the model to first master the correct syntax before optimizing for functional correctness. λacc is a hyperparameter and the effect of it is discussed in Appendix B.5.

- • Format Reward (Rformat) The format reward is a binary signal that assesses the structural integrity of the model’s output. It verifies that the generated response contains all required special tokens in the correct order and follows predefined rules.

1 if the output format is valid 0 otherwise

(6)

Rformat =

If Rformat is 0, the total reward for the trajectory is nullified, creating a strong imperative for the model to learn the required output structure.

- • Tool Reward (Rtool) The tool reward provides a fine-grained evaluation of the tool-calling process, with a score ranging from 0 to 4. We employ a hierarchical scoring system where each level must be passed to proceed to the next.

- I. Invocation Structure (Score 1): A score of 1 is awarded if the tool call is correctly encapsulated within the <tool call> and </tool call> tokens. If not, the score is 0, and no further tool evaluation occurs.

- II. Tool Name Validity (Score 2): If the structure is correct, we verify that the invoked tool name exists in the set of available tools, T . If the name is valid, the score becomes 2.
- III. Parameter Name Correctness (Score ∈ [2,3]): If the tool name is valid, we assess the parameter names. A partial score is awarded based on the proportion of correctly named parameters. A perfect match yields a score of 3. The score is calculated as:

Rtool = 2 + |paramscorrect name| |paramstotal|

(7)

- IV. Parameter Content Validity (Score ∈ [3,4]): Finally, if all parameter names are correct (base score of 3), we evaluate the parameter values for semantic and contextual validity. The final score is proportional to the number of correct values, reaching a maximum of 4.

Rtool = 3 + |paramscorrect content| |paramstotal|

(8)

- • Accuracy Reward (Raccuracy) The accuracy reward evaluates the final outcome of the reasoning process, providing a clear signal based on the correctness of the model’s final answer.

Raccuracy =

4 if the final answer is correct 0 otherwise

(9)

This multi-faceted reward structure guides the model toward not only achieving the correct final answer but also mastering the intermediate steps of correct formatting and precise tool invocation.

- B EXPERIMENT DETAILS

- B.1 TASK DEFINITION

We evaluate our approach across a diverse suite of three challenging tasks to validate whether our approach can help enhance the reasoning ability.

Visual Spatial Planning We adopt the FrozenLake scenario (Towers et al., 2024) to evaluate models’ spatial planning and verification abilities. The navigation task requires the model to generate a safe path from the start to the goal while avoiding holes, which demands accurate perception of the grid map and robust sequential reasoning to plan multi-step trajectories. The verification task instead focuses on state checking, determining whether a given location or a proposed path is safe, which isolates the perception and reasoning components of the planning pipeline. Together, these tasks expose two fundamental challenges for VLMs: (i) precise visual perception of spatial layouts under varying map sizes, and (ii) reliable reasoning over action sequences to ensure safety and goal completion.

Concretely, we evaluate models on two benchmarks. The first is VSPO, a dataset we construct to assess out-of-distribution generalization. During training, we use maps of sizes 4 × 4, 6 × 6, and

- 8 × 8, while reserving maps of sizes 3 × 3, 5 × 5, 7 × 7, and 9 × 9 for testing. This setup not only probes the model’s ability to generalize to unseen spatial configurations but also examines whether it truly learns to leverage tool usage for problem solving. The second is the original VSP benchmark (Wu et al., 2024), where we adopt the navigation and verification tasks to further test visual-spatial reasoning and state-checking capabilities under standardized settings.

Jigsaw The Jigsaw task (Noroozi & Favaro, 2016) evaluates a model’s ability to reconstruct holistic understanding from fragmented visual inputs. Specifically, the model must infer the correct spatial arrangement of shuffled image patches and reason about their part–whole relationships. This requires fine-grained perception to capture local details, as well as global reasoning to integrate them into a coherent whole. The key challenges lie in bridging local–global consistency and maintaining semantic alignment across patches, making it a strong test of visual compositionality and structural reasoning.

Concretely, we evaluate models on two benchmarks. The first is Jigsaw-COCO, where we construct training and test splits based on the COCO train 2017 dataset (Lin et al., 2014). We extract the topleft, top-right, and bottom-left patches of each image to form the training set, while reserving the bottom-right patches for testing. This design allows us to probe the model’s out-of-distribution generalization and examine whether it truly learns to invoke tool usage for solving the puzzle. The second is the Jigsaw benchmark from BLINK (BLINK-J) (Fu et al., 2024), which provides a standardized evaluation of fine-grained visual reasoning and compositional understanding under more challenging and diverse settings.

GUIQA The GUIQA task is designed to evaluate a model’s sophisticated capabilities in finegrained visual understanding and critical information extraction from GUIs. In this task, a model is provided with a GUI screenshot and an associated question, where the main difficulty lies in precisely grounding UI elements on a dense layout, comprehending their functional roles, and performing multi-step reasoning by integrating scattered information.

The evaluation is conducted on two distinct datasets. The first is the GUIChat (Chen et al., 2024), which specifically probes the model’s capacity for interactive, dialogue-based comprehension of webpage screenshots. Models are required to process complex queries related to visual information, human-centric needs, world knowledge, and reasoning. The second is the WebQA from the WebMMU (Awal et al., 2025). It offers a structured evaluation across three distinct categories. Agentic Action tests the ability to understand UI elements like buttons and menus in order to formulate the necessary user actions, complete with precise spatial grounding. General Visual Comprehension assesses how well the model can extract and synthesize information from varied page components, including text, images, and graphics. And Multi-step Reasoning demands complex inference, numerical calculations, and comparisons across different parts of the UI. To better evaluate the effectiveness of our visual tools in GUI-based scenarios, we focus on the agent acting subset of the English split of WebMMU and report results on this subset in the main table to highlight the impact of our approach. Some of the complete WebMMU results are provided in Appendix B.4.3.

General VQA General VQA aims to evaluate the general visual ability of models beyond structured visual reasoning tasks, focusing on open-ended visual understanding and reasoning in unconstrained real-world scenarios. Unlike structured benchmarks that provide well-defined goals and tool usage patterns, general VQA tasks often require flexible perception, commonsense reasoning, and robust interpretation of diverse visual content without explicit procedural guidance.

To assess the general visual capabilities of our approach in such settings, we consider two representative benchmarks: V*(Wu & Xie, 2024) and HRBench(Wang et al., 2025b). V* evaluates a model’s ability to perform fine-grained visual search and object attribute recognition in highresolution, complex visual scenes, focusing on precise localization and relationship reasoning that goes beyond simple image understanding. HRBench assesses the perception and reasoning capabilities of Multimodal Large Language Models on ultra-high-resolution images (up to 8K), testing fine-grained single-instance and cross-instance visual understanding, including detailed attribute extraction and spatial relationships in larger scenes.

- B.2 PROMPTS The system prompt used for guiding the tool-planning model is provided in Figure 8.
- B.3 IMPLEMENTATION DETAILS

We developed AdaReasoner Framework, an end-to-end framework that orchestrates the entire lifecycle of our tool-planning models, from data curation to evaluation. At the heart of this framework

### System Tool Prompt

You are a visual assistant capable of solving visual reasoning problems. You can rely on your own capabilities or use external tools to assist in solving. Available Tools In your response, you can use the following tools: {Tool List} Steps for Each Turn

- 1. Think: First, silently analyze the user’s request to understand the goal. This thinking process should be enclosed in <think> and </think> tags.
- 2. Decide Action: Based on your thinking, decide on one of the following two actions:

- - If you need to use a tool: Generate your tool call, enclosed between <tool call> and </tool call> tags. Do not generate a <response> in this turn.

- - If you have enough information to answer: Generate your final, user-facing answer, enclosed between <response> and </response> tags. Do not generate a <tool call> in this turn. Output Format: Your output must always begin with your thought process. After the <think> block, you must provide either a <tool call> or a <response>, but never both in the same turn.

- Case 1: Tool Use is Required <think> Your thoughts and reasoning </think> <tool call> {“name”: “Tool name”, “parameters”: {“Parameter name”: “Parameter content”, “...”: “...”}} </tool call>

- Case 2: Ready to Respond to the User <think> Your thoughts and reasoning </think> <response> Your final response </response> Important Notes

- 1. You must always include the <think> field to outline your reasoning. Provide one of <tool call> or <response>. You must not include both <tool call> and <response> in the same turn because they are mutually exclusive. If tool usage is required, you must instead include both <think> and <tool call>, and omit <response> for that turn. If no further tool usage is required and ready to answer the user’s question, you can then use <think> to summarize your reasoning and include <response> with your final answer, and this indicates the ends of the conversation.

- 2. You can only invoke a single tool call at a time in the <tool call> fields. The tool call should be a JSON object with a “name” field and a “parameters” field containing a dictionary of parameters. If no parameters are needed, leave the ”parameters” field an empty dictionary. All images have their coordinate origin at the top-left corner.

- 3. Some tools require image input. You do not need to generate or upload the actual image data simply refer to an image using a placeholder in the form of “img n”. There may be multiple images present in the dialogue. Besides the original image, additional images may appear as a result of prior tool calls (e.g., edited images returned by visual editing tools). You are free to select which image to use as input for the next tool. The index n in “img n” refers to the image’s position in the dialogue history:

- - The original image is always referred to as “img 1”.

- - Each subsequent image, including those returned from tools, is assigned “img 2”, “img 3”, and so on, in the order they appear in the dialogue. For example:{“parameters”: {“image”: “img 1”, “other params”: “other values”}}

- 4. All image coordinates used must be in absolute pixel values, not relative or normalized coordinates.
- 5. At the end, provide your final answer by placing it inside boxed{}, and wrap the entire final output inside <response></response> tags.

Figure 8: Our system employs tool prompts to guide models in learning how to use tools effectively.

Base Model Qwen2.5-VL-7B-Instruct Vision Tower Frozen True MM Projector Frozen True Finetuning Type Full DeepSpeed Stage ZeRO-3

Model

Max Samples 332,649 Cutoff Length 35,536 Preprocessing Workers 64

Dataset

Batch Size per Device 1 Gradient Accumulation Steps 2 Effective Batch Size 2 Learning Rate 1e-5 Epochs 3 LR Scheduler cosine Warmup Ratio 0.1 Mixed Precision bfloat16

Training

Logging Steps 10 Checkpoint Save Steps 100

Logging / IO

Train/Validation Split 90% / 10% Eval Batch Size per Device 1 Eval Steps 100

Evaluation

Table 7: Tool Cold Start (SFT) Training Configuration and Hyperparameters.

is the Tool Server, a unified, MCP-like service that manages all available tools, from simple offline utilities to compute-heavy online expert models.

- B.3.1 DATA CURATION

During the data curation stage, we employ our AdaDataCuration module, which leverages the Tool Server to automatically generate high-quality cold-start trajectories. Specifically, we first design abstract, optimal problem-solving blueprints for each task, consisting of a tool-call chain and chainof-thought (CoT) placeholders. We then prompt Gemini-2.5-Flash to fill these placeholders with detailed CoT reasoning. Finally, the Tool Server executes the corresponding tool calls and integrates the results into the dialogue, yielding a complete and coherent training instance.

- B.3.2 TOOL COLD START STAGE

During the cold-start stage, these trajectories are used for full-parameter supervised fine-tuning, for which we adopt the LLaMA Factory framework (Zheng et al., 2024). The key configurations and hyperparameters are summarized in Table 7.

- B.3.3 TOOL GRPO STAGE

Following SFT, the model is further refined in the Tool-GRPO stage using AdaTG, our custom reinforcement learning framework inspired by Sheng et al. (2024); Zheng et al. (2025), which also relies on the Tool Server for live tool interactions. Specifically, AdaTG enables online reinforcement learning by allowing the model to interact with tools in real time through the Tool Server, receiving execution feedback at each planning step. Combined with the Tool-GRPO optimization objective, this interactive training process explicitly aligns the policy with long-horizon tool planning behaviors, resulting in a robust and adaptive tool planning model. The key configurations and hyperparameters of Tool GRPO stage are summarized in Table 8.

- B.3.4 DETAILS OF OUR FINAL MODEL

During the training of our final model, AdaReasoner-7B, we adopt a two-stage training pipeline consisting of a cold-start stage followed by a Tool-GRPO stage. In the Tool Cold Start (TC) stage, we fine-tune the model using data from VSP, Jigsaw, and WebQA. During this phase, we enable Adaptive Learning to facilitate efficient adaptation across diverse task structures and reasoning

Max Prompt Length 8192 tokens Max Response Length 20480 tokens Train Batch Size 32 Shuffle True Filter Overlong Prompts True

Data

Strategy FSDP Gradient Checkpointing True PPO Mini-batch Size 8 PPO Micro-batch Size / GPU 1 Max Token Len / GPU (PPO) 16384 Grad Clip 1.0 Clip Ratio (PPO) 0.2 PPO Epochs 1 Entropy Coeff 0.0 Use KL Loss False Actor LR 1e-6 Weight Decay 0.01 FSDP Param Offload True FSDP Optimizer Offload True # Nodes / GPUs 1 node, 8 GPUs

Policy

Engine vLLM Temperature 1.0 Top-p 1.0 Top-k -1 # Samples per Prompt (n) 4 Dtype bfloat16 Tensor Model Parallel Size 2 Max # Batched Tokens 32768 GPU Memory Utilization 0.65 Enforce Eager False Chunked Prefill False

Rollout

Tool-Agent Max Turns per Episode 10

Strategy FSDP LR 1e-5 Weight Decay 0.01 PPO Epochs 1 Grad Clip 1.0

Critic

Advantage Estimator GRPO Gamma 1.0 Lambda 1.0 Use KL in Reward False KL Coef 0.0 Norm Adv by Std in GRPO True

Algorithm

Table 8: Key configurations and hyperparameters used in the Tool GRPO stage.

patterns. In the subsequent Tool GRPO (TG) stage, we further train the model using a broader set of tasks, including VSP, Jigsaw, WebQA, and additional visual search data. Adaptive Learning remains enabled throughout this stage to continuously adjust the model’s behavior as it interacts with different tools and task distributions. This two-stage training strategy allows AdaReasoner-7B to progressively acquire strong reasoning capabilities while maintaining robustness and generalization across diverse multimodal tasks.

In Table 6, we report the performance differences of several prior methods under their original evaluation settings and our unified new setting. We observe that the performance gaps mainly stem from changes in tool types and definitions, as well as differences in the input order of images and questions. In contrast, our model exhibits strong robustness on the V* benchmark: its performance remains stable even when the image–question order is swapped or when the tool definition is changed, demonstrating superior adaptability to variations in tool definitions and evaluation protocols.

- B.4 EVALUATION DETAILS

- B.4.1 BASELINES

We evaluate our method against a comprehensive set of strong baselines. First, we include proprietary models with state-of-the-art multimodal capabilities, including GPT-5-20250807 (OpenAI, 2025), Claude-sonnet-4-20250514 (Anthropic, 2025), and Gemini-2.5-flash (Comanici et al., 2025). Second, we consider open-source MLLMs, namely Qwen-2.5-VL-32B/72B-Instruct (Bai et al., 2025) and InternVL-3-78B (Zhu et al., 2025a), which are selected as primary testbeds due to their strong performance in visual understanding and reasoning, enabling us to assess the scalability of our approach across different model sizes.

- B.4.2 EVALUATION METHOD

We design AdaEval, a unified evaluation framework for tool planning that supports the evaluation of both tool-planning models and non–tool-planning models. To ensure fair and consistent comparison, all models in our experiments are evaluated exclusively within the AdaEval framework under the same evaluation protocol.

For benchmarks that require open-ended or subjective judgment, including VStar, WebMMU, and GUIQA, we adopt an LM-as-a-Judge evaluation strategy. Specifically, we use Qwen-2.5-VL-72B as the judge model. The prompt used for judgment is illustrated in Figure 9.

System Tool Prompt

You are an expert evaluator. Your goal is to determine if a [Model Answer] correctly and factually answers a [Question] when compared against a [Standard Answer]. Core Evaluation Principle:

The [Model Answer] is considered consistent if it contains the essential key information present in the [Standard Answer]. The [Model Answer] is allowed to be much more verbose, conversational, and include additional correct context or explanations. Your primary task is to verify the presence of the core facts, not to penalize extra information. If a question asks for specific formatting like coordinates or tables, but the model identifies the correct core element textually, it should still be considered consistent.

- - Consistent (Judgement: 1): The [Model Answer] successfully identifies the main point or action from the [Standard Answer]. For example, if the standard answer is to “click button A”, the model answer is consistent if it mentions clicking or interacting with “button A”, even if it’s surrounded by other text.
- - Inconsistent (Judgement: 0): The [Model Answer] fails to mention the key information, provides contradictory information, or hallucinates a different solution. Output Format: Just output ‘Judgement: 1’ or ‘Judgement: 0’. Do not output anything else.

Figure 9: System Prompt Used for the LM-as-a-Judge Evaluation

In addition, we conduct a human evaluation on the V* benchmark by manually assessing the extracted model predictions of Qwen 2.5 VL 7B and AdaReasoner-7B. The human judgment scores are consistent with the scores produced by our evaluation framework, providing strong evidence for the reliability and correctness of our automated evaluation.

- B.4.3 DETAILED RESULTS

Due to space constraints and for clarity of presentation, we report abridged results in the main paper and provide the full results here. Specifically, in Section 3.1, we evaluate the contributions of TC and TG under single-task fine-tuning. The summarized results are reported in Table 2, with the corresponding detailed results provided in Table 13. In Section 3.3, we present the generalization performance in Table 4, and report the full benchmark breakdown in Table 12. Finally, in Section 3.4, we report the main results in Table 5, with the complete version shown in Table 13.

- B.5 ABLATION STUDY

We systematically adjust λtool and λacc to evaluate their influence on learning dynamics and final performance. Specifically, we train the model on the same VSP task data for 100 RL steps under different reward-weight settings, monitor the training curves to ensure convergence, and then evaluate each checkpoint’s performance. The results are summarized in the table 9.

As shown in table 9, the model’s performance consistently improves as the ratio λtool : λacc increases. This indicates that larger tool rewards not only accelerate convergence during RL training but also lead to significantly better final performance. These results validate that our tool-reward design is effective and plays a crucial role in helping the model learn tool calling more efficiently and robustly.

λtool : λacc VSP (%) VSPO (%)

Nav Verify Overall Nav Verify Overall

- 0:1 51.83 95.00 71.45 41.78 75.58 57.37

- 1:2 49.50 95.80 70.55 36.44 94.29 63.11

- 1:1 64.00 96.40 78.73 48.56 96.23 70.54

- 2:1 90.33 96.80 93.27 70.33 96.36 82.34

Table 9: Ablation on reward-weight configurations for VSP and VSPO.

- C FURTHER ANALYSIS

- C.1 VISUAL TOOLS HELP OVERCOME SCALE-BASED LIMITATIONS

Avg. Upper Boundary: 96.2%

Claude Sonnet 4: 56.3%

Qwen 2.5 VL 3B Qwen 2.5 VL 7B AdaReasoner 3B AdaReasoner 7B

30

40

50

60

70

80

90

100

Accuracy(%)

26.7%

31.6%

94.7%

97.6%

Base Models

Our Models

Avg. Boundary

Claude-Sonnet-4

Figure 10: Overcoming scale-based limitations with tool augmentation. On the VSP task, our tools boost the performance of both 3B and 7B models, elevating them from disparate baselines to a near-uniform high performance.

The results shown in section 3.1 reveal that tool augmentation can redefine the performance ceiling of MLLMs by overcoming scale-based limitations. As illustrated in Figure 10, while the baseline performance of 3B and 7B models is disparate and low, our tool-augmented versions both achieve near-perfect accuracy (94.7% and 97.6%). This indicates that the primary performance bottleneck has shifted from the model’s intrinsic scale to the extrinsic quality of the tools it wields. Consequently, this establishes a powerful paradigm where even smaller, more efficient models can achieve state-of-the-art results, contingent not on their size, but on the instruments they are equipped with.

- C.2 WHY VISUAL TOOLS HELP

Our framework decomposes complex reasoning tasks into manageable steps, each resolved either by the model itself or by a high-precision external tool. This design fundamentally shifts the problem-solving burden: instead of requiring flawless internal reasoning, the model’s primary task becomes effective tool planning. By delegating precise sub-tasks to reliable tools, the model is freed to focus on its core competencies of judgment, synthesis, and integrating the resulting outputs.

Perception VSP-Verify Reasoning Accuracy

Model

Point Loc. Acc. ↑ Base w/ Line w/ Point

Qwen 2.5 VL 3B 2.47 50.91 57.92 (+7.01) 49.09 (-1.82) Qwen 2.5 VL 7B 47.01 50.85 57.68 (+6.83) 57.87 (+7.02) Qwen 2.5 VL 32B 6.54 53.12 61.31 (+8.19) 87.87 (+34.75) Qwen 2.5 VL 72B 50.00 52.34 61.57 (+9.23) 87.53 (+35.19)

Our POINT Tool 100.0 – – –

- Table 10: From Perception to Reasoning: Impact of our POINT tool. (Left) Comparison of start-point localization accuracy between Qwen 2.5 VL models and our specialized tool. (Right) Zero-shot reasoning performance on VSP-Verify task when augmented with tool-generated visual context (Line/Point). The significant gains in 32B/72B models demonstrate that high-precision perception is the key to unlocking agentic reasoning.

Perception Tools Help MLLMs to See Our framework leverages expert perception tools to overcome the intrinsic perceptual limitations of MLLMs. As shown in Table 10, in VSP-verification, our

expert POINT tool achieves perfect localization accuracy (100.0% vs. ∼ 50.0% for baselines), and providing its coordinate output as context boosts the downstream zero-shot reasoning performance by an average of +18.79 points. This principle holds even with imperfect tools: for the Jigsaw task, our DETECTBLACKAREA tool achieves only 72.6% accuracy, yet it still provides a significant perceptual advantage, underscoring the value of delegating such challenges to specialized tools.

Manipulating Tools help MLLMs to verify Our manipulating tools empower the model to formulate and subsequently verify its own hypotheses. For example, in the VSP-Verify task, we teach the model to call DRAW2DPATH to explicitly draw a red line on the frozen lake question picture. The problem is thus converted to verifying whether the red line crosses the blue ice holes. As shown in Table 10, even under a zero-shot context-appending setting, the DRAWLINE does help improve the judge accuracy of the model, yielding an average performance improvement of +7.82 points. Similarly, in the Jigsaw task, the model can actively invoke the INSERTIMAGE tool to compose and evaluate different candidate pieces by inserting them into the original image, enabling more informed decision-making.

High-Quality Trajectory data help MLLMs to plan While augmenting context with tool outputs is effective for zero-shot reasoning, this strategy alone is insufficient for achieving optimal performance. Tool-Cold-Start addresses this gap by explicitly teaching the model two foundational capabilities: how to use tools correctly and how to recognize the patterns where they should be applied. As shown in Table 2, for the 7B models, adding the Tool-Cold-Start phase before ToolGRPO yields a massive performance improvement of +24.93 points on VSP and +19.82 points on Jigsaw compared to using Tool-GRPO alone. Besides this, the inclusion of reflection data during the Cold-Start phase provides further benefits to the model’s reasoning. As shown in Table 3, when ASTAR search is disabled, training with reflection data yields a substantial improvement over the no-reflection checkpoints (91.36 vs. 67.27).

- C.3 THE DUAL ROLE OF COLD-START SUPERVISION

A central finding of our work concerns the dual role of the Tool Cold Start (TC, SFT) phase, which highlights a critical trade-off between imparting expert knowledge and preserving a model’s exploratory freedom. Our results suggest that the decision to include a supervised pre-training stage is not universally beneficial, but rather highly contingent on the nature of the task.

For complex, structured tasks with discernible optimal solutions, such as VSP and Jigsaw, the SFT phase provides a decisive advantage. In these scenarios, discovering an effective tool-use trajectory from scratch is a non-trivial exploration problem for the model due to its inherent reasoning or knowledge deficits. By exposing the model to high-quality, deterministic solution paths, the Tool Cold Start phase effectively bootstraps the learning process, instilling a strong inductive bias towards a correct strategy. The empirical results in Table 2 validate this unequivocally: for our 7B models, adding this SFT phase before Tool-GRPO yields a massive performance improvement of +24.93 points on VSP and +19.82 points on Jigsaw compared to using Tool-GRPO alone.

Conversely, for open-ended and highly generalized domains like GUIQA, the limitations of this predefined guidance become apparent. In such settings, the optimal tool-use strategy is often unknown even to human designers, making any human-designed trajectory likely sub-optimal. We find that a rigid SFT phase can inadvertently restrict the model’s exploratory freedom during subsequent RL by creating a strong policy bias, which hinders the discovery of more effective, emergent strategies. This effect is clearly observed in our results for the 7B model on the WebMMU benchmark, where the standalone Tool-GRPO approach actually outperforms the combined pipeline (72.97 vs. 68.16).

This dichotomy suggests a key principle for training tool-augmented agents: while injecting expert knowledge via SFT is a powerful method for tasks with well-defined solution spaces, a pure reinforcement learning approach like Tool-GRPO may be superior for more dynamic and general tasks that benefit from unconstrained exploration.

VSPO VSP WebMMU

Model

Nav Verify Overall Nav Verify Overall Avg. Act. Comp. Reason.

Qwen2.5-VL 3B 5.67 50.91 26.53 7.50 49.80 26.73 45.39 55.89 51.82 34.95 + Direct SFT 27.42 49.66 38.15 34.50 44.00 38.82 46.54 61.38 54.46 32.31 + Direct GRPO 2.78 50.00 24.55 18.33 50.00 32.73 48.44 56.30 51.49 41.41 + TC (cold-start) 14.67 84.81 47.01 23.33 84.40 51.09 35.03 44.72 42.24 24.82 + TG (tool-GRPO) 11.22 50.00 29.10 22.67 50.00 35.09 58.88 72.15 62.05 47.87 + TC + TG 73.00 98.44 84.73 92.17 97.80 94.73 63.48 81.71 57.43 53.01

- ∆ vs. base 67.33 47.53 58.20 84.67 48.00 68.00 18.09 25.82 5.61 18.06 Qwen2.5-VL 7B 5.22 48.96 25.39 12.33 47.00 28.09 59.08 67.48 69.31 48.46

+ Direct SFT 33.68 51.30 42.18 42.67 51.40 46.64 55.62 65.65 63.70 44.79 + Direct GRPO 10.33 49.48 28.38 12.50 51.40 30.18 70.19 83.54 69.31 60.94 + TC (cold-start) 31.58 94.01 61.69 41.00 93.60 64.91 51.63 64.63 54.13 41.12 + TG (tool-GRPO) 65.89 52.47 59.70 88.17 55.20 73.18 72.97 88.62 66.34 64.61 + TC + TG 73.44 98.70 85.09 96.33 99.20 97.64 68.16 82.32 67.33 58.30

- ∆ vs. base 68.22 49.74 59.70 84.00 52.20 69.55 9.08 14.84 −1.98 9.84

- Table 11: Sub-task Breakdown for Single Task Study. Detailed results for VSPO (Navigation and Verification), VSP (Navigation and Verification), and WebMMU (Avg. = overall average, Act. = Agentic Action, Comp. = Visual Comprehension, Reason. = Multi-step Reasoning). Best is bold, second-best is underlined. See Table 2 for complete results including Jigsaw, BLINK-J, and GUIChat.

Model

VSPO VSP WebMMU

Nav Verify Overall Nav Verify Overall Avg. Act. Comp. Reason.

Qwen2.5-VL 7B 5.22 48.96 25.39 12.33 47.00 28.09 59.08 67.48 69.31 48.46 + TC 9.11 49.35 27.66 12.83 49.20 29.36 43.16 41.87 58.75 37.15 + Rnd TC 7.33 49.35 26.71 14.17 49.80 30.36 40.51 42.89 52.15 33.63 + TG 1.78 50.00 24.01 14.67 50.00 30.73 56.71 67.68 55.78 49.19 + TC + TG 2.56 50.00 24.43 9.83 49.80 28.00 55.15 59.76 59.08 50.07 + Rnd TC + TG 35.00 84.42 57.78 27.50 71.00 47.27 59.62 69.11 64.36 50.66

Rnd TC + Rnd TG 47.78 95.84 69.94 65.33 95.20 78.91 60.98 70.93 66.34 51.40 ∆ vs. base +42.56 +46.88 +44.55 +53.00 +48.20 +50.82 +1.90 +3.46 -2.97 +2.94

- Table 12: Sub-task Breakdown for Generalization Study. Detailed results for VSPO (Navigation and Verification), VSP (Navigation and Verification), and WebMMU (Avg. = overall average, Act. = Agentic Action, Comp. = Visual Comprehension, Reason. = Multi-step Reasoning) from Table 4. Rnd TC and Rnd TG indicate randomized training on trajectories from 1-shot to 3-shot prompts. Best is bold, second-best is underlined.

Model

VSPO VSP V*

Nav Verify Overall Nav Verify Overall Attr. Spatial.

Closed-Source Models

Gemini 2.5 flash 15.44 68.96 40.12 34.50 76.40 53.55 74.78 59.21 GPT 5 26.89 42.86 34.25 48.17 64.60 55.64 72.17 78.95 Claude 4 sonnet 37.56 67.92 51.56 48.17 66.00 56.27 59.13 60.53

Open-Source Models

Qwen 2.5 VL 3B 5.67 50.91 26.53 7.50 49.80 26.73 38.26 52.63 Qwen 2.5 VL 7B 5.22 48.96 25.39 12.33 47.00 28.09 64.35 61.84 Qwen 2.5 VL 32B 7.56 53.12 28.56 24.33 45.40 33.91 72.17 72.37 Qwen 2.5 VL 72B 17.22 52.34 33.41 28.00 52.40 39.09 79.13 81.58 InternVL3 78B 7.22 52.60 28.14 21.67 51.20 35.09 80.87 81.58

Tool-Planning Models

Qwen 2.5 VL 7B + Tools 11.89 48.96 28.98 17.83 45.60 30.45 49.57 63.16 Qwen 2.5 VL 72B + Tools 29.56 51.43 39.64 40.83 50.00 45.00 62.61 71.05 GPT 5 + Tools 38.11 69.87 52.75 61.67 83.00 71.36 66.96 75.00 DeepEyes 4.11 17.27 10.18 9.00 16.00 12.18 69.57 64.47 PixelReasoner 7.33 37.79 21.38 12.67 38.80 24.55 48.70 60.53 AdaReasoner 7B 47.78 98.31 71.08 63.00 96.80 78.36 64.35 80.26 ∆ vs. base +42.56 +49.35 +45.69 +50.67 +49.80 +50.27 0.00 +18.42

- Table 13: Sub-task breakdown for Main Results. Detailed results for VSPO (Navigation and Verification), VSP (Navigation and Verification), and V* (Attr. = Attribute Recognition, Spatial. = Spatial Relationship Reasoning) from Table 5. Best is bold, second-best is underlined.

