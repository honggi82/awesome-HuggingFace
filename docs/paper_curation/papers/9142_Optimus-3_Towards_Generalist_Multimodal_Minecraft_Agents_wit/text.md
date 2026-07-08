# Optimus-3: Dual-Router Aligned Mixture-of-Experts Agent with Dual-Granularity Reasoning-Aware Policy Optimization

Zaijing Li, Rui Shao, Member, IEEE, Yuquan Xie, Gongwei Chen, Weili Guan, Dongmei Jiang, Yaowei Wang, Member, IEEE, and Liqiang Nie, Senior Member, IEEE https://cybertronagent.github.io/Optimus-3.github.io/

## arXiv:2506.10357v2[cs.AI]10Feb2026

Abstract—Developing generalist agents capable of solving open-ended tasks in visually rich, dynamic environments remains a core pursuit of embodied AI. While Minecraft has emerged as a compelling benchmark, existing agents often suffer from fragmented cognitive abilities, lacking the synergy between reflexive execution (System 1) and deliberative reasoning (System 2). In this paper, we introduce Optimus-3, a generalist agent that organically integrates these dual capabilities within a unified framework. To achieve this, we address three fundamental challenges. First, to overcome the scarcity of reasoning data, we propose a Knowledge-Enhanced Automated Data Generation Pipeline. It synthesizes high-quality System 2 reasoning traces from raw System 1 interaction trajectories, effectively mitigating hallucinations via injection of domain knowledge. We release the resulting dataset, OptimusM4, to the community. Second, to reconcile the dichotomous computational requirements of the dual systems, we design a Dual-Router Aligned MoE Architecture. It employs a Task Router to prevent task interference via parameter decoupling, and a Layer Router to dynamically modulate reasoning depth, creating a computational “Fast Path” for System 1 and a “Deep Path” for System 2. Third, to activate the reasoning capabilities of System 2, we propose Dual-Granularity Reasoning-Aware Policy Optimization (DGRPO) algorithm. It enforces Process-Outcome Co-Supervision via dual-granularity dense rewards, ensuring consistency between the thought process and the answer. Extensive evaluations demonstrate that Optimus-3 surpasses existing state-of-the-art methods on both System 2 (21% on Planning, 66% on Captioning, 76% on Embodied QA, 3.4× on Grounding, and 18% on Reflection) and System 1 (3% on Long-Horizon Action) tasks, with a notable 60% success rate on open-ended tasks.

Index Terms—Open-World Agent, Multimodal Large Language Model, Reinforcement Learning.

✦

#### 1 INTRODUCTION

Building generalist agents that can solve open-ended tasks in visually rich, open-world environments is a long-standing vision of embodied AI [1]–[4]. Among such environments, Minecraft [5], [6] has emerged as a compelling benchmark for studying open-world agents, due to its diverse scenes and objects, vast action space, and the open-ended nature of tasks. Recent advances in Minecraft [7]–[12] have empowered agents with Multimodal Large Language Models (MLLMs) [13]–[15], enabling impressive performance on programmatic, pre-defined objectives. However, these agents often exhibit brittle behavior when confronted with dynamic and openended instructions, where success necessitates both imme-

- • Zaijing Li and Yaowei Wang are with the School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen Campus), Shenzhen 518055, China and Pengcheng Laboratory, Shenzhen 518000, China E-mail: lzj14011@gmail.com, wangyw@pcl.ac.cn
- • Rui Shao, Yuquan Xie, and Liqiang Nie are with School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen Campus), Shenzhen 518055, China E-mail: rshaojimmy@gmail.com, nieliqiang@gmail.com
- • Weili Guan and Gongwei Chen are with the School of Information Science and Technology, Harbin Institute of Technology (Shenzhen Campus), Shenzhen 518055, China and Pengcheng Laboratory, Shenzhen, China and Shenzhen Loop Area Institute, Shenzhen, China E-mail: honeyguan@gmail.com, cgwfeel@163.com
- • Dongmei Jiang is with Pencheng Laboratory, Shenzhen 518000, China E-mail: jiangdm@pcl.ac.cn
- • The corresponding authors are Rui Shao and Liqiang Nie.

diate responsiveness to dynamic changes and deliberate analysis of visual cues.

To achieve this, we argue that an agent must emulate the Dual-Process Theory of human cognition [16]: synergizing

- System 1 (fast, reflexive, intuitive) and System 2 (slow, deliberative, analytic) processes. As illustrated in Figure 1, consider the instruction “Craft a diamond sword based on the current inventory”. This open-ended task demands a seamless interplay between two cognitive modes. System 2 takes charge of the deliberative reasoning loop, encompassing Planning, Reflection, and notably, Active Perception (Grounding, Embodied QA). We classify these visual tasks as
- System 2 processes because they transcend passive detection: they demand a deliberative analysis of visual evidence to verify the factual existence of queried entities before decisionmaking. System 1 governs the high-frequency visuomotor loop, executing low-latency control commands (Action) to interact with the environment. Existing Minecraft agents, however, typically possess only a fragmented subset of these abilities, lacking the organic synergy required for robust performance. This raises a fundamental research question: Can we develop a generalist agent that organically integrates System 1 action loops with System 2 reasoning capabilities within a unified framework? Toward this goal, we identify three fundamental challenges centered around the acquisition, modeling, and activation of these dual capabilities.

[Figure 1]

###### ③ Planning (System 2)

Human: Based on your current materials, how can you obtain a diamond sword step by step?

### Optimus-3

Optimus-3: To get a diamond sword, follow these steps in order: S1. Dig down to mine cobblestones. S2. Craft furnace. S3. Mine iron ores. S4. Smelt iron ingot. S5. Craft iron pickaxe. S6. Mine diamonds. S7. Craft diamond sowrd.

[Figure 2]

###### ④ Action (System 1)

###### ② Grounding (System 2)

Craft A Diamond Sword Based on The Current Inventory.

[Figure 3]

[Figure 4]

Human: Show me how to craft iron pickaxe.

Human: Locate the tool you have.

Optimus-3: <think> I can observe a stone pickaxe in the inventory. </think>

[Figure 5]

Optimus-3: [action sequence]

[Figure 6]

[Figure 7]

[Figure 8]

###### ⑤ Reflection (System 2)

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Human: Reflect on current status.

① Captioning (System 2)

###### ⑥ Embodied QA (System 2)

Optimus-3: <think> I have now obtained the necessary materials for crafting a diamond sword: two diamonds and one wooden stick. </think> <answer> Continue. </answer>

[Figure 13]

Human: Describe the environment and inventory in the view.

Optimus-3: <answer> I’m standing on a grassy plain. The inventory bar shows various items, including a stone

Human: Determine if the task is complete.

[Figure 14]

pickaxe, a crafting table, seeds, raw meat, and flowers. </answer>

Optimus-3: <think> I can observe a diamond sword in the current scene. </think> <answer> Complete. </answer>

- Fig. 1: Given the task Craft a diamond sword based on the current inventory, Optimus-3 employs Captioning to perceive and interpret the inventory information, Grounding to select appropriate tools, Planning to generate sub-goals based on available materials, Action to execute these sub-goals sequentially, Reflection to assess the current task state, and Embodied QA to verify whether the task has been successfully completed.

- Challenge I: Scarcity of Domain-Specific System 2 Reasoning Traces. While raw gameplay data (action trajectories) is abundant [17], [18], there is a critical scarcity of data capturing the reasoning processes, such as hierarchical planning, visual grounding, and reflection. Manually annotating these complex cognitive traces is prohibitively expensive. A naive strategy is to leverage general-domain MLLMs [13]–[15] for automated annotation. However, these models, while possessing strong general reasoning, lack the “knowledge” of Minecraft (e.g., crafting recipes, physics rules). Consequently, they frequently generate unfeasible plans or hallucinated responses [13], [19]. Bridging the gap between general-purpose reasoning and domain-specific knowledge to construct high-quality System 2 reasoning traces remains a significant hurdle.
- Challenge II: Computational Conflicts in Coupling System 1 and System 2. System 1 (Action) demands highfrequency inference with low latency, relying on shallow, reflexive processing of local context. In contrast, System 2 (Planning, Reflection) operates at a lower frequency but requires deep, computation-intensive reasoning over long horizons. Standard dense architectures or conventional Mixture-of-Experts (MoE) [20], [21] enforce a uniform computational depth, resulting in a dilemma: they are either too computationally heavy for real-time control or too shallow for complex reasoning. Furthermore, training these heterogeneous tasks within a shared parameter space often leads to task interference [22], [23]. Designing an architecture that adaptively allocates computational resources, provides a fast path for System 1 and a deep path for System 2, and simultaneously avoids task interference is highly challenging.

Challenge III: Optimizing System 2 Reasoning under Sparse Outcome Feedback. Given Minecraft’s infinite diversity in scenes and dynamics, Supervised Fine-Tuning (SFT) is insufficient for agent training. While Reinforcement Learning (RL) [25], [26] offers a pathway to activate such reasoning capabilities through exploration [27], [28], adapting it to Minecraft presents distinct challenges. (i) Visual-Logic Misalignment: Unlike text-only tasks, Minecraft requires reasoning over high-dimensional visual inputs. Existing methods [25], [26], [28] often struggle to ground reasoning in visual evidence, leading to perceptual hallucinations. (ii) Lack of Process Supervision: Standard RL typically relies on sparse outcome rewards, treating the reasoning process as a black box. This coarse supervision fails to penalize flawed intermediate logic that accidentally yields correct answers, hindering the acquisition of strictly valid and grounded reasoning chains.

In this paper, we propose Optimus-3, a generalist agent in open-world of Minecraft, which is endowed with comprehensive capabilities in perception, planning, action, and reflection (depicted in Figure 2). To address the aforementioned challenges, we propose targeted improvements across three key dimensions: data generation, model architecture, and training methodology.

Knowledge-Enhanced Data Generation Pipeline for System 2 Tasks. To address the scarcity of domain-specific reasoning data (Challenge I), we introduce a knowledgeenhanced pipeline that synthesizes high-quality reasoning traces from raw action trajectories. Distinct from conventional pipelines that rely solely on the generative priors of generalist MLLMs, our framework explicitly injects Minecraft-specific

###### A. Overview of Optimus-3

###### B. Dual-Router Aligned MoE Architecture

###### C. Performance Comparison

###### Planning

###### EQA

Optimus-3 (Ours) Previous SOTA Qwen2.5-VL GPT-4o

Add & Norm

Two sheep.

- Step 1: Chop a tree.
- Step 2: Craft 4 planks.
- Step3:Craftcraftingtable. Grounding

LayerGating

- Layer L-1

Layer L

- Layer L-2

[Figure 15]

[Figure 16]

###### Action

Action

ES EP EV EA

EG ER

Planning

[forward] [jump]

...

95

94

###### Captioning

92 78

I’m standing on a grassy plain. The inventory bar shows...

###### Reflection

Task Gating

Layer 1

Continue.

[Figure 17]

###### Layer Router

Task Router

23 20

Captioning

Reflection

###### Dual-Router Aligned MoE

###### Skipping

Selection

66

56 47

78

18

[Figure 18]

[Figure 19]

1 2 3 ... L-2 L-1 L

ViT Text Tokenizer

Layer

Expert EP EV EA EG ER

46

shared expert

softmax

softmax

MLP

MLP

How to get an iron sword?

[Figure 20]

task expert

80 81

Show me how to mine stone.

Embodied QA

Describe the image. How many sheep in the view?

Grounding

visual token

Locate the stone pickaxe.

text token

Reflect current situation.

- Fig. 2: (A): Overview of Optimus-3. Given observations and instructions, Optimus-3 couples System-1 fast reaction (Action) and System-2 deliberate reasoning (Embodied QA, Planning, Grounding, Reflection) within the Dual-Router Aligned MoE architecture. (B): The details of Dual-Router Aligned MoE architecture. Horizontally, Task Router assigns each input to its corresponding task expert together with a shared knowledge expert. Vertically, Layer Router accelerates latency-sensitive action inference by selectively skipping intermediate layers. Both routing decisions are made once before the forward pass. (C): Performance comparison of Optimus-3 against current task-specific SOTA agents, GPT-4o [24], and Qwen2.5-VL [13].

knowledge (e.g., expert models, knowledge bases, environment feedback) as ground-truth constraints. It employs such knowledge to guide and verify the accuracy of the reasoning content, significantly reducing the hallucinated content in the annotations. This yields OptimusM4, a large-scale, Multi-Modal Multi-task Minecraft dataset that provides the rigorous supervision needed to bootstrap the agent’s reasoning capabilities.

Dual-Router Aligned MoE Architecture for System 1/2 Synergy. To reconcile the computational conflicts between reflexive action and deliberative reasoning (Challenge II), we propose the Dual-Router Aligned MoE architecture. As depicted in Figure 2, it structurally enforces the organic coupling of System 1 and System 2 through a novel twodimensional routing mechanism. Horizontally, we employ a Task Router to assign distinct experts to heterogeneous tasks, achieving orthogonal parameter decoupling to prevent task interference. Vertically, we introduce a Layer Router that dynamically modulates the inference depth based on the task’s cognitive complexity. It constructs a computational “Fast Path” with reduced layer activation for latency-sensitive System 1 actions, while preserving a “Deep Path” utilizing the full network depth for complex System 2 reasoning. This design breaks the redundancy of static models, enabling efficient, conflict-free coexistence of dual cognitive processes.

Dual-Granularity Reasoning-Aware Policy Optimization for System 2. To endow agents with rigorous System 2 capabilities, we propose a Dual-Granularity ReasoningAware Policy Optimization (DGRPO) algorithm. Unlike conventional RL that relies on sparse outcome signals, DGRPO establishes a novel Process-Outcome Co-Supervision paradigm to mitigate inconsistencies between reasoning process and answers. Central to this framework is our Dual-Granularity Dense Reward mechanism, designed to address specific reasoning failures: (i) The DependencyAware Synthesis Reward incorporates domain knowledge graphs to enforce topological consistency, ensuring the

agent’s thought process strictly adheres to hierarchical logic; (ii) The Hallucination-Aware Consistency Reward acts as a fine-grained perceptual verifier, explicitly penalizing nonexistent entities within the reasoning trace to align high-level thought with low-level visual evidence. By providing dense feedback on reasoning process, DGRPO compels the agent to “think-before-act,” significantly enhancing robustness and generalization in open-ended dynamic environments.

We conducted comprehensive evaluations in the openworld environment of Minecraft. Experimental results show that Optimus-3 outperforms both general MLLMs and previous SOTA agents in Minecraft across a wide range of tasks. Compared to previous SOTA, Optimus-3 achieves improvements of 21% on Planning, 3% on Long-Horizon Action, 66% on Captioning, 76% on Embodied QA, 3.4× on Grounding, and 18% on Reflection, respectively. Notably, on Open-Ended tasks, Optimus-3 achieves an average success rate of 60%, whereas existing agents almost never succeed due to their lack of multi-dimensional capabilities. In summary, our contributions are as follows:

- • We propose Optimus-3, the first generalist Minecraft agent that organically integrates System 1 action loops with System 2 reasoning capabilities (planning, perception, reflection) within a unified framework.
- • We propose a knowledge-enhanced data generation pipeline that synthesizes high-quality System 2 reasoning traces from interaction trajectories. Based on it, we release Multi-Modal, Multi-task Minecraft dataset OptimusM4 for the community.
- • We design the Dual-Router Aligned MoE architecture, which resolves computational conflicts by decoupling task parameters horizontally and adapting reasoning depth vertically, enabling efficient System 1/2 synergy.
- • We propose a Dual-Granularity Reasoning-Aware Policy Optimization algorithm. It introduces a process-outcome co-supervision mechanism to align reasoning chains with visual evidence, enabling robust decision-making

[Figure 21]

C D E

Instruction Instruction

Instruction

[Figure 22]

Instruction

[Figure 23]

[Figure 24]

[Figure 25]

Add & Norm

Policy

MLLM

MLLM

[Figure 26]

Action

- A

- B

Optimus-3

latent tokens

sub-goals

- 1. chop a tree
- 2. craft 4 planks
- 3. craft 2 sticks

[Figure 27]

[Figure 28]

Instruction

Self-Attention

[Figure 29]

LLM

[Figure 30]

[Figure 31]

Multi-dimensional Abilities

Policy

Policy

def craft_planks(num): choose_table() for i in num:

Action Planning EQA Grounding Reflection Captioning

move_logs() yield act

Action

Action

- Fig. 3: Different agent framework in Minecraft. (A) Goal-conditioned policy which based on Transformer-XL architecture. (B) Function calling, which employs LLM to generate executable functions. (C) (M)LLM as the high-level planner, which then employs a goal-conditioned policy to generate low-level actions. (D) MLLM generates latent tokens that serve as conditioning inputs for the policy. (E) End-to-end MoE architecture (Ours) which endowed with multi-dimensional capabilities.

under dense feedback.

a task router to achieve orthogonal parameter decoupling, a layer router implements adaptive reasoning depth.

Reinforcement Learning. Early deep Reinforcement Learning (RL) algorithms [46]–[48] established the foundations of learning control policies from high-dimensional feedback but suffered from instability and limited scalability to large models. Proximal Policy Optimization (PPO) [25] has since become the mainstream on-policy algorithm for continuous control and LLM fine-tuning, thanks to its clipped surrogate objective that stabilizes policy updates, yet it remains sampleinefficient and sensitive to reward shaping. Building on PPO, Reinforcement Learning from Human Feedback (RLHF) [49], [50] trains a reward model from human preferences and then applies PPO-style optimization to align models with nuanced human objectives, but introduces substantial annotation cost and training instability. To simplify this pipeline, Direct Preference Optimization (DPO) [26] removes the explicit RL loop by recasting preference alignment as a supervised objective over paired responses, yielding more stable and implementation-friendly training. More recently, Group Relative Policy Optimization (GRPO) [28] replaces the critic in PPO with group-wise relative baselines, significantly reducing memory and computation for largescale LLMs while preserving on-policy updates. DeepSeekR1 [27] enhances the reasoning ability of LLM via GRPO, yet its reward design remains relatively coarse, lacking fine-grained supervision for thinking process and final answers. In this paper, we propose Multimodal ReasoningAugmented Reinforcement Learning, which assigns taskspecific fine-grained rewards to heterogeneous tasks and leverages multimodal reasoning to guide the model to focus on informative visual cues.

#### 2 RELATED WORK

Minecraft Agents. The existing Minecraft agent frameworks are illustrated in Figure 3. Early work [6], [17], [18], [29], [30] build goal-conditioned policies based on Transformer-XL architecture, refer to Figure 3 (A). Such policies have limited capability in instruction-following and long-horizon planning, and thus can only execute a few simple, atomic tasks. Figure 3 (B) show that several works [7], [31]–[35] leverage Large Language Models (LLM) to generate executable code, enabling agent-environment interaction through function calling mechanisms. However, this API-call paradigm is fundamentally different from how humans execute low-level actions. To endow agents with human-like coupling between high-level planning and low-level execution, several work [8]–[12] employ MLLMs as planners and use the policy as the executor, refer to Figure 3 (C). Moreover, some work [36], [37] leverage MLLMs to generate latent tokens as conditions for the policy, rather than using explicit textual instructions (Figure 3 (D)). Despite significant advances, these agents still underperform open-ended tasks depicted in Figure 1, due to their lack of multidimensional competencies. As shown in Figure 3 (E), we develop an end-to-end generalist agent Optimus-3, which is equipped with comprehensive capabilities in Minecraft.

Mixture-of-Experts Architectures. In recent years, Mixture of Experts (MoE) architectures [20], [39] have garnered significant attention in the field of Large Language Models (LLMs), owing to their scalability and sparsity in activation [40]– [42]. To further enhance expert specialization, DeepSeekMoE [23] proposed more differentiated expert training strategies, significantly boosting downstream task performance, albeit with increased training complexity. Recently, MoE architectures have been incorporated into MLLMs [21], [22], [43]– [45], significantly enhancing their generalization capabilities and computational efficiency. Despite these advancements, existing MoE architectures still face notable challenges in load balancing and routing optimization [22]. In this paper, we propose a dual-router aligned MoE architecture, in which

#### 3 OPTIMUS-3

In this section, we introduce the framework of Optimus3, designed to organically integrate System 1 action loops with System 2 reasoning capabilities. First, we introduce the Knowledge-Enhanced Data Generation Pipeline (Sec. 3.1), which synthesizes high-quality System 2 reasoning traces

[Figure 32]

Task Pool

[Figure 33]

[Figure 34]

###### Planning

###### Embodied QA

[Figure 35]

###### Grounding

Step1: Chop a tree.

l diamond pickaxe l golden sword

<question> Are there any sheep in the current view? </question> <think> The image shows a player's first-person perspective in Minecraft. The player is standing on a ... </think> <answer> Yes, there are one sheep. </answer>

<think> The image shows a pig on the left side of the screen... </think> <answer>

- Step 2: Craft 4 planks.
- Step 3: Craft crafting table.

###### l ......ironleggings Knowledge Graph

...

[Figure 36]

STEVE-1

[Figure 37]

Obs-Action Pairs

[Figure 38]

[Figure 39]

[Figure 40]

action

Filtering

[Figure 41]

[Figure 42]

</answer>

Env Feedback

W obs

QA Query Env Feedback

Grounding DINO

[Figure 43]

###### Captioning

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Reflection

[Figure 48]

GPT-4o

GPT-4o

- System 1
- System 2

The image shows a player's first-person perspective in Minecraft. The player is standing on a grassy plain and is holding a stone pickaxe in their right hand. In the foreground, there is a white sheep.

My goal is chop a tree. I can see some tress ahead, and I can move forward to chop them.

[Figure 49]

Env Feedback

Env Feedback

- Fig. 4: Knowledge-Enhanced Data Generation Pipeline. The knowledge source is in green. Given a task pool, we utilize a knowledge graph [11] to generate task plans, forming the planning dataset. These plans are then used as instructions for STEVE-1 [18], which interacts with the environment to produce the action dataset. During this process, we randomly sample images and employ expert models [19], [38] with environmental feedback to generate the captioning, embodied QA, and grounding datasets.

from raw System 1 trajectories. Then, we detail the DualRouter Aligned MoE Architecture (Sec. 3.2), specifically engineered to reconcile the computational conflicts between fast, reflexive actions and slow, deliberative reasoning. Subsequently, in Sec 3.3, we elaborate on the Multimodal Reasoning-Augmented RL, a training paradigm designed to activate the agent’s System 2 capabilities. Finally, Sec 3.4 outlines our multi-stage training strategy.

agent state, inventory, and surrounding objects) as groundtruth for GPT-4o [24] to generate detailed captions. These captions serve as the thinking process for System 2 tasks, enabling the agent to reason based on visual observations. The annotation process of System 2 tasks is as follows:

- • Embodied QA: We employ DeepSeek-VL2 [19] to generate question-answer pairs derived from the generated captions, ensuring captions and QA pairs are factually aligned with the environment feedback.
- • Visual Grounding: We incorporate Grounding DINO [54] as a vision expert to annotate objects, providing precise bounding boxes that general MLLMs often miss.
- • Reflection: We employ GPT-4o [24] to annotate execution status based on environment feedback, creating traces of self-correction.

###### 3.1 Knowledge-Enhanced Data Generation Pipeline

Currently, while raw gameplay data (System 1 action trajectories) is abundant, datasets capturing the System 2 reasoning traces are scarce. Relying on general-purpose MLLMs for automated annotation often fails, as they lack the domain-specific knowledge of Minecraft, leading to unfeasible plans or hallucinated responses. For instance, most of generic MLLMs [19], [24], [51], [52] lack an understanding of Minecraft crafting recipes (object synthesis relationships), rendering them incapable of generating feasible plans. Furthermore, these models frequently exhibit hallucinations [53] in vision-centric tasks, such as dense captioning, Embodied Question Answering (EQA), and visual grounding. To address this, we propose a knowledge-enhanced automated data generation pipeline. It injects external knowledge (e.g., domain knowledge graphs, expert models, environmental feedback) as ground-truth constraints to guide and verify the generation process.

We utilize these knowledge sources to filter out hallucinations, and construct Multi-Modal Multi-task Minecraft dataset OptimusM4. It provides the high-fidelity System 2 supervision required to bootstrap the agent’s reasoning capabilities. Data visualizations and detailed statistics are presented in Figure 5.

###### 3.2 Dual-Router Aligned MoE Architecture

Unified modeling of System 1 and System 2 introduces a fundamental architectural dilemma: these two modes impose distinct computational requirements. System 1 (Action) demands high-frequency, latency-sensitive processing of local contexts, whereas System 2 (Planning, Reflection) requires deep, computation-intensive reasoning over long horizons. Standard dense architectures enforce a uniform depth, making them either too slow for real-time control or too shallow for complex reasoning. Furthermore, co-training these heterogeneous tasks often leads to gradient interference and negative transfer [20], [21]. To resolve these conflicts, we propose the Dual-Router Aligned MoE, a task-aware architecture that structurally enforces System 1/2 coupling through a two-dimensional routing mechanism (Figure 2).

As depicted in Figure 4, we first construct a task pool by collecting general items from Minecraft Wiki. For each rollout, we utilize a domain knowledge graph [11] to enforce the topological correctness of crafting paths, ensuring the generated plan is feasible. We then employ an expert policy [18] to sequentially execute the planned sub-goals. Trajectories verified as successful via environmental feedback signals are archived as action data (observation-action pairs). During execution, visual frames are sampled at a fixed frequency. Subsequently, we utilize environmental feedback (including

Input: Given that you already have 1 furnace, 1 crafting_table, 3 iron_ingot, how to get 1 iron_boots ?

Output: I need to carry out the following steps in sequence: Step 1: chop trees to get 1 logs. Step 2: craft 2 stick. Step 3: dig down and mine 3 cobblestone. Step 4: craft 1 stone_pickaxe. Step 5: dig down and mine 4 iron_ore. Step 6: smelt 1 iron_ingot. Step 7: craft 1 iron_boots.

ReflectionPlanningEmbodiedQACaptioningGroundingAction

Input: Given that you already have 1 cobblestone, how to get 1 stone_pickaxe ?

Output: I need to carry out the following steps in sequence: Step 1: chop trees to get 3 logs. Step 2: craft 9 planks. Step 3: craft 4 stick. Step 4: craft 1 crafting_table. Step 5: craft 1 wooden_pickaxe. Step 6: dig down and mine 2 cobblestone. Step 7: craft 1 stone_pickaxe.

Input: Describe this image.

Output: The image shows a Minecraft gameplay scene where the player is using a crafting table in an outdoor grassy area. The crafting interface is open, and the player is in the process of crafting a wooden pickaxe. In the crafting grid, there are two wooden sticks placed vertically in the center and bottom-middle slots, and six wooden planks placed across the top row. The player's inventory is visible at the bottom of the screen, containing only wooden planks. The background reveals grass blocks and the crafting table itself.

[Figure 50]

Output: <think> The image is a first-person Minecraft view of a player inside a small, vertical shaft or hole underground. Visible are the sides of the shaft made up of stone, dirt, andesite, and the bottom of the hole. There are no visible entities, mobs, or structures within the hole just natural blocks. The hotbar is visible at the bottom of the screen: the player currently holds a wooden plank block in their hand. The lighting suggests it is underground, with no direct sunlight visible, and there are no weather effects present. </think> <answer> The player is in an underground shaft or a narrow hole, surrounded by stone, dirt, and granite blocks.</answer>

Input: What type of environment is the player in?

[Figure 51]

Output: <think> This image is a first-person view from the game Minecraft. The player is near the shore of a river or lake, facing a grassy landscape with birch and oak trees spread throughout, characteristic of a forest or flower forest biome. Several patches of flowers and grass are visible on the ground. The sky is clear with a few clouds. The status bar at the bottom shows full health (10 hearts), full hunger (10 drumsticks). The hotbar is visible at the bottom, currently empty with no items in any slots, and the player’s hand is empty, indicating nothing is held. </think> <answer> The player has 10 hearts of health (full health). </answer>

Input: How many hearts of health does the player currently have?

[Figure 52]

Input: Trace the outline of the cow visible outdoors.

Output: <think> The image shows a first-person view from within the game Minecraft, displaying a grassy plains biome under a blue sky with some clouds. In the foreground, there are multiple cows visible: one cow's face is looking directly at the player just ahead, while another cow is partially visible to the left. </think>

[Figure 53]

[Figure 54]

Input: Locate the position of the horse in the view.

Output: <think> The image shows a grassy plains biome with several trees in the background. A grey horse is visible in the middle distance, standing on the grass. In the foreground, there is a fire with orange flames and black smoke particles rising, likely caused by adjacent lava or burning netherrack. </think>

[Figure 55]

[Figure 56]

Input: determine whether the task can continue.

Output: <think> The player has full health and hunger bars, indicating no imminent danger or status issues. There are no hostile mobs visible, and the environment is a typical underground mining tunnel composed mostly of stone blocks. Thus, the task of crafting a diamond sword can continue without any critical interruptions. </think> <answer>The task can continue. </answer>

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Input

Mouse:(2,0) Keyboard:w

Mouse:left_click Keyboard:w

Mouse:left_click Keyboard:null

Mouse:left_click Keyboard:w

Mouse:(-3,2) Keyboard:w

Mouse:(1,0) Keyboard:null

Mouse:left_click Keyboard:w

Mouse:left_click Keyboard:null

Output

###### (a) # Samples per Task (b) Biomes Distribution

(c) Tech-level Distribution

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Ocean 2% Desert 12%

19k

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Iron 19%

Forest 18%

14k 15k 12k

Wood 25%

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Beach

10k

- 7% Hill
- 8%

2k

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Diamond 13%

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Redstone 10%

Plain 46%

Stone 20%

Captioning

Reflection

Grounding

Planning

Action

QA

Gold 13%

E

Cave 7%

- Fig. 5: Data visualization and statistics of the OptimusM4 dataset. Top: Representative samples across the Planning, Captioning, Embodied QA, Grounding, Reflection, and Action tasks. Bottom: Statistical overview, detailing sample counts, biome distribution, and tech-level distribution for Action data.

- 3.2.1 Horizontal Routing: Orthogonal Parameter Decoupling To prevent gradient interference between conflicting tasks (e.g., visual grounding vs. action prediction), we employ a semantic Task Router. Unlike token-level soft routing which can lead to load imbalance [22], it deterministically directs input tokens to topologically distinct parameter spaces based on the task type T .

For the l-th layer, we replace the standard Feed-Forward Network (FFN) with a hybrid expert module comprising a Shared Knowledge Expert ES and Task-Specific Experts ET . The output hidden state hl is computed as:

hl = xl ⊕ ( ES(xl)

General Knowledge

+ ET (xl)

Task-Specific

), (1)

where xl denotes the input features (after Attention and Normalization). Here, ES captures universal semantic patterns across tasks to facilitate positive transfer, while ET is exclusively activated for task T . This orthogonal design ensures that the optimization of System 1 control policies is isolated from System 2 reasoning tasks, maintaining a shared representational foundation without task interference.

- 3.2.2 Vertical Routing: System-1/2 Adaptive Reasoning Complementing the horizontal separation, we introduce a Layer Router to align the model’s reasoning depth with the cognitive complexity of the task. Grounded in the DualProcess Theory [16], this router implements a Dynamic Compute Allocation mechanism. It recognizes that reflexive tasks (System 1) prioritize inference speed, whereas deliberative tasks (System 2) demand reasoning depth.

Formally, we model the vertical routing as a layer selection problem based on task-specific importance. Let eT denote the task embedding. We employ a layer router to project the task representation into a layer-wise importance distribution α ∈ RL via a Softmax function:

##### α = Softmax(MLP(eT )), (2)

where αl quantifies the contribution of the l-th layer to the current task. To achieve adaptive inference, we introduce a filtering threshold τ to dynamically determine the set of active layers ΦT :

##### ΦT = {l ∈ {1,...,L} | αl ≥ τ}. (3)

During the forward pass, layers with importance scores below the threshold are bypassed via residual connections. The state transition is formulated as:

##### xl+1 = xl + I(l ∈ ΦT ) · Attn(xl) + MoEBlockl(xl) , (4)

where I(·) is the indicator function. This mechanism naturally induces a cognitive dichotomy:

- • System 2 Mode (Deep Path): For reasoning-intensive tasks (e.g., Planning), the router yields a high-entropy distribution or uniformly high importance across critical

reasoning blocks, resulting in |ΦT | ≈ L to ensure deep reasoning.

- • System 1 Mode (Fast Path): For interaction-intensive tasks (e.g., Action), the router assigns high probability mass only to a few essential perceptual and control layers. This filters out redundant intermediate computation (i.e., |ΦT | ≪ L), significantly reducing latency.

By selectively skipping calculations for non-essential layers, the Dual-Router Aligned MoE achieves a dynamic equilibrium between computational efficiency and reasoning capability, effectively synthesizing the reflexive nature of System 1 and the deliberative depth of System 2 within a unified framework.

3.3 Dual-Granularity Reasoning-Aware Policy Optimization

In the open-ended and dynamic environment of Minecraft, relying solely on Supervised Fine-Tuning (SFT) is insufficient. While Reinforcement Learning (RL) [28] offers a solution to activate reasoning capabilities through exploration [27], it lacks fine-grained supervision of the reasoning process. To this end, we propose the Dual-Granularity ReasoningAware Policy Optimization (DGRPO) algorithm. It explicitly activates System 2 capabilities by establishing a ProcessOutcome Co-Supervision paradigm, utilizing dense rewards to enforce logical consistency between the intermediate reasoning and the final answer.

3.3.1 Overview of Framework

DGRPO consists of two progressive phases: a visualreasoning cold-start phase and a preference alignment phase via reinforcement learning.

- Phase 1: Visual-Reasoning Cold Start. To initialize the model with basic reasoning capabilities, we perform finetuning using Chain-of-Thought (CoT) [55] templates. Unlike standard text-only CoT, our approach compels the model to explicitly describe the visual scene within its reasoning trace, thereby grounding high-level reasoning in low-level visual evidence. The objective is defined as:

LSFT = −

T

t=1

log πθ (yt | xv,xins,y<t), y = [y(think);y(ans)].

(5) Here, xv and xins denote visual and instructional inputs, respectively. The output y is decomposed into a reasoning process y(think) and a final answer y(ans). This phase serves to mitigate initial hallucinations and activate the model’s reasoning capabilities.

- Phase 2: Optimization via DGRPO. To further robustify the reasoning process, we employ Group Relative Policy Optimization (GRPO) [28] as the optimization backbone. GRPO is particularly advantageous for large-scale MLLMs as it eliminates the need for a separate value function critic. Instead, it utilizes the group average reward as a baseline, reducing computational overhead while stabilizing training. Formally, given an input x, the policy πθ samples a group of G outputs {y1,y2,...,yG}. The optimization objective is formulated as:

|yi|

G

1 G

1 |yi|

min rθ (x,yi)Ai,t,

t=1

i=1

clip rθ (x,yi),1 − ε,1 + ε Ai,t − λDKL

(6)

πθ (yi,t | x,yi,<t) πθ

, (7)

rθ (x,yi) =

(yi,t | x,yi,<t)

old

###### Planning

###### Grounding

###### Reflection

###### Embodied QA

Input

Input

Input

Input

[Figure 92]

[Figure 93]

[Figure 94]

Given that you already have 3 planks, 1 stick, how to get 1 wooden_hoe ?

What tool is the player currently holding and using?

Locate the pig in the view.

Determine whether the task “craft a golden” axe can continue.

Thinking reward

Thinking reward

Thinking reward

Thinking reward

[Figure 95]

The image shows... There are two sheep on the left side of the image, and there is no pig in the picture.

wooden_hoe: {'stick': 2, 'planks': 2, } stick: {'planks': 2} planks: {'logs': 2} crafting_table: {'planks': 4}

The image shows... where the player is underground, surrounded by stone blocks and mining a crafting table using a iron pickaxe.

The player is underground, mining with a stone pickaxe, and both health and hunger levels are sufficient for continued task execution. There are no indicators that progress toward the task cannot continue. Therefore, the task can proceed successfully.

PlanPlanning ning

PlanPlanning ning

PlanPlanning ning

PlanPlanning ning

[Figure 96]

-0.15

+0.75

-0.20

-0.50

[Figure 97]

[Figure 98]

Answer reward

Answer reward

Answer reward

[Figure 99]

Answer reward

[ 383, 254, 433, 372 ]

- step 1: chop trees to get 2 logs
- step 2: craft 3 planks
- step 3: craft 1 stick
- step 4: craft 1 crafting_table
- step 5: craft 1 wooden_hoe

I will analyze the questions in relation to the visual content to derive the correct answers. The player is holding and using a stone pickaxe.

[Figure 100]

[Figure 101]

+0.80

+0.95

ground-truth: [ 359, 218, 423, 346 ] predict: [ 383, 254, 433, 372 ] IoU = 0.36

No elements within the visual display will affect task execution. The task can continue.

[Figure 102]

+0.95

+0.72

[Figure 103]

Input

Output

Reward Function

Rewards

Advantages

1

1

1

Dependency-Aware Synthesis Reward

[Figure 104]

2

2

2

...

...

...

Hallucination-Aware Consistency Reward

- Fig. 6: Visualization examples of the task-specific fine-grained reward functions in DGRPO. For the Planning task, we design a Dependency-Aware Synthesis Reward, which treats the item’s crafting dependency path as thinking reward and assigns fine-grained step-wise supervision as answer reward. For vision-related tasks, we introduce a Hallucination-Aware Consistency Reward that penalizes hallucinated items in the reasoning process and the final answer.

Rthinking encourages the model to explicitly verify material prerequisites in its thought chain, while Ranswer ensures the final plan sequence matches the optimal path.

πref (yi,t | x,yi,<t) πθ (yi,t | x,yi,<t) − log

πref (yi,t | x,yi,<t) πθ (yi,t | x,yi,<t) − 1,

DKL =

Hallucination-Aware Consistency Reward. For tasks requiring precise visual grounding (Embodied QA, Reflection, Grounding), the primary failure mode is perceptual hallucination [53]. We introduce the Hallucination-Aware Consistency Reward (RHAC) to penalize the generation of non-existent entities during reasoning. Let Sgold denote the set of groundtruth objects present in the scene. The reward function is defined as:

(8) where Ai,t denotes the advantage function, ε and λ are hyperparameters, and πθ

old and πref represent the old policy and the reference model, respectively.

- 3.3.2 Dual-Granularity Dense Reward Design Standard GRPO algorithm relies on sparse outcome rewards (e.g., success/failure), which treat the reasoning process as a black box. Crucially, such answer-only supervision fails to penalize flawed reasoning that accidentally yields the correct response (i.e., spurious correlations). DGRPO addresses this by introducing a Process-Outcome Co-Supervision mechanism. We design task-specific dense rewards that provide fine-grained feedback on both the reasoning process

N

1 N

I(yn(think) ∈ Sgold)

##### +rtask(y(ans),y(gold))

RHAC(y) =

##### .

n=1

Rans

Rthink

(10) Here, Rthink explicitly measures perceptual precision (analogous to 1 − CHAIR [56]), forcing the agent to ground its reasoning in visual perception to answer questions. The answer reward Rans adapts to the specific task format:

(Rthink) and the final answer (Rans). As illustrated in Fig. 6, the reward system is tailored for two primary task categories.

Dependency-Aware Synthesis Reward. In long-horizon planning tasks, the core challenge is strictly adhering to the hierarchical synthesis logic of Minecraft. We propose the Dependency-Aware Synthesis Reward (RDAS) to enforce topological consistency with the domain knowledge graph. Let Gsyn be the ground-truth synthesis dependency graph. The total reward RDAS is a weighted sum of the thinking reward Rthink and plan accuracy reward Rans:

- • Embodied QA: We utilize ROUGE-L F1 score to measure the accuracy of the agent’s responses.
- • Reflection: We utilize binary success indicators to measure accuracy of the agent’s decisions.
- • Visual Grounding: We implement a strict IoU-Density Reward to penalize imprecise bounding boxes. Given the Intersection-over-Union (IoU) score u between predicted and ground-truth boxes:

M

N

 

1 N

1 M

I(ym(think) ∈ Gsyn)

I(yn(ans) = yn(gold))

1, if u ≥ α ηu, if β ≤ u < α 0, if u < β

##### ,

RDAS(y) =

+

(11)

m=1

n=1

Rans =



Rthink

Rans

(9) where I(·) is the indicator function, and M and N represent the number of reasoning steps and plan steps, respectively.

Here, α and β are hyperparameters in the range (0,1), and η is a weighting coefficient.

TABLE 1: Hyperparameter setting for each training phase.

###### Stage-1 Stage-2 Stage-3

Hyperparameter Stage-1 Stage-2 Stage-3 Optimizer AdamW AdamW AdamW Learning Rate 5.0e-5 3.0e-5 1.0e-6 Epochs 2 2 20 Batch Size 12 8 16 Gradient Accumulation 16 16 8 Warm Up Ratio 0.25 0.25 Max Pixels 234416 234416 234416 Num Generations - - 5 Max Prompt Length 2048 2048 2048

SFT

Coldstart

###### RL

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Shared Expert

Shared Expert

Shared Expert

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Task Experts

Task Experts

Task Experts

- Fig. 7: The stage-wise training strategy. Stage-1: We train the shared knowledge expert via Supervised Fine-Tuning (SFT). Stage-2: We train the task experts with multimodal reasoning fine-tuning (Coldstart). Stage-3: We further train the task experts with Reinforcement Learning (RL).

into the predefined task set. For Action tasks, following prior work [12], we employ VPT [17] as action head to generate low-level actions. We collect 230k samples for training stage1, 58k samples for the training stage-2, and 5k samples for training stage-3. These datasets are sourced from our proposed OptimusM4 as well as previous works [8], [12], [17]. All experiments were conducted on 8× NVIDIA L40 GPUs. The hyperparameter setting are shown in Table 1.

Additionally, we add a Format Reward with weight λf to each task, requiring the agent’s response to follow the format < think > ... < /think >< answer > ... < /answer >.

###### 3.4 Training Strategy

To ensure stable convergence, we adopt a three-stage training strategy (as illustrated in Fig. 7) for Optimus-3.

- Stage-1: In this stage, we initialize the shared knowledge expert from a dense model, then train it via supervised finetuning. The shared knowledge expert is trained across all tasks to capture common semantic representations.
- Stage-2: In this stage, we initialize the task experts from shared expert, then freeze the shared knowledge expert. The task experts are trained via the fine-tuning (coldstart) to activate multimodal reasoning capabilities. While the Action expert is trained via imitation learning [12].
- Stage-3: We refine the System 2 experts (Planning, Perception, Reflection) via the DGRPO algorithm. By optimizing against the proposed dual-granularity rewards, the agent learns to generate reasoning chains that are both logically sound and visually faithful.
- 4 EXPERIMENTS

###### 4.2 Evaluation on System 2 Tasks

Evaluation Tasks & Metrics. To comprehensively evaluate the multi-dimensional cognitive abilities of Optimus-3, we construct a MineSys2 Benchmark comprising five categories: Planning, Captioning, Embodied QA, Grounding, and Reflection. For Planning and Reflection tasks, evaluation samples are 103 and 64, respectively. We employ average accuracy (Acc) as the evaluation metric. For the Captioning and Embodied QA tasks, the evaluation includes 134 and 400 samples, respectively. We adopt an LLM-as-Judge [60] approach, employing GPT-4 [24] to assign a score from 1 to 10 for each sample. The average score is then normalized to a value between 0 and 1. For the Grounding tasks, we construct 500 evaluation samples, and use IOU@0.5 [61] as the metric.

###### 4.1 Experiments Setting

Baselines. For the MineSys2 Benchmark, we compare Optimus-3 against generalist MLLMs (GPT-4o [24], Qwen2.5VL [13], Gemini-1.5-pro [52]), and various variants of Qwen2.5-VL [13] that post-trained on OptimusM4 dataset.

Environment. Following the standard evaluation protocol adopted in prior studies [8]–[11], [18], we conduct experiments in the open-world environment of Minecraft on the MineRL [5] platform. It is an archetypal open-world environment that offers rich diversity in resources and biomes, thereby requiring agents to operate under longhorizon and highly variable conditions. In MineRL, the agent receives instructions and observations then produces mouse/keyboard control actions at 20 Hz. For each episode, the agent is spawned without any initial equipment and is initialized at random biomes and positions, leading to substantial variation in environmental dynamics and task difficulty. Therefore, Minecraft serves as a suitable and challenging testbed for evaluating open-world agents.

Results Analysis. As depicted in Table 2, compared to all baselines, Optimus-3 achieves the highest performance across all System 2 tasks. The first four rows of Table 2 reveal the limited capabilities of existing generalist MLLMs, due to the fact that they have not been post-trained in the Minecraft domain. In contrast, Qwen2.5-VL-SFT, which post-trained on OptimusM4, shows a notable performance boost. Furthermore, after applying our proposed DGRPO method, Qwen2.5-VL-RL achieves a 52% improvement in Grounding. More importantly, we observe that the various variants of Qwen2.5-VL exhibit task interference, while Optimus-3 consistently achieves superior performance across tasks. Moreover, compared to token-level routing, the proposed task-level routing demonstrates superior performance on tasks such as Captioning, Planning, and Grounding. We attribute this improvement to the Dual-route Aligned MoE architecture in Optimus-3, which allocates task-specific experts via task router, thereby effectively mitigating conflicts among heterogeneous tasks.

Implementation details. We initialize Optimus-3 with the weights of Qwen2.5-VL-7B [13]. It comprises a large language model (LLM) and a ViT-based [57] visual encoder. Then we adapt it into MoE architecture, comprising one shared knowledge expert and five task-specific experts dedicated to planning, perception, action, grounding, and reflection. Subsequently, we implement the Task Router by fine-tuning a lightweight Sentence-BERT [58] model to classify instructions

- TABLE 2: Evaluation Results of Optimus-3 on MineSys2 benchmark. #Params denotes activated parameters. SFT and RL refer to supervised fine-tuning and reinforcement learning, respectively.

Model #Params

Planning Captioning EQA Grounding Reflection Acc Score Score IoU@0.5 Acc Generalist Multimodal Large Language Model

GPT-4o [24] - 0.20 0.46 0.33 - 0.31 Gemini-1.5-pro [52] - 0.19 0.33 0.33 - 0.34 DeepSeek-VL2 [19] 4B 0.07 0.49 0.51 0.29 0.47 Qwen2.5-VL [13] 3B 0.03 0.37 0.40 0.58 0.47 Qwen2.5-VL [13] 7B 0.05 0.47 0.46 0.18 0.56 Qwen2.5-VL [13] 32B 0.07 0.51 0.49 0.34 0.53 Qwen2.5-VL [13] 72B 0.09 0.52 0.54 0.32 0.53

Post-trained Multimodal Large Language Model

Qwen2.5-VL-SFT [13] 3B 0.76 0.64 0.69 0.69 0.47 Qwen2.5-VL-RL [13] 3B 0.74 0.65 0.70 0.71 0.48 Qwen2.5-VL-SFT [13] 7B 0.79 0.68 0.71 0.52 0.53 Qwen2.5-VL-RL [13] 7B 0.76 0.71 0.68 0.79 0.56 Optimus-3 [token routing] 6.8B 0.88 0.66 0.77 0.75 0.58 Optimus-3 [task routing] 6.8B 0.94 0.78 0.81 0.80 0.66

- TABLE 3: Main Result of Optimus-3 on Long-Horizon Benchmark. We report SR on each task group. † denotes we employ a hierarchical architecture agent from prior work [12], where an MLLM serves as the planner, followed by STEVE-1 [18] acting as the policy to sequentially generate actions. ∗ represents reproduced results under the same settings as other baselines. Optimus-3-Action denotes Optimus-3 trained only on action trajectories.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Method Wood Stone Iron Gold Diamond RedStone Armor

Multimodal Large Language Model† GPT-3.5 [59] 0.40± 0.15 0.20± 0.13 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.00± 0.00 GPT-4o [24] 0.47± 0.23 0.23± 0.09 0.05± 0.04 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.00± 0.00

- Gemini-1.5-pro [52] 0.41± 0.14 0.21± 0.10 0.03± 0.02 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.00± 0.00 Qwen2.5-VL [13] 0.28± 0.15 0.06± 0.03 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.00± 0.00 Qwen2.5-VL-SFT [13] 0.76± 0.11 0.36± 0.07 0.11± 0.05 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.00± 0.00

Goal-conditioned Policy in Minecraft VPT [17] [NeurIPS’22] 0.18± 0.15 0.07± 0.05 0.00± 0.00 0.00± 0.00 0.01± 0.01 0.00± 0.00 0.00± 0.00 GROOT [29] [ICLR’24] 0.34± 0.17 0.17± 0.10 0.08± 0.05 0.01± 0.01 0.01± 0.01 0.03± 0.02 0.04± 0.02 MineCLIP [6] [NeurIPS’22] 0.23± 0.16 0.12± 0.08 0.06± 0.05 0.00± 0.00 0.00± 0.00 0.00± 0.00 0.02± 0.02 STEVE-1 [18] [NeurIPS’23] 0.45± 0.22 0.22± 0.19 0.08± 0.06 0.00± 0.00 0.05± 0.03 0.00± 0.00 0.07± 0.05 Agents in Minecraft Voyager∗ [10] [NeurIPS’23] 0.87± 0.25 0.32± 0.15 0.08± 0.06 0.02± 0.02 0.01± 0.01 0.00± 0.00 0.14± 0.09 DEPS [10] [NeurIPS’23] 0.77± 0.13 0.48± 0.09 0.16± 0.08 0.00± 0.00 0.01± 0.01 0.00± 0.00 0.10± 0.18 MP5∗ [8] [CVPR’24] 0.89± 0.23 0.73± 0.21 0.43± 0.18 0.10± 0.08 0.09± 0.08 0.17± 0.08 0.19± 0.18 JARVIS-1 [9] [TPAMI’25] 0.93± 0.14 0.89± 0.07 0.36± 0.06 0.07± 0.03 0.08± 0.03 0.16± 0.07 0.15± 0.19

- Optimus-1 [11] [NeurIPS’24] 0.98± 0.02 0.92± 0.04 0.46± 0.09 0.08± 0.05 0.11± 0.05 0.25± 0.03 0.19± 0.22
- Optimus-2 [12] [CVPR’25] 0.99± 0.02 0.93± 0.04 0.53± 0.03 0.09± 0.01 0.13± 0.02 0.28± 0.03 0.21± 0.19 Optimus-3-Action 0.93± 0.03 0.87± 0.05 0.49± 0.04 0.03± 0.04 0.03± 0.02 0.09± 0.05 0.15± 0.22

- Optimus-3 0.99± 0.01 0.95± 0.02 0.55± 0.03 0.10± 0.02 0.15± 0.02 0.29± 0.02 0.23± 0.16

###### 4.3 Evaluation on System 1 Tasks

Evaluation Tasks & Metrics. To evaluate the robustness of Optimus-3 in executing long-horizon action sequences, we follow the experimental setup of prior work and conduct experiments on the Long-Horizon benchmark [11]. It comprises 67 tasks spanning 7 technology groups: Wooden (10 tasks), Stone (9 tasks), Iron (16 tasks), Gold (6 tasks), Diamond (7 tasks), Redstone (6 tasks), and Armor (13 tasks). Each task consists of multiple sub-steps that the agent must execute sequentially to reach the final goal. For each task, we perform at least 30 rollouts from different initial positions, and we report the average Success Rate (SR) with standard deviation on each task group as the evaluation metric.

Baselines. We follow the hierarchical architecture of agents from prior work, employing an MLLM (GPT-3.5 [59], GPT4o [24], Qwen2.5-VL [13], Gemini-1.5-pro [52]) as a planner to generate sub-goals, followed by STEVE-1 as a policy to sequentially generate actions. Moreover, we employ current SOTA agents in Minecraft (DEPS [10], Jarvis-1 [9], Optimus-1 [11], and Optimus-2 [12]) as baselines.

Results Analysis. As shown in Table 3, Optimus-3 achieves the highest success rate across all 7 task groups, with particularly strong performance on the Diamond Group, attaining a SR of 15%. It is worth emphasizing that Optimus3 performs both high-level planning and low-level action execution within an end-to-end architecture, whereas all

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

- TABLE 4: Performance comparison on Wooden Pickaxe , Stone Sword , Iron Ingot , Golden Shovel , and Diamond Sword

[Figure 128]

. We report Completion Rate (CR) and Success Rate (SR) for each task. † denotes we employ a hierarchical architecture agent from prior work [12]. ∗ represents reproduced results under the same settings.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Wooden Pickaxe Stone Sword Iron Ingot Golden Shovel Diamond Sword Avg CR SR CR SR CR SR CR SR CR SR CR SR

Model

Multimodal Large Language Model† GPT-5-Instant [62] 0.22 0.05 0.18 0.05 0.14 0.00 0.12 0.00 0.12 0.00 0.15 0.02 GPT-5-Thinking [62] 0.28 0.10 0.24 0.05 0.16 0.00 0.14 0.00 0.12 0.00 0.18 0.03 Gemini-2.5-Flash [63] 0.24 0.10 0.22 0.10 0.16 0.05 0.14 0.05 0.14 0.00 0.18 0.06 Gemini-2.5-Pro [63] 0.36 0.25 0.30 0.20 0.28 0.10 0.22 0.05 0.14 0.00 0.26 0.12 Qwen2.5-VL-7B [13] 0.31 0.20 0.26 0.20 0.24 0.10 0.19 0.05 0.09 0.00 0.21 0.11 Agents in Minecraft Voyager∗ [7] [NeurIPS’23] 0.16 0.10 0.16 0.05 0.14 0.00 0.14 0.00 0.08 0.00 0.14 0.03 DEPS∗ [10] [NeurIPS’23] 0.14 0.10 0.14 0.10 0.12 0.05 0.12 0.00 0.10 0.05 0.12 0.06 MP5∗ [8] [CVPR’24] 0.29 0.20 0.19 0.20 0.18 0.15 0.18 0.05 0.16 0.10 0.20 0.14 Jarvis-1∗ [9] [TPAMI’25] 0.19 0.15 0.18 0.15 0.18 0.10 0.20 0.10 0.12 0.05 0.17 0.11

- Optimus-1∗ [11] [NeurIPS’24] 0.21 0.20 0.18 0.20 0.18 0.15 0.17 0.15 0.12 0.10 0.17 0.16
- Optimus-2∗ [12] [CVPR’25] 0.22 0.25 0.21 0.15 0.18 0.15 0.19 0.10 0.19 0.15 0.19 0.16

- Optimus-3 0.89 0.75 0.86 0.70 0.79 0.65 0.75 0.55 0.69 0.35 0.79 0.60

[Figure 134]

Original Clear Knowledge

100

| |5.0<br><br>47.0 46.0<br><br>18.0 18.0<br><br>59.0<br><br>55.0<br><br>62.0<br><br>94.0<br><br>78.0<br><br>81.0 80.0| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

80

[Figure 135]

[Figure 136]

60

Acc(%)

40

20

[Figure 137]

[Figure 138]

0

Planning Captioning Embodied QA Grounding

- Fig. 8: Ablation Study on Training Data. Original refers to the original Qwen2.5-VL-7B, Clear indicates the Optimus-3 posttrained on data without knowledge, Knowledge represents the Optimus-3 post-trained on OptimusM4.

Fig. 9: Comparison of the average per-action inference time on Log , Crafting Table , Stone Pickaxe , and Diamond Pickaxe . Optimus-3-w/o-router denotes the variant of Optimus-3 without Layer Router. The 50ms mark corresponds to Minecraft’s real-time interaction requirement, i.e., 20Hz.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

baseline agents [9]–[12] rely on additional external tools or task-specific modules to bridge these two stages. The results in Rows 1-4 of Table 3 show that generic MLLMs struggle to produce effective long-horizon plans, which we attribute to the absence of domain-specific fine-tuning in Minecraft. In contrast, Qwen2.5-VL-SFT is fine-tuned on our proposed OptimusM4 dataset, its planning capability improves substantially. It highlights the critical role of OptimusM4 in injecting Minecraft-specific knowledge and structured long-horizon planning skills. Moreover, compared with the variant (Optimus-3-Action) trained solely on action-trajectory data, Optimus-3 trained on a diverse set of task types achieves a substantially higher success rate. We attribute this improvement to the richer domain knowledge contained in the different task families, which enhances the model’s representations. While shared knowledge expert in Optimus-3 enables such knowledge transfer across tasks.

open-ended scenarios, we construct a suite of five openended tasks corresponding to the Wooden, Stone, Iron, Gold, and Diamond tech levels. For each episode, the agent is randomly initialized at a different location, and endowed with a set of initial resources sampled from a predefined resource pool. This setup requires the agent to leverage its multi-dimensional abilities, e.g., perception, planning, and reflection, to make adaptive decisions conditioned on the current state. For each task, we perform 20 rollouts and employ the average Success Rate (SR) and average Completion Rate (CR) as evaluation metrics.

CR quantifies the agent’s progress toward solving an open-ended task by measuring how many of the following stages it successfully completes in order: (1) Captioning, (2) Grounding, (3) Planning, (4) Action, and (5) Embodied QA.

###### 4.4 Evaluation on Open-ended Tasks

Evaluation Tasks & Metrics. To evaluate the performance of Optimus-3 to jointly deploy its diverse capabilities in

Formally, let K denote the total number of stages and let ki be the number of stages completed in the i-th rollout. Given

Coarse-grained reward Dependency-Aware Synthesis Reward Hallucination-Aware Consistency Reward

Planning

Embodied QA

Grounding

Reflection

0.82

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

0.80

0.66

0.94

0.80

0.64

0.78

0.62

0.92

0.76

0.78

0.60

Accuracy

Accuracy

Score

Score

0.74

0.90

0.58

0.76

0.72

0.56

0.88

0.74

0.70

0.54

0.68

0.52

0.86

0.72

0.50

0.66

0 500 1k 1500

0 500 1k 1500 2k 2500

0 1k 2k 3k

0 200 400 600 800

Steps

Steps

Steps

Steps

- Fig. 10: Performance comparison between coarse-grained reward (GRPO) and our fine-grained rewards (DGRPO). Our DependencyAware Synthesis Reward and Hallucination-Aware Consistency Reward enable stable and consistent performance improvements, whereas using coarse-grained rewards leads to unstable RL training due to the lack of fine-grained supervision.

- TABLE 5: Ablation study of Optimus-3 on training stages. We report accuracy of Planning, Captioning, EQA, and Grounding on each stage.

SFT Coldstart RL Planning Captioning EQA Grounding

0.05 0.47 0.46 0.18 0.79 0.68 0.71 0.52 0.86 0.72 0.73 0.68 0.76 0.71 0.68 0.79 0.94 0.78 0.81 0.80

N rollouts, the CR is defined as:

N

1 N

ki K

, (12)

CR =

i=1

Baselines. For the Open-ended Tasks, we compare Optimus3 against generalist MLLMs (GPT-5 [24], Qwen3-VL [13],

- Gemini-2.5-pro [52]), and current SOTA agents in Minecraft (Jarvis-1 [9], Optimus-1 [11], and Optimus-2 [12]). Results Analysis. As shown in Table 4, Optimus-3 achieves the highest task success rate and completion rate across all open-ended tasks. Notably, on the most challenging Diamond Sword task, Optimus-3 attains a success rate of 35% and a completion rate of 69%, demonstrating a substantial margin over all baselines. In contrast, generic MLLMs achieve consistently low success and completion rates on open-ended tasks, as they often fail to generate adaptive plans conditioned on the specific situation (e.g., the current inventory and surrounding environment). Meanwhile, even strong baselines such as Jarvis-1 [9] and Optimus-1 [11] struggle to generalize to open-ended settings. We attribute this to the fact that they are neither explicitly designed nor trained to support multi-dimensional abilities beyond planning and action execution. It highlights the advantage of Optimus-3: by integrating multi-dimensional capabilities into an end-to-end framework, it can better adapt to open-ended scenarios and solve tasks that require coherent perception– grounding–planning–execution–reflection loops.

###### 4.5 Ablation Study

In this section, we conduct comprehensive ablation studies to validate the effectiveness of our approach and summarize our key findings.

High-quality training data is essential for effective MLLM post-training. Experimental results in Table 2 reveal that both

Optimus-3 and Qwen2.5-VL-SFT benefit substantially from training on OptimusM4 dataset. Furthermore, we conduct an ablation study to investigate the role of knowledge injection in our data-generation pipeline. As shown in Figure 8, removing the knowledge-augmentation mechanism (denoted as Clear) leads to substantial performance degradation, with drops of 81% on Planning, 32% on Embodied QA, and 23% on Grounding, respectively. We attribute this degradation to the fact that generic MLLMs lack Minecraftspecific domain knowledge and thus produce low-quality synthetic data when the knowledge signal is removed. This result highlights the critical importance of our proposed knowledge-enhanced data generation pipeline in providing high-quality supervision for post-training the MLLM backbone. Moreover, the data collection process of OptimusM4 incurred only $300 in API costs, and was completed using 4× NVIDIA L40 GPUs over 36 hours, demonstrating the cost-efficiency of the pipeline.

The Dual-Router Aligned MoE architecture is crucial for heterogeneous multi-task learning. As shown in Table 2, dense models suffer from severe task interference when trained on heterogeneous multi-tasks, leading to suboptimal performance across all task types. Although the Optimus-3 variant with token routing outperforms the dense model on Planning and Embodied QA, it exhibits noticeable degradation on Captioning and Grounding. We attribute this behavior to the intrinsic difficulty of token-level routing, which must simultaneously address load balancing across experts and joint optimization over conflicting task objectives. In contrast, Optimus-3 with task routing achieves the best performance on all tasks. It indicates that explicitly assigning task-specific experts for heterogeneous tasks substantially mitigates task interference.

On the other hand, we conduct an ablation study to evaluate the impact of the Layer Router on latency-sensitive tasks, as illustrated in Figure 9. The results indicate that the Optimus-3 variant without the Layer Router suffers from an average per-action latency exceeding 50ms, thus failing to meet Minecraft’s 20Hz interaction requirement. In contrast, introducing the Layer Router substantially accelerates the inference of Optimus-3 while preserving its high success rate. Remarkably, with the layer router enabled, the Optimus-3 with 6.8B parameters attains an inference speed comparable to Optimus-2 with 1.3B parameters. It highlights the role

###### Input

###### GPT-4o Qwen2.5-VL (SFT) Optimus-3

The scene depicts a peaceful moment in Minecraft where the player is exploring or building near a natural landscape. The presence of the sheep suggests that this area might be suitable for farming or gathering resources like wool. The inventory indicates that the player has been active, collecting materials such as sand, dirt, and grass blocks, as well as crafting tools like a pickaxe and sword.

This Minecraft screenshot depicts a firstperson view during sunset in a plains biome near an ocean, featuring grassy terrain, sparse trees with vines, and a white sheep visible near the crosshair. The player holds a partially used iron pickaxe and has a hotbar containing sand, dirt, oak planks, raw mutton, white wool, apples, and the pickaxe. With full health, full hunger, and experience level 1.

The player is standing on a grassy plain and is holding a stone pickaxe in their right hand. In the foreground, there is a white sheep. In the background, there are several trees and vines hanging down. Beyond the trees, there is a body of water visible on the left side of the image. The player's inventory bar shows various items, including a wooden pickaxe, a crafting table, seeds, raw meat, and a flower.

Describe the current screen in details.

[Figure 143]

CaptioningEQAGroundingPlanning

[Figure 144]

How many pigs are in the scene?

The visible mobs include four pigs in the foreground.

In the scene, there are three pigs visible. There are three pigs in the scene.

[Figure 145]

[Figure 146]

[Figure 147]

Grounding the pig in the scene.

NA

- Step 1: chop trees to get 3 logs
- Step 2: craft 5 planks
- Step 3: craft 3 stick
- Step 4: craft 1 wooden pickaxe
- Step 5: dig down and mine 1 cobblestone
- Step 6: craft 1 stone pickaxe

- Step 1: chop trees to get 3 logs
- Step 2: craft 5 planks
- Step 3: craft 3 stick
- Step 4: craft 1 wooden pickaxe
- Step 5: dig down and mine 2 cobblestone
- Step 6: craft 1 stone pickaxe

Given that you already have 1 crafting table, 1 cobblestone, how to get 1 stone sword?

- Step 1: obtain a stick
- Step 2: craft the stone sword

- Fig. 11: Visual comparison of Optimus-3 (ours), Qwen2.5-VL (tuned on our data), and GPT-4o. Red highlights indicate errors, while blue highlights denote correct outputs.

of the layer router in adapting the effective model depth to the task-dependent reasoning complexity. Taken together, these findings highlight the effectiveness of our proposed Dual-Router Aligned MoE architecture in simultaneously resolving conflicts among heterogeneous tasks and balancing inference complexity with real-time latency constraints.

Dual-Granularity Reasoning-Aware Policy Optimization further enhances the agent’s capabilities. As shown in Table

- 5, removing the DGRPO method (both the coldstart and RL stages) leads to performance drops of 16%, 13%, 12%, and 35% on Planning, Captioning, EQA, and Grounding, respectively. These results highlight the pivotal role of DGRPO in enhancing the performance of Optimus-3 in dynamic and diverse scenarios. Furthermore, we conduct experiments to investigate the importance of our proposed fine-grained reward design. As shown in Figure 10, compared with using only the final answer correctness as a coarse feedback signal, our Dependency-Aware Synthesis Reward and Hallucination-Aware Consistency Reward yield consistently better performance on Planning, Embodied QA, Grounding, and Reflection. A critical ablation on the Embodied QA reveals that the model fails to converge when restricted to standard answer-level supervision. It shows the limitations of outcome-based RL in complex visual reasoning: the absence of process supervision renders the model vulnerable to visual hallucinations. In contrast, our DGRPO mitigates this issue by imposing strict penalties on non-existent entities generated within the reasoning chain.

###### 4.6 Qualitative Analysis

In this section, we present qualitative visualizations to illustrate the behavior of Optimus-3 across different task types. As depicted in Figure 11, we provide a visual comparison between Optimus-3, Qwen2.5-VL-7B [13], and GPT-

- 4o [24], highlighting their differences in performance and behavior across non-action tasks. We observe that GPT-4o exhibits hallucinations in captioning and embodied QA, lacks grounding capabilities, and produces unreasonable plans. In contrast, Qwen2.5-VL-SFT, which is fine-tuned on our OptimusM4 dataset, shows reduced hallucination, acquires grounding and planning abilities, and generates more coherent outputs. Notably, Optimus-3 accurately performs visionrelated tasks and produces well-structured plans conditioned on instructions, demonstrating its superior perception and reasoning in the Minecraft environment.

Moreover, as shown in Figure 12, we compare Optimus3 with Jarvis-1 [9] on the open-ended instruction, Craft a diamond sword based on the current inventory. We observe that Jarvis-1 fails to accurately capture the fine-grained item information in the inventory (hotbar), which leads to planning trajectories containing unnecessary steps (e.g., crafting a Wooden Pickaxe and a Stone Pickaxe ). Moreover, the semantic gap between its planning and action modules causes it to eventually produce an incorrect target item. In contrast, Optimus-3 precisely understands the current context and generates an appropriate plan, successfully crafting the Diamond Sword step-by-step. We attribute this performance to Optimus-3’s integration of System 1 action loops and System 2 reasoning capabilities within a unified framework, which enables it adapts more effectively to the diverse situations encountered in Minecraft.

[Figure 148]

[Figure 149]

[Figure 150]

- 5 CONCLUSION

In this paper, we presented Optimus-3, a unified generalist agent that organically integrates System 1 action loops with System 2 reasoning capabilities within an end-to-end framework. To overcome the challenges of data scarcity,

Long-horizon Task: Craft a Diamond Sword based on the current inventory.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

perception chop mine crafting crafting mine

Jarvis-1

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

crafting mine crafting smelt crafting failure

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

perception grounding crafting crafting mine mine

Optimus-3

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

mine mine smelt crafting crafting success

- Fig. 12: Visualization of Optimus-3 and Jarvis-1 executing the open-ended task “Craft a Diamond Sword based on the current inventory”. Correct actions are highlighted in green , unnecessary actions in yellow , and erroneous actions in red .

architectural conflict, and open-world generalization, we contributed advances along three dimensions. First, we introduced a Knowledge-Enhanced Data Generation Pipeline that samples high-fidelity System 2 reasoning traces from raw interaction trajectories. By leveraging domain constraints to filter hallucinations, we constructed and released the OptimusM4 dataset. Second, we proposed the Dual-Router Aligned MoE architecture to address the computational conflict between the two systems. Through horizontal parameter decoupling and vertical depth adaption, it efficiently maintains a “Fast Path” for reflexive control and a “Deep Path” for deliberative reasoning. Third, we developed the Dual-Granularity Reasoning-Aware Policy Optimization (DGRPO) algorithm. It establishes a Process-Outcome CoSupervision mechanism, utilizing dual-granularity rewards to align reasoning chains with visual evidence. Extensive experiments demonstrate that Optimus-3 achieves superior performance across diverse tasks, marking a significant step toward achieving general-purpose embodied intelligence in complex, open-ended worlds.

- [6] L. Fan, G. Wang, Y. Jiang, A. Mandlekar, Y. Yang, H. Zhu, A. Tang, D.-A. Huang, Y. Zhu, and A. Anandkumar, “Minedojo: Building open-ended embodied agents with internet-scale knowledge,” Advances in Neural Information Processing Systems, vol. 35, pp. 18343– 18362, 2022. 1, 4, 10
- [7] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar, “Voyager: An open-ended embodied agent with large language models,” arXiv preprint arXiv:2305.16291, 2023. 1, 4, 11
- [8] Y. Qin, E. Zhou, Q. Liu, Z. Yin, L. Sheng, R. Zhang, Y. Qiao, and J. Shao, “Mp5: A multi-modal open-ended embodied system in minecraft via active perception,” arXiv preprint arXiv:2312.07472,

2023. 1, 4, 9, 10, 11

- [9] Z. Wang, S. Cai, A. Liu, Y. Jin, J. Hou, B. Zhang, H. Lin, Z. He, Z. Zheng, Y. Yang et al., “Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models,” arXiv preprint arXiv:2311.05997, 2023. 1, 4, 9, 10, 11, 12, 13
- [10] Z. Wang, S. Cai, G. Chen, A. Liu, X. Ma, and Y. Liang, “Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents,” arXiv preprint arXiv:2302.01560, 2023. 1, 4, 9, 10, 11
- [11] Z. Li, Y. Xie, R. Shao, G. Chen, D. Jiang, and L. Nie, “Optimus-1: Hybrid multimodal memory empowered agents excel in longhorizon tasks,” arXiv preprint arXiv:2408.03615, 2024. 1, 4, 5, 9, 10, 11, 12
- [12] Z. Li, Y. Xie, R. Shao, G. Chen, and L. Nie, “Optimus-2: Multimodal minecraft agent with goal-observation-action conditioned policy,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9039–9049. 1, 4, 9, 10, 11, 12
- [13] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang et al., “Qwen2. 5-vl technical report,” arXiv preprint arXiv:2502.13923, 2025. 1, 2, 3, 9, 10, 11, 12, 13
- [14] G. Chen, L. Shen, R. Shao, X. Deng, and L. Nie, “Lion: Empowering multimodal large language model with dual-level visual knowledge,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 26540–26550. 1, 2
- [15] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in neural information processing systems, vol. 36, 2024. 1, 2
- [16] D. Kahneman, Thinking, fast and slow. macmillan, 2011. 1, 7
- [17] B. Baker, I. Akkaya, P. Zhokov, J. Huizinga, J. Tang, A. Ecoffet, B. Houghton, R. Sampedro, and J. Clune, “Video pretraining (vpt): Learning to act by watching unlabeled online videos,” Advances in Neural Information Processing Systems, vol. 35, pp. 24639–24654,

2022. 2, 4, 9, 10

- [18] S. Lifshitz, K. Paster, H. Chan, J. Ba, and S. McIlraith, “Steve-1: A generative model for text-to-behavior in minecraft,” Advances in Neural Information Processing Systems, 2023. 2, 4, 5, 9, 10
- [19] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, Y. Sun et al., “Deepseek-vl: towards real-world vision-

#### REFERENCES

- [1] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse domains through world models,” arXiv preprint arXiv:2301.04104,

2023. 1

- [2] W. Tan, Z. Ding, W. Zhang, B. Li, B. Zhou, J. Yue, H. Xia, J. Jiang, L. Zheng, X. Xu et al., “Towards general computer control: A multimodal agent for red dead redemption ii as a case study,” arXiv preprint arXiv:2403.03186, 2024. 1
- [3] M. A. Raad, A. Ahuja, C. Barros, F. Besse, A. Bolt, A. Bolton, B. Brownfield, G. Buttimore, M. Cant, S. Chakera et al., “Scaling instructable agents across many simulated worlds,” arXiv preprint arXiv:2404.10179, 2024. 1
- [4] S. Yang, J. Walker, J. Parker-Holder, Y. Du, J. Bruce, A. Barreto, P. Abbeel, and D. Schuurmans, “Video as the new language for real-world decision making,” arXiv preprint arXiv:2402.17139, 2024. 1
- [5] W. H. Guss, B. Houghton, N. Topin, P. Wang, C. Codel, M. Veloso, and R. Salakhutdinov, “Minerl: A large-scale dataset of minecraft demonstrations,” arXiv preprint arXiv:1907.13440, 2019. 1, 9

- language understanding,” arXiv preprint arXiv:2403.05525, 2024. 2, 5, 10
- [20] A. Vats, R. Raja, V. Jain, and A. Chadha, “The evolution of mixture of experts: A survey from basics to breakthroughs,” Preprints, 2024. 2, 4, 5
- [21] Y. Huang, Z. Wang, Z. Yuan, Y. Ding, R. Gong, J. Guo, X. Liu, and J. Zhang, “Modes: Accelerating mixture-of-experts multimodal large language models via dynamic expert skipping,” arXiv preprint arXiv:2511.15690, 2025. 2, 4, 5
- [22] L. Shen, G. Chen, R. Shao, W. Guan, and L. Nie, “Mome: Mixture of multimodal experts for generalist multimodal large language models,” Advances in neural information processing systems, vol. 37, pp. 42048–42070, 2025. 2, 4, 7
- [23] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu et al., “Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models, 2024,” URL https://arxiv. org/abs/2401.06066, 2024. 2, 4
- [24] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023. 3, 5, 9, 10, 12, 13
- [25] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017. 2, 4
- [26] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53728–53741, 2023. 2, 4
- [27] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025. 2, 4, 7
- [28] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024. 2, 4, 7
- [29] S. Cai, B. Zhang, Z. Wang, X. Ma, A. Liu, and Y. Liang, “Groot: Learning to follow instructions by watching gameplay videos,” in The Twelfth International Conference on Learning Representations, 2023. 4, 10
- [30] J. Deng, Z. Wang, S. Cai, A. Liu, and Y. Liang, “Open-world skill discovery from unsegmented demonstrations,” arXiv preprint arXiv:2503.10684, 2025. 4
- [31] S. Liu, H. Yuan, M. Hu, Y. Li, Y. Chen, S. Liu, Z. Lu, and J. Jia, “Rlgpt: Integrating reinforcement learning and code-as-policy,” arXiv preprint arXiv:2402.19299, 2024. 4
- [32] S. Liu, Y. Li, K. Zhang, Z. Cui, W. Fang, Y. Zheng, T. Zheng, and M. Song, “Odyssey: Empowering agents with open-world skills,” arXiv preprint arXiv:2407.15325, 2024. 4
- [33] H. Li, X. Yang, Z. Wang, X. Zhu, J. Zhou, Y. Qiao, X. Wang, H. Li, L. Lu, and J. Dai, “Auto mc-reward: Automated dense reward design with large language models for minecraft,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 16426–16435. 4
- [34] Z. Li, X. Xu, Z. Xu, S. Lim, and H. Zhao, “Larm: Large autoregressive model for long-horizon embodied intelligence,” arXiv preprint arXiv:2405.17424, 2024. 4
- [35] S. Zhou, T. Zhou, Y. Yang, G. Long, D. Ye, J. Jiang, and C. Zhang, “Wall-e: World alignment by rule learning improves world modelbased llm agents,” arXiv preprint arXiv:2410.07484, 2024. 4
- [36] Z. Wang, S. Cai, Z. Mu, H. Lin, C. Zhang, X. Liu, Q. Li, A. Liu, X. Ma, and Y. Liang, “Omnijarvis: Unified vision-language-action tokenization enables open-world instruction following agents,” arXiv preprint arXiv:2407.00114, 2024. 4
- [37] S. Cai, Z. Wang, K. Lian, Z. Mu, X. Ma, A. Liu, and Y. Liang, “Rocket1: Mastering open-world interaction with visual-temporal context prompting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 12122–12131. 4
- [38] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European Conference on Computer Vision. Springer, 2024, pp. 38–55. 5
- [39] S. Mu and S. Lin, “A comprehensive survey of mixture-ofexperts: Algorithms, theory, and applications,” arXiv preprint arXiv:2503.07137, 2025. 4

- [40] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, “Gshard: Scaling giant models with conditional computation and automatic sharding,” arXiv preprint arXiv:2006.16668, 2020. 4
- [41] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022. 4
- [42] N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat et al., “Glam: Efficient scaling of language models with mixture-of-experts,” in International conference on machine learning. PMLR, 2022, pp. 5547–5569. 4
- [43] Y. Li, S. Jiang, B. Hu, L. Wang, W. Zhong, W. Luo, L. Ma, and M. Zhang, “Uni-moe: Scaling unified multimodal llms with mixture of experts,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 4
- [44] X. V. Lin, A. Shrivastava, L. Luo, S. Iyer, M. Lewis, G. Ghosh, L. Zettlemoyer, and A. Aghajanyan, “Moma: Efficient early-fusion pre-training with mixture of modality-aware experts,” arXiv preprint arXiv:2407.21770, 2024. 4
- [45] J. Wu, X. Hu, Y. Wang, B. Pang, and R. Soricut, “Omni-smola: Boosting generalist multimodal models with soft mixture of lowrank experts,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14205–14215. 4
- [46] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” Nature, vol. 518, pp. 529–533,

2015. 4

- [47] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” arXiv preprint arXiv:1509.02971, 2015. 4
- [48] J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz, “Trust region policy optimization,” in International conference on machine learning. PMLR, 2015, pp. 1889–1897. 4
- [49] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” Advances in neural information processing systems, vol. 30, 2017. 4
- [50] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27730–27744, 2022. 4
- [51] X. Dong, P. Zhang, Y. Zang, Y. Cao, B. Wang, L. Ouyang, X. Wei, S. Zhang, H. Duan, M. Cao et al., “Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in visionlanguage large model,” arXiv preprint arXiv:2401.16420, 2024. 5
- [52] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang et al., “Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context,”

- arXiv preprint arXiv:2403.05530, 2024. 5, 9, 10, 12

[53] Z. Bai, P. Wang, T. Xiao, T. He, Z. Han, Z. Zhang, and M. Z. Shou, “Hallucination of multimodal large language models: A survey,”

- arXiv preprint arXiv:2404.18930, 2024. 5, 8

- [54] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, C. Li, J. Yang, H. Su, J. Zhu et al., “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” arXiv preprint arXiv:2303.05499, 2023. 5
- [55] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24824–24837, 2022. 7
- [56] A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, and K. Saenko, “Object hallucination in image captioning,” arXiv preprint arXiv:1809.02156, 2018. 8
- [57] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” ICLR, 2021. 9
- [58] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” arXiv preprint arXiv:1908.10084, 2019. 9
- [59] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in

- neural information processing systems, vol. 35, pp. 27730–27744, 2022. 10
- [60] D. Li, B. Jiang, L. Huang, A. Beigi, C. Zhao, Z. Tan, A. Bhattacharjee, Y. Jiang, C. Chen, T. Wu et al., “From generation to judgment: Opportunities and challenges of llm-as-a-judge,” arXiv preprint arXiv:2411.16594, 2024. 9
- [61] Z. Chen, R. Zhang, Y. Song, X. Wan, and G. Li, “Advancing visual grounding with scene knowledge: Benchmark and method,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 15039–15049. 9
- [62] A. Singh, A. Fry, A. Perelman, A. Tart, A. Ganesh, A. El-Kishky, A. McLaughlin, A. Low, A. Ostrow, A. Ananthram et al., “Openai gpt-5 system card,” arXiv preprint arXiv:2601.03267, 2025. 11
- [63] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva,

I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025. 11

