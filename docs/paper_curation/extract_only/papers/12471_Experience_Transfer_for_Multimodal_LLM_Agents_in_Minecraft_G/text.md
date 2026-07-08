## Experience Transfer for Multimodal LLM Agents in Minecraft Game

# arXiv:2604.05533v1[cs.AI]7Apr2026

Chenghao Li1, Jun Liu1, Songbo Zhang1, Huadong Jian1, Hao Ni2, Lik-Hang Lee3, Sung-Ho Bae4, Guoqing Wang1, Yang Yang1, Chaoning Zhang1∗

[Figure 1]

“Although tools, weapons, and armor differ in materials (wood, stone, iron, diamond), the patterns in their crafting materials are consistent. Therefore, one can apply transfer learning based on the successfully crafted recipes, without needing to relearn the crafting recipe for each individual item.”

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

“The wooden sword deals 4 damage, the stone sword deals 5 damage, and the iron sword deals 6 damage. Is the diamond sword even stronger? Assuming that the better material, the greater damage.”

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

###### +

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

+ ? ? ? ? ? ? ? ? ? ? ? ?

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

+ =

? ? ?

[Figure 60]

[Figure 61]

“A gray, rock-like surface with a stony texture, embedded with particles of various colors. Can these also be smelted to produce new materials?”

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

+

[Figure 77]

[Figure 78]

[Figure 79]

More harm?

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

craft mine

smelt

[Figure 94]

[Figure 95]

[Figure 96]

- 4 damage mine smelt

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- 5 damage

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

- 6 damage

“Given the process “stone pickaxe → mine iron → smelt → iron pickaxe,” we can analogically infer “iron pickaxe → mine copper → smelt → copper pickaxe.” In other words, the task structure remains the same, only the materials differ. This demonstrates how one can infer an unseen process from a remembered task sequence and engage in reflective practice.”

Reflection

Process

Reuse

[Figure 110]

Chin

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

craft

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

“The surface is composed of horizontal wooden planks, with subtle gaps between each plank to create a sense of layered assembly. The differences lie in their colors: dark olive brown, pinkish brown, light taupe, pale khaki, reddish chestnut, ash white, dark mahogany, and dark umber. Same look, same function.”

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

“Memory: The central area is centered around a well or bell tower, surrounded by villagers’ houses. Farmland is usually located on the outskirts of the

“Observe: In front of me is the village’s central well. To

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

gather resourc es, I should..

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

village or near water sources, growing crops such as wheat, carrots, potatoes, or beetroot. Some buildings, such as the grapher’s house or the library, may also contain resources like books and paper.”

“Each type of plank can be used individually to build a boat.”

.”

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Figure 1. Conceptual illustration of Echo. The agent learns from experience and discovers transferable patterns, enabling interpretable analogy-based reasoning and cross-task generalization. In some classical methods, such as DEPS [44] and JARVIS-1 [45], ICL is mainly used to retrieve few-shots from the memory bank to assist in generating sub-task sequences for the current goal. Echo, on the other hand, treats ICL learning as an active process — it proactively retrieves potentially new tasks from the memory bank for validation and execution.

### Abstract

Multimodal LLM agents operating in complex game environments must continually reuse past experience to solve new tasks efficiently. In this work, we propose Echo, a transfer-oriented memory framework that enables agents to derive actionable knowledge from prior interactions rather than treating memory as a passive repository of static records. To make transfer explicit, Echo decomposes reusable knowledge into five dimensions: structure, attribute, process, function, and interaction. This formulation allows the agent to identify recurring patterns shared across

∗ Corresponding author. 1University of Electronic Science and Technology of China; 2KAIST; 3The Hong Kong Polytechnic University; 4Kyung Hee University. Project: https://github.com/CatworldLee/Echo

different tasks and infer what prior experience remains applicable in new situations. Building on this formulation, Echo leverages In-Context Analogy Learning (ICAL) to retrieve relevant experiences and adapt them to unseen tasks through contextual examples. Experiments in Minecraft show that, under a from-scratch learning setting, Echo achieves a 1.3×–1.7× speed-up on object-unlocking tasks. Moreover, Echo exhibits a burst-like chain-unlocking phenomenon, rapidly unlocking multiple similar items within a short time interval after acquiring transferable experience. These results suggest that experience transfer is a promising direction for improving the efficiency and adaptability of multimodal LLM agents in complex interactive environments.

### 1. Introduction

Driven by embodied intelligence and complex interactive tasks, planning-oriented multimodal large language model (MLLM) agents have rapidly emerged [36, 38, 42, 44, 45, 51, 58]. Representative works such as Voyager [42] and JARVIS-1 [45] in environments like Minecraft demonstrate open-ended exploration driven by a “perception-reasoningaction-memory” loop, where agents continuously selfimprove through environmental feedback. These systems reveal the potential to decompose goals, plan subtasks, and invoke tools without large-scale task-specific supervision. Such agents [42, 45] typically integrate chain-of-thought reasoning from language models with environment interaction trajectories to generate executable strategies, codes, and action sequences—paving the way for agents that move from “speaking” to “doing.”

Alongside this trend, the demand for task memory and reasoning-based planning becomes particularly prominent. Long-term persistent memory enables an agent to reuse skills and support reasoning about the current task in complex scenarios [37, 42, 45]. Cross-modal perception enables fine-grained scene understanding [34]. Moreover, explainable reasoning–action coupling allows for hierarchical planning [1] and self-verification [48] under uncertainty. Existing research has shown that explicit memory structures—such as spatiotemporal event indices [36], multimodal knowledge graphs [23], and structured graph memories [16, 32]—together with retrieval augmentation (extracting examples from historical trajectories, rules, and recipes) can significantly enhance retrieval efficiency, environmental understanding, and long-horizon planning stability, providing support for multi-step reasoning and strategy generation [42, 45].

However, despite the promising outlook, most existing methods still treat “transfer” only superficially. Memory is often regarded as a passive warehouse [7, 25, 42, 45, 61], an index of past behaviors, or a library of reusable skills [42, 45], while the deeper structures that enable experience to be truly transferable remain underexplored. Experience transfer lies at the heart of embodied intelligence [13, 42]. Once an agent has accumulated sufficient experience and memory, it should be able to infer new knowledge from past experiences—many patterns repeat themselves in subtle, structural ways. Similar landscapes hide similar logic: the same crafting patterns, the same material hierarchies, the same causal chains linking raw resources to useful tools. In worlds like Minecraft, the “shape prototypes” of tools and armor, the “substitutability” among material families, the common processing chains (gathering → smelting → crafting), and even the functional symmetry between weapons all reveal recurring motifs of transfer. If such transfer axes-shape, material, process, and function—can be explicitly represented and aligned with multimodal embed-

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Tech Line

Rapid Unlocking

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

1.3×Faster

[Figure 196]

1.7×Faster

[Figure 197]

[Figure 198]

Figure 2. Comparison of item unlocking progress across different agents. The x-axis represents the iteration steps, and the y-axis indicates the number of unique items unlocked. Our method shows a significantly faster progression, exhibiting a “rapid unlocking” phenomenon in the mid-stage, where similar items are unlocked in an explosive manner. Compared to previous methods (MP5 [38], Voyager [42], JARVIS-1 [45], and MrSteve [36]), our approach achieves equivalent milestones about 1.3×–1.7× faster.

dings, an agent would no longer need to relearn what it already knows. Instead, it could rapidly adapt, recombine, and repurpose its existing knowledge to thrive in unfamiliar worlds.

To bridge this gap, we introduce a memory-transferaugmented MLLM planning agent Echo (Figure 1), designed to learn “how to transfer and when to transfer memory”. The agent decomposes environmental and experiential knowledge into five explicit transfer dimensionsStructural, Attribute, Procedural, Functional, and Interaction—forming a unified description of “what the world is, how it evolves, what can be done, and how to act.” Within this representation, a Contextual State Descriptor (CSD) encodes visual, textual, and interactive signals into compact, comparable semantic snapshots, aligning multimodal information across the transfer axes. Finally, a structured incontext analogical learning (ICAL) [19] module allows the agent to retrieve, adapt, and build upon past experiences, enabling interpretable reasoning, dynamic self-verification, and knowledge reuse across worlds and tasks.

As shown in the Figure 2, after accumulating a certain amount of knowledge during the cold-start phase, our method exhibits a significant increase in learning ability during the mid-to-late phase (item unlocking), resulting in an “explosive” item unlocking phenomenon. In addition, we conducted several experiments to evaluate cross-world adaptation, task generalization, and continuous learning efficiency in complex interactive tasks. The results show that our method, by combining explicit transfer axes, ICAL, self-consistency checking, and memory replay, significantly improves the model’s learning efficiency, task success rates, and interpretability, particularly in long-term learning and cross-task transfer scenarios.

Our main contributions are as follows:

- • To the best of our knowledge, we are the first to explore how to explicitly transfer multimodal memory experiences in multimodal embodied agents.
- • To address the challenges of structuring and transferring multimodal memory, we propose five explicit transfer dimensions and combine them with ICAL to effectively organize and leverage multimodal memory.
- • To validate the effectiveness of our approach, we compared it with four agents [36, 38, 42, 45] with different focuses. In learning from scratch tasks, our agent demonstrated clear advantages in learning ability.

### 2. Related Work

Embodied Agents in Minecraft. Open-world agents learning in complex environments [4, 6, 9, 12, 14, 15, 17, 24, 29, 40–42, 50–53, 55–57, 59]. MineDojo [14] and Voyager [42] leveraged internet-scale knowledge and large models to achieve open-ended exploration. Some works enhanced the agents’ active perception capabilities [38, 55, 58]. The Optimus series [25, 26] and Jarvis series [45, 46] adopted modular or hierarchical architectures to enable skill reuse. GITM [7] and Odyssey [30] combined large-scale pretraining with skill libraries to achieve exploratory transfer. In complex interactive environments (e.g., Minecraft), agents need to possess cross-modal perception and longterm memory capabilities to associate contextual information and reuse past experiences. For example, MrSteve [36] proposed a What-Where-When memory model that supports event retrospection based on temporal–spatial indexing; VistaWise [16] constructed a cross-modal knowledge graph to enable efficient knowledge storage and semantic association; and the GROOT series [7, 8] employed graph neural networks to build structured memory.

In-Context Learning. In-Context Learning [11, 19] was first systematically introduced in GPT-3 [5], referring to the ability to perform new tasks without parameter updates by providing a few demonstrations within the input context. ICL exhibits strong few-shot generalization capabilities. Existing research mainly focuses on three directions: (1) Prompt design and example selection: improving ICL accuracy and robustness [18, 22, 27, 39, 43, 54, 60]; (2) Retrieval-augmented ICL: combining external knowledge bases to dynamically retrieve demonstrations and mitigate context-length limitations [3, 10, 21, 33]; (3) Structured ICL: explicitly modeling structural patterns-e.g., tables, logical chains, or code—to improve transfer [28, 31, 47]. In multimodal domains, models such as CoCa [49] and Flamingo [2] can rapidly adapt to visual question answering and image understanding with few image–text examples. In interactive environments, ICL facilitates policy and skill transfer; e.g., ReAct [48] integrates reasoning traces and action trajectories for context-aware planning.

###### (a) (b)

Read

FuncInterStructAttrProc

different state transitions, different causal elations

###### structure the transfer

AgentInteract

Interact

###### hard to transfer

ContextualStateDescriptor

ComplexRealWorld

ComplexRealWorld

understand,plan,reason

understand,plan,reason

MLLMs

Agent

Observe

MLLMs

Observe

Control

Control

stabilize generation

Instruction Tuning

MLLM hallucination

unstable control

Write

Figure 3. Overview of motivation and Problem Framework. (a) Traditional MLLM-based agents struggle to generalize across complex real-world environments due to different state transitions and causal relations (hard to transfer) and may exhibit unstable control arising from hallucinations. (b) The proposed Structured In-Context Learning framework introduces a unified CSD that decomposes environmental knowledge into five explicit transfer dimensions.

### 3. Proposed Method

Motivation and Challenges. In open, interactive, and multimodal environments, our goal is to use retrieval proactively to achieve rapid, stable, and interpretable generalization to new tasks, rather than merely serving current objectives reactively [42, 45]. At present, we face two major challenges. First, the real world exhibits highly complex regularities and structural dependencies. Distinct tasks often involve significantly different state transitions and causal relationships, making effective transfer and generalization inherently difficult. Second, MLLMs tend to suffer from hallucinations and uncontrolled reasoning in open-ended scenarios [20], undermining both the stability and verifiability of their generalization performance.

As illustrated in Figure 3, these challenges can lead to unstable control and limited transferability in traditional MLLM-based agents when deployed in complex real-world environments. To address these issues, we propose a Structured ICL framework [19] based on explicit transfer dimensions (§3.1). This framework decomposes environmental representations and historical experiences into multiple dimensions within the multimodal space, explicitly modeling task transfer relationships along five axes: Structural, Attribute, Procedural, Functional, and Interaction. By modeling these dimensions explicitly, the model can semantically interpret the correspondence and similarity between tasks, thereby enabling interpretable cross-task alignment and analogy-based reasoning.

The entire process is grounded in a unified CSD (§3.2) that integrates heterogeneous environmental and experiential information. Building upon this representation, an instruction-tuned MLLM performs structured reasoning and planning, achieving efficient and stable generalization to new tasks without parameter updates.

#### 3.1. Explicit Transfer Dimensions

This section anchors the five “learning-by-analogy” dimensions within a unified, MLLM-driven framework for explicit knowledge transfer. In this formulation, a single MLLM serves as the central mechanism to represent, align, and evaluate cross-modal correspondences along five interpretable axes: structural, attribute, procedural, functional, and interaction.

These five dimensions are not arbitrary-they form a holistic grammar of understanding for any agent operating in open-world environments. From a theoretical standpoint, an embodied agent seeking to transfer or reconstruct knowledge must simultaneously grasp three fundamental questions:

- • What the world is like—captured through its structures and attributes;
- • How the world changes—revealed through its procedures and functions;
- • How the agent engages with the world—embodied in its interactions.

These aspects are deeply interdependent and hierarchically organized. The structural and attribute dimensions ground understanding in the physical and spatial regularities of the environment, providing the static scaffolding upon which transfer can occur. The procedural and functional dimensions capture the world’s dynamics—its causes, effects, and transformations—enabling the reuse and adaptation of strategies across contexts. Finally, the interaction dimension closes the loop, modeling the continuous feedback between perception and action that allows an agent to learn not only from observation but from participation. Although our approach appears similar to MrSteve’s “What-WhereWhen Memory,” [36] MrSteve primarily aims to introduce episodic memory into agents, whereas our method focuses on leveraging structured memory to enable task transfer.

- • Structural Axis—“How the world is organized.” Focuses on spatial layout and hierarchical relationships—how entities are arranged and connected. It provides the geometric framework of the environment, helping the agent understand spatial structure, reachability, and coherence during transfer.
- • Attribute Axis—“What physical properties things have.” Describes the visual and physical traits of objects, such

- as color, texture, hardness, and material composition. It enables reasoning about substitution, support, and compatibility in different contexts.

###### Meta Data (JSON)

###### Structural Data (JSON)

###### Functional Data (JSON)

{ Description: "There is a furnace one block in front of the Agent. Iron ore is

{ Agent Position:[12,64,-3], Facing Direction:West, Time Step:10234, World Time:Day, Biome:Plains, Current Task:Smelting Iron, Objects:[Furnace,Ore,...] }

{ "furnace": "A processing device used to convert ore and fuel into finished products.", "iron_ore": ... "charcoal": ... "iron_ingot":

placed in the top slot, charcoal in the bottom slot, and an iron ingot in the right output slot." }

... }

###### Attribute Data (JSON)

###### Procedural Data (JSON)

###### Interaction Data (JSON)

{ Description: "Iron ore occupies the top slot of the furnace and charcoal the bottom slot as fuel. When ignited, the furnace smelts the ore, producing iron ingots that ..." }

{ Interaction: "The Agent walks up to the furnace and opens its interface..." History: [ "The Agent collected iron ore and charcoal.", "The Agent ...] }

{ "furnace": " color": "gray", " shape": "cube", " texture": "rough stone surface"

"iron_ore": ... "charcoal": ... "iron_ingot": ... }

Figure 4. Overview of the CSD schema.

- • Procedural Axis—“How the world changes.” Captures the causal rules and state transitions that define how actions alter the environment. It models sequences and dependencies—clarifying what to do, when, and why.
- • Functional Axis—“What things do.” Describes the purpose and role of objects—what they can do and how they contribute to tasks. It supports semantic-level generalization and creative reuse across domains.
- • Interaction Axis—“How the agent interacts with the world.” Characterizes the perception–action loop, showing how operations lead to feedback and environmental response. It connects knowledge to execution, ensuring actions are both understandable and performable.

To enable the agent to better transfer knowledge across the five dimensions, we design a standardized CSD-ICAL paradigm (§ 3.2).

#### 3.2. Contextual State Descriptor

This section introduces the overall design goals, data schema, generation pipeline, and system contracts of the CSD. The core idea of CSD is to compress heterogeneous multimodal inputs—visual, textual, and interactive—into a unified, comparable, and verifiable semantic snapshot. This unified representation enables Echo to perform stable retrieval and reasoning within a structured semantic space. Specifically, the CSD organizes multimodal content along explicit transfer dimensions, including structural, attribute, procedural, functional, and interaction axes. By aligning these axes through vectorized encoding and quality evaluation, the CSD serves as a reliable foundation for interpretable cross-task generalization and anomaly detection.

Design Goals and Constraints. A CSD consists of six core components: metadata and five semantic dimensions. The meta field records the generation timestamp, source environment, and model versions. The five fields—struct, attr, proc, func, and inter—correspond to the five transfer dimensions, as illustrated in Fig. 4, each containing symbolic content as well as global embeddings for fast vector-based retrieval. In addition, during training, we apply instruction

Iteratively In Context Analogical Learning

Task Selection-Most successful tasks/ Recently learned tasks <Meta>;<Attr>;<Struct>;<Func>;<Proc>;<Inter>

Retrieval

Longterm Memory Bank

Key-Value Query

NextTransfer

Learning Examples-Compute semantic similarity <Meta>;<Attr>;<Struct>;<Func>;<Proc>;<Inter>

Dimension

Planner

Transferred Knowledge-Action seqs & Auxiliary context

Figure 5. ICL-based analogical learning workflow using the CSD memory bank.

fine-tuning to enable the MLLM to produce well-formatted CSD structures more reliably. Throughout this process, the model learns from large numbers of structured task examples to align task descriptions with evidence across the five semantic axes, generating normalized outputs that follow a unified specification. The training data consists of multimodal task instructions, historical execution traces, and verifier feedback, enabling the model to develop basic capabilities in comprehension, structured organization, and consistency assurance when generating CSD.

ICL-Based Analogical Learning. As illustrated in Fig. 5, CSD entries are aligned with the Minecraft task workflow: only successful tasks are written to long-term memory. The CSD library is periodically maintained offline through consolidation, cleaning, deduplication, and clustering. Clustered CSD supports knowledge inference and pattern abstraction (e.g., extending “smelting iron ore → iron ingot” to “smelting gold ore → gold ingot” or deriving new crafting routes), enabling autonomous knowledge expansion. Based on this representation, we build an ICL-based analogical workflow. (1) Task Selection: choose a representative task (e.g., most successful or most recently learned) and extract its complete CSD; (2) Example Retrieval: retrieve the top-K most relevant tasks by computing multidimensional semantic similarity across five CSD components (attr, struct, func, proc, Inter); (3) ICL Context Construction: combine these samples to form the ICL input context; (4) New Task Induction: the model generalizes from the context and outputs potential new tasks only as action sequences; (5) Execution & Validation: the actions are executed and evaluated; successful trajectories are stored, while failures are logged. This workflow enables continuous experience accumulation, knowledge transfer, and autonomous task discovery.

#### 3.3. Overall Iterative Process

The system follows the classical agent model [42, 45] and achieves efficient use of internal knowledge for open-world task transfer by combining ICAL with explicit transfer axes, thereby improving the agent’s performance and capabilities in the open world, as shown in Figure 6.

Perception Layer

Memory Layer

Environment State

e.g. <position>, <health>, <inventory>, <task status>

Transfer

Long-termMemoryBank e.g.facts, experiences,failures,CSD

Visual Input

e.g. <camera>, <GUI>

MLLM

Visual Caption

meta,struct,attr,proc,func,inter

ContextualStateDescriptor

scene caption, object list, spatial relations

Decision Layer

Prompt Builder

instructions, environment data

Planner (MLLM)

plan and command sequence

Pre-checker

AffectsGameWorld

e.g. resource / position

Execution Layer

e.g.current objective,constraints

Action

Short-term Memory

Result Observation

execution results, position, inventory, health, screenshots

No Successful？ Yes

Task Manager Update

Error Recovery

Read

calls MLM to repair command

task progress/ next subgoal

Write

Figure 6. Overview of our iterative framework. The system performs perception, memory retrieval, planning, verification, and execution in a loop. A three-layer architecture (perception, decision, execution) interacts with short- and long-term memory to support structured ICAL and case-based transfer.

Transfer System Formalization. The iterative reasoning process is formalized as follows: (1) Memory: The memory M = {mi} stores multimodal trajectories, their CSDs, plans, validation results, and execution traces. It maintains dual representations: symbolic graphs for interpretability and vector embeddings for fast retrieval. (2) Transfer Space (Five Axes): The transfer space T = {struct,attr,proc,func,inter} represents the structural, attribute, procedural, functional, and interaction dimensions of transferable knowledge. (3) Retrieval Operator: The retrieval operator SK = R(xt,M,T) retrieves K exemplars and their corresponding cross-axis similarity evidence from the memory bank. (4) Instruction-tuned MLLM: The instruction-tuned MLLM, denoted as fθ (with frozen parameters θ), performs structured in-context learning:

[πt,Asst] = fθ(xt,SK,protocol), (1)

where πt represents a hierarchical plan, and Asst corresponds to self-verification assertions. (5) Verifier: The verifier {pass,fail} = V(πt,Asst,xt), report ensures the internal logical consistency and external task feasibility of the plan and assertions. (6) Executor: The executor Exec(πt) → tracet executes the plan and collects the resulting trajectory. (7) Memory Update: The memory update function M′ = U(M,tracet) updates both symbolic and vector channels for continual learning.

Table 1. From-scratch learning in Minecraft (Success@0→10 / Success@0→30). Higher is better. Results are averaged over worlds, map variants, and resource configurations. denotes full model; denotes component disabled.

Recipe (Succ@0→10 / 30) ↑ Functional Eq. (Succ@0→10 / 30) ↑ Crafting Chain (Succ@0→10 / 30) ↑ Utility Blocks (Succ@0→10 / 30) ↑ Bed Iron Pickaxe Shield BridgeEq SmeltEq WeaponEq WeaponSet ToolBench ArmorSet CraftGrid CraftTable Furnace

Method

Memory-enhanced Models

Voyager [42] 35.0 / 62.5 30.0 / 57.5 25.0 / 50.0 17.5 / 40.0 15.0 / 35.0 15.0 / 32.5 22.5 / 45.0 27.5 / 55.0 17.5 / 40.0 45.0 / 77.5 35.0 / 65.0 25.0 / 47.5 Self Verification [42] 25.0 / 47.5 25.0 / 45.0 17.5 / 37.5 12.5 / 27.5 10.0 / 25.0 7.5 / 20.0 15.0 / 32.5 17.5 / 40.0 10.0 / 27.5 30.0 / 55.0 22.5 / 45.0 12.5 / 32.5 MrSteve [36] 22.5 / 40.0 20.0 / 37.5 17.5 / 35.0 47.5 / 77.5 45.0 / 77.5 42.5 / 72.5 15.0 / 32.5 17.5 / 35.0 12.5 / 27.5 25.0 / 45.0 17.5 / 35.0 10.0 / 22.5 MP5 [38] 40.0 / 67.5 37.5 / 65.0 35.0 / 60.0 45.0 / 75.0 42.5 / 70.0 37.5 / 65.0 35.0 / 65.0 35.0 / 62.5 30.0 / 57.5 42.5 / 72.5 35.0 / 65.0 27.5 / 52.5 Patroller [38] 35.0 / 55.0 32.5 / 52.5 27.5 / 47.5 35.0 / 60.0 35.0 / 55.0 30.0 / 50.0 27.5 / 50.0 25.0 / 45.0 22.5 / 42.5 35.0 / 57.5 30.0 / 50.0 22.5 / 37.5 JARVIS-1 [45] 60.0 / 87.5 50.0 / 85.0 50.0 / 80.0 47.5 / 77.5 45.0 / 75.0 40.0 / 70.0 37.5 / 72.5 42.5 / 75.0 30.0 / 65.0 52.5 / 80.0 55.0 / 82.5 35.0 / 67.5 SelfCheck [45] 45.0 / 75.0 42.5 / 70.0 37.5 / 65.0 35.0 / 60.0 32.5 / 57.5 27.5 / 50.0 25.0 / 50.0 27.5 / 57.5 17.5 / 40.0 45.0 / 72.5 32.5 / 60.0 20.0 / 45.0

###### Few-shot Variants

- Echo (1-shot) 50.0 / 80.0 50.0 / 80.0 42.5 / 75.0 37.5 / 60.0 32.5 / 65.0 27.5 / 50.0 37.5 / 72.5 32.5 / 67.5 20.0 / 60.0 55.0 / 92.5 55.0 / 75.0 35.0 / 62.5

- Echo (2-shot) 62.5 / 90.0 50.0 / 87.5 52.5 / 80.0 35.0 / 65.0 37.5 / 62.5 40.0 / 65.0 37.5 / 72.5 40.0 / 70.0 22.5 / 67.5 55.0 / 92.5 55.0 / 87.5 35.0 / 62.5 Echo (4-shot) 62.5 / 92.5 50.0 / 87.5 52.5 / 85.0 40.0 / 72.5 35.0 / 62.5 40.0 / 65.0 37.5 / 72.5 40.0 / 75.0 22.5 / 67.5 55.0 / 92.5 55.0 / 87.5 35.0 / 65.0 Echo (8-shot) 62.5 / 92.5 52.5 / 87.5 55.0 / 87.5 50.0 / 80.0 47.5 / 80.0 45.0 / 75.0 40.0 / 77.5 42.5 / 82.5 27.5 / 67.5 57.5 / 92.5 55.0 / 87.5 37.5 / 70.0

### 4. Experiment

Experimental Objectives. Integrate multimodal long-term memory with transfer learning to address the open-world distribution shift problem. (1) To demonstrate that explicit transfer axes combining structured ICAL outperform methods based solely on memory or policy transfer in cross-world and cross-task generalization; (2) To show that consistency self-checking contributes to stability in longhorizon planning; (3) To illustrate how memory replay (continual learning) leads to progressive performance improvement and enhanced interpretability.

#### 4.1. Cross-world Learning from Scratch

Multi-task Evaluation Under Cold-starting. As shown in Table 1, the primary objective of this section is to evaluate an agent’s generalization ability and learning efficiency from scratch under open-world testing. Specifically, this experiment examines whether a model, when initialized with no prior task knowledge, can rapidly acquire crafting and reasoning skills within the first 10 and 30 learning episodes. Evaluation Tasks and Metrics. The tasks are grouped into four families: Recipe (e.g., Iron Pickaxe, Bed, Shield) evaluates structural and shape-based recipe transfer. Functional Equivalence (BridgeEq, SmeltEq, WeaponEq) tests the agent’s ability to perform functionally equivalent reasoning—i.e., using alternative equivalent items when the required ones are unavailable. Crafting Chain (WeaponSet—crafting wooden, stone, and iron swords; ToolBench—crafting a stone pickaxe, shovel, and hoe; ArmorSet—crafting a full set of iron armor) measures multi-step dependency reasoning. Finally, Utility Blocks (Crafting Grid, Crafting Table, Furnace) evaluate the agent’s ability to correctly utilize functional blocks to complete short-horizon, dependency-based tasks. Success@0→10 and Success@0→30 represent average

success rates within the first 10 and 30 episodes, respectively (higher is better).

2-Shot is Competitive. The comparison includes several strong baselines and ablation variants: Voyager [42], MrSteve [36], MP5 [38], and JARVIS-1 [45], each with corresponding component removal experiments. Voyager [42] without the Self-Verification mechanism shows a significant drop in performance, indicating that self-verification is crucial for stable learning. MrSteve performs best on functional-equivalence tasks but is weaker on structural and multi-step reasoning tasks. MP5 demonstrates consistent robustness, and its Patroller module (inspection module) [38] significantly improves task success rates under complex visual variations. JARVIS-1 stands out as the most stable overall baseline, achieving leading scores across nearly all task families. Removing its SelfCheck module [45] leads to a 10–20 point drop in success rates for most tasks, further confirming that self-consistency checking is vital for cross-world stability. Echo exhibits competitive performance even in the 2-shot setting.

Few-Shot Gains Saturate. In the few-shot setting, our method introduces ICAL during planning with 1–8 contextual examples. Results show steady improvement as the number of examples (k) increases, especially for multi-step and long-term tasks. The 1-shot variant already matches or surpasses most baselines, while the 4-shot and 8-shot variants achieve the highest overall success rates—up to 62.5/92.5 on the Recipe and Crafting Table families. Although functional-equivalence tasks in the 8-shot setting do not reach JARVIS-1 [45]’s absolute peak, their learning curves are smoother, indicating that ICAL effectively accelerates cross-world adaptation during cold-start learning.

#### 4.2. Continuous Learning Test

This experiment (Fig. 8) aims to evaluate the proposed method’s continuous learning efficiency and stability un-

Recipe

0%

Task Axis Correlation

KeepOnly

1.0

|0.83|0.37|0.35|0.29|0.30|
|---|---|---|---|---|
|0.20|0.67|0.43|0.73|0.34|
|0.18|0.71|0.73|0.40|0.50|
|0.21|0.48|0.71|0.24|[Figure 199]<br><br>0.71|

- -15%

- -10%

- -5%

- -15%

- -10%

- -5%

0%

Functional Eq.

AttributeStructuralProceduralFunctionalInteraction

- -15%

- -10%

- -5%

- -15%

- -10%

- -5%

0%

Crafting Chain

AttributeStructuralProceduralFunctionalInteraction

- -15%

- -10%

- -5%

- -15%

- -10%

- -5%

0%

Utility Blocks

AttributeStructuralProceduralFunctionalInteraction

- -15%

- -10%

- -5%

[Figure 200]

Recipe

0.8

Functional Eq.

0.6

0%

0%

0%

0%

0.4

Crafting Chain

Remove

- -15%

- -10%

- -5%

0.2

Utility Blocks

0.0

AttributeStructuralProceduralFunctionalInteraction

AttributeStructuralProceduralFunctionalInteraction

- Figure 7. Comparison of task performance when keeping or removing individual design axes. Left: bar charts for each task showing median performance change under “Keep Only” and “Remove” scenarios. Right: correlation heatmap between task outcomes and design axes (thicker borders indicate stronger correlations).

0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

0

5

10

15

20

25

30

35

40

45

50

55

60

(a) Continuous Learning Group 1 (Detailed)

Ours

JARVIS-1

MP5

MrSteve

Voyager

0 5 10 15 20 25 30

0

10

20

30

40

50

60

(b) Learning Group 2 (Coarse)

0 5 10 15 20 25 30

0

10

20

30

40

50

60

(c) Smooth + Confidence Interval

Episode

SuccessRate(%)

- Figure 8. Continuous learning performance comparison. The figure shows the success rate (%) over 31 training episodes (0–30) across five agents: Ours, JARVIS-1, MP5, MrSteve, and Voyager. The shaded region (episodes 5–15) highlights the fast learning phase of our method. Compared to all baselines, our method demonstrates a faster learning rate in the mid-phase.

cess rate ranking is: Echo (45), MP5 (43) [38], JARVIS1 (35) [45], MrSteve (33) [36], and Voyager (18) [42]. From a “fast start vs. strong finish” perspective, JARVIS1 [45] excels in the cold-start phase (0–10) due to its pre-trained policy library and self-checking mechanism, but its growth plateaus beyond 20 episodes, indicating limited scalability in multi-task and long-horizon reasoning. Although ours starts slower, it maintains steady gains from episodes 10–30, validating that explicit transfer axes (Attr/Struct/Proc/Func/Inter) and structured ICL foster stronger long-term planning and knowledge reuse once sufficient experience is accumulated. This trend is consistent with the task-family sensitivity and stability analyses in §4.1 and §4.3.

#### 4.3. Ablation of Explicit Transfer Axes

Single-Axis Removal/Retention. As shown in Figure 7, this experiment aims to verify the necessity and advantages of using explicit transfer axes in multi-axis similarity modeling. By comparing two ablation settings-Keep-Only (retaining only one axis) and Remove (removing one axis)we quantify the independent contribution and sensitivity of each semantic axis across four task families (Recipe, Functional Eq., Crafting Chain, Utility Blocks). The goal is to examine whether explicit axis alignment provides higher stability and interpretability than implicit holistic similarity modeling. The setup involves four task families and five explicit transfer axes, resulting in ten ablation scenarios: five Remove-Axis (removing one axis) and five Keep-OnlyAxis (retaining only one axis). The evaluation metric is the relative change in success rate (∆Success%), where negative values indicate performance degradation.

der open-world tasks. Specifically, we focus on the following aspects: (1) the adaptation speed during the early coldstart phase (episodes 0–10); (2) the improvement and stability during the mid-to-late phase (episodes 20–30) driven by long-horizon planning and memory replay; and (3) the crossover points and final convergence performance compared with strong baselines, including JARVIS-1, MP5, MrSteve, and Voyager. The goal is to validate the effectiveness of our method in achieving sustainable adaptation and cross-task generalization in long-term learning scenarios.

Axis Interactions. The experimental results show that different semantic transfer axes have a significant impact on task performance:

Mid-to-Late Phase Performance. Our method shows slower progress in the early episodes but accelerates rapidly after episode 10, reaching a stable plateau around 46–48% in the final stage. In contrast, JARVIS-1 learns faster initially but saturates after 20 episodes, while MP5 improves steadily, and both MrSteve and Voyager remain

- • The Attribute axis is crucial for recipe tasks. Removing it leads to a significant drop in Recipe (-11%).
- • The Structural axis affects functional equivalence and crafting chain tasks. Removing it causes declines in Func-

- at lower performance levels. At episode 30, the suc-

- tional Eq. (-7%) and Crafting Chain (-9%).
- • The Procedural axis has the greatest impact on longhorizon tasks (e.g., Crafting Chain). Removing it leads to severe degradation (-12%).
- • The Functional axis dominates functional equivalence tasks. Removing it almost disables the tasks (-9%).
- • The Interaction axis affects short-term tasks. Removing it causes a major drop in Utility Blocks (-7%).

The experimental results show that removing a single axis causes a significant performance drop, much greater than the drop when only retaining a single axis. The Structural axis primarily affects geometric and recipe-related tasks; the Attribute axis is crucial for functional equivalence and crafting chain tasks; the Procedural axis has a decisive impact on long-horizon tasks like Crafting Chain and Crafting Table; the Functional axis dominates functional equivalence tasks; and the Interaction axis strongly influences utilities operation tasks. Explicit transfer axes help clarify the dependencies between tasks and effectively locate the sources of errors, such as process reasoning errors or interface positioning errors.

#### 4.4. Case Study

This section demonstrates an experimental example for interpretability analysis. The target sample is retrieved and matched through the Func axe (since the functions of planks and stone are similar). After performing ICL, a similar example for the stone pickaxe is inferred. The specific process is shown in the Figure 9. The key CSD for the entire transfer process is shown below.

Transfer Example. Crafting a Wooden Pickaxe. Task Steps. (1) Convert oak logs into oak planks. (2) Turn planks into sticks. (3) Attempt to craft the pickaxe directly but fail, realizing that a crafting table is required. (4) Craft and place the crafting table. (5) Arrange planks and sticks on the crafting table to craft the wooden pickaxe.

Retrieved Item Descriptions (Func Content). Oak Planks: Used as materials for crafting tools, and as building materials. Stone: Serve as components for crafting tools and construction materials.

ICAL Result. Crafting a Stone Pickaxe. Task Steps. (1) Use the wooden pickaxe to mine stone blocks and acquire stone. (2) Collect planks and craft planks into sticks. (3) Craft and place the crafting table. (4) Arrange stones and sticks on the crafting table to craft the stone pickaxe.

Analysis. The key point of this transfer is the functional similarity between materials (oak planks and stone), which allows the task structure for crafting a wooden pickaxe to be applied to crafting a stone pickaxe. ICAL recognizes the pattern in task steps (gather materials, use the crafting table, arrange materials) and adapts it, transferring the knowledge from one task to the other.

|[Figure 201]|
|---|

[Figure 202]

(a) Find planks (b) Craft wooden pickaxe

|[Figure 203]|
|---|

|[Figure 204]|
|---|

(c) Find stones (d) Craft stone pickaxe

Figure 9. Transferring from a wooden pickaxe to a stone pickaxe.

### 5. Conclusion and Discussion

Research Focus Comparison. Compared with representative works [38, 42, 45], our approach emphasizes skill acquisition and learning rather than exploration or perception. As a result, Echo is less effective in actively exploring unfamiliar environments. For instance, MP5 employs active perception to continuously gather new information [38], whereas Echo relies more on prior knowledge and retrieval, which weakens its performance in information-sparse settings. In addition, our method shows a slower initial learning rate.

Applicability to the Real Physical World. Our method is evaluated mainly in Minecraft, an open-ended and complex but still idealized environment with simple and consistent rules [35]. While such predictability facilitates efficient skill learning and transfer, it also limits real-world applicability. Compared with Minecraft, real-world tasks are more diverse, ambiguous, and causally complex, making transfer learning more reliant on the reasoning and generalization abilities of large models. Therefore, skill transfer in the real world is unlikely to be as straightforward as in Minecraft.

Conclusion. This study explores explicit multimodal memory transfer in multimodal embodied agents. To address challenges in structuring and transferring multimodal memory, we propose five explicit transfer dimensions and integrate them with ICAL for effective organization and use. Compared with four baseline agents focusing on different aspects, our model achieves superior learning efficiency and task generalization in learning-from-scratch settings. We hope this framework inspires future research on enhancing planning and reasoning in interactive multimodal agents.

Acknowledgments. This work was partially supported by the National Natural Science Foundation of China under grant 62572104, and 62220106008.

### References

- [1] Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for hierarchical planning. Conference on Neural Information Processing Systems (NeurIPS), 36:22304– 22325, 2023. 2
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Conference on Neural Information Processing Systems (NeurIPS), 35: 23716–23736, 2022. 3
- [3] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. International Conference on Learning Representations (ICLR), 2024. 3
- [4] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Conference on Neural Information Processing Systems (NeurIPS), 35:24639– 24654, 2022. 3
- [5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Conference on Neural Information Processing Systems (NeurIPS), 33:1877–1901,

2020. 3

- [6] Shaofei Cai, Zihao Wang, Xiaojian Ma, Anji Liu, and Yitao Liang. Open-world multi-task control through goal-aware representation learning and adaptive horizon prediction. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 13734–13744, 2023. 3
- [7] Shaofei Cai, Bowei Zhang, Zihao Wang, Xiaojian Ma, Anji Liu, and Yitao Liang. Groot: Learning to follow instructions by watching gameplay videos. In The Twelfth International Conference on Learning Representations, 2023. 2, 3
- [8] Shaofei Cai, Bowei Zhang, Zihao Wang, Haowei Lin, Xiaojian Ma, Anji Liu, and Yitao Liang. Groot-2: Weakly supervised multi-modal instruction following agents. In The Thirteenth International Conference on Learning Representations, 2024. 3
- [9] Qi Chai, Zhang Zheng, Junlong Ren, Deheng Ye, Zichuan Lin, and Hao Wang. Causalmace: Causality empowered multi-agents in minecraft cooperative tasks. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 14410–14426, 2025. 3
- [10] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. Dense x retrieval: What retrieval granularity should we use? In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 15159–15177, 2024. 3
- [11] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. A survey on in-context learning. In Confer-

- ence on Empirical Methods in Natural Language Processing (EMNLP), pages 1107–1128, 2024. 3
- [12] Yubo Dong, Xukun Zhu, Zhengzhe Pan, Linchao Zhu, and Yi Yang. Villageragent: A graph-based multi-agent framework for coordinating complex task dependencies in minecraft. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 16290–16314, 2024. 3
- [13] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: an embodied multimodal language model. In ICML (International Conference on Machine Learning), pages 8469–8488,

2023. 2

- [14] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Conference on Neural Information Processing Systems (NeurIPS), 35:18343–18362, 2022. 3
- [15] Yicheng Feng, Yuxuan Wang, Jiazheng Liu, Sipeng Zheng, and Zongqing Lu. Llama-rider: Spurring large language models to explore the open world. In Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), pages 4705–4724, 2024. 3
- [16] Honghao Fu, Junlong Ren, Qi Chai, Deheng Ye, Yujun Cai, and Hao Wang. Vistawise: Building cost-effective agent with cross-modal knowledge graph for minecraft. In EMNLP (Conference on Empirical Methods in Natural Language Processing), 2025. 2, 3
- [17] William H Guss, Brandon Houghton, Nicholay Topin, Phillip Wang, Cayden Codel, Manuela Veloso, and Ruslan Salakhutdinov. Minerl: a large-scale dataset of minecraft demonstrations. In International Joint Conference on Artificial Intelligence (IJCAI), pages 2442–2448, 2019. 3
- [18] Or Honovich, Uri Shaham, Samuel Bowman, and Omer Levy. Instruction induction: From few examples to natural language task descriptions. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 1935– 1952, 2023. 3
- [19] Xiaoyang Hu, Shane Storks, Richard L Lewis, and Joyce Chai. In-context analogical reasoning with pre-trained language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023. 2, 3
- [20] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems (TIS), 43(2):1–55, 2025. 3
- [21] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research (JMLR), 24

(251):1–43, 2023. 3

- [22] Hyuhng Joon Kim, Hyunsoo Cho, Junyeob Kim, Taeuk Kim, Kang Min Yoo, and Sang-goo Lee. Self-generated in-context

- learning: Leveraging auto-regressive language models as a demonstration generator. arXiv preprint arXiv:2206.08082, 2022. 3
- [23] Jonathan Leung, Yongjie Wang, and Zhiqi Shen. Knowledge retrieval in llm gaming: A shift from entity-centric to goaloriented graphs. arXiv preprint arXiv:2505.18607, 2025. 2
- [24] Hao Li, Xue Yang, Zhaokai Wang, Xizhou Zhu, Jie Zhou, Yu Qiao, Xiaogang Wang, Hongsheng Li, Lewei Lu, and Jifeng Dai. Auto mc-reward: Automated dense reward design with large language models for minecraft. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 16426– 16435, 2024. 3
- [25] Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, and Liqiang Nie. Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks. Conference on Neural Information Processing Systems (NeurIPS), 37:49881–49913, 2024. 2, 3
- [26] Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, and Liqiang Nie. Optimus-2: Multimodal minecraft agent with goal-observation-action conditioned policy. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [27] Jiachang Liu, Dinghan Shen, Yizhe Zhang, William B Dolan, Lawrence Carin, and Weizhu Chen. What makes good incontext examples for gpt-3? In Deep Learning Inside Out Workshop (DeeLIO), pages 100–114, 2022. 3
- [28] Sheng Liu, Haotian Ye, Lei Xing, and James Y Zou. Incontext vectors: Making in context learning more effective and controllable through latent space steering. In International Conference on Machine Learning (ICML), pages 32287–32307, 2024. 3
- [29] Shaoteng Liu, Haoqi Yuan, Minda Hu, Yanwei Li, Yukang Chen, Shu Liu, Zongqing Lu, and Jiaya Jia. Rl-gpt: Integrating reinforcement learning and code-as-policy. Conference on Neural Information Processing Systems (NeurIPS), 37: 28430–28459, 2024. 3
- [30] Shunyu Liu, Yaoru Li, Kongcheng Zhang, Zhenyu Cui, Wenkai Fang, Yuxuan Zheng, Tongya Zheng, and Mingli Song. Odyssey: Empowering minecraft agents with openworld skills. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 2025. 3
- [31] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 8086–8098, 2022. 3
- [32] Jianwen Luo, Yiming Huang, Jinxiang Meng, Fangyu Lei, Shizhu He, Xiao Liu, Shanshan Jiang, Bin Dong, Jun Zhao, and Kang Liu. Gate: Graph-based adaptive tool evolution across diverse tasks. arXiv preprint arXiv:2502.14848, 2025. 2
- [33] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in retrieval-augmented large language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5303–5315, 2023. 3
- [34] Nicola Messina, Giuseppe Amato, Andrea Esuli, Fabrizio Falchi, Claudio Gennaro, and St´ephane Marchand-Maillet.

- Fine-grained visual textual alignment for cross-modal retrieval using transformer encoders. ACM Transactions on Multimedia Computing, Communications, and Applications (ACM TOMM), 17(4):1–23, 2021. 2
- [35] Ruaridh Mon-Williams, Gen Li, Ran Long, Wenqian Du, and Christopher G Lucas. Embodied large language models enable robots to complete complex tasks in unpredictable environments. Nature Machine Intelligence, 2025. 8
- [36] Junyeong Park, Junmo Cho, and Sungjin Ahn. Mrsteve: Instruction-following agents in minecraft with what-wherewhen memory. In International Conference on Learning Representations (ICLR), 2025. Poster. 2, 3, 4, 6, 7
- [37] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In UIST (ACM Symposium on User Interface Software and Technology), 2023. 2
- [38] Yiran Qin, Enshen Zhou, Qichang Liu, Zhenfei Yin, Lu Sheng, Ruimao Zhang, Yu Qiao, and Jing Shao. Mp5: A multi-modal open-ended embodied system in minecraft via active perception. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 3, 6, 7, 8
- [39] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In NAACL (Conference of the North American Chapter of the Association for Computational Linguistics), pages 2655–2671,

2022. 3

- [40] Elias Stengel-Eskin, Archiki Prasad, and Mohit Bansal. Regal: Refactoring programs to discover generalizable abstractions. In International Conference on Machine Learning (ICML), pages 46605–46624. PMLR, 2024. 3
- [41] Ryan Volum, Sudha Rao, Michael Xu, Gabriel DesGarennes, Chris Brockett, Benjamin Van Durme, Olivia Deng, Akanksha Malhotra, and William B Dolan. Craft an iron sword: Dynamically generating interactive game characters by prompting large language models tuned on code. In Wordplay 2022 workshop, 2022.
- [42] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024. 2, 3, 5, 6, 7, 8
- [43] Xinyi Wang, Wanrong Zhu, Michael Saxon, Mark Steyvers, and William Yang Wang. Large language models are latent variable models: Explaining and finding good demonstrations for in-context learning. Conference on Neural Information Processing Systems (NeurIPS), 36:15614–15638, 2023. 3
- [44] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: interactive planning with large language models enables open-world multi-task agents. In Conference on Neural Information Processing Systems (NeurIPS), 2023. 1, 2
- [45] Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Jinbing Hou, Bowei Zhang, Haowei Lin, Zhaofeng He, Zilong Zheng, Yaodong Yang, Xiaojian Ma, and Yitao Liang. Jarvis-1: Open-world multi-task agents with memory-

- augmented multimodal language models. IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI), 2024. 1, 2, 3, 5, 6, 7, 8
- [46] Zihao Wang, Shaofei Cai, Zhancun Mu, Haowei Lin, Ceyao Zhang, Xuejie Liu, Qing Li, Anji Liu, Xiaojian Shawn Ma, and Yitao Liang. Omnijarvis: Unified vision-languageaction tokenization enables open-world instruction following agents. Conference on Neural Information Processing Systems (NeurIPS), 37:73278–73308, 2024. 3
- [47] Zhe Yang, Damai Dai, Peiyi Wang, and Zhifang Sui. Not all demonstration examples are equally beneficial: Reweighting demonstration examples for in-context learning. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 13209–13221, 2023. 3
- [48] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2022. 2, 3
- [49] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. Transactions on Machine Learning Research, 2022. 3
- [50] Shu Yu and Chaochao Lu. ADAM: An embodied causal agent in open-world environments. In International Conference on Learning Representations (ICLR), 2025. 3
- [51] Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, Yibei Liu, Chenghao Li, Qigan Sun, Shuai Yuan, Fachrina Dewi Puspitasari, Dongshen Han, Guoqing Wang, Sung-Ho Bae, and Yang Yang. Text summarization via global structure awareness. In International Conference on Learning Representations (ICLR), 2026. Poster. 2
- [52] Jiaquan Zhang, Chaoning Zhang, Shuxu Chen, Xudong Wang, Zhenzhen Huang, Pengcheng Zheng, Shuai Yuan, Sheng Zheng, Qigan Sun, Jie Zou, Lik-Hang Lee, and Yang Yang. Learning global hypothesis space for enhancing synergistic reasoning chain. In International Conference on Learning Representations (ICLR), 2026. Poster.
- [53] Malu Zhang, Shuai Wang, Jibin Wu, Wenjie Wei, Dehao Zhang, Zijian Zhou, Siying Wang, Fan Zhang, and Yang Yang. Toward energy-efficient spike-based deep reinforcement learning with temporal coding. IEEE Computational Intelligence Magazine, 20(2):45–57, 2025. 3
- [54] Yiming Zhang, Shi Feng, and Chenhao Tan. Active example selection for in-context learning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022. 3
- [55] Zhonghan Zhao, Wenhao Chai, Xuan Wang, Boyi Li, Shengyu Hao, Shidong Cao, Tian Ye, and Gaoang Wang. See and think: Embodied agent in virtual environment. In European Conference on Computer Vision (ECCV), pages 187–204. Springer, 2024. 3
- [56] Zhonghan Zhao, Kewei Chen, Dongxu Guo, Wenhao Chai, Tian Ye, Yanting Zhang, and Gaoang Wang. Hierarchical auto-organizing system for open-ended multi-agent navigation. In International Conference on Learning Representations (ICLR), Workshop on Large Language Model (LLM) Agents, 2024.

- [57] Pengcheng Zheng, Chaoning Zhang, Jiarong Mo, GuoHui Li, Jiaquan Zhang, Jiahao Zhang, Sihan Cao, Sheng Zheng, Caiyan Qin, Guoqing Wang, and Yang Yang. LLaVA-FA: Learning fourier approximation for compressing large multimodal models. In International Conference on Learning Representations (ICLR), 2026. Poster. 3
- [58] Sipeng Zheng, Jiazheng Liu, Yicheng Feng, and Zongqing Lu. Steve-eye: Equipping llm-based embodied agents with visual perception in open worlds. In International Conference on Learning Representations (ICLR), 2024. 2, 3
- [59] Xinyue Zheng, Haowei Lin, Kaichen He, Zihao Wang, Qiang Fu, Haobo Fu, Zilong Zheng, and Yitao Liang. Mcu: An evaluation framework for open-ended game agents. In International Conference on Machine Learning (ICML), 2025. 3
- [60] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. In International Conference on Learning Representations (ICLR),

2022. 3

- [61] Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, et al. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144, 2023. 2

