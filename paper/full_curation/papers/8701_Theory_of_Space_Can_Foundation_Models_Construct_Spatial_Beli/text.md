# arXiv:2602.07055v1[cs.AI]4Feb2026

[Figure 1]

THEORY OF SPACE: CAN FOUNDATION MODELS CONSTRUCT SPATIAL BELIEFS THROUGH ACTIVE EXPLORATION?

Pingyue Zhang1,∗,†, Zihan Huang∗, Yue Wang4,∗, Jieyu Zhang3,∗, Letian Xue1, Zihan Wang1, Qineng Wang1, Keshigeyan Chandrasegaran2, Ruohan Zhang2, Yejin Choi2, Ranjay Krishna3, Jiajun Wu2, Li Fei-Fei2, Manling Li1,†

1Northwestern University 2Stanford University 3University of Washington 4Cornell University pingyuezhang@u.northwestern.edu, manling.li@northwestern.edu

∗Equal contribution †Corresponding author

ABSTRACT

Spatial embodied intelligence often operates under partial observability, where agents must act to acquire missing information rather than passively consume complete observations. In such settings, progress depends on actively selecting informative actions that reduce uncertainty and support the construction of spatial understanding. While multimodal foundation models have shown strong performance on passive multimodal perception and reasoning tasks, their ability to support active, self-directed exploration under partial observability has not been systematically studied. In particular, it remains unclear whether and how these models can decide what to observe next in order to build and maintain a coherent spatial belief over time. We therefore propose THEORY OF SPACE, defined as an agent’s ability to actively acquire information through self-directed, active exploration and to construct, revise, and exploit a spatial belief from sequential, partial observations. We implement THEORY OF SPACE using a benchmark with textual and visual environments. Rather than solving specific tasks, the goal is curiositydriven exploration to build a complete, accurate spatial belief. A core innovation is spatial belief probing: we prompt it to reveal its internal spatial belief as a cognitive map at each step, letting us measure the quality of its underlying spatial model. Our evaluation of state-of-the-art models on a suite of downstream tasks reveals critical bottlenecks: (1) The Active-Passive Gap: Performance degrades when agents must autonomously gather information (e.g., GPT-5.2: 0.57→0.46); (2) Inefficiency: Models explore in an unsystematic way and with high redundancy, failing to match the efficiency of program-based proxies while producing no better results. Through belief probing, we diagnose that perception acts as an initial bottleneck, yet global beliefs suffer further from instability that causes spatial knowledge to degrade over time. Finally, using a false belief paradigm to test belief revision, we uncover Belief Inertia where agents fail to overwrite obsolete priors. This issue exists in text agents but is notably severe in vision-based models.

Website https://theory-of-space.github.io/ Code https://github.com/mll-lab-nu/Theory-of-Space Data https://huggingface.co/datasets/MLL-Lab/tos-data

1 INTRODUCTION

Spatial embodied intelligence relies on active exploration. Unlike disembodied systems that passively process fixed observations, an embodied agent could take actions to alter its position in the environment as exploration, selectively acquiring observations needed to construct spatial knowledge for various spatial tasks. Cognitive science shows that such active exploration leads to substantially better spatial understanding than passively receiving the same information, even when observations are identical (Held & Hein, 1963; Chrastil & Warren, 2012; 2013). But exploration isn’t simply

[Figure 2]

###### Environment Topdown Active Exploration Spatial Belief

[Figure 3]

[Figure 4]

Turn 90; Observe

Evaluation (Pairwise): Where is relative to

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- 1

- 2

- 3

- 4

- 5

front-right, mid distance

[Figure 9]

Northeast, slightly far

[Figure 10]

[Figure 11]

[Figure 12]

Turn 90; Observe

Evaluation (Rotation): Object order if rotate clockwise

[Figure 13]

[Figure 14]

front, mid distance

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Turn 180; Go to ; Observe

Representation Map each object to 2D coord

v

[Figure 23]

[Figure 24]

front, near

[Figure 25]

|[Figure 26]|
|---|

[Figure 27]

[Figure 28]

Turn 90; Observe

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

front-left, mid distance

[Figure 33]

[Figure 34]

vision observation text observation

...

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Go to ; Turn 180; Observe

[Figure 39]

front, near

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

predicted 2D coord

[Figure 44]

[Figure 45]

front, mid distance front-left, mid distance

[Figure 46]

[Figure 47]

agent action

Turn 90; Observe

Figure 1: THEORY OF SPACE: active exploration, probed belief, and evaluation. Left: a top-down view of agent trajectory under partial observability in multiple-room scenes. Middle: the agent’s action loop of moving, rotating, and observing in text- or vision-based environments, receiving egocentric observations and updating an internal belief. Right: evaluation through exploitation of the belief in spatial tasks and direct probing via probed cognitive maps.

about collecting more observations. It is about efficiency, acting under uncertainty to target what is unknown or ambiguous in the agent’s spatial belief and maximize information gain.

We propose THEORY OF SPACE as a framework that explicitly treats exploration as a first-class decision-making problem, decoupled from any single downstream task, focusing on opening the box of the agent’s internal spatial belief. Just as Theory of Mind (ToM) measures how agents model the hidden mental states of others, THEORY OF SPACE assesses an agent’s ability to model the hidden physical structure of the world. We define THEORY OF SPACE as an embodied agent’s ability to actively construct, revise in a dynamic environment, and exploit an internal spatial belief formed through active exploration. Beyond end-task evaluation, THEORY OF SPACE directly probes what the agent knows, what remains uncertain, and how effectively its actions reduce those uncertainties, measured by the number of exploration steps and the uncertainty resolved per action. Figure 1 provides an overview of THEORY OF SPACE’s active exploration, belief probing, and end-task evaluation.

We apply THEORY OF SPACE to evaluate multimodal language models, which are promising candidates for embodied agents. By integrating vision and language, they support unified perception, reasoning, and action over time, yet existing foundation-model benchmarks offer little insight into these capabilities. Most current benchmarks fall into two categories: passive (Weston et al., 2015; Shi et al., 2022; Yang et al., 2025c; Gholami et al., 2025; Yang et al., 2025a), where the agent is only asked to reason over given observations, and task-driven (Gordon et al., 2018; Shridhar et al., 2020b; Li et al., 2025; Yang et al., 2025b), where the agent must achieve a specific goal (e.g., “find the red chair”).

In this work, we propose to systematically evaluate the active process of spatial belief construction. Unlike passive benchmarks, our THEORY OF SPACE benchmark requires agents to actively explore via moving, rotating, and observing to build coherent global beliefs. We implement a scalable environment using ThreeDWorld (Gan et al., 2021) and Objaverse (Deitke et al., 2022) that provides Text-based and Vision-based worlds to localize perception versus reasoning failures. After active exploration, we evaluate the process along two axes: (i) belief exploitation via spatial downstream tasks that probe route-level and survey-level knowledge (Siegel & White, 1975; Montello, 1998); and (ii) exploration efficiency via the number of exploration steps and the accumulated information gain curve over steps, capturing how quickly an agent reduces uncertainty rather than merely increasing coverage. Finally, we design scripted proxy agents that execute strong reference trajectories to disentangle exploration from reasoning. Our evaluation of state-of-the-art foundation models reveals both promising capability in the pure text world and striking limitations in the vision world under THEORY OF SPACE. Active exploration remains a primary bottleneck. Models perform reasonablely well in passive setting, but degrade when they must actively gather information (e.g.,

GPT-5.2: 57.1 → 46.0; GEMINI-3 PRO: 60.5 → 57.3; Figure. 2). We also find a major efficiency gap: rule-based proxy agents reach target coverage in ∼ 9 steps, whereas foundation models explore redundantly, requiring ≥ 14 steps without improving belief accuracy. Thus, even when models can reason about spatial tasks (as reflected in passive performance), they fail to autonomously structure the information-gathering needed to solve them.

Beyond downstream task scores, a core contribution of THEORY OF SPACE is explicit cognitive-map probing, which provides a direct window into the agent’s latent spatial belief as it is constructed and revised. Rather than treating the agent as a black box whose internal state is only inferred from final answers, we prompt the model to expose its evolving cognitive map during exploration, enabling measurement of both belief accuracy and belief uncertainty at each step. This probingbased assessment uniquely supports finegrained diagnosis of how models represent space: it reveals that while perception acts as an initial bottleneck, global beliefs also suffer severely from instability, causing knowledge to degrade over time. This allows us to track belief evolution over time, attribute failures to specific representational breakdowns, and evaluate whether an agent truly “knows what is uncertain” rather than merely producing plausible outputs.

Claude-4.5-Sonnet GLM-4.6V GPT-5.2 Gemini-3-Pro Qwen3-VL

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Vision: Evaluation Performance vs Exploration Cost

| |[Figure 53]| | | | | | |
|---|---|---|---|---|---|---|---|
| |[Figure 54]| |[Figure 55]| | | | |
| |[Figure 56]| | | |[Figure 57]| | |
| | | | | | | | |
| |[Figure 58]| | |[Figure 59]| |[Figure 60]| |
| |[Figure 61]| | |[Figure 62]| | | |
| | | | | | | | |

60%

EvaluationAccuracy

50%

40%

30%

20%

10%

0%

8 10 12 14 16 18 20

Exploration Cost

Figure 2: Evaluation accuracy vs. exploration cost for active exploration in vision-world. Faded icons mark the passive setting, where the agent gets a pre-generated exploration history and only reasons.

Finally, to evaluate the mechanics of dynamic spatial updating, we introduce a False Belief paradigm. By altering the environment (relocating or reorienting objects) after the agent’s initial exploration, we uncover a phenomenon we term spatial belief inertia: agents (particularly in vision-based settings) struggle to overwrite obsolete spatial priors with new sensory evidence. Despite directly observing the new configuration, models persist in their initial, now incorrect coordinates. This reveals a critical failure in spatial memory revision, where foundational models lack the plasticity to revise their internal cognitive maps in response to physical changes.

An important direction for future work is to extend THEORY OF SPACE beyond single-agent settings to multi-agent exploration, where additional challenges arise around coordination and aligning (or sharing) spatial beliefs across agents.

- 2 THEORY OF SPACE

To build agents with spatial intelligence, we argue for evaluating not merely passive reasoning, but the active, self-directed construction of spatial belief from partial observations. We introduce THEORY OF SPACE, a conceptual counterpart to Theory of Mind (ToM). While ToM models hidden mental states of others, THEORY OF SPACE models uncertain, currently unobserved structure of space.

### Definition: THEORY OF SPACE

Ability to construct, revise, and exploit an internal spatial belief.

Here, an internal spatial belief is a mental model (Taylor & Tversky, 1992) of spatial layout and relations maintained in working memory and updated from partial observations. We formalize THEORY OF SPACE within a partially observable framework over a spatial structure S ∈ S. The agent interacts with S to generate a history ht = (o0:t,a0:t), where o and a denote observations and actions. We define THEORY OF SPACE as the capacity to manipulate a probabilistic belief Bt through three core operations:

- 1. Construct: To form a globally consistent internal spatial belief by actively seeking out and integrating partial observations. Formally, the agent integrates ht to approximate the true posterior, denoted as Bt(S) ≈ P(S | ht).
- 2. Revise: To dynamically update the internal belief by using new information acquired through further exploration to resolve conflicts with prior beliefs. Upon an environmental shift S → S′, the agent utilizes exploratory actions ∆h to minimize the divergence from the new ground truth, i.e., Bt+∆t → P(S′ | ht+∆t).
- 3. Exploit: To utilize the current belief to support spatial tasks. The agent utilizes a policy

π conditioned on the belief, π(at | Bt), to perform a downstream task T . In a benchmark context, we measure the value of belief by the performance metric J achieved by this policy:

#### J (π(·|Bt),T ).

- 2.1 A PARADIGM FOR ASSESSING THEORY OF SPACE OF LARGE FOUNDATION MODELS

We propose a new paradigm for Assessing THEORY OF SPACE of large foundation models, which consists of three essential components below.

Task-Agonistic Active Exploration to Move From Passive Viewer to Active Explorer. Evaluating THEORY OF SPACE requires a shift from downstream tasks to exploration, i.e., how an agent explores and decides “what to see next”. In detail, we place the agent in a partially observable environment and explicitly challenge the LLM/VLM agent to actively select actions for itself, including moving, rotating, observing, and terminating. The primary goal is not to complete a downstream task or follow pre-collected trajectories, but to build a general-purpose internal model from its own self-directed exploration with minimal cost. This process encompasses both initial Belief Construction and dynamic Belief Revision. Inspired by the false belief paradigm in Theory of Mind (Wimmer & Perner, 1983) and spatial belief revision (Knauff et al., 2013), we evaluate whether an agent can detect dynamic environmental changes and correctly revise its internal belief during exploration. This demonstrates the ability to customize beliefs given evolving observations. Consequently, the model must identify what remains uncertain and actively terminate exploration only upon acquiring sufficient evidence to form an accurate and responsive internal map.

Belief Exploitation Assessment. To translate THEORY OF SPACE into concrete evaluation tasks, we draw insights from the development of spatial representations (Siegel & White, 1975; Montello, 1998) and define two tasks to measure an agent’s ability to exploit its internal belief for goal-directed behavior: (1) Belief on Route evaluates a path-based understanding of space organized around landmarks such as pairwise spatial relationships along a egocentric navigation; (2) Belief on Survey assesses a map-like “bird’s-eye view” that represents space allocentrically, allowing for the inference of global relationships.

Explicit Probing of the Internal Spatial Belief. Behavioral success such as whether the agent finds the chair cannot directly reveal the quality of agent’s internal model. We require the agent to explicitly represent its spatial belief by probing its cognitive map at any point of exploration. Cognitive maps are structured allocentric representations of space, which is well-established in neuroscience (Tolman, 1948; O’Keefe & Dostrovsky, 1971; Hafting et al., 2005). Thus, we use cognitive maps as the canonical representation of the hidden structure of space. In our implementation, we probe the agent’s internal belief by requiring it to externalize a structured cognitive map. We evaluate the map’s Correctness, and we diagnose reasoning breakdowns with dynamic signals that capture how reliably observations are integrated, tracked over time, and kept coherent across local and global structure. Additionally, we explicitly test the agent’s belief on uncertainty by identifying unobserved regions to measure its uncertainty modeling. This shifts the evaluation from behavioral success to a direct assessment of representational competence, giving us a window into the agent’s spatial belief development.

- 3 BENCHMARKING THEORY OF SPACE ABILITY FOR FOUNDATION MODELS

Unlike task-driven benchmarks that only test task completion, we aim to answer “can the agent form a global environmental belief through exploration?”. We structure the benchmarking into two phases. In the Exploration Phase I, the agent interacts with the environment to construct spatial belief by selecting and executing actions in the action space in § 3.1, and gather a sequence of local

observations to integrate them into a unified spatial belief. In the Reasoning Phase II, the agent is asked to conduct spatial tasks (detailed in § 3.2).

- 3.1 SPATIAL ENVIRONMENT CONSTRUCTION

To ensure controlled experimentation, we procedurally generate multi-room indoor layouts on an N × M grid. Each scene is populated with n indoor objects, each assigned a 2D integer coordinate and a cardinal orientation from (N, S, E, W). The agent begins at a random position, is informed of the total number of rooms and the names of all objects in the scene, and then starts exploration. Following the Gym-style interface (Brockman et al., 2016), we define procedurally generated, highly scalable environments in which each random seed deterministically instantiates a distinct multi-room layout.

Action Space in the Environment. The agent’s interaction with the world is designed to focus on high-level decision-making rather than low-level motor control: Goto to move directly to a currently visible object; Rotate to turn in place by 90◦,180◦, or 270◦; Observe to perceive visible objects in the 90◦ field of view; and Query to obtain a visible object’s absolute 2D coordinates. We additionally assign costs of 1 to Observe and 2 to Query, encouraging Query to be used only when necessary to resolve ambiguity. However, across all models Query is invoked only rarely, so we restrict attention to Observe and measure exploration efficiency by step count instead of action cost.

Observation Feedback from a Text-Vision Parallel Environment. We offer both text-based and vision-based environments, enabling diagnostic analysis of spatial reasoning. Each Observe action returns both textual and visual feedback from a 90◦ field of view. The Text World provides symbolic observations with discrete bins for direction and distance (e.g., “chair is front-left and near”, detailed below), isolating pure spatial reasoning. The Visual World instead supplies ego-centric RGB images rendered in ThreeDWorld (Gan et al., 2021) with Objaverse assets (Deitke et al., 2022), requiring perception to recover spatial relations. To calibrate perception in the visual setting, we provide two reference images, indicating unit distance (1 grid unit) / angle (a 22.5◦ angular cone), and showing all objects with their names and canonical “front” orientation, respectively. Details are shown in Appendix ¶ A.1

Spatial Relation Representation. To ensure that agents perceive and communicate about space using a consistent language across tasks and modalities, we discretize spatial relationships for directions and distances. For allocentric direction, we discretize into eight 45◦ bins aligned with the four cardinal and four intercardinal directions, denoted compactly as {N,NE,E,SE,S,SW,W,NW}. Each bin spans 45◦ around its heading (e.g., N = [−22.5◦,22.5◦)). For egocentric direction, within a 90◦ forward field of view (FOV), we use five labels: front-left [−45◦,−22.5◦), front-slight-left [−22.5◦,0), front 0◦, front-slight-right (0,22.5◦], and front-right (22.5◦,45◦]. For distance, measured in map units independent of direction, we define six bins: same = 0, near (0,2], mid (2,4], slightly far (4,8], far (8,16], and very far (16,32].

- 3.2 DOWNSTREAM SPATIAL TASKS

We use open-ended questions rather than multiple-choice questions to reduce the risk of knowledge leakage. Drawing on prior work (Siegel & White, 1975; Montello, 1998), we define tasks to evaluate an agent’s Route and Survey knowledge, shown in Table 1. Route belief captures how an agent encodes paths and spatial relations from an egocentric step-by-step perspective. Survey belief is a map-like, allocentric representation. An overview of the tasks is present in Figure 3.

- 3.3 ASSESSMENT DIMENSIONS

We define assessment dimensions that align with the core THEORY OF SPACE abilities: construction and revision are evaluated via exploration efficiency and belief quality, while exploitation is evaluated via task success.

(D1) Belief Construction Efficiency. Measures how efficiently the agent collapses spatial uncertainty during exploration. We quantify this using a normalized information gain metric, E. Let M be the number of possible positions for any object at the start of exploration (a uniform prior), and let Ci be the number of positions for object i that remain consistent with all observations gathered by the agent

[Figure 63]

[Figure 64]

###### Route Survey

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Belief on Route: Reasoning about paths and Belief on Survey: Building an allocentric map landmark relations

[Figure 69]

[Figure 70]

[Figure 71]

###### Pairwise Direction

###### Allocentric Map

[Figure 72]

What's spatial direction between

What's 2D coordinate of

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

###### Perspective taking

[Figure 78]

[Figure 79]

[Figure 80]

From perspective, where is

###### Mental Rotation

[Figure 81]

[Figure 82]

What's object sequence if rotate clockwise

[Figure 83]

###### Perspective Determine

[Figure 84]

[Figure 85]

You observe at your front-left, which object's perspective is it?

[Figure 86]

[Figure 87]

###### Location2View

[Figure 88]

###### Action2View

If you move to (0, 2), where is ?

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

After actions , where is ?

[Figure 93]

[Figure 94]

[Figure 95]

###### View2Action

###### View2Location

[Figure 96]

[Figure 97]

You observe at your front-right, generate action sequence

[Figure 98]

[Figure 99]

If you see at your right front, where are you

- Figure 3: THEORY OF SPACE exploitation task suite: it covers route-level egocentric reasoning and survey-level allocentric mapping. Route tasks evaluate path-based inference and egocentric observations. Survey tasks test global mapping, geometric transformation, and perspective conversion. Together they cover both local navigation reasoning and global spatial abstraction.

Dynamic Group Belief on Route Belief on Survey Static Pairwise Relation (direction)

Allocentric Mapping (alloc.map) predict global coordinates (and headings) for all objects.

report allocentric direction and distance from A to B.

Mental Rotation (ment.rot) predict the sequence of front-facing objects during a 360◦ self-rotation. Location2View (loc2view) given a global pose, predict the observation (one object in FOV with ego bins/distances).

Perspective Taking (persp.take) output the observation from a specified object’s perspective. Action-to-View (act2view) given a sequence of Goto/Rotate, predict the final observation (one object in FOV with ego direction/distance bins).

Forward Dynamics

Perspective Decision (perc.dec) infer which object’s perspective the agent is currently adopting. View-to-Action (view2act) recover an action sequence that produces a target observation.

View2Location (view2loc) localize the agent (and optionally orientation) from a target observation under the map.

Backward Dynamics

- Table 1: Task suite comparison: Route belief emphasizes egocentric, step-by-step path reasoning; Survey belief emphasizes allocentric mapping and novel view inference.

N i=1 log2 max(1,Ci)

(calculated by AC-3 algorithm). The efficiency is calculated as E = 1 −

N log2 M . This score ranges from 0 (no information gained, Ci = M) to 1 (all objects perfectly localized, Ci = 1). Note that it can also be used to calculate the accumulated information gain at each step. Information gain is mainly used in text-based environments, since vision-based environments have direct access

to scenes without such ambiguity. Therefore, for vision-based environments, we directly use node coverage to measure exploration efficiency.

Belief Representation and Quality Assessment. A core contribution of THEORY OF SPACE is disentangling spatial memory from spatial inference. We structurally decompose the probed cognitive map into two components:

- • (D2) The Cognitive Map (Observed): Measures fidelity and coherent integration of observations over time. We evaluate using two criteria: (1) Correctness, alignment with ground truth, computed as a composite of positional, directional, and facing accuracy; and (2) dynamic reasoning diagnostics, including Perception quality, Self-tracking, Stability, and Local ↔ Global Consistency, reflecting internal coherence such as the absence of contradictions within the relational graph and between maps and relations.
- • (D3) The Uncertainty Map (Unobserved): Measures how well the agent models plausible hypotheses about unobserved regions. We assess Uncertainty Modeling by providing a candidate set of positions formed by randomly sampled points from both observed and unobserved areas, and measuring the agent’s ability to identify valid locations via F1.

This separation lets us diagnose whether failures stem from misestimating the observed world or from insufficient reasoning about what remains unobserved.

- (D4) Belief Revision. Measures the agent’s ability to revise its spatial belief under latent environment changes. We evaluate this using the False Belief task (§5.3), where objects are covertly manipulated (translated or rotated) following the initial exploration. The agent must re-explore to detect these discrepancies; we measure the accuracy of these identified changes (both object identity and transformation type) using the F1 score. Furthermore, we introduce Belief Inertia to quantify whether belief revision remain biased toward obsolete priors.
- (D5) Belief Exploitation Success. Measures task success when the agent must utilize its spatial belief. For tasks involving spatial relations (direction, persp.take, action2view), we score direction and distance separately, awarding 0.5 for each correct component. For tasks that output coordinates (view2loc, alloc.map), we compute a coordinate similarity score.

- 3.4 EXPLORATION STRATEGIES

To rigorously evaluate spatial cognition, we distinguish between two capabilities: the ability to acquire information (exploration) and the ability to synthesize it (reasoning). We present two evaluation settings: (i) Active Exploration, where the agent must plan actions to reduce uncertainty, and (ii) Passive Comprehension, where the agent reasons over standardized logs generated by scripted proxies.

Uncertainty-Driven On-Policy Exploration. We conduct active evaluation to understand agent ability in exploring the environment to gather necessary information in building spatial belief. In this setting, the evaluated agent must plan and execute its own information-gathering policy. At each step, the agent selects an action based on its observation history and current objective, then receives new observations (text or image). Exploration continues until the agent issues an exploration termination or reaches the step budget. Success requires balancing two goals: maximizing coverage of unknown relations while minimizing action cost. This setting directly reveals whether the agent can recognize what it does not yet know and actively reduce uncertainty through exploration.

Passive Exploration via Scripted Proxy Agents. Evaluating THEORY OF SPACE requires disentangling two intertwined factors: how well an agent explores, and how well it reasons about the observations gathered. An agent may fail either due to a suboptimal exploration policy (missing key evidence) or a deficiency in integrating observations into a coherent belief. To isolate the latter, we introduce proxy agents as an exploration control. In this setting, evaluated models are fed a fixed stream of observations generated by a proxy agent. By enforcing a standardized exploration path, we eliminate variance caused by exploration failures, allowing for a fair evaluation of core reasoning abilities across different architectures. We design two scripted proxies to provide standardized exploration logs. The SCOUT agent is used for visual environments, who rotates at each location to guarantee all objects are observed. Leveraging visual cues like distance, these compact logs are sufficient for accurate belief construction. The STRATEGIST agent is used for text environments, which

follows a belief-driven edge-coverage policy and actively selects viewpoints to maximally reduce ambiguity in coarse symbolic observations. It is implemented with AC-3 constraint propagation to prune inconsistent hypotheses and ensure relations are uniquely determined. Implementation details for both agents appear in Appendix ¶ A.1.

- 4 EVALUATION AND ANALYSIS

We evaluate a set of state-of-the-art proprietary and open-source foundation models. They are evaluated on both passive and active settings described in § 3.4. Unless otherwise specified for ablations, all experiments use three connected 6 × 6 rooms with 4 objects in each (total 12 objects). To enable a like-for-like comparison between the text and vision settings, we instantiate identical room layouts across modalities. We use 384 × 384 images in the vision setting. We generate 100 scenes and create three questions per task per scene, yielding 3 × 9 × 100 = 2700 questions per setting. We mainly evaluate six foundation models: GPT-5.2 (OpenAI, 2025), GEMINI-3 PRO (Google, 2025), CLAUDE-4.5-SONNET (Anthropic, 2025), GLM-4.6V (Zhipu AI Team, 2025), QWEN3-VL (Bai et al., 2025) (235B-A22B-Thinking), and INTERNVL-3.5 (Wang et al., 2025) (241B-A28B). For closed-source reasoning models GPT-5.2, GEMINI-3 PRO, and CLAUDE-4.5-SONNET, we set the temperature to 1 and the maximum number of tokens to 32768. For all other models, we set the temperature to 0. INTERNVL-3.5 supports at most 10 images, so we omit it for the vision-based world setting.

directionpersp.takeperc.dec.act2viewview2act alloc.mapment.rotloc2viewview2loc

Static (S) Dynamic (D) Static (S) Dynamic (D)

Methods Avg.step Route Survey Avg.

Vision-based World Proprietary Models

GPT-5.2 17.2 40.0 36.7 56.2 43.8 40.3 43.4 59.7 56.9 37.8 46.0 GEMINI-3 PRO 13.6 56.3 36.7 68.2 47.2 54.0 63.5 73.0 65.4 52.2 57.3 CLAUDE-4.5 SONNET 19.6 23.7 23.3 18.7 33.3 10.7 37.4 34.7 33.7 50.9 29.6 Open-source Models

|GLM-4.6V 15.0 15.8 18.5 3.3 14.0 0.7 18.9 8.0 18.5 31.8 QWEN3-VL 16.3 16.8 23.3 13.4 24.8 5.7 25.8 16.3 21.5 43.7<br><br>|14.4 21.3|
|---|---|
|HUMAN 9.8 94.5 100.0 100.0 100.0 93.4 93.4 100.0 100.0 86.7 HUMAN WITH TOOL⋆ 11.1 100.0 100.0 100.0 100.0 97.8 100.0 100.0 100.0 93.4<br><br>|96.4 99.0|

Text-based World Proprietary Models

GPT-5.2 11.4 68.8 70.5 80.3 71.0 53.7 77.9 81.0 79.1 66.0 72.0 GEMINI-3 PRO 13.5 78.0 79.2 90.6 75.3 76.3 81.0 94.0 83.3 76.2 81.5 CLAUDE-4.5 SONNET 18.7 65.3 65.3 79.0 62.7 51.7 68.8 76.3 57.0 67.0 65.9 Open-source Models

|GLM-4.6V 14.5 20.8 19.7 12.7 21.8 3.7 13.9 9.3 22.7 26.2 INTERNVL-3.5 15.0 28.8 44.8 26.0 36.8 7.3 31.0 27.7 33.8 38.9 QWEN3-VL 14.1 32.3 45.7 48.2 33.3 11.7 36.4 34.7 35.7 49.9<br><br>|16.8 30.6 36.8|
|---|---|
|HUMAN 10.8 87.8 82.1 100.0 85.5 86.8 66.6 100.0 95.6 75.8 HUMAN WITH TOOL⋆ 12.8 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 91.2<br><br>|86.7 99.0|

- Table 2: Exploitation Performance (%) of Belief Construction via Active Exploration. Models autonomously plan actions and are evaluated on exploration cost, route-level reasoning, and surveylevel reasoning across text- and vision-based environments. GEMINI-3 PRO leads every task and all reasoning metrics, while GPT-5.2 achieves the lowest exploration cost in text-world. Humans outperform in both settings, especially in vision. ⋆Humans can use instruments such as protractors and compasses to infer object positions precisely.

Active Exploration Results. We evaluate models as active agents, where they must autonomously explore the environment to build their spatial belief and terminate the exploration process by their own. This setting tests the full THEORY OF SPACE pipeline, requiring the agent to simultaneously plan an efficient information-gathering trajectory, integrate observations, and maintain a coherent cognitive map under uncertainty. The agent’s performance is measured by its Exploration Efficiency as shown in § 3.3 and its final accuracy on the downstream spatial tasks. The agent has a maximum of 20 exploration steps. Table 2 presents the active performance of the models, providing a holistic view of their ability to translate curiosity into knowledge. Figure 4 illustrates information gain over

directionpersp.takeperc.decact2viewview2act alloc.mapment.rotloc2view view2loc

Static (S) Dynamic (D) Static (S) Dynamic (D)

Methods Route Survey Avg.

Vision-based World Proprietary Models

GPT-5.2 47.3 35.0 63.9 54.5 49.3 64.8 83.3 50.3 65.6 57.1 GEMINI-3 PRO 63.8 36.3 57.5 49.0 58.0 67.2 85.3 70.4 57.0 60.5 CLAUDE-4.5 SONNET 47.3 33.5 37.7 40.8 15.7 54.8 58.3 44.7 54.8 43.1 Open-source Models

GLM-4.6V 11.5 24.5 4.7 19.0 2.7 22.9 11.7 20.0 33.6 16.7 QWEN3-VL 20.8 28.3 22.7 16.7 4.7 33.2 21.7 27.3 40.8 24.9

Text-based World Proprietary Models

GPT-5.2 84.5 88.2 97.0 89.0 76.0 96.3 98.3 94.8 89.2 90.4 GEMINI-3 PRO 82.7 92.7 97.0 87.5 75.7 86.2 91.3 85.7 80.0 86.5 CLAUDE-4.5 SONNET 73.0 80.7 90.7 77.7 59.0 76.9 74.3 59.2 70.7 73.6 Open-source Models

GLM-4.6V 22.3 39.8 25.0 25.3 4.7 21.2 9.0 27.0 35.7 23.4 INTERNVL-3.5 36.7 67.8 42.7 41.2 8.7 37.3 19.3 38.7 43.8 37.4 QWEN3-VL 40.8 69.3 56.5 50.0 17.7 42.8 40.3 42.5 54.6 45.6

- Table 3: Exploitation Performance (%) of Belief Construction via Passive Observations. Models are evaluated as passive comprehension agents on Route- and Survey-level reasoning using standardized observation logs from scripted proxy explorers, decoupling exploration from belief construction across text- and vision-based environments. GEMINI-3 PRO leads most tasks in the vision-based world and achieves the best overall average, while GPT-5.2 leads the text-based world and attains the best overall average.

the course of the exploration turns. GPT-5.2 acquires substantial information early on, but its rate of gain slows in later turns, resulting in lower cumulative information gain than GEMINI-3 PRO and CLAUDE-4.5 SONNET. Moreover, none of the models achieves full coverage relative to the proxy agent. We benchmarked three human subjects across five text and five vision scenes. Humans consistently outperformed foundation models in both domains, particularly in vision. Intuitively, humans scored higher in vision than text as visual information is easier to process. With tools, they achieved near-perfect accuracy

Passive Exploration Results. We evaluate models on trajectories generated by rule-based proxy agent to understand a model’s core spatial reasoning ability regardless of its exploration strategy. The performance of various models in both text-based and vision-based environments is summarized in Table 3. As evaluated, the results show a clear separation: GPT-5.2 and GEMINI-3 PRO lead by a wide margin over other systems, particularly open-source models. A substantial modality gap persists, with text performance far better than vision performance for all models.

### Key Findings: Modality Gap

• Modality Gap Exists: text significantly outperforms vision.

Overall, active accuracies underperform the passive setting. Incomplete exploration leads to drops:

- Figure 4 shows that GPT-5.2 gathers information quickly but often terminates prematurely, leaving uncertainty and lowering active scores relative to passive. Compared to the strategist proxy, which achieves full certainty, models remain less thorough. A second critical disparity is the efficiency gap. In the vision domain, the SCOUT proxy reaches target coverage in ≈ 9 steps, whereas autonomous models expend significantly more actions with no performance benefit. This inefficiency is further highlighted in the text domain. While our primary text experiments utilize the STRATEGIST proxy for maximum coverage, we additionally evaluated the SCOUT proxy in text world. The text-based SCOUT similarly averages ≈ 9 steps. When following these concise trajectories, GPT-5.2 and GEMINI-3 PRO achieve accuracies of 83.9 and 86.7, respectively. These scores surpass their active exploration performance (72.0,81.5 for GPT-5.2 and GEMINI-3 PRO, as in Table 2), demonstrating that models perform better when guided by a short, efficient proxy path than when exploring autonomously.

Text-based World

|Methods 2-room|4-room|
|---|---|
|pass. act. steps<br><br>|pass. act. steps|
|GPT-5.2 92.3 77.8 6.2 GEMINI-3 PRO 86.7 80.6 6.2<br><br>|86.5 66.0 16.4 81.2 77.7 19.7|

Vision-based World

|Methods 2-room<br><br>|4-room|
|---|---|
|pass. act. steps<br><br>|pass. act. steps|
|GPT-5.2 59.3 51.5 10.8 GEMINI-3 PRO 58.3 57.8 6.6<br><br>|52.6 40.3 23.2 56.2 51.5 19.7|

- Table 4: Exploitation Performance (%) for Multi-Room Settings (2-room and 4-room). pass. for passive avg acc, act. for active avg acc, steps for average steps.

1 3 5 7 9 11 13 15 17 19 21 Steps

0.2

0.4

0.6

0.8

1.0

InformationGain

Accumulated Information Gain

glm-4.6v

gemini-3-pro-preview

qwen3-vl-235b-a22b-thinking

claude-sonnet-4-5

gpt-5.2

internvl3.5-241b-a28b

Strategist

80% Samples Ends

Figure 4: Accumulated information gain over exploration steps in the text world.

Different Room Settings. For the two best-performing models, GPT-5.2 and GEMINI-3 PRO, we further evaluate reasoning and exploration under different room configurations: a four-room setting and two three-room settings. In the four-room setting, the main room connects to the other three rooms. Table 4 reports results across different room settings. As the number of rooms increases, exploration cost rises accordingly. For both GPT-5.2 and GEMINI-3 PRO, performance declines as the room number increases, and the active–passive performance gap widens with room number. Moreover, GEMINI-3 PRO requires nearly the same number of exploration steps in the text-only and vision-based environments. Detailed results are in Appendix ¶ B.

Key Findings: Active Exploration as the Bottleneck

- • Performance and Efficiency Deficit: Active agents score lower than reasoning on rule based program histories, and explore less efficiently than the program.
- • Incomplete Coverage: Active agent fails to achieve complete information coverage.
- • Complexity-Widened Gap: The active versus passive difference grows with environment scale; GEMINI-3 PRO degrades least.

Exploration Pattern Manual inspection of agent exploration histories reveals distinct behavioral patterns. For GPT-5.2, the active-passive performance gap stems from unsystematic exploration. Specifically, the agent tends to prioritize any newly discovered door, immediately jumping to inspect it and often leaving the current room partially unexplored. This is compounded by object omission and path redundancy. In contrast, GEMINI-3 PRO adopts a more methodical “rotate-and-scan” strategy, scanning its surroundings before transitioning to new rooms, which is a behavior mirroring the SCOUT proxy agent. Further examples are provided in Appendix ¶ C.

- 5 HOW DO FOUNDATION MODELS MANAGE INTERNAL SPATIAL BELIEF?

In this section, we use the THEORY OF SPACE belief-probing mechanism (as proposed in §2.1) to diagnose how MLLMs manage internal spatial beliefs and move beyond treating the agent as a black box. Figure 5 shows the example of how we probe the belief of agent at each exploration step

- 5.1 COGNITIVE MAP PROBING

Instead of treating the spatial belief as a black box, we probe the agent’s internal state to distinguish verifying known facts from hypothesizing about the unknown. The agent externalizes its belief via a structured JSON containing a Cognitive Map, which records objects currently or previously observed within the field of view.

Representation. For consolidated map, the agent presents its belief as a single, allocentric cognitive map serialized in structured JSON. The map maintains (i) a global layout anchored to the agent’s initial pose, and (ii) a local snapshot that records only the currently visible objects with the current pose as origin to diagnose immediate perceptual errors.

Exploration History Spatial Belief Probing

generate cognitive map

[Figure 100]

Turn 90; Observe

[Figure 101]

|[Figure 102]| | | | |
|---|---|---|---|---|
| | |[Figure 103]<br><br>[Figure 104]| | |
| | | | | |

[Figure 105]

[Figure 106]

front-right, mid distance

[Figure 107]

| | | | | |
|---|---|---|---|---|

[Figure 108]

Turn 90; Observe

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

[Figure 109]

[Figure 110]

front, mid distance

[Figure 111]

[Figure 112]

select unexplored points

Turn 180; Go to ; Observe

[Figure 113]

| | | | | | |
|---|---|---|---|---|---|

[Figure 114]

front, near

| | | | | | |
|---|---|---|---|---|---|
| | | |[Figure 115]| |A|
| |B| | | | |
| | | | | | |
| | | | |C|D|

[Figure 116]

Turn 90; Observe

[Figure 117]

[Figure 118]

front-left, mid distance

[Figure 119]

[Figure 120]

[Figure 121]

Go to ; Turn 180; Observe front, mid distance front-left, mid distance

C, D unexplored

[Figure 122]

[Figure 123]

[Figure 124]

|[Figure 125]|
|---|

starting point Agent pose possible position predicted position observed point unobserved point

[Figure 126]

[Figure 127]

[Figure 128]

Terminate

|A|
|---|

|C|
|---|

- Figure 5: Internal Spatial Belief Probing. At each step, the agent executes an action, receives an observation, and updates its spatial belief. We probe this belief by prompting the agent to (i) output a JSON-structured cognitive map of all observed objects and (ii) select the next unexplored position from a top-down view given a set of labeled candidate points. For clarity, the figure shows the probing process for a single step.

Metrics. We evaluate consolidated map using three complementary metrics. Positional accuracy (pos.acc) is the Euclidean similarity between predicted and true object coordinates: (K/N)·e−RMSE/L, where RMSE is the root mean squared error between predicted and ground-truth object positions, L is the RMS ℓ2-norm of the positions of all objects in the scene, and K/N is the coverage (the ratio of the number of predicted objects K to the number of ground-truth objects N). Directional accuracy (dir.acc) is the accuracy of directional relationship between each pair of objects. Facing accuracy (facing.acc) is the fraction of objects whose predicted facing matches the ground truth.

Using global and local belief representations, we compute a set of diagnostic scores at each turn t (all per-turn except Correctness, which is computed only at the final turn after termination). Unless noted, scores are averaged over turns and scenes:

- • Correctness (final): Measures the accuracy of the agent’s terminal global spatial belief. At the last turn, we evaluate the predicted global map and report a composite score given by the (equally weighted) mean of the three metrics defined above, with weights 1/3 each. We compute dir.acc only for correctness, since the global cognitive map prioritizes consistent pairwise spatial relations.
- • Perception: Measures how accurately the agent interprets newly observed local structure. We compare the predicted local map to the ground-truth local map for the current field of view (FOV), counting only objects that appear in the FOV for the first time.
- • Self-tracking: Measures how well the model estimates its own pose over time. We infer the agent’s pose from the predicted global map and compare it against the ground-truth agent state.
- • Local ↔ Global consistency: Measures whether new local evidence is incorporated into the global belief coherently. Within the same turn, we compare local and global predictions to verify that newly perceived structure is integrated without contradictions.

ori. pos. overall ori. pos. ori. pos. ori. pos. ori. pos.

Selftracking (%)

Uncertainty (%) Vision-based World

Local↔ Global (%)

Methods Correctness (%) Perception (%)

Stability (%)

GPT-5.2 20.2 42.0 32.2 33.5 72.4 57.9 58.7 65.4 56.4 93.3 64.7 53.7 GEMINI-3 PRO 32.2 62.5 52.1 43.8 68.5 52.9 68.3 61.8 62.0 98.8 73.9 70.2

Text-based World

GPT-5.2 91.0 75.1 80.0 100 86.8 96.4 86.0 96.7 67.6 98.0 86.7 64.5 GEMINI-3 PRO 92.5 75.5 81.4 99.9 88.2 91.6 84.8 90.8 67.7 99.9 85.2 79.2

Table 5: Spatial Belief Quality via Cognitive Map Probing. We measure final map correctness and turn-level perception, local global consistency, stability, self-tracking, and uncertainty in textvs. vision-worlds. ori. for orientation and pos. for position. Across models, vision lags text on all metrics, with the largest drop on orientation and stability.

• Stability: Measures whether beliefs about previously observed objects remain non-degrading over time. For each previously observed object, at every subsequent turn we check that its predicted state does not worsen; the per-check score is 1 if the prediction is no worse than in the previous turn.

Results in Table 5 indicate a substantial modality gap between vision and text: performance drops markedly in the vision setting across all metrics, not just belief Correctness. Self-tracking does not appear to be a primary bottleneck, models can often maintain an accurate belief about their own pose. Perception remains a key limitation for state-of-the-art models in visual world settings. In particular, recognizing an object’s facing direction is especially challenging: agents frequently fail to infer orientation and achieve near-chance (or worse) facing Correctness. This weakness is consistent with Table 2, where agents perform poorly on perspective-taking tasks (about 36% accuracy). Stability & Decay. Crucially, the metric reveals that spatial beliefs are highly brittle not just for orientation, but also for position. While Perception scores indicate that models can capture local spatial details with reasonable accuracy, this initial fidelity fails to translate into final map Correctness. This performance gap highlights a critical failure in state maintenance: even when objects are correctly perceived initially, the agent frequently overwrites these verified facts with incorrect predictions in subsequent turns. Thus, the low final Correctness stems not solely from perceptual errors, but from the cumulative effect of unstable belief updates, where valid spatial memories degrade over the course of the episode.

### Key Findings: Cognitive Map Failures (Orientation, Stability, and Belief Drift)

- • Orientation Gap: Vision perception is a bottleneck, especially for object orientation.
- • Unstable Map: Beliefs about previously observed objects degrades over time.
- • Belief Drift: New updates corrupt earlier correct perceptions, lowering final correctness.

Cognitive Map Validation & Correlation. To validate the utility of the probed cognitive map and investigate whether it faithfully reflects the agent’s reasoning process, we first conducted two ablation studies:

- • Sufficiency Test (Oracle Map): We conditioned the model on the ground-truth cognitive map before generating answers for evaluation. Performance rose to near-perfect levels (≈ 95% for both models in both worlds). This confirms that our cognitive map representation captures all necessary information for the tasks; performance bottlenecks stem from the agent’s inability to accurately construct the map, not the representation format itself.
- • Alignment Test (Explicit Reasoning): We prompted the model to explicitly generate the cognitive map before answering the evaluation questions. This resulted in a slight performance degradation compared to direct answering.

These results reveal an externalization gap: the model’s latent internal spatial belief is richer or more accurate than the discretized JSON output it produces. While it is a lossy compression of the agent’s true internal state, the explicit map remains a strong diagnostic signal. We support this

claim by computing the Pearson correlation between the agent’s cognitive map Correctness and downstream task performance. To ensure a robust correlation, we calculate the average performance across five independent cognitive map runs for each sample. As shown in Table 6, belief correctness is consistently and positively correlated with downstream success in both modalities, with all correlations significant (p < .001). The association is stronger in vision (r=0.570/0.645) than in text (r=0.418/0.466). The stronger vision correlation suggests that perception-driven mapping errors and unstable belief updates more directly translate into task failures. Thus, we establish map probing as a validated diagnostic proxy for failure analysis. While acknowledging that correlation does not imply causality, we treat the explicit map as a robust, albeit conservative, signal for diagnosing reasoning breakdowns rather than definitive evidence.

### Key Findings: Maps as a Diagnostic Proxy

• Lossy but Diagnostic: Though a lossy compression, map correctness correlates significantly with downstream success, making it a strong diagnostic signal.

- 5.2 UNCERTAINTY MAP PROBING

Methods Text (%) Vision (%)

GPT-5.2 41.8 57.0 GEMINI-3 PRO 46.6 64.5

Table 6: Pearson correlation (r) between spatial-belief correctness and downstream evaluation performance. All correlations are significant (p < .001).

To probe an agent’s ability to model uncertainty, we provide it with a top-down view of the scene in which all objects are removed, and we overlay a set of candidate points. These points are sampled randomly and include both previously observed and unobserved locations. The agent’s task is to identify which candidate points remain unobserved, thereby revealing its belief over unseen regions.

Representation. The agent receives an empty top

down map that shows only the candidate points and its current position, with no objects present. The agent must select the points that have not yet been observed. In the text based world, the top down map is represented as an N × M symbolic grid, where different symbols denote the agent, gates, and candidate points. In the vision based world, all objects are removed and the agent instead receives a top down image of the environment, check examples in Appendix ¶ A.1. We use F1 to evaluate selected points.

1 3 5 7 9 11 13 15 17 19 21 Steps

0.2

0.4

0.6

0.8

InfoGain

Accumulated Info Gain & Cognitive Map Correctness

0.0

0.2

0.4

0.6

0.8

1.0

CognitiveMapCorrectness

gpt-5.2 (InfoGain)

gpt-5.2 (CogMap Text)

gpt-5.2 (CogMap Vision)

gemini-3-pro (InfoGain)

gemini-3-pro (CogMap Text)

gemini-3-pro (CogMap Vision)

Figure 6: Accumulated Information Gain and Cognitive Map Correctness over steps.

We report Uncertainty scores in Table 5. GEMINI-3 PRO models uncertainty better than GPT-5.2 in both text- and vision-based settings. These results help explain the information gain and cognitive map trends in Figure 6. GPT-5.2 achieves higher initial information gain (i.e., it ramps up faster), likely because it quickly commits to an explore-the-doors strategy. However, it generalizes poorly to unobserved regions, reflected by the subsequent plateau in Figure 6: additional steps yield little marginal gain. In contrast, although GEMINI-3 PRO improves more slowly at the beginning, its cognitive map accuracy continues to increase with exploration, suggesting it keeps collecting useful evidence and progressively resolving uncertainty.

- 5.3 BELIEF REVISION TASK

Spatial intelligence requires not only mapping static environments but also maintaining beliefs under non-stationarity. Inspired by false belief protocols in developmental psychology (Wimmer & Perner, 1983; Baron-Cohen et al., 1985) and spatial belief revision (Knauff et al., 2013), we introduce a dynamic perturbation task to probe the agent’s ability to discard obsolete priors and reintegrate new evidence.

Task Protocol. Following the initial exploration phase, we introduce a discrete environmental shift: a subset of k = 4 objects are stochastically relocated or reoriented. The agent, retaining its memory (exploration history), must actively re-explore the environment to identify the state changes. This requires the agent to detect conflicts between its internal belief state and new sensory observations.

Metrics. We evaluate performance along four complementary axes:

- • Identification Accuracy (F1): How precisely the agent pinpoints which objects changed. We compute the F1 score for detecting the subset of objects whose position or orientation shifted.
- • Average Steps: How efficiently the agent revises its beliefs to completion. We report Total Steps needed to identify all changes, and Redundancy Steps, defined as the number of steps taken after the last changed object has been observed. Ideally, Redundancy → 0, indicating the agent recognizes when updating is complete.
- • Belief Correctness: How accurate the updated beliefs are on the changed subset. We compute correctness as in §5.1, but restrict evaluation to changed objects to isolate the fidelity of reexploration.
- • Belief Inertia: Whether updating remains systematically biased toward obsolete priors. To quantify attraction back to pre-shift beliefs, we test whether the residual error of the updated

belief aligns with the direction of the old belief. For each shifted object i, let boldi denote the pre-shift belief, bnewi the post-revision belief, and ginew the post-shift ground truth. Define the prior-offset and post-revision error vectors: vi = boldi − ginew,ei = bnewi − ginew. We define positional inertia as

e⊤i vi ∥ei∥∥vi∥ + ϵ Directional alignment (cos θi)

∥bnewi − boldi ∥2 2σ2

sposi =

·exp −

Proximity weight (wi)

#### .

Here cosθi is large when the remaining error after updating still points toward the obsolete location, while wi downweights such alignment when the belief has moved far from boldi . We set σ to a dynamic noise scale: the RMS localization error on the first re-observed unchanged objects during re-exploration; ϵ ensures numerical stability. Under unbiased updating, E[sposi ] ≈ 0, whereas sposi > 0 indicates systematic pull toward the obsolete prior. For orientation shifts, we measure inertia via sorii = 1 ϕnewi = ϕoldi , where ϕ denotes the predicted orientation. It flags failures to overwrite the obsolete facing direction.

Table 7 corroborates the modality gap observed in previous sections: vision-based agents significantly underperform their text-based counterparts. This performance drop is characterized by increased exploration redundancy and lower accuracy in identifying changed objects. Notably, while belief inertia persists across both modalities, it is markedly more severe in vision-based agents, particularly regarding object orientation. Vision models frequently fail to overwrite their initial spatial memory, persisting with obsolete facing estimates despite new visual evidence. This also suggests that fine-grained orientation estimation remains a critical bottleneck for visual spatial reasoning.

### Key Findings: Vision Deficiencies & Belief Inertia

- • Vision-based Revision Failures: Vision agents suffer from excessive exploration redundancy and poor accuracy in identifying object shifts.
- • Belief Inertia: Agents, especially vision-based ones, persist in obsolete spatial coordinates despite new observations.

all red. ori. pos. ori. pos. ori. pos.

Belief Correctness (%) ↑

Belief Inertia (%) ↓

Methods Avg. Steps ↓ Identification (%) ↑

Text-based World GPT-5.2 6.92 0.55 97.9 98.4 89.5 69.7 5.5 12.5 GEMINI-3 PRO 7.79 0.18 98.7 98.8 91.8 72.9 7.9 5.7

Vision-based World

GPT-5.2 13.06 6.20 14.3 68.0 16.7 42.9 68.9 34.7 GEMINI-3 PRO 10.29 3.23 23.9 82.5 30.3 63.1 51.1 14.4

Table 7: Belief updating under environmental shifts. After relocating/reorienting k=4 objects, we evaluate change identification, re-exploration cost (including redundancy (red.)), and belief correctness/update in text- vs. vision-worlds. Vision agents require more redundant steps and show severe orientation inertia, failing to overwrite obsolete facing beliefs despite new evidence.

- 6 RELATED WORK

Passive Spatial Reasoning. Early paradigms treat spatial reasoning as static inference: given a textual description, agents answer relational queries (Weston et al., 2015; Shi et al., 2022; Mirzaee et al., 2021; Li et al., 2024). Other benchmarks probe understanding from a single image, asking for relative directions, topological relations, or metric attributes (Ma et al., 2024; Deng et al., 2025; Cheng et al.,

- 2024; Chen et al., 2024; Liao et al., 2024; Kamath et al., 2023). Multi-view and video benchmarks raise difficulty by requiring cross-view integration, egocentric–allocentric conversion, and temporal consistency (Yang et al., 2025c; Xu et al., 2025; Wu et al., 2025; Yeh et al., 2025; Gholami et al.,
- 2025; Zhou et al., 2025b). Recent works explicitly adopt cognitive maps: VSI-Bench (Yang et al., 2025a) shows map formation improves video QA, and MindCube (Yin et al., 2025) demonstrates that predicting layouts boosts multi-view reasoning. While informative, these benchmarks remain disembodied, as agents reason only over pre-collected trajectories.

Active Exploration for Spatial Understanding. Research has also examined agents that actively explore, but their exploration is usually tied to task-specific goals rather than building a general spatial belief. Embodied question answering benchmarks evaluate agents by whether they can gather evidence to answer questions (Das et al., 2018; Gordon et al., 2018; Majumdar et al., 2024; Ginting et al., 2025; Ren et al., 2024). Instruction-following settings extend household tasks to long horizons and realistic scenes, often with dialog or language grounding (Shridhar et al., 2020b; Kim et al., 2024; Shridhar et al., 2020a; Puig et al., 2018; Padmakumar et al., 2022; Gao et al., 2022). Navigation benchmarks stress path execution and generalization across diverse environments (Anderson et al., 2018; Jain et al., 2019; Ku et al., 2020; Krantz et al., 2020; Nguyen & III, 2019; Wang et al., 2024; Zhao et al., 2025). Spatial reference tasks focus on grounding natural-language descriptions in embodied search (Qi et al., 2019; Zhou et al., 2025a), and manipulation (Jiang et al., 2023; Mees et al., 2022; Srivastava et al., 2022; Wu et al., 2023). While existing benchmarks incorporate active perception, they largely rely on task-driven foraging. This paradigm conflates the efficiency of environmental exploration with downstream task performance, often fostering brittle spatial representations that lack generalizability (Bonawitz et al., 2011). Beyond the above task-driven active exploration, EXCALIBURZhu et al. (2023) also considers task-agnostic exploration, but its RL training can induce goal leakage and encodes maps implicitly in policy weights. In contrast, we study zero-shot foundation-model agents with no environment-specific training for task-agnostic exploration, emphasizing exploration efficiency via minimal-cost uncertainty reduction (rather than coverage), and evaluating not only task success but also the belief construction process via explicit belief probing.

- 7 CONCLUSIONS

We introduce THEORY OF SPACE, which asks whether foundation models can function as spatial agents under partial observability: not merely answering questions from fixed views, but actively acquiring information through self-directed exploration to construct, revise, and exploit an internal spatial belief. Building on this framing, we contribute a new evaluation paradigm centered on task-agnostic active exploration, downstream spatial tasks for belief exploitation assessment, and

explicit probing of internal beliefs via cognitive-map externalization. We implement THEORY OF SPACE in a multimodal environment that instantiates parallel text- and vision-based worlds, enabling controlled diagnosis of failures across symbolic versus perceptual observation streams. A key strength of this design is that it makes spatial belief measurable rather than implicit. By requiring models to externalize evolving cognitive maps and uncertainty over unobserved regions, THEORY OF SPACE evaluates more than end task accuracy: it reveals the correctness, internal consistency, and temporal dynamics of belief formation, and quantifies how localized mistakes propagate into global map corruption over time. Empirically, active exploration is a major bottleneck: end-task performance drops and exploration is less efficient than passive viewing, with the gap widening as room complexity increases. Belief probes make these error sources explicit: in vision, perception error often appears early, and models also exhibit belief instability, where correct information is later overwritten or forgotten, cascading into inconsistencies and lower map fidelity. Finally, when environments change and previously held beliefs must be revised, models exhibit strong belief inertia. They fail to overwrite obsolete priors, and this inertia is especially pronounced for vision-based models, particularly for orientation and facing updates. Taken together, THEORY OF SPACE reframes spatial evaluation from “can the model answer?” to “can the model build and maintain a coherent, revisable spatial world model through efficient information gathering?” We hope this benchmark and its belief-centric measurements provide a foundation for developing models with (i) uncertainty-aware and efficient exploration policies, (ii) robust state/belief maintenance under long horizons, and (iii) reliable mechanisms for revising beliefs when the world changes.

REFERENCES

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko S¨underhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visuallygrounded navigation instructions in real environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3674–3683, 2018.

Anthropic. System card: Claude sonnet 4.5. https://assets.anthropic.com/ m/12f214efcc2f457a/original/Claude-Sonnet-4-5-System-Card.pdf, September 2025. System card (PDF).

Shuai Bai, Yuxuan Cai, Ruizhe Chen, ..., and Ke Zhu. Qwen3-vl: The next generation multimodal llm from qwen / alibaba cloud. arXiv preprint, 2025. URL https://arxiv.org/abs/2511. 21631.

Simon Baron-Cohen, Alan M Leslie, and Uta Frith. Does the autistic child have a “theory of mind”? Cognition, 21(1):37–46, 1985.

Elizabeth Bonawitz, Patrick Shafto, Hyowon Gweon, Noah D Goodman, Elizabeth Spelke, and Laura Schulz. The double-edged sword of pedagogy: Instruction limits spontaneous exploration and discovery. Cognition, 120(3):322–330, 2011.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455–14465, 2024.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.

Elizabeth R Chrastil and William H Warren. Active and passive contributions to spatial learning. Psychonomic bulletin & review, 19(1):1–23, 2012.

Elizabeth R Chrastil and William H Warren. Active and passive spatial learning in human navigation: acquisition of survey knowledge. Journal of experimental psychology: learning, memory, and cognition, 39(5):1520, 2013.

Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. Embodied question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1–10, 2018.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects, 2022. URL https://arxiv.org/abs/2212.08051.

Nianchen Deng, Lixin Gu, Shenglong Ye, Yinan He, Zhe Chen, Songze Li, Haomin Wang, Xingguang Wei, Tianshuo Yang, Min Dou, et al. Internspatial: A comprehensive dataset for spatial reasoning in vision-language models. arXiv preprint arXiv:2506.18385, 2025.

Chuang Gan, Jeremy Schwartz, Seth Alter, Damian Mrowca, Martin Schrimpf, James Traer, Julian De Freitas, Jonas Kubilius, Abhishek Bhandwaldar, Nick Haber, Megumi Sano, Kuno Kim, Elias Wang, Michael Lingelbach, Aidan Curtis, Kevin Feigelis, Daniel M. Bear, Dan Gutfreund, David Cox, Antonio Torralba, James J. DiCarlo, Joshua B. Tenenbaum, Josh H. McDermott, and Daniel L. K. Yamins. Threedworld: A platform for interactive multi-modal physical simulation, 2021. URL https://arxiv.org/abs/2007.04954.

Xiaofeng Gao, Qiaozi Gao, Ran Gong, Kaixiang Lin, Govind Thattai, and Gaurav S. Sukhatme. Dialfred: Dialogue-enabled agents for embodied instruction following. IEEE Robotics and Automation Letters, 7(4):10049–10056, 2022. Also available as arXiv:2202.13330.

Mohsen Gholami, Ahmad Rezaei, Zhou Weimin, Yong Zhang, and Mohammad Akbari. Spatial reasoning with vision-language models in ego-centric multi-view scenes. arXiv preprint arXiv:2509.06266, 2025.

Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Reinke, Bandi Jai Krishna, Navid Kayhani, Oriana Peltzer, David D. Fan, Amirreza Shaban, Sung-Kyun Kim, Mykel J. Kochenderfer, Ali akbar Agha-mohammadi, and Shayegan Omidshafiei. Enter the mind palace: Reasoning and planning for long-term active embodied question answering, 2025. URL https: //arxiv.org/abs/2507.12846.

Google. Gemini 3 pro: Model card. https://storage.googleapis.com/ deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf, November 2025. Model card (GA/preview update for Gemini 3 Pro).

Daniel Gordon, Aniruddha Kembhavi, Mohammad Rastegari, Joseph Redmon, Dieter Fox, and Ali Farhadi. Iqa: Visual question answering in interactive environments. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4089–4098, 2018.

Torkel Hafting, Marianne Fyhn, Sturla Molden, May-Britt Moser, and Edvard I Moser. Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052):801–806, 2005.

Richard Held and Alan Hein. Movement-produced stimulation in the development of visually guided behavior. Journal of comparative and physiological psychology, 56(5):872, 1963.

Vihan Jain, Gabriel Magalh˜aes, Alexander Ku, Ashish Vaswani, Eugene Ie, and Jason Baldridge. Stay on the path: Instruction fidelity in vision-and-language navigation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019.

Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. In Proceedings of the 40th International Conference on Machine Learning (ICML), 2023. arXiv preprint arXiv:2210.03094.

Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s” up” with vision-language models? investigating their struggle with spatial reasoning. arXiv preprint arXiv:2310.19785, 2023.

Taewoong Kim, Cheolhong Min, Byeonghwi Kim, Jinyeon Kim, Wonje Jeung, and Jonghyun Choi. Realfred: An embodied instruction following benchmark in photo-realistic environments. In ECCV, 2024.

Markus Knauff, Leandra Bucher, Antje Krumnack, and Jelica Nejasmic. Spatial belief revision. Journal of Cognitive Psychology, 25(2):147–156, 2013.

Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In European Conference on Computer Vision, pp. 104–120. Springer, 2020.

Alexander Ku, Peter Anderson, Roma Patel, Eugene Ie, and Jason Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.

Fangjun Li, David C Hogg, and Anthony G Cohn. Reframing spatial reasoning evaluation in language models: A real-world simulation benchmark for qualitative reasoning. arXiv preprint arXiv:2405.15064, 2024.

Manling Li, Shiyu Zhao, Qineng Wang, Kangrui Wang, Yu Zhou, Sanjana Srivastava, Cem Gokmen, Tony Lee, Li Erran Li, Ruohan Zhang, Weiyu Liu, Percy Liang, Li Fei-Fei, Jiayuan Mao, and Jiajun Wu. Embodied agent interface: Benchmarking llms for embodied decision making, 2025. URL https://arxiv.org/abs/2410.07166.

Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quantitative spatial reasoning in large vision-language models. arXiv preprint arXiv:2409.09788, 2024.

Wufei Ma, Haoyu Chen, Guofeng Zhang, Yu-Cheng Chou, Celso M de Melo, and Alan Yuille. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. arXiv preprint arXiv:2412.07825, 2024.

Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16488–16498, 2024.

Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3), 2022. Also available as arXiv:2112.03227.

Roshanak Mirzaee, Hossein Rajaby Faghihi, Qiang Ning, and Parisa Kordjmashidi. Spartqa:: A textual question answering benchmark for spatial reasoning. arXiv preprint arXiv:2104.05832, 2021.

Daniel R. Montello. A new framework for understanding the acquisition of spatial knowledge in large-scale environments. Spatial and temporal reasoning in geographic information systems, pp. 143–154, 1998.

Khanh Nguyen and Hal Daum´e III. Help, anna! visual navigation with natural multimodal assistance via retrospective curiosity-encouraging imitation learning. In arXiv preprint arXiv:1909.01871, 2019.

John O’Keefe and Jonathan Dostrovsky. The hippocampus as a spatial map: preliminary evidence from unit activity in the freely-moving rat. Brain research, 1971.

OpenAI. Gpt-5.2 system card. https://cdn.openai.com/pdf/ 3a4153c8-c748-4b71-8e31-aecbde944f8d/oai_5_2_system-card.pdf, August 2025. System card.

Aishwarya Padmakumar, Jesse Thomason, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gokhan Tur, and Dilek Hakkani-Tur. Teach: Task-driven embodied agents that chat. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 2017–2025, 2022.

Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. Virtualhome: Simulating household activities via programs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. URL https://arxiv. org/abs/1806.07011.

Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, and Anton van den Hengel. Reverie: Remote embodied visual referring expression in real indoor environments. arXiv preprint arXiv:1904.10151, 2019.

Allen Z Ren, Jaden Clark, Anushri Dixit, Masha Itkina, Anirudha Majumdar, and Dorsa Sadigh. Explore until confident: Efficient exploration for embodied question answering. arXiv preprint arXiv:2403.15941, 2024.

Zhengxiang Shi, Qiang Zhang, and Aldo Lipani. Stepgame: A new benchmark for robust multihop spatial reasoning in texts. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pp. 11321–11329, 2022.

Mani Shridhar, Roozbeh Mottaghi, Yonatan Bisk, Luke Zettlemoyer, and Dieter Fox. Alfworld: Aligning text and embodied environments for interactive task learning. arXiv preprint arXiv:2010.03768, 2020a.

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10740–10749, 2020b.

Alan W. Siegel and Sheldon H. White. The development of spatial representations of large-scale environments. Advances in Child Development and Behavior, 10:9–55, 1975.

Sanjana Srivastava, Chengshu Li, Michael Lingelbach, Roberto Mart´ın-Mart´ın, Fei Xia, Kent Elliott Vainio, Zheng Lian, Cem Gokmen, Shyamal Buch, Karen Liu, Silvio Savarese, Hyowon Gweon, Jiajun Wu, and Li Fei-Fei. Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments. In Aleksandra Faust, David Hsu, and Gerhard Neumann (eds.), Proceedings of the 5th Conference on Robot Learning, volume 164 of Proceedings of Machine Learning Research, pp. 477–490. PMLR, 08–11 Nov 2022.

Holly A Taylor and Barbara Tversky. Spatial mental models derived from survey and route descrip-

tions. Journal of Memory and language, 31(2):261–292, 1992. Edward C Tolman. Cognitive maps in rats and men. Psychological review, 55(4):189, 1948. Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu,

Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Zhaowei Wang, Hongming Zhang, Tianqing Fang, Ye Tian, Yue Yang, Kaixin Ma, Xiaoman Pan, Yangqiu Song, and Dong Yu. Divscene: Benchmarking lvlms for object navigation with diverse scenes and objects. arXiv preprint arXiv:2410.02730, 2024.

Jason Weston, Antoine Bordes, Sumit Chopra, Alexander M Rush, Bart Van Merri¨enboer, Armand Joulin, and Tomas Mikolov. Towards ai-complete question answering: A set of prerequisite toy tasks. arXiv preprint arXiv:1502.05698, 2015.

Heinz Wimmer and Josef Perner. Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children’s understanding of deception. Cognition, 13(1):103–128, 1983.

Haoning Wu, Xiao Huang, Yaohui Chen, Ya Zhang, Yanfeng Wang, and Weidi Xie. Spatialscore: Towards unified evaluation for multimodal spatial understanding. arXiv preprint arXiv:2505.17012, 2025.

Yue Wu, Xuan Tang, Tom M. Mitchell, and Yuanzhi Li. Smartplay: A benchmark for llms as intelligent agents. arXiv preprint arXiv:2310.01557, 2023.

Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models. arXiv preprint arXiv:2505.17015, 2025.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10632–10643, 2025a.

Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, and Tong Zhang. Embodiedbench: Comprehensive benchmarking multi-modal large language models for visiondriven embodied agents, 2025b. URL https://arxiv.org/abs/2502.09560.

Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multi-image spatial intelligence. arXiv preprint arXiv:2505.23764, 2025c.

Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, and Yi Ma. Seeing from another perspective: Evaluating multi-view understanding in mllms. arXiv preprint arXiv:2504.15280, 2025.

Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. arXiv preprint arXiv:2506.21458, 2025.

Yong Zhao, Kai Xu, Zhengqiu Zhu, Yue Hu, Zhiheng Zheng, Yingfeng Chen, Yatai Ji, Chen Gao, Yong Li, and Jincai Huang. Cityeqa: A hierarchical llm agent on embodied question answering benchmark in city space. arXiv preprint arXiv:2502.12532, 2025.

Zhipu AI Team. Glm-4.6v: Native multimodal foundation model. Technical report / model release,

2025. URL https://z.ai/blog/glm-4.6v. Available online; native multimodal vision + reasoning model (128k context) among open-source LLMs.

Enshen Zhou, Jingkun An, Cheng Chi, Yi Han, Shanyu Rong, Chi Zhang, Pengwei Wang, Zhongyuan Wang, Tiejun Huang, Lu Sheng, and Shanghang Zhang. Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. arXiv preprint arXiv:2506.04308, 2025a.

Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, and Achuta Kadambi. Vlm4d: Towards spatiotemporal awareness in vision language models. arXiv preprint arXiv:2508.02095, 2025b.

Hao Zhu, Raghav Kapoor, So Yeon Min, Winson Han, Jiatai Li, Kaiwen Geng, Graham Neubig, Yonatan Bisk, Aniruddha Kembhavi, and Luca Weihs. Excalibur: Encouraging and evaluating embodied exploration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14931–14942, 2023.

APPENDIX

- A TECHNICAL DETAILS A.1 BENCHMARK CONSTRUCTION

We expose the ToS world as a Gym-like interface (Brockman et al., 2016): agents interact in discrete steps under partial observability at a resolution of 384×384 to construct and revise an internal spatial belief, which we later exploit in evaluation tasks. Scenes are procedurally generated multi-room layouts on an N×M grid with n named indoor objects (each with integer (x,y) and heading in {N,E,S,W}) and a randomized agent spawn pose. We restrict multi-room layouts to a tree topology: the room–adjacency graph is connected and acyclic (no loops).

Text-based World At each step, OBSERVE returns a symbolic snapshot of objects in the current room within a 90◦ forward FOV. For every visible object we provide discretized egocentric direction (e.g., front-left) and distance bins (e.g., near/mid/far), plus object identity and facing when determinable. Egocentric observations are rendered with a 90-degree field of view (FOV), discretized into angular and distance bins as specified in Figure 7a. Visibility is room-bounded; doorways act as transparent portals only when the agent stands in them, enabling dual-room visibility. Optional noise modules perturb bins for ablations.

Vision-based World We procedurally generate scenes in a 3D simulator with two controllable parameters: the level (number of rooms) and the object count per room. Objects are drawn from a library of 293 distinct 3D models, grouped into 6 categories and 37 subtypes, primarily everyday household items (see Figure 7b). To ensure diversity, each object type appears at most once in a given scene.

[Figure 129]

###### Model Overview

Furniture Transportation Utensils Appliances Toys Clothing

9%

9%

31%

14%

18% 19%

(a) Field of view (FOV) specification for the agent in our tasks. The FOV spans 90° in front of the agent and is divided into angular bins (e.g., front, front-slight left, front-left) and distance ranges (near [0,2], mid [2,5], far [5,10]). This egocentric perception defines how spatial relations are observed and reported.

(b) Distribution of all 3D models used in our vision tasks.

Created with Try for free at moqups.com

### Figure 7: Demonstration figures for FOV and 3D model distribution

For task setup, we additionally generate instructional (Figure 8) and orientation (Figure 9) images that serve as references for the agent in vision-world. We include both images in the vision prompt. Object placement follows validity constraints (e.g., collision avoidance, minimum spacing), and random seeds control reproducibility across environments.

[Figure 130]

- Figure 8: Example of distance cues in the vision prompt. The colored cylinders illustrate objects placed at different distances from the agent: yellow at 2 m, blue at 1 m, red at 2 m, and green at 3 m, providing calibration for mapping visual observations to discretized distance bins.

[Figure 131]

- Figure 9: Object appearance and orientation cues in the vision prompt. Objects with facing direction are shown from both the front and side views, while objects without inherent orientation are displayed only from the front view. This provides the agent with consistent visual references for recognizing shape and facing.

Information Gain Calculation We use the AC-3 arc-consistency algorithm to maintain, for each object, a domain of feasible grid cells. Initially, every object’s domain spans the entire 20 × 20 map. Each new observation is compiled into unary and binary constraints (e.g., egocentric direction/distance bins, room visibility/occlusion, and ALLDIFFERENT to prevent collisions). When a constraint is

added, AC-3 iteratively prunes any cell in one object’s domain that is unsupported by the domains of related objects, propagating revisions along incident arcs until a fixed point is reached (all arcs are consistent). While AC-3 alone does not guarantee global consistency, in our setting all constraints are derived from a valid trajectory; therefore the ground-truth assignment remains supported and is never pruned, ensuring that domains stay non-empty throughout propagation.

Proxy agents We implement two scripted proxies to provide strong, reproducible baselines.

SCOUT. From its spawn pose, the agent performs a 360° sweep (four cardinal ROTATE+OBSERVE actions) to capture all views at the initial location. It then follows a fixed room-visitation order: upon discovering a doorway, it enters the adjacent room, executes the same sequential sweep, and repeats this “visit–sweep–advance” routine until every room has been observed at least once.

STRATEGIST. The first stage mirrors SCOUT: a panoramic sweep to register all currently visible objects. Thereafter, within the current room the agent maintains, for each object, a set of feasible positions (“domain”) induced by accumulated observations. At each turn it: (i) selects the object with the largest remaining domain (highest positional uncertainty); (ii) moves to a viewpoint that best constrains this object (e.g., near it or along a sightline that intersects the most candidate cells); (iii) at that viewpoint, orients to test pairwise relations: it computes unresolved pairwise directions between the target object and all others in the room, identifies the direction bin with the highest outstanding count, and OBSERVEs in that orientation first. The procedure iterates until all objects in the room are resolved (domains shrink to singletons), then proceeds to the next unvisited room and repeats.

Prompts We show the detailed designs of our prompts for exploration in Figure 10, evaluation prompts in Figure 11, cognitive map prompts in Figure 12, and top-down view for uncertainty modeling in Figure 13.

Observation Instructions

| | |
|---|---|
|[Relationship] bearing in degrees; distance is Euclidean. Use binned labels. Bearing is a degree in [-180, 180]; 0° is front. +: clockwise, -: counterclockwise.<br><br>[Orientation]<br><br>-forward/backward/right/left (ego) or north/east/south/west (allo).<br>-When agent faces north: forward<br><br><br>= north, right = east, etc.<br><br>-Gate's orientation: report wall position (e.g., 'on left wall'). [Binned relationship reporting] EgoFront (egocentric, object-to-agent); Cardinal (object-to-object). Egocentric angle bins (0° is front):<br><br>-[-45°,-22.5°)→front-left,<br>-[-22.5°,0°)→front-slight-left,<br>-0°→front,<br>-(0°,22.5°]→front-slight-right,<br>-(22.5°,45°]→front-right,<br>-otherwise→beyond-fov.<br><br><br>Cardinal angle bins (45° each):<br><br>-(-22.5°,22.5]→north,<br>-(22.5°,67.5]→north east,<br>-(67.5°,112.5]→east,<br>-(112.5°,157.5]→south east,<br>-(157.5°,202.5]→south,<br>-(202.5°,247.5]→south west,<br>-(247.5°,292.5]→west,<br>-(292.5°,337.5]→north west,<br><br><br>Distance bins:<br><br>-=0→same distance,<br>-(0,2]→near, (2,4]→mid distance,<br>-(4,8]→slightly far,<br>-(8,16]→far,<br>-(16,32]→very far,<br>-32→extremely far.<br>| |
| | |

Action Instructions

| | |
|---|---|
|You can jump to objects within and across rooms, turn, and observe.<br><br>When you are at a door, you can see objects from both connected rooms (within FOV).<br><br>Available Actions: {actions}<br><br>Action Grammar (HARD CONSTRAINT): Actions: [ <M>* <F> ] <M> = "JumpTo(OBJ)" | "Rotate(DEG)" <F> = "Observe()" | "Query(OBJ)" | "Term()" Constraints:<br><br>- Zero, one or more <M>. No JumpTo at first step.<br>- Exactly one <F>, and it must be the final action.<br>- No more than one Observe().<br>- Term() may appear only alone.<br>- Any violation is invalid.<br><br>Examples: {examples} Rules:<br><br>- Observe action only reports from your current position and facing direction. If you jump multiple times, the final Observe() action gives the view only from your last position.<br>- Actions execute in order. Field of view: {field_of_view}°.<br><br><br>Observe and Query action have costs: {costs}| |
| | |

###### Exploration Prompt (text)

Goal & Setting Prompt Exploration Prompt (vision)

Exploration Prompt - False Belief

| | |
|---|---|
|You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y)| |
| | |

| | |
|---|---|
|[Goal] Minimize total COST while building a complete and accurate map of the environment.<br><br>[Multi-room rules] (may exist multiple rooms):<br><br>- Your vision is confined to your current room.<br>- Doors block vision between rooms.<br>- Exception: When located in a doorway, door is open and invisible, you can see into both connected rooms.<br>- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.<br>| |
| | |

Here is an example of your observation: blue cylinder 1 m straight ahead; red cylinder 2 m straight ahead; yellow cylinder 2 m at 45° to your front-left; green cylinder 3 m at 22.5° to your front-slight-right: {image_placeholder}

You have returned to the initial position and face north. There are {n_changes} objects in the room that have been changed (position or orientation). Note one object is either moved or rotated, not both. Goal: Explore the room again and identify which objects have been changed and how with minimum costs. Use the same action set as the exploration phase. You must use the Term(changes="...") action to submit your answer and terminate. Format: Term(changes="object1: change_type, object2: change_type") Example: Term(changes="apple: position, chair: orientation") You have a maximum of {max_steps} steps.

Format & Rules Instructions

coordinates.

| | |
|---|---|
|[Rules]<br><br>- Achieve complete coverage with the fewest steps;<br>- Prefer actions that reveal more unknowns; avoid redundancy<br>- FOV is 90°, you can NOT see objects outside your FOV.<br>- Track your current and initial pose<br>| |
| | |

Exploration Prompt (vision)

| | |
|---|---|
|You are a spatial reasoner in a 3D simulated environment. The world is rendered in 3D but abstracted into a discrete 2D grid of size N×M. Every entity, including yourself, is represented by integer coordinates (x, y) on this grid.| |
| | |

The image shows all objects in the room. Each tile is numbered (1-N) in the top-left, matching the object order in the room layout. For items with a facing direction, two copies are shown side-by-side: the left copy has its front facing the camera; the right copy has its front facing left. Items without a meaningful facing direction are shown once. {image_placeholder}

### Figure 10: Exploration prompts

## Route Survey

Pairwise Direction

Allocentric Map

You return to your starting position and face north. From a Top-Down map, describe where {obj_name} is relative to {anchor_name}.

Treat your starting position as the origin (0, 0) while facing north. Report allocentric coordinates using (x right/east, y up/north). Objects: {object_list}.

Answer format: <cardinal direction>, <distance> Example: north-west, near

Answer format: (x0, y0); (x1, y1); ... in the same order. Example: (1, 0); (-2, 3); (0, -1)

Perspective Taking

Now you jump to {anchor_name}'s direction, facing its direction. Describe where {obj_name} is relative to you.

Mental Rotation

You return to your starting position and face north. You will perform a full 360-degree rotation by continuously turning {turn_direction} in place. Assume all walls are removed (you can see through walls), so every object is visible.

Answer format: <ego direction>, <distance> Example: front-left, near

Perspective Determine

Focus on this set of objects: {object_pool}. List them in the exact order they appear directly ahead while you rotate. If two objects share a bearing, place the nearer one first.

Now you jump to an object's position, facing its direction. You observe that {observation}. Which object are you standing at?

Answer format: <object_name> Example: lamp

Answer format: <object_name1>, <object_name2>, ... Example: mug, sofa, plant

Action2View

You return to your starting position and face north. You will execute the following action sequence: {actions}

Location2View

{origin_instruction}You move to {loc} and face {direction}. What is the egocentric relation of {target}?

After executing the actions, what is the ego relation of {target} relative to you?

Answer format: <ego direction>, <distance> Example: front, near

Answer format: <direction>, <distance> Example: front, near

View2Action

You return to your starting position and face north. Then you have executed an action sequence and changed to a new location and facing direction. You observe the following: {final_obs}

View2Location

You move to a new location and face {orientation}. {observations} {origin_instruction}What is your new 2D coordinate (x, y)?

What action sequence led to this final view? The action sequence must be valid and only contain move actions.

Answer format: (x, y) Example: (2, -1)

Answer format: <sequence of move actions> Example: JumpTo(lamp), Rotate(90)

### Figure 11: Evaluation prompt design. We show the prompt for each evaluation task.

###### Cognitive Map Prompt - Base

###### Cognitive Map Prompt - Base (False Belief)

###### Cognitive Map Prompt - Uncertainty map (text)

### Fog Probe {symbol_def}

[Cognitive Map (JSON)] Represent the scene as a JSON map.

[Cognitive Map (JSON)] Represent the scene as a JSON map.

The map displays candidate points labeled A-Z. Select the points that are located in unexplored/unobserved regions.

[Schema (shared)]

[Schema (shared)]

- - position: [x, y] integers
- - facing: "north|south|east|west" (global) or "+x|-x|+y|-y" (local/rooms)

[General rules (shared)]

- - MUST include facing key if the object has facing direction.

- - position: [x, y] integers
- - facing: "north|south|east|west" (global) or "+x|-x|+y|-y" (local/rooms)

[General rules (shared)]

- - Include only observed objects.
- - MUST include facing key if the object has facing direction.

Map: {symbol_map}

Example: {{

"unexplored": ["A", "C"] }}

| | |
|---|---|
| | |

###### Cognitive Map Prompt - Uncertainty map (vision)

### Fog Probe The map visualizes the environment:

###### Cognitive Map Prompt - Global

###### Cognitive Map Prompt - Local

###### Cognitive Map Prompt - Global (False Belief)

- - **North is Up.**
- - **Coordinates**: (0, 0) is at the bottom-left. X points Right (East), Y points Up (North).
- - **Grey points**: All positions with integer coordinates.
- - **Red letters (A-Z)**: Candidate points to evaluate.
- - **Blue point**: Your current position.

[Local Cognitive Map]

[Global Cognitive Map]

[Global Cognitive Map]

- - Structure: include an "objects" dict; each object's position and facing are relative to the agent at time of writing.
- - Frame: must include "origin":"agent". Always keep in mind that the origin is the agent's current position and orientation.
- - +y: facing forward
- - when facing +y: +x -> right, -x -> left, -y -> backward
- - All positions/facings relative to this frame.
- - Content: "objects" dict; include all objects and doors in your current field of view; exclude agent.
- - Facing: use "+x|-x|+y|-y" (local axes).

- - Grid: concise global map on an N×M grid.
- - Frame: origin [0,0] is your initial position; your initial facing direction is north.
- - Content: include ALL objects and gates; include the agent
- - Facing: use "north|south|east|west" (cardinal direction only).

- - Grid: concise global map on an N×M grid.
- - Frame: origin [0,0] is your initial position; your initial facing direction is north.
- - Content: include all observed objects and gates; include the agent
- - Facing: use "north|south|east|west" (cardinal direction only).

NOTE: you must include all objects and gates.

Select the candidate points (Red letters) that are located in

[Example] {

**unexplored/unobserved** regions. Map: {image_map} Example: {{

[Example] {

"agent": {"position": [2, 3], "facing": "east"},

"agent": {"position": [2, 3], "facing": "east"},

[Example] {

"chair": {"position": [2, 4], "facing": "north"},

"chair": {"position": [2, 4], "facing": "north"},

"origin": "agent", "objects": { "chair": {"position": [0, 1], "facing": "-x"} }

"sofa": {"position": [5, 1], "facing": "west"} }

"sofa": {"position": [5, 1], "facing": "west"} }

"unexplored": ["A", "C"] }}

}

### Figure 12: Belief probing prompt design. We use these prompts to ask the model to output a cognitive map or select unobserved points.

Symbol Map (text) Image Map (vision)

- ● : empty cell
- ● : object
- ● : agent position

[Figure 132]

# : wall

################## #....J.########### #...D..###########

. : empty cell A–K: object

+ : door

- #A.....########### #I.....########### #......########### #..F..L########### ######*########### #......########### #.H....+......#### #......#......#### #.C.G..#......####
- #B.E...#......#### #..K...#......#### ########......#### ##################

* : agent position

##### Figure 13: The symbol map and the image map provide parallel representations of the same environment for text and vision settings in uncertainty probing prompts.

- B EVALUATION SETUPS

To enable a like-for-like comparison between the text and vision settings, we instantiate identical room layouts across modalities. Concretely, we generate 100 evaluation instances with IDs 0–99; for each ID, we use the ID itself as the random seed to drive task sampling in both environments. This seed tying guarantees deterministic layouts and bit-for-bit reproducibility across modalities.

Additional Results We show detailed results for different room settings including two-room and four-room layouts. In both the two-room and four-room settings, we use the same room size and the same number of objects per room as in the three-room setting. For the four-room setting, we connect the main room with all the others. We evaluate GPT-5.2 and GEMINI-3 PRO, the two best-performing models. Additionally, we tested higher resolution, but found no performance gain. Table 8 and 9 report passive and active performance of the two-room setting. Table 10 and 11 report passive and active performance of the three-room setting. As the number of rooms increases, exploration cost rises accordingly. The results also underscore the importance of efficient exploration: in the four-room setting, which demands more strategic exploration, the gap between active and passive performance becomes substantially larger.

directionpersp.takeperc.decact2viewview2act alloc.mapment.rotloc2viewview2loc

Static (S) Dynamic (D) Static (S) Dynamic (D)

###### Methods Route Survey Avg.

Vision-based World Proprietary Models

GPT-5.2 39.2 37.3 63.3 53.8 58.3 68.2 92.7 52.3 68.6 59.3 GEMINI-3 PRO 57.8 33.9 53.8 48.5 58.7 64.6 83.3 54.7 69.8 58.3

Text-based World Proprietary Models

GPT-5.2 85.3 92.0 99.0 90.0 83.0 97.2 99.7 89.5 95.2 92.3 GEMINI-3 PRO 88.2 86.7 91.7 87.3 79.3 90.1 92.7 81.5 82.9 86.7

- Table 8: Exploitation Performance (%) via Passive Observations under two rooms settings.

directionpersp.takeperc.dec.act2viewview2act alloc.mapment.rotloc2viewview2loc

Static (S) Dynamic (D) Static (S) Dynamic (D)

Methods Avg.cost Route Survey Avg.

Vision-based World Proprietary Models

GPT-5.2 10.8 41.3 36.2 48.2 49.0 54.7 56.9 72.0 45.2 59.7 51.5 GEMINI-3 PRO 6.6 51.7 36.3 63.0 47.2 56.0 63.4 85.0 50.3 67.5 57.8

Text-based World Proprietary Models

GPT-5.2 6.2 68.7 67.3 90.0 76.8 64.0 83.4 92.7 73.7 83.7 77.8 GEMINI-3 PRO 6.2 76.0 68.3 89.0 77.2 72.7 83.1 96.0 77.5 86.2 80.6

- Table 9: Exploitation Performance (%) via Active Exploration under two rooms settings.

directionpersp.takeperc.decact2viewview2act alloc.mapment.rotloc2viewview2loc

Static (S) Dynamic (D) Static (S) Dynamic (D)

Methods Route Survey Avg.

Vision-based World Proprietary Models

GPT-5.2 47.0 37.7 59.7 38.3 40.3 60.1 73.7 50.5 65.9 52.6 GEMINI-3 PRO 63.5 35.5 58.7 42.8 43.0 64.4 81.7 48.8 67.4 56.2

Text-based World Proprietary Models

GPT-5.2 83.8 88.2 94.3 86.8 62.7 94.8 93.7 82.0 92.5 86.5 GEMINI-3 PRO 81.2 91.3 96.7 82.2 68.3 76.8 81.3 74.2 79.0 81.2

- Table 10: Exploitation Performance (%) via Passive Observations under four rooms settings.

directionpersp.takeperc.dec.act2viewview2act alloc.mapment.rotloc2viewview2loc

Static (S) Dynamic (D) Static (S) Dynamic (D)

###### Methods Avg.cost Route Survey Avg.

Vision-based World Proprietary Models

GPT-5.2 23.2 41.2 33.2 49.0 30.8 30.7 32.5 49.7 40.5 55.4 40.3 GEMINI-3 PRO 19.7 59.8 34.2 60.3 34.7 46.0 56.8 62.7 44.0 64.8 51.5

Text-based World Proprietary Models

GPT-5.2 16.4 65.3 69.0 74.3 62.8 44.3 66.6 76.3 57.5 77.8 66.0 GEMINI-3 PRO 19.7 76.3 77.2 91.7 73.3 64.3 77.0 83.7 74.0 81.9 77.7

- Table 11: Exploitation Performance (%) via Active Exploration under four rooms settings.

- C ADDITIONAL VISUALIZATION EXAMPLES

We include concrete examples of task formats and answer styles with open-ended, format-constrained outputs in Figure 14.

Cognitive map output by models We visualize the turn-by-turn cognitive maps (in Figures 15 and 16 of GPT-5.2, comparing them against ground-truth maps. The performance is noticeably stronger in text-based environments than in vision-based ones.

Exploration pattern examples by models We include representative trajectories from each model to illustrate the active exploration patterns identified in our analysis, shown in Figure 17, 18, 19, 20, and 21 . These examples highlight how different models manifest recurring exploration behaviors: for instance, GPT-5.2 often adopts a “finding-gate” strategy, rotating until a doorway is detected before moving toward it, while other models more frequently repeat redundant checks. All figures mark the agent’s position and orientation explicitly, with actions annotated beneath each frame and a shared legend provided for each trajectory.

Analysis Platform We also include some demonstrations in Figure 22, 24, 23, 25, and 26 of our designed platform for better analysis

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

Pairwise Direction Q: You return to your starting position and face north. From a Top-Down map, describe where shelf is relative to truck. A: south east, mid distance

Allocentric Map

Q: Treat your starting position as the origin (0, 0) while facing north. Report allocentric coordinates using (x right/east, y up/north). Objects: shelf, truck, lamp.

A: [[12, -1], [10, 1], [0, 4]]

Perspective Taking

Q: Now you jump to backpack's direction, facing its direction. Describe where chair is relative to you.

Mental Rotation

A: front-left, mid distance

Q: You return to your starting position and face north. You will perform a full 360-degree rotation by continuously turning counterclockwise in place. Assume all walls are removed (you can see through walls), so every object is visible. Focus on this set of objects: bike, pan, television. List them in the exact order they appear directly ahead while you rotate. If two objects share a bearing, place the nearer one first.

Perspective Determine

Q: Now you jump to an object's position, facing its direction. You observe that truck is front-left, mid distance, facing backward; shelf is front, mid distance. Which object are you standing at?

A: laptop

A: ['television', 'pan', 'bike']

Action2View

Q: You return to your starting position and face north. You will execute the following action sequence:

Location2View

- 1. Jump to the object at front-right, mid distance.
- 2. Rotate(-90)
- 3. Jump to the object at front-right, mid distance.
- 4. Rotate(-180)

Q: Treat the green door as the new 'origin' (0, 0). You move to (2, -5) and face north. What is the egocentric relation of pan?

After executing the actions, what is the ego relation of bike relative to you?

A: front, mid distance

A: front-right, mid distance

View2Action

Q: You return to your starting position and face north. Then you have executed an action sequence and changed to a new location and facing direction. You observe the following: pan is at front-right, slightly far, facing backward; truck is at front-right, mid distance, facing forward; laptop is at front, mid distance, facing backward What action sequence led to this final view? A: [['rotate', 90], ['jumpto', 'green door'], ['jumpto', 'shelf'], ['rotate', 180]]

View2Location Q: You move to a new location and face north. You observe: pan is at front, mid distance, facing right; truck is at front-right, mid distance, facing left; green door is at front-slight-left, slightly far, on left wall Treat the green door as the new 'origin' (0, 0). A: [2, -5]

- Figure 14: Examples of task formats and answer styles used. Each block illustrates a spatial reasoning task type in our suite (Route-level and Survey-level), including the corresponding input context and an example open-ended answer that must follow a strict output format. In the vision setting, textual scene descriptions in the questions are replaced by rendered observation images.

agent

basket

cabinet

cap

chair

dog

fridge

initial_position

lamp

plant

scooter

shoppingcart

teddybear

television

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Turn 1 - Pred

Turn 2 - Pred

Turn 3 - Pred

Turn 4 - Pred

Turn 5 - Pred

Turn 6 - Pred

Turn 7 - Pred

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

Turn 1 - GT

Turn 2 - GT

Turn 3 - GT

Turn 4 - GT

Turn 5 - GT

Turn 6 - GT

Turn 7 - GT

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

Figure 15: GPT-5.2’s turn-by-turn cognitive map in text world during exploration.

agent bike

cabinet cap

chair initial_position

shelf shoes

suitcase table

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Turn 1 - Pred

Turn 2 - Pred

Turn 3 - Pred

Turn 4 - Pred

Turn 5 - Pred

Turn 6 - Pred

Turn 7 - Pred

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

Turn 1 - GT

Turn 2 - GT

Turn 3 - GT

Turn 4 - GT

Turn 5 - GT

Turn 6 - GT

Turn 7 - GT

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

Figure 16: GPT-5.2’s turn-by-turn cognitive map in vision world during exploration.

[Figure 145]

- Figure 17: Example trajectory illustrating GPT-5.2’s door-finding strategy and systematic sweeping pattern: Upon detecting a door, the agent navigates toward it and executes a strategic rotation to maximize environmental coverage. The process terminates once all target objects have been successfully identified.

[Figure 146]

##### Figure 18: Example trajectory illustrating GPT-5.2’s omission pattern: Observing the door too early may lead the agent to skip the rest of the exploration, causing incomplete environmental discovery.

[Figure 147]

##### Figure 19: Example trajectory illustrating GEMINI-3 PRO’s door-finding strategy and systematic sweeping pattern in vision world: Upon detecting a door, the agent navigates toward it and executes a strategic rotation to maximize environmental coverage. The process terminates once all target objects have been successfully identified.

[Figure 148]

##### Figure 20: Example trajectory illustrating GEMINI-3 PRO’s object sweeping pattern mostly found in text world: Orbit the starting object using it as the pivot point. Randomly select an observed door to jump to a new object, then resume pivoting around the new target in a continuous loop.

[Figure 149]

##### Figure 21: Example trajectory illustrating CLAUDE-4.5 SONNET’s exploration pattern: There is no clear exploration pattern.

[Figure 150]

Figure 22: Platform designed by us for analysis (chart)

[Figure 151]

- Figure 23: Visualization Platform for analysis: Metrics for active exploration in text world

[Figure 152]

- Figure 24: Visualization Platform for analysis: Metrics for active exploration in vision world

[Figure 153]

Figure 25: Visualization Platform for analysis: one turn of active exploration in text-world, including agent’s action and cognitive map.

[Figure 154]

##### Figure 26: Visualization Platform for analysis: one turn of active exploration in vision-world

