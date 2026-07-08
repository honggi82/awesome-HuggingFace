[Figure 1]

## Communicating about Space: Language-Mediated Spatial Integration Across Partial Views

# arXiv:2603.27183v2[cs.CV]1Apr2026

Ankur Sikarwar1,2* Debangan Mishra3* Sudarshan Nikhil3 Ponnurangam Kumaraguru3 Aishwarya Agrawal1,2,4 1Mila – Quebec AI Institute 2Université de Montréal 3IIIT Hyderabad 4Canada CIFAR AI Chair

### Abstract

Humans build shared spatial understanding by communicating partial, viewpoint-dependent observations. We ask whether Multimodal Large Language Models (MLLMs) can do the same, aligning distinct egocentric views through dialogue to form a coherent, allocentric mental model of a shared environment. To study this systematically, we introduce COSMIC, a benchmark for Collaborative Spatial Communication. In this setting, two static MLLM agents observe a 3D indoor environment from different viewpoints and exchange natural-language messages to solve spatial queries. COSMIC contains 899 diverse scenes and 1,250 question– answer pairs spanning five tasks. We find a capability hierarchy, MLLMs are most reliable at identifying shared anchor objects across views, perform worse on relational reasoning, and largely fail at building globally consistent maps, performing near chance, even for frontier models. Moreover, we find thinking capability yields gains in anchor grounding, but is insufficient for higher-level spatial communication. To contextualize model behavior, we collect 250 human–human dialogues. Humans achieve 95% aggregate accuracy, while the best model, Gemini-3-Pro-Thinking, reaches 72%, leaving substantial room for improvement. Moreover, human conversations grow more precise as partners align on a shared spatial understanding, whereas MLLMs keep exploring without converging, suggesting limited capacity to form and sustain a robust shared mental model throughout the dialogue. Our code and data is available at link.

### 1. Introduction

Spatial intelligence extends beyond individual perception. Through communication and collaboration, humans trans-

*Equal contribution

form local observations into shared spatial mental models [6, 10, 15, 20, 30]. Imagine two friends trying to meet at a large park they have never visited. One says, “I am near the lamp post beside the fountain,” and the other replies, “I see a tall tree and the fountain.” Neither has a complete view, but through dialogue they integrate these utterances into a shared spatial model. They ground common landmarks (fountain), infer relative orientation, and progressively refine hypotheses about unseen regions. Language here acts as a scaffold for spatial communication [6, 15], enabling behaviors like frame-of-reference switching, grounding shared objects across views (anchor objects), clarification and repair, and the overall synthesis of complementary partial views.

This ability to build shared spatial models through communication is increasingly central for modern AI systems. As MLLMs are now expected to operate as collaborative partners rather than passive tools, deployed across embodied robotics [28], AR/VR platforms, and multi-user interactive systems [23], spatial reasoning becomes inherently distributed, with no single agent having access to the full environment. Unlike single-agent settings [33–35], these distributed scenarios require agents to communicate observations, reconcile conflicting interpretations, and build a consistent shared model of the environment, all through natural language. Yet despite rapid progress on spatial perception and reasoning benchmarks [17, 33, 35, 37], the capacity of MLLMs for this kind of communication-driven spatial integration remains largely unexplored.

We address this gap with COSMIC, a diagnostic benchmark designed to test whether MLLM agents can build a shared spatial understanding through dialogue. The benchmark places two agents in the same indoor scene from distinct egocentric perspectives, requiring multi-turn dialogue to integrate complementary observations and jointly solve a QA task (Fig. 1). Our key contributions are as follows:

1. We introduce the task of building shared spatial under-

[Figure 2]

- Figure 1. Left: MLLM agents attempt to communicate and build a spatial mental model for answering questions in COSMIC. Right: Answerer and Helper agents integrate distinct egocentric views via communication to answer the question. Humans demonstrate efficient and precise strategies while MLLM agents are more verbose, inefficient, and fail to build and maintain a robust shared mental model.

standing through natural-language dialogue across partial, egocentric views. To support systematic evaluation of this capability, we present COSMIC, a diagnostic benchmark comprising 899 procedurally generated diverse indoor scenes and 1,250 question-answer pairs spanning five tasks evaluating object-level, relation-level, and maplevel spatial reasoning capabilities.

- 2. We conduct zero-shot evaluations of state-of-the-art opensource and proprietary MLLMs, revealing a deterioration in their capabilities from anchor grounding to relational reasoning to cognitive mapping. Through systematic failure mode analysis, we further demonstrate that crossview grounding and geometric reasoning constitute the primary limitations in current MLLMs.
- 3. We collect 250 human–human dialogues and provide a systematic comparison of human and model dialogues in collaborative spatial reasoning, revealing that humans converge rapidly through targeted, information-dense exchanges grounded in precise spatial descriptions, while MLLM agents resort to verbose, exploratory dialogue that remains shallow and fails to build a coherent shared spatial model (see Fig. 1 for an example).

### 2. Related Work

Single-view Spatial reasoning. Early studies revealed that vision–language models such as CLIP and BLIP-VQA strug-

gle with even basic spatial relationships [13]. Subsequent work systematically evaluated MLLMs on spatial reasoning, with benchmarks such as What’sUp [13] and VSR [19] probing basic positional understanding, and SpatialRGPTBench [5] extending evaluation to broader aspects of spatial cognition. More recent benchmarks, including OmniSpatial [12] and [26], further expand the scope to complex realworld scenarios and tasks requiring mental rotation, folding, and navigation. However, these benchmarks are limited to reasoning over single-view.

Multi-view spatial reasoning. Recent benchmarks extend spatial evaluation beyond single views by presenting multiple images of a scene to a single model [14, 16, 33]. MINDCUBE [36] probes spatial mental models through questions involving translations, rotations, and perspectivetaking, while Ego3D-Bench [11] and All-Angles Bench [35] evaluate models on egocentric multi-view observations. Finetuning with reinforcement learning or supervised objectives can partially mitigate model’s limitations [35, 36] on spatial reasoning. All of these benchmarks remain centralized, with a single model reasoning over all available views. COSMIC addresses the fundamentally harder distributed setting where two agents, each with partial observations, are required to reason jointly.

Multi-agent cooperation and communication. Prior work has examined multi-agent LLM and MLLM systems on col-

[Figure 3]

- Figure 2. Overview of the COSMIC benchmark. Each task pair shows the Answerer’s view (left) and the Helper’s view (right), along with the question and options posed to the Answerer.

laborative tasks. Benchmarks such as [1, 18] evaluate how LLMs handle missing information through sustained dialogue to solve textual puzzles. In [22, 32], multimodal agents with complementary observations cooperate to solve games, while [38, 39] explore similar settings in navigation. Other works propose frameworks for improving reasoning and assessing collaborative behavior in communication-based environments [3, 4, 8]. However, multi-agent spatial reasoning under egocentric partial observability remains understudied. Cognitive maps. Cognitive science conceptualizes scene understanding through cognitive maps, internal representations encoding environmental geometry [21, 29]. Recent work investigates whether MLLMs can form such map-like representations [9, 33, 36]. Unlike humans, who routinely build such representations collaboratively from partial observations, these works focus on centralized, single-model settings. COSMIC bridges this gap by studying map-level understanding in a distributed, communication-based multiagent setting.

- 3. COSMIC Benchmark

tions within a 3D indoor room environment. Each agent receives their own egocentric RGB view (IA & IH), providing them with partial but complementary perspectives of the environment. The Answerer is additionally given a question and is required to collaborate with the Helper through multiturn natural-language dialogue, integrating complementary observations across both views to arrive at the correct answer (see Fig. 1).

COSMIC comprises five tasks spanning OBJECT-LEVEL, RELATION-LEVEL, and MAP-LEVEL collaborative spatial skills. Importantly, this hierarchical decomposition serves

- as a diagnostic framework, isolating whether failures arise
- at the level of cross-view grounding, relational reasoning, or allocentric integration. Together, these levels enable finegrained identification of bottlenecks in current MLLMs. The five tasks are organized across three levels as follows:

OBJECT-LEVEL. (i) ANCHOR RECOGNITION. This task requires agents to identify which objects appear in both their views, testing the fundamental skill of establishing common reference objects across viewpoints. (ii) GLOBAL COUNTING. Building on anchor recognition, this task requires agents to aggregate object instances across views, correctly disambiguating which instances are shared across views and which are only visible to one agent to avoid double-counting

#### 3.1. Overview

Our task involves two static MLLM-based agents, an Answerer (A) and a Helper (H), positioned at different loca-

[Figure 4]

- Figure 3. Benchmark curation pipeline. Our pipeline involves scene generation, sampling complementary agent viewpoints, generating questions using templates and unique object description followed by paraphrasing the questions.

or omissions (see Fig. 2 top for examples).

RELATION-LEVEL. (iii) RELATIVE DISTANCE. In this task, agents must infer which object is closest or farthest from a target object. The target and candidate objects are distributed across both agent’s views, requiring agents to fuse their partial observations to reason about relative proximity. (iv) RELATIVE DIRECTION. In this task, the Answerer must infer the egocentric direction of a target object absent from its own view. Resolving this requires both agents to coordinate a cross-view perspective transformation from the Helper’s allocentric descriptions into the Answerer’s egocentric frame (Fig. 2 middle).

MAP-LEVEL. (v) COGNITIVE MAPPING. This task evaluates whether agents can communicate and combine their partial egocentric observations into a map-like allocentric representation of the environment (Fig. 2 bottom). Specifically, the answerer is presented with a candidate top-down map and tasked with judging whether it accurately represents the spatial layout of the shared environment.

Task formulation. All tasks in COSMIC are posed as multiplechoice questions with one correct answer and three carefully constructed distractors to discourage superficial heuristics (see the supplementary for details on distractor construction). The exception is COGNITIVE MAPPING, which we frame as a binary judgment of whether a presented top-down map is correct or incorrect, rather than free-form map generation, since evaluating freely generated maps would require generative assessment beyond the reliable capabilities of current MLLMs.

#### 3.2. Benchmark Curation

Scene Generation. To generate 3D indoor environments, we build upon Infinigen [24], a procedural generation framework for synthesizing photorealistic 3D scenes. We extend it with a customized pipeline that enables fine-grained control over indoor spatial layouts, object distribution, and dualview sampling. Viewpoints are sampled to ensure controlled partial overlap, with each pair of views sharing anchor objects while retaining objects exclusive to each perspective (Fig. 3). This produces high-fidelity scenes with complementary egocentric viewpoints tailored for collaborative reasoning, spanning diverse room types including Living Rooms, Bedrooms, Bathrooms, Kitchens, and Dining Rooms (Fig. 4 left).

Question Generation. Given the object sets from both views, we first filter out visually ambiguous objects to prevent low-level perceptual noise from confounding spatial reasoning evaluation. Specifically, objects that are too small, indistinct, or partially occluded are excluded. Next, to uniquely refer to specific objects when constructing questions, each object is assigned descriptors based on its color, size, or neighboring objects. Descriptors are programmatically chosen to uniquely identify a single object instance within the scene, yielding references such as “purple door next to a yellow cabinet”, “wall art above the cabinet”, etc (Fig. 3). Objects that cannot be assigned a unique descriptor are excluded from question generation. Finally, we employ fixed templates to generate questions spanning objects across both views ensuring that correct answers strictly require integrating information from both viewpoints. For instance, a global counting question “What is the total number of shelves in the room?” is generated from a template “What is the total number of <obj> in the room?”. The template-generated questions then undergo a paraphrasing stage to introduce linguistic diversity across the benchmark (see the supplementary for more details).

Data Filtering and Human Verification. To ensure that COSMIC serves as a rigorous evaluation of collaborative spatial reasoning, we implement a multi-stage filtering and verification pipeline. (i) CROSS-VIEW NECESSITY FILTERING. We remove questions that can be answered through commonsense biases rather than genuine cross-view integration. For instance, an MLLM may correctly infer the relative direction of a center table by exploiting the prior that a sofa typically faces a center table, without requiring any information from the Helper’s view. To filter out such cases, we prompt a strong MLLM (Qwen3-VL-235B-A22B) with only the Answerer’s view for three trials per question, excluding any question the model answers correctly in all three runs. (ii) HUMAN QUALITY ASSURANCE. Following automated filtering, all remaining question-answer pairs undergo manual verification by the authors to ensure linguistic clarity, resolve spatial and color ambiguities, and confirm the correctness of ground-truth answers.

Dataset Statistics. COSMIC consists of 899 indoor room scenes paired with 1,250 unique question-answer instances, with 250 multiple-choice questions per task, each drawn from a unique scene. Fig. 4 (center-top) shows the distribu-

[Figure 5]

- Figure 4. COSMIC benchmark composition. Left: Scenes from our benchmark. Center Top: Distribution of room types. Center Bottom: Distribution of scene clutter (number of object instances per scene). Right Top: Object-category frequencies across the benchmark. Right Bottom: Word cloud representing the most frequent spatial and object-related terms in the dataset.

tion of room types across tasks, reflecting sufficient scene diversity per subtask. The benchmark spans more than 23 distinct object categories, with Doors and Plant Containers being the most frequent (Fig. 4 right-top). Scenes vary considerably in clutter, with a mean of 17.71 object instances per scene and ranging from 6 to 31 (Fig. 4 center-bottom), adding to the scene-level complexity of the benchmark. Finally, the word cloud in Fig. 4 (right-bottom) shows that spatial and object-related terms dominate the question vocabulary, reflecting the focus of our benchmark.

### 4. Evaluation on the COSMIC Benchmark

Baseline Models. We evaluate a range of recent state-ofthe-art MLLM models. For open-source models, we include InternVL3.5 [31], Qwen3-VL [2], and Gemma-3 [27]. Among closed-source baselines we include GPT-5.2 [25], Gemini-3-Flash and Gemini-3-Pro [7]. For GPT-5.2 and Gemini-3-Flash, we report both no-thinking and thinking settings.

Evaluation Setup and Metric. We conduct zero-shot evaluation of all models. Performance is measured by binary accuracy, where a response is considered correct if and only if the model selects the correct answer option. This applies uniformly across all tasks, including COGNITIVE MAPPING. We report per-task accuracy with 90% bootstrap confidence intervals, averaged over four runs for open-source models and two runs for closed-source models.

Multi-turn Dialogue Protocol. Each instance unfolds as a multi-turn conversation, initiated by the Answerer. The Helper responds based on its own view and the dialogue history. This alternating exchange continues until the Answerer decides it has gathered sufficient information and explicitly terminates the conversation, or until a fixed turn

limit of 10 rounds is reached. Both agents have full access to the dialogue history throughout, and all communication proceeds exclusively through natural language with no parameter sharing or hidden state exchange between agents. Once the dialogue concludes, the Answerer is prompted to produce a final answer. For thinking models, we allow models to perform explicit intermediate reasoning before generating each dialogue message.

Model Inputs and Role Conditioning. Both Answerer and Helper agents are instantiated from the same underlying MLLM. Answerer agent’s input includes {IMAGE IA, QUESTION q, ANSWER OPTIONS, TASK INSTRUCTION, DIALOGUE HISTORY}, with the additional inclusion of the candidate top-down map for COGNITIVE MAPPING task. The Helper’s input consists of {IMAGE IH, TASK INSTRUCTION, DIALOGUE HISTORY}. Role-conditioning prompts are appended to the system prompt (“You are the Answerer”,

“You are the Helper”) to enforce behavior specialization (see supp. for system prompts).

Human Study. We additionally collect human–human dialogues under the same two-agent protocol on a subset of COSMIC, comprising 250 questions (50 per task), which we refer to as COSMIC-HUMAN. This serves to establish (i) a human performance baseline and (ii) a basis for systematic comparison of model and human communication patterns. We conduct in-lab sessions with university students to gather these responses. The interface mirrors the COSMIC setup, with Answerer and Helper roles, each observing only their egocentric view and jointly solving the task via multi-turn chat, subject to the same 10-round turn limit. Early termination is allowed.

[Figure 6]

[Figure 7]

- Figure 5. Top: Evaluation on COSMIC. Error bars denote 90% confidence intervals computed via bootstrap resampling. Dashed lines indicate chance levels (25% for 4-choice MCQ, 50% for binary map tasks and 30% for overall). Bottom: Evaluation on COSMICHUMAN.

### 5. Results on COSMIC

#### 5.1. Main Results

MLLMs significantly underperform humans. Fig. 5 (Bottom) reports per-task accuracy on COSMIC-HUMAN. Even the strongest model, Gemini-3-Pro-Thinking, achieves an average accuracy of only 71.82%, falling far short of the human baseline of 95.22%. This gap of over 23% underscores that COSMIC poses a substantial challenge for current MLLMs, highlighting that communicating about space remains a fundamentally difficult capability with significant room for improvement. Crucially, the gap is not uniform, it is narrowest on object-level tasks, where MLLMs show partial competence, but widens considerably on RELATIVE DIRECTION and COGNITIVE MAPPING, where humans maintain near-ceiling accuracy. The contrast is most striking on COGNITIVE MAPPING, where humans achieve 94% accuracy while even frontier models perform near chance, suggesting that humans possess a form of collaborative spatial intelligence that current MLLMs fundamentally lack.

Closed-source models consistently outperform opensource ones. Across all models on COSMIC (Fig. 5 Top), closed-source models consistently outperform open-source ones, with Gemini-3-Pro-Thinking and Gemini-3-FlashThinking (71.64%, 67.88%) leading closed-source models, while Qwen3-VL-32B-Instruct (avg. 52.47%) leads opensource models, substantially outperforming InternVL3.538B (avg. 39.45%) and Gemma-3-27B (avg. 36.22%). Among the open-source models, scaling yields mixed re-

[Figure 8]

[Figure 9]

Figure 6. Left: Performance Hierarchy in MLLMs across Object, Relation and Map-level. Right: Two-Agent Communication vs Single Agent.

sults, Gemma-3 (12B: 37.43% vs. 27B: 36.22%) and InternVL3.5 (8B: 37.39% vs. 38B: 39.45%) show no statistically significant improvement from their smaller to larger checkpoints, as evidenced by overlapping confidence intervals, while Qwen3-VL benefits meaningfully from scale (8B: 40.08% vs. 32B: 52.47%) with non-overlapping confidence intervals.

MLLMs’ performance degrades from Anchors to Maps. For the closed-source systems, we observe a capability hierarchy across tasks, with model accuracy declining from object-level to map-level reasoning, with statistically significant gaps in most cases (Fig. 6 Left). We see in Fig. 5 (Top) that ANCHOR RECOGNITION is the easiest task for most models (Qwen3-VL-32B: 66.59%, GPT-5.2-Thinking:

76.83%, Gemini-3-Pro-Thinking: 91.99%), yet performance on even this most fundamental skill remains far from robust across the model spectrum. Performance degrades further on GLOBAL COUNTING and RELATIVE DISTANCE, which additionally require aggregating multiple object instances across views and cross-view metric reasoning respectively.

The decline is steeper on RELATIVE DIRECTION (Gemini-3-Pro-Thinking: 46.21%, GPT-5.2-Thinking: 51.61%). Unlike RELATIVE DISTANCE, where agents can reason about proximity by comparing distances relative to shared anchor objects, RELATIVE DIRECTION requires the Answerer to transform the Helper’s allocentric spatial descriptions, typically expressed relative to shared anchors, into its own egocentric frame of reference. This allocentricto-egocentric transformation proves fundamentally challenging for current MLLMs, even the strongest models. COGNITIVE MAPPING demands an even harder operation, jointly integrating both egocentric views into a globally consistent allocentric map of the full environment. Models collapse entirely on this task, with even frontier models near the 50% chance baseline, indicating that full egocentric-to-allocentric integration across views remains beyond the reach of current MLLMs.

Thinking helps object and metric reasoning but not geometric integration. The capability hierarchy above raises a natural question of whether explicit reasoning can recover some performance at higher levels. We test this by enabling thinking for GPT-5.2 and Gemini-3-Flash (see Fig. 5 Top). Thinking yields consistent, statistically significant gains on ANCHOR RECOGNITION (Gemini-3-Flash 77.78% vs. 87.19%, GPT-5.2 64.34% vs. 76.83%) and RELATIVE DISTANCE (Gemini-3-Flash 76.81% vs. 88.00%, GPT-5.2 58.01% vs. 74.01%), suggesting that deliberate reasoning helps models more carefully match object descriptions and estimate metric relations across views. However, thinking yields no gain on RELATIVE DIRECTION or COGNITIVE MAPPING for either model family (Fig. 5 Top). This dissociation reveals that the bottleneck at higher levels is not a failure of reasoning but a fundamental deficit in the geometric understanding required to reconcile egocentric observations into a shared spatial model. For GLOBAL COUNTING, the effect of thinking is not consistent across models.

Communication introduces additional difficulty beyond single-agent reasoning. To isolate the impact of communication in collaborative spatial reasoning, we compare single-agent and two-agent performance for Qwen3-VL32B-Instruct and Gemini-3-Flash-Thinking as representatives of open-source and closed-source model families respectively (Fig. 6 Right). In the single-agent setting, each model is given both egocentric views simultaneously and asked to answer directly without dialogue. Both models perform substantially worse when required to communicate, with Qwen3-VL-32B-Instruct dropping from 64.62%

to 52.62% and Gemini-3-Flash-Thinking dropping from 78.38% to 68.07%. This consistent performance gap indicates that the challenge of COSMIC is not solely attributable to the spatial reasoning tasks themselves, but is meaningfully compounded by the demands of coordinating through natural language and maintaining a coherent shared spatial model across turns. We believe that the time is now ripe to hold our MLLMs to higher standards by testing not just their spatial reasoning capabilities but also their capability to communicate that reasoning in natural language. We hope that COSMIC will serve as a new standard benchmark for evaluating this capability.

#### 5.2. Failure Mode Analysis

Why do frontier MLLMs fail at language-mediated spatial integration? To investigate, the authors manually examine agent–agent conversations and analyze where communication and reasoning breaks down. We review 150 failed instances (30 per task) for the best-performing model (Gemini3-Pro-thinking), labeling each conversation with different error categories corresponding to failures at different stages of the dialogue (single conversation may exhibit multiple error types). We then compute the distribution of each error category over all errors made across the 150 conversations (Fig. 7). We describe three main error categories below (see the supp. for more details).

- 1. PERCEPTUAL FAILURES. These failures encompass two subcategories. Object Recognition Failure occurs when an agent entirely misses a visible object or misclassifies it as a different category, often triggered by environmental factors such as adverse lighting, cluttered backgrounds, or unusual viewing angles. Attribute Mislabelling refers to an agent hallucinating or misidentifying object properties such as color or size, causing its partner to search for an object that does not match what is visible in their view (see Fig. 7).
- 2. CROSS-VIEW GROUNDING FAILURES. This category encompasses systemic breakdowns in establishing shared anchor objects across views. We identify three subcategories. Referential Ambiguity arises when an agent generates utterances with underspecified object descriptions that fail to uniquely bind to a single instance in the partner’s view, particularly in cluttered scenes. Instance Merging occurs when two agents erroneously conclude that distinct object instances visible in their respective views refer to the same entity, collapsing them into one (see Fig. 7 Top-Right). Instance Duplication is the converse failure, where a single entity visible in both views is treated as two separate instances, with each agent believing it refers to a different object.
- 3. GEOMETRIC & RELATIONAL FAILURES. This category encompasses structural breakdowns in reasoning about spatial relations and transforming egocentric observations

[Figure 10]

- Figure 7. Failure mode analysis on COSMIC. Qualitative examples illustrate the three failure categories, Perceptual Failure (Object Recognition Failure), Cross-View Grounding Failure (Instance Merging), and Geometric and Relational Failure (Layout Understanding Failure). The bar chart shows the distribution of failure modes across tasks, and the donut chart shows the overall breakdown.

into a unified allocentric representation. PerspectiveTaking Failures occur when an agent incorrectly maps its partner’s spatial descriptions onto its own egocentric frame of reference, resulting in systematic orientation errors such as left-right mirroring or front-back inversions. Layout Understanding Failure occurs when an agent fails to reason about how objects are arranged relative to one another in 3D space from its 2D egocentric view, preventing it from constructing a coherent mental model of the scene’s spatial layout (see Fig. 7 Bottom-Left).

Fig. 7 shows the distribution of failure modes across tasks. Perceptual Failures represent the smallest share of errors overall (19.70%), suggesting that low-level scene perception is comparatively reliable. Cross-View Grounding Failures are the dominant error category overall (46.09%), accounting for the largest share of failures, representing a primary bottleneck in collaborative spatial reasoning. They are most prominent on ANCHOR RECOGNITION and GLOBAL COUNTING (67.85% & 64% respectively), consistent with the crossview instance binding demands of these tasks. Moving up the task hierarchy, Geometric and Relational Failures become increasingly dominant, constituting the vast majority of errors on RELATIVE DIRECTION and COGNITIVE MAP-

PING (57.33% & 69.44% respectively), reflecting that these tasks evaluate a holistic understanding of the room’s spatial layout that goes well beyond object-level reasoning. Note that cross-view grounding failures remain present across all tasks, consistent with it being a fundamental skill for any form of cross-view spatial reasoning.

Overall, these results reveal a cascading failure dynamic where unresolved grounding errors can propagate through the dialogue and compound into geometric and relational failures that dominate on higher-level tasks. Since agents communicate exclusively through natural language, a single misidentified object or ambiguous reference early in the conversation can corrupt the shared spatial model that subsequent reasoning depends on.

#### 5.3. Human vs. Model Dialogue: How Different Is Collaboration?

To investigate how the nature of communication differs between human-human and model-model, we compare the dialogues of human pairs on COSMIC-HUMAN with those of MLLM agents.

MLLMs Produce Verbose but Spatially Shallow Exchanges. Fig. 8 (Top) shows mean words per conversation

[Figure 11]

- Figure 8. Communication Efficiency & Information Dynamics on COSMIC-HUMAN. Top: Communication Efficiency (Verbosity vs. Accuracy). Bottom: Information Dynamics (Unique Objects Mentioned vs. Conversation Turn Number). Shaded regions denotes variance across conversations.

against average accuracy for all models and human pairs. Notably, humans achieve the highest accuracy (95.22%) while using least words per conversation (avg. 199.65 words). In contrast, MLLM agents produce substantially more verbose exchanges (avg. across models 438.48) yet achieve considerably lower accuracy (avg. across models 50.48%). Communication verbosity and accuracy are largely uncorrelated across models (pearson r = 0.37, p = 0.26), indicating no statistically significant relationship between the length of a dialogue and its effectiveness. For instance, Qwen3VL-8B-Instruct generates the most words (613.30) among open-source models yet achieves among the lowest accuracies (37.90%). This suggests that more verbose conversation does not necessarily lead to more effective conversation. We hypothesize that humans are efficient because of robust spatial priors, built over a lifetime of navigating and communicating about physical environments, which support targeted, uncertainty-reducing exchanges. In contrast, models compensate for weaker spatial representations with verbose but less informative dialogue.

MLLMs Fail to Converge While Humans Systematically Narrow the Hypothesis Space. To understand how human and model dialogues differ in their information dynamics, we track the mean number of unique objects mentioned per turn across conversation turns (Fig. 8 Bottom). Human pairs begin with a moderately high number of object mentions in

[Figure 12]

Figure 9. Dialogue Repair behavior in Gemini-3-Pro-Thinking.

the first turn and taper sharply over next turns, consistent with quickly locking onto a set of verified anchor objects and thereafter exchanging targeted, spatially precise updates to their mental model of the scene. This behavior is also qualitatively evident in Fig. 1, where human pairs quickly ground the sofa and peach window as shared anchors, and thereafter narrow down to targeted, spatially precise exchanges. In contrast, MLLM agents show a slower and less consistent decline, with GPT-5.2-Thinking converging gradually and Gemini-3-Pro-Thinking and Gemini-3-Flash-Thinking sustaining a high rate of new object mentions throughout, as reflected in their persistently elevated curves in Fig. 8 (Bottom). As also seen in Figure 1, rather than anchoring on shared objects early, MLLM agents persistently enumerate new scene elements (the white couch, black desk, white desk, and TV), across every turn without converging on a shared reference frame from which to resolve the query. This pattern of continuous exploration rather than convergence directly explains the verbosity of MLLM conversations and their inability to build a coherent shared spatial model across turns.

MLLMs Rarely Recover from Flawed Reasoning Trajectories. Dialogue Repair refers to the metacognitive ability of agents to identify and correct erroneous reasoning trajectory during the conversation, a capacity that is critical for robustly building a shared spatial model. We quantify such behavior in MLLMs and compare it against human pairs. For each conversation in COSMIC-HUMAN, we employ a strong MLLM (Gemini-3-Flash-Thinking) as an automated judge, providing it with full task context including both egocentric views, the question & options, the ground-truth answer, and the dialogue transcript (see supp). The judge produces a binary label indicating whether a repair event occurred (see Fig. 9).

We define the dialogue repair rate as the fraction of conversations where agents successfully recovered from a

wrong reasoning trajectory, out of all conversations where a wrong reasoning trajectory was observed (see supp.). Models exhibit far less dialogue repair than humans. Human pairs achieve a dialogue repair rate of 79.31%, while MLLM agents fall substantially short of this, with the bestperforming model Gemini-3-Pro-Thinking reaching only 28.04% and Qwen3-VL-32B achieving just 7.8%.

Together, these analyses reveal that the failures of MLLMs reflect both a communication deficit and a deeper spatial reasoning deficit, with each compounding the other. Models struggle both to convey spatial observations effectively and to update their spatial model from their partner’s utterances, a bidirectional breakdown that accumulates into an increasingly incoherent shared spatial representation.

### 6. Conclusion and Future Work

We introduced COSMIC and evaluated a broad set of frontier MLLMs. Our evaluation reveals a consistent capability hierarchy, current MLLMs show partial success at anchor grounding but deteriorate on relational reasoning and perform near chance on cognitive mapping. Human pairs achieve rapid convergence through targeted, informationdense exchanges, whereas MLLM agents resort to verbose dialogue that fails to yield a shared spatial model across turns.

These findings point to several directions for future work. Progress on higher-level spatial tasks will likely require moving beyond linguistic chain-of-thought toward explicit visual reasoning mechanisms that support internal geometric verification and mental rotation. Structured spatial communication protocols such as anchorfirst grounding conventions, explicit reference frame agreement, or sketch-based spatial descriptions, could help mitigate referential ambiguity and improve cross-view grounding. Finally, agents must develop proactive dialogue repair strategies to detect and correct flawed reasoning midconversation rather than accumulating contradictory spatial representations across turns. Addressing these gaps is essential for the development of capable, collaborative agents.

### References

- [1] Kartikeya Badola, Jonathan Simon, Arian Hosseini, Sara Marie Mc Carthy, Tsendsuren Munkhdalai, Abhimanyu Goyal, Tomáš Koˇcisk`y, Shyam Upadhyay, Bahare Fatemi, and Mehran Kazemi. Multi-turn puzzles: Evaluating interactive reasoning and strategic dialogue in llms. arXiv preprint arXiv:2508.10142, 2025. 3
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 5
- [3] Justin Chen, Swarnadeep Saha, and Mohit Bansal. Reconcile: Round-table conference improves reasoning via consensus

- among diverse llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7066–7085, 2024. 3
- [4] Pei Chen, Shuai Zhang, and Boran Han. Comm: Collaborative multi-agent, multi-reasoning-path prompting for complex problem solving. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 1720–1738, 2024. 3
- [5] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision language models,

2024. 2

- [6] Herbert H. Clark. Using language. Cambridge University Press, 1996. 1
- [7] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 5
- [8] Yilun Du, Shuang Li, Antonio Torralba, Joshua B Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Forty-first international conference on machine learning, 2024. 3
- [9] Xiangjun Gao, Zhensong Zhang, Dave Zhenyu Chen, Songcen Xu, Long Quan, Eduardo Pérez-Pellitero, and Youngkyoon Jang. Map2thought: Explicit 3d spatial reasoning via metric cognitive maps, 2026. 3
- [10] Simon Garrod and Anthony Anderson. Saying what you mean in dialogue: A study in conceptual and semantic coordination. Cognition, 27(2):181–218, 1987. 1
- [11] Mohsen Gholami, Ahmad Rezaei, Zhou Weimin, Sitong Mao, Shunbo Zhou, Yong Zhang, and Mohammad Akbari. Spatial reasoning with vision-language models in ego-centric multiview scenes, 2025. 2
- [12] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models, 2026. 2
- [13] Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9161–9175, 2023. 2
- [14] Kanghee Lee, Injae Lee, Minseok Kwak, Kwonyoung Ryu, Jungi Hong, and Jaesik Park. Spatialmosaic: A multiview vlm dataset for partial visibility. arXiv preprint arXiv:2512.23365,

2025. 2

- [15] Stephen C. Levinson. Space in language and cognition: Explorations in cognitive diversity. Language, 79(3):620–622,

2003. 1

- [16] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025. 2
- [17] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi

- Zhang, Yongliang Shen, et al. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025. 1
- [18] Yuxuan Li, Aoi Naito, and Hirokazu Shirado. Hiddenbench: Assessing collective reasoning in multi-agent llms via hidden profile tasks. arXiv preprint arXiv:2505.11556, 2025. 3
- [19] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning, 2023. 2
- [20] Nora S Newcombe. Spatial cognition. Memory and Cognitive Processes, 3:113–163, 2004. 1
- [21] John O’keefe and Lynn Nadel. The hippocampus as a cognitive map. Oxford university press, 1978. 3
- [22] Timothy Ossowski, Danyal Maqbool, Jixuan Chen, Zefan Cai, Tyler Bradshaw, and Junjie Hu. Comma: A communicative multimodal multi-agent benchmark. arXiv preprint arXiv:2410.07553, 2024. 3
- [23] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023. 1
- [24] Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, et al. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21783–21794,

2024. 4

- [25] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 5
- [26] Ilias Stogiannidis, Steven McDonagh, and Sotirios A. Tsaftaris. Mind the gap: Benchmarking spatial reasoning in visionlanguage models, 2025. 2
- [27] Gemma Team. Gemma 3. 2025. 5
- [28] Stefanie Tellex, Thomas Kollar, Steven Dickerson, Matthew Walter, Ashis Banerjee, Seth Teller, and Nicholas Roy. Understanding natural language commands for robotic navigation and mobile manipulation. In Proceedings of the AAAI conference on artificial intelligence, pages 1507–1514, 2011. 1
- [29] Edward C Tolman. Cognitive maps in rats and men. Psychological review, 55(4):189, 1948. 3
- [30] Barbara Tversky. Structures of mental spaces: How people think about space. Environment and Behavior, 35(1):66–80,

2003. 1

- [31] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 5
- [32] Zelai Xu, Zhexuan Xu, Xiangmin Yi, Huining Yuan, Xinlei Chen, Yi Wu, Chao Yu, and Yu Wang. Vs-bench: Evaluating vlms for strategic reasoning and decision-making in multiagent environments. 2025. 3
- [33] Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal

- large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10632–10643, 2025. 1, 2, 3
- [34] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, et al. Mmsi-bench: A benchmark for multiimage spatial intelligence. arXiv preprint arXiv:2505.23764, 2025.
- [35] Chun-Hsiao Yeh, Chenyu Wang, Shengbang Tong, Ta-Ying Cheng, Ruoyu Wang, Tianzhe Chu, Yuexiang Zhai, Yubei Chen, Shenghua Gao, and Yi Ma. Seeing from another perspective: Evaluating multi-view understanding in mllms. arXiv preprint arXiv:2504.15280, 2025. 1, 2
- [36] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, Saining Xie, Manling Li, Jiajun Wu, and Li Fei-Fei. Spatial mental modeling from limited views, 2025. 2, 3
- [37] Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025. 1
- [38] Ziheng Zhang, Minghao Chen, Suguo Zhu, Tingting Han, and Zhou Yu. Mmcnav: Mllm-empowered multi-agent collaboration for outdoor visual language navigation. In Proceedings of the 2025 International Conference on Multimedia Retrieval, pages 1767–1776, 2025. 3
- [39] Ruolin Zhu, Shaobin Li, Zixing Zhu, Jing Jia, and Min Yang. Ca-vln: Collaborative agents in mllm-powered visuallanguage navigation. Sensors (Basel, Switzerland), 26(4): 1254, 2026. 3

[Figure 13]

## Communicating about Space: Language-Mediated Spatial Integration Across Partial Views

## Supplementary Material Overview of Supplementary

Table 1. Templates used for question generation. Below, object category is a placeholder for an object category, for example sofa, while object description is a placeholder for the unique description of a given object instance, for example green sofa near a white door.

- 1. COSMIC Benchmark Design (Sec. 7)
- 2. Evaluation Protocol and Reproducibility (Sec. 8)
- 3. Failure Mode Analysis (Sec. 9)
- 4. Human Data Collection Interface (Sec. 10)
- 5. Multi-turn Dialogue (Sec. 11)
- 6. Dialogue Repair (Sec. 12)
- 7. Model and Human Conversations (Sec. 13)
- 8. Case Study (Sec. 14)
- 9. Compute Resources (Sec. 15)
- 10. Broader Impact (Sec. 16)
- 11. Limitations (Sec. 17)

Task Template Anchor Recognition Which of the following objects

is visible in both your and your partner’s views of the room?

Global Counting What is the total number of <object category> in the room?

Relative Direction From your perspective, in which direction is <object description>?

### 7. COSMIC Benchmark Design

Relative Distance Which of the following objects is closest to / farthest from the <object description>?

This section provides additional details on the COSMIC benchmark curation pipeline. We also provide several examples from our benchmark in Figs. 12 to 16.

Cognitive Mapping Is this top-down map of the room correct?

#### 7.1. Scene Generation

Each environment E contains a set of objects O =

- o1,...,oM, with the Answerer (A) and Helper (H) agents placed at distinct viewpoints. Answer and Helper receive their respective egocentric views IA and IH. The set of visible objects in each view, OA = {oj ∈ O | oj is visible in IA } and OH = {oj ∈ O |
- oj is visible in IH }, are designed such that their union (OA ∪ OH ≈ O) covers most of the environment, while their intersection (OA ∩ OH ̸= ∅) provides shared anchor objects that allows cross-view grounding and perspective alignment between agents. Overall, this formulation transforms spatial reasoning from a single-agent perception task into a collaborative spatial reasoning task.

a given camera if at least three of its eight bounding box corners are both within the camera frustum and unoccluded by intervening geometry. Camera poses are resampled until the resulting visibility sets satisfy OA ∩ OH ̸= ∅ and OA ∪ OH ≈ O.

#### 7.2. Question Generation

To generate questions for different tasks as described in Sec. 7.3, we rely on predefined templates to produce the base question strings shown in Tab. 1. These templategenerated questions are then passed through a LLM paraphrasing stage gpt-4o-mini, which rephrases them to introduce linguistic diversity and naturalness while preserving their semantic content. Tab. 2 demonstrates examples of these paraphrases.

To ensure that the visibility sets OA and OH satisfy these properties by construction, we employ a controlled dual-view camera sampling strategy. Both cameras are placed at a fixed altitude of 1.5m above the floor with a fixed horizontal pitch, producing natural egocentric views at human eye level. Yaw is sampled independently and uniformly over [−180◦,180◦] for each camera, ensuring diverse viewing directions across scenes. To determine OA and OH, we employ a ray-casting visibility algorithm. For each mesh object in the scene, we project all eight corners of its axis-aligned bounding box into camera space and cast rays from the camera origin toward each corner. An object is considered visible from

#### 7.3. Distractor Design and Task Difficulty

We describe the distractor design and the task difficulty for each task below:

ANCHOR RECOGNITION. The primary difficulty in this task lies in communicating the overlap between two egocentric observations precisely. An example question for this task is “Which of the following objects is present in both

Table 2. Examples of template-generated questions and their corresponding paraphrased versions.

Before Paraphrasing After Paraphrasing Question: Which of the following objects is visible in both your and your partner’s views of the room? Options:

Question: Which of the following objects is present in both your and your partner’s views of the room? Options:

- (A) floor purple Lamp
- (B) black shutter Window
- (C) green Plant Container next to a black Desk
- (D) Lamp near a black shutter Window

- (A) floor purple lamp
- (B) window with black shutters
- (C) green plant container located next to a black desk
- (D) lamp near a window with black shutters

Question: What is the total number of Monitor in the room? Options:

Question: What is the total number of computer monitors in the room? Options:

- (A) 1
- (B) 2
- (C) 3
- (D) 4

- (A) 1
- (B) 2
- (C) 3
- (D) 4

Question: Which of the following objects is closest to the floor purple Lamp? Options:

Question: Which of the following objects is nearest to the floor lamp with purple color? Options:

- (A) beige curtain Window
- (B) Monitor near a green Plant Container
- (C) brown curtain Window
- (D) green Plant Container next to a black Desk

- (A) window with beige curtains
- (B) computer monitor located near a green plant container
- (C) window with brown curtains
- (D) green plant container located next to a black desk

Question: From your perspective, in which direction is Window near a Monitor? Options:

Question: From your perspective, in which direction is the window that is near a computer monitor? Options:

- (A) behind
- (B) front
- (C) left
- (D) right

- (A) behind
- (B) front
- (C) left
- (D) right

Question: Is this top-down map of the room correct? Question: Is this top-down layout of the room accurate?

your and your partner’s views of the room?” The question template for this task is provided in Tab. 1. Agents must dynamically reference objects via their attributes or relational cues (e.g., “the yellow cabinet next to the door”), since multiple object instances of the same category with similar visual properties may appear across the two views, making category labels alone insufficient to determine whether two agents are referring to the same physical instance. This mirrors the referential grounding challenges that arise in realworld cluttered environments, where precise, discriminative descriptions are essential for establishing shared reference.

The correct answer is the object present in OA ∩OH. We construct three distractors (see Fig. 12 top-left) as follows, (i) an object visible exclusively to the Answerer (OA \ OH) (e.g. window with green curtains), (ii) an object visible exclusively to the Helper (OH \ OA) (e.g. blue shelf), and (iii) an object from the same semantic category as the correct answer but differing in at least one discriminative attribute (e.g.,

color, size, or spatial relation). This third distractor, drawn exclusively from one agent’s view, is specifically designed to prevent models from succeeding via category-level matching alone, requiring instead attribute- and relation-based reasoning (e.g. window with beige curtains). When a scene does not contain a same-category object with differing attributes, this slot is filled by an object sampled from another available category in the scene (e.g. gray toilet located next to a blue shelf).

GLOBAL COUNTING. The central difficulty in this task is cross-view deduplication. The question template for this task is provided in Tab. 1, with a representative example being

“What is the total number of shelves in the room?”

Building on ANCHOR RECOGNITION, in this task, agents must jointly determine which object instances across their two views refer to the same physical entity and which are distinct, communicating this through language to avoid both double-counting shared instances and omitting instances

visible only to one agent. This demands simultaneous crossview aggregation and instance-level disambiguation, requiring agents to produce and interpret precise, discriminative descriptions of objects within the same category.

Questions are generated by aggregating all instances of a target object category across both views and querying the total count, i.e., |Oans ∪ Ohelp|. Distractors are constructed to target two canonical failure modes (see Fig. 13 top-left for an example), overcounting, corresponding to the naive sum |Oans| + |Ohelp| (i.e., double-counting shared instances), and undercounting (i.e., omitting certain instances). For remaining distractors, we sample a count value close to the ground truth.

RELATIVE DISTANCE. The difficulty of this task stems from the need to compare metric distances across two spatially disjoint views. A typical question takes the form,

“Which of the following objects is closest to the white computer desk?” (see Tab. 1 for question template). Agents cannot directly observe all candidate objects and must instead align their respective partial views through communication before a meaningful distance comparison is possible.

To generate questions, we select an anchor object to be the target object. We then sample four candidate objects for the options from each agent’s exclusive view (OA \ OH and OH \ OA). Distances are then computed between the bounding box of the target object and each candidate. The correct answer is the closest (or farthest) candidate from the target object. We enforce a minimum distance margin between the correct answer and the next closest candidate to ensure the difference is geometrically unambiguous.

RELATIVE DIRECTION. This task requires a cross-view perspective transformation. An example question for this task is “From your perspective, in which direction is the computer desk located?” where the target object (computer desk) is visible only to the Helper. Answerer agent must infer the direction of the target object with respect to its own view by grounding via shared anchor objects and mapping its partner’s spatial descriptions into its own frame of reference. The difficulty increases in scenes with little overlap across views, since the perspective transformation needs to be performed with limited shared grounding.

Answer options correspond to eight egocentric directions, Front, Front-Right, Right, Behind-Right, Behind, BehindLeft, Left, and Front-Left, representing angular offsets of 45◦. A direction is assigned when the object’s bearing falls within a ±10◦ window centered on the corresponding orientation angle.

COGNITIVE MAPPING. This task requires agents to jointly construct a coherent allocentric (top-down) representation of the environment from their complementary egocentric views, making it the most demanding task in the benchmark. Rather than just reasoning about individual objects or pairwise relations, agents must integrate their partial obser-

vations into a globally consistent top-down map. A typical question takes the form, “Is this top-down map of the room correct?” (see Tab. 1 for the question template, subsequently paraphrased using an LLM).

We construct the correct map by projecting object positions from the scene onto a 2D top-down grid. Distractor maps are generated by swapping the positions of Helperexclusive objects in the correct map, producing layouts that contain the correct set of objects but placed at incorrect spatial locations. The options for this task are binary, i.e. either Yes or No. Maps are rendered as a visual image and provided directly to the Answerer agent as input.

### 8. Evaluation Protocol and Reproducibility

Given the multi-turn dialogue setup of COSMIC, where each conversation involves up to 10 rounds of messages and the final answer depends on the full reasoning trajectory accumulated across turns, model performance exhibits higher variance than is typical in single-turn evaluation settings. We therefore strongly recommend that future work report results averaged over multiple independent runs, with 90% confidence intervals computed via bootstrap resampling (10000 iterations), consistent with the evaluation protocol adopted in this work. Single-run evaluations might produce unreliable estimates of true model performance.

### 9. Failure Mode Analysis

In addition to the qualitative samples presented in the main paper, Fig. 10 shows representative failure cases of Gemini3-Pro-Thinking on COSMIC, covering the remaining error categories from our failure mode analysis.

In Fig. 10 (top-left), we show an example of PERCEPTUAL FAILURE (Attribute Mislabelling). The Helper mislabels the color of the window’s blinds (“might be the orange curtains option”) introducing perceptual uncertainty that propagates into the Answerer’s reasoning leading to an incorrect prediction.

Fig. 10 (top-right) shows an example of CROSS-VIEW GROUNDING FAILURE (Referential Ambiguity). The Helper produces object descriptions that are insufficiently discriminative (“I also see a dark brown door to the left of that shelf”) to uniquely identify a single object instance, leaving the Answerer unable to establish a reliable shared referent.

Fig. 10 (bottom-left) illustrates CROSS-VIEW GROUNDING FAILURE (Instance Duplication). The agents fail to recognize that the white cabinet described by each agent is the same physical instance (“My white cabinet does not match your description”), incorrectly treating it as two distinct objects across the two views.

Finally, we show a case of GEOMETRIC & RELATIONAL FAILURE (Perspective-Taking Failure) in Fig. 10 (bottomright). Both agents incorrectly infer each other’s viewpoint,

[Figure 14]

Figure 10. Failure Mode Analysis. Qualitative samples illustrating different failure categories, Perceptual Failure (Attribute Mislabelling), Cross-View Grounding Failure (Referential Ambiguity and Instance Duplication), and Geometric & Relational Failure (Perspective-Taking Failure). Green ticks denote the ground truth answer.

the Helper asserts “the red cabinet is located on the wall directly behind you” and the Answerer acknowledges “I understand that you are likely facing the wall behind me”, constructing mutually inconsistent spatial models that compound into an erroneous directional judgment.

### 10. Human Data Collection Interface

The human dialogue data collection interfaces for the Answerer and Helper are shown in Fig. 21 and 22, respectively, with the corresponding task instructions shown in Fig. 19 and Fig. 20. Both interfaces display the participant’s egocen-

tric image alongside a chat box for multi-turn dialogue. The Answerer’s interface additionally presents the question and multiple-choice answer options, with a submit button that becomes accessible once the dialogue concludes, either after

- 10 messages are exchanged or when the Answerer chooses to end the chat early.
- 11. Multi-turn Dialogue

For each message generation step during the multi-turn conversation, we use a temperature of 1.0 and a maximum of 8192 completion tokens. The system prompts provided to

the Answerer and Helper agents are shown in Sec. 11.1. The dialogue initiation prompts are shown in Sec. 11.2, the task description prompts are shown in Sec. 11.4, and the final prompt used to elicit the answer from the Answerer after dialogue concludes is shown in Sec. 11.3.

#### 11.1. System Prompts for Answerer and Helper Answerer System Prompt

- 1. You will be participating in a COLLABORATIVE TASK to answer a question.
- 2. You are the ANSWERER AGENT.
- 3. You will be connected to a HELPER AGENT.
- 4. In this task, you and the HELPER AGENT will receive one image each that shows different views of the same room.
- 5. You have to chat and collaborate with the HELPER AGENT to answer your question correctly.
- 6. Overall, your role is to answer your question correctly by having a conversation with the HELPER AGENT.

###### Helper System Prompt

- 1. You will be participating in a COLLABORATIVE TASK.
- 2. You are the HELPER AGENT.
- 3. You will be connected to an ANSWERER AGENT.
- 4. In this task, you and the ANSWERER AGENT will receive one image each that shows different views of the same room.
- 5. You have to chat and collaborate with the ANSWERER AGENT to help them answer their question correctly.
- 6. Overall, your role is to help the ANSWERER AGENT by having a conversation with them.

#### 11.2. Prompts for Multi-turn Dialogue

###### Answerer Prompt for Multi-turn Dialogue

<Answerer’s View> <Map View (given only for cognitive mapping task)>

- 1. The provided image is your view of the room.
- 2. HELPER AGENT also receives one image that shows a different view of the same room.
- 3. You will be given a multiple-choice question with different options. Only one of the options is correct.
- 4. You can send only one message at a time. You cannot send consecutive messages. You have to wait for the HELPER AGENT to respond before you can send your next message.
- 5. You can send a maximum of <max-num-turns> messages to the HELPER AGENT.
- 6. After the conversation is over, you will be asked to provide the answer. <Task Description>

Note: When you are ready to answer the question, you can terminate the conversation early by saying ‘TERMINATE’. Use exact word ‘TERMINATE’ in your response.

Goal: <Answerer Goal> QUESTION: <Question> OPTIONS: <Options>

Begin the conversation with the HELPER AGENT. You MUST generate all your messages in this format

ANSWERER AGENT: <RESPONSE>. Do not deviate from this format.

###### Helper Prompt for Multi-turn Dialogue

<Helper’s View>

- 1. The provided image is your view of the room.
- 2. ANSWERER AGENT also receives one image that shows a different view of the same room.
- 3. You can send only one message at a time. You cannot send consecutive messages. You have to wait for the ANSWERER AGENT to respond before you can send your next message.
- 4. You can send a maximum of <max-num-turns> messages to the ANSWERER AGENT. <Task Description>

Goal: <Helper Goal> Begin the conversation with the ANSWERER AGENT by responding to their first message. You MUST generate all your messages in this format HELPER AGENT: <RESPONSE>. Do not deviate from this format. <First-message-from-answerer-agent>

#### 11.3. QA Prompt after Multi-turn Dialogue

Answerer QA Prompt after Multi-turn Dialogue

Now you need to answer the multiple-choice question based on your view and the conversation with the HELPER AGENT.

QUESTION: <Question> OPTIONS: <Options>

Instructions:

- 1. Select the correct answer from the given options. Make sure to select only one of the options from the given options A, B, C, or D.
- 2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER> or <ANSWER>C</ANSWER> or <ANSWER>D</ANSWER>.

Answerer QA Prompt after Multi-turn Dialogue (Cognitive Mapping)

Now you need to answer the multiple-choice question based on your view and the conversation with the HELPER AGENT.

QUESTION: <Question> OPTIONS: <Options>

Instructions:

- 1. Select the correct answer from the given options. Make sure to select only one of the options from the given options A, or B.
- 2. Format your response like <ANSWER>A</ANSWER> or <ANSWER>B</ANSWER>.

#### 11.4. Task Description Prompts

###### Anchor Recognition

Answerer:

- 1. The task is to find the object that is common in both your and your partner’s views.
- 2. Only one of the objects in the options will be common to both views, while the other objects in the options will be present in only one of the views of the room - either the answerer’s or the helper’s. Helper:

- 1. The task is to find the object that is common in both your and your partner’s views.
- 2. Only one of the objects in the options will be common to both views, while the other objects in the options will be present in only one of the views of the room - either the answerer’s or the helper’s.

###### Global Counting

Answerer:

- 1. The task is to find the count of a given object.
- 2. You and your partner must make sure that you are counting the total number of unique instances of that object in the room while preventing overcounting or undercounting, as there may be some common objects in both views.
- 3. Note: Cabinets and Shelves refer to the entire furniture, not different compartments within a specific piece of furniture. Helper:

- 1. The task is to find the count of a given object.
- 2. You and your partner must make sure that you are counting the total number of unique instances of that object in the room while preventing overcounting or undercounting, as there may be some common objects in both views.
- 3. Note: Cabinets and Shelves refer to the entire furniture, not

different compartments within a specific piece of furniture.

###### Relative Distance

Answerer:

- 1. The task is to find which of the objects in the options is either the farthest or the closest to the object mentioned in the question.
- 2. The object mentioned in the question is visible to both you and your partner.
- 3. The objects in the options are visible either in only your view or only in your partner’s view but not in both views.
- 4. Important: The correct answer is based on both views combined. An object that looks closest or farthest from your perspective may not be correct, and the right choice might be an object you cannot see at all. Helper:

- 1. The task is to find which of the objects in the options is either the farthest or the closest to the object mentioned in the question.
- 2. The object mentioned in the question is visible to both you and your partner.
- 3. The objects in the options are visible either in only your view or only in your partner’s view but not in both views.
- 4. Important: The correct answer is based on both views combined. An object that looks closest or farthest from your perspective may not be correct, and the right choice might be an object you cannot see at all.

###### Relative Direction

Answerer:

- 1. In this task, the Answerer must determine the direction of a target object from their own viewpoint.
- 2. The Answerer cannot see the object directly -- it is visible only to the Helper.
- 3. Since the Answerer cannot see the object directly, to identify where the object is located, the Answerer must communicate with the Helper and use the information obtained to infer its direction relative to themselves.
- 4. Note: Here, the directions are relative to the Answerer’s orientation, i.e., their egocentric viewpoint.
- 5. Directions (like front, front-left, front-right, etc.) describe where something is based on the Answerer’s facing direction, not on what they can currently see. Even if the object is outside the view, it can still be called front-left or front-right if it lies in that direction relative to the Answerer. Helper:

- 1. In this task, the Answerer must determine the direction of a target object from their own viewpoint.
- 2. The Answerer cannot see the object directly -- it is visible only to the Helper.
- 3. Since the Answerer cannot see the object directly, the Helper must communicate with the Answerer to provide the information needed to answer the question.
- 4. Note: Here, the directions are relative to the Answerer’s orientation, i.e., their egocentric viewpoint.
- 5. Directions (like front, front-left, front-right, etc.) describe where something is based on the Answerer’s facing direction, not on what they can currently see. Even if the object is outside the view, it can still be called front-left or front-right if it lies in that direction relative to the Answerer.

###### Cognitive Mapping

Answerer:

- 1. The task is to identify if the provided map accurately depicts the top-down layout of the room.
- 2. You and the Helper observe different, partial views of the room, and neither view is complete on its own. The full layout can only be inferred by communicating and combining information from both views.
- 3. Evaluate the map only by the spatial arrangement of the objects it shows. Focus exclusively on the object categories listed in the legend, ignore any other items, and do not consider objects placed on top of other objects in your judgment. Helper:

- 1. The task is to identify if the provided map accurately depicts the top-down layout of the room.
- 2. The map is only provided to the answerer agent.

### 12. Dialogue Repair

The automated MLLM judge (Gemini-3-Flash-Thinking) labels each conversation in COSMIC-HUMAN for presence and resolution of flawed reasoning trajectories. The judge is provided with the full task context, including both egocentric views, the question and answer options, the ground-truth answer, and the complete dialogue transcript. The judge then assigns one of three labels (see Sec. 12.1 for detailed prompt). A label of 0 indicates that no flawed reasoning trajectory was observed. A label of 1 indicates that a flawed reasoning trajectory was present and successfully repaired. A label of -1 indicates that a flawed reasoning trajectory was present but went undetected and uncorrected. Each conversation is judged over two independent runs, and only conversations where both runs agree on the assigned label are retained. The dialogue repair rate is then computed as the fraction of conversations labeled 1 out of all conversations labeled 1 or -1.

#### 12.1. Prompts for Dialogue Repair Judge

MLLM Judge Prompt - Dialogue Repair

You are evaluating the presence of dialogue repair and backtracking behaviour in the following two-agent conversation. You must detect when an agent goes down a wrong reasoning path and is unable to backtrack on its errors thereby arriving at the wrong answer. A wrong reasoning path corresponds to a cascade of errors that the agents make in their utterances which end up propagating in the conversation, and the agents are unable to recover from thereby causing the agents to answer a question wrongly.

IMPORTANT: This is NOT an individual utterance-level evaluation. Do NOT score based on isolated phrases like "let me revisit" or "I might be wrong." The evaluation must be done at the full conversation / task level. Agents might also make some errors and correct them later on but you are measuring the overall dialogue repair with the goal of the collaborating agents to identify the correct answer.

SCORING: You must judge whether:

- - The agents go down an incorrect reasoning path and continue without correcting it, OR - The agents recognize an incorrect reasoning path and successfully backtrack and recover. - You can ignore the mistakes in the grammar and spelling while evaluating the agents Scoring Criteria (Conversation-Level Score):
- -1 = The conversation contains a wrong reasoning path, and the agents continue reasoning based on that error without correcting it. They go down the wrong path and never recover.

- 0 = The conversation contains no wrong reasoning path. Reasoning remains logically consistent and aligned with the correct interpretation of the task throughout.
- 1 = The conversation contains a wrong reasoning path, but the agents recognize the broader mistake and successfully correct their reasoning, recovering from the error before finalizing the answer.

Evaluate the entire conversation and assign a single score based on overall presence of wrong reasoning path and recovery.

Output format: Return only a single integer and an explanation for your scoring. Strictly maintain the below format: <score>-1, 0, or 1</score> <explanation>Your reasoning behind your score</explanation> Do not provide any additional text.

<Answerer-and-Helper-views> <Question-to-agents> <Options> <Correct-answer> <Conversation-between-agents>

### 13. Model and Human Conversations

We present example dialogues from both MLLM agents (Gemini-3-Pro-Thinking) and human participants across all five COSMIC tasks in Fig. 17 and Fig. 18, respectively. Each

figure displays one conversation per task, showing the egocentric views available to each agent, the multiple-choice options, full multi-turn dialogue, agent’s prediction, and the ground truth. Together, these examples illustrate the qualitative differences in communication strategy between human pairs and MLLM agents.

### 14. Case Study

Fig. 11 presents a dialogue between two MLLM agents (Gemini-3-Pro-Thinking) collaborating on a Relative Distance task. After establishing the scene context, the Answerer commits an early error, it references only one black desk (“I also see a black desk on that wall, but it is next to a window with white blinds”) while failing to recognize the second desk located near the window with brown curtains. This constitutes an object recognition failure, where a relevant object in the scene is entirely omitted. As a result, subsequent reasoning proceeds under an incorrect grounding assumption regarding the queried black desk.

Despite this initial error, the agents demonstrate partial cross-view spatial reasoning. For example, they correctly infer that their viewpoints correspond to opposite walls of the room, indicating a basic level of cross-view layout understanding. Additionally, the Helper agent accurately determines that options B and D are the farthest from the anchor object. This inference reflects cross-view distance reasoning, since the window with brown curtains near the blue cabinet (option D) is outside the Helper’s field of view.

Nevertheless, the final answer remains incorrect. The agents ultimately select the black desk after reasoning that the pink cabinet separating it from the orange-shuttered window is wider than the lamp separating it from the desk. However, because the initial object grounding was incorrect, the reasoning chain is grounded to the wrong desk. Consequently, the agents fail to recover from the early perception error, leading to an incorrect final prediction.

Overall, in this example, while the model exhibited some positive behaviors at the utterance level, their effects were negated by compounding failures that the agents were unable to detect or recover from.

### 15. Compute Resources

All experiments were conducted on 4 NVIDIA A100 80GB GPUs. Open-source models were served using vLLM with tensor parallelism across all 4 GPUs, with each model taking approximately 2.5 hours to evaluate on the full benchmark. Closed-source model evaluations were conducted via their respective APIs.

### 16. Broader Impact

By providing a systematic diagnostic benchmark for collaborative spatial communication, COSMIC aims to surface

concrete bottlenecks that must be addressed before MLLMbased collaborative systems can be reliably deployed in realworld. The specific failure modes identified here, namely cross-view grounding errors, perspective transformation failures, inability to construct globally consistent spatial maps, and more, have direct implications for a range of applications such as assistive home robotics, multi-agent warehouse coordination, and AR-based remote guidance, among others, where precise alignment of spatial understanding between agents is not merely beneficial but essential for safe and effective operation.

The distributed, communication-based setting of COSMIC more closely mirrors how spatial reasoning actually occurs in human-AI teaming scenarios than existing singleagent benchmarks. The ability to build shared spatial mental models through language is foundational to a wide range of everyday human activities, and becomes a hard prerequisite as AI systems are increasingly deployed alongside humans in shared physical environments such as warehouses, hospitals, and construction sites, where grounding spatial references, resolving ambiguous descriptions, and maintaining a coherent shared representation of the environment is important. Our findings suggest that current MLLMs are far from meeting this bar, and we hope COSMIC serves as a concrete target for developing agents that can participate fluently in the kinds of spatially grounded collaborative interactions that humans engage in naturally.

The benchmark also has implications for the design of spatial communication protocols in multi-agent AI systems. The contrast between human and model dialogue strategies, where humans converge rapidly through anchor-first grounding while models explore redundantly, suggests that explicitly structured spatial communication conventions such as enforcing reference frame agreement or anchor establishment early in dialogue could meaningfully improve model performance. Such protocols could inform the design of more robust human-AI and AI-AI interfaces for spatially grounded tasks.

We do not foresee any direct negative societal impacts from this work.

### 17. Limitations

COSMIC evaluates collaborative spatial reasoning in static, controlled indoor environments generated from a limited set of object categories. While this enables systematic benchmarking useful for diagnosing bottlenecks in MLLMs, it does not capture the full complexity of real-world spatial communication, which involves dynamic scenes, continuous viewpoint changes, and a far richer vocabulary of objects. Similarly, agents in COSMIC operate from fixed viewpoints with no ability to actively gather additional visual information, whereas real collaborative agents can move, reorient, and request clarification through action rather than language

[Figure 15]

Figure 11. Conversation between agents (Gemini-3-Pro-Thinking). Although the agents demonstrate successful behaviors on some atomic steps, errors in the dialogue negate these successes, leading them to select the incorrect answer (option A).

alone.

The benchmark evaluates spatial communication through multiple-choice questions, which, while controlled and reproducible, constrains the space of possible responses. In particular, the binary formulation of the Cognitive Mapping task sidesteps the harder problem of free-form map generation and evaluation, which remains an open challenge. Additionally, performance is measured purely by final answer accuracy, without directly rewarding the quality of the

dialogue itself, meaning a model that arrives at the correct answer through sub-optimal reasoning is indistinguishable from one that reasons perfectly.

Finally, our human study, while providing a valuable baseline and qualitative reference point, is conducted with university students in a controlled lab setting. This population may not be representative of the broader range of spatial communication strategies employed across different demographic groups or levels of spatial expertise.

[Figure 16]

###### Figure 12. Samples from COSMIC Benchmark (Anchor Recognition).

[Figure 17]

###### Figure 13. Samples from COSMIC Benchmark (Global Counting).

[Figure 18]

###### Figure 14. Samples from COSMIC Benchmark (Relative Distance).

[Figure 19]

###### Figure 15. Samples from COSMIC Benchmark (Relative Direction).

[Figure 20]

###### Figure 16. Samples from COSMIC Benchmark (Cognitive Mapping).

[Figure 21]

Figure 17. Representative dialogues from Gemini-3-Pro-Thinking on COSMIC. The tick mark denotes the ground truth answer and the red-highlighted option indicates the answerer’s prediction. From top to bottom, ANCHOR RECOGNITION (top-left), GLOBAL COUNTING (top-right), RELATIVE DISTANCE (middle-left), RELATIVE DIRECTION (middle-right), and COGNITIVE MAPPING (bottom). Each panel displays the Answerer’s and Helper’s egocentric views, the multiple-choice options, and the multi-turn dialogue exchanged between the two agents.

[Figure 22]

Figure 18. Representative dialogues from Humans on COSMIC. The tick mark denotes the ground truth answer and the red-highlighted option indicates the answerer’s prediction. From top to bottom, ANCHOR RECOGNITION (top-left), GLOBAL COUNTING (top-right), RELATIVE DISTANCE (middle-left), RELATIVE DIRECTION (middle-right), and COGNITIVE MAPPING (bottom). Each panel displays the Answerer’s and Helper’s egocentric views, the multiple-choice options, and the multi-turn dialogue exchanged between the two agents.

[Figure 23]

###### Figure 19. Instruction page for the human dialogue data collection interface (Answerer)

[Figure 24]

###### Figure 20. Instruction page for the human dialogue data collection interface (Helper)

[Figure 25]

Figure 21. Human Dialogue Data Collection Interface (Answerer)

[Figure 26]

Figure 22. Human Dialogue Data Collection Interface (Helper)

