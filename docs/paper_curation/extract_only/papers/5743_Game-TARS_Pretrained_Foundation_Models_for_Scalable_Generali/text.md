# arXiv:2510.23691v1[cs.AI]27Oct2025

[Figure 1]

## Game-TARS: Pretrained Foundation Models for Scalable Generalist Multimodal Game Agents

### Bytedance Seed

See Contributions section for a full author list.

### Abstract

We present Game-TARS, a generalist game agent trained with a unified, scalable action space anchored to human-aligned native keyboard–mouse inputs. Unlike API- or GUI-based approaches, this paradigm enables large-scale continual pre-training across heterogeneous domains, including OS, web, and simulation games. Game-TARS is pre-trained on over 500B tokens with diverse trajectories and multimodal data. Key techniques include a decaying continual loss to reduce causal confusion and an efficient Sparse-Thinking strategy that balances reasoning depth and inference cost. Experiments show that Game-TARS achieves about 2 times the success rate over the previous sota model on open-world Minecraft tasks, is close to the generality of fresh humans in unseen web 3d games, and outperforms GPT-5, Gemini-2.5-Pro, and Claude-4-Sonnet in FPS benchmarks. Scaling results on training-time and test-time confirm that the unified action space sustains improvements when scaled to cross-game and multimodal data. Our results demonstrate that simple, scalable action representations combined with large-scale pre-training provide a promising path toward generalist agents with broad problem-solving abilities.

Date: October 27, 2025 Correspondence: yujia.qin@bytedance.com; shiguang.sg@bytedance.com

aProject: https://seed-tars.com/game-tars

[Figure 2]

Figure 1 Game-TARS achieves a higher level of performance compared to humans, domain experts, and general VLMs in unseen 3D virtual environments, including open-world [19], FPS games [78], web games, and simulators [11].

### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 2 Continual Pre-Training for Generalist Game Agent . . . . . . . . . . . . . . . . . . . . . . . 5

- 2.1 A Scalable and Generalist Action Space for Computer-Use Agents . . . . . . . . . . . . . . . 5
- 2.2 Native Sparse ReAct Pretraining via Thinking Aloud . . . . . . . . . . . . . . . . . . . . . . . 6
- 2.3 Continual Pre-training with Decaying Loss Function . . . . . . . . . . . . . . . . . . . . . . . 8

- 3 Post-Training for Generalist Game Agent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 3.1 Instruction Following . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.2 Multimodal Prompts Elicit In-Context Learning . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 3.3 Sparse Thinking via Reinforcement Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 3.4 Long Episodic Memory Compression with Sparse Thought . . . . . . . . . . . . . . . . . . . . 11
- 3.5 Post-Training with Cross-Source Agentic Trajectories . . . . . . . . . . . . . . . . . . . . . . . 12

- 4 Experiment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- 4.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 4.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 4.2.1 Experimental Setups . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.2.2 Evaluation on Open-World Minecraft . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- 4.2.3 Evaluation on Unseen Benchmarks and Simulators . . . . . . . . . . . . . . . . . . . . 16

- 4.3 Analysis and Discussions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- 4.3.1 Scaling Experiments on Training Datasets and Inference Steps . . . . . . . . . . . . . 18
- 4.3.2 Ablation on sparse thinking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 4.3.3 Validation on Continual Pre-Training Loss . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 5 Related Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- 5.1 Game Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 5.2 Generalist Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 5.3 Broader AI Agents . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- 6 Conclusions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- A Rollout Trajectories . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- A.1 Rollouts in Minecraft . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- A.2 Rollouts in Unseen Games . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- B Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- B.1 Prompts of Game-TARS on Minecraft . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- B.2 Prompts of Game-TARS on Unseen Games . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

### 1 Introduction

Building Generalist Artificial Agents capable of seamlessly interacting with complex and dynamic digital environments has emerged as a key research path toward achieving Artificial General Intelligence (AGI) [10, 41, 46]. Video games, with their diverse task objectives, intricate interaction logic, and rich visual information, provide an ideal platform for both training and evaluating such agents [7, 58, 65]. Despite substantial progress, existing approaches still face significant challenges in creating truly scalable agents with broad generalization capabilities [51, 61, 67]. One of the core bottlenecks is the definition of the action space: traditional agents often rely on highly customized, pre-defined action sets tailored for specific environments [1, 47, 48]. This tight coupling between the agent and the environment severely limits the agent’s generalization, making it difficult to adapt to new tasks, particularly in open-world settings [4, 19].

To overcome these limitations, we propose a novel framework for pre-training general game agents. Our central contribution introduces a fundamental shift in paradigm: we move away from application-specific high-level action instructions [66, 70] and instead define a scalable, unified action space directly tied to low-level computer input devices, such as the keyboard and mouse. This native human interaction paradigm ensures that the agent can operate within any graphical user interface (GUI) environment, laying the foundation for large-scale, cross-domain pre-training.

Utilizing this unified action space, we have developed a comprehensive agent training process that includes continual pre-training and post-training stages on a vision-language model [3, 12, 21, 44]. In the continual pre-training phase, we employ an efficient, native ReAct paradigm [82] that incorporates Sparse Thinking—interleaving reasoning and action only at critical decision points to mirror human cognitive patterns. This is achieved by combining offline chain-of-thought synthesis [75] with an online think-aloud protocol [14] to collect and refine high-quality, causally consistent multimodal trajectory data. In the post-training phase, we implement a series of targeted fine-tuning strategies to systematically enhance the agent’s core capabilities. These strategies include: enforcing strict adherence to instructions through automated action space augmentation; fostering dynamic context learning via multimodal prompts and long-range trajectory learning; refining the agent’s decision-making by enforcing efficient Sparse Thinking through rejection fine-tuning; and developing long-term memory abilities via a dual-layer memory architecture.

To extend the agent’s capabilities beyond gaming, we also incorporate cross-domain agent trajectories from fields such as code generation, GUI automation, and research tasks into the post-training data. This diverse mix of data aims to transform the model from a specialized game-player into a versatile general computer-user with broader problem-solving capabilities. The work presented in this paper offers a comprehensive, end-to-end solution for developing the next generation of generalist agents.

We first validated the effectiveness of the unified action space and large-scale continual pretraining by testing the agent’s capabilities in the open-world game Minecraft [19]. Unlike grounding-based [55] or API-based [66] actions, we found that the unified action space, when trained with less than 10B data, did not initially outperform GUI-based action agents in terms of success rate. However, the key advantage of the unified action space lies in its scalability—large-scale data can be efficiently collected, and the agent can be pretrained using a consistent format across all games. Our scaling experiments demonstrate that this unified action space provides an excellent foundation for the realization of generalist agents. When trained on a large corpus of general computer-use data, Game-TARS, based on the unified action space, achieved approximately a 2x performance improvement in Minecraft compared to previous state-of-the-art expert models [74].

Furthermore, we conducted extensive evaluations on a wide range of untapped benchmarks, including FPSgame Vizdoom [78], 3D Simulator Miniworld [11], and online web games [49]. The results highlight the exceptional generalization ability of our generalist agent. Our mini version of Game-TARS outperformed renowned pre-trained models such as Gemini-2.5-Pro [12], GPT-5 [46], and Claude-4-Sonnet [2] on various tasks, showcasing superior performance across diverse domains. These results provide compelling evidence of the broad applicability and robustness of the unified action space and continual pretraining approach, establishing Game-TARS as a formidable agent in a wide variety of environments.

[Figure 3]

###### Figure 2 Generalist Game Agent Game-TARS. Game-TARS can interpret and respond to various human instructions across diverse environments using a single neural network with a consistent set of weights. It was pre-trained on a wide range of multimodal datasets, including vision-language question-answering, captioning, over 20k hours of game trajectories, GUI agent trajectories, and more.

- Table 1 Taxonomy of computer-use agent action spaces, contrasting high-level, environment-specific APIs with the low-level, general-purpose device controls.

Action Hierarchy Agent Action Level Action Definition

MCP Application Search(query) Retrieve the relevant URLs for the query.

[Figure 4]

MCP Application ReadUrl(url) Get the content in URL.

MCP Application CreateFile(filePath) Create a new file at the specified file path.

Code Application Run(script) Execute a script file or program.

Code Application Compile(code) Compile source code into an executable. GUI System-Level(Desktop) Drag(x1, y1, x2, y2) Drag from (x1, y1) to (x2, y2)

GUI System-Level(Desktop) scroll(200) Scrolls up by 200 units.

GUI System-Level(Desktop) Hotkey(key) Presses the specified hotkey.

GUI System-Level(Desktop) LeftDouble(x, y) Double-clicks at (x, y).

GUI System-Level(Mobile) LongPress(x, y) Long presses at (x, y).

GUI System-Level(Mobile) PressHome() Presses the “home” button.

- - Device-Level(Mouse) mouseMove(dx, dy) Move the mouse in the dx and dy directions.

- - Device-Level(Mouse) mouseClick(buttons) Click the mouse buttons.

- - Device-Level(Keyboard) keyPress(keys) Press buttons in keyboard.

### 2 Continual Pre-Training for Generalist Game Agent

Our objective is to model an inherently multi-task generalist agent. We formulate this agent as a autoregressive, parameterized policy, πθ. The policy’s role is to map the agent’s history and current perception to a decision. At any given timestep t, the agent receives an Observation (ot) from the environment (specifically, the current visual image frame). It also maintains an internal memory state (ht), which serves as a compressed summary of all past interactions. Following the ReAct paradigm [82], the agent’s decision-making process is modeled as a two-stage sequence:

- • Reasoning (rt): An internal cognitive process that encompasses context analysis, memory retrieval, planning, and self-reflection.
- • Action (at): An external interaction with the environment.

The policy πθ is responsible for generating this entire reasoning and action chain. The core policy function at timestep t is thus defined as:

(rt,at) ∼ πθ(· | ot,ht) (1)

where the memory state ht is a function of all preceding steps, i.e., ht = f((ri,ai,oi)ti=0−1). We define a complete ReAct cycle—comprising the agent’s reasoning, its action, and the environment’s subsequent observation—as

a single "step": (rt,at,ot). Therefore, a complete trajectory (τ) of length T, representing the full log of the agent’s interaction, is formalized as a sequence of these steps:

τ = (r0,a0,o0),(r1,a1,o1),...,(rT,aT,oT) (2) Our training objective is to optimize the parameters θ of our policy πθ using a large dataset of these trajectories.

#### 2.1 A Scalable and Generalist Action Space for Computer-Use Agents

A critical challenge in constructing a generalist agent is the definition of an action space that is both broadly applicable and highly scalable. Traditional agents, such as those trained in environments like Minecraft [4, 22] or StarCraft [65], typically operate on a predefined, task-specific action space. These action spaces, whether encapsulated via game-internal protocols (e.g., Minecraft Protocol) or code-level APIs [66], necessitate the design and implementation of a unique instruction set for each new environment or task. While effective for specialized domains, this approach suffers from a fundamental limitation: the tight coupling of the action space to the environment and task, which severely constrains the model’s generality and scalability. Adapting the agent to a novel game or application often requires substantial manual effort in redefining action spaces, engineering API wrappers, and annotating data, thereby impeding generalization to open-world scenarios.

To partially mitigate this issue, prior work has introduced agents that operate on Graphical User Interfaces (GUIs), such as UI-TARS [49], by identifying and manipulating GUI elements. Although such methods improve generality, we observe that their action spaces remain constrained by the underlying operating system (OS) or specific UI frameworks. Significant discrepancies in GUI element structures and interaction modalities across Windows, macOS, and Linux, or even between different software applications, prevent the realization of a truly universal action space.

To transcend these limitations, we propose a more fundamental solution: abstracting the agent’s action space to a lower level and grounding it directly in the most universal input devices for human-computer interaction—the keyboard and mouse. By doing so, the agent’s action instruction set is decoupled from any high-level application or OS-specific API, achieving direct alignment with a human user’s physical operations. We term this the Human-Native Interaction paradigm. This paradigm ensures an unprecedented level of universality for the action space across all computer graphical environments.

Specifically, we define the agent’s action space as a composition of three primitive operations:

- • mouseMove(dx, dy): This action simulates the two-dimensional movement of the mouse cursor on the screen. The parameters dx and dy represent the relative displacement along the horizontal and vertical axes, respectively.
- • mouseClick(button): This action simulates a mouse button click event. The button parameter specifies the target button, such as left, right, or middle.
- • keyPress(keys): This action simulates the press and release of one or more keyboard keys. The keys parameter can be a single key (e.g., keyPress(w)) or a combination of keys (e.g., keyPress(ctrl, c)).

We augment this core set with supplementary actions such as Think, LongThinking, and No-Op to facilitate more complex reasoning and behavior. The complete action space and its descriptions are provided in Table 1.

The principal advantage of grounding the action space in these input primitives is its exceptional scalability, enabling our generalist agent to operate in any standard computing environment without modification, much like a human user. More importantly, in contrast to traditional discrete action spaces centered on highlevel maneuvers, our Human-Native Interaction paradigm is inherently time-aligned. Each action primitive corresponds directly to a physical input at a specific point in time. This design obviates the need for special modeling of action durations or complex temporal dependencies, as the agent can autonomously learn the timing and rhythm of its operations through experience. Consequently, whether interacting with 2D pixel-based games, complex 3D open-world environments, or standard desktop applications, the agent employs the same unified and stable action space. This consistency establishes a robust foundation for large-scale, cross-domain Continual Pre-Training.

#### 2.2 Native Sparse ReAct Pretraining via Thinking Aloud

Prior research has demonstrated that co-modeling reasoning and action as a unified output for an agent significantly enhances model generalization and performance [49, 59, 82]. Recent methods usually employ a pre-trained vision language model [21] to inject reasoning steps into the raw trajectory data τ = {(ot,at)}Tt=0 from human gameplaying, translating the observation-action pairs into the ReAct format. However, previous studies have shown that retrospectively adding reasoning to recorded action sequences is exceptionally difficult and inefficient [49, 67], as accurately reconstructing the annotator’s real-time thought process is nearly impossible. On the other hand, our time-aligned, high-frequency action modeling (with action intervals of only 50-100ms) results in action sequences with numerous repetitive and continuous operations, during which a human player’s cognitive state does not significantly change. Consequently, forcing the generation of a reasoning step at every action step is both impractical and unnecessary. Building upon this, we have designed an online think-aloud data collection and processing pipeline to generate high-quality, native Sparse ReAct trajectories for pre-training.

Unified Trajectory Data Paradigm. We define the pre-training trajectories in a sparse-thinking format, formalized as:

τ = (r0,a0,o0),...,(an,on),...,(rm,am,om),...,(rT,aT,oT). (3)

[Figure 5]

- Figure 3 The pipeline of Think-Aloud data collection and post-processing. This process captures and synchronizes three types of original inputs (screen, keyboard, and mouse, audio), refines sparse-thinking through the ASR-LLM pipeline, and uses a timestamp aligner to synthesize the final (Instruction, Observation, Thinking, Action) datasets.

Here, some timesteps m contain the complete tuple of reasoning, action, and observation (rm,am,om), while other timesteps n only contain action-observation pairs (an,on). To ensure the structural integrity of each trajectory, we mandate that reasoning steps must be included at the beginning (t = 0) and end (t = T) of the trajectory, corresponding to planning and summarization, respectively.

Online Think-Aloud Data Collection. Inspired by Deitke et al. [14], we instead adopt a think-aloud protocol, requiring annotators to verbalize their thoughts in real-time via audio while performing tasks, as demonstrated in Figure 3. This method allows us to collect ReAct sequences containing native, in-situ reasoning. To ensure the density and quality of the reasoning data, we instruct annotators to provide thoughts for at least three key moments per minute.

Data Recording and Preliminary Processing. Our data acquisition system utilizes a parallel recording program to synchronously capture a player’s screen frames (as observations ot), low-level mouse and keyboard input signals (as actions at), and the audio stream from the "think-aloud" process (as raw thoughts rt). After collection, we use an ASR model [52] to transcribe the audio into text, which is then refined using a Large Language Model (LLM) to remove noise and enhance logical consistency.

Causal Alignment via Visual Anchors. The precise alignment of multi-modal data, particularly ensuring the causality of observation-action pairs, is a critical step. In practice, we found that due to minor latencies in the recording system, relying solely on timestamps causes action signals to lag behind their corresponding screen frames. This temporal misalignment severely disrupts the data’s causal relationship. If used directly for training, it can lead the agent model to degenerate into an Inverse Dynamics Model (IDM) [4]—where the model learns to infer actions from state differences rather than deciding based on the current state—which would significantly impair its performance in practical applications. We developed a realignment method based on visual anchors to overcome this issue. We use the on-screen mouse cursor as a visual anchor and employ a visual grounding model [67] to precisely track its pixel position in each frame. By comparing the tracked changes in cursor position with the recorded mouse movements (dx, dy), we can accurately realign each action signal to the exact observation frame in which it was executed, thereby completely correcting the timestamp bias and restoring the causal chain.

Trajectory Construction and Refinement. After achieving causal alignment, we construct training sequences

centered on observation frames. For the time window between any two consecutive frames, all recorded keyboard and mouse operations are merged into a single action event; a lack of operations is marked as a no-op. Finally, to reinforce the semantic association between reasoning and action, we implement a thought timestamp relocation strategy. This involves intelligently adjusting the timestamp of a thought to align with the subsequent action to which it is most semantically relevant, ultimately producing highly synchronized and causally consistent (observation ot, reasoning rt, action at) training sequences.

#### 2.3 Continual Pre-training with Decaying Loss Function

In our auto-regressive training framework, the agent predicts an action a at each timestep t based on the comprehensive history of all preceding observations and actions, denoted by Ht = (o1,a1,o2,a2,...,at−1,ot). A critical challenge we identified in our preliminary experiments is the severe imbalance in the action distribution within non-Markovian agent trajectories [87], and this trend intensifies as the trajectory length increases. Specifically, temporally persistent actions, where at = at−1, constitute the vast majority (over 90%) of the action sequence1. This high frequency of repetitive, low-entropy actions causes the standard cross entropy loss to be dominated by these easy samples and leads to the causal confusion in imitation learning [13], thereby masking the learning signal from the more challenging and crucial action transitions (where at ̸= at−1). Previous methods, such as VPT [4], remove all no-ops from the trajectory; however, this disrupts the complete action distribution, as waiting is necessary in some scenarios. To mitigate this issue, we introduce a history-aware loss re-weighting scheme that dynamically attenuates the loss contribution from consecutive identical actions. Formally, we redefine the training loss for a trajectory τ = (o1,a1,...,oT,aT) as:

T

ωt log P(at | Ht) (4)

L = −

t=1

The weight ωt for the action at timestep t is determined by its consistency with the immediate past. Let kt be the number of consecutive timesteps including t where the action has remained unchanged (i.e., at = at−1 = ... = at−k

and at−k

t+1. The weight ωt is then defined as an exponential decay: ωt = γk

t ̸= at−k

t

t−1 (5)

Here, γ ∈ (0,1) is a decay factor hyperparameter (in our implementation, γ = 1/2). This formulation assigns a full weight of 1 (since γ0 = 1) to any action that is new or different from the previous one, while exponentially reducing the weight for each subsequent repetition. By systematically down-weighting the loss from monotonous action sequences, this method compels the model to focus more of its capacity on learning the high-entropy decision boundaries associated with action changes, leading to a more robust and capable policy.

### 3 Post-Training for Generalist Game Agent

After a large-scale continual pre-training on over 20k game hours (about 500B tokens), we introduce a post-training stage to further enhance the agent’s execution capabilities and interactive intelligence in specific tasks. This stage focuses on strengthening three core abilities: instruction following, sparse thinking, and long-term memory.

#### 3.1 Instruction Following

To build a generalist agent that can interact seamlessly across different games and application environments, the primary task is to ensure it can strictly follow the instructions for a given task, particularly in understanding and using the action space of a specific environment. Although continual pre-training endows the model with powerful general interaction abilities, it also leads to the learning of a prior distribution over actions based on a massive dataset. This distributional bias presents two major challenges during the post-training phase:

1The detailed value is different on different games.

[Figure 6]

Figure 4 Distribution of different game types in the Game-TARS training dataset.

- • Action Space Violation: The model tends to output the actions defined in the pre-training data, but they are invalid in the current task. For example, even if the System Prompt specifies the UpArrow as the jump key, the model might still output Space to perform a jump out of habit.
- • Behavioral Inertia: The model may ignore current observations and instructions, instead relying on the frequency distribution of actions to minimize the loss function. This manifests as the model frequently outputting common actions from the pre-training data (e.g., the W key) rather than making decisions based on the current context.

Automatic Action Space Augmentation. To address these issues, we propose a fine-tuning strategy called Automatic Action Space Augmentation. The core idea of this strategy is to decouple the semantic function of an action from its specific key binding during training. We preserve the semantics of actions in the original trajectory data but randomly replace their corresponding key bindings. For instance, in a trajectory, we might randomly replace the original W key used for "move forward" with the X key, while simultaneously defining the function of X as "move forward" in the System Prompt. Through this random substitution, we compel the model to rely on the System Prompt to understand the currently valid action space and its semantics, rather than merely memorizing and reproducing high-frequency action patterns from the pre-training data. Furthermore, this method effectively mitigates the problem of highly imbalanced action distributions in the training data, preventing the model from gaming the loss function by exploiting this prior distribution.

Inverse-Dynamics Prediction. To further reinforce the model’s understanding of action semantics, we introduce Inverse-Dynamics Prediction as an auxiliary training task. This task requires the model to infer the action at that caused a state transition, given the current observation ot and the next observation ot+1. This is formally represented as:

at ∼ P(· | A,[oi,ai]ti=1−1,ot,ot+1), (6) where A is the predefined action space described in natural language. This task compels the model to learn the causal relationship between actions and changes in the environmental state, thereby fostering a deeper understanding of the actual effects of actions. Finally, throughout the entire post-training phase, we provide explicit task instructions (Instruction, I) in the System Prompt for each video trajectory. This shifts the model’s prediction target from simple imitation learning to conditional instruction following, with the final training objective expressed as:

P(at | I,A,ot).

[Figure 7]

Figure 5 Game-TARS is trained on a wide range of games, including adventure, shooting, role-playing, and racing.

#### 3.2 Multimodal Prompts Elicit In-Context Learning

Another core insight of the post-training stage is that the agent should be able to learn from historical interactions and dynamically adjust its behavioral strategies. This is crucial for enhancing its generalization capabilities in unseen games. We define this ability as the agent’s In-Context Learning (ICL) [16, 26]: the agent dynamically adjusts the distribution of its future actions, including the selection of discrete actions and the scaling of continuous actions, by observing the effects of previous actions on the environment.

Contextual Understanding of Discrete Actions via Multimodal Prompts. We found that relying solely on language-based System Prompts is often insufficient to accurately and comprehensively express the semantics of all actions. In complex action games, the function of certain keys (e.g., special skills or combos) is highly context-dependent and can even vary between characters. Such complex semantics are difficult to describe exhaustively with text. To address this, we introduce a multimodal prompting mechanism. For each game, we select video clips demonstrating key "action-effect" pairs to provide as in-context examples for the agent. By observing these examples, the agent can intuitively understand the actual effect of specific actions in that environment, rather than relying on textual system prompts. During training, we even randomly mask the textual action descriptions, A, in the System Prompt, forcing the agent to infer the action space from the multimodal context.

Adaptive Calibration of Continuous Actions. For continuous actions, such as mouse movements, the practical significance of their numerical values varies dramatically across different environments. In-game mouse sensitivity settings, differences in user hardware, and personalized key bindings can all cause the same (dx,dy) values to produce vastly different in-game responses. To enable the agent to adapt to this dynamism, we adopt the following strategies:

- • Long-Horizon, Non-Markovian Trajectory Training: We use non-Markovian trajectory data for autoregressive training and extend the length of a single trajectory to at least 80 consecutive frames. This ensures that the agent has a sufficiently long history of interactions to reference for calibration when making decisions.
- • Initial Action Loss Masking: To prevent the gradient direction from being distorted by uncalibrated actions at the beginning of a trajectory (due to a lack of historical context), we mask the action loss for the first T mask steps, excluding them from gradient propagation.
- • Continuous Action Data Augmentation: We apply random scaling to the mouse movements in the trajectories. For each training segment, we multiply all (dx,dy) values by a random scaling factor k, while ensuring their absolute values do not exceed a cap of 180. This forces the agent to learn to infer the correct scaling factor from the feedback of historical actions, rather than relying on a fixed numerical range.

Finally, to further enhance the model’s ability to learn long-term dependencies, our training data also includes a subset of ultra-long sequences with trajectories exceeding 240 steps.

#### 3.3 Sparse Thinking via Reinforcement Fine-Tuning

During the post-training phase, we observed that both the reasoning data synthesized offline by LLMs and that recorded online by humans exhibit a degree of abstraction, preventing their content from precisely guiding subsequent action generation. Furthermore, the "think-aloud" data collection mechanism led some players to pause game actions while verbalizing their thoughts, resulting in a large volume of reasoning text being associated with "no-op" actions. This decoupling of reasoning and action weakens the model’s ability to learn the causal relationship between them.

Locating the sparse thinking Steps. Not all steps within a trajectory require explicit reasoning for the agent to act correctly. To optimize the fine-tuning process, we first identify the timestamps where reasoning is most critical. We employ a system prompt to let Game-TARS generate the actions only, πaction, which is required to predict the next action based solely on the history of observations and actions. We then evaluate Game-TARS on the offline trajectories. A timestep t is identified as requiring reasoning augmentation if the action-only model fails to predict the ground-truth action agtt . This suggests that a simple reactive policy is insufficient at this stage, and a more deliberative reasoning process is required. This procedure yields a set of timestamps Sr targeted for reasoning augmentation:

Sr = t ∈ 1,...,T | I(πaction(a | ot,(oi,ai)ti=0−1) = agtt ) = 0 (7)

Where I(·) is the indicator function used to identify unmatched generated actions. The subsequent Rejection Fine-Tuning is then applied exclusively to the timestamps within this set Sr.

Refining Thought-Action Chains via Rejection Sampling. To strengthen the causal link between reasoning and action, we adopted a Rejection Fine-Tuning (RFT) approach to refine and enhance the offline trajectory data [25]. We use the actions from the original trajectories as a critic to filter and generate high-quality reasoning-action pairs through Rejection Sampling.

Specifically, for any given timestep t in a trajectory τ, we have the agent model π generate a candidate thought rtcand based on the observation ot, and then predict an action apredt based on this thought. We compare the predicted action with the ground-truth action from the human demonstration, agtt . The candidate thought rtcand is considered valid and retained for subsequent fine-tuning only if the predicted action matches the ground-truth action. This process can be formally described as:

Accept rtcand iff I(π(at | ot,rtcand) = agtt ) = 1 (8)

where I(·) is the indicator function. Through this method, we filter out a significant amount of abstract reasoning that is irrelevant to the actual behavior, retaining only those thought chains that directly lead to correct actions.

After obtaining high-quality reasoning data through rejection sampling, we perform two post-processing steps:

- • Thought Consolidation: We found that in segments where the same action is performed consecutively, the effective reasoning content also shows a high degree of repetitiveness. Therefore, we merge these consecutive, identical thoughts, retaining only the instance at the first timestep to reduce data redundancy.
- • LLM Rewriting and Density Control: We further utilize an LLM to rewrite the filtered reasoning texts, making their expression more concise and their logic clearer. Concurrently, we actively control the density of reasoning in the trajectories (i.e., the proportion of steps with reasoning to the total number of steps) to better align with a natural human cognitive rhythm.

#### 3.4 Long Episodic Memory Compression with Sparse Thought

To succeed in complex tasks requiring long-range planning and memory, an agent must be able to integrate historical information that far exceeds the standard context window length. To this end, we have designed a

[Figure 8]

Figure 6 Memory Mechanism in Game-TARS Agent is divided into two areas: context and summary, corresponding to long-term and short-term memory, respectively. The context memory retains a complete record of observations, thoughts, and actions for M steps. In contrast, the summary memory only retains the thinking content for N steps. Older information (beyond t − M − N) is popped using a sliding window approach.

two-tiered memory mechanism that remains consistent during both training and inference, effectively balancing high-fidelity recent memory with highly compressed long-term memory.

- Tier 1: Short-Term Contextual Memory. This tier serves as the agent’s "working memory" and is stored directly within the model’s context window. It maintains an interleaved sequence in the form of a queue, containing the most recent observations and their corresponding ReAct pairs, refined through the "sparse thinking" process. This sequence is updated using a sliding window, with a standard context length of 32k tokens. However, because encoded visual images consume a substantial number of tokens, the temporal span of interaction history that this window can accommodate is limited.
- Tier 2: Long-Term Summary Memory. When historical information is moved out of the short-term context window, it is compressed and stored in the second memory tier. To overcome the storage bottleneck imposed by visual tokens, we leverage the output of the "sparse" mechanism—sparse and highly refined thought texts—as summaries of historical events. Each step that is evicted from the context no longer retains its large visual observation data; instead, only the corresponding thought text is preserved. Thanks to the sparsity of thoughts and the high compression ratio of textual representation, this method allows the agent to retain a long-term task context of over 2000 interaction steps at an extremely low token cost.

Ultra-Long Context Training. To further enhance the agent’s ability to process and utilize long-range dependencies, we have included ultra-long sequences in our training data, with context window lengths reaching up to 128k tokens. By training on data of this scale, the agent learns to reason and plan over much broader time horizons, thereby improving its performance on tasks that require long-term memory.

#### 3.5 Post-Training with Cross-Source Agentic Trajectories

Our ultimate goal is to build a generalist agent whose capabilities extend beyond the domain of gaming. Previous research has shown that incorporating diverse agentic trajectory data from various domains into the training process can significantly improve a model’s general cognitive abilities, such as reflection, planning, and decision-making [67]. Therefore, in the post-training stage, we not only use massive amounts of game data but also introduce a range of cross-source agentic trajectories, with the aim of cultivating a more comprehensive generalist agent.

We posit that the visual understanding and rapid-response capabilities learned through continual pre-training in game environments provide the agent with a solid "sensory-motor" foundation. Building on this, we introduce several types of non-gaming agent data to strengthen its higher-level cognitive functions:

- • Code Agent Trajectories: This data comes from agents performing programming tasks and includes the

###### Table 2 Training recipes in different stages of GameTARS agent.

Stage Pre-Training Post-Training Image Sizes (640,360)∼(1920,1080) (640,360)∼(1280,720) Image Tokens 208B 10B Text Tokens 326B 29B Total Tokens 526B 40B Game Number ∼100 ∼500 Game Hours ∼ 20k ∼ 2k Freeze ViT N Y Maximum Learning Rate 2e-5 1.4e-5 Minimum Learning Rate 2e-6 2e-6 Batch Size 2048 512 Maximum Token Sequence Length 32,000 128,000 Maximum Image Sequence Length 80 300 Training Steps 16,000 10,000 Warmup Steps 1000 750

entire process from problem comprehension and logical decomposition to code implementation. By learning from these trajectories, the agent can enhance its structured thinking and long-range logical reasoning abilities [45, 50, 81].

- • GUI Agent Trajectories: This data covers the process of completing specific tasks in desktop or web applications (e.g., booking a flight, managing files). It strengthens the agent’s ability to understand and execute multi-step instructions in non-gaming environments [48, 55].
- • MCP Agent Trajectories: This data records an agent’s process of information retrieval, document reading, content synthesis, and summarization to complete complex research tasks. Learning from this data can significantly enhance the agent’s ability to plan, reflect, and integrate knowledge within the ReAct loop [8, 20, 47, 54, 68].

By co-training on these diverse data sources alongside game data, we aim to achieve a synergistic learning effect: game data solidifies the foundational interaction skills, while cross-domain agentic data builds more abstract and general problem-solving capabilities on top of this foundation. This process ultimately facilitates the model’s transformation from a generalist game agent into a general computer-use agent.

### 4 Experiment

#### 4.1 Training Details

The training process for our agent comprises two main phases: continual pre-training and post-training (see

- Table 2 for detailed training parameters). During the early stages of this project, we explored a multi-stage training strategy, consisting of the following steps: 1) training on pure action trajectories; 2) continuing training on offline-synthesized ReAct trajectories; 3) further training on online-collected human "think-aloud" ReAct trajectories; and 4) fine-tuning with general multimodal data to enable capability annealing. However, our experimental results indicated that this staged approach had a detrimental effect on the model’s final performance. As a result, we adopted a unified, single-stage continual pre-training method, blending all data sources together, and trained the model on over 500 billion tokens. Subsequently, during the post-training phase, we fine-tuned the model on approximately 20 billion carefully curated high-quality tokens.

Model Details. We trained three versions of Game-TARS, namely Game-TARS-MoE-mini, Game-TARSMoE-Large, and Game-TARS-Dense. Game-TARS-MoE-mini is initialized from the pre-trained checkpoint of Seed-VL-1.5 [21], and leverages all of its continual-training data. The architecture includes a 632M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM with 2.5B active parameters. Game-TARS-MoE-large is a larger version of MoE. To ensure a fair comparison with other open-source models, we trained GameTARS-Dense using the Qwen2.5-VL-7B-Instruct checkpoint [3] and the VeOmni training strategies [38]. Due to training and data limitations, we only used game data for continual pretraining of the dense Game-TARS.

Dataset Details. Our training dataset represents an unprecedented scale, both in terms of breadth and depth. The game data includes over 500 popular games covering a wide range of genres. For detailed statistics on the types, durations, and diversity of the game data, please refer to Figure 4. The overall token distribution across different data sources is as follows:

- 1. Pure Action Game Trajectories: 61 billion (61B) tokens across 50 open-world games.
- 2. Offline-Synthesized ReAct Trajectories: 45 billion (45B) tokens across 50 games. These data and the pure-action section use initial game data from different sources.
- 3. Online Human Think-Aloud ReAct Trajectories: 127 billion (127B) tokens across 500 games. We selected games with a focus on reasoning in this part of the data and enhanced the diversity of the games.
- 4. Cross-Domain Agent Data: over 20 billion (20B) tokens including deep research, codex, and GUI trajectories [45, 76, 79].
- 5. General Multimodal Data: 200 billion (200B) tokens, including question-answering, image captioning, visual question answering (VQA), and video captioning [21, 56].

#### 4.2 Main Results

##### 4.2.1 Experimental Setups

To comprehensively evaluate the generalization capabilities of our agent model, we focused on testing its performance in games and environments that were not encountered during the training phase. We carefully selected a set of representative benchmarks to assess the agent’s adaptability and performance across a range of tasks and environments. The selected environments include:

- 1. Minecraft Although the model was exposed to a large amount of Minecraft data during training, the open-ended nature and task diversity of this environment make it an ideal platform for testing the model’s ability to perform new tasks in a familiar setting. We utilized a variety of tasks from Minedojo [19] and MCU [35] to ensure the model was tested on a broad spectrum of challenges, including task completion efficiency, flexibility, and reasoning capabilities in this open-world environment.
- 2. Web Games To ensure the novelty of the evaluation environment, we selected multiple games from the online platform Poki for testing. These games span across genres such as racing, puzzle, and other action types, providing an effective means of testing the model’s zero-shot generalization ability in entirely new visual and interaction paradigms. This evaluation serves to measure the model’s reasoning and decision-making performance in previously unseen games and environments.
- 3. Vizdoom Vizdoom is a semi-realistic 3D environment based on the classic game Doom, commonly used to evaluate an agent’s decision-making ability in fast-paced FPS scenarios that require rapid responses. By testing in this dynamic environment, we can assess the agent’s ability to react quickly to rapidly changing situations and evaluate its decision-making speed and accuracy under competitive, high-stakes conditions [78].
- 4. MiniWorld MiniWorld is a minimalist 3D environment simulator designed to test an agent’s navigation, object interaction, and basic physical reasoning skills. In this environment, the agent is required to perform basic tasks such as finding target objects, avoiding obstacles, and handling physical simulations. It serves as an effective testbed for evaluating the model’s ability to manage simple physical and interactive tasks [11].

These benchmarks encompass a variety of game types, task complexities, and interaction modalities, ensuring a comprehensive evaluation of the agent’s generalization ability in new, unseen environments. By testing across these diverse setups, we aim to verify the model’s performance in unfamiliar settings, providing insights into its potential for general-purpose intelligence.

- Table 3 Evaluation results in open-world Minecraft. The dense model of GameTARS is trained exclusively on game datasets, without incorporating datasets from other agents trajectories or general visual question-answering.

Embodied Tasks GUI Tasks Combat Tasks

Agents

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Steps ASR Steps ASR Steps ASR Policy-based Agents in Minecraft

VPT [4] .00 .00 377 6.0±11.4 .00 .20 398 0.8±3.3 .00 .15 396 3.6±7.7 STEVE-1 [34] .10 .25 384 8.0±17.0 .00 .30 391 3.2±8.4 .00 .20 395 3.9±12.0 Dreamer V3 [22] .00 .00 382 3.0±10.2 - - - - .00 .05 370 3.1±11.4

VLM-based Agents in Minecraft JARVIS-VLA [31] .25 .80 305 30.0±35.4 .35 .45 339 25.1±23.9 .00 .20 352 18.5±22.7 OmniJARVIS [72] .00 .25 363 24.4±31.1 .00 .00 393 3.0±7.5 .00 .20 371 8.5±17.9 OpenHA [74] .10 .45 287 30.1±13.9 .25 .30 314 32.5±9.2 .00 .30 316 31.9±13.7 UI-TARS-1.5 [55] .30 .60 290 42.1±20.4 .30 .50 320 36.7±17.2 .00 .60 346 31.0±16.4

Our Generalist Game Agent

GameTARS-Dense∗ .35 .85 373 50.4±20.7 .40 .55 406 39.1±27.5 .10 .70 372 38.1±24.6 GameTARS-MoE-mini .70 .85 188 72.0±21.5 .65 .55 288 55.4±17.6 .40 .75 271 66.1±25.2

##### 4.2.2 Evaluation on Open-World Minecraft

We first conducted a systematic evaluation of the agent’s instruction-following capability on unseen tasks within the Minecraft environment. For this purpose, we adopted the MCU benchmark [35], which contains more than 800 diverse and easy-to-verify tasks. By evaluating on this large and heterogeneous task set, we can quantitatively assess the agent’s ability to understand novel instructions and generalize them into concrete action sequences. To ensure fairness, we manually curated 800+ distinct initial environments such that each task could be completed within 600 steps. We further controlled the random seed to guarantee that the agent had not been exposed to these environments during training. Each task was paired with a unique natural language instruction that guided the agent’s behavior. Performance was measured using Average Success Rate and Average Completion Steps for evaluating instruction-following ability. As baselines, we compared against agents trained exclusively on Minecraft, including both policy-based and VLM-based approaches.

To provide a more fine-grained analysis, we grouped the tasks into three categories: (1) Embodied tasks, which evaluate exploration and object interaction in 3D space; (2) GUI tasks, which assess crafting and manipulation in 2D interfaces; and (3) Combat tasks, which focus on survival and combat capabilities. Each task was evaluated at least three times to reduce randomness in results.

As shown in Table 3, our generalist agent Game-TARS substantially outperforms all expert baselines trained solely on Minecraft. In particular, the Game-TARS-MoE-mini variant achieved the highest success rates across all task categories, with significant improvements in both instruction-following accuracy and sample efficiency. These results confirm that large-scale continual pretraining with a unified action space enables strong generalization even in complex, open-world environments.

A closer inspection of the results highlights several key findings. First, Game-TARS exhibits particularly strong performance in GUI tasks, where the success rate reached 55.0%, surpassing all prior baselines by a large margin. This demonstrates the effectiveness of the unified action space in handling tasks that require reasoning and execution across both 2D graphical interfaces and 3D embodied environments. Second, in embodied tasks, Game-TARS achieved a remarkable 72.0% success rate while completing tasks with substantially fewer steps, underscoring its ability to generalize complex exploration and interaction behaviors. Finally, in combat tasks, the model achieved 66.0% success, more than doubling the performance of the strongest VLM-based baseline.

Taken together, these results validate our central hypothesis: large-scale continual pretraining with a unified, device-level action space not only boosts overall task success but also enhances the agent’s adaptability to diverse instruction modalities. Game-TARS therefore establishes a new state-of-the-art in open-world Minecraft benchmarks and provides compelling evidence of the viability of generalist game agents.

##### 4.2.3 Evaluation on Unseen Benchmarks and Simulators

[Figure 15]

- Figure 7 Evaluation results in Web 3D games on Poki. Evaluation results in Web 3D games on Poki. Metrics include checkpoint time (Race), levels completed (Jump-Only), short-horizon score (Temple Run), and survival score (Airplane Flying).

[Figure 16]

- Figure 8 Evaluation results in FPS Vizdoom benchmark across four maps [78]. The reported metric is the average episode reward. Game-TARS-mini significantly outperforms all baselines.

[Figure 17]

Figure 9 Evaluation results in Miniworld Simulator [11].

The goal of Game-TARS is not to achieve a higher score on trained games, but to provide a good foundation model for a generalist game agent. Therefore, further evaluation is conducted on unseen games, including web games, FPS Doom, and 3D simulators. We first evaluated our model in browser-based 3D web games to test its zero-shot generalization ability in entirely novel visual and interaction paradigms. We selected four representative games from the Poki platform2: Race (racing), Jump-Only (action), Temple Run (endless running), and Airplane Flying (3D navigation). Evaluation metrics were task-specific: in Race, we measured the time to reach the first checkpoint (lower is better); in Jump-Only, the number of levels completed within 20 seconds; in Temple Run, the score achieved within 20 seconds; and in Airplane Flying, the cumulative score before failure.

2https://poki.com/

[Figure 18]

Figure 10 (Left) Inference-time Scaling Experiments. Game-TARS can improve the task success rates through exploration in inference-time steps. (Right) Training-time Scaling Experiments. The performance of Game-TARS improves as the training data increases, including in-domain, out-of-domain, and other agentic trajectory tokens.

All games were wrapped into a browser sandbox environment to ensure a controlled and reproducible evaluation3. We used GPT-5 and Human players as baselines. For the human baseline, we recruited at least four individuals with no prior experience in the selected games and reported their best scores. For GPT-5 and our agent, we conducted at least five runs per game and reported the best-of-5 (Bo5) results. Both models were provided with identical system prompts, including basic game rules and action space definitions.

As illustrated in Figure 7, Game-TARS-mini demonstrated competitive generalization compared to human players, achieving near-human or superior performance across most games. Notably, the model outperformed fresh human players in Jump-Only and Airplane Flying, highlighting its ability to adapt to novel tasks and mechanics without task-specific training.

We further evaluated our agent in Vizdoom [78], a classic FPS environment commonly used for benchmarking decision-making under fast-paced and adversarial conditions. We selected four standard maps: Battle-1, Battle-2, Defend the Line, and Defend the Center. The evaluation metric was the cumulative reward achieved per episode. Baselines included leading general-purpose models such as GPT-5 [46], Claude-4-Sonnet [2], and Gemini-2.5-Pro [12].

Results in Figure 8 show that Game-TARS-mini consistently outperformed all baselines across almost all maps, achieving substantial gains in both survival and combat effectiveness. Beyond raw scores, qualitative analysis revealed that the agent exhibited advanced FPS behaviors, including effective corner-peeking, evasive maneuvers to dodge projectiles, and adaptive targeting strategies. These results underscore the strong transferability of behaviors learned from diverse game data to previously unseen FPS environments.

To further assess the agent’s ability to generalize to tasks requiring navigation, memory, and exploration, we evaluated it in the MiniWorld 3D simulator. This environment presents simplified but diverse tasks involving spatial reasoning, obstacle avoidance, and object interaction. Across nearly all tested maps, Game-TARS demonstrated robust performance, successfully navigating new environments and completing tasks despite having no prior exposure to MiniWorld during training.

Together, these results confirm the broad generalization capability of Game-TARS across unseen environments, ranging from browser-based games and FPS simulators to navigation-focused 3D environments. The findings provide strong evidence that large-scale continual pretraining with a unified action space enables the emergence of transferable skills applicable far beyond the training distribution.

3Due to the inference latency of the model, we packaged these web games into a sandbox and used time-pause to ensure causal interaction.

###### Table 4 Ablation experiments on different thinking patterns in GameTARS across different evaluation benchmarks.

Method Minecraft Doom Race ↓ 2048 Avg. Tokens ↓

No-thinking 55% 13.2 12.4s 445 22 Greedy Thinking 45% 7.2 11.3s 986 68 Efficient Thinking 63% 11.5 9.3s 924 37

#### 4.3 Analysis and Discussions

In this section, we present a set of analyses to provide deeper insights into the design and performance of Game-TARS. Our discussion covers three main aspects: sparse thinking, loss decay across training paradigms, and scaling experiments. These analyses aim to ensure a fair and systematic evaluation of the model’s capabilities and limitations.

##### 4.3.1 Scaling Experiments on Training Datasets and Inference Steps

We first conducted scaling experiments on the continual pre-training datasets to examine how performance improves with increasing amounts and diversity of training data. The results are summarized in Figure 10, and several key insights emerge:

- 1. Specified action space performs well with small datasets but is hard to scale. When using visual grounding–based actions (e.g., GUI action spaces), agents achieve relatively strong performance with limited training data. However, such action spaces require extensive preprocessing of raw trajectories and do not naturally transfer across heterogeneous games. As a result, their scalability and ceiling performance are inherently constrained (blue curve).
- 2. Unified action space requires more data initially but scales better. By contrast, the proposed unified device-level action space is highly scalable but not natively supported by pre-trained VLMs. As a result, its performance lags behind GUI-based approaches when trained on small datasets. However, after pre-training on large-scale game data, its upper-bound performance approaches that of GUI-based action spaces (orange curve).
- 3. Cross-game scaling brings substantial gains. The unified action space enables consistent representation across diverse games, allowing large-scale aggregation of heterogeneous data. Incorporating 200B additional tokens from other games led to a significant improvement in Minecraft performance, boosting the success rate from 42% to 65%. This observation echoes the Bitter Lesson in AI: general and simple representations, when combined with scaling in data and compute, yield stable and long-term improvements [60].
- 4. Multimodal and cross-domain agentics data further enhance generalization. Extending pre-training to include multimodal data (e.g., captioning, VQA, video understanding) and agent trajectories from other domains led to an additional improvement, pushing performance to 72%. This finding demonstrates that generalist agent models can benefit from heterogeneous sources of action–trajectory data, acquiring broadly transferable decision-making capabilities.

Together, these results confirm the scalability advantages of the unified action space: while GUI-based representations may provide a short-term advantage in low-data regimes, only the unified paradigm enables sustained performance gains through cross-game and cross-domain scaling.

##### 4.3.2 Ablation on sparse thinking

To better understand the role of reasoning in decision-making, we conducted ablation experiments on different thinking patterns, including No-thinking, Greedy Thinking, and sparse thinking. The results are reported in Table 4.

The necessity of explicit reasoning is strongly task-dependent. In reasoning-intensive games such as 2048, where planning and long-horizon inference are critical, incorporating thinking at every step significantly

- Table 5 Ablation experiments on the validation of decaying loss. The ASR and N-Acc represents average success rates on online evaluations and non-repetitive accuracy on offline datasets, respectively.

Online Embodied Tasks Online Combat Tasks Online GUI Tasks Offline Evaluation ASR Diversity Steps ASR Diversity Steps ASR Diversity Steps Accuracy N-Acc

Model

w/o Decaying Loss 44.90% 60% 168 14.54% 50% 305 37.75% 80% 244 59% 12% w/ Decaying Loss 46.15% 90% 172 22.22% 60% 247 42.26% 100% 164 47% 39%

improves performance (986 average tokens consumed per step). By contrast, in fast-paced environments such as Doom, excessive reasoning (Greedy Thinking) introduces latency and even degrades performance. One primary source of degradation is hallucinated reasoning loops, where the agent incorrectly assumes that certain goals have already been achieved and prematurely terminates actions.

The No-thinking baseline performs relatively well in reaction-heavy games like Doom but fails to generalize to tasks that require compositional reasoning (e.g., crafting tasks in Minecraft or the strategic planning required in 2048). On the other hand, Greedy Thinking—forcing the agent to reason at every step—incurs substantial computational overhead, increases inference cost, and amplifies the risk of cascading reasoning errors.

Our proposed sparse thinking strategy achieves the best trade-off between performance and inference efficiency. The agent learns to selectively engage in reasoning at key decision points (e.g., planning in Minecraft crafting tasks or assessing critical survival situations in Doom), while re-using or following previously established reasoning in routine steps. This reduces redundant computation, lowers average token consumption (37 tokens/step vs. 68 tokens/step in Greedy Thinking), and achieves state-of-the-art performance across most benchmarks (63% success rate in Minecraft, 924 points in 2048).

Interestingly, qualitative trajectory analysis shows that the agent develops an implicit meta-cognition mechanism: it chooses when to reason and when to act reflexively. For instance, in Minecraft crafting sequences, the agent reasons carefully at the start of a task (to identify needed resources and steps) but executes subsequent mining or crafting actions without redundant re-planning. Similarly, in 2048, it reasons at every move due to the inherent combinatorial structure of the game, while in VizDoom it minimizes reasoning to maintain responsiveness.

These findings highlight that reasoning is not universally beneficial; rather, its effectiveness depends on the interaction between task structure and agent inference cost. Sparse thinking emerges as a general strategy that balances adaptability, efficiency, and robustness, enabling the agent to approximate human-like decision-making where deliberation is applied selectively at critical junctures.

##### 4.3.3 Validation on Continual Pre-Training Loss

We further investigated the impact of applying a decaying continual pre-training loss. The analysis was conducted from both training dynamics and downstream evaluation perspectives (Table 5).

From a purely optimization perspective, introducing a decaying loss initially leads to a slightly higher absolute training loss compared to the non-decaying baseline. Nevertheless, the overall decreasing trend remains stable, and both approaches converge to comparable final loss values. This observation indicates that the decaying strategy does not compromise convergence stability, while potentially reshaping the optimization trajectory.

We then conducted ablation studies on downstream online environments, measuring task-level Average Success Rate (ASR), trajectory diversity, and average completion steps. The results reveal several advantages of the decaying loss: 1) Improved task success rates across embodied, combat, and GUI tasks, with particularly large gains in combat scenarios (from 14.5% to 22.2%). 2) Substantially shorter trajectories, indicating that agents are less likely to become stuck or generate redundant action loops. 3) Enhanced behavioral diversity, especially in GUI and embodied tasks, where diversity increased from 60–80% to nearly 100%.

These improvements suggest that decaying loss helps mitigate the tendency of agents to fall into repetitive behaviors, thereby producing more robust and efficient action sequences.

To further probe this effect, we performed offline action prediction experiments across 1,000 trajectories containing more than 100k steps. We report two metrics: 1) Average action prediction accuracy across all steps. 2) Non-repetitive accuracy, i.e., accuracy on steps where the predicted action differs from the previous one.

Interestingly, while the non-decaying baseline achieves higher overall accuracy (59% vs. 47%), the decaying loss yields a dramatic improvement in non-repetitive accuracy (39% vs. 12%, a +28% absolute gain). This indicates that models trained without decaying loss tend to exploit dataset bias by repeatedly copying the previous action, “hacking” the accuracy metric without genuinely improving decision quality. In contrast, decaying loss discourages such degenerate strategies, forcing the agent to learn meaningful state-dependent action prediction.

The ablation results demonstrate that although decaying loss sacrifices some global prediction accuracy, it provides substantial benefits in downstream performance by reducing repetition, increasing action diversity, and improving robustness across online tasks. This aligns with the broader goal of continual pre-training: to shape learning signals in a way that promotes adaptive, generalizable behaviors rather than exploiting dataset artifacts.

### 5 Related Works

Our work builds upon and extends several key research areas in artificial intelligence: specialized game-playing agents, the emerging paradigm of generalist agents, and agents designed for broader computer interaction tasks. We situate Game-TARS by drawing a critical distinction based on the agent’s action space, arguing that a unified, human-friendly paradigm is fundamental to achieving true scalability and generality.

#### 5.1 Game Agents

The history of AI is deeply intertwined with games as benchmarks for intelligence. Foundational work in closed-world games demonstrated superhuman performance but relied on highly specialized action spaces. Deep Q-Networks (DQN) mastered numerous Atari games from raw pixels using a low-dimensional, discrete action space [40]. AlphaGo and its successors achieved world-champion level Go play via deep neural networks and Monte Carlo Tree Search, with actions tied to the game’s formal rules [58]. Similarly, AlphaStar excelled in StarCraft II using a custom, game-specific API, limiting transferability [65]. While highly successful, these agents’ tight coupling of action representation to specific environments remains a bottleneck for generalization.

The open-world environment of Minecraft has become a key testbed for more generalizable agents. VPT pioneered learning behavioral priors from large-scale video data using the native human interface (keyboard/mouse), aligning with our approach, though training was confined to Minecraft [4]. Other methods like DreamerV3 learn world models via RL, solving hard-exploration tasks without human data but lacking demonstrated cross-domain generalization [22].

The rise of pre-trained vision-language models (VLMs) and large language models (LLMs) has enabled new possibilities for multi-task game agents in open worlds. Voyager [66] leverages LLMs for long-horizon planning but executes actions via high-level APIs (e.g., JavaScript code), creating a semantic gap by learning to sequence functions rather than acquiring sensorimotor skills [33, 70, 71, 83, 86]. Cradle demonstrated strong instruction-following across multiple games using a unified agentic workflow [61]. Other approaches achieve strong performance via VLM-based fine-tuning (SFT) or RL within specific games [18, 29, 55, 72, 84]. Furthermore, general AI models like Gemini-2.5 [12], GPT-5 [46], and Claude 4 [2] are increasingly evaluated on games (e.g., Chess, Pokémon), highlighting gaming as a crucial benchmark for foundational model capabilities [57, 85].

#### 5.2 Generalist Agents

The pursuit of a single agent capable of mastering multiple domains has led to the development of generalist architectures. Early DQN research showed that a unified RL algorithm could achieve superhuman performance across many Atari games, albeit requiring separate training runs for each [40]. Gato was a seminal work

demonstrating a multi-modal, multi-task, multi-embodiment agent trained with a single set of weights by serializing all data into a token sequence for a Transformer [53]. However, its heterogeneous, discretized action space still required domain-specific tokenization, falling short of a truly universal interface. More recently, the SIMA project shares a core philosophy with Game-TARS: training an agent to follow natural language instructions across diverse 3D environments using a generic, human-like interface (pixels in, keyboard/mouse out) [51]. This marks a critical shift toward a unified action paradigm. However, initial SIMA agents were evaluated primarily on short-horizon tasks and limited in-domain games, underscoring the learning challenge. VPT built a foundational model for reinforcement learning in an open world by pre-training on IDM and video data from the internet. However, this foundational model can only be used for Minecraft and is difficult to generalize to other game scenarios [4]. Recently, with the rise of pre-trained models such as LLM [25, 39, 64] and VLM [2, 12, 46], developing a pre-trained foundation model for agents has become mainstream, such as embodied foundation models [9, 17, 24, 42, 62] and computer-use foundation models [48, 63, 67, 69]. Game-TARS builds on this shared vision, demonstrating enhanced performance and scalability through large-scale continual pre-training on vision language models.

#### 5.3 Broader AI Agents

From a broader perspective, there is significant potential for AI agents based on VLM or LLM to be implemented in various fields, including computer use and embodied agents [5, 6, 17, 28, 43]. Research into GUI and Web Agents (e.g., using Mind2Web [15] or UI-TARS [67]) often focuses on grounding high-level plans to specific UI elements [23, 30, 36, 37, 68, 77, 80]. By operating directly on low-level keyboard/mouse outputs, Game-TARS learns an end-to-end policy that implicitly addresses grounding. Similarly, agents for software engineering like Claude [2] and SWE-agent [27, 45, 81] tackle complex, logic-driven tasks. Protocols such as MCP [1] have introduced standardized mechanisms for flexible tool integration, enabling agents to seamlessly interact with search engines, file parsers, or external APIs [20, 32, 59, 73]. Incorporating data from such domains in post-training enhances Game-TARS’ high-level reasoning, complementing its foundational sensorimotor skills.

### 6 Conclusions

We present Game-TARS, a generalist game agent utilizing a unified, scalable action space based on native keyboard-mouse inputs. This human-native paradigm bypasses the limitations of API/GUI approaches, enabling large-scale continual pre-training (>500B tokens) across diverse domains (game trajectories, ai agents trajectories, multimodal data). Game-TARS demonstrates strong instruction-following capabilities and employs an efficient Sparse Thinking strategy, balancing reasoning depth with inference cost. Experiments show Game-TARS achieves 2x SOTA success rate in Minecraft, nears human-level generalization in unseen web games, and outperforms leading models (GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet) on FPS benchmarks. Scaling results confirm that simple, scalable action representations paired with massive pre-training offer a promising path toward generalist computer-use agents.

### Contributions

Authors Zihao Wang1,2, Xujing Li1, Yining Ye1, Junjie Fang1, Haoming Wang1, Longxiang Liu1, Shihao Liang1, Junting Lu1, Zhiyong Wu1, Jiazhan Feng1, Wanjun Zhong1, Zili Li1, Yu Wang1, Yu Miao1, Bo Zhou1, Yuanfan Li1, Hao Wang1, Zhongkai Zhao1, Faming Wu1, Zhengxuan Jiang3, Weihao Tan1, Heyuan Yao1, Shi Yan1, Xiangyang Li1, Yitao Liang2, Yujia Qin1, Guang Shi1

Afflications 1ByteDance Seed, 2Peking University, 3M-A-P Acknowledgment We greatly thank Muyao Li for discussions. And we appreciate Cheng Yuan and Hao Chen’s efforts on the Game-TARS website.

### References

- [1] Anthropic. Introducing the model context protocol, 2024. URL https://www.anthropic.com/news/ model-context-protocol.
- [2] anthropic. Introducing claude 4, 2025. URL https://www.anthropic.com/news/claude-4.
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

- [4] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems, 35:24639–24654, 2022.

- [5] Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh. Rt-h: Action hierarchies using language. arXiv preprint arXiv:2403.01823, 2024.

- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

- [7] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

- [8] ByteDance. Seed-thinking-1.6, 2025. URL https://seed.bytedance.com/zh/seed1_6.
- [9] Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025.

- [10] Yuheng Cheng, Ceyao Zhang, Zhengwen Zhang, Xiangrui Meng, Sirui Hong, Wenhao Li, Zihao Wang, Zekai Wang, Feng Yin, Junhua Zhao, et al. Exploring large language model based intelligent agents: Definitions, methods, and prospects. arXiv preprint arXiv:2401.03428, 2024.

- [11] Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo de Lazcano, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and Jordan Terry. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. CoRR, abs/2306.13831, 2023.

- [12] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

- [13] Pim De Haan, Dinesh Jayaraman, and Sergey Levine. Causal confusion in imitation learning. Advances in neural information processing systems, 32, 2019.

- [14] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv e-prints, pages arXiv–2409, 2024.

- [15] Xiang Deng, Kelvin Guu, Panupong Pasupat, Afra Akyürek, Sheng Zhuang, Wenlong Chen, Tatsunori Hashimoto, Kelvin Guu, and Percy Liang. Mind2web: Towards a generalist agent for the web. In NeurIPS Datasets and Benchmarks, 2023. URL https://arxiv.org/abs/2306.06070.

- [16] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.

- [17] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

- [18] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, et al. Palm-e: An embodied multimodal language model. 2023.

- [19] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35:18343–18362, 2022.

- [20] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536, 2025.

- [21] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

- [22] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

- [23] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024.

- [24] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π0.5: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054,

2025. URL https://arxiv.org/abs/2504.16054.

- [25] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

- [26] Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. arXiv preprint arXiv:2210.03094, 2(3):6, 2022.

- [27] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

- [28] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Grace Lam, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. In Pulkit Agrawal, Oliver Kroemer, and Wolfram Burgard, editors, Proceedings of The 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pages 2679–2713. PMLR, 06–09 Nov

2025. URL https://proceedings.mlr.press/v270/kim25c.html.

- [29] Hao Li, Xue Yang, Zhaokai Wang, Xizhou Zhu, Jie Zhou, Yu Qiao, Xiaogang Wang, Hongsheng Li, Lewei Lu, and Jifeng Dai. Auto mc-reward: Automated dense reward design with large language models for minecraft. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16426–16435, June 2024.

- [30] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981, 2025.

- [31] Muyao Li, Zihao Wang, Kaichen He, Xiaojian Ma, and Yitao Liang. Jarvis-vla: Post-training large-scale vision language models to play visual games with keyboards and mouse. arXiv preprint arXiv:2503.16365, 2025.

- [32] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Torl: Scaling tool-integrated rl, 2025. URL https://arxiv.org/abs/ 2503.23383.
- [33] Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, and Liqiang Nie. Optimus-1: Hybrid multimodal memory empowered agents excel in long-horizon tasks. Advances in neural information processing systems, 37: 49881–49913, 2024.

- [34] Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, and Sheila McIlraith. Steve-1: A generative model for text-to-behavior in minecraft. Advances in Neural Information Processing Systems, 36:69900–69929, 2023.

- [35] Haowei Lin, Zihao Wang, Jianzhu Ma, and Yitao Liang. Mcu: A task-centric framework for open-ended agent evaluation in minecraft. arXiv preprint arXiv:2310.08367, 2023.

- [36] Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, and Jiaya Jia. Arpo: End-to-end policy optimization for gui agents with experience replay. arXiv preprint arXiv:2505.16282, 2025.

- [37] Run Luo, Lu Wang, Wanwei He, Longze Chen, Jiaming Li, and Xiaobo Xia. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.

- [38] Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, et al. Veomni: Scaling any modality model training with model-centric distributed recipe zoo. arXiv preprint arXiv:2508.02317, 2025.

- [39] AI Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. https://ai.meta.com/blog/llama-4-multimodal-intelligence/, 2025.

- [40] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

- [41] Meredith Ringel Morris, Jascha Sohl-Dickstein, Noah Fiedel, Tris Warkentin, Allan Dafoe, Aleksandra Faust, Clement Farabet, and Shane Legg. Levels of agi for operationalizing progress on the path to agi. arXiv preprint arXiv:2311.02462, 2023.

- [42] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv: 2503.14734, 2025.

- [43] Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, et al. Open x-embodiment: Robotic learning datasets and rt-x models. arXiv preprint arXiv:2310.08864, 2023.

- [44] OpenAI. Gpt-4v(ision) system card. https://openai.com/index/gpt-4v-system-card/, 2023. Accessed: 202504-23.
- [45] openai. Introducing swe-bench verified, 2024. URL https://openai.com/index/ introducing-swe-bench-verified/.
- [46] OpenAI. Introducing gpt 5, 2025. URL https://openai.com/index/introducing-gpt-5/.
- [47] OpenAI. Introducing deep research - openai. https://openai.com/index/introducing-deep-research/, 2025.
- [48] OpenAI. Computer-using agent (cua). https://openai.com/index/computer-using-agent/, 2025. Research preview / blog.
- [49] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

- [50] Qwen. Qwen-3-coder. https://qwenlm.github.io/blog/qwen3-coder, 2025.
- [51] Maria Abi Raad, Arun Ahuja, Catarina Barros, Frederic Besse, Andrew Bolt, Adrian Bolton, Bethanie Brownfield, Gavin Buttimore, Max Cant, Sarah Chakera, et al. Scaling instructable agents across many simulated worlds. arXiv preprint arXiv:2404.10179, 2024.

- [52] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR, 2023.

- [53] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.

- [54] Bytedance Seed. Seed-thinking-v1. 5: Advancing superb reasoning models with reinforcement learning. Technical report, Technical report, ByteDance, 2025. URL https://github. com/ByteDance-Seed . . . , 2025.
- [55] ByteDance Seed. Ui-tars-1.5. https://seed-tars.com/1.5, 2025.
- [56] ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.

- [57] Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, et al. Korgym: A dynamic game platform for llm reasoning evaluation. arXiv preprint arXiv:2505.14552, 2025.

- [58] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354–359, 2017.

- [59] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and JiRong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning. arXiv preprint arXiv:2503.05592, 2025.

- [60] Richard Sutton. The bitter lesson. Incomplete Ideas (blog), 13(1):38, 2019.

- [61] Weihao Tan, Wentao Zhang, Xinrun Xu, Haochong Xia, Ziluo Ding, Boyu Li, Bohan Zhou, Junpeng Yue, Jiechuan Jiang, Yewen Li, et al. Cradle: Empowering foundation agents towards general computer control. arXiv preprint arXiv:2403.03186, 2024.

- [62] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025.

- [63] Kimi Team. Kimi k2: Open agentic intelligence, 2025. URL https://arxiv.org/abs/2507.20534.
- [64] Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [65] Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. nature, 575(7782):350–354, 2019.

- [66] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

- [67] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025.

- [68] Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, and Heng Ji. Acting less is reasoning more! teaching model to act efficiently, 2025. URL https://arxiv.org/abs/2504.14870.
- [69] Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, et al. Opencua: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123, 2025.

- [70] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Shawn Ma, and Yitao Liang. Describe, explain, plan and select: interactive planning with llms enables open-world multi-task agents. Advances in Neural Information Processing Systems, 36:34153–34189, 2023.

- [71] Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Jinbing Hou, Bowei Zhang, Haowei Lin, Zhaofeng He, Zilong Zheng, Yaodong Yang, et al. Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

- [72] Zihao Wang, Shaofei Cai, Zhancun Mu, Haowei Lin, Ceyao Zhang, Xuejie Liu, Qing Li, Anji Liu, Xiaojian Shawn Ma, and Yitao Liang. Omnijarvis: Unified vision-language-action tokenization enables open-world instruction following agents. Advances in Neural Information Processing Systems, 37:73278–73308, 2024.

- [73] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, and Yitao Liang. Rat: Retrieval augmented thoughts elicit context-aware reasoning in long-horizon generation. arXiv preprint arXiv:2403.05313, 2024.

- [74] Zihao Wang, Muyao Li, Kaichen He, Xiangyu Wang, Zhancun Mu, Anji Liu, and Yitao Liang. Openha: A series of open-source hierarchical agentic models in minecraft. arXiv preprint arXiv:2509.13347, 2025.

- [75] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, F. Xia, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Neural Information Processing Systems, 2022.

- [76] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

- [77] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024.

- [78] Marek Wydmuch, Michał Kempka, and Wojciech Jaśkowski. Vizdoom competitions: Playing doom from pixels. IEEE Transactions on Games, 2018. IEEE Transactions on Games outstanding paper award 2022.

- [79] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

- [80] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.

- [81] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. arXiv preprint arXiv:2405.15793, 2024.

- [82] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

- [83] Haoqi Yuan, Chi Zhang, Hongcheng Wang, Feiyang Xie, Penglin Cai, Hao Dong, and Zongqing Lu. Skill reinforcement learning and planning for open-world long-horizon tasks. arXiv preprint arXiv:2303.16563, 2023.

- [84] Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems, 37:110935–110971, 2024.

- [85] Alex L Zhang, Thomas L Griffiths, Karthik R Narasimhan, and Ofir Press. Videogamebench: Can vision-language models complete popular video games? arXiv preprint arXiv:2505.18134, 2025.

- [86] Ceyao Zhang, Kaijie Yang, Siyi Hu, Zihao Wang, Guanghe Li, Yihang Sun, Cheng Zhang, Zhaowei Zhang, Anji Liu, Song-Chun Zhu, et al. Proagent: Building proactive cooperative ai with large language models. arXiv preprint arXiv:2308.11339, 2023.

- [87] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

## Appendix

### A Rollout Trajectories

- A.1 Rollouts in Minecraft

[Figure 19]

Figure 11 Game-TARS has mastered advanced combat skills. In battles with creepers, he avoids damage from creeper explosions by constantly retreating and also uses jumping strikes to increase his damage.

[Figure 20]

Figure 12 Game-TARS can use tools and interact with objects in the environment.

[Figure 21]

- Figure 13 Game-TARS have good long-term memory, allowing them to remember the movement trajectory of foxes during the process of killing them, and to track the foxes over a long period through navigation and exploration skills.

[Figure 22]

- Figure 14 Game-TARS has great action consistency and can break obsidian blocks that require a 10-second long press to obtain.

[Figure 23]

- Figure 15 Game-TARS can use unified actions to freely switch and interact between the embodied 3D interface and the 2D GUI interface.

- A.2 Rollouts in Unseen Games

###### Table 6 Rollout trajectory of Game-TARS on Poki Plane game. We skip some timestamps.

t Observation Model Response

[Figure 24]

Thought: Just entered a new scene, and there are obstacles ahead. I need to quickly adjust the direction to avoid them; I can’t let the plane crash, or else the game will be over. Action: no_op

0

...

[Figure 25]

6 Action: keyPress(d)

...

[Figure 26]

Thought: The position of the obstacle ahead has changed, need to adjust direction to the right, maintain distance from the obstacle, and continue flying forward. Action: keyPress(d)

14

...

[Figure 27]

Thought: There are new obstacles appearing again, need to adjust the direction to the left to avoid this, cannot let the plane hit any obstacles. Action: keyPress(a)

21

...

[Figure 28]

Thought: The obstacles in front are arranged quite densely, so I need to adjust the direction to the right to avoid them; I can’t let the plane hit them, otherwise the game will be over. Action: keyPress (d)

35

...

###### Table 7 Rollout trajectory of Game-TARS on Temple Run game. We skip some timestamps.

t Observation Model Response

- 0

[Figure 29]

Thought: A diamond-shaped obstacle appeared ahead, I need to jump up quickly to avoid it, otherwise I will crash and lose health. Action: keyPress(arrowup)

- 1 Action: no_op

[Figure 30]

...

[Figure 31]

Thought: There was a stone wall ahead, you had to crouch down to pass through, otherwise you would be blocked, so quickly press the crouch key. Action: keyPress (arrowdown)

12

...

[Figure 32]

24 Action: keyPress(arrowleft)

...

[Figure 33]

Thought: There are diamonds and stone pillars ahead; be careful to avoid the stone pillars while jumping over the diamonds, and stay focused. Action: no_op

37

...

[Figure 34]

Thought: There is a gap ahead, you need to avoid it, otherwise you will fall. Press the right button to ensure a safe passage. Action: keyPress(arrowright)

53

...

[Figure 35]

67 Action: keyPress(arrowleft)

...

[Figure 36]

82 Action: keyPress(arrowup)

...

[Figure 37]

109 Action: keyPress(arrowup)

...

###### Table 8 Rollout trajectory of Game-TARS on Poki Race game. We skip some timestamps.

t Observation Model Response

[Figure 38]

Thought: The game interface has now loaded, and I need to start driving the vehicle. First, I have to hold down the acceleration key to get the car moving; only then can I start the timer. I must ensure that I maintain acceleration from the very beginning and not let my speed drop so that I can reach the finish line as quickly as possible. Action: keyPress(arrowup)

0

...

- 6

[Figure 39]

Thought: I see the "GO!" signal now, it’s time to accelerate with all my might and pick up speed. The road ahead looks straight, so I’ll stay in the middle lane to avoid hitting the trees or other cars on either side. I need to pay attention to the vehicles ahead, but for now, I’ll focus on accelerating and aim to get my speed up right from the start. Action: keyPress(arrowup)

- 7 Action: keyPress(arrowup)

[Figure 40]

...

[Figure 41]

Thought: The speed has already reached over 100, and the road ahead is still very straight, so continue to accelerate. Now I need to pay attention to the vehicles in front, but currently there are no other cars, so I can keep accelerating. At the same time, I need to ensure that I’m in the center of the lane; this way it’s both safe and helps maintain speed, avoiding veering off the road which could lead to a slowdown. Action: keyPress(arrowup)

17

...

[Figure 42]

Thought: The car on the right has passed, now return to the middle lane and continue to accelerate. The speed is already 200; maintain this speed. The road ahead is still straight, as long as I don’t deviate from the road, I can keep going at high speed. I need to pay attention and look further ahead to anticipate any potential situations. Action: keyPress(arrowup)

56

...

- 82 Action: keyPress(arrowup)

[Figure 43]

- 83

[Figure 44]

Thought: The car in front has already passed, now return to the middle lane and continue to accelerate. The speed is still 200, the road remains straight, and there are no other vehicles around, so I can maintain this speed. I need to pay attention to the mini-map to ensure I’m in the center of the lane; this way it’s both safe and allows me to keep my speed, avoiding deceleration due to drifting off the road. Action: keyPress(arrowup)

- 84 Action: keyPress(arrowup)

[Figure 45]

...

- 100

[Figure 46]

Thought: The road ahead is starting to turn, so I need to adjust my direction in advance and stay in the center of the road. When turning, I can’t reduce speed too much, so I need to steer early, turn slowly while continuing to accelerate, and try to maintain my speed. This way is both safe and allows me to keep the fastest speed. Action: keyPress(arrowup, arrowright)

- 101 Action: keyPress(arrowup)

[Figure 47]

...

[Figure 48]

130 Action: keyPress(arrowup, arrowright)

...

[Figure 49]

150 Action: keyPress(arrowup, arrowright)

...

[Figure 50]

Thought: The runway is still very straight, and the speed is maintained at 200. There are still no other vehicles around, so I can continue to accelerate. Pay attention to the mini-map to ensure I’m in the center of the road, while also looking further ahead to anticipate any potential situations. This way, I can maintain the fastest speed. Action: keyPress(arrowup)

180

...

[Figure 51]

220 Action: keyPress(arrowup, arrowleft)

...

### B Prompts B.1 Prompts of Game-TARS on Minecraft

###### System Prompt of Game-TARS on Unseen Poki Plane Game

You are an AI agent performing tasks in Minecraft based on given instructions, action history, and visual observations (screenshots). Your goal is to take the next optimal action to complete the task. ## Output Format Thought: ... Action: ...

## Action Space

- * mouseMove(dx, dy) # Move the mouse position; dx and dy represent horizontal and vertical movement, respectively.
- * mouseClick(’left’ or ’right’) # left click or right click the mouse

- - left # Attack; In GUI, pick up the stack of items or place the stack of items in a GUI cell; when used as a double click (attack - no attack - attack sequence), collect all items of the same kind present in inventory as a single stack.
- - right # Place the item currently held or use the block the player is looking at. In GUI, pick up the stack of items or place a single item from a stack held by mouse.

* keyPress(keys) # press the keyboard buttons

- - w # Move forward.
- - s # Move backward.
- - a # Strafe left.
- - d # Strafe right.
- - e # Open or close inventory and the 2x2 crafting grid.
- - space # Jump.
- - q # Drop a single item from the stack of items the player is currently holding. If the player presses ctrl-Q then it drops the entire stack. In the GUI, the same thing happens except to the item the mouse is hovering over.
- - 1-9 # Switch active item to the one in a given hotbar cell.
- - left.ctrl # Move fast in the current direction of motion.
- - left.shift # Move carefully in current direction of motion. In the GUI it acts as a modifier key: when used with attack it moves item from/to the inventory to/from the hotbar, and when used with craft it crafts the maximum number of items possible instead of just 1.

- * no_op # wait and do not interact with the world

If multiple actions are activated, use and connect. Use Chinese in Thought part. Provide a brief plan in the Thought section, specifying your next move and objective. If this step does not require thought, you can only output Action. Your history thoughts will accumulate continuously in history conversations.

## User Instruction {instruction}

#### B.2 Prompts of Game-TARS on Unseen Games

###### System Prompt of Game-TARS on Unseen Poki Plane Game

You are an AI agent playing "Plane Avoid Barrier" game, based on given instructions, action history, and visual observations (screenshots). Your goal is to take the next optimal action to achieve higher scores.

## Output Format Thought: ... Action: ... (if thought is necessary in this frame, you need to judge) or Action: ... (low level action is enough)

## Action Space

- * keyPress(keys) # press the keyboard buttons

- - a # Strafe left.
- - d # Strafe right.

- * no_op # wait and do not interact with the world.

## User Instruction Avoid the obstacle ahead.

###### System Prompt of Game-TARS on Unseen Poki Race Game

You are an AI agent playing "Drive Freedom" game, based on given instructions, action history, and visual observations (screenshots). Your goal is to take the next optimal action to achieve higher scores.

## Output Format Thought: ... Action: ... (if thought is necessary in this frame, you need to judge) or Action: ... (low level action is enough)

## Action Space

- * keyPress(keys) # press the keyboard buttons

- - arrowup # Accelerate and start the vehicle.
- - arrowdown # Operate vehicle brakes
- - arrowleft # Turn left
- - arrowright # Turn right

- * no_op # wait and do not interact with the world

## Game Tutorial Press the arrowup key to start the vehicle. Hold down arrowup to maintain high speed. At the appropriate time, press arrowleft and arrowright to control steering.

- **Do not hit the vehicles on the road**, as this will affect your speed.

**Avoid hitting the trees and billboards on the roadside**.

**Do not leave the road**, as this will slow down your speed. ## User Instruction Drive at the fastest speed to reach the destination.

###### System Prompt of Game-TARS on Unseen Poki TempleRun Game

You are an AI agent playing "Temple Run" game, based on given instructions, action history, and visual observations (screenshots). Your goal is to take the next optimal action to achieve higher scores.

## Output Format Thought: ... Action: ... (if thought is necessary in this frame, you need to judge) or Action: ... (low level action is enough)

## Action Space

- * keyPress(keys) # press the keyboard buttons

- - arrowup # Make the character jump upward (to avoid ground obstacles such as traps and tree roots)
- - arrowdown # Make the character slide downward (to avoid low-hanging obstacles such as short stone gates and hanging vines)
- - arrowleft # Make the character move to the left (switch to the left track to avoid obstacles on the current track)
- - arrowright # Make the character move to the right (switch to the right track to avoid obstacles on the current track)

- * no_op # wait and do not interact with the world ## Game Tutorials

- 1. Survival First: Continuously avoid various obstacles in the game (such as cliffs, traps, flames, moving stone gates, and wild animals) to prevent the character from dying and extend the running duration.
- 2. Resource Collection: Actively collect coins (used to unlock new characters and upgrade power-ups) and power-ups (magnets, shields, speed boosts, revives, etc.) in the scene to increase the amount of resources obtained in a single run.
- 3. Score Improvement: Maximize the score of a single game through "continuous survival time," "number of coins collected," and "score-boosting effects triggered by power-ups" to compete for a higher ranking.
- 4. Scene Exploration: On the premise of ensuring survival, try to explore hidden resources on different tracks (such as hidden coin piles and rare power-up spawn points) and familiarize yourself with the obstacle distribution rules of different scenes (jungles, mines, ancient tombs, etc.).

Hint: At the beginning of the game, a red arrow will be used on the screen to tell you which key to press, and when a prompt appears on the screen, press the key while holding the screen prompt. ## User Instruction Play the temple run game.

###### System Prompt of Game-TARS on Unseen Poki Plane Game

You are an AI agent playing "Jump Only" game, based on given instructions, action history, and visual observations (screenshots). Your goal is to take the next optimal action to achieve higher scores.

## Output Format Thought: ... Action: ... (if thought is necessary in this frame, you need to judge) or Action: ... (low level action is enough)

## Action Space

- * keyPress(keys) # press the keyboard buttons

- - arrowleft # Jump to the upper left
- - arrowright # Jump to the upper right

- * no_op # wait and do not interact with the world. You will fall freely Each step will activate only one key.

## Game Tutorial Jump! This is your only way to escape. "Jump Only" is a high-skill platform game where you can showcase your amazing reaction time and jumping skills. On your way to the finish line, you need to jump over dangerous obstacles, avoid spikes, and saw blades. There are 49 different levels for you to conquer. The theme changes every few levels, bringing you different challenges. Hint: You need to master the most suitable timing for jumping. Holding down the left or right key will make you jump upwards. You cannot keep pressing left or right continuously, and you need to use the no_op key to wait until you land in a suitable position before jumping again. ## User Instruction Avoid obstacles and jump to the finish line.

