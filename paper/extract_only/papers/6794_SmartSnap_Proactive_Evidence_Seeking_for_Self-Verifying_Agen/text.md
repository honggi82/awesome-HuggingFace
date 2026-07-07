# arXiv:2512.22322v2[cs.CL]6Jan2026

## SmartSnap: Proactive Evidence Seeking for Self-Verifying Agents

###### Youtu-Agent Team∗

Agentic reinforcement learning (RL) holds great promise for the development of autonomous agents under complex GUI tasks, but its scalability remains severely hampered by the verification of task completion. Existing task verification is treated as a passive, post-hoc process: a verifier (i.e., rule-based scoring script, reward or critic model, and LLM-as-a-Judge) analyzes the agent’s entire interaction trajectory to determine if the agent succeeds. Such processing of verbose context that contains irrelevant, noisy history poses challenges to the verification protocols and therefore leads to prohibitive cost and low reliability. To overcome this bottleneck, we propose SmartSnap, a paradigm shift from this passive, post-hoc verification to proactive, in-situ self-verification by the agent itself. We introduce the Self-Verifying Agent, a new type of agent designed with dual missions: to not only complete a task but also to prove its accomplishment with curated snapshot evidences. Guided by our proposed 3C Principles (Completeness, Conciseness, and Creativity), the agent leverages its accessibility to the online environment to perform self-verification on a minimal, decisive set of snapshots. Such evidences are provided as the sole materials for a general LLM-as-a-Judge verifier to determine their validity and relevance. Experiments on mobile tasks across model families and scales demonstrate that our SmartSnap paradigm allows training LLM-driven agents in a scalable manner, bringing performance gains up to 26.08% and 16.66% respectively to 8B and 30B models. The synergizing between solution finding and evidence seeking facilitates the cultivation of efficient, self-verifying agents with competitive performance against DeepSeek V3.1 and Qwen3-235B-A22B.

Date: December 26, 2025 Correspondence: caishaofei@stu.pku.edu.cn; yuleiqin@tencent.com Code: https://github.com/TencentYoutuResearch/SmartSnap Data: https://huggingface.co/collections/yolay/smartsnap

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

### 1 Introduction

[Figure 6]

Figure 1. Success rate on AndroidLab [1] across model families and scales. Compared with the vanilla prompting (PT), our SmartSnap brings significant gains via fine-tuning (FT) and reinforcement learning (RL) without relying on sophisticated rule-based verifiers and task-specific reward models. The developed self-verifying agents learn to complete tasks and curate snapshot evidences in a complementary manner, achieving competitive performance with larger LLMs.

*Full author list in contributions.

Task Instruction

3C Principles & Task Instruction

Task Instruction

… …

…

Agent Env

##### Agent Env

Agent Env

𝑠 𝑎 𝑠 𝑎 … 𝑠 𝑎 𝑠 𝑎 𝑠 𝑎 … 𝑠 𝑎 𝑒 𝑒

𝑠 𝑎 𝑠 𝑎 … 𝑠 𝑎

###### Submit Complete Trajectory

Submit Self-Verified Evidence

###### Record Inner Env State

[Figure 8]

[Figure 9]

[Figure 10]

##### LLM-as-a-Judge LLM-as-a-Judge

Scoring Script

𝑅 Reward

𝑅

𝑅

Reward Reward

[Figure 11]

Is the evidence related to the task? Is the evidence succinct enough? Does evidence support agent claim?

Is the inner env state strictly matches the ground truth state?

Is it a successful trajectory to complete the given task?

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Full Trajectory Processing & High Hallucination Risk.

Task-Specific Scripting & Internal State Access.

Self-Contained Procedure & Task Generalization.

(a) Rules Verify State/Answer

(b) L/VLMs Verify Trajectory

(c) L/VLMs Verify Evidence

User Prompt System Prompt 𝑠 Env State 𝑎 Tool Call 𝑒 Agent Curated Evidence

- Figure 2. Three strategies for agent verification distinguished by their inputs to the verifier: (a) the task-specific script accessing the ground-truth state; (b) the full trajectory with noisy context; and (c) the agent-curated evidence set.

The recent rapid advancement of Large Language Models (LLMs) [2, 3, 4, 5] has catalyzed the emergence of general-purpose agents capable of understanding human intent and operating complex digital interfaces. The vision is to automate daily tasks on devices like computers [6, 7, 8] and smartphones [9, 1, 10, 11], promising a significant boost in human productivity. The challenge of scaling the training of such agents, a pursuit now often termed Agentic Reinforcement Learning (Agentic RL) [12, 13], critically hinges on a scalable and automated verification mechanism [14]. In traditional, constrained environments such as video games [15, 16], verification is unambiguous; success is determined either by explicit reward signals like a game score, or through rule-based scripts that directly access the environment’s internal state for validation (as depicted in Figure 2a). Conversely, for agents in open-ended digital worlds, the spectrum of tasks is vast, and their success criteria are not amenable to simple manual specification. Although early studies on open-ended tasks attempted to use manually designed, rule-based scripts to verify task completion [17, 18], these approaches remain labor-intensive and unscalable towards diverse real-world tasks.

To overcome this problem, recent research has shifted towards employing Vision-Language Models (VLMs) [19] as automated verifiers [20, 21, 22, 23] (Figure 2b). This paradigm offers immense scalability and generalization, as a single VLM can, in principle, verify a wide range of tasks without per-task engineering. However, this shift exposes a deeper, architectural flaw in the prevailing agent paradigm: the agent’s execution is entirely decoupled from the subsequent verification process, reducing verification to a difficult, passive, and post-hoc analysis of potentially ambiguous behaviors. In this paradigm, the agent itself is verification-agnostic; it performs actions without any consideration of proving its success beforehand. Consequently, the full cognitive burden of verification falls upon the VLM verifier, which is forced to review the entire, often noisy and ambiguous, interaction trajectory in a brute-force manner. Such passive verification design leads directly to two critical drawbacks: (1) it incurs prohibitively high verification costs (in both API fees and latency), and (2) it places a heavy load on the VLMs, increasing the risk of hallucinations and false positive judgments [20].

We propose SmartSnap, a paradigm shift from the flawed paradigm of passive external verification to proactive self-verification for building GUI agents. Specifically, we redesign the core responsibilities of the agent itself and introduce the concept of a Self-Verifying Agent. It fulfills two-fold missions: to Execute and to Verify. Upon completing a task, instead of passively awaiting judgment, the agent leverages its privileged position within the live environment sandbox (that is often destroyed or unavailable by the time of post-hoc

verification). It actively reviews its interaction history and, crucially, can creatively execute additional, evidence-oriented actions to generate more decisive proof. This entire curation process, guided by what we term the 3C Principles of Evidence Curation (Completeness, Conciseness, and Creativity), yields a concise yet decisive set of evidence. We posit that this dual mission creates a synergistic learning loop: the requirement to find and present evidence compels the agent to develop a deeper, more holistic understanding of the task, as the core knowledge and capabilities required for successful execution and effective proof are deeply intertwined. Through this proactive self-verification, the inputs to the VLM verifier are reduced from the full trajectory to a minimal, curated evidence set, fundamentally resolving the trade-off between verification cost and reliability. Beyond delivering a final verdict, we equip the verifier to provide fine-grained feedback on evidence quality, facilitating a reward-shaping process that accelerates the agent’s acquisition of evidence curation skills. As shown in Figure 1, both fine-tuning (FT) and RL under our SmartSnap paradigm lead to significant performance gains on various LLM families and scales.

Our main contributions are threefold:

- • We propose the SmartSnap, a new paradigm that shifts the burden of verification from the external handcrafted verifier to the Self-Verifying Agent. This fundamentally enhances the efficiency, scalability, and robustness of the verification process by reducing the cognitive load of verifier under GUI tasks.
- • We formalize the 3C Principles of evidence curation (Completeness, Conciseness, and Creativity), providing a theoretical foundation for training self-verifying agents.
- • We design an efficient learning framework where an LLM/VLM verifier provides structured, dense feedback. This is achieved through intrinsic Reward Shaping, not only to judge task completion but also to explicitly guide the agent in improving the quality of its evidence curation.

### 2 Related Works and Preliminaries

#### 2.1 Agents based on Large Language Models

Agents based on LLMs [3, 5, 4] are entities that leverage the world knowledge acquired from web-scale pre-training and RL post-training to perform instruction understanding, perception, task planning, and decision-making. A prominent line of research focuses on enabling these agents to operate general-purpose electronic devices, such as computers [7, 8] and smartphones [9, 1, 10], to fulfill users’ daily instructions or to play games [24, 25, 26, 27]. Existing approaches can be broadly categorized into two paradigms based on their training methodology: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL). SFTbased methods [28, 29, 7] typically rely on expert demonstration trajectories recorded by human annotators for learning. Distillation [30, 31, 32] from stronger LLMs as teacher also provides effective supervision. Nevertheless, a primary limitation of this approach is its low data collection efficiency and high cost. To improve data utility, UI-TARS-2 [33] employs a rejection sampling strategy, where successful trajectories from online interaction with the environment are automatically filtered and used for iterative fine-tuning. Nevertheless, these methods still struggle to eliminate the dependency on a large corpus of high-quality, successful trajectories. To further reduce human intervention, researchers [9, 18, 22, 23, 34] have turned to the RL paradigm, which enables learning from reward signals. However, the core challenge for RL-based methods lies in designing the reward function. Early works, such as UI-R1 [17] and InfiGUI-R1 [18], relied on handcrafted, rule-based reward functions; however, this approach proved to be labor-intensive, unscalable, and prone to overfitting, thereby lacking generalization. To overcome these limitations, subsequent works like WebRL [22] and AutoGLM [23] proposed training an Outcome Reward Model (ORM) to automatically judge if the final environmental state corresponds to task completion, thereby avoiding per-task rule design. More recently, approaches such as Zero-GUI [20] have leveraged the power of VLMs to review all

screenshots captured during a task for a comprehensive judgment. While this method cleverly mitigates agent hallucination and reduces the false positive rate through mechanisms like voting, the prohibitively high cost of full-trajectory review and the extreme demands on the VLM’s long-range reasoning capabilities remain significant drawbacks.

- 2.2 Group Relative Policy Optimization

Inspired by recent works [35, 3], we adopt Group Relative Policy Optimization (GRPO) as our learning algorithm. GRPO is a policy gradient method that innovatively eliminates the need for a separate critic network. Its key mechanism is to compute the advantage function by normalizing the reward of each trajectory against the statistics (mean and standard deviation) of a group of trajectories sampled from the same policy. This approach significantly reduces the training cost and memory overhead associated with traditional actor-critic methods. For each task instruction q, the training process begins by sampling a group of G trajectories,

{τ1, . . . , τG}, with rewards, {R1, . . . , RG}, from the current policy πθold. The policy parameters θ are then updated by maximizing the following objective function:

J =

1 G

G

∑

i=1

min (ri(θ)Ai,clip (ri(θ),1 − ϵ,1 + ϵ) Ai) ,

ri(θ) =

πθ(τi|q) πθold(τi|q)

, Ai =

Ri − mean({R1, · · · , RG}) std({R1, · · · , RG})

.

(1)

- 3 Method

- 3.1 Overall Pipeline and Problem Formulation

We formalize the process of a self-verifying agent’s task execution and evidence curation within an interactive environment as an augmented Markov Decision Process (MDP). We model the environment as a tuple M = (S, A′, P). A′ is an augmented action space, defined as the union of two distinct sets of actions: A′ = Aexec ∪ Acurate, where Aexec denotes the actions for direct environment interaction (e.g., click(x,

y), type(’text’), screenshot()). A state s ∈ S can represent a snapshot of the environment or a tool response. P(st+1|st, at) is the state transition function, where at ∈ Aexec. Acurate is the set of actions used for submitting proof of task completion. A task is defined by a language sentence I. The agent’s behavior is governed by a unified policy πθ(at|s≤t, a<t, I), which maps the history observations and the instruction I to a probability distribution over the augmented action space A′. By interacting with the environment, the policy πθ generates a trajectory τ = (s0, a0, s1, . . . , sN, aN). The trajectory terminates when the agent selects a special terminal action aN ∈ Acurate, such as submit(’text’,E). The ’text’ denotes the final message T and E refers to the curated evidence set. We can use any state-of-the-art LLMs as the verifier ψ. It acts as a scoring function which takes the instruction I, text message T, the evidence E, and evaluation guidance G as input, returning a reward R = ψ(G, I, T, E).

- 3.2 Grounding Evidence in Tool Interactions

A central challenge in our SmartSnap framework is defining trustworthy evidence. A naive approach is relying on the agent’s self-declaration, where the agent generates a textual summary like, “Based on the instruction, I performed actions X and Y, observed Z, and thus the task is complete.” This is inherently untrustworthy,

as the agent is incentivized to omit failures to hack the verifier. A slightly more robust method is using the final state screenshot. However, this approach is also inherently brittle for two key reasons. First, the final state is not always informative; for instance, an agent might successfully fill a table but then navigate back to a generic home page before concluding, leaving the final screenshot useless. Second, verifying certain tasks requires capturing the difference before and after a critical action (e.g., filling the correct arguments into the app as required), where a single, static screenshot is insufficient to prove. To overcome these limitations, we formally define a single piece of evidence, an exhibit, as an atomic tuple representing a direct cause-and-effect event: (at, st+1) where at is the action (i.e., tool call) executed by the agent, and st+1 is the direct observation (i.e., tool response). This interaction pair is an objective, unalterable fact rather than the agent’s subjective summary. This atomic definition elegantly simplifies the agent’s task: instead of generating descriptive text, the agent learns to output a set of integer indices E. These indices pinpoint the most critical interactions within its full history, and this curated evidence is then programmatically formatted using the standard OpenAI chat template:

StringJoin ({F(ae, se+1)|e ∈ E}) , (2)

where StringJoin(·) denotes the pythonic concatenation operation of string and F represents the OpenAI chat template formatting operation. This interaction tuple links a specific action to its immediate, multimodal outcome, and therefore, the agent’s core learning objective becomes the selection of a minimal yet decisive evidence set E from its entire interaction history.

#### 3.3 Foundational Principles of Evidence Curation

To guide the agent’s evidence curation behavior, we establish the 3C Principles that govern it: Completeness, Conciseness, and Creativity. These principles are formulated as meta-instructions and injected into the agent’s system prompt to constrain its decision-making process.

Principle 3.1 (Completeness and Conciseness) The ideal evidence set should maximize the signal of proof while minimizing the noise of redundancy.

Operationally, the system prompt drives the agent to review its entire history of tool interactions after task execution. The Completeness principle requires the agent to ensure that all pivotal tool interactions are included within the final evidence set E. From a statistical perspective, the goal is to maximize the True Positive (TP) rate of the verifier. This prevents a correctly executed trajectory from being wrongly penalized due to incomplete evidence. Conversely, the Conciseness principle is crucial for minimizing the False Positive (FP) rate. This pursuit stems from a dual consideration of efficiency and robustness. For an LLM/VLM verifier, a shorter, more information-dense submission reduces its cognitive load and the likelihood of hallucination induced by irrelevant details [20]. From another perspective, avoiding excessively long inputs for the verifier mitigates the risk of context rot [36] (i.e., the degradation of the LLM verifier’s performance as the context length increases).

Principle 3.2 (Creativity) If the necessary evidence does not exist in the interaction history, the agent is encouraged to create it by taking additional actions.

This principle elevates the agent from a passive historian of its actions to a proactive investigator. It establishes a crucial distinction between two types of agent behavior: task-oriented actions, which are taken to achieve the task’s goal, and evidence-oriented actions, which may be taken after the goal is met, with the sole intent of generating a better proof. For instance, consider the task of installing a browser extension. The final task-oriented action is clicking the “Add extension” button in a confirmation dialog. The immediate feedback for this action—the dialog disappearing—does not actually confirm that the extension has been successfully loaded into the browser, thus constituting weak evidence. A creative agent, after performing this click, will shift its attention from the web content to the browser’s own user interface, proactively inspecting the “toolbar area” for the new extension’s icon. This inspection forms a perfect evidence-oriented action.

Task: Record an income of 8000 CNY in the books, and mark it as salary.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Tap(180, 335)

Type(“8000”)

###### …

Tap(305, 820)

Core Evidence

…

Step 0 Step 7 Step 8 Step 9

Regular Task Execution Steps Self-Verification Steps

- Figure 3. An example of self-verification for evidence curation. The agent decomposes the task into an actionable checklist where the date, the amount, and the category tag are to be confirmed during stepwise task completion. The proactive step of taking snapshots that list the target transaction provides a definitive evidence for task completion.

Such capability to create evidence is the key to achieving true robustness in agentic RL: 1) It allows the agent to understand the indirect consequences of its actions and to search for an optimal proof within a broader state space that includes the entire sandbox (e.g., a complete application UI), thereby satisfying both the completeness and conciseness criteria. 2) It encourages the agent to focus on the state transition and get acquainted with the environment feedback. 3) It implicitly incentivizes reasoning and reflection of agent for active collection of evidence according to the environment feedback.

#### 3.4 Evidence Verification and Reward Shaping

To provide a dense and informative reward signal that guides the agent in mastering the complex evidence curation process, we designed a principled verification framework. This process is executed by an LLM/VLM Verifier following detailed instructions, and its structured feedback is used to construct a multi-component reward function.

Evidence Validity Check. As the first stage of verification, the verifier determines if the agent’s submitted evidence is relevant to the task instruction. Critically, evidence that unambiguously proves failure (e.g., a screenshot showing Wi-Fi is still on for a “turn off Wi-Fi” task) is also considered valid. Evidence is only deemed invalid if it is entirely irrelevant. We provide a small positive reward Rvalidity (e.g., +0.5) for valid evidence and a penalty otherwise. This auxiliary signal encourages the agent to navigate towards relevant application screens, reinforcing its ability to identify key steps even if the task ultimately fails.

Strict Evidence-to-Task Grounding. Only when the agent claims success does the verifier proceed to this core stage of scrutiny. This stage adheres to a “success only upon unequivocal proof” rule, conducting a rigorous, multi-faceted examination of the evidence. This includes checks for (1) Zero Assumptions: The Verifier must not infer or “fill in the gaps” for any states or actions that are not explicitly and unambiguously present in the evidence. What is not shown in the evidence is assumed not to have happened. (2) Traceable Reasoning: Every factual claim made by the Verifier in its analysis must be directly traceable to a specific piece of evidence (e.g., “In the screenshot from Round 3...”). This mechanism promotes literal description over potentially hallucinatory summarization. This stage results in a outcome reward Rcomplete ∈ {1,0}.

Behavioral and Formatting Regularization. To further refine the agent’s curation behavior, we introduce two additional auxiliary reward components: (1) Format Reward (Rformat). This signal constrains the agent’s output to adhere to the format specified in its system prompt strictly. Any formatting error incurs a fixed negative penalty, which is crucial for stable interaction with downstream modules. (2) Conciseness Penalty (Rconcise). To enforce the Principle of Conciseness, we include a penalty term proportional to the size of the evidence, such as Rconcise = −λ · size(E), where λ is a small coefficient. This incentivizes the agent to find the most succinct proof, given that completeness is met.

Ultimately, these four components form a composite reward function, Rtotal, which provides rich and dense feedback for the agent’s end-to-end learning. The final reward is a sum of these components:

R = Rformat + Rvalidity + Rcomplete + Rconcise (3) We optimize the policy using GRPO [35] under the VeRL [37] implementation.

### 4 Experiment

#### 4.1 AndroidLab

Environments. We perform experiments on the AndroidLab [1] benchmark, a reproducible evaluation platform featuring 138 tasks across nine apps, which run on predefined Android Virtual Devices (AVDs). For our work, we utilize only the defined tasks and their corresponding initial environments. We do not use AndroidLab’s built-in, rule-based evaluation system, which determines task completion by decomposing tasks into sub-goals and matching specific UI tree elements. Avoiding this rule-based reward function is a key advantage of our Self-Verifying paradigm, as we believe designing such functions incurs a prohibitively high manual cost. Furthermore, our observation utilizes the compressed XML tree representation provided by Androidlab, rather than high-resolution screenshots. This choice allows us to focus on the agent’s planning and decision-making capabilities, rather than being limited by insufficient perceptual abilities. Our action space adopts the native action space provided by Androidlab, which includes Tap, Swipe, Type, Long Press, Home, and Back. We additionally add a submit tool to allow the agent to present the evidence snapshot.

Table 1. Statistics on the distribution of tasks from AndroidLab [1].

Number of Tasks (Percentage) per App Category Calendar Zoom Bluecoins PiMusic Maps.me Contacts Cantook Clock Setting Total

Split

Training 92 (12.67%) 0 (0.00%) 81 (11.15%) 58 (7.98%) 86 (11.84%) 55 (7.57%) 84 (11.57%) 43 (5.92%) 227 (31.26%) 726 (100%) Validation 14 (10.14%) 5 (3.62%) 15 (10.86%) 12 (8.69%) 15 (10.86%) 15 (10.86%) 12 (8.69%) 27 (19.56%) 23 (16.66%) 138 (100%)

Datasets. To ensure comparability, we prepare the same 726 training tasks as those released in Android Instruct dataset [1]. We use the same tasks for both cold-start SFT and RL. Specifically for the SFT dataset, it consists of two components: 1) We follow the VeRL agent loop [37] to build a simple agent framework that allows the agent LLM to interact with the Android environment. All the 726 training tasks are fed into the framework for trajectory generation where both task completion and evidence submission are performed. To ensure diversity of the collected trajectories, we adopt both the DeepSeek V3.1 [38] and the Qwen3-235B-A22B [39] as agents for rollout. 2) We update the original response data of the Android Instruct dataset with those from DeepSeek V3.1 and Qwen3-235B-A22B. In total, we obtain around 550K trainable QA pairs from 30K trajectories. We randomly sample 100K QA pairs as the cold-start dataset. Table 1 provides the detailed statistics on the distribution of tasks in AndroidLab [1].

#### 4.2 Implementation Details

Agent. In the present study, we employ various baselines of different architectures, size, and edition as the backbone for Self-Verifying Agent: LLaMA3.1-8B-Instruct [40], Qwen2.5-7B-Instruct [41], Qwen3-8BInstruct [39], and Qwen3-32B-Instruct [39]. It is noted that due to the limit of context length, we use the instruct mode for Qwen3 series rather than the verbose reasoning mode. The selection of backbone models (e.g., size) is based on the requirement that the model must possess instruction following capability for comprehension of the 3C principles defined in the system prompt. Furthermore, unlike the majority of computer use agents [9, 7, 20] that compress history into a fixed-length context, we do not bring any agent scaffold into training and use the entire trajectory for end-to-end optimization. Such design enjoys two benefits: 1) the Self-Verifying agent can review the complete context for task-oriented reflection and evidence preparation; 2) the independence of complicated memory management in existing scaffold frameworks enables broader compatibility and easy accessibility. We design a system prompt of approximately 4k tokens, which specifies the agent’s required output format, as illustrated in Figure 2.

Verifier. For the verifier, we utilize the DeepSeek-R1 model [3]. Since the agent’s evidence is also in the form of compressed XML, DeepSeek-R1, which lacks multimodal understanding but possesses strong reasoning capabilities, is highly suitable. The system prompt designed for the verifier is shown in Figure B. We implement a structured reward system: if the agent provides evidence relevant to task verification, a reward of 0.2 is given. If, based on this, the agent successfully completes the task, an additional 0.8 reward is granted. To mitigate verifier hallucinations, we use DeepSeek-R1 to evaluate the same evidence three times; the task is only considered complete if at least two evaluations pass. Additionally, if the agent’s output does not adhere to the format prescribed by the system prompt, a fixed penalty of −1.0 is applied.

Training. We use the VeRL [37] for both cold-start SFT and the RL with SmartSnap. For SFT experiments, we set the training batch size as 32, number of epochs as 5, learning rate default as 1e − 5 with cosine decay, and maximum context length of 32K. For RL, we set the training batch size as 8, ppo mini batch size as 8, group size as 8, KL coefficient β = 0, constant learning rate default as 1e − 6, maximum number of interaction turns as 30, the total number of training steps as 180 (2 epochs), and the maximum context length of 32K. All experiments are conducted on 64 GPUs.

Evaluation For the validation of task completion under AndroidLab environment, we follow [1] to use the LLM-as-a-Judge (GLM4) [42] for fair comparison. It analyzes all the interaction trajectories (e.g., traces and xml snapshots) for final scoring.

#### 4.3 Main Results

As shown in Table 2, our SmartSnap enables the LLM-driven agents to master interaction with the Android environment during RL the stage, boosting the task success rate (SR) substantially. Specifically, SmartSnap (Qwen3-8B-Instruct) and SmartSnap (Qwen3-32B-Instruct) respectively achieve on-par performance with DeepSeek-V3.1 and Qwen3-235B-A22B, showcasing quite promising results with regard to their model scales. These two models support both instruct (i.e., empty thinking <think>\n\n</think>) and reasoning modes during inference, but we use the default instruct mode due to the limit of multi-turn context in the present study. All the models under investigation achieve performance gains over 16% across model families and sizes, confirming the generalization of SmartSnap. It is noted that for LLaMA3.1-8B-Instruct, its function call mode does not natively support ReAct [43] where the reasoning process and the tool calling are interleaved

- Table 2. Results on AndroidLab with XML mode. PT, FT, and RL stand for prompting, fine-tuning, and reinforcement learning, respectively. SR, Sub-SR, RRR, and ROR stand for Success Rate, Sub-Goal Success Rate, Reversed Redundancy Ratio, and Reasonable Operation Ratio, respectively. For all these metrics, a higher value means better performance. The best and second-best results are marked in bold and underlined, respectively. We do not report RRR score if SR < 5.

Type Model SR Sub-SR RRR ROR PT GPT-4o 25.36 30.56 107.45 86.56 PT GPT-4-1106-Preview 31.16 38.21 66.34 86.24 PT Gemini-1.5-Pro 18.84 22.40 57.72 83.99 PT Gemini-1.00 8.70 10.75 51.80 71.08 PT GLM4-Plus 27.54 32.08 92.35 83.41 PT DeepSeek-V3.1 36.23 40.95 81.01 94.63 PT Qwen3-235B-A22B 34.78 38.76 83.35 89.48 Act-only∗ PT LLaMA3.1-8B-Instruct‡ 2.17 3.62 – 52.77 FT† [1] LLaMA3.1-8B-Instruct‡ 23.91(+21.74%) 30.31 75.58 92.46 PT LLaMA3.1-8B-Instruct 5.07 6.28 52.77 51.82 FT† [1] LLaMA3.1-8B-Instruct 20.28(+15.21%) 26.13 69.44 90.43 FT (ours) LLaMA3.1-8B-Instruct 23.91(+18.84%) 30.36 37.96 83.23 RL (ours) LLaMA3.1-8B-Instruct 31.15(+26.08%) 38.03 81.28 95.80 ReAct PT Qwen2.5-7B-Instruct 12.32 14.98 67.56 78.52 FT† [1] Qwen2.5-7B-Instruct 20.28(+7.96%) 27.05 35.52 62.46 FT (ours) Qwen2.5-7B-Instruct 30.15(+17.83%) 36.59 49.19 73.28 RL (ours) Qwen2.5-7B-Instruct 30.43(+18.11%) 35.20 102.30 96.36 PT Qwen3-8B-Instruct 10.14 12.38 66.21 67.15 FT† [1] Qwen3-8B-Instruct 19.56(+9.41%) 25.60 38.69 65.18 FT (ours) Qwen3-8B-Instruct 26.81(+16.66%) 31.09 72.16 69.85 RL (ours) Qwen3-8B-Instruct 36.23(+26.08%) 41.96 88.04 94.49 PT Qwen3-32B-Instruct 18.12 21.80 91.99 87.57 FT† [1] Qwen3-32B-Instruct 22.46(+4.34%) 28.20 39.28 65.50 FT (ours) Qwen3-32B-Instruct 28.98(+10.86%) 35.92 97.79 97.33 RL (ours) Qwen3-32B-Instruct 34.78(+16.66%) 40.26 89.47 93.67

∗ LLaMA3.1 models only natively support tool calling w/o reasoning. † The Android Instruct dataset [1] is used for fine-tuning where self-

verification is not performed. ‡ The official results are cited here for comparison.

during the rollout progress. Therefore, its act-only paradigm restricts the explicit semantic representation of state analyses (reflection) and sub-goal decomposition (planning) and thereafter leads to lower upper limit. Such reasoning-free agent might mechanically repeat its sub-optimal or even incorrect steps, leading to ineffective acquisition of skill-sets. This is also confirmed by its comparatively lower reversed redundancy ratio (RRR) where redundant steps are more likely to occur.

Furthermore, with respect to the performance comparison of SFT, we find that our Self-Verifying mode achieves much higher task SR than the vanilla Android Instruct dataset [1]. Note that we adopt exactly the same 726 tasks without performing data augmentation for task distribution expansion, ensuring fair comparison. Our SFT allows the agent to not only knows how to perform the task but also, more importantly, understands which kinds of evidences are required and appropriate for curation. Such self-verifying nature is quite indispensable to the generalization of agent as the guidance on the preparation of key evidences implicitly cultivates the problem decomposition capabilities. Each evidence can be viewed as a milestone of each sub-task along the path towards completion and thereafter such a curation process contributes to a superior SR. In contrast, the vanilla fine-tuning only causes parameterized memorization of solutions. It leverages LLMs of various basic capacity (e.g., knowledge QA, reasoning and agentic performance) towards a similar level of 22% SR. This means that the mechanic imitation of expert demonstration (e.g., trajectories from a stronger LLM) without evidence-oriented self-verification cannot bring each model to its extreme.

- Table 3. Results of success rates per category on AndroidLab with XML mode. PT, FT, and RL stand for prompting, fine-tuning, and reinforcement learning, respectively. For all tasks, a higher value means better performance.

Success Rate per App Category Calendar Zoom Bluecoins PiMusic Maps.me Contacts Cantook Clock Setting

Type Model

PT LLaMA3.1-8B-Instruct 0.00 0.00 0.00 0.00 0.00 13.33 28.57 15.38 8.33 FT† [1] LLaMA3.1-8B-Instruct 7.14 40.00 40.00 0.00 7.14 33.33 41.66 15.38 18.18 FT (ours) LLaMA3.1-8B-Instruct 7.14 20.00 26.66 8.33 0.00 60.00 33.33 14.81 39.13 RL (ours) LLaMA3.1-8B-Instruct 7.14 60.00 20.00 8.33 0.00 53.33 50.00 37.03 47.82 PT Qwen2.5-7B-Instruct 0.00 40.00 10.00 10.00 0.00 20.00 16.66 18.51 14.28 FT† [1] Qwen2.5-7B-Instruct 14.28 40.00 13.33 8.33 0.00 46.66 33.33 18.51 21.73 FT (ours) Qwen2.5-7B-Instruct 21.42 40.00 40.00 8.33 0.00 46.66 41.66 33.33 43.47 RL (ours) Qwen2.5-7B-Instruct 14.28 20.00 33.33 8.33 0.00 60.00 50.00 37.03 34.78 PT Qwen3-8B-Instruct 7.14 66.66 0.00 8.33 0.00 15.38 11.11 4.00 15.00 FT† [1] Qwen3-8B-Instruct 7.14 40.00 6.66 8.33 0.00 33.33 50.00 18.51 26.08 FT (ours) Qwen3-8B-Instruct 21.42 60.00 53.33 8.33 0.00 16.66 45.45 40.74 41.66 RL (ours) Qwen3-8B-Instruct 14.28 60.00 33.33 33.33 13.33 53.33 50.00 40.74 39.13 PT Qwen3-32B-Instruct 0.00 80.00 13.33 0.00 0.00 33.33 33.33 22.22 17.39 FT† [1] Qwen3-32B-Instruct 14.28 60.00 26.66 0.00 0.00 40.00 50.00 25.92 13.04 FT (ours) Qwen3-32B-Instruct 7.14 40.00 13.33 8.33 0.00 53.33 50.00 40.74 39.13 RL (ours) Qwen3-32B-Instruct 7.14 80.00 40.00 25.00 0.00 60.00 50.00 38.46 39.13

† The Android Instruct dataset [1] is used for fine-tuning where self-verification is not performed.

Table 3 provides the detailed success rates per category. Compared with the naive prompting and the vanilla instruction fine-tuning, our SFT and RL achieve performance gains consistently on most of the app categories except Maps.me. Specifically, we find that the relative performance gains are closely associated with the distributions of training tasks. As shown in Table 1, the tasks from App Setting are dominating and thereafter steady improvement is observed on this domain for both SFT and RL stages. This highlights that for further studies on mobile agents, the preparation of training data should emphasize: 1) the balance between tasks of different apps, and 2) the diversity of tasks within each app. Moreover, we observe that the difficulty of tasks also plays a critical role in RL effectiveness. The almost zero variation of performance (both SFT and RL) on App Maps.me, regardless of model families and scales, implies that there exists a huge knowledge gap between the existing pretrained LLMs and the mobile domains. Without intensive knowledge injection, the agent itself cannot explore effectively to handle tasks such as finding the shortest path via a specific transportation or locating the destination that meets all target requirements.

#### 4.4 Qualitative Analysis

We present a comprehensive analysis and highlight several key findings from our RL dynamics.

###### Takeaways

- • The agents manage to submit only relevant and persuasive evidences with the number of evidences converging towards 1.5 on average.
- • The agents learn to complete tasks and submit evidences in a more compact and efficient interaction manner without mindless trials and errors.
- • The RL itself increases the training reward consistently with a decreasing intra-group variance, showcasing that the agents gradually overfit the few available training tasks.
- • The fluctuation of performance on complex domains like Calendar, Maps.me, and Zoom suggests that agents struggle at deriving robust strategies without knowledge supplement.

[Figure 33]

[Figure 34]

[Figure 35]

(a) Number of Evidence (Training). (b) Number of Evidence (Validation). (c) Number of Agent Turns (Training).

[Figure 36]

[Figure 37]

[Figure 38]

(d) Number of Agent Turns (Validation). (e) Response Length (Training). (f) Response Length (Validation).

Figure 4. Agent behavior evolution (RL dynamics).

Behavior Evolution To gain deeper insights into the RL process of SmartSnap, we illustrate the evolution of agent behaviors in Figure 4.

- • Number of Evidence: The number of evidences is penalized to be smaller than 3 in our intrinsic reward design. It not only prevents potential reward hacking where the overwhelming evidences might cheat the LLM-as-a-Judge in return for higher rewards, but also encourages the agent to create or select only truly relevant evidences for verification. For training, the Act-only LLaMA3.1-8B exhibits a downward-then-upward trend while the ReAct Qwen series demonstrate more complex patterns. We find that the LLaMA3.1-8B tends to submit a few last screenshots as evidences at the early training stages, and then proposes to submit only the last one as training progresses. Its relatively limited

[Figure 40]

[Figure 41]

(a) Training Reward (mean). (b) Training Reward (std.).

[Figure 42]

[Figure 43]

[Figure 44]

(c) Validation on Bluecoins. (d) Validation on Calendar. (e) Validation on Cantook.

[Figure 45]

[Figure 46]

[Figure 47]

(f) Validation on Clock. (g) Validation on Contacts. (h) Validation on Maps.me.

[Figure 48]

[Figure 49]

[Figure 50]

(i) Validation on Pi Music. (j) Validation on Settings. (k) Validation on Zoom.

Figure 5. Agent performance variation (RL dynamics).

evidence creativity suggests that a brief thinking before acting is beneficial to the maintenance of evidences.

- • Number of Agent Turns: All models across families and scales undergo a decreasing trend of interaction turns for both training and validation. Given the monotonous increase of training reward, we believe SmartSnap encourages agents to complete tasks in a more efficient manner.
- • Response of Length: In line with the number of agent turns, the response length decreases accordingly. During the RL stage, agents tend to exploit the existing successful patterns and becomes more proficient when they encounter the same tasks again during the second training epoch.

Performance Variation To investigate the performance variation across training steps, we provide the both the training reward and the validation accuracy in Figure 5. Note that the accuracy score reported here is delivered by the LLM-as-a-Judge, which reflects the performance of both task solving and evidence curation capabilities.

- • Training Reward: It is observed that the average training reward consistently improves, demonstrating that the agent indeed learns to complete the tasks and curate valid snapshot evidences. The decreasing intra-group standard deviation also suggests that during RL, the agent gradually overfits the few 726 tasks with a group of trajectories either full success or failure. To maintain effective RL signals, we believe the construction of a larger, difficulty-balanced training set is indispensable.
- • Validation Accuracy: The validation accuracy increases on most app domains except Calendar, Maps.me, and Zoom. On these three apps, we find that agents of all model families and scales are struggling with a fluctuating performance close to zero. One reason is that the testing samples from these categories are too scarce to provide smooth estimation of performance. Another underlying reason is that agents fail to develop robust tactics to handle these tasks, revealing a potential knowledge gap of LLMs under mobile scenarios. One would have to perform continual pre-training (CPT) to inject domain-specific knowledge, allowing shallow-yet-braod coverage of operation skills [44, 45].

Case Study We randomly choose two examples respectively from the apps Bluecoins and Settings. One of the typical failure pattern is that the agent misunderstands the clickable items in the app interfaces and repeats its wrong actions consistently. In Figure 6, the agent assumes there exists an entrance button that one must tap first for interaction. It consistently taps the ok/cancel button to reveal and hide the sidebar without proceeding to performing the task directly. After exploration during RL rollout, the agent suddenly notices that there exists a floating button (+) in the bottom right area. It performs a tentative tap and jumps to the detailed page for noting down expenses, which leads to outcome success and incentivizes such policy through RL. To self-verify the task completion, the agent not only chooses the last screenshot but also those intermediate ones that specify the correct in-filling of parameters such as amount, date, and expense type. Such behavior encourages the agent to reflect whether the preceding steps truly meet the requirement of the task. Another case in Figure 7 also shows that initially the agent fails to escape from the control panel of Network & Internet. It hallucinates that the current page corresponds to Display setting and thereafter triggers off consistent tapping for theme adjustment. After RL, the agent masters the trick of search for shortcut access to the dark theme switch. In this case, the agent finds a more convenient solution with only few interaction turns. Thereafter, only the last snapshot that reveals the turned-on switch is adopted as the evidence for verifying that the dark theme is ready. Such efficient interaction and evidence curation process is in line with the decreasing number of agent turns during RL training.

[Figure 53]

- Figure 6. The agent trajectories before and after SmartSnap RL of one testing example of the app Bluecoins (Qwen3-8B). Initially, the agent fails to comprehend the instruction properly and consistently taps the top-left ok/cancel button to find the entrance towards creating a new expense note. After SmartSnap RL, the agent learns to utilize the floating button for creating new notes with correct in-filling of amount and date. In addition, it grasps the evidence-based self-verification during stepwise reasoning, where snapshots that respectively prove the completion of filling amount, date, and expense type are submitted for verification.

[Figure 55]

- Figure 7. The agent trajectories before and after SmartSnap RL of one testing example of the app Settings (Qwen3-8B). Initially, the agent fails to act according to its thought process, entering the Network & Internet by mistake rather than the Display. It fails to jump out of the loop and keeps repeating the erroneous actions. After SmartSnap RL, the agent learns to utilize the search tool in the app and quickly pinpoints the correct page for the theme switch. In addition, it grasps the evidence-based self-verification during stepwise reasoning, where the exact snapshot with the dark theme switch turned on is submitted for verification.

### 5 Conslusion

In this paper, we addressed a critical bottleneck in scaling agentic reinforcement learning: the inefficiency and unreliability of passive, post-hoc verification of task completion. By introducing the Self-Verifying Agent, we propose SmartSnap, a fundamental paradigm shift that relocates the burden of proof from a distant verifier to the agent itself. Guided by the 3C Principles (Completeness, Conciseness, and Creativity), our framework empowers agents to not only execute complex GUI tasks but also to proactively curate a minimal, decisive evidence set that simplifies the verification process. Experimental results on the mobile benchmark AndroidLab demonstrate that our proactive approach indeed reduces the costly verification associated with LLMs-based verifier while enhancing the overall success rate. The synergistic learning between task completion and evidence curation reinforces a mechanism of developing robust and self-aware autonomous agents. We believe that self-verification is a vital step toward creating scalable and reliable agentic systems in open-ended, real-world environments.

###### Limitations and Future Directions There exist two major limitations.

- • In the present study, we do not perform comprehensive CPT and SFT for preparation of a RL-ready starting point. The ineffectiveness of RL on certain task domains suggest that the agent is in lack of domain-specific knowledge that must be parameterized via additional training. In this case, we believe the collection of corpus for downstreaming agentic tasks (e.g., mobile operations) is imperative in the future. It lays a foundation for cultivating skills of both task solution and evidence creation.
- • The current experiments are limited to one benchmark due to the resource-intensive nature of RL with sandbox environments. Unlike maths and coding tasks, the complex environments such as web explorer, android system, and linux OS require scheduling of multiple sandbox services with high CPU and memory specifications. RL with such environments poses a severe challenge to engineering for high concurrency and low latency. We aim to extend the Self-Verifying paradigm to more heterogeneous environments with diverse benchmarks (e.g., WebShop [46] and WebArena [47]).

### Contributions

Authors Shaofei Cai1,2† Yulei Qin1 Haojia Lin1 Zihan Xu1 Gang Li1 Yuchen Shi1 Zongyi Li1 Yong Mao1 Siqi Cai1 Xiaoyu Tan1 Yitao Liang2 Ke Li1 Xing Sun1

Affiliations 1Tencent Youtu Lab 2Institute for Artificial Intelligence, Peking University

† Work during Internship at Tencent.

Equal Contribution Shaofei Cai Yulei Qin

### References

- [1] Yifan Xu, Xiao Liu, Xueqiao Sun, Siyi Cheng, Hao Yu, Hanyu Lai, Shudan Zhang, Dan Zhang, Jie Tang, and Yuxiao Dong. Androidlab: Training and systematic benchmarking of android autonomous agents. arXiv preprint arXiv:2410.24024, 2024.
- [2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [3] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.
- [4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [5] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [6] Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, et al. Ufo2: The desktop agentos. arXiv preprint arXiv:2504.14603, 2025.
- [7] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.
- [8] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37: 52040–52094, 2024.
- [9] Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Foundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025.
- [10] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.

- [11] Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, and Yuxiao Dong. Mobilerl: Online agentic reinforcement learning for mobile gui agents,

2025. URL https://arxiv.org/abs/2509.18119.

- [12] Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, et al. The landscape of agentic reinforcement learning for llms: A survey. arXiv preprint arXiv:2509.02547, 2025.
- [13] Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou, Zehan Qi, Hanyu Lai, Yifan Xu, Rui Lu, Hongning Wang, Jie Tang, and Yuxiao Dong. Agentrl: Scaling agentic reinforcement learning with a multi-turn, multi-task framework, 2025. URL https://arxiv.org/abs/ 2510.04206.
- [14] Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning, 2025. URL https://arxiv.org/abs/2511.22570.
- [15] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.
- [16] Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng Wu, Qingwei Guo, et al. Mastering complex control in moba games with deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 6672–6679, 2020.
- [17] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. CoRR, 2025.
- [18] Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, and Fei Wu. Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint arXiv:2504.14239, 2025.
- [19] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [20] Chenyu Yang, Su Shiqian, Shi Liu, Xuan Dong, Yue Yu, Weijie Su, Xuehui Wang, Zhaoyang Liu, Jinguo Zhu, Hao Li, Wenhai Wang, Yu Qiao, Xizhou Zhu, and Jifeng Dai. Zerogui: Automating online gui learning at zero human cost. arXiv preprint arXiv:2505.23762, 2025.
- [21] Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, et al. Opencua: Open foundations for computer-use agents. arXiv preprint arXiv:2508.09123, 2025.
- [22] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. arXiv preprint arXiv:2411.02337, 2024.
- [23] Xiao Liu, Bo Qin, Dongzhu Liang, Guang Dong, Hanyu Lai, Hanchen Zhang, Hanlin Zhao, Iat Long Iong, Jiadai Sun, Jiaqi Wang, et al. Autoglm: Autonomous foundation agents for guis. arXiv preprint arXiv:2411.00820, 2024.
- [24] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Shawn Ma, and Yitao Liang. Describe, explain, plan and select: interactive planning with llms enables open-world multi-task agents. Advances in Neural Information Processing Systems, 36:34153–34189, 2023.
- [25] Shaofei Cai, Bowei Zhang, Zihao Wang, Haowei Lin, Xiaojian Ma, Anji Liu, and Yitao Liang. Groot-2: Weakly supervised multi-modal instruction following agents. arXiv preprint arXiv:2412.10410, 2024.

- [26] Shaofei Cai, Zihao Wang, Kewei Lian, Zhancun Mu, Xiaojian Ma, Anji Liu, and Yitao Liang. Rocket1: Mastering open-world interaction with visual-temporal context prompting. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12122–12131, 2025.
- [27] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.
- [28] Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. arXiv preprint arXiv:2406.08451, 2024.
- [29] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317, 2024.
- [30] Xiao Cui, Yulei Qin, Yuting Gao, Enwei Zhang, Zihan Xu, Tong Wu, Ke Li, Xing Sun, Wengang Zhou, and Houqiang Li. Sinkhorn distance minimization for knowledge distillation. arXiv preprint arXiv:2402.17110, 2024.
- [31] Xiao Cui, Mo Zhu, Yulei Qin, Liang Xie, Wengang Zhou, and Houqiang Li. Multi-level optimal transport for universal cross-tokenizer knowledge distillation on language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 23724–23732, 2025.
- [32] Xiao Cui, Yulei Qin, Wengang Zhou, Hongsheng Li, and Houqiang Li. Optical: Leveraging optimal transport for contribution allocation in dataset distillation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15245–15254, 2025.
- [33] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025.
- [34] Yulei Qin, Xiaoyu Tan, Zhengbao He, Gang Li, Haojia Lin, Zongyi Li, Zihan Xu, Yuchen Shi, Siqi Cai, Renting Rui, Shaofei Cai, Yuzheng Cai, Xuan Zhang, Sheng Ye, Ke Li, and Xing Sun. Learn the ropes, then trust the wins: Self-imitation with progressive exploration for agentic reinforcement learning, 2025. URL https://arxiv.org/abs/2509.22601.
- [35] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [36] Kelly Hong, Anton Troynikov, and Jeff Huber. Context rot: How increasing input tokens impacts llm performance. Technical report, Chroma, July 2025. URL https://research.trychroma.com/ context-rot.
- [37] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [38] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [39] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [40] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [41] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- [42] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024.
- [43] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv.org/abs/ 2210.03629.
- [44] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025. URL https://arxiv.org/abs/2504.13837.
- [45] Charlie Zhang, Graham Neubig, and Xiang Yue. On the interplay of pre-training, mid-training, and rl on reasoning language models, 2025. URL https://arxiv.org/abs/2512.07783.
- [46] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable realworld web interaction with grounded language agents, 2023. URL https://arxiv.org/abs/2207. 01206.
- [47] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents, 2024. URL https://arxiv.org/abs/2307.13854.

### A Prompts

The detailed prompts used in the present study are respectively provided in Figures 8, 9, 10, and 11. Specifically, the 3C principles are detailed in Figure 8, which explicitly state the paradigm of task completion criteria and self-verification procedures. We use the simple task instruction formatted as Figure 9.

The tool schemas are provided in Figures 10 and 11, containing tool definition of get_current_xml, tap, type, long_press, swipe, back, home, wait, enter, and submit. All the implementations of these tools are based on the adb_command Android Controller [1].

3C Principles

# Agent Instructions

- ## 1. Core Identity & Mission You are a professional Android operation agent assistant. Your primary mission is to **successfully** complete the user’s instructions, then call the ‘submit‘ tool to prove you have finished the task. The evidence you provide to this tool will signal your success.

- * **Solvability Principle:** Assume every user request has a valid solution within the app. Your task is not to determine *if* it can be done, but **how** it is done. **Never give up or state that a task is impossible.** * **Unwavering Persistence:** You must exhaust all possible options. If you get stuck, re-evaluate the screen, consider alternative interpretations of the user’s request, and explore different UI elements. Instead of reporting failure, detail your attempts and pivot to a new strategy.
- * **Focus:** Concentrate on the task at hand.
- * **Language:** Do not translate proper nouns into English.

- ## 2. Operational Protocol

- * **Initial Observation:** **Before you think or act, you must call ‘get_current_xml‘ to get the first observation of the screen.**
- * **Interaction Rules:**
- * Operate only on elements you can see in the current view.
- * On dropdowns, try typing the option directly first.
- * When typing text, do not include surrounding quotes (‘""‘).
- * Please pay special attention to whether the correct date and time are selected on the current interface when using date-related software.
- * If an irrelevant notification dialog box appears, please close it immediately to prevent it from interfering with the task.
- * **Exploration Strategy:**
- * When the path is unclear, explore the app by navigating and tapping elements. Once the path is clear, proceed directly.
- * If you see a scrollable area and have not found the required content, you must use ‘swipe‘ to explore.
- * **If you are stuck, systematically explore all tappable elements, scroll in every possible direction (up, down, left, right), and reconsider the task’s objective from a different perspective.**
- * **Format:** Before each action, you must output a thought process. This helps to clarify your planning before taking actions.

- ## 3. Finalization: Using the ‘submit‘ Tool

- * The ‘submit‘ tool call is the **absolute final output** of your task. It must be called **exactly once**.
- * Before you call ‘submit‘, your final thought process **must** include a recap to prepare the evidence. An evidence is defined as a **Round ID**.
- * You are encouraged to make a few additional tool calls to gather the necessary evidence before calling ‘submit‘, which we call
- **Creative Verification**.

* Your ‘evidences‘ parameter must contain the 1-3 decisive IDs selected during your recap, based on the **"Creative Verification" principle** (showing the final, visible result).

Figure 8. The 3C principles that specify the behavior of self-verification agents.

For the scoring prompt via LLM-as-a-Judge, we provide the detailed verifier prompt in Figure 12 and 13. It specifies not only the judgment criteria for evidence evaluation, but also the necessary app-related knowledge for agent task comprehension.

###### Task Instruction

# Task Instruction: {task_content}

Figure 9. The user prompt for task instruction.

###### Agent Tool Schema (Part 1)

{"type": "function", "function": "name": "get_current_xml", "description": "This function is used to get the current XML representation of the smartphone screen without performing any action. It returns the latest XML of the current screen state.", "parameters": "type": "object", "properties": {}, "required": []} {"type": "function", "function": "name": "tap", "description": "This function is used to tap a UI element shown on the smartphone screen by simulating a tap action within the specified rectangular area defined by the coordinates (x1, y1) and (x2, y2). A simple use case is tap(462,1693,619,1870), which taps the center of the UI element, calculated to be at [540.5,1781.5]. Return a string that contains the latest XML of the current screen.", "parameters": "type": "object", "properties": "x1": "type": "integer", "description": "The x-coordinate of the top-left corner of the rectangle.", "y1": "type": "integer", "description": "The y-coordinate of the top-left corner of the rectangle.", "x2": "type": "integer", "description": "The x-coordinate of the bottom-right corner of the rectangle.", "y2": "type": "integer", "description": "The y-coordinate of the bottom-right corner of the rectangle.", "required": ["x1", "y1", "x2", "y2"]} {"type": "function", "function": {"name": "type", "description": "This function is used to insert text input in an input field/box. text_input is the string you want to insert and must be wrapped with double quotation marks. A simple use case can be type(Hello,¨ world!¨), which inserts the string Hello,¨ world!ïnto the input area on the smartphone screen. This function is only callable when you see a keyboard showing in the lower half of the screen. Return a string that contains the latest XML of the current screen.", "parameters": {"type": "object", "properties": {"text_input": {"type": "string", "description": "The text string to input using the keyboard."}}, "required": ["text_input"]}}} {"type": "function", "function": {"name": "long_press", "description": "This function is used to long press a UI element shown on the smartphone screen. The element is identified by the rectangular area defined by the coordinates (x1, y1) and (x2, y2). The function calculates the center of this area and performs a long press action at that point. A simple use case can be long_press(462,1693,619,1870), which long presses the UI element labeled on [540.5,1781.5]. Return a string that contains the latest XML of the current screen.", "parameters": {"type": "object", "properties": {"x1": {"type": "integer", "description": "The x-coordinate of the top-left corner of the rectangle."}, "y1": {"type": "integer", "description": "The y-coordinate of the top-left corner of the rectangle."}, "x2": {"type": "integer", "description": "The x-coordinate of the bottom-right corner of the rectangle."}, "y2": {"type": "integer", "description": "The y-coordinate of the bottom-right corner of the rectangle."}}, "required": ["x1", "y1",

- "x2", "y2"]}}} {"type": "function", "function": {"name": "swipe", "description": "This function simulates a swipe gesture on a smartphone screen, which can be applied to UI elements like scroll views or slide bars. The swipe starts from the center of a rectangular area defined by (x1, y1) and (x2, y2), then moves in a specified direction for a certain distance. Return a string that contains the latest XML of the current screen.", "parameters": {"type": "object", "properties": {"x1": {"type": "integer", "description": "The x-coordinate of the top-left corner of the rectangle."}, "y1": {"type": "integer", "description": "The y-coordinate of the top-left corner of the rectangle."}, "x2": {"type": "integer", "description": "The x-coordinate of the bottom-right corner of the rectangle."},
- "y2": {"type": "integer", "description": "The y-coordinate of the bottom-right corner of the rectangle."}, "direction": {"type":

"string", "description": "The direction of the swipe (üp¨, down¨ ¨, ¨left¨, right¨ ¨)."}, "dist": {"type": "string", "description": "The distance of the swipe, with options ¨long¨, medium¨ ¨, short¨ ¨. Defaults to medium¨ ¨."}}, "required": ["x1", "y1", "x2", "y2", "direction", "dist"]}}}

Figure 10. The available tools (part 1) that are used to interact with the Android system.

###### Agent Tool Schema (Part 2)

{"type": "function", "function": {"name": "back", "description": "Simulates a back button press. This method navigates the user back to the previous screen or state in the application or operating system. Return a string that contains the latest XML of the current screen.", "parameters": {"type": "object", "properties": {}, "required": []}}} {"type": "function", "function": {"name": "home", "description": "Simulates pressing the home button. This method takes the user to the home screen of the device, minimizing the current application or context. It’s akin to exiting the current state and returning to the main dashboard or operating system’s primary interface. Return a string that contains the latest XML of the current screen.", "parameters": {"type": "object", "properties": {}, "required": []}}} {"type": "function", "function": {"name": "wait", "description": "This function is used to wait for a specified amount of time (in seconds). It can be useful when waiting for UI elements to load or animations to complete.", "parameters": {"type": "object", "properties": {"seconds": {"type": "number", "description": "The number of seconds to wait."}}, "required": ["seconds"]}}} {"type": "function", "function": {"name": "enter", "description": "This function is used to press the Enter key on the smartphone. It simulates pressing the Enter key and returns the latest XML of the current screen after the action completes.", "parameters": {"type": "object", "properties": {}, "required": []}}} {"type": "function", "function": {"name": "launch", "description": "Launches a specified application on the device.The app parameter should be the name of the application to launch. Return a string that contains the latest XML of the current screen.", "parameters": {"type": "object", "properties": {"app": {"type": "string", "description": "The name of the application to launch (e.g., Chrome¨ ¨, Calculator¨ ¨, Settings¨ ¨)."}}, "required": ["app"]}}}

{"type": "function", "function": {"name": "submit", "description": "Submit the evidences when completes the task.", "parameters": {"type": "object", "properties": {"message": {"type": "string", "description": "A message to print before exiting."}, "evidences": {"type": "array", "description": "A list of integers representing the IDs of the decisive evidence steps that led to the successful completion of the task. If the task was not completed successfully, this list can be empty. An individual piece of evidence is a **Tool Call**. This includes its unique ID, its input parameters, and its output result. Each tool call is assigned a unique number ‘x‘, formatted as ‘[TOOL CALL ID: x]‘"}}, "required": ["message", "evidences"]}}}

###### Figure 11. The available tools (part 2) that are used to interact with the Android system.

## Role and Goal You are a meticulous and strict Verifier for an Android automation agent. Your primary role is to determine if the agent has successfully completed a given task, based on its response and the evidence it provides. ## Inputs You Will Receive

- 1. **Task Description:** The original instruction given to the agent.
- 2. **Agent’s Final Response:** The agent’s textual summary of its performance. This is the primary basis for your first step of judgment.
- 3. **Agent’s Evidence:** A set of key evidence, which can be screenshots or XML data, collected by the agent. This is the sole basis for your second step of judgment. ## Core Decision-Making Process Your judgment must follow a strict, hierarchical process: ### Preliminary Analysis: Evaluate Evidence Validity Before determining task success or failure, you must first evaluate the validity of the final evidence submitted by the agent.

- * **What is Valid Evidence (‘<ValidEvidence>True</ValidEvidence>‘)?**
- * The evidence shows a UI screen or XML state that is directly relevant to the task.
- * **Key Point:** Evidence that allows you to definitively conclude ‘FAILURE‘ is also **valid evidence**. For example, if the task is to "turn off Wi-Fi," and the evidence is a screenshot of the Settings app with the Wi-Fi toggle still ’On’, this is *valid* evidence because it allows you to make a confident judgment (in this case, failure).
- * **What is Invalid Evidence (‘<ValidEvidence>False</ValidEvidence>‘)?**
- * The evidence is completely irrelevant to the task, preventing any meaningful judgment. For example, the task is to add a contact in the Contacts app, but the final evidence is a screenshot of the Weather app.
- * **Crucial Rule: If ‘<ValidEvidence>‘ is ‘False‘, then the ‘<Verdict>‘ must be ‘FAILURE‘.**

- ### **Step 1: Evaluate the Agent’s Response** This is your first and most critical task. Start by examining the **Agent’s Final Response**.

- * If the agent’s response admits **failure**, claims the **task is impossible**, states that **information cannot be found**, or makes any form of **excuse or justification for failure (quibbling)**, you must **immediately assign a verdict of FAILURE**.
- * In this scenario, you do **not** need to analyze the evidence. Your reasoning should simply state that the agent failed to accomplish its primary objective.

- ### **Step 2: Verify the Agent’s Claim of Success** You proceed to this step **only if** the agent explicitly claims in its response that it has **successfully completed the task**. In this step, you must conduct a strict verification of the **Agent’s Evidence**:

- * **Strict Verification:** The evidence must unequivocally and fully prove the task’s completion. Any ambiguity or missing step means the task has failed.
- * **Full Task Alignment:** Carefully compare the evidence against every requirement in the Task Description. Partial completion does not count as success. The final state shown in the evidence must perfectly match the desired outcome of the task.
- * **No Assumptions:** Do not infer any actions or states that are not explicitly shown in the evidence. For example, if the task was to "turn off Wi-Fi," the evidence must clearly show the Wi-Fi icon in the "off" state.
- * **Handling Conflicting Evidence:** When analyzing evidence, adhere to the following principles if you encounter conflicting information:
- * **Timestamp Priority:** If multiple screenshots conflict on a key piece of information, you must prioritize the one with the
- **later timestamp**, as it represents the most recent state.
- * **Result Priority:** If an in-process screenshot conflicts with a final result screenshot, you must **always trust the final result screenshot** to determine the task’s outcome.
- * **No Assumptions or Hallucinations:** Your analysis must be **100%** based on the provided evidence. It is **strictly forbidden** to mention any detail, text, or UI element that is not explicitly and clearly visible in the evidence. If the evidence is blurry or incomplete, you should state this and fail the task, rather than guessing the content.
- * **(New) Traceable Reasoning:** This is the most important rule. Inside your ‘<Reasoning>‘ tag, every statement of fact you make must be traceable. Explicitly state which piece of evidence (e.g., "the screenshot from Round 3") supports your observation. **Quote or describe literally, do not summarize or infer.**
- * **Incorrect Example:** "Round 3 shows the event creation UI with the title ’work’." (This is a summary and prone to hallucination).
- * **Correct Example:** "In the screenshot from Round 3, I see an input field labeled ’Title’, and the text inside that field is ’work’." (This is a literal description of observations; it fails if the word ’work’ isn’t there). (Continued below in Figure 13)

Figure 12. The LLM-as-a-Judge prompt for verification of the evidence submitted by the agents (part 1).

(Continued from previous Figure 12) ## Application-Specific Hints To help you make more accurate judgments, here are some common-sense rules and conventions for specific types of applications. Use these hints to better interpret the true meaning of the evidence (screenshots or UI data).

- #### 1. Finance & Expense Apps

- * **Expenses:** Unless specified otherwise, expense amounts should typically be represented as **negative numbers** (e.g., ‘-$50.00‘) or be clearly labeled with words like ’Expense’, ’Payment’, or ’Spent’. If the task was to record an expense, but the evidence shows a positive number without an expense label, the task has failed.
- * **Income:** Income amounts should be **positive numbers** (e.g., ‘+$100.00‘ or ‘100.00‘) or be clearly labeled with words like ’Income’, ’Received’, or ’Deposit’.
- * **Transaction Status:** Pay close attention to statuses like ’Processing’, ’Pending’, ’Completed’, or ’Failed’. If the task was to ’complete a transfer’, the evidence must show a **’Completed’** status, not just ’Processing’.

- #### 2. System Settings & Controls

- * **Toggle States:** For features like Wi-Fi, Bluetooth, or Airplane Mode, pay attention to the **visual state of the toggle switch** (e.g., highlighted/grayed out, slider to the right/left) to determine if it is on or off.
- * **Connection Status:** ’Connected’ is different from ’On but not connected’. If a task is to ’connect to a Wi-Fi network’, the evidence must show a **successful connection to the specific network**, not just that the Wi-Fi is enabled.

- #### 3. E-commerce & Order Apps

- * **Order Status:** If the task is to ’place an order’, the final evidence should show ’Order Successful’, ’Awaiting Payment’, or ’Order Created’, not just the items sitting in the shopping cart.
- * **Shopping Cart:** ’Adding an item to the cart’ and ’placing an order’ are two different tasks. For the former, evidence of the item in the cart is sufficient. For the latter, the evidence must show that the checkout process has been completed.

- #### 4. Date, Calendar & Booking Apps

* **Date Selection Status:** When a task involves selecting a specific date (e.g., setting a reminder, booking a hotel, or checking a schedule), you must carefully inspect the calendar view in the evidence. Verify that the **correct date** is in a **"selected"** visual state (often indicated by a highlight, a circular background, or another distinct marker). Simply viewing the correct month is insufficient; the target date must be explicitly selected.

- #### 5. Text Input & Form Filling

* **Correct Input Location:** If the task requires entering text into a specific field (e.g., ’Title’, ’Note’, ’Username’), you must verify the evidence meticulously. Confirm that the text content is not only correct but is also located in the **correct input field**. For instance, if the task was to enter "Meeting" in the "Title" field, but the evidence shows "Meeting" entered in the "Location" field, the task has failed.

- #### 6. Clock, Alarm & Timer Apps

- * **Activation Status Verification:** When a task involves setting an alarm, timer, or stopwatch, it is not enough to check if the time and label are correct. You must verify that its **final activation state** matches the task’s intent.
- * **Alarms:** If the task is to "set an alarm," the corresponding toggle switch in the evidence must be in the **"On" or "Activated"** state. If the alarm was created but the switch is off, the task has failed.
- * **Timers/Stopwatches:** If the task is to "start a timer" or "run a stopwatch," the final evidence must show that it is **actively running** (e.g., the time is dynamically changing), not just set to an initial value. ## Your Output Format: You must provide your verdict in the following structure: <Reasoning> (Provide a clear, step-by-step analysis. First, state if the evidence is valid, and then use that to determine the final task verdict.) </Reasoning> <ValidEvidence>{True/False}<ValidEvidence> <Verdict>{SUCCESS/FAILURE}<Verdict>
- **NO OTHER TEXT IS ALLOWED INSIDE THE <Verdict> TAG.**

**Do not be misled if the agent’s response implies the task is unanswerable, even if it provides supporting evidence. It is highly probable that the agent simply failed to navigate to the correct user interface. In such cases, you should directly assign a verdict of FAILURE.** ## Task Description {task_instruction} ## Agent’s Final Claim {submit_message}

Figure 13. The LLM-as-a-Judge prompt for verification of the evidence submitted by the agents (part 2).

