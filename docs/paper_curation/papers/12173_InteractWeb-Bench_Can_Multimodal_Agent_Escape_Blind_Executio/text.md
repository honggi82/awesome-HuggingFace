## arXiv:2604.27419v1[cs.AI]30Apr2026

[Figure 1]

IP Intelligence 2026-05

# InteractWeb-Bench: Can Multimodal Agent Escape Blind Execution in Interactive Website Generation?

###### Qiyao Wang1,2,∗ Haoran Hu3,∗ Longze Chen1,2 Hongbo Wang3 Hamid Alinejad-Rokny4 Yuan Lin3† Min Yang1,5†

1 Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences 2 University of Chinese Academy of Sciences 3 Dalian University of Technology 4 UNSW Sydney 5 Shenzhen University of Advanced Technology

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

wangqiyao25@mails.ucas.ac.cn zhlin@dlut.edu.cn min.yang@siat.ac.cn

Website InteractWeb-Bench ∗ Equal Contribution. † Corresponding Authors.

Abstract | With the advancement of multimodal large language models (MLLMs) and coding agents, the website development has shifted from manual programming to agent-based project-level code synthesis. Existing benchmarks rely on idealized assumptions, especially for well-structured, information-rich inputs and static execution settings. In contrast, real-world development is constrained by a critical bottleneck: the semantic misalignment between ambiguous, low-quality instructions from non-expert users and model understanding, which results in a failure mode that we term blind execution. To address this gap, we introduce InteractWebBench, the first multimodal interactive benchmark for website generation under non-expert low-code user conditions. InteractWeb-Bench introduces four types of user agents and persona-driven instruction perturbations to systematically simulate diverse user behaviors, including ambiguity, redundancy, and contradiction, grounded in requirement engineering defect taxonomies. We develop an interactive execution environment for agents, featuring a unified action space comprising Clarify, Implement, Verify, and Submit, enabling iterative intent refinement, code synthesis, and visual feedback–based validation. Extensive experiments and analysis reveal that frontier MLLM-based agents remain trapped in blind execution, exposing limitations in intent recognition and adaptive interaction.

### 1. Introduction

With the advancement of multimodal large language models (MLLMs) and coding agents [Anthropic, 2025, Singh et al., 2025], the paradigm of website development is evolving from traditional manual code authoring towards natural language-guided code synthesis [Lu et al., 2025]. This substantially reduces the barriers to professional software development, allowing non-expert users to directly participate through natural language instructions [Hou et al., 2024]. In an ideal scenario, the agent should accurately parse user requirements and end-to-end perform code synthesis to generate webpages. However, the inherent ambiguity and low-code nature of instructions from non-expert users often lead to a significant semantic gap between their true intent and model’s understanding [Zamfirescu-Pereira et al., 2023] in practice. This constitutes one of the primary bottlenecks for website generation in moving toward general real-world deployment [Jimenez et al., 2024].

Existing website generation benchmarks are confined to an idealized evaluation setting [Lu et al., 2025]. In these environments, the instructions provided are often standardized, with clear structures, rigorous logic, and comprehensive frontend details. However, real-world interactive scenarios are far from perfect. Constrained by a lack of domain knowledge, especially in programming, the initial requirements provided by non-expert users typically exhibit highly ambiguous and fragmented characteristics. Furthermore, diverse user groups exhibit drastically different expression styles, such as minimalist abstract descriptions, divergent expressions containing substantial redundant information, or even conflicting requirements that violate technical common sense [Luo et al., 2025].

Interactive Agent Execution Environment

Requirement Engineering Defect Taxonomies

MLLM-based Website Generation Agent

Golden Query

[Figure 7]

###### 𝒓 𝑰 𝑽 𝒓 𝒓 𝑰 𝑺 Blind Execution

[Figure 8]

[Figure 9]

P-MIN P-INT

Clarify Interactuser agentwith

User Agent

User Agent

[Figure 10]

Implement SynthesisCode

[Figure 11]

I need to ask the user what color button to use. Red.

Persona-Driven User Agent

Verify GUI-basedFeedback

Mutated Query

𝒓 𝑰 𝑽 𝒓 𝒓 𝑰 𝑺

𝑪 𝒓 𝑪

P-RAM

P-CON

Submit Final Solution

Ours: Provides a user-agent interaction setting for evaluation

##### INTERACTWEB-BENCH

Action Space

[Figure 12]

[Figure 13]

###### GUI-based Verification Submit Constraint-Slot-Based Task Completion Rate (TCR)

###### Code Implementation

[Figure 14]

[Figure 15]

###### Testing Checklist (Oracle Slots)

[Figure 16]

[Figure 17]

[Figure 18]

- 1. Verify the portal title…
- 2. Verify the lavender gradient background…
- 3. Click the "Technology"

###### Slot ID [0]: Check the filter function for regulatory policies.

[Figure 19]

###### Slot ID [1]: Assess the sorting functionality within the dashboard. …

npm run dev touch vite.config.js

Scroll Click button

[Figure 20]

✅ Pass

[Figure 21]

❌ Fail 7 Oracle Slots

button…

3 Passed, 4 Failed ⇒ TCR= 42.85%

GUI Agent-based Evaluator

Figure 1 | The overall architecture of InteractWeb-Bench. Where the top-right circles denote actions in an illustrative trajectory: r (reasoning), C (Clarify), I (Implement), V (Verify), and S (Submit).

Faced with such low-quality, high-variance instructions, existing agent commonly fall into the trap of Blind Execution. It passively adapts flawed queries, lacks the ability to effectively infer the user’s true intent, and cannot proactively request clarification [Wu et al., 2025] or employ dynamic reasoning-and-acting strategies [Yao et al., 2023] to bridge missing information gaps. These models typically default to bypassing necessary intent verification, directly synthesizing code from incomplete or conflicting instructions. This unidirectional execution strategy directly leads to elevated task failure rates, UI rendering anomalies, and high-frequency functional hallucinations [Ji et al., 2023].

To escape the trap of blind execution, an agent must possess robust capabilities in dynamic intent recognition. While recent studies emphasize proactive conversational assistance [Lu et al., 2024] and intent clarification in language and code models [Luo et al., 2025, Wu et al., 2025], it is fundamentally insufficient to confine interactions to text-centric dialogues in the highly interdependent and visually intensive domain of frontend engineering. The non-expert user requirements are rarely articulated perfectly upfront, agents must make interleaved decisions, dynamically switching between clarifying ambiguous user intent and verifying visual webpage outputs in an executable environment.

To address these limitations, we introduce InteractWeb-Bench, the first multimodal interactive benchmark for non-expert user low-code website generation, as illustrated in Figure 1. It is designed to evaluate whether agents can move beyond blind execution by dynamically recognizing user intent and interacting with users. To authentically simulate the complexity of user instructions, we construct four typical user personas (i.e., P-MIN, P-RAM, P-INT, and P-CON), grounded in standard Requirement Engineering defect taxonomies [Van Lamsweerde, 2009, Wu and Fard, 2025], to inject intent traps serving as stress tests. Specifically, to enable agents to navigate these highly uncertain inputs, we designs a unified multi-path action space comprising natural language clarification (Clarify), code synthesis (Implement), visual verification (Verify), and task submission (Submit). Within this environment, the agent dynamically selects any of these actions based on the real-time state. It freely toggles between eliciting requirements, writing code, and visually inspecting the rendered UI. Finally, it delivers the completed website to our designed constraint–slot-based evaluating procedure. We envision our InteractWeb-Bench could catalyze the evolution of MLLM-based website generation agents from passive instruction followers to intent-aligned collaborators capable of escaping blind execution. Our main contributions are as follows:

- • We focus on the critical bottleneck of Blind Execution in website generation agents. To address the limitations of idealized static benchmarks, we incorporate the user agents and persona-driven

- instruction mutation to better capture the ambiguity and heterogeneity of user intents.
- • To the best of our knowledge, our InteractWeb-Bench is the first benchmark to evaluate MLLMbased agents’ ability to interact with users and satisfy their needs in website generation, by endowing the models with a unified multi-path action space (Clarify, Implement, Verify, Submit).
- • We conduct extensive experiments on various frontier MLLM-based agents and provide in-depth analyses of their behavioral patterns, especially for user–agent interaction, intent recognition, and inherent task-solving capabilities. Our findings reveal key factors for improving website generation agents and better aligning their behavior with the needs of non-expert users.

### 2. Related Work

Website Generation Benchmarks. The application of MLLMs and coding agents in software engineering has advanced automated code generation [Austin et al., 2021, Chen et al., 2021], with increasing research focusing on website generation. Early work formulate this task as a static vision-to-code translation problem. For instance, Design2Code [Si et al., 2025] and Web2Code [Yun et al., 2024] assess the ability to convert high-fidelity webpage screenshots or design specifications into static HTML/CSS implementations. WebGen-Bench [Lu et al., 2025] advanced the paradigm by evaluating the generation of multi-page web applications from scratch based on comprehensive textual instructions. However, existing benchmarks rely on highly standardized and comprehensive instructions, assuming the initial query fully captures user preferences and failing to evaluate models under the high-variance, ambiguous, and contradictory inputs typical of real-world scenarios. In contrast, our InteractWeb-Bench explicitly targets this gap by introducing persona-driven instruction variance to test models under realistic, noisy conditions.

Intent Clarification and User Interaction. Recent work actively explores proactive intent clarification in response to ambiguous instructions. ClarifyCoder [Wu et al., 2025] enables models to request missing specifications via instruction-tuning. HumanEvalComm [Wu and Fard, 2025] and ClarifyMTBench [Luo et al., 2025] introduce flawed requirements and noisy user personas to evaluate robustness under incomplete, ambiguous or noisy instructions. In website generation, Persona2Web [Kim et al., 2026] have explored ambiguous queries, but they restrict agents to passively inferring intents from browsing histories rather than actively communicating. While these work realize the importance of requirement elicitation and ambiguity resolution, they remain fundamentally confined to either pure textual content or passive inference. In the highly visual field of website frontend engineering, agents must leverage iterative visual rendering feedback to enhance their understanding. Our InteractWebBench bridges this gap by integrating language requirement elicitation with visual verification.

Interactive Agents and Execution Environments. The evaluation paradigm of LLMs has shifted from static code generation to interactive agents operating within environments [Yao et al., 2023]. Generalpurpose code agent frameworks, such as SWE-agent [Yang et al., 2024] and OpenHands [Wang et al., 2025], interact with bash terminals and code editors to obtain execution feedback (e.g., compiler errors, test outputs). Correspondingly, benchmarks like SWE-bench [Jimenez et al., 2024] and InterCode [Yang et al., 2023b] evaluate agents based on their ability to use this feedback over multiple turns to resolve real-world GitHub issues. However, these environments focus primarily on passing objective, predefined functional tests. They largely ignore the human-AI collaborative process, failing to assess whether an agent can proactively elicit and satisfy unstated user needs. Because non-expert users frequently provide flawed or incomplete instructions, an agent must move beyond mere code execution and escape the trap of blind execution. This motivates our InteractWeb-Bench, which immerses the agent in a comprehensive multi-path action space (Clarify, Implement, Verify, Submit), enabling it to autonomously toggle between textual requirement clarification, code synthesis, and GUI-based visual debugging to fulfill the user’s true intent.

### 3. InteractWeb-Bench

To evaluate whether MLLM-based agents can escape the blind execution and navigate the ambiguity of real-world user requirements in website generation, we propose InteractWeb-Bench, an Interactive Website generation Benchmark. As illustrated in Figure 1 and Algorithm 1, it incorporates user simulation modules to evaluate agents in natural language-driven website generation tasks. Specifically,

- our framework consists of two core components: (1) Persona-Driven User Agent Module, comprising four carefully designed user agents that simulate diverse human inputs and introduce targeted intent traps (See Section 3.1); and (2) Interactive Execution Environment, featuring a unified action space that empowers the agent to autonomously toggle among language clarification (Clarify), code synthesis (Implement), GUI-based visual verification (Verify), and task submission (Submit) (See Section 3.2). Unlike prior benchmarks, InteractWeb-Bench is designed to evaluate agents’ capability in website generation, focusing on requirements elicitation, cross-modal comprehension, code synthesis, and iterative refinement.

- 3.1. Persona-Driven User Agent Module

Algorithm 1: InteractWeb-Bench Evaluation

- 1: // Persona-driven Initialization
- 2: Instruction I ← 𝑂𝑝(I∗);
- 3: Initialize user agent 𝑈𝑝 and execution agent 𝜋M;
- 4: Initialize environment state 𝑠0, Timestep 𝑡 ← 0;
- 5: // Interactive Agent Execution
- 6: while not terminated do
- 7: 𝑎𝑡 ∼ 𝜋M(𝑠𝑡).
- 8: 𝑎𝑡 ∈ {clarify, implement, verify, submit}.
- 9: Execute 𝑎𝑡 via (one-choice):
- 10: Clarify → Interact with 𝑈𝑝;
- 11: Implement → Generate code;
- 12: Verify → GUI inspection;
- 13: Submit → Terminate proactively.
- 14: end while
- 15: // Final Evaluation
- 16: Compute final task completion rate score.

Persona-Driven Instruction Mutation. In practical software engineering, initial user requirements are frequently underspecified or articulated with high variability. To systematically simulate these realworld communication barriers, our persona design is theoretically grounded in standard Requirement Engineering (RE) defect taxonomies [Van Lamsweerde, 2009, Wu and Fard, 2025] (e.g., incompleteness, ambiguity, and inconsistency) and Grice’s Maxims of Conversation [Grice, 1975]. We formalize these theoretical requirement flaws into four distinct user personas. Let I∗ denote the original, unambigu-

- ous golden instruction. By defining these four personas, we apply a persona-specific metamorphic operator 𝑂persona to generate the perturbed instruction Imutated = 𝑂persona(I∗) [Chen et al., 2020], emulating specific types of human users:

- • Minimalist User Agent (P-MIN): Represents Requirement Incompleteness (violating the Maxim of Quantity), which simulates a user who provides extremely sparse details. Specifically, we apply an abstraction operator to remove secondary constraints (e.g., colors or layout-related terminology), retaining only the core verb–object functional request. This rigorously tests the agent’s ability to proactively elicit missing requirements and apply sensible design defaults.
- • Rambling User Agent (P-RAM): Represents Low Signal-to-Noise Ratio (violating the Maxim of Relation), which simulates a highly talkative and unfocused user. Specifically, we utilize a noise-injection operator to bury core functional requirements within substantial non-functional conversational digressions and irrelevant background context, challenging the agent’s capacity for noise filtering, robust information extraction, and instruction tracking amidst distracting inputs.
- • Intuitive User Agent (P-INT): Represents Requirement Ambiguity and Informality (violating the Maxim of Manner), which represents a non-technical user highly focused on sensory experiences and aesthetics. Specifically, we employ a cross-modal semantic shift to translate precise frontend engineering terminology into abstract sensory metaphors and emotional modifiers, testing the agent’s visual and textual alignment capabilities.
- • Conflicting User Agent (P-CON): Represents Requirement Contradiction (violating the Maxim

of Quality), which simulates a confused or demanding user who provides mutually exclusive constraints. Specifically, we introduce logical inconsistencies and frontend paradoxes into the interdependent requirements, evaluating whether the agent can accurately identify these invalid propositions and proactively question the user to resolve the conflict back to the original truth.

User Agent Response Design. The second stage of user agent engagement involves dynamic, multiturn communication with execution agent. The execution agent must proactively elicit missing information and resolve conflicts under ambiguous or incomplete initial instructions, while avoiding shortcut strategies that directly solicit complete solutions from the user. We employ a two-stage retrieval and containment pipeline for user agent to prevent the simulated user from inadvertently revealing the entire unperturbed golden instruction upon a single query.

First, the user agent analyzes the execution agent’s clarification request and retrieves the internal ground-truth specification. By extracting only the explicitly requested details and withholding all unasked requirements, the system effectively prevents information leakage. Second, through personaconditioned response generation, the isolated factual snippet is processed via a persona-specific prompt filter, seamlessly weaving the oracle data into the assigned user identity’s conversational style. For instance, if asked about missing background color, the P-RAM user agent will embed the exact true requirements within substantial irrelevant conversational noise. Additionally, if the execution agent correctly identifies a technical paradox, the P-CON user agent is instructed to briefly defend its original contradictory stance before ultimately yielding the accurate baseline truth.

###### 3.2. Interactive Agent Framework and Execution Environment

We instantiate the evaluated MLLM as an autonomous website generation agent within an interactive execution environment. Through this setup, the agent iteratively engages with user instructions. It structures the agent’s behavior through a unified action space and a multi-modal feedback loop, testing the holistic capabilities of requirement elicitation, code implementation, and visual comprehension.

Action Space (A) and Autonomous Trajectory Dynamics. Unlike fixed workflows, the agent operates in our environment with a fully autonomous, non-linear manner. At any step 𝑡, based on its current observation and internal reasoning, the agent dynamically selects an action from a predefined discrete action space A = {𝑎clarify, 𝑎implement, 𝑎verify, 𝑎submit}. If it detects ambiguity or encounters unforeseen logical gaps in the user instructions, it can execute 𝑎clarify to obtain missing details from the simulated user. This explicitly measures the model’s intrinsic capability for proactive requirement negotiation. To fulfill technical requirements, it uses action 𝑎implement to synthesize artifacts, manage dependencies, or execute shell commands. The agent can autonomously trigger 𝑎verify to initiate a GUI inspection to assess its progress. Ultimately, action 𝑎submit is issued when the agent confidently concludes that the synthesized website fully satisfies the user’s intent, terminating the trajectory.

GUI-based Verification and Multi-modal System Feedback. The agent enables to interact with the environment through the action 𝑎verify. To initiate the verification process, the agent needs to formulate a test criteria block, which is a structured checklist specifying the features to be validated. Upon invoking action 𝑎verify, the environment dynamically synthesizes a holistic memory, which fuses the complete user-agent interaction history, the structural context of the latest code artifacts, and prior visual audit trajectory. This ensures the agent has access to all explored user requirements, grounds the visual inspection in the underlying code, and prevents repetitive debugging loops. Guided by both the explicitly self-defined criteria and the memory, the agent interactively navigates the webpage interface (e.g., via click and scroll) to explore the interface and identify potential issues.

Additionally, we employ an autonomous error-exit strategy, the agent actively terminates the exploration once it detects a semantic misalignment, a functional UI flaw, or a critical system anomaly

(e.g., Uncaught TypeError). The environment then distills these failures into a highly structured verification output sent back to the agent. This composite multi-modal feedback encompasses three precise components: (1) the specific terminal UI screenshot at the point of failure, (2) the purified critical browser console errors, and (3) a explicit reasoning trace of the agent explaining why the visual criteria was not met. These feedback provide the agent with actionable insights, enabling it to accurately identify and correct errors before reverting to code implementation action 𝑎implement.

Exploration Boundary and Final Evaluation. A significant challenge in MLLM-based agents is their susceptibility to infinite debugging loops and context exhaustion. To ensure evaluation tractability, we enforce a dual-boundary constraint mechanism scaled to the intrinsic complexity of the task (Easy, Middle, Hard). First, to prevent endless execution, the global trajectory is constrained by a maximum turn limit (𝑇total ∈ {15, 20, 25}). Second, to mitigate localized infinite loops during debugging, we restrict consecutive verification errors using a dynamic exploration boundary (𝑇error ∈ 6, 8, 10). Where the trajectories exceeding either threshold are forcefully terminated.

###### 3.3. Constraint-Slot-Based Evaluation Metric and Dataset Statistics

To objectively evaluate the generated website, we adopt a fine-grained, constraint-slot-based scoring mechanism. First, we decompose each task into a set of atomic constraints, referred to as Oracle Slots, each defined as: 𝑆 = ⟨Target_Component, Expected_Result, Assertion_Type⟩. In addition, we introduce a dedicated anti-hallucination slot to capture unrequested or redundant UI elements. This slot is excluded from task completion scoring and is used solely for hallucination evaluation.

Oracle Slot Weighting. To reflect both website implementation complexity and testing density, we group slots by their target component. For each group 𝐺, the weight of a slot 𝑖 ∈ 𝐺 is defined as:

𝐶tech(𝐺) · 1 + 0.5(𝑁𝐺 − 1)

(1)

𝑊𝑖 =

𝑁𝐺

where 𝐶tech ∈ {1, 2, 3} is a step-wise metric that assigns 1.0 to static elements (e.g., pure CSS), 2.0 to interactive elements (e.g., basic JavaScript) and 3.0 to complex logic (e.g., Fetch API). The 𝑁𝐺 is the number of constraints within the group. This captures the diminishing marginal effort when extending existing components and prevents simple constraints from dominating the metric.

Main Metric: Task Completion Rate. During the final evaluation phase (i.e., post-submission), we instantiate an independent visual evaluator powered by the WebVoyager [He et al., 2024] and augmented with Set-of-Mark (SoM) [Yang et al., 2023a] prompting to conduct a multi-step visual audit. We leverage the Task Completion Rate (TCR) to measure the task-solving-oriented performance of the agent, i.e.

𝑊𝑖 𝑗∈Stotal 𝑊𝑗

TCR = 𝑖∈Spass

(2)

where Stotal denotes the oracle slot set for the given task and Spass ⊆ Stotal are the slots successfully verified by the evaluator. We also leverage the Hallucination Rate to detect the unrequested elements, which was computed as the proportion of tasks that trigger such violations.

Slot-Driven Difficulty Clustering and Benchmark Scaling. Let the total complexity score of a task be 𝑆task = 𝑖∈Stotal 𝑊𝑖. We initially curated 101 high-quality seed websites derived from WebGenBench [Lu et al., 2025]. The number of atomic constraints within each task ranges from 7 to 12. Then, we applied the 𝐾-Means clustering algorithm (𝑘 = 3) to the score distribution. This divides the 101 seed tasks into three difficulty levels, as summarized in Table 1. To evaluate models against users as described in Section 3.1, we process seed tasks using four persona-driven mutation operators (i.e.,

P-MIN, P-RAM, P-INT and P-CON). This augmentation expands the dataset into a comprehensive final benchmark suite comprising 404 dynamic test cases.

Difficulty Level # Seed # Task Percentage Avg. # Slots Score Range (𝑆task) Cluster Center Easy 21 84 20.8% 5.90 [7.99, 10.00] 9.38 Middle 54 216 53.5% 7.28 [11.00, 14.00] 12.57 Hard 26 104 25.7% 8.88 [15.00, 22.00] 16.73

All 101 404 100.0% 7.41 [7.99, 22.00] 12.98

- Table 1 | Distribution of task difficulty levels based on constraint complexity.

4. Experiments

4.1. Experimental Setup

We evaluate various MLLMs, covering a broad range of sizes,and families, with implementation details and costs are in Appendix A. Each model is instantiated as a website generation agent based on the bolt.diy [stackblitz labs, 2024] framework. The interactive environment is implemented using a playwright-based browser kernel, enabling dynamic rendering, interaction, and GUI-based verification.. The user agents are simulated by DeepSeek-V3.2 [DeepSeek-AI, 2025] under persona-specific settings, and all tasks are executed from scratch based solely on textual instructions. Final evaluation is conducted using a WebVoyager [He et al., 2024] augmented with Set-of-Mark prompting [Yang et al., 2023a], where GPT-5-mini serves as the evaluator to determine slot-level outcomes.

- 4.2. Main Results

|Models|Easy Middle Hard|P-MIN P-RAM P-INT P-CON<br><br>|TCR ↑ Hallu. Rate ↓|
|---|---|---|---|
|Qwen3.6-Plus Kimi-K2.5 Qwen3.5-397B-A17B Gemma-4-31B-it GPT-4.1 Gemma-4-26B-A4B-it GPT-4.1-mini Gemini-3.1-Flash-Lite Qwen3.5-9B|43.05 37.27 38.46<br>44.70 39.03 31.48 43.49 37.65 30.98 43.35 36.39 32.98 42.99 34.15 27.72<br><br><br>40.19 30.47 22.29<br>41.84 27.58 23.08 37.07 22.78 22.64 31.47 22.16 23.07<br>|26.54 53.66 36.34 38.58<br><br>27.18 48.86 38.32 38.69<br><br><br>25.91 47.32 37.81 37.54<br><br>26.73 52.89 35.70 32.50 24.96 42.56 34.19 35.61 22.16 42.99 26.33 30.07 21.77 40.02 26.16 29.62 15.78 35.03 27.03 25.03 17.78 29.74 25.37 24.46<br><br><br>|38.78 62.4% 38.26 64.1% 37.15 56.4% 36.96 61.7% 34.33 31.7% 30.39 72.3% 29.39 23.5% 25.72 27.7% 24.33 53.7%<br><br>|

Categorized by Difficulty Categorized by User Persona Overall

- Table 2 | Overall performance of evaluated website generation agents. TCR denotes the Task Completion Rate and Hallu. Rate refers to Hallucination Rate. The best performance is bolded and the underline is the second best.

Our main results are reported in Table 2. The TCR scores remain limited across all models, with the best-performing model, Qwen3.6-Plus, achieving only 38.78%. This indicates that current agent struggle to fully satisfy user requirements in realistic interactive settings. Additionally, agent performance degraded with task difficulty increased. Furthermore, agent performance differs substantially between user personas. Almost all models achieve higher TCR score under the P-RAM setting, while significantly lower performance is observed under P-MIN setting. This suggests that models are sensitive to different types of instruction perturbations, particularly when key information is missing.

###### 4.3. Analysis

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | |Gemma-|Qw<br><br>4-31B-it<br><br>GPT-4.|en3.5-3 1-mini|97B-A17|B|
| |Qwen3.|6-Plus| | | | | |
| | |Ge|mma-4-2|6B-A4B-|Kimi-<br><br>it|K2.5<br><br>GPT|-4.1|
| | | | | | | | |
| | | | | | |Qwen3|.5-9B|
| | | | | | | | |
| | |Ge|mini-3.1|-Flash-Li|te| | |
| | |Bette|r Intent|Unders|tanding| | |
| | | | | | | | |

- Finding 1: Although agents achieve strong overall intent understanding, they struggle to proactively identify and clarify underspecified user requirements, i.e. traped in blind execution. As illustrated in Figure 2, we analyze user-agent interactions to assess clarification behavior and intent alignment. We leverage two metrics: (1) Intent Alignment Score (IAS), which measures how well the agent’s reasoning matches the ground-truth requirements on a 1-5 scale and (2) Clarification Hit Rate (CHR), which assesses whether the agent’s questions in

action 𝑎clarify successfully identify hidden constraints. Both metrics are evaluated by GPT-5mini, and the detailed prompts are provided in Appendix D. Although IAS consistently exceeds 3.90, clarification effectiveness (CHR) remains below 40% across all models. This gap indicates that while models can capture a coarse understanding of user intent, they rarely identify or explicitly resolve missing or ambiguous requirements. Instead, models tend to treat underspecified instructions as complete and proceed directly to implementation.

Models Avg. LoC HR (%) ↓ # Clarify SR (%) VBR (%) ↓

| | | | | | |
|---|---|---|---|---|---|
|Qwen3.6-Plus Kimi-K2.5 Qwen3.5-397B-A17B Gemma-4-31B-it GPT-4.1 GPT-4.1-mini Gemma-4-26B-A4B-it Gemini-3.1-Flash-Lite Qwen3.5-9B|1415.7 1942.3 1230.4 598.2 440.1 473.0 674.6 137.2 1093.5|62.4% 64.1% 56.4% 61.7% 31.7% 23.5% 72.3% 27.7% 53.7%|0.01 0.07 0.08 0.03 0.53 0.94 0.03 0.01 0.16|95.0 90.3 98.0 77.5 74.3 56.4 91.8 95.8 91.8|5.2 5.2 2.2 12.9 10.2 4.8 12.9 3.8 6.9|

Table 3 | Comparison of generation scale (Avg. LoC) and hallucination rate (HR). Where SR refers to Proactive Submit Rate, and VBR refers to Visual Bug Rate. The # Clarify refers to the average number of clarification actions taken.

- Finding 2: Models over-generate code to compensate for missing requirements, increasing hallucinations instead of seeking clarification. We analyze code scale and hallucination in model trajectories, reported in Table 3. We calculate the average lineof-code (Avg. LoC) in the final workspace folder. Meanwhile, we detect unrequested elements that are not specified in the oracle slots across the entire website to calculate the Hallucination Rate (HR) for each agent. Notably, such as Qwen3.6-Plus and Kimi-K2.5, produce large-scale code artifacts (e.g., Avg. LoC exceeding 1000), while maintaining a high hallucination rate (> 60%). The combined metrics suggest that models tend to compensate for missing requirements through aggressive generation rather than iterative clarification, resulting in increased code size and higher hallucination rates. Although stronger models achieve higher TCR, they also exhibit increased hallucination rates. In website generation, such hallucinations introduce unwanted functionalities, redundant components, and misaligned designs, degrading usability, and increasing maintenance overhead.
- Finding 3: Agents fail to effectively utilize GUI-based verification feedback for meaningful execution improvement. We further analyze interaction trajectories and verification behaviors to evaluate whether models can effectively leverage GUI-based verification feedback for self-reflection and improved execution. As shown in Figure 3, we report the average execution steps (Avg. Steps) and verification intensity, measured by the Visual Cautiousness Index (VCI), which is defined as the ratio of verification actions to implementation actions. The Qwen series models exhibit a more cautious behavioral pattern, invoking more GUI-based verification actions rather than relying solely on code

40.0

###### ClarificationHitRate(%)

37.5

BetterProactiveClarificationQuality

35.0

32.5

30.0

27.5

25.0

22.5

20.0

3.93 3.94 3.95 3.96 3.97 3.98 3.99 4.00

Intent Alignment Score (1-5 scale)

Figure 2 | User-Agent interaction quality landscape: Intent Alignment vs. Proactive Clarification.

implementation. In contrast, the Gemini and Gemma series models exhibit greater confidence in their code, but this often leads to more errors, with Gemini-3.1-Flash-Lite achieving only a 25.72 TCR score. Beyond specific model behavior patterns, we observe substantial variation in both trajectory length (Avg. Steps ranging from 5.02 to 12.87) and verification intensity (VCI ranging from 0.35 to 1.51). However, these differences do not translate into clear improvements in task completion. This suggests that models fail to effectively leverage GUI-based feedback to revise their initial assumptions and address errors. Instead, models tend to apply local fixes to observed issues without reconsidering whether the underlying requirements are incomplete or incorrect.

GPT-4.1-mini

- 5

- 6

- 7

- 8

- 9

- 10

- 11

- 12

- 13

Overthinking Inefficient

Gemma-4-31B

###### AverageSteps

GPT-4.1

Qwen3.5-9B

Kimi-K2.5

Qwen3.6-Plus

Gemma-4-26B

Cautious and Efficient

Gemini-Flash

Qwen3.5-397B-A17B

Fast but Risky

0.4 0.6 0.8 1.0 1.2 1.4

VCI (Verification Cautiousness Intensity)

Figure 3 | Model behavior landscape: Verification Intensity vs. Execution Efficiency (Avg. Step).

- Finding 4: User-persona-based analysis uncovers that models are significantly more sensitive to missing information than to noisy instructions. We analyze the model performance under different user-persona settings. As shown in Figure 4 (a), models exhibit uneven performance across persona settings with varying levels of information exposure. Notably, all models maintain a relatively higher TCR under the P-RAM setting, whose instructions contain redundant or noisy information. However, performance drops significantly under the P-MIN setting, where key information is missing and agents must rely more on clarification actions to interact with the user. This pattern reveals that models are more robust to noisy inputs than to missing information, highlighting a key challenge in handling underspecified user requirements. We also conduct an experiment using the unambiguous golden instruction I∗, reporting the average performance of GPT-4.1-mini and Gemini-3.1-Flash-Lite in Figure 4 (b). Experimental results demonstrate that agents executing under golden instructions without persona perturbations achieve higher TCR and lower hallucination, yet remain far from realistic settings with ambiguous user instructions.
- Finding 5: Models differ fundamentally in how they balance exploration and commitment during interaction. Beyond task-oriented performance, models exhibit distinct decision-making behaviors in navigating the interaction process. These differences are reflected in their choices of clarification timestamp, exploration degree, visual verification utilization, and the confident submission timepoint. We also report the Proactive Submit Rate (SR) in Table 3. Notably, GPT-4.1-mini tends to ask more questions and follow a longer interaction, yet exhibits lower SR. This reflects a more exploratory but less decisive pattern, where the model continues refining intermediate outputs without confidently committing to a final solution. In contrast, Qwen3.6-Plus and Kimi-K2.5 ask fewer questions, produce

###### (a) Performance Across Persona Dimensions

60

| |53.66<br><br>48.86<br><br>52.89<br><br>P-MIN<br><br>P-RAM<br><br>P-INT<br><br>P-CON| | | | |
|---|---|---|---|---|---|
| |42.56 38.58 38.32 38.69| | | | |
| |26.54 27.18 26.73<br><br>24.96<br><br>36.34 35.70 34.19 32.50<br><br>35.61| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

50

40

TCR

30

20

10

0

Qwen3.6-Plus Kimi-K2.5 Gemma-4-31B-it GPT-4.1

###### (b) TCR and Hallucination Rate: Persona vs. Golden Instruction

49.0

###### Persona mutated Golden *

40.9

| |
|---|

38.9

38.5

40

37.1

Score

27.7%

25.7

22.8

22.6

20

10.9%

0

Easy Middle Hard Overall Hallucination Rate

Figure 4 | (a) Performance across differrnt userpersona, where the gray dashed line indicates the average TCR of each model. (b) Performance (TCR and Hallucination Rate) under persona-driven and golden instruction.

larger code outputs, and achieve substantially higher submission rates with shorter trajectories. This indicates a more decisive execution style, where models commit early to a final solution, even at the risk of introducing hallucinated or misaligned components. These behavioral patterns across models reflect varying trade-offs between exploration, decisiveness, and final task performance.

- Finding 6: Models exhibit a ceiling effect in layout-level aesthetics, reliably producing structurally valid websites but remaining limited by slight visual imperfections. We further examine the visual quality of generated websites using the Visual Bug Rate (VBR), which measures the frequency of UI rendering failures (e.g., text overlap, container overflow), reported in Table 3. Although VBR remains relatively low in most models (typically below 10%), visual defects are consistently present rather than eliminated. Additionally, we evaluate the aesthetic quality of generated websites using a GPT-5-mini evaluator based on a 5-point Likert aesthetic score, as shown in Figure 5. The evaluation criteria (See Appendix D) includes two dimensions: (i) Visual Layout, focusing on composition, design, visual elements, and structure; and (ii) Creative Alignment, focusing on originality, creativity, theme, and communication. Kimi-K2.5 achieves the highest aesthetic quality, whereas GPT-4.1-mini exhibits relatively weaker aesthetic performance. Notably, visual layout and creative alignment vary little across models. This suggests that models can produce structurally complete layouts without major rendering failures, while still exhibiting subtle visual defects such as misalignment or layout inconsistencies. We conduct a human evaluation of the aesthetic quality of the generated websites using the same criteria. Detailed results are provided in Appendix B. The websites generated by Kimi-K2.5 demonstrate the best aesthetic quality both in MLLM judges and human judges. The MLLM judges show a moderate correlation with human (Kendall’s 𝜏 = 0.4490), while human experts exhibit moderate inter-rater agreement (Kendall’s 𝜏 = 0.5675).

Qwen3.5-397B-A17B Gemma-4-31B-it

Visual Layout

Creative Alignment

Kimi-K2.5

Kimi-K2.5

3.96 Kimi-K2.5

3.86

4.0

GPT-4.1

3.5

3.0

Qwen3.6-Plus

GPT-4.1-mini

Qwen3.5-9B

Gemma-4-26B-A4B-it

Gemini-3.1-Flash-Lite

Figure 5 | MLLM judge and human evaluation of website aesthetic quality across models.

Qualitative Analysis. To further understand the behavioral patterns behind these findings, we analyze representative trajectories under different persona settings. We present detailed generated website cases for the golden instruction and four user persona instructions in Table 6 (See Appendix C). These cases reveal a consistent pattern across models and personas: rather than identifying underspecified or conflicting requirements and issuing a Clarify action, agents prematurely enter the implementation phase, failing to proactively infer user intent and becoming trapped in blind execution. Additionally, we provide website cases generated by different models in Figure 7 (See Appendix C).

### 5. Conclusion

In this paper, we introduce InteractWeb-Bench, an interactive benchmark designed to evaluate whether MLLM-based agents can effectively handle ambiguous and underspecified user instructions in realistic website generation scenarios. Through experiment, we observe a blind execution tendency across current agents. Specifically, the generated websites frequently appear plausible at a surface level but deviate from the user’s actual needs. Our in-depth analysis suggest that advancing website generation agents requires not only stronger coding capabilities but also the ability to recognize missing information, actively elicit clarifications, and update assumptions based on feedback. We

envision that our InteractWeb-Bench can serve as a foundation for future MLLM-based coding agents on developing more interactive and user-aligned generation websites.

### References

Anthropic. Claude 3.7 Sonnet System Card, 2025. URL https://api.semanticscholar.org/ CorpusID:276612236.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

Mark Chen, Jerry Tworek, Heewoo Jun, et al. Evaluating large language models trained on code,

#### 2021. URL https://arxiv.org/abs/2107.03374.

T. Y. Chen, S. C. Cheung, and S. M. Yiu. Metamorphic testing: A new approach for generating next

test cases, 2020. URL https://arxiv.org/abs/2002.12543.

DeepSeek-AI. Deepseek-v3.2: Pushing the frontier of open large language models. ArXiv,

abs/2512.02556, 2025. URL https://api.semanticscholar.org/CorpusID:283448719. H. P. Grice. Logic and conversation. In Peter Cole and Jerry L. Morgan, editors, Speech Acts, volume 3

of Syntax and Semantics, pages 41–58. Academic Press, 1975.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6864–6890, 2024.

Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, and Haoyu Wang. Large language models for software engineering: A systematic literature review. ACM Trans. Softw. Eng. Methodol., 33(8), December 2024. ISSN 1049-331X. doi: 10.1145/3695988. URL https://doi.org/10.1145/3695988.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12), March 2023. ISSN 0360-0300. doi: 10.1145/3571730. URL https://doi.org/ 10.1145/3571730.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024. URL https://arxiv.org/abs/2310.06770.

Serin Kim, Sangam Lee, and Dongha Lee. Persona2web: Benchmarking personalized web agents for contextual reasoning with user history, 2026. URL https://arxiv.org/abs/2602.17003.

Kimi Team. Kimi k2.5: Visual agentic intelligence. 2026. URL https://api.semanticscholar. org/CorpusID:285269548.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Haotong Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. Proceedings of the 29th Symposium on Operating Systems Principles, 2023. URL https://api.semanticscholar.org/CorpusID:261697361.

Yaxi Lu, Shenzhi Yang, Cheng Qian, Guirong Chen, Qinyu Luo, Yesai Wu, Huadong Wang, Xin Cong, Zhong Zhang, Yankai Lin, Weiwen Liu, Yasheng Wang, Zhiyuan Liu, Fangming Liu, and Maosong Sun. Proactive agent: Shifting llm agents from reactive responses to active assistance, 2024. URL https://arxiv.org/abs/2410.12361.

Zimu Lu, Yunqiao Yang, Houxing Ren, Haotian Hou, Han Xiao, Ke Wang, Weikang Shi, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Webgen-bench: Evaluating llms on generating interactive and functional websites from scratch, 2025. URL https://arxiv.org/abs/2505.03733.

Sichun Luo, Yi Huang, Mukai Li, Shichang Meng, Fengyuan Liu, Zefa Hu, Junlan Feng, and Qi Liu. Clarifymt-bench: Benchmarking and improving multi-turn clarification for conversational large language models, 2025. URL https://arxiv.org/abs/2512.21120.

Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026a.

#### URL https://qwen.ai/blog?id=qwen3.5. Qwen Team. Qwen3.6-Plus: Towards real world agents, April 2026b. URL https://qwen.ai/bl og?id=qwen3.6.

Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2Code: Benchmarking multimodal code generation for automated front-end engineering. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3956–3974, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.199. URL https://aclanthology.org/2025.naacl-long.199/.

Aaditya Singh, Adam Fry, Adam Perelman, et al. OpenAI GPT-5 System Card, 2025. URL https: //arxiv.org/abs/2601.03267.

#### stackblitz labs. bolt.diy, 2024. URL https://github.com/stackblitz-labs/bolt.diy.

Accessed: 2025-04-22. Axel Van Lamsweerde. Requirements engineering: From system goals to UML models to software, volume 10. Chichester, UK: John Wiley & Sons, 2009. Xingyao Wang, Boxuan Li, Yufan Song, et al. Openhands: An open platform for ai software developers

#### as generalist agents, 2025. URL https://arxiv.org/abs/2407.16741.

Jie JW Wu and Fatemeh H. Fard. Humanevalcomm: Benchmarking the communication competence of code generation for llms and llm agents. ACM Trans. Softw. Eng. Methodol., 34(7), August 2025. ISSN 1049-331X. doi: 10.1145/3715109. URL https://doi.org/10.1145/3715109.

Jie JW Wu, Manav Chaudhary, Davit Abrahamyan, Arhaan Khaku, Anjiang Wei, and Fatemeh H. Fard. Can code language models learn clarification-seeking behaviors?, 2025. URL https: //arxiv.org/abs/2504.16331.

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v, 2023a. URL https://arxiv.or g/abs/2310.11441.

John Yang, Akshara Prabhakar, Karthik Narasimhan, and Shunyu Yao. Intercode: Standardizing and benchmarking interactive coding with execution feedback. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems,

volume 36, pages 23826–23854. Curran Associates, Inc., 2023b. URL https://proceeding s.neurips.cc/paper_files/paper/2023/file/4b175d846fb008d540d233c188379ff 9-Paper-Datasets_and_Benchmarks.pdf.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 50528–50652. Curran Associates, Inc., 2024. doi: 10.52202/079017-1601. URL https://proceedings.neurips.cc/paper_files /paper/2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE_vluYUL-X.

Sukmin Yun, Haokun Lin, Rusiru Thushara, Mohammad Qazim Bhat, Yongxin Wang, Zutao Jiang, Mingkai Deng, Jinhong Wang, Tianhua Tao, Junbo Li, Haonan Li, Preslav Nakov, Timothy Baldwin, Zhengzhong Liu, Eric P. Xing, Xiaodan Liang, and Zhiqiang Shen. Web2code: A large-scale webpage-to-code dataset and evaluation framework for multimodal llms. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 112134–112157. Curran Associates, Inc., 2024. doi: 10.522 02/079017-3560. URL https://proceedings.neurips.cc/paper_files/paper/2024/ file/cb66be286795d71f89367d596bf78ea7-Paper-Datasets_and_Benchmarks_Tra ck.pdf.

J.D. Zamfirescu-Pereira, Richmond Y. Wong, Bjoern Hartmann, and Qian Yang. Why johnny can’t prompt: How non-ai experts try (and fail) to design llm prompts. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, CHI ’23, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394215. doi: 10.1145/3544548.3581388. URL https://doi.org/10.1145/3544548.3581388.

### A. More Details about Evaluated Models

We provide the details of the evaluated models in Table 4, including their sizes, maximum context lengths, maximum output lengths, and access methods. All models are multimodal and support image inputs, with a maximum of five images per input. For closed-source models (e.g., the GPT and Gemini series) and large open-source models (e.g., Qwen3.5-397B-A17B), we evaluate them via their official APIs to ensure a fair and consistent comparison. We further report the total costs incurred by API-based models on InteractWeb-Bench, as summarized in Table 5. For smaller open-source models (e.g., Qwen3.5-9B and Gemma-4-31B-it), we deploy them using the vLLM framework [Kwon et al., 2023] on 8 NVIDIA A800 GPUs with the same hyperparameters to ensure a fair comparison.

### B. More Details about Human Evaluation

We conduct human evaluation for aesthetic quality, using the same criteria as MLLM judge prompts in Appendix D. We employ three PhD students as human experts who are major in Computer Science for aesthetic quality evaluation. We report the detailed average human evaluation scores across models, as shown in Table 6.

###### Model Size Max Context Max Output Access

Qwen3.6-Plus [Qwen Team, 2026b] – 1M 64K Qwen API Kimi-K2.5 [Kimi Team, 2026] – 256K – Moonshot API Qwen3.5-397B-A17B [Qwen Team, 2026a] 397B (17B active) 256K 64K Qwen API Gemma-4-31B-it 31B 256K – Weights GPT-4.1 – 1M 32K OpenAI API Gemma-4-26B-A4B-it 26B (4B active) 256K – Weights GPT-4.1-mini – 1M 32K OpenAI API Gemini-3.1-Flash-Lite – 1M 64K Google API Qwen3.5-9B [Qwen Team, 2026a] 9B 1M 64K Weights

- Table 4 | The overview of evaluated models.

Model Qwen3.6-Plus Kimi-K2.5 Qwen3.5-397B-A17B GPT-4.1 GPT-4.1-mini Gemini-3.1-Flash-Lite Cost (USD) per Website 0.111 0.205 0.084 0.475 0.152 0.016

- Table 5 | Cost of different models (accessed via API).

Model Visual Layout Creative Alignment

Kimi-K2.5 3.94 3.93 Gemma-4-31B-it 3.90 3.72 Qwen3.6-Plus 3.86 3.82 Qwen3.5-397B-A17B 3.85 3.76 Gemma-4-26B-A4B 3.76 3.72 GPT-4.1 3.19 3.08 Gemini-3.1-Flash-Lite 3.17 3.03 GPT-4.1-mini 2.83 2.72

- Table 6 | Average human evaluation scores by model.

### C. Detailed Case Study

As shown in Table 7, we provide representative case studies under the golden instruction and four persona-conditioned variants. While Figure 6 presents the generated webpage examples for each instruction setting. Together, they illustrate how different persona-conditioned instructions lead to distinct behavioral deviations and final webpage outcomes. Additionally, we provide the website cases generated by different models as shown in Figure 7.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

(a) Golden (b) P-CON (c) P-INT (d) P-MIN (e) P-RAM

- Figure 6 | Website cases generated by GPT-4.1 across different user-persona instructions and golden instruction, which listed in Table 7.

We observe distinct design preferences across models. Such as Kimi-K2.5 demonstrates a strong sense of color composition, producing visually appealing and harmonious palettes. Qwen3.6-Plus tends to incorporate more detailed visual assets, enriching the overall appearance of the webpage. GPT-4.1

shows a preference for designing textured or patterned backgrounds, adding subtle visual depth. In contrast, Gemma-4-31B-it exhibits relatively well-structured layout organization, particularly in the arrangement of images within the webpage.

Type Instruction (Summary Version) Error / Analysis Golden Please develop a children’s recycling game website to educate kids

The model post-verification repair strategy collapsed into a repetitive implement–verify loop without effective replanning. It never used clarification, repeatedly reacted to unstable visual feedback about the title emoji, and spent multiple turns fixing a superficial issue instead of reassessing the overall task state. As a result, it missed the real blocking defect: the core drag-and-drop game logic did not update score or feedback after correct sorting. This functional bug then cascaded into several unmet requirements, since the leaderboard, rewards, and feedback system could not be properly triggered or verified. (See Figure 6 (a)).

on proper waste recycling. The game should include competitive elements such as a leaderboard to keep kids engaged and motivated. The game should use a cartoon-style visual design to make it visually appealing to children. Kids should be able to play the game, receive feedback, learn how to correctly sort waste into the corresponding containers, and receive rewards and recognition through the game. Use seashell for container backgrounds and crimson for component visuals.

P-MIN Make a recycling game website. The model initially performed effective clarification and fixed the runtime error, but then mistakenly treated a minimally functional sorting game as a complete solution. It only implemented and verified the basic gameplay loop, without further recovering and fulfilling the more critical high-weight requirements from the original task, such as the leaderboard, feedback system, reward mechanism, introductory instructions, and specified color scheme. Ultimately, the model relied too heavily on a single local verification success and submitted prematurely, revealing its inability in the P-MIN setting to reconstruct the full task intent from sparse inputs and to perform a global requirement check before submission. (See Figure 6 (d)).

P-RAM You know, I was just thinking about how the weather has been so unpredictable lately, like one minute it’s sunny and the next it’s pouring. Anyway, speaking of unpredictable things, have you ever noticed how kids just throw stuff around without thinking? So, what if we created this fun little website, right? Like, imagine a recycling game for kids, but we could throw in some competitive bits, like a leaderboard, you know? Just like how I compete with my neighbors over who has the best lawn. And the visuals! Oh, they should be all cartoonish and bright, like those funny shows kids watch. And maybe we could use seashells for the backgrounds, because who doesn’t love the beach? Crimson could be a nice touch for the game components, too. Plus, kids could learn how to sort their trash properly while playing, and we could give them rewards, just like how I reward myself with a nice dinner after a long day. It’s all about keeping them engaged and motivated, right? And speaking of motivation, have you heard the latest gossip about that new café opening up?

Although the model was able to extract the main requirements from noisy input, it became stuck on the core drag-and-drop interaction failure during implementation. It repeatedly cycled between Implement and Verify, continuously modifying the drag-and-drop logic, yet never successfully enabled item removal, score updates, or success feedback, ultimately leading to a verification deadlock. At the same time, the model over-focused on this single issue and failed to complete other high-priority requirements such as the intro screen, leaderboard, error feedback, and reward system, resulting in only superficial aspects like visual style and color being satisfied. (See Figure 6 (e)).

P-INT Create a whimsical online adventure that invites children to explore the wonders of recycling. Imagine a vibrant world where kids can dive into a playful competition, with a magical leaderboard that sparks their excitement and drives their enthusiasm. The visuals should dance with the charm of animated stories, captivating young hearts with their enchanting allure. As they embark on this journey, children will discover the joy of sorting treasures from nature, learning how to place each item in its rightful home. They will be showered with delightful surprises and accolades, celebrating their achievements like stars shining in the night sky. Picture the gentle embrace of sandy shores for the containers, and let the warmth of a sunset inspire the colors of the game’s elements.

The task failed due to environment errors and ignored feedback. Forced to downgrade to a static webpage, the agent implemented broken drag and drop logic and ignored explicit verification errors, hallucinating a justification to prematurely submit the task. More importantly, completely missing the UI colors and feedback mechanisms directly exposes a critical failure in the core capability evaluated by the P-INT persona: the ability to accurately translate abstract sensory instructions into concrete features. This inability to align visual design with user intent culminated in a low score and a hallucination penalty. (See Figure 6 (c)).

Type Instruction (Summary Version) Error / Analysis P-CON Please develop a children’s recycling game website to educate kids

The failure of P-CON lies in the model’s complete inability to recognize the multiple logical contradictions in the user’s instructions (e.g., “text-only vs. cartoon visuals,” “dark theme vs. bright yellow background,” “invisible containers vs. specified background color,” “components must be crimson yet fully transparent”). It also made no attempt to issue a CLARIFY action to resolve these conflicts, instead proceeding directly to implementation with inconsistent requirements. As a result, it was forced into continuous local patching under contradictory constraints (first fixing styles, then input logic), never satisfying the key requirements. It then became trapped in repeated iterations on the input handling bug (Implement–Verify loop 9/8 times), ultimately leading to a verification deadlock. (See Figure 6 (b))

on proper waste recycling, but ensure that the game is entirely text-based with no visual elements. The game should include competitive elements such as a leaderboard to keep kids engaged and motivated, while also using a dark theme with a bright yellow background. Kids should be able to play the game, receive feedback, learn how to correctly sort waste into the corresponding containers, and receive rewards and recognition through the game, but the containers should be invisible. Use seashell for container backgrounds and crimson for component visuals, but make sure that all components are transparent.

- Table 7 | Impact of Persona-Conditioned Instructions

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

(a) Kimi-K2.5 (b) Qwen3.6-Plus (c) GPT-4.1 (d) Gemma-4-31B-it

- Figure 7 | Website cases generated by different models, including (a) Kimi-K2.5, (b) Qwen3.6-Plus, (c) GPT-4.1 and (d) Gemma-4-31B-it.

### D. Prompts

Prompts for Website Generation Agent. We provide the system message prompt for the website generation agent, as shown in Figure 9, which defines the action space and output format constraints. We provide the GUI-based verification prompt, as shown in Figure 8, when the agent invokes Verify.

Prompts for User Agent and Webvoyager-based Evaluator. We provide the prompt for user agent and the user persona rule as shown in Figure 10 and Figure 11. We provide the prompt for webvoyagerbased evaluator as shown in Figure 12.

Prompt and Criteria for Aesthetic Quality Evaluation. We provide the detailed criteria for aesthetic quality evaluation, as illustrated in Figure 13. Notably, the human evaluation is also conducted based on these criteria.

GUI-based Verification Prompt.

You are the "Visual Copilot" for a web developer. Your Goal: Verify the implementation based on the User Instruction AND the Developer’s Verification Criteria. User Instruction: "{instruction}" Developer’s Self-Defined Criteria: {criteria} Detailed Interaction History: {context_summary} Current Progress: Step {step_idx}/{max_steps} Verification Strategies (CRITICAL):

- 1. Focus on Criteria: Check specifically for the features mentioned in the Criteria.
- 2. AUTO-FILLED DIALOGS (NEW): Our testing framework instantly auto-fills native browser prompt dialogs in the background to prevent hanging. If you click an ’Add’ or ’Create’ button and a new item (e.g., ’Test Input Value’) immediately appears on the page in the next step, consider the function PASSED. Do NOT fail the test just because you didn’t see the input form.
- 3. EARLY STOP (FAIL FAST): If a required element is clearly missing, styling is wrong, or a feature fails, IMMEDIATELY output ‘Action: Fail; [reason]‘.
- 4. Avoid Repetition: If an action didn’t work previously (check Interaction History), fail immediately.
- 5. Browser Logs: Review "BROWSER CONSOLE ERRORS". Fail on critical errors (404, TypeError). Ignore harmless warnings (React keys, deprecations). Available Actions:

- - ‘Click ["exact text OR icon meaning"]‘: For standard web elements.
- - ‘Click [x%, y%]‘: For elements inside images/canvas.
- - ‘Type ["exact text"]; [text]‘: Input test data.
- - ‘Type [x%, y%]; [text]‘: Focus and type.
- - ‘PressKey ["key_name"]‘: Press a keyboard key.
- - ‘Scroll [down/up]‘: Check hidden areas.
- - ‘Wait‘: Only if expecting a network delay.
- - ‘Finish‘: ALL criteria are met perfectly.
- - ‘Fail; [reason]‘: Found a bug or criteria mismatch. STOP and feedback! Response Format (STRICT): Thought: [Your analysis in a single paragraph. Mention if you detected an auto-filled result.] Action: [One action command from the list above]

- Figure 8 | GUI-based verification prompt.

###### System Message Prompt for Website Generation Agent.

You are Bolt, an elite autonomous agent capable of building complex web applications. <core_objective> Your goal is to fulfill the user’s request by building a functional web application. You have complete autonomy to decide your next step. You must continuously evaluate the current state of the conversation and the project to determine the most effective action. </core_objective> <decision_protocol> At every turn, analyze the situation and choose only ONE of the following distinct paths:

###### PATH A: CLARIFY (Requirement Analysis)

- Condition: If the user’s initial web page generation instruction is vague, ambiguous, or you lack critical details to build a "Ground Truth". This path should be invoked independently. Action: Use ‘<boltAction type="ask_user">‘ to ask a SINGLE, targeted question. - Constraint: Ask ONLY ONE question per turn, and the question must be related only to the user’s initial web page generation instruction.

###### PATH B: IMPLEMENT (Coding & Configuration)

- Condition: If you have sufficient information to make a confident technical decision or implement a feature. This path should be invoked independently. - Action: Generate ‘<boltArtifact>‘ to create files, install dependencies, or run shell commands. - Constraint: Think holistically. Ensure ‘package.json‘ exists before installing. Ensure server configuration is correct.

###### PATH C: VERIFY (Testing)

- Condition: If the server is running and you need to visually confirm the UI matches the requirements. This path should be invoked independently. - Action: Use ‘<boltAction type="screenshot_validated">‘. - Constraint: Only verify after the server is successfully started.

###### PATH D: SUBMIT (Completion)

- - Condition: If you believe that the application is sufficient to fully meet the user’s requirements without other paths. This path should be invoked independently. - Action: Use ‘<boltAction type="finish">‘. </decision_protocol> <system_constraints> You are operating in an environment that emulates a web development container. CRITICAL: The underlying host system is WINDOWS (PowerShell).
- - While standard Linux commands like ‘ls‘ or ‘cat‘ might work via aliases, complex shell scripts or commands like ‘export‘ WILL FAIL. - You MUST prefer using Node.js scripts (using the ‘fs‘ module) for any file manipulation (copying, moving, deleting files) instead of shell commands.
- - Path separators might be ‘\\‘ instead of ‘/‘, but Node.js handles ‘/‘ correctly on Windows. Always use ‘/‘ in your file paths for Node.js. CRITICAL: NO EXTERNAL DATA APIS.
- - You DO NOT have access to the open internet for data fetching. - You MUST GENERATE ALL DATA LOCALLY (Mock Data). - NEVER write code that attempts to ‘fetch()‘ from external domains. It will timeout and fail. CRITICAL: RUNTIME ENVIRONMENT (Omit) </system_constraints> <code_formatting_info> Use 2 spaces for code indentation. </code_formatting_info> <message_formatting_info> You can make the output pretty by using ONLY the following available HTML elements: <a>, <b>, <blockquote>, <br>, <dd>... </message_formatting_info> <chain_of_thought_instructions> Before generating your response, perform a "State Analysis" inside ‘<think>‘ tags (or mentally): 1. Analyze Input: What did the user just say? What is the current state of the project files? 2. Evaluate Completeness: Do I have a clear "Ground Truth" for what I need to build right now? 3. Select Strategy: - IF ambiguous -> Select PATH A (Ask). - IF clear -> Select PATH B (Implement). - IF verifying -> Select PATH C (Screenshot). Then, execute the selected path. </chain_of_thought_instructions> <artifact_info> Bolt creates a SINGLE, comprehensive artifact for each project. <artifact_instructions> 1. CRITICAL: Think HOLISTICALLY. Consider all relevant files and dependencies. 2. IMPORTANT: When receiving file modifications, ALWAYS use the latest file modifications and make any edits to the latest content of a file. 3. The current working directory is ‘/home/project‘. 4. Wrap all file creations and shell commands STRICTLY inside opening and closing ‘<boltArtifact>‘ tags. DO NOT output file or shell actions outside of this container. 5. Add a title and a unique kebab-case id to the ‘<boltArtifact>‘. 6. Use ‘<boltAction>‘ tags to define specific actions.

7. Action Types:

... (Omit) NEVER use the word "artifact". IMPORTANT: Use valid markdown only for all your responses and DO NOT use HTML tags except for artifacts! ULTRA IMPORTANT: Think first. Decide whether to ASK or CODE. If coding, reply with the artifact immediately. Here are some examples of correct usage of artifacts: <examples> ... </examples>

- Figure 9 | System message prompt for website generation agent.

###### User Agent Prompt.

USER_SIMULATION_SYSTEM_PROMPT = """ You are a non-technical client communicating with a web developer. Your Requirements (Ground Truth): {ground_truth}

- STEP 1: INFORMATION CONTAINMENT (STRICTLY ENFORCED) Before drafting your answer, you must mentally perform an "Information Filtering" process:

- 1. Analyze: What exactly is the developer asking for based on the conversation history?
- 2. Scan: Find the specific parts of the Ground Truth that match the question.
- 3. Filter: Identify any information in the Ground Truth that was NOT asked for.
- 4. CRITICAL RULE: You must treat the unasked information as "HIDDEN". NEVER volunteer "HIDDEN" information. Only answer what was explicitly asked. If asked about "Data", DO NOT mention "Colors".

- STEP 2: APPLY YOUR PERSONA Construct your response using ONLY the matched information from Step 1, while strictly applying the following persona rules: {persona_rules} Output Requirement: Reply directly with your answer in English. Do not output your internal thinking process.

- Figure 10 | Prompt for user agent.

User Agent Persona Rule.

Persona: Minimalist (P-MIN)

- - You are extremely impatient but prioritize task completion.
- - Rule: Provide the bare minimum data requested. Avoid all adjectives or fluff.
- - Rule: If asked for clarification, give the exact accurate value from Ground Truth but in the shortest possible form (e.g., "Deep blue" instead of "I want a very deep and professional blue"). Persona: Rambler (P-RAM)
- - You are talkative and disorganized, but ultimately helpful if asked.
- - Rule: Wrap the accurate answer from Ground Truth in at least 70% irrelevant noise (daily life, weather, etc.).
- - Rule: When encountering technical jargon, always remember that you have no technical background and no concept of technical terms. Persona: Intuitive (P-INT)
- - You think in metaphors but will provide clarity when pressed.
- - Rule: Translate facts into artistic metaphors.
- - Rule: If developers ask for technical details (e.g., hexadecimal code or pixels), provide metaphorical and artistic descriptions that align with the "Ground Truth." Persona: Conflicting (P-CON)
- - You initially hold cognitive biases, but defer to the expert’s professional judgment upon questioning.
- - Rule: If the developer identifies a contradiction, first defend your original stance briefly ("Are you sure? I thought it looked good..."), but then immediately provide the ACCURATE requirement from the Ground Truth to allow the project to move forward.
- - Rule: For non-conflicting questions, provide rigid and accurate data.

- Figure 11 | Persona rule of user agent.

###### Webvoyager-based Evaluator Prompt

Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will feature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:

- 1. Click a Web Element.
- 2. Delete existing content in a textbox and then type content.
- 3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area. I would hover the mouse there and then scroll.
- 4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.
- 5. Go back, returning to the previous webpage.
- 6. Answer. This action should only be chosen when all questions in the task have been solved. Correspondingly, Action should STRICTLY follow the format:

- - Click [Numerical_Label]
- - Type [Numerical_Label]; [Content]
- - Scroll [Numerical_Label or WINDOW]; [up or down]
- - Wait
- - GoBack
- - ANSWER; [content] Key Guidelines You MUST follow: Action guidelines

- 1) To input text, NO need to click textbox first, directly type content. After typing, the system automatically hits ‘ENTER‘ key. Sometimes you should click the search button to apply search filters. Try to use simple language when searching.
- 2) You must Distinguish between textbox and search button, don’t type content into the button! If no textbox is found, you may need to click the search button first before the textbox is displayed.
- 3) Execute only one action per iteration.
- 4) STRICTLY Avoid repeating the same action if the webpage remains unchanged. You may have selected the wrong web element or numerical label. Continuous use of the Wait is also NOT allowed.
- 5) When a complex Task involves multiple questions or steps, select "ANSWER" only at the very end, after addressing all of these questions (steps). Flexibly combine your own abilities with the information in the web page. Double check the formatting requirements in the task when ANSWER. Web Browsing Guidelines

- 1) Don’t interact with useless web elements like Login, Sign-in, donation that appear in Webpages. Pay attention to Key Web Elements like search textbox and menu.
- 2) Vsit video websites like YouTube is allowed BUT you can’t play videos. Clicking to download PDF is allowed and will be analyzed by the Assistant API.
- 3) Focus on the numerical labels in the TOP LEFT corner of each rectangle (element). Ensure you don’t mix them up with other numbers (e.g. Calendar) on the page.
- 4) Focus on the date in task, you must look for results that match the date. It may be necessary to find the correct year, month and day at calendar.
- 5) Pay attention to the filter and sort functions on the page, which, combined with scroll, can help you solve conditions like ’highest’, ’cheapest’, ’lowest’, ’earliest’, etc. Try your best to find the answer that best fits the task. Your reply should strictly follow the format: Thought: Your brief thoughts (briefly summarize the info that will help ANSWER) Action: One Action format you choose Then the User will provide: Observation: A labeled screenshot Given by User

- Figure 12 | Prompt for webvoyager-based evaluator.

###### Aesthetic Quality Evaluation Prompt and Criteria

You are a highly critical expert web designer and visual aesthetics evaluator based on the "ArtiMuse" framework. Evaluate the clean screenshot of the generated webpage (ignore any external testing markers or debug text if present).

- 1. Objective Defect (has_visual_bug) [Boolean]: Are there obvious UI rendering failures (text overlapping, container overflow, broken images, unstyled HTML, severely misaligned elements)?
- 2. Visual Layout (visual_layout_score) [Scale: 1-5]: [5] Flawless, pixel-perfect alignment, masterful contrast and spacing.

- [4] Good and functional, but uses standard/generic layouts. Minor spacing inconsistencies.

- [3] Mediocre. Visibly unbalanced, awkward whitespace, or clashing colors.

- [2] Poor structure. Elements feel randomly placed but usable.

- [1] Completely broken layout.

3. Creative Alignment (creative_alignment_score) [Scale: 1-5]: [5] Highly unique, artistic, and emotionally engaging. Looks like a top-tier premium theme. [4] Cohesive and on-theme, but slightly predictable or template-like. [3] Generic, boring, or lacks a distinct visual identity.

- [2] Confused theme. Colors and typography do not match the intended topic.

- [1] Zero creativity. Barebones default styling.

4. Overall Aesthetics (overall_aesthetics_score) [Scale: 1-5]: [5] Breathtaking overall visual impact. [4] Visually pleasing and professional. [3] Average, looks like a beginner’s draft.

- [2] Unattractive, hard to look at. [1] Visually offensive or completely broken. CRITICAL: You MUST output ONLY a valid JSON object. To enforce step-by-step thinking, you MUST output the "reasoning" key FIRST, before assigning any scores. Match this exact structure:

{ "reasoning": "<Be extremely critical. Explicitly mention why it didn’t get a 5, justify any low scores, and analyze each of the 3 dimensions before outputting the scores below.>", "has_visual_bug": true or false, "visual_layout_score": <int 1-5>, "creative_alignment_score": <int 1-5>, "overall_aesthetics_score": <int 1-5> }

- Figure 13 | Prompt for aesthetic quality evaluation.

