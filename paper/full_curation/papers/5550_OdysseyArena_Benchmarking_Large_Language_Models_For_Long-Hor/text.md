# arXiv:2602.05843v2[cs.CL]4Jun2026

## ODYSSEYARENA: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions

Hang Yan1∗ Fangzhi Xu1∗ Qiushi Sun2∗ Jinyang Wu3 Zixian Huang4

Muye Huang1 Jingyang Gong2 Zichen Ding4 Kanzhi Cheng5 Yian Wang6

Xinyu Che1 Zeyi Sun4 Jian Zhang1 Zhangyue Yin7 Haoran Luo8 Ben Kao2

Qika Lin6†

### Abstract

The advancement of Large Language Models (LLMs) has catalyzed the development of autonomous agents capable of navigating complex environments. However, existing evaluations primarily adopt a deductive paradigm, where agents execute tasks based on explicitly provided rules and static goals, often within limited planning horizons. Crucially, this neglects the inductive necessity for agents to discover latent transition laws from experience autonomously, which is the cornerstone for enabling agentic foresight and sustaining strategic coherence. To bridge this gap, we introduce ODYSSEYARENA, which re-centers agent evaluation on long-horizon, active, and inductive interactions. We formalize and instantiate four primitives, translating abstract transition dynamics into concrete interactive environments. Building upon this, we establish ODYSSEYARENA-LITE for standardized benchmarking, providing a set of 120 tasks to measure an agent’s inductive efficiency and long-horizon discovery. Pushing further, we introduce ODYSSEYARENACHALLENGE to stress-test agent stability across extreme interaction horizons (e.g., > 200 steps). Extensive experiments on 15+ leading LLMs reveal that even frontier models exhibit a deficiency in inductive scenarios, identifying a critical bottleneck in the pursuit of autonomous discovery in complex environments.

### 1 Introduction

The emergence of Large Language Models (LLMs; [8, 2]) has sparked unprecedented interest in autonomous agents that can perceive environments, make decisions, and take actions to accomplish complex tasks. These AI agents are increasingly deployed across diverse domains—from robotics [29] and game playing [1] to scientific discovery [26] and business automation [35]. As the capabilities of LLMs expand, so does the demand for evaluation benchmarks that can reliably assess agent performance in realistic, dynamic settings.

While recent benchmarks have evolved from static question answering toward interactive decisionmaking [32, 28, 5], they predominantly assess a deductive mode of intelligence where agents rely on

∗Equal contribution. †Correspondence at linqika@nus.edu.sg

1Xi’an Jiaotong University, 2The University of Hong Kong, 3Tsinghua University, 4Shanghai AI Laboratory, 5Nanjing University, 6National University of Singapore, 7Fudan University,8Nanyang Technological University

Preprint.

specified success criteria, whereas the realistic deployment requires the agents to actively probe, react to feedback, and iteratively adjust their actions through self-improvement [36]. Finally, inductive reasoning from interaction represents a critical evaluation gap in current benchmarks, as most protocols assess deductive compliance to provided explicit instructions rather than the capacity to infer latent rules and transition dynamics from real-time interactive experience.

(a) Deductive (b) Inductive

[Figure 1]

[Figure 2]

[Figure 3]

Induced

[Figure 4]

Env.

[Figure 5]

Rules

[Figure 6]

Rules

Act.

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Reason

Act.

[Figure 12]

[Figure 13]

[Figure 14]

?

Obs.

Obs.

Figure 1: Comparison between deductive and inductive settings in multi-turn agentic tasks.

extensive prior knowledge to complete tasks. As illustrated in Figure 1, such evaluation overlooks the essential inductive mode, in which agents are required to actively explore and induce the hidden rules underlying the environment. This omission restricts the evaluation of an agent’s proficiency in complex environments where rules are not pre-specified.

We identify three critical capability dimensions that remain largely unaddressed in the current landscape. First, extremely long-horizon interaction is often neglected by existing widely used agent benchmarks that restrict episodes to fewer than 50 steps, which fails to capture the core strategic coherence and error accumulation challenges inherent in sequences of thousands of steps. Second, active exploration and trial-and-error are frequently bypassed by environments that provide a fully

In order to bridge this gap, we introduce ODYSSEYARENA, a suite of interactive environments that re-centers evaluation on long-horizon, active, and inductive reasoning, which entails inferring latent transition laws from empirical interactions. We formalize environments as generative state transition functions: (st+1,rt) = T (st,at), where st ∈ S is the latent state and at ∈ A is the agent action. The transition function T implicitly encodes the environment’s rules and regularities, which agents must actively induce from interaction in order to anticipate outcomes, plan effectively, and optimize behavior over long horizons. To systematically study an agent’s inductive capacity, we decompose T into a taxonomy of four representative structural primitives: discrete latent rules, continuous stochastic dynamics, multi-objective periodic patterns, and relational dependencies.

To facilitate empirical investigation into these abstract dynamics, we materialize these primitives into four diverse ODYSSEYARENA environments. These environments are curated to be computationally efficient and lightweight while remaining functionally representative of real-world systems, providing a tractable and scalable testbed for the community. Specifically, Turn On Lights grounds discrete Boolean logic in interdependent bulb configurations, while AI Trading presents continuous stochastic dynamics through multivariate stock-factor relationships. Energy Dispatch requires agents to uncover periodic efficiency patterns under multi-objective constraints, and Repo System engages agents in deducing the topological dependencies among software package versions in a virtual environment.

For practical evaluation, we provide ODYSSEYARENA-LITE as a standardized benchmark of 120 curated tasks. This suite serves as a representative of a scalable task distribution, optimized for high evaluation throughput while preserving the core challenges of active and inductive discovery. Each task maintains interaction horizons that are computationally tractable yet sufficiently non-trivial to necessitate long-horizon planning and the active induction of latent rules. In addition, we release ODYSSEYARENA-CHALLENGE, a stress-test suite with 1,000+ steps per task and 10 tasks per environment, intended to probe the limits of agent persistence, reasoning stability, and the ability to maintain coherent strategies over extremely long horizons. Together, these two settings balance accessibility, efficiency, and scalability, supporting both rapid iteration on current agents and rigorous evaluation of next-generation capabilities.

We evaluated over 15 trending LLMs, spanning proprietary models and open-source models across different scales. Overall, commercial models consistently outperform open-source alternatives, with Gemini 3 Pro Preview achieving the highest success rate of 44.17. Despite this, even the strongest commercial models broadly remain far below human-level performance across four environments in ODYSSEYARENA-LITE, highlighting substantial gaps in long-horizon reasoning, active exploration, and inductive generalization. Beyond these aggregate results, we conducted a detailed, fine-grained analysis of agent behavior across different environments and tasks, revealing patterns and failure modes that provide actionable insights for designing more capable and robust autonomous agents.

Our primary contributions are as follows:

- (1) Novel Perspectives For Agent Evaluation: We propose a novel evaluation paradigm centered on the capacity for autonomous discovery. This shift refocuses agentic intelligence on the induction of latent world dynamics through long-horizon and active interaction.
- (2) Reliable and Scalable Evaluation : We instantiate the inductive paradigm evaluation into ODYSSEYARENA, establishing ODYSSEYARENA-LITE as a standard suite for efficient evaluation.
- (3) Extensive Evaluations and Insights: Through an extensive evaluation of 15+ top-tier LLMs, we characterize the inductive bottleneck as a fundamental barrier to autonomous discovery while establishing rigorous benchmarking results for future research.

### 2 Related work

Interactive Benchmarks. Interactive benchmarks for LLM agents have evolved from grounded language understanding in simplified grid-worlds [23, 3] to sophisticated digital [7, 40, 32] and real-world systems [35, 39]. Despite this progress, a critical bottleneck remains in temporal depth: most environments favor short horizons or trajectories [21, 25]), which fails to capture the error accumulation phenomenon and the decay of long-term planning consistency [17]. Furthermore, many existing protocols [30, 20] bypass exploratory requirements by providing gold instructions or detailed API docs. ODYSSEYARENA and derived benchmarks bridge these gaps by introducing long-horizon tasks that demand coherent internal states and robust recovery strategies over extended interaction sequences.

Table 1: Comparison of representative multi-turn agentic benchmarks. Ind. indicates whether inductive reasoning is required. Horizon denotes the number of steps required to complete a task, categorized as short (<50), long (50–100), or XLong (>100). Deploy describes the evaluation setup required to run the environment, with APIbased deployment being the most lightweight.

Benchmark Ind. Horizon Deploy BabyAI [2019] ✓ Long Simulator ALFWorld [2021] ✓ Short Simulator GAIA [2023] ✗ Long Offline WebArena [2024] ✗ Short Docker OSWorld [2024] ✗ Long Docker AndroidWorld [2025] ✗ Short Emulator BrowseComp [2025] ✗ Long API ODYSSEYARENA ✓ X-Long API

Inductive Reasoning. Current agentic frameworks, such as ReAct [38] and Reflexion [22], primarily rely on deductive reasoning or test-time interactions [24, 33, 34] to apply internal knowledge or provided rules. However, the nature of intelligence necessitates inductive reasoning to infer latent rules and transition dynamics from raw observations [13]. While static benchmarks like ARC [4] and Zebra-Logic [15] evaluate rule synthesis, they remain passive and fail to capture the active discovery loop essential for autonomous agents [15]. While interactive environments like Mars [27] facilitate exploratory interaction, they struggle to decouple pure induction from pre-trained knowledge priors. In contrast with typical agent benchmarks shown in Table 1, this work necessitates inducing latent world structures through extremely long-horizon interactions, aligning with the concept of world models [11] while posing higher-order symbolic challenges for future agentic intelligence.

### 3 ODYSSEYARENA

#### 3.1 Preliminaries

Interactive environments can be characterized as generative processes where the environment’s response to an action at ∈ A at a latent state st ∈ S is governed by a transition function T :

(st+1,rt) = T (st,at). (1)

In this framework, T encapsulates the unobservable regularities and constraints that dictate the system’s evolution. Unlike standard reinforcement learning paradigms that often focus on policy optimization within known or fixed MDPs, ODYSSEYARENA emphasizes latent structure induction. To achieve long-horizon planning and optimal decision-making, agents must autonomously discover the functional form of T through strategic interaction.

To systematically characterize environment dynamics, we decompose the landscape of environment dynamics into a taxonomy of four orthogonal structural primitives, representing the fundamental

###### OdysseyArena

###### Turn On Lights Repo System

###### Energy Dispatch

###### AI Trading

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

Stock 1 Stock 2 Stock 3

Capacity Demand Budget 130 530

“Welcome to Repo System”

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Today 0/10

|[Figure 32]<br><br>>>> repo tree|
|---|

[Figure 33]

[Figure 34]

|>>>>>>toggletoggle light1light1<br><br>[Figure 35]|
|---|

Thermal Wind Solar Battery

HOLD 0 HOLD 0 HOLD 0

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Info

[Figure 40]

[Figure 41]

CASH 1000 [Next Day News]

-- run.py,

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

100KW 25KW 15KW 0/10 KW

-- core

[Figure 51]

|>>>>>>{ ”sell”:toggle light1{ “S1”: 0,<br><br>[Figure 52]<br><br>“S2”: 0 ,<br>“S3”: 0 },<br><br><br>“buy”: { “S1”: 100,<br><br>“S2”: 70 ,<br>“S3”: 0 } }<br>|
|---|

Stability Carbon

-- smoke.py

Metrics

[Figure 53]

0.95 0.64

|>>>>>>toggletoggle light1light2<br><br>[Figure 54]|
|---|

-- check_io.py,

-- eval/run_evaluatoion.py

[Figure 55]

|[Figure 56]<br><br>>>> { “Thermal ”: 100, “Wind”: 20, “Solar”: 20, “Battery”: -10 }|
|---|

[Figure 57]

|[Figure 58]<br><br>>>> pip install pkg1==1.0|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

…

Demand Unsatisfied Last Day

[Figure 67]

|>>>>>> toggletogglelight1light1<br><br>[Figure 68]|
|---|

[Figure 69]

[Figure 70]

successfully install pkg1==1.0

[Figure 71]

Capacity Demand Budget 165 750

[Figure 72]

Stock 1 Stock 2 Stock 3

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

|[Figure 79]<br><br>>>> pip install pkg2==1.0|
|---|

Today 10/10

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Thermal Wind Solar Battery

HOLD 200 HOLD 150 HOLD 50

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Info

[Figure 94]

|>>>>>> toggletoggle light1light3<br><br>[Figure 95]|
|---|

pkg2==1.0 is conflict with pkg1==1.0

[Figure 96]

[Figure 97]

>>>>>>{”sell”:toggle{light1“S0”: 100, }

CASH 200 [Next Day News]

99.9KW 18.8KW 20.5KW 10/10KW

…

Stability Carbon

[Figure 98]

…

|>>>>>>{toggle”sell”:light1{ “S1”: 200,<br><br>[Figure 99]<br><br>“S2”: 150 ,<br>“S3”: 0 },<br><br><br>“buy”: { “S1”: 0,<br><br>“S2”: 0 ,<br>“S3”: 50 } }<br>|
|---|

Metrics

[Figure 100]

|>>>>>>toggletoggle light1light5<br><br>[Figure 101]|
|---|

0.91 0.69

|[Figure 102]<br><br>>>> python run.py|
|---|

[Figure 103]

[Figure 104]

|[Figure 105]<br><br>>>> { “Thermal ”: 100, “Wind”: 40,<br><br>“Solar”: 40, “Battery”: 5 }|
|---|

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

…

[Figure 113]

All required packages installed Packages are compatible

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

TASK SUCCESS Your Profit Ratio is: +12.88% Running successfully

TASK SUCCESS

- Figure 2: Demonstrations of four ODYSSEYARENA environments: Turn On Lights, AI Trading, Energy Dispatch, and Repo System. For clarity, we omit the task prompts here and present only the interaction trajectories. Full prompts are provided in Appendix C.

mathematics that lead to complex real-world systems [6, 14]. By isolating these irreducible structures, we define a comprehensive set of world-modeling challenges that an autonomous agent must navigate:

- • Discrete Symbolic Rules: The transition is governed by Boolean logic over N bits, where s ∈ {0,1}N. This requires the agent to perform symbolic hypothesis testing to uncover the latent causal dependencies and logical couplings.
- • Continuous Stochastic Dynamics: The system evolves through a continuous state space s ∈ Rd

according to st+1 = f(st,at) + ϵ, where T incorporates a latent functional signal f and noise ϵ. This necessitates statistical inference to disentangle underlying regularities from fluctuations.

- • Periodic Temporal Patterns: The transition function exhibits cyclic regularities defined by a period P, such that T (s,a,t) ≈ T (s,a,t + P). This necessitates identifying long-range temporal dependencies to optimize multi-objective trade-offs.
- • Relational Graph Structures: The environment is defined by a graph G = (V,E), where transitions involve non-local interactions between entities. Success requires relational reasoning over the topological constraints that govern global state changes.

This taxonomy ensures a comprehensive evaluation, as each structure induces a distinct cognitive requirement—ranging from logical deduction to relational abstraction—that is irreducible to the others. ODYSSEYARENA integrates these primitives into a diverse suite of environments designed to assess an agent’s fundamental capacity for world-structure induction, as illustrated in Figure 2. We will introduce the respective environments in detail as follows.

#### 3.2 Env I: Turn On Lights

Overview. This environment instantiates the discrete symbolic rules primitive by simulating a network of N interdependent lights. The agent aims to reach a target configuration s = 1, representing the state where all lights are illuminated, through a sequence of toggling actions. Dynamics are governed by latent Boolean couplings that remain fixed within one trial but vary across episodes, where an intervention on a single light may trigger a deterministic cascade of state changes across the network. Consequently, success requires active exploration to infer the underlying logic and the correct activation sequence.

Hidden Rules. State transitions are governed by latent discrete rules that define how each action influences the configuration of lights. For each episode, these rules are instantiated by randomly combining Boolean operators to create a unique logical network, while ensuring that the resulting dependencies are solvable. As a result, an action targeting a particular light may deterministically affect multiple lights through indirect toggling or conditional activation. The dependencies remain

[Figure 121]

- Step 1

[Figure 122]

Support Auto-Validation

- Step 2

Manual-crafted Configuration

[Figure 123]

LLM Agent

Working Memory

Input

Support Difficulty Control

[Figure 124]

[Figure 125]

Thought + Action

Obs.

[Figure 126]

Task Success

[Figure 127]

ifEnd

Task Generator

init() step() eval()

[Figure 128]

Task Failure

[Figure 129]

Setup

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Check True Action Obs.

states continue

Config Files

[Figure 136]

API-based Environment Interface

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

…

- Figure 3: Overview of the benchmark architecture, illustrating the environment configuration initialization (left) and the interaction loop between the LLM agent and the environment step logic (right).

fixed within an episode but vary across episodes, and are not directly observable to the agent. Consequently, the agent must infer the underlying logical structure through deliberate intervention and observation of state changes.

Action and Observation Spaces. At each step, the agent takes a discrete action to activate/toggle a single light. Although actions are defined over individual lights and remain fixed across episodes, their effects are not necessarily localized due to hidden logical couplings. After each interaction, the agent observes the on/off status of all N lights. No information about the underlying logical dependencies or transition rules is revealed. As a result, while the environment is fully observable with respect to the surface state, it is partially observable in terms of the latent transition dynamics induced by the hidden rule configuration.

Task Evaluation. An episode is considered successful if all lights are turned on before the interaction budget is exhausted. Performance is measured by the task success rate. Solving the task efficiently requires the agent to identify and exploit the latent logical rules, rather than relying on myopic or brute-force exploration.

#### 3.3 Env II: AI Trading

Overview. This environment instantiates the continuous stochastic dynamics primitive, where the agent manages a multi-asset portfolio to maximize cumulative returns rt. Market transitions are driven by latent factors obscured by stochastic fluctuations, with daily news serving as indirect hints for future price movements. Unlike reactive tasks, success here depends on the agent’s ability to induce the underlying market regularities, requiring it to disentangle the meaningful signal f from noise to execute optimal multi-step, long-horizon trading strategies.

Hidden Rules. Price transitions are governed by the function st+1 = Wzt + ϵ, where s ∈ Rd denotes asset returns. In this formulation, W represents a latent factor loading matrix that maps

a vector of unobserved market factors zt to the price changes of d assets. While zt and the noise ϵ fluctuate at each timestep, the structural relationships defined by W remain invariant within an episode but vary across tasks. Consequently, agents must treat the sequence of interactions as a noisy observation process to identify the functional form of W for effective long-horizon planning.

Action and Observation Spaces At each step t, the agent perceives an observation ot ∈ O comprising historical prices and news-derived indicators. Since ot serves as a noisy proxy for zt, the agent must utilize the information within ot to estimate the underlying market state. Based on the perceived ot, the agent issues a combinatorial action at ∈ A specifying buy and sell quantities across all assets. To enable portfolio reallocation within a single decision cycle, the environment enforces sequential execution, processing sell orders before buy orders, requiring the agent to jointly reason about asset selection, trade timing, and capital allocation under the transition logic T .

Task Evaluation. Performance is measured by cumulative return over the trading horizon, adjusted for transaction costs and risk constraints. Successful performance requires learning and exploiting the latent stochastic dynamics over time, as opposed to reactive or single-step trading strategies.

#### 3.4 Env III: Energy Dispatch

Overview. This environment instantiates periodic temporal patterns by modeling a long-horizon energy dispatch problem in a dynamic power grid. The agent acts as a dispatcher, allocating thermal, wind, solar, and battery resources to meet daily demand under budget constraints. Unlike deductive optimization tasks, the environment enforces strict constraints: repeated budget or demand violations trigger immediate termination, simulating irreversible system failure. Achieving a stable, low-carbon supply therefore requires anticipating latent efficiency cycles and planning over extended horizons to avoid systemic collapse.

Hidden Rules. The core challenge lies in the discrepancy between the agent’s planned dispatch and the actual power generation. The system evolves according to a transition logic where the realized output Preal is determined by the agent’s rated action at modulated by a latent efficiency vector Et, formally Preal ≈ at ⊙ Et. Here, Et represents the time-varying efficiency for each power source. Crucially, the efficiency factors for wind and solar are governed by distinct, unobserved periodic functions Et ≈ Et+T with unique periods T. Since these fluctuations are not directly observable, the agent must infer the underlying periodic structures from historical output gaps.

Action and Observation Space. At each timestep, the agent receives ot containing the current electricity demand Dt and operating budget Bt. Based on these requirements, the agent issues a continuous action at ∈ R4 specifying the rated output for each generation type and the net charge/discharge command for the battery. The battery introduces inter-temporal dependencies, allowing the agent to buffer energy across time steps. Crucially, the environment enforces a strict safety protocol: repeated violations of demand satisfaction or budget limits trigger an early termination, simulating an irreversible grid collapse.

Task Evaluation. Success is defined by a rigorous multi-objective criterion. First, the agent must ensure survival by maintaining system stability without triggering early termination over the full horizon H (e.g., 120 days). Second, upon completion, the aggregate performance must satisfy predefined thresholds for both Carbon Intensity (C < τc) and Grid Stability (S > τs).

#### 3.5 Env IV: Repo System

Overview. This environment models a realistic software repository management scenario, where the agent must configure a Python project to execute successfully. The agent interacts with a partially observable dependency ecosystem, in which resolving local failures may introduce new global inconsistencies. Success requires systematic diagnosis, relational reasoning over dependencies, and careful planning under non-monotonic side effects.

Hidden Rules. The transition logic T is defined by a latent dependency graph G = (V,E). Here, nodes V represent software packages with specific versions, and directed edges E represent compatibility constraints (e.g., Pkg A v1.0 requires Pkg B ≥ v2.0). The system state st ⊆ V represents the currently installed environment configuration. Crucially, the graph G is hidden from the agent. A transition st+1 = T (st,at) triggered by an installation command involves a rigorous resolution process: the system automatically installs required ancestors and uninstalls conflicting nodes to maintain local consistency, potentially altering parts of st not targeted by at (side effects).

Action and Observation Space. The agent perceives the environment through ot, consisting of terminal outputs, file structures, and execution logs. Since G is latent, ot serves as a sparse signal revealing only the graph “broken edges” (e.g., ImportError). Based on these error traces, the agent issues discrete symbolic actions at ∈ Ashell (e.g., pip install). Since actions act as high-level graph mutation operators, the agent must sequence them to navigate the combinatorial state space, deducing the topology of G to resolve conflicts without explicitly observing the full picture.

Task Evaluation. The task is evaluated by whether the agent achieves a globally consistent environment in which the full project executes successfully. Partial or intermediate fixes (e.g., running individual sub-scripts) receive no reward if the global entry point fails, as local improvements may conflict with overall correctness. This strictly binary protocol enforces global consistency, emphasizing long-horizon planning, relational reasoning over interdependent components, and robustness to delayed and indirect effects over myopic error patching.

- 4 ODYSSEYARENA-LITE and ODYSSEYARENA-CHALLENGE

Building upon the infinite task space provided by ODYSSEYARENA, we derive two distinct benchmarking protocols to serve as standardized instantiations designed to evaluate agent performance across different difficulties: We primarily introduce ODYSSEYARENA-LITE as a suite tailored for efficient and reproducible performance assessment. Additionally, we provide ODYSSEYARENACHALLENGE for stress-testing agent stability and inductive resilience over extreme interaction horizons. The benchmark construction process and sample details are described below.

- 4.1 Task configuration and curation

To derive verifiable benchmarks from ODYSSEYARENA, we formalize each task as a deterministic instance sampled from a bounded parameter distribution (see Appendix A for details). This transition from abstract primitives to concrete evaluation is achieved by decomposing each instance into its structural configuration and temporal trajectory.

Structural Configuration. The configuration initializes latent rules T that remain invariant throughout an episode. In Turn On Lights, this corresponds to stochastically generating a DAG with specified density and logical operator ratios; in AI Trading and Repo System, it instantiates linear coefficient matrices and package dependency graphs, respectively. To ensure solvability, constraint satisfaction algorithms are applied during curation to guarantee a valid solution path for every rule set.

Temporal Trajectory. To eliminate stochasticity, we pre-determine all time-varying factors in ODYSSEYARENA as fixed sequences within the task metadata. For environments with dynamic states such as AI Trading and Energy Dispatch, components including daily factor fluctuations, efficiency curves, and resource budgets are pre-calculated. This ensures that the environment’s evolution is not subject to runtime randomness, allowing for an identical and fair comparison across different agents and experimental trials.

Task Sampling Strategy. We sample tasks by defining valid ranges for key environment parameters, such as the number of entities or the depth of logical dependencies. This sampling process is specifically calibrated to modulate task difficulty across two protocols. ODYSSEYARENA-LITE occupies a tractable parameter set for efficient evaluation, while ODYSSEYARENA-CHALLENGE targets the empirical limits of these ranges to stress-test agent stability over extreme horizons. This approach enables a characterization of the inductive bottleneck across varying environmental challenges. Detailed human anno processes are in Appendix D.

- 4.2 Task statistics

Under the ODYSSEYARENA-LITE setting, we construct a fixed evaluation suite by sampling 30 tasks for each environment. The maximum allowed number of interaction steps is environment-specific, reflecting the inherent complexity of the underlying dynamics. Specifically, the step limits are set to 200 for TurnOnLights, and 120 for AI Trading, Energy Dispatch, and Repo Management, respectively. This configuration yields a balanced benchmark that supports reliable comparison across environments while remaining computationally efficient for online evaluation.

While ODYSSEYARENA-LITE serves as our primary evaluation setting and the basis for all main experiments in this paper, we additionally introduce ODYSSEYARENA-CHALLENGE as a more demanding variant designed for stress-testing advanced agents, which substantially extends the required reasoning horizon, with tasks often exceeding 1,000 interaction steps. This setting targets failure modes that do not manifest under standard budgets, such as long-term credit assignment breakdown and compounding planning errors. Due to its significantly higher computational cost, ODYSSEYARENA-CHALLENGE is not used in our main evaluations, but is provided as an optional benchmark for future research.

- 5 Experiments

#### 5.1 Experimental settings

We evaluate over 15 trending LLMs on ODYSSEYARENA-LITE, encompassing proprietary frontiers such as Gemini 3 Pro Preview, Gemini 2.5 Pro [8], GPT-5, and Grok 4 Fast, alongside open-source series including DeepSeek-V3.2 [16], gpt-oss-120b [19], Qwen3 series [37], Llama 3 series [10],

Table 2: Performance comparison on four environments. We provide three different reasoning effort of gpt-oss-120b. For AI Trading environment, we report the profit rate and Best@4 is calculated based on the highest profit of each task. For other three environments, we report the success rate. Colored Rows represent proprietary models. The best results are in bold.

Turn On Lights AI Trading Energy Dispatch Repo System

Model

Avg@4 Pass@4 Avg@4 Best@4 Avg@4 Pass@4 Avg@4 Pass@4 Human 81.67 100.00 +92.55% +197.23% 25.00 60.00 77.50 100.00

Gemini 3 Pro Preview 44.17 76.67 +67.71% +76.94% 30.00 36.67 65.83 80.00 GPT-5 28.33 40.00 +17.32% +20.47% 23.33 40.00 62.50 83.33 Gemini 2.5 Pro 29.17 50.00 +33.02% +40.12% 10.83 26.67 50.00 66.67 gpt-oss-120b (high) 27.50 40.00 +23.27% +27.47% 0.00 0.00 18.33 33.33 DeepSeek-V3.2 18.33 36.67 +8.62% +12.88% 0.00 0.00 48.33 76.67 Grok 4 Fast 14.17 40.00 +5.70% +11.52% 0.00 0.00 38.33 60.00 Qwen3-235B-A22B-Instruct 15.00 43.33 +11.26% +17.67% 0.00 0.00 15.83 36.67 Qwen3-30B-A3B-Instruct 11.67 26.67 +4.76% +8.94% 0.00 0.00 26.67 50.00 gpt-oss-120b (medium) 16.67 40.00 +3.21% +7.09% 0.00 0.00 2.50 6.67 GLM-4-32B-0414 14.17 33.33 +3.14% +7.24% 0.00 0.00 9.17 30.00 gpt-oss-120b (low) 7.50 13.33 +2.02% +5.70% 0.00 0.00 9.17 26.67 Llama 3.3 70B Instruct 6.67 16.67 +0.77% +2.01% 0.00 0.00 19.17 40.00 Qwen3-4B-Instruct 0.00 0.00 +1.67% +6.95% 0.00 0.00 13.33 26.67 Llama 3.1 8B Instruct 6.67 20.00 +0.55% +3.07% 0.00 0.00 0.00 0.00 GLM-4-9B-Chat 0.00 0.00 -0.18% +0.41% 0.00 0.00 0.00 0.00

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

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

[Figure 161]

[Figure 162]

[Figure 163]

- Figure 4: Success rate comparison of w/ and w/o rules in Turn On Lights. We select Llama 3.3 70B Instruct, GLM-4-32B-0414, Qwen3-235BA22B-Instruct, DeepSeek-V3.2, Grok 4 Fast, GPT-5, Gemini 3 Pro Preview for illustration.

Table 3: Failure mode of 47 failed trajectories from Gemini 3 Pro Preview, in which planning is not the bottleneck. Notably, each trajectory may contain multiple failure modes.

Failure Mode Count Percentage

Exploration Limitations 38 80.85% Memory Constrains 27 57.45% Behavior Stagnation 13 27.66% Planning Inability 2 4.26%

Total 47 100.00%

and GLM-4 series [9]. Each test case is executed four times to report both Avg.@4 and Pass@4 success rates. Notably, for Repo System, we use Best@4 to represent the best profit of 4 generated trajectories. To manage the extreme interaction horizons, our standard prompts retain only the history of actions and environment feedback, omitting intermediate reasoning traces from previous steps to ensure context efficiency. Additional details regarding model specifications, reasoning effort, and prompt templates are provided in Appendix C.1 and C.2.

#### 5.2 Main results.

The General Performance Gap. Table 2 illustrates a performance disparity between SOTA LLMs and human. Unlike human participants who successfully resolve tasks by isolating causal variables and distilling latent rules through instinctive trial-and-error, LLMs exhibit a marked deficiency in autonomous rule induction. This gap highlights a fundamental failure in current agents to internalize latent world dynamics from experience, leading to significant performance degradation as interaction horizons extend.

Proprietary Model and the Scaling Limit. Frontier proprietary models, led by Gemini 3 Pro Preview, consistently provides the current performance ceiling and substantially outperform other counterparts across most environments. Despite this advantage, the ubiquitous failure in Energy Dispatch underscores a critical architectural limitation shared across the spectrum: an inability to synthesize periodic patterns over extended observation windows (∼20 steps). This suggests that while increased scale enhances deductive compliance, it remains insufficient to overcome the inductive bottleneck for robust world-structure modeling.

Qwen3-235B-A22B-Instruct

Gemini 2.5 Pro DeepSeek-V3.2 GLM-4-9B-Chat

Grok 4 Fast

Llama 3.3 70B Instruct

GPT-5

Gemini 3 Pro Preview

random

0.4

0.60

SuccessRate

0.3

0.45

0.2

0.30

0.1

0.15

0.0

0.00

0 40 80 120 160 200

0 30 60 90 120

Step

Step

(b) Repo System

(a) Turn On Lights

- Figure 5: Success Rate against step in two environments. Most of the success rate curve saturate with extended interaction steps. More results are in Appendix B.4.

[Figure 164]

Infeasible Region

Infeasible Region

Figure 6: Model performance is significantly related to loop ratio. Infeasible region indicates that a high Loop Ratio results in an inability to solve long-horizon inductive reasoning tasks.

Deductive Proficiency vs. Inductive Deficiency. To disentangle the root of failure, we evaluate agents with explicit access to latent transition rules. Figure 4 shows that frontier models achieve near-perfect success when the underlying logic is provided, yet falter significantly without it. This contrast identifies a fundamental asymmetry: LLMs excel at deductive reasoning but lack the inductive capacity to autonomously synthesize environment mechanics from experience. These findings confirm that the primary bottleneck in ODYSSEYARENA is the discovery of world dynamics rather than the complexity of the task logic itself.

- 6 Analysis

In this section, we first introduce the frequent failure modes of SOTA LLMs in Section 6.1, followed by detailed analyzes of how these phenomena degrade model performance in Section 6.2, Section 6.3 and Appendix B.7. We include more analytical experiments, such as the difference between successful and unsuccessful trajectories in AppendixB.2, step usage to finish a task in Appendix B.10, token efficiency for task completion in Appendix B.11, comparison of in context learning in Appendix B.9 and the results of ODYSSEYARENA-CHALLENGE in Appendix B.13.

#### 6.1 Failure modes of current LLM agents

We manually analyze 47 failure trajectories from Gemini 3 Pro Preview, revealing four error categories in Table 3, which stem primarily from limitations in exploration, memory, and action adaptation.

The most prevalent issue is exploration limitation (38/47), where agents prematurely converge to local optima and default to suboptimal strategies without systematic hypothesis testing, which is detailed in Section 6.2. Closely related are memory constraints (27/47), where agents successfully gather relevant information but fail to retrieve or prioritize key evidence over long interaction histories, which is detailed in Appendix B.7. We also observe behavior stagnation (13/47), where agents repeatedly execute invalid actions despite explicit negative feedback, especially in discrete-action environments, which is detailed in Section 6.3. Notably, planning inability is rare (2/47), suggesting that forming high-level plans is not the primary bottleneck.

#### 6.2 Exploration limitation: performance saturation in long-horizon

As illustrated in Figure 5, the success rate curves reveal that the interaction budget beyond an initial exploratory phase yields negligible marginal gains for most models. This plateau suggests a fundamental inductive bottleneck in long-horizon scenarios, where extended interaction fails to rectify the absence of a coherent internal world model. Furthermore, weaker models frequently underperform the random baseline, underscoring an inherent inability to extract latent regularities from environmental feedback. These findings indicate that the primary barrier is not interaction volume but the underlying capacity for inductive discovery.

#### 6.3 Behavior stagnation: action loops and inductive stagnation

We reveal a prevalent failure mode characterized by persistent “action loops,” where models repeat invalid operations despite receiving negative environmental feedback. As shown in Figure 6, a higher loop ratio directly correlates with diminished success rates, identifying a critical inability to capture hidden rules during interaction. This repetitive behavior signifies inductive stagnation, as agents fail to synthesize latent world laws from unsuccessful trials to refine their long-term strategy. Consequently, these cycles highlight the failure to transform trial-and-error into active discovery, underscoring the gap between deductive compliance and inductive world-modeling. Details are in Appendix B.8.

### 7 Conclusion

This work introduces a paradigm shift in agentic evaluation by transitioning from deductive instructionfollowing to long-horizon, active and inductive modeling. We formalize abstract environment dynamics into four structural primitives and instantiate them via ODYSSEYARENA, and then establish a standardized framework for the reproducible evaluation. The observed low-performance plateau across multiple flagships reveals a fundamental inductive bottleneck that scaling alone cannot satisfy the necessity of moving beyond deductive compliance. Future research toward more agentic intelligence should prioritize architectures capable of distilling latent transition laws from raw experience, bridging the gap between passive rule-following and active discovery in complex, dynamic worlds.

### References

- [1] Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models. Nature Human Behaviour, 9(7):1380–1390, May 2025. ISSN 2397-3374. doi: 10.1038/s41562-025-02172-y. URL http://dx.doi.org/10.1038/s41562-025-02172-y.
- [2] Anthropic AI. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1:1, 2024. URL https://assets.anthropic.com/m/61e7d27f8c8f5919/original/ Claude-3-Model-Card.pdf.
- [3] Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. Babyai: A platform to study the sample efficiency of grounded language learning. In International Conference on Learning Representations, 2019. URL https://iclr.cc/virtual/2019/poster/733.
- [4] François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. URL https://arxiv.org/abs/1911.01547.
- [5] Andy Chung, Yichi Zhang, Kaixiang Lin, Aditya Rawal, Qiaozi Gao, and Joyce Chai. Evaluating long-context reasoning in llm-based webagents. In NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning, 2025. URL https:// openreview.net/forum?id=oxj422wRvO.
- [6] Andy Clark. The dynamical challenge. Cognitive science, 21(4):461–481, 1997. URL https://www.sciencedirect.com/science/article/abs/pii/S0364021399800305.
- [7] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023. URL https://openreview.net/forum?id= kiYqbO3wqw.
- [8] Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. URL https://arxiv.org/abs/2507.06261.
- [9] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024. URL https: //arxiv.org/abs/2406.12793.

- [10] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/ 2407.21783.
- [11] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. Advances in neural information processing systems, 31, 2018. URL https://proceedings.neurips.cc/paper_files/paper/2018/file/ 2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf.
- [12] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. URL https://dl.acm.org/doi/abs/10.1145/ 3600006.3613165.
- [13] Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and brain sciences, 40:e253, 2017. URL https://pubmed.ncbi.nlm.nih.gov/27881212/.
- [14] Jintang Li, Ruofan Wu, Xinzhou Jin, Boqun Ma, Liang Chen, and Zibin Zheng. State space models on temporal graphs: A first-principles study. Advances in Neural Information Processing Systems, 37:127030–127058, 2024. URL https://openreview.net/forum?id=UaJErAOssN.
- [15] Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. Zebralogic: On the scaling limits of llms for logical reasoning. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=sTAJ9QyA6l.
- [16] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025. URL https://arxiv.org/ abs/2512.02556.
- [17] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In ICLR, 2024. URL https://openreview.net/forum?id=zAdUB0aCTQ.
- [18] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations, 2023.
- [19] OpenAI. gpt-oss-120b & gpt-oss-20b model card. gpt-oss model card, 1:1, 2025. URL https://arxiv.org/abs/2508.10925.
- [20] Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/pdf?id=2GmDdhBdDk.
- [21] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William E Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Kenji Toyama, Robert James Berry, Divya Tyamagundlu, Timothy P Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=il5yUQsrjC.
- [22] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023. URL https://openreview.net/pdf?id= vAElhFcKW6.

- [23] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021. URL https: //openreview.net/pdf?id=0IOX0YcCdTn.
- [24] Qiushi Sun, Zhangyue Yin, Xiang Li, Zhiyong Wu, Xipeng Qiu, and Lingpeng Kong. Corex: Pushing the boundaries of complex reasoning through multi-model collaboration. arXiv preprint arXiv:2310.00280, 2023.
- [25] Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, et al. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5555–5579, 2025.
- [26] Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, et al. Scienceboard: Evaluating multimodal autonomous agents in realistic scientific workflows. arXiv preprint arXiv:2505.19897, 2025.
- [27] Xiaojuan Tang, Jiaqi Li, Yitao Liang, Song-Chun Zhu, Muhan Zhang, and Zilong Zheng. Mars: Situated inductive reasoning in an open-world environment. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=3qoQ6AolAz.
- [28] Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, Jeffrey Hui, Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, et al. Michelangelo: Long context evaluations beyond haystacks via latent structure queries. CoRR, 2024. URL https://openreview.net/forum?id=jdc57bqY3u.
- [29] Jiaqi Wang, Enze Shi, Huawen Hu, Chong Ma, Yiheng Liu, Xuhui Wang, Yincheng Yao, Xuan Liu, Bao Ge, and Shu Zhang. Large language models for robotics: Opportunities, challenges, and perspectives. Journal of Automation and Intelligence, 4(1):52–64, 2025. URL https://www.sciencedirect.com/science/article/pii/S2949855424000613.
- [30] Weixuan Wang, Dongge Han, Daniel Madrigal Diaz, Jin Xu, Victor Rühle, and Saravan Rajmohan. Odysseybench: Evaluating llm agents on long-horizon complex office application workflows. arXiv preprint arXiv:2508.09124, 2025. URL https://arxiv.org/abs/2508. 09124.
- [31] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025. URL https://arxiv.org/pdf/2504.12516.
- [32] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024. URL https://openreview.net/forum?id= tN61DTr4Ed#discussion.
- [33] Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Jun Liu, Qika Lin, and Zhiyong Wu. ϕdecoding: Adaptive foresight sampling for balanced inference-time exploration and exploitation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13214–13227, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.647. URL https://aclanthology.org/2025.acl-long.647/.
- [34] Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, and Zhiyong Wu. Genius: A generalizable and purely unsupervised self-training framework for advanced reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13153–13167, Vienna, Austria, July

2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/ 2025.acl-long.644. URL https://aclanthology.org/2025.acl-long.644/.

- [35] Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Zhiruo Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Melroy Maben, Raj Mehta, Wayne Chi, Lawrence Keunho Jang, Yiqing Xie, Shuyan Zhou, and Graham Neubig. Theagentcompany: Benchmarking LLM agents on consequential real world tasks. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/ forum?id=LZnKNApvhG.
- [36] Hang Yan, Xinyu Che, Fangzhi Xu, Qiushi Sun, Zichen Ding, Kanzhi Cheng, Jian Zhang, Tao Qin, Jun Liu, and Qika Lin. Tide: Trajectory-based diagnostic evaluation of test-time improvement in llm agents. arXiv preprint arXiv:2602.02196, 2025. URL https://arxiv. org/abs/2602.02196.
- [37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.
- [38] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2023. URL https://openreview.net/forum?id= WE_vluYUL-X.
- [39] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik R Narasimhan. {$\tau$}-bench: A benchmark for \underline{T}ool-\underline{A}gent-\underline{U}ser interaction in real-world domains. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=roNSXZpUDN.
- [40] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations,

2024. URL https://openreview.net/forum?id=oKn9c6ytLx.

### A Task curation

#### A.1 Turn On Lights curation

This task instantiates the discrete symbolic rules primitive by simulating a network of N interdependent lights. The agent aims to reach the target configuration s = 1 (where all lights are illuminated) from an initial state s0 = 0. The environment’s dynamics are governed by a transition function (st+1,rt) = T (st,at) that encapsulates unobservable regularities in the form of latent boolean logic. Success necessitates the systematic discovery of these hidden logical dependencies through strategic interaction and hypothesis testing.

#### A.1.1 Structural configuration

Dependency Structure Construction. Each light Li ∈ L = {L0,...,LN−1} is associated with an activation condition ϕi, a propositional formula over other light states. We define a strict partial order ≺ in L such that if Lj appears in ϕi, then Lj ≺ Li. This ensures that each light’s condition only references lights with smaller indices:

ϕi = f(Lj

,Lj

1

), where j1,...,jk < i. (2)

,...,Lj

2

k

The minimal element L0 is defined with a constant activation condition (ϕ0 ≡ True) and its role as the entry point of the causal chain is hidden. Specifically, we apply a random mapping σ : L → L. Consequently, the agent cannot rely on numerical preference or ordering of IDs to infer the dependency graph; instead, it must perform systematic interactions to inductively reason and subsequently bootstrap its knowledge of the transition function T .

Toggle Mechanism and State Transitions. Let s ∈ {0,1}N denote the state vector. When an agent performs an action at to toggle light Li, the transition function T updates the state as follows:

st+1,i = ¬st,i if ϕi(st) = True st,i if ϕi(st) = False

(3)

If ϕi(st) = False, the agent receives only a generic failure message. The presence of negation (¬) introduces non-monotonic dynamics: turning a light on may satisfy one condition while violating another, necessitating complex inductive reasoning to navigate the state space.

#### A.1.2 Resolvability and diversity

Guaranteed Resolvability. Every generated instance is verified to have at least one valid solution through an exhaustive search. An instance is accepted only if: (1) there exists a path to the goal state s = 1, and (2) the minimum steps required to finish the task is larger than a predefined threshold.

The partial order structure provides a constructive guarantee for latent structure induction: since L0 is always accessible and each subsequent ϕi depends only on its predecessors, a solution path always exists.

Task Diversity. To ensure a robust evaluation, we leverage the following parameters to generate a diverse suite of environment instances:

- • State Space Scaling (N): By modulating the number of lights N, we control the exponential growth of the state space s ∈ {0,1}N. This allows for the creation of tasks ranging from localized logic puzzles to complex systems with 2N possible configurations.
- • Logical Combination: By varying the density of the predecessors Lj ≺ Li and the specific mixture of {∧,∨,¬} operators, each instance presents unique transition regularities.

#### A.2 AI Trading curation

This task instantiates the continuous stochastic dynamics primitive by simulating a multi-stock management scenario. Agents must autonomously infer the latent dependency matrix W to maximize cumulative reward rt. The challenge lies in performing statistical inference to unravel the underlying market signal f from stochastic fluctuations ϵ over a fixed trading horizon.

#### A.2.1 Structural configuration

Dependency Matrix Construction. The relationship between latent market factors and asset returns is modeled via a transition matrix W ∈ Rd×K, where d denotes the number of stocks and K the number of unobserved market factors. At each time step t, the environment generates a factor change vector zt ∈ RK. The resulting stock returns st+1 are governed by:

st+1 = Wzt + ϵ, ϵ ∼ N(0,σ2) (4)

where s ∈ Rd represents the vector of price returns. The price of the stock i at time t + 1 is updated to pt+1,i = pt,i + st+1,i. The matrix W remains invariant within an episode to represent the unobservable regularities of the specific market task.

Trading Mechanism and State Transitions. The action at consists of a set of buy/sell operations. Specifically, the agent’s action at specifies the buy/sell and quantity for each stock at every step. The transition function T then updates the agent’s portfolio, which consists of available cash and a vector of stock holdings. The reward rt is determined by the sum of cash and the current market value of all held stocks. The agent must inductively reason about the hidden matrix W by observing how price changes correlate with market factors over time, enabling them to execute strategic, long-horizon trading decisions.

#### A.2.2 Temporal trajectory

To ensure reproducible and fair comparisons across different agents, the environment’s stochastic elements are pre-determined within the task metadata. The complete timeline of the changes of z and

ϵ is generated at the beginning of each task and stored in the configuration. This approach ensures that while the market factors fluctuate, the environment’s evolution remains deterministic for any given task instance. Consequently, the optimal trading strategy is theoretically computable with complete knowledge of the dependency matrix W and the pre-generated factor timeline, providing a consistent benchmark for evaluating agent performance.

#### A.2.3 Resolvability and diversity.

Guaranteed Resolvability. There is no failure in this environment, and every trajectory finishes each task with different profit. So all the tasks are resolvable.

Task Diversity. We generate diverse tasks by modulating the following parameters:

- • Dimensionality Scalability (d,K): By scaling the number of stocks d and market factors K, we control the complexity of the inference problem. A larger d expands the action space, while a higher K increases the difficulty of disentangling the latent factors from the observations.
- • Dependency Sparsity: The density of non-zero entries in W determines the complexity of the structural relationships. Sparse matrices require the agent to identify a few factor-stock couplings, whereas dense matrices test the agent’s ability to model global market correlations.
- • Signal-to-Noise Ratio: By adjusting the variance of ϵ, we modulate the information scarcity. This forces agents to distinguish between persistent structural signals defined by W and stochastic fluctuations.

#### A.3 Energy Dispatch curation

This task simulates an energy grid dispatch scenario where agents must allocate power generation resources to satisfy electricity demand while maintaining grid stability, budget constraints, and carbon emission targets. Given 4 generation sources (thermal, wind, solar, and battery) with respective capacities, the agent specifies daily rated power allocations at ∈ RK over a horizon H. The challenge lies in adapting to time-varying renewable efficiency while balancing multiple competing objectives.

#### A.3.1 Structural configuration

To evaluate the agent’s capacity for world-structure induction, we model the discrepancy between planned and realized power through a latent efficiency vector Et and real generated energy Preal ≈ at ⊙ Et.

Multi-Objective Constraints and State Transitions. Each day, the environment evaluates the following constraint satisfaction criteria:

- • Demand: Total realized supply Preal,t must meet demand Dt.
- • Budget: Total cost must not exceed budget Bt.
- • Carbon: The ratio of cumulative generated thermal energy

H τ=1 Pthermal,τ

H

τ=1(Pthermal,τ,Pwind,τ,Psolar,τ) must remain below target τc.

- • Stability: Grid stability penalizes large allocation changes between consecutive days. Additionally, a violation of either demand or budget could significantly influence the grid stability.

Consecutive violations (3 days in ODYSSEYARENA-LITE) of demand or budget constraints trigger early termination, simulating an irreversible grid collapse.

#### A.3.2 Temporal trajectory

For renewable sources wind and solar, Et is governed by a five-level hierarchical generative process that simulates multi-scale temporal dependencies:

Ewind/solar,t = Clip Ebase(t mod T) + δ⌊t/T⌋ + ϵt (5)

where T ∈ [15,25] denotes the hidden period length randomly sampled. The generative logic is structured as follows:

- • Base Pattern (Ebase): It is constructed as a piecewise-linear sequence where each segment (2–5 days) is assigned a random efficiency baseline to simulate the consistency of weather. To introduce intra-period complexity, we add stochastic spikes (with a probability of 5%) representing extreme weather.
- • Cyclic Variation (δ⌊t/T⌋): To prevent the agent from relying on the memorization of a fixed efficiency curve, each full cycle incorporates a unique random offset δ. This mimics the changes between different months or seasons, requiring the agent to continuously recalibrate its internal model.
- • Micro-Fluctuations (ϵt): A high-frequency Gaussian noise ϵt ∼ N(0,0.012) is added to the daily output, simulating real-world stochastic.
- • Value Clipping: Finally, the efficiency is clipped to domain-specific ranges ([0.6,1.05] for wind and [0.65,1.1] for solar) to ensure physical realism and prevent extreme outliers.

Additionally, the efficiency of thermal is near to 1 with minimum fluctuation and the battery efficiency is a constant 1 without any fluctuation. A positive action abattery,t > 0 denotes discharging stored energy to the grid, while abattery,t < 0 denotes charging from excess generation.

The efficiency curves are unobservable to the agent. Consequently, the agent must inductively reason for the stable periodic signal from transient fluctuations by comparing its historical rated actions at with the real power outputs Preal,t.

To ensure reproducibility and fair comparison, all time-varying factors (Dt,Bt,Et) are predetermined as fixed sequences within the task metadata. This ensures that the environment’s evolution is fully deterministic given the agent’s sequence of actions, allowing for identical experimental trials across different models.

#### A.3.3 Resolvability and diversity

Guaranteed Resolvability. Each instance is guaranteed to be feasible through careful parameter design. Total capacity exceeds peak demand, and budgets are set as Bt = 4.2×Dt to provide adequate financial space. Efficiency values and objective thresholds are tuned to ensure that a foresighted dispatch policy can successfully complete the horizon H without triggering early termination, while leaving substantial space for diverse strategies.

Task Diversity. Our generation framework provides fine-grained control over task difficulty through several orthogonal parameters:

- • Temporal Dynamics: Varying period lengths T for wind and solar creates complex interference patterns that the agent must disentangle.
- • Constraint Tightness: Adjusting targets τc and τs modulates the precision required to balance competing carbon and stability goals.

#### A.4 Repo System curation

This task instantiates the relational graph structures primitive by simulating a software repo dependency resolution scenario. The agent must discover a valid configuration of packages that satisfies all latent constraints in a dependency graph G = (V,E).

#### A.4.1 Structural configuration

Dependency Graph Construction. The environment is defined by a latent graph G = (V,E), where each node v ∈ V represents a specific version of a software package. Edges E represent directed compatibility constraints; for instance, an edge (vi,vj) may indicate that version vi of Package A requires version vj of Package B. To ensure a structured challenge, we generate a random topological ordering over packages to prevent circular dependencies, while allowing version-level constraints to create complex requirements.

To simulate ubiquitous dependencies on foundational libraries (e.g., NumPy, PyTorch), we designate 1-2 packages as base libraries at the root of the topological order. A proportion of other packages depend on these base libraries with varying version constraints. There are two types of constraints:

(1) the base library and the dependent library should be the same version. For example, the version of base library A is 1.1, then the version of dependent package B should be 1.1 as well. (2) the base library and the dependent library should be the same main version. For example, the version of base library A is 1.1, then the version of dependent package B should be 1.X.

Transition Logic and Side Effects. The state represents the set of currently installed package versions. When an agent issues an installation action at, the transition function st+1 = T (st,at) simulates a rigorous resolution process. Unlike simple state updates, T may induce side effects, such as automatically upgrading, downgrading, or uninstalling conflicting packages to satisfy the constraints in G. We implement four types of resolution behaviors: (1) Ensure: automatically install missing dependencies; (2) Force-high/low: coercing dependencies to extreme compatible versions; and (3) Pin: locking a package to a specific version. These behaviors introduce non-monotonic dynamics, where the sequence of actions a1,a2 may result in a different final state s than a2,a1, forcing the agent to reason about the order of interventions.

#### A.4.2 Resolvability and diversity

Guaranteed Resolvability. Every instance is verified to have at least one valid goal state through a solution-first generation strategy. We first sample a ground-truth configuration and then construct the edges E such that they provably contain this solution. All dependencies and constraints are then generated to provably include the ground-truth packages version.

- A.5 Action space for ODYSSEYARENA

For the four environments in ODYSSEYARENA: Turn On Lights, AI Trading, Energy Dispatch, and Repo System, we list detailed action space and their description in Table 4.

Notably, the agent can only execute one action in Turn On Lights and Repo System environments. For AI Trading environments, the agent should first sell and then buy stocks within one step. For Energy Dispatch environment, the agent can plan for thermal, wind, solar, and battery together. However, charging and discharging cannot be executed simultaneously in one step.

- A.6 License We agree to release our datasets under CC-BY 4.0 license.
- A.7 Limitation

To ensure reproducibility and fair comparison, all stochastic elements in each environment are pre-determined as fixed sequences within the task metadata. While this guarantees deterministic evaluation across agents, it removes the non-stationary dynamics inherent in real interactive systems, where environment responses may evolve or co-adapt with the agent over time.

### B Details of analysis

#### B.1 Detailed comparison between w/ and w/o rules

To additionally demonstrate the performance between w/ and w/o rules for SOTA models, we provide detailed results in Table 5. From the extended results, we argue the same conclusion that SOTA models are strong deductive reasoners rather than good inductive reasoners.

#### B.2 Comparison between successful and unsuccessful trajectories

We manually analyze 16 Turn On Lights problems where Gemini 3 Pro Preview exhibit partial success rather than absolute success or failure. We find that only two problems are due to random guesses. Moreover, this intra-problem variance mainly stems from exploration limitations (unstable exploration). For example, if activating light3 requires light2 (on) and light1 (off), and the agent

- Table 4: Action Space of and description for ODYSSEYARENA. All parameters are wrapped in <>. Environment Action Description

The parameter light is the specific index of the light you want to toggle. This action is successfully executed only if its latent logical condition is satisfied.

Turn On Lights Toggle <light>

Purchase specific shares of stock and sell specific shares of stock. Multiple stocks can be acquired or sold simultaneously. The buy action is successful only if the cash is enough. If the selling shares exceed the current holdings, the environment defaults to selling the entire existing position.

{

"Buy":<stocks, shares>, "Sell":<stocks, shares>

AI Trading

}

Dispatch p1, p2, p3 units of thermal, wind, and solar power re-

{

"Thermal":<p1>, "Wind":<p2>, "Solar":<p3>, "Battery":<p4>

spectively. For p4, a negative value represents charging and positive represents discharging. Notably, charging is truncated upon reaching the battery’s max capacity, and the discharging is limited to the available capacity if it exceeds the current state of charge. The cost of the four energies should be limited within the total budget.

Energy Dispatch

}

Inspect the repository’s directory structure to identify script paths.

repo tree/ls

Install the package. Supports specific version (==) and range constraints (>, >=, <, <=).

Repo System

pip install <package>

pip uninstall <package> Remove the specified package

from the configuration.

pip list List all installed packages with

version identifiers. python <script>

Execute a specified script to verify configuration or trigger entry points.

activates light3 right after activating light2, it may wrongly conclude that only light2 is necessary, ignoring light1.

#### B.3 Reasoning boosts inductive reasoning

In Table 2, we further compare gpt-oss-120b across varying reasoning budgets (low, medium, and high). The results reveal that LLMs demonstrate better inductive performance with more reasoning budget. For example, in Turn On Lights, gpt-oss-120b (high) achieves average success rate of 27.50%, while gpt-oss-120b (medium) achieves 16.67% and gpt-oss-120b (low) achieves 7.50%.

- Table 5: Performance comparison between w/o rules and w/ rules settings. We provide three different reasoning effort of gpt-oss-120b. For AI Trading environment, we report the profit rate and pass@4 is calculated based on the highest profit of each task. For other three environments, we report the success rate. Colored Rows represent proprietary models.

Turn On Lights AI Trading Energy Dispatch Repo Management w/o rules w/ rules w/o rules w/ rules w/o rules w/ rules w/o rules w/ rules

Model

Gemini 3 Pro Preview 44.17 100.00 +67.71% +135.48% 30.00 16.67 65.83 96.67 GPT-5 28.33 100.00 +17.32% +132.02% 23.33 13.33 62.50 96.67 Gemini 2.5 Pro 29.17 100.00 +33.02% +126.24% 10.83 20.00 50.00 96.67 gpt-oss-120b (high) 27.50 100.00 +23.27% +141.63% 0.00 0.00 18.33 50.00 DeepSeek-V3.2 18.33 96.67 +8.62% +118.88% 0.00 0.00 48.33 56.67 Grok 4 Fast 14.17 100.00 +5.70% +62.96% 0.00 0.00 38.33 36.67 Qwen3-235B-A22B-Instruct 15.00 90.00 +11.26% +123.66% 0.00 0.00 15.83 33.33 Qwen3-30B-A3B-Instruct 11.67 50.00 +4.76% +94.00% 0.00 0.00 26.67 43.33 gpt-oss-120b (medium) 16.67 100.00 +3.21% +100.27% 0.00 0.00 2.50 30.00 GLM-4-32B-0414 14.17 60.00 +3.14% +18.50% 0.00 0.00 9.17 6.67 gpt-oss-120b (low) 7.50 36.67 +2.02% +26.40% 0.00 0.00 9.17 26.67 Llama 3.3 70B Instruct 6.67 33.33 +0.77% -0.93% 0.00 0.00 19.17 50.00 Qwen3-4B-Instruct 0.00 10.00 +1.67% +60.52% 0.00 0.00 13.33 16.67 Llama 3.1 8B Instruct 6.67 10.00 +0.55% +0.18% 0.00 0.00 0.00 3.33 GLM-4-9B-Chat 0.00 0.00 -0.18% +0.34% 0.00 0.00 0.00 0.00

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Qwen3-235B-A22B-Instruct

Gemini 3 Pro Preview

DeepSeek-V3.2

Grok 4 Fast

random

Llama 3.3 70B Instruct

Gemini 2.5 Pro

GLM-4-9B-Chat

GPT-5

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

0.4

0.60

0.60

SuccessRate

0.3

0.45

0.45

0.2

0.30

0.30

0.1

0.15

0.15

0.00

0.0

0.00

0 40 80 120 160 200

0 30 60 90 120

0 30 60 90 120

Step

Step

Step

Repo System

Turn On Lights

AI Trading

Qwen3-30B-A3B-Instruct

Llama 3.1 8B Instruct

gpt-oss-120b (low)

GLM-4-32B-0414

gpt-oss-120b (medium)

gpt-oss-120b (high)

random

Qwen3-4B-Instruct

0.24

| | | | | |
|---|---|---|---|---|
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

0.24

0.24

0.18

SuccessRate

0.18

0.16

0.12

0.12

0.08

0.06

0.06

0.00

0.00

0.00

0 40 80 120 160 200

0 30 60 90 120

0 30 60 90 120

Step

Step

Step

Repo System

Turn On Lights

AI Trading

- Figure 7: All data for Success Rate against Step. We do not plot for Energy Dispatch environment due to its complex success conditions.

#### B.4 More results and implementation details of evolution progress

More Results of Evolution Progress. We provide detailed results for 15 models and a random baseline in Figure 7. We can observe that: (1) In Turn On Lights and Repo System environments, most models demonstrate saturation as the interaction step increases, indicating that long-horizon remains a bottleneck for SOTA LLM models. (2) Some models, e.g. GLM-4-9B-Chat, perform equivalent or even worse than random baseline, indicating their inability of discovering hidden rules of the environment. (3) Models demonstrate different performance curves, suggesting that we should evaluate each model from multiple timestamps.

Implementation Details. For the Turn On Lights and Repo Management environments, we plot evolution curves with the interaction step index on the x-axis and the cumulative task success rate on the y-axis, defined as the proportion of successfully completed tasks from the beginning to the

current step. This formulation captures the progression of task completion over time and reflects the agent’s inductive reasoning capability as the interaction horizon increases. At any given step, a higher y-axis value indicates a stronger inductive reasoning performance in the current step. For the AI Trading environment, the y-axis represents the profit rate relative to the initial capital, which is then averaged across all tasks in this environment. We do not report evolution curves for the Energy Dispatch environment, as task success is determined by complex, multi-constraint conditions that do not admit a well-defined step-wise success metric.

Notably, as shown in the results above, model performance varies across different interaction steps, indicating that evaluation at a single step is insufficient. Therefore, it is necessary to assess model performance across multiple steps to obtain a more comprehensive and reliable understanding of their inductive reasoning capabilities.

[Figure 180]

- (g)

(f)

(e)

(d)

(c)

(b)

(a)

Turn On Lights

[Figure 181]

0 20 40 60 80 100 120

Steps Energy Dispatch

[Figure 182]

0 20 40 60 80 100 120

Steps Repo System

[Figure 183]

0.0000

0.0111

0.0222

0.0333

0.0444

0.0556

StepDensityDistribution

Figure 8: Step density distribution for LLM models: (a) Gemini 3 Pro Preview, (b) GPT-5, (c) Gemini 2.5 Pro, (d) gpt-oss-120b (high), (e) DeepSeek-V3.2, (f) Grok 4 Fast, (g) Qwen3-235B-A22B-Instruct,

- (h) gpt-oss-120b (medium), (i) Qwen3-30B-A3B-Instruct, (j) GLM-4-32B-0414, (k) gpt-oss-120b (low), (l) Llama 3.3 70B Instruct, (m) Qwen3-4B-Instruct (n) Llama 3.1 8B Instruct, (o) GLM-4-9BChat.

(h)

(i)

(j)

(k)

(l)

(m)

(n)

(o)

0 20 40 60 80 100 120 140 160 180 200

Steps

Easy(50.67%) Medium(40.67%) Hard(32.00%)

- (a)
- (b)
- (c)
- (d)
- (e)
- (f)
- (g)
- (h)
- (i)
- (j)
- (k)
- (l)
- (m)
- (n)
- (o)
- (p)

[Figure 184]

[Figure 185]

Task

Task

Repo System

Energy Dispatch

- Figure 9: Task success status (based on pass@4). Each row represents: (a) Human, (b) Gemini3 Pro Preview, (c) GPT-5, (d) Gemini 2.5 Pro, (e) gpt-oss-120b (high), (f) DeepSeek-V3.2, (g) Grok 4 Fast, (h) Qwen3-235B-A22B-Instruct, (i) gpt-oss-120b (medium), (j) Qwen3-30B-A3B-Instruct, (k) GLM-4-32B-0414, (l) gpt-oss-120b (low), (m) Llama 3.3 70B Instruct, (n) Qwen3-4B-Instruct, (o) Llama 3.1 8B Instruct, (p) GLM-4-9B-Chat. Dark green cells indicate tasks solved by Human. Green cells indicate tasks solved by LLM agents. Gray cells indicate unsolved tasks. We report the average success rate in Repo System across all LLMs for each subset (Easy, Medium and Hard).

#### B.5 More results and implementation details of task success status

We provide detailed results of task success status for 14 models in Energy Dispatch and Repo System in Figure 9. We can observe that there is a significant gap between human and SOTA LLMs. Notably, most LLMs fail all tasks in Energy Dispatch, indicating an inability of long-horizon inductive reasoning.

#### B.6 Analyze of task success status

We provide an overview of solved tasks and unsolved tasks in Figure 9. Additionally, for Repo System, we simplify the difficulty representation using the number of required packages and separate all tasks into easy, medium and hard sets based on lights number . Human can solve all the 30 tasks, while the best LLM (Gemini 3 Pro Preview) can only solve 25 tasks. Notably, the we do not simplify or provide the difficulty level of Energy Dispatch due to its complex multi-object goal.

#### B.7 Memory constrain: the impact of memory usage

We conduct a comparison experiment in Repo System environment, where different length of interaction history is provided to the agent, which is detailed in Table 6. The following results demonstrate that the best performance occurs with about 100 turns of context window, indicating that current agents cannot inductively reasoning across overlong memory (about 200 turns). And merely about 50 turns cannot provide sufficient information for inductive reasoning.

Table 6: Performance in Repo System with different memory usage.

Avg@4 Pass@4 200 turns 150 turns 100 turns 50 turns 200 turns 150 turns 100 turns 50 turns

Model

Gemini 3 Pro Preview 65.83 70.00 76.67 70.83 80.00 83.33 86.67 83.33 gpt-oss-120b-high 18.33 19.17 23.33 11.67 33.33 33.33 36.67 20.00

#### B.8 Loop implementation details

Given a trajectory dataset, a global measure of interaction volume is obtained by aggregating the total number of executed actions across all trajectories. For Turn On Lights and Repo System environments, we additionally analyses the Loop Ratio for each model. To characterize the inability of inductive reasoning, we focus on repetitive execution of previously taken action that yields no effective task progress.

Concretely, we define the state as the on/off of each light for Turn On Lights environment and the version of each package for Repo System environment. Within each trajectory, we then detect whether an agent replays the same pair (state, action) immediately after completing it and the two interactions make no task progress. We then sum the action counts of all such immediately repeated pairs and normalize this quantity by the total number of actions. This normalized proportion defines the Loop Ratio, which measures how much of the agent’s interaction is spent on consecutive, unproductive repetition.

Under this formulation, a smaller Loop Ratio indicates that the agent is more capable to find hidden rules through inductive reasoning.

#### B.9 Eliciting inductive reasoning is NOT trivial

We conduct experiments with in-context examples in Repo System environment. We additionally provide one-shot example consisting of 12 actions for completion. Results in Table 7 demonstrate that merely providing in-context examples cannot strengthen inductive reasoning ability of agents. More techniques such as memory management or systematically exploration encouragement should be involved in future agents.

#### B.10 Step distribution analysis

We plot the distribution of the total steps number required to complete the task for each trajectory, including both successful and failed trajectories. In the Turn On Lights and Repo System environments, step usage serves as a proxy for inductive reasoning efficiency, where fewer steps indicate stronger inductive reasoning capability. In the Energy Dispatch environment, step usage corresponds to the number of days during which energy is supplied without violating the three-day consistency constraint. A higher step usage represents a better inductive reasoning capability. Notably, every trajectory in AI Trading environment has 120 steps in ODYSSEYARENA-LITE, so we do not plot a step distribution figure for this environment. The results are in Figure 8.

Table 7: In-Context Learning Performance Comparison in Repo System

Model Original Avg@4 In-Context Avg@4 Original Pass@4 In-Context Pass@4

Gemini 3 Pro Preview 65.83 66.67 80.00 81.67 gpt-oss-120b-high 18.33 17.50 33.33 30.00

We can observe a sharp concentration around max step limit (200 steps for Turn On Lights and 120 steps for AI Trading), indicating that most SOTA models demonstrate limited inductive reasoning capability in this environment and can not solve the task within pre-defined steps limit.

Additionally, in Energy Dispatch environment, most models exhibit step distributions that are sharply concentrated at relatively small values, indicating an inability to satisfy the demand and budget constraints for three consecutive days. Conversely, for Gemini 3 Pro Preview, Gemini 2.5 Pro, and GPT-5, the step distribution is sharply concentrated around 120 steps, suggesting that these models can satisfy the demand and budget constraints over the evaluated horizon. Notably, task success is additionally governed by the carbon and stability constraints and sustaining 120 days does not necessarily imply success.

Token Usage

Token Usage

0 5 10 15 20 25

0 2 4 6 8 10

GLM-4-9B-Chat

GLM-4-9B-Chat

GLM-4-32B-0414

GLM-4-32B-0414

Llama 3.3 70B Instruct

Llama 3.3 70B Instruct

Llama 3.1 8B Instruct

Llama 3.1 8B Instruct

Qwen3-4B-Instruct

Qwen3-4B-Instruct

Qwen3-30B-A3B-Instruct

Qwen3-30B-A3B-Instruct

Qwen3-235B-A22B-Instruct

Qwen3-235B-A22B-Instruct

0.00 0.05 0.10 0.15 0.20 0.25

0.00 0.01 0.01 0.01 0.02 0.02 0.03 0.04

Token Efficiency

Token Efficiency

(a) Turn On Lights

###### (b) AI Trading

Token Usage

Token Usage

0.0 0.2 0.4 0.6 0.8 1.0 1.2

0.0 0.5 1.0 1.5 2.0

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |

GLM-4-9B-Chat

GLM-4-9B-Chat

GLM-4-32B-0414

GLM-4-32B-0414

Llama 3.3 70B Instruct

Llama 3.3 70B Instruct

Llama 3.1 8B Instruct

Llama 3.1 8B Instruct

Qwen3-4B-Instruct

Qwen3-4B-Instruct

| | |
|---|---|
| | |

Qwen3-30B-A3B-Instruct

Qwen3-30B-A3B-Instruct

| | |
|---|---|
| | |

Qwen3-235B-A22B-Instruct

Qwen3-235B-A22B-Instruct

0.00 0.00 0.00 0.01 0.01 0.01

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75

Token Efficiency

Token Efficiency

(c) Energy Dispatch

(d) Repo System

- Figure 10: The line chart represents token usage and the bar chart represents token efficiency. Token Usage is measured in units of 106, while Token Efficiency is reported in units of 10−6.

#### B.11 Token efficiency analysis

We analyze token efficiency, defined as the contribution of each token to the final success rate (or trading profit), and calculate it as the success rate (or trading profit) divided by the total token usage. We provide detailed analysis of GLM series, Llama series and Qwen series in Figure 10.

Results in Figure 10 demonstrate that GLM-4-32B-0414 is the most token-efficient in Turn On Lights and AI Trading environment and Llama 3.3 70B Instruct is the most token-efficient in Repo System environment.

Interestingly, model GLM-4-9B-Chat generates negative profit in AI Trading environment, so its token efficiency is negative. Moreover, models from the three series fail all the tasks in Energy Dispatch environment.

Notably, although Qwen achieves best success rate (or trading profit) among the three series, their consistent high token usage makes their token efficiency extremely low.

#### B.12 Different strategies for AI trading

In this subsection, we provide more methods for AI Trading environment and analyze their profit. Notably, the strategies in this subsection do not require LLMs and is implemented through only python scripts.

Optimal Strategy. We introduce an optimal strategy that operates under the assumption of perfect information regarding future market dynamics. Specifically, with full access to the price trajectory of all stocks, this method adopts a greedy strategy: at each time step, it allocates the entire portfolio to the stock with the highest single-day return ratio. In scenarios where all stocks exhibit a downward trend, keep all assets as cash.

Conservative Strategy. The fundamental principle of this strategy is to delay trading until sufficient historical data is collected. Specifically, wait and do nothing for at least num_factors + 2 days. This threshold ensures that the linear system of equations is overdetermined or has a unique solution before any parameter estimation occurs, thereby avoiding instability in the early stages.

We first estimate the dependency matrix M, which maps factor changes to price changes. We then constructs the price and factor matrices to solve for M using linear algebra solvers and buy/sell stocks based on the hidden rules we find.

Progressive Strategy. This strategy adopts an aggressive mechanism, starting trading from the third day rather wait for more information.

We employ an incremental learning framework based on the least square method. At any given time step, we utilize the entire history of available price and factor changes accumulated up to that point to estimate the dependency matrix. Unlike approaches that rely on fixed data windows or static thresholds, this strategy dynamically updates its understanding of the market. As new data become available each day, the linear regression model is re-calculated using the full dataset, allowing the estimated parameters to evolve continuously.

This approach fundamentally represents a strategic trade-off between estimation precision and market timing. In the initial stages, the model operates with higher risk because the dataset is sparse, which may lead to significant estimation errors and suboptimal trading decisions. The estimator naturally converges towards the true parameters as more information is gathered.

Correlation Strategy. While progressive strategy attempt to solve for the entire dependency matrix simultaneously by considering the joint influence of all factors, the correlation strategy operates on the assumption that these factors are statistically independent.

For every individual pair of stocks and factors, we utilize the full historical dataset available to compute this simple regression coefficient. By iterating through all combinations, the model updates the dependency matrix element by element in each step.

Rolling Window Strategy. The Rolling Window strategy posits that recent market behaviors are more predictive of immediate future trends. Consequently, the strategy limits its estimate scope to a fixed-size trailing window (specifically set to 15 days), ensuring that the model remains sensitive to structural shifts in the market environment.

On any given trading day t, we construct the dependency matrix using only data pairs from the most recent w days through the least squares method. Old data points that fall outside this window are systematically ignored in the calculation.

Ridge Regression Strategy. This strategy modifies the core learning logic of the progressive strategy by introducing a regularization mechanism.

Instead of simply minimizing the prediction error through the least squares method, we apply a strict penalty to the magnitude of the regression coefficients. By artificially shrinking the coefficients, the algorithm effectively suppresses the noise that arises when factors are too similar, preventing any single factor from exerting an unrealistically large influence on the trading decision.

Table 8: Comparison of different strategies, human and models. For LLM models, we provide the best results from both proprietary models and open-source models. ∆ represents the delta of average profit ratio against optimal strategy. Colored Rows represent proprietary models.

Method Avg. Profit Ratio ∆ Optimal Strategy +211.13% – Conservative Strategy +192.23% -18.90% Progressive Strategy +197.33% -13.80% Correlation Strategy +181.51% -29.62% Rolling Window Strategy +197.31% -13.82% Ridge Regression Strategy +192.63% -18.50% Human Annotation 92.55% -118.58%

Gemini 3 Pro Preview +67.71% -143.42% Qwen3-235B-A22B-Instruct +11.26% -199.87%

[Figure 186]

[Figure 187]

Table 9: Performance of three LLMs in Repo System of ODYSSEYARENA-CHALLENGE and ODYSSEYARENA-LITE. Colored Row represents proprietary models. We additionally provide the performance gap.

Model Lete Challenge ∆

Gemini3 Pro Preview 65.83 10.00 -55.83 Qwen3-235B-A22B-Instruct 15.83 0.00 -15.83 Qwen3-30B-A3B-Instruct 26.67 0.00 -26.67

[Figure 188]

[Figure 189]

[Figure 190]

Comparison of Strategies We provide results of the above mentioned strategies, human annotation and two SOTA models in Table 8. We can observe that various strategies achieve different performance. Notably, SOTA models demonstrate a significant performance gap against both human annotation and the strategies proposed in this subsection, indicating a huge space for inductive reasoning ability optimization.

#### B.13 ODYSSEYARENA-CHALLENGE results

For our proposed extremely long-horizon and complex dataset ODYSSEYARENA-CHALLENGE, we test Repo System and report the results in Table 9, indicating that the long-horizon scenario is still the bottleneck of current LLM’s inductive reasoning.

### C Main results details

#### C.1 Baseline settings

For Qwen3-4B-Instruct and Qwen3-235B-A22B-Instruct, we use the version Qwen3-4B-Instruct2507 and Qwen3-235B-A22B-Instruct-2507. For Llama3.1-8B-Instruct and Llama3.3-70B-Instruct, we use the checkpoint updated in February, 2025. For GLM-4-9B-Chat, we use the checkpoint updated in January, 2025. For DeepSeek-V3.2, we use the non-thinking version. The reasoning effort of GPT-5 is medium. We additionally report different reasoning effort of gpt-oss-120b: low, medium, high.

Experiments of open-source models are conducted on NVIDIA H200 GPUs, Intel Platinum 8480+ Processor CPU with 56 cores and 1 TB memory. For all experiments, we set the temperature to 0.6 using vLLM engine [12] 0.8.5.post1.

For generation, we use </action> and </finish> as stop tokens. We then parse the generated action within each <action></action> <finish></finish> pair.

For AI Trading environment, we keep the 50 most recent interactions as the memory. For Energy Dispatch environment, we keep the 40 most recent interactions as the memory. For other two environments, we keep all interactions as memory.

#### C.2 Evaluation prompts

Each environment provides the model with three components: system prompt, history records, and current state observation.

Turn On Lights Environment The Lights environment provides the following information:

- • System Prompt: Concise goal description (light all bulbs), rule explanation, action format (integer index).

- • History: Interaction history with explicit feedback: Action: {action}, Feedback: {feedback}, State: {obs}.
- • Current State: Visual representation using symbols.

AI Trading Environment The Trade environment provides the following state information:

- • System Prompt: Trading objectives, market dynamics explanation, action format (JSON buy/sell combinations).
- • History: Interaction history formatted as market_info + Action: + action, containing stock prices, holdings, cash, total value, and news hints.
- • Current State: Current day, stock prices and holdings, cash, total value, and next day’s news (predictive price change hints).

Energy Dispatch Environment The Energy environment provides the following state information:

- • System Prompt: Environment description (four generation types), constraints (demand, budget, stability, carbon), dynamic target thresholds, action format.
- • History: Interaction history formatted as state + Action: + action, containing daily status metrics, previous generation results, supply/demand balance, and financial status.
- • Current State: Day number, status indicators (stability, carbon, battery), previous step results, current demand, and the next day’s budget.

Repo System Environment The Repo environment provides the following state information:

- • System Prompt: Detailed environment description, debugging strategies, error interpretation guidelines, action format (command strings).
- • History: Interaction history in structured format: Feedback:{result}\n\n=== Step {n} ===\n»> Command: {command}, separated by double newlines.
- • Current State: Execution result of the last command (success message or error details).

### D Human annotation details

In this section, we provide detailed information regarding the human annotation process conducted to evaluate ODYSSEYARENA-LITE, in accordance with responsible NLP research practices.

#### D.1 Instructions given to participants

We provide annotators with the full text of instructions and a comprehensive guide prior to the task. To facilitate a clear understanding of the evaluation criteria, we designed an intuitive annotation interface. A screenshot of this interface is presented in Figure 11. Moreover, we provide the user instructions for all environments.

We detail the System Prompts of the four above mentioned environments in the following part.

Turn On Lights Environment System Prompt You are an intelligent agent. ### Goal: Your mission is to light on all the bulbs. However, the accessibility of the bulbs is based on the current condition of other bulbs. You need to learn the hidden rule behind the environment and complete the task. ### Action Space: The action space is based on the index of bulbs. For example, you would like to light on / off the first bulb, you should output <action>0</action> to toggle the state of the bulb. ### History Action and Feedback: {history_trajectories} ### Current State: {the_state_of_each_light}

Now think step by step and choose the next action to act in the environment. You are encouraged to act actively to derive the environment dynamics. Output ONLY one action in the format: <action>n</action>

AI Trading Environment System Prompt

### Goal: Your mission is to maximize your total portfolio value by buying and selling stocks. The market prices are influenced by underlying variables F, and each day’s news provides hints about future price changes. You need to learn the hidden dynamics of the simulated market and make decisions accordingly. Please note that the underlying meaning of variables may differ from the real stock.

### Action Space: You can take actions in the form of buying or selling multiple stocks each day. You can combine buy and sell in one action. The environment will first execute all sell actions, then all buy actions. You cannot spend more cash than you have or sell stocks you don’t own.

**Action Format Examples:**

- To buy 10 shares of S0 and 20 shares of S2, and sell 10 shares of S1: <action>{{"buy": {{"S0": 10, "S2": 20}}, "sell": {{"S1": 10}}}}</action>

- - To only buy: <action>{{"buy": {{"S0": 5}}, "sell": {{}}}}</action>
- - To do nothing: <action>{{"buy": {{}}, "sell": {{}}}}</action>

**Important:**

- - Stock symbols and numbers should NOT have quotes
- - Use valid JSON format inside <action></action> tags
- - If you cannot afford a purchase or don’t own enough shares to sell, that part of the action

will be ignored ### History Actions and Feedback: {history_trajectories} ### Current State: {cash_and_the_price_of_each_stock} Think carefully step by step and decide your next action. You are encouraged to act proactively, using the news to predict future price changes, and to improve your strategy over time. Provide your action in the format: <action>...</action>

You are an intelligent energy system operator managing a Dynamic Energy Grid. Your goal is to achieve a safe, stable, and low-carbon electricity supply across a long planning horizon. Each day, you adjust the composition of generation resources within strict physical and economic limits. To perform well, you must learn and exploit hidden temporal patterns from the history.

# ENVIRONMENT OVERVIEW This environment simulates a long-horizon national power grid with four generation types: Thermal — highly reliable, carbon-intensive, lowest cost. Wind — highly variable, driven by seasonal cycles. Solar — variable, driven by seasonal cycles. Battery (Storage) — A storage buffer that can charge or discharge based on the capacity. Its carbon footprint is determined by the source of energy used for charging. Each day t, the system evolves according to underlying temporal dynamics. The agent must design the next day’s rated generation scheme while anticipating these dynamics.

## Demand & Budget The allocation scheme must strictly satisfy both demand and budget constraints. current_demand (MW) — electricity required today. current_budget — tomorrow’s maximum allowable total generation cost.

## Generation Cost Model (Unit Prices) Each generation type has a fixed unit cost per MW of rated output: Thermal: cheapest (e.g., 3.0 cost/unit) Wind: moderate cost (e.g., 5.0 cost/unit) Solar: more expensive (e.g., 6.0 cost/unit) Battery: operational cost (Charge/Discharge), very low (e.g., 0.1 cost/unit)

## Grid Stability To maintain a stable grid, the agent must avoid large day-to-day changes in the rated outputs. Sudden increases or decreases (ramping) reduce stability, which affects overall performance. A good strategy adjusts gradually, anticipating future needs rather than reacting abruptly. Violating the daily budget or failing to meet the demand would largely damage system stability.

## Carbon Intensity Thermal generation emits carbon. To maintain a clean and sustainable city, the agent should limit the proportion of thermal power while still meeting demand and respecting budget constraints. This creates a non-trivial trade-off between cost, stability, and carbon impact.

## Season & Efficiency Actual generation is not equal to rated generation. It depends on a time-varying efficiency term: actual_output = rated_output × efficiency (t) Efficiency changes periodically over time. Solar and Wind share different periods. Agent is required to derive the hidden temporal rules from the history observation. Because actual output fluctuates around rated output, the agent must leave safety margins and learn the temporal structure.

# Objective The agent needs to simulate across a long planning horizon (120 Turns). The task is successful only if the final metric **Stability > {target_stability}, Carbon < {target_carbon} **. Notably, violation of daily cost or demand constraints for 3 consecutive steps would lead to termination.

# Action Space Each day, the agent must decide the rated generation for the next day within the capacity limit: Thermal (MW), Rated Power Command, [0,600], Must be non-negative. Wind (MW), Rated Power Command, [0,350], must be non-negative. Solar (MW), Rated Power Command, [0,250], must be non-negative. Battery (MW), Net Flow Command, battery capacity=80, Bidirectional: Negative = Charge (Consumption), Positive = Discharge (Supply).

- **Action Format Example 1**: <action>{{"thermal": 400.0, "wind": 10.0, "solar": 35.0, "battery": -15.0}}</action> Interpretation: The agent sets the Rated Power for Thermal, Wind, and Solar to 400 MW, 10 MW, and 35 MW, respectively. Additionally, the agent commands the battery to consume 15 MW from the grid for charging. This 15 MW consumption will be drawn from the total supply available from the three generation units.
- **Action Format Example 2**: <action>{{"thermal": 350.0, "wind": 25.0, "solar": 15.0, "battery": 10.0}}</action> Interpretation: The agent sets the Rated Power for Thermal, Wind, and Solar to 350 MW, 25 MW, and 15 MW, respectively. Additionally, the agent commands the battery to supply 10 MW of power to the grid (discharging). This 10 MW is added to the total supply from the three generation units.

# History Action and Feedback: {history_trajectories}

# Current State: {last_day_info_and_today_info}

**Important Note:**

- - Set Rated Capacity above Actual Demand to save room for the efficiency gap (Rated vs. Actual output) and forecast uncertainty.
- - Keep daily cost within the budget and meet the daily demand, violation of either cost and supply for three consecutive steps would lead to immediate, irreversible grid collapse.
- - Stability and Carbon are long-term average metric. After 120-turn, stability must be > {target_stability}, Carbon must be < {target_carbon}.

Now think step by step and choose the next action to act in the environment. You are encouraged to act actively to derive the environment dynamics. Output ONLY one action within the tag of <action></action>.

Repo Management Environment System Prompt (Part I) You are an intelligent computer-using agent. # Environment Overview You are interacting with a simulated Python project setup environment. This environment mimics real-world difficulties of configuring a repo:

- - Partial information (no full dependency graph)
- - Object-level runtime failures (module/symbol/kwarg), not explicit version instructions
- - Non-monotonic side-effects: installing one package may upgrade/downgrade other

packages

- Hidden rules that may only trigger in specific sub-modules or late-stage scripts

Repo System Environment System Prompt (Part II)

# Repo Hierarchy & Debugging The repo is hierarchical: it contains multiple runnable scripts under subdirectories. You can debug incrementally by running sub-scripts (to locate which subsystem fails), but the final goal is to make the entire project pass. Use:

- - ‘repo tree‘ (or ‘repo ls‘) to list available scripts in the repo.
- - ‘python <script_path>‘ to run a specific sub-script and fix it step by step.
- - ‘python run.py‘ to run the whole project (a sequence of entry points). This is the only

command that ends the episode with success. # Goal Your ultimate goal is to make: ‘python run.py‘ execute successfully. # Action Space (ONE command per step)

- - Install Python:
- - ‘pip install python==3.10‘
- - Install packages:
- - ‘pip install pkgX‘
- - ‘pip install pkgX==1.2‘ (note: if you output x.y.z, it will be interpreted as x.y)
- - ‘pip install pkgX>=1.1,<2.0‘
- - Uninstall packages:
- - ‘pip uninstall pkgX‘
- - Inspect environment:
- - ‘pip list‘
- - Inspect repo structure:
- - ‘repo tree‘ / ‘repo ls‘
- - Execute scripts:
- - ‘python run.py‘
- - ‘python core/smoke.py‘ (example; use ‘repo tree‘ to discover actual paths) Other commands (e.g., ‘–upgrade‘) are not supported. # How to Interpret Errors (Important) Errors are meant as clues without directly stating version ranges:
- - ‘ModuleNotFoundError: No module named pkgX‘ usually means pkgX is missing.
- - ‘ImportError: cannot import name ’S’ from pkgX.mod‘ often means pkgX version does not export that symbol.
- - ‘TypeError: ... got an unexpected keyword argument kw‘ indicates signature/API mismatch. If the message says "during project entry", adjust the provider package used by the project. If it says "while importing caller_pkg", it indicates a caller->provider incompatibility. Because installations can trigger side effects, a later fix may break an earlier sub-script. Use sub-scripts to localize failures, but always re-run ‘python run.py‘ to confirm global consistency.

# History Action and Feedback: {history_trajectories}

# Current Environment Feedback: {specific_feed_back_of_command}

Now think step by step and choose the next action. Output exactly ONE action inside <action></action>, e.g. <action>pip install pkg0==2.1</action>.

[Figure 191]

Figure 11: Screenshot of the user interface and instructions provided to the human annotators.

Turn On Lights Environment User Instruction Assume there are 3 bulbs (indices 0, 1, 2), all initially off. Example Logic (only shown in examples, these rules are hidden in actual tasks, users need to infer):

- • B0: True # Represents B0 can be turned on under any circumstances
- • B1: B0 # Represents B1 can only be turned on when B0 is on
- • B2: not B1 and B0 # Represents B2 can only be turned on when B1 is off and B0 is on

#### Example Steps:

- 1. Step 1: Input action 1, click “Execute Action”

- • Environment state after execution: ◦ ◦ ◦
- • Environment feedback: B1 remains inactive... remaining bulbs should be in specific mode.
- • Reason: B1 can only be turned on when B0 is on, but B0 is off, so B1 cannot be turned on.

- 2. Step 2: Input action 0, click “Execute Action”

- • Environment state after execution: • ◦ ◦
- • Environment feedback: Toggled B1 to True
- • Reason: B0 can be turned on at any time.

- 3. Step 3: Input action 2, click “Execute Action”

- • Environment state after execution: • ◦ •
- • Environment feedback: Toggled B2 to True
- • Reason: B2 can only be turned on when B1 is off and B0 is on, so B2 was turned on.

- 4. Step 4: Input action 1, click “Execute Action”

- • Environment state after execution: • • • (Task completed)
- • Environment feedback: Toggled B1 to True
- • Reason: B1 can only be turned on when B0 is on, so B1 was turned on.

#### Tips:

- • • indicates bulb is lit
- • ◦ indicates bulb is not lit
- • Each bulb’s availability may depend on other bulbs’ states
- • You need to discover hidden rules through experimentation
- • Maximum 200 steps allowed

31

Goal: Light all bulbs (all bulbs display as •)

- AI Trading Environment User Instruction (Part I)

Scenario Description You are a stock trader who needs to perform buy and sell operations across multiple trading days, achieving maximum returns within 120 days.

#### Important Concepts:

- • S0, S1: Stock codes (Stocks), representing 2 different stocks that can be bought and sold
- • F0, F1: Market factors (Factors), representing market factors that affect stock prices

- – News will report changes in these factors (e.g., “F0 rose slightly (+0.03)”)
- – Factor changes affect stock prices through dependency matrix
- – You need to predict stock price changes based on news, then trade

- • Please check news, e.g., “F0 rose slightly (+0.03) | F1 decreased significantly (-0.10)” predict which stocks will rise/fall based on factor changes
- • Buying is limited by cash
- • Selling is limited by holdings

#### Available Actions:

- • Buy Stock: Input positive number to buy (e.g., S0 input 100 means buy 100 shares of S0)
- • Sell Stock: Input negative number to sell (e.g., S0 input -50 means sell 50 shares of S0)
- • Buying is limited by cash, selling is limited by holdings

Example: Example Logic (only shown in examples, these rules are hidden in actual tasks, users need to infer):

- • Matrix corresponding to S0, S1, F0, F1 is:

0.1 0.2 −0.3 0.4

- • Represents:

- – F0 rises 1 point, S0 rises 0.1 points; F0 rises 1 point, S1 falls 0.3 points
- – F1 rises 1 point, S0 rises 0.2 points; F1 rises 1 point, S1 rises 0.4 points

#### Initial Environment in This Example:

- • You have 100 cash
- • S0 initial price is 1, S1 initial price is 2
- • This example is a simple demonstration, only keeps 2 days (actual task is 120 days)

- AI Trading Environment User Instruction (Part II) Example Steps:

#### Step 1 (Day 1):

- • Environment state before execution: Tomorrow F0 rose significantly (+0.10) | F1 rose slightly (+0.05)
- • Stock prices before execution: S0 1.00, S1 2.00, Cash 100
- • Action: Buy 100 shares of S0
- • Reason: S0 tomorrow’s price = 1.00 + (0.1×0.10) + (0.2×0.05) = 1.00 + 0.01 + 0.01

- = 1.02 (up 2%), while S1 tomorrow’s price is S1 = 2.00 + ((-0.3)×0.10) + (0.4×0.05)
- = 2.00 - 0.03 + 0.02 = 1.99 (down 0.5%). S0 rises while S1 falls, so buy S0. Buying 100 shares of S0 costs 100, cash becomes 0.

#### Step 2 (Day 2):

- • Environment state before execution: Tomorrow F0 decreased significantly (-0.15) | F1 rose significantly (+0.10)
- • Stock prices before execution: S0 1.02, S1 1.99, Cash 0, Holdings 100 shares of S0
- • Sell 100 shares of S0, buy about 51 shares of S1
- • Reason: S0 tomorrow’s price = 1.02 + (0.1×(-0.15)) + (0.2×0.10) = 1.02 - 0.015 + 0.02 = 1.025 (slight rise 0.5%), while S1 tomorrow’s price is S1 = 1.99 + ((-0.3)×(0.15)) + (0.4×0.10) = 1.99 + 0.045 + 0.04 = 2.075 (up 4.3%). S1 rise is much greater than S0, so sell S0 and buy S1. Selling 100 shares of S0 gets 102, can buy about 51 shares of S1 (102/1.99 = 51.26, rounded to 51 shares, costs about 101.49).

#### Step 3 (Day 3):

- • Environment state before execution: Tomorrow F0 stable (0.00) | F1 rose significantly (+0.20)
- • Stock prices before execution: S0 1.025, S1 2.075, Cash 0.51, Holding 51 shares of S1
- • Action: No operation (or use remaining cash to buy small amount of S1)
- • Reason: S0 tomorrow’s price = 1.025 + (0.1×0) + (0.2×0.20) = 1.025 + 0.04 = 1.065 (up 3.9%), while S1 tomorrow’s price is S1 = 2.075 + ((-0.3)×0) + (0.4×0.20) = 2.075 + 0.08 = 2.155 (up 3.9%). Both stocks have similar rises, but S1 absolute rise is larger (0.08 vs 0.04), and already holding S1, so maintain position.

Final State: 51 shares of S1, price 2.155 per share, total value about 109.91 (51×2.155), plus remaining cash about 0.51, total value about 110.42, return rate about 10.42%

Energy Dispatch Environment User Instruction (Part I) Scenario Description

You need to manage an energy grid, balancing generation, demand, and budget while meeting stability and carbon emission targets, completing at least 120 days of tasks. If demand violations or budget violations occur for three consecutive days, the task will fail immediately. Task Objectives:

- • Completion Days: Complete at least 120 days
- • Stability Target: Final average stability must be ≥ target value (shown in status)
- • Carbon Emission Target: Final carbon emission ratio must be ≤ target value (shown in status)
- • Violation Limit: 3 consecutive days of demand violations or budget violations will cause task failure

#### Available Actions:

- • Thermal: Input thermal power generation (≥0)
- • Wind: Input wind power generation (≥0)
- • Solar: Input solar power generation (≥0)
- • Battery: Input battery operation

- – Negative value = charging (e.g., -20)
- – Positive value = discharging (e.g., 20)
- – 0 = no battery usage
- – Battery has maximum capacity limit of 80

#### Actual Generation Calculation:

- • Actual generation = Input generation × efficiency coefficient
- • After actual generation, store to battery, this stage has no loss
- • For example, input thermal 10, wind 20, solar 30, battery storage 10. Thermal efficiency 0.9, wind efficiency 1.1, solar efficiency 1
- • Then actual generation: 10 × 0.9 + 20 × 1.1 + 30 × 1 = 61
- • Applied to grid (subtract battery storage): 61 − 10 = 51

#### Stability Requirements:

- • Daily generation configuration changes cannot be too large, otherwise it will cause grid instability
- • Stability calculation considers: generation configuration change magnitude (ramping), budget violations, demand violations
- • If budget violation or demand violation occurs, stability will decrease significantly
- • Important: Insufficient stability will not directly terminate the task, but will be used to judge task success after completion. So you need to adjust strategy timely to improve stability

Energy Dispatch Environment User Instruction (Part II) Carbon Emission Requirements:

- • Carbon emission ratio = Historical cumulative thermal actual generation / Historical cumulative total actual generation
- • When task is completed, carbon emission ratio must be ≤ target value
- • Need to control thermal power proportion of total generation throughout the task
- • Important: Excessive carbon emissions will not directly terminate the task, but will be used to judge task success after completion. So you need to adjust strategy timely to reduce carbon emissions

#### Violation Explanation:

- • Demand Violation: Actual supply < demand
- • Budget Violation: Actual cost > budget
- • Insufficient stability or excessive carbon emissions do not count as violations
- • Three consecutive days of violations will directly terminate and fail the task
- • Important: Only demand violations and budget violations will increase consecutive violation days. Insufficient stability and excessive carbon emissions are not violations but affect final results

Example: Scenario Description:

- • Thermal, wind, solar unit prices are 2, 4, 6 per unit respectively, battery operation cost 0.1 yuan/unit
- • Carbon emission ratio target ≤ 0.81 (i.e., thermal proportion ≤ 0.19)
- • Stability target ≥ 0.5
- • This example demonstrates 6 days, actual task requires completing 120 days

Example Logic (only shown in examples, these rules are hidden in actual tasks, users need to infer):

- • Thermal efficiency sequence: [1.0, 1.0, 1.0, 0.9, 1.1, 1.0] (randomly fluctuates around 1)
- • Wind efficiency sequence: [1.1, 1.0, 1.1, 1.0, 1.1, 1.0] (cycle every 2 days)
- • Solar efficiency sequence: [0.9, 1.0, 1.1, 0.9, 1.0, 1.1] (cycle every 3 days)

#### Important Notes:

- • In actual tasks, efficiency coefficients are hidden and need to be inferred from historical data
- • Need to balance cost, stability, carbon emissions, and demand satisfaction
- • Insufficient stability and excessive carbon emissions will not directly terminate the task but will affect final task completion conditions
- • Only demand violations and budget violations will increase consecutive violation days, 3 consecutive days of violations will cause task failure
- • When violations occur, need to adjust strategy timely to avoid consecutive violations
- • In actual problems, you cannot see the specific calculation process of stability coefficient, you only see a result, please adjust strategy based on this result

Repo System Environment User Instruction (Part I) Scenario Description You need to configure Python environment and install correct package versions so that the project can run successfully: python run.py Available Commands:

- • pip install python==3.10 - Install Python version
- • pip install pkg0==1.2 - Install package (supports version constraints)
- • pip uninstall pkg0 - Uninstall package
- • pip list - View current environment status
- • repo tree - View repository structure
- • python run.py - Run project (success means task completed)

#### Example Hidden Rules (users need to discover in actual tasks):

- • Requires python>=3.10
- • Requires pkg1==1.0
- • Requires pkg2>=1.2,<=2.0
- • Requires pkg3<=1.0
- • All version numbers of pkg3 must match pkg1 (including integer and decimal parts)
- • Major version number of pkg2 must match pkg1 (integer part)

#### Example Steps:

- Step 1: Input pip install python==3.10, click “Execute Action” Environment feedback: Successfully installed python==3.10 Reason: Successfully installed
- Step 2: Input python run.py, click “Execute Action”

- Environment feedback: ModuleNotFoundError: No module named ’pkg1’.

- Reason: pkg1 not installed

Step 3: Input pip install pkg1==1.0, click “Execute Action”

- Environment feedback: Successfully installed pkg1==1.0

- Reason: Successfully installed pkg1==1.0

- Step 4: Input python run.py, click “Execute Action” Environment feedback: ModuleNotFoundError: No module named ’pkg2’. Reason: pkg2 not installed
- Step 5: Input pip install pkg2==2.0, click “Execute Action” Environment feedback: Successfully installed pkg2==2.0

- Reason: Successfully installed pkg2==2.0

- Step 6: Input python run.py, click “Execute Action” Environment feedback: RuntimeError: ABI mismatch detected between ’pkg6’ and dependent packages. Reason: Major version number of pkg2 does not match pkg1

#### D.2 Recruitment and Compensation

This approach ensures a high level of expertise, as all recruited annotators are students with a background in Artificial Intelligence and possess the necessary domain knowledge to accurately assess model performance against the specific criteria defined for ODYSSEYARENA. Annotators are compensated at a rate of about $ 15 per hour.

Repo Syetem Environment User Instruction (Part II)

- Step 7: Input pip install pkg2==1.0, click “Execute Action” Environment feedback: Successfully installed pkg3==1.0 Reason: Successfully installed pkg3==1.2
- Step 8: Input python run.py, click “Execute Action” Environment feedback: ModuleNotFoundError: No module named ’pkg3’. Reason: pkg2 not installed
- Step 9: Input pip install pkg3==1.0, click “Execute Action” Environment feedback: Successfully installed pkg3==1.0

- Reason: Successfully installed pkg3==0.1

- Step 10: Input python run.py, click “Execute Action” Environment feedback: RuntimeError: tightly-coupled components are out of sync with ’pkg1’. Reason: All version numbers of pkg3 must match pkg1 (including integer and decimal parts)
- Step 11: Input pip install pkg3==1.0, click “Execute Action” Environment feedback: Successfully installed pkg3==1.0

- Reason: Successfully installed pkg3==1.0

- Step 12: Input python run.py, click “Execute Action” Environment feedback: Task completed! Project ran successfully! Reason: All conditions met Tips:

- • Packages may have dependencies and version conflicts
- • Need to carefully handle version constraints
- • Maximum 120 steps allowed

Goal: Successfully run python run.py so that the project can execute normally

#### D.3 Consent

Informed consent was obtained from all annotators, with the mutual understanding that the collected data would be utilized for research purposes and released as a public dataset.

#### D.4 Annotation Process

To mitigate individual bias, each example in our dataset was annotated by four independent annotators. The annotation workflow consisted of two stages:

- 1. One-shot Demonstration: Participants were first presented with simplified examples for each environment. These examples served as tutorials and did not overlap with the test dataset.
- 2. Main Annotation: Annotators labeled the generated outputs strictly adhering to the specific rules defined within each environment.

#### D.5 Inter-Annotator Agreement

To ensure the reliability of the human annotated data, we conducted inner agreement analysis across four experimental environments. For tasks with discrete outcomes (Turn On Lights, Energy Dispatch, and Repo System), where performance can be explicitly judged as a binary success metric, Fleiss’ Kappa was employed to account for multiple annotators. The resulting coefficients for these environments indicate a moderate level of consensus consistent. Furthermore, for the AI

Table 10: Inter-rater reliability analysis across four experimental environments. For tasks with discrete success metrics (Turn On Lights, Energy Dispatch, and Repo System), we report Fleiss’ Kappa (κ). For the continuous reasoning task (AI Trading), the Intraclass Correlation Coefficient (ICC) is employed.

Environment Success Rate Type Metric Inner-Annotation Score

Turn On Lights Discrete Fleiss’ κ 0.42 Energy Dispatch Discrete Fleiss’ κ 0.40 AI Trading Continuous Intraclass Correlation Coefficient 0.12 Repo System Discrete Fleiss’ κ 0.18

Trading environment, which yields continuous performance metrics (profit rate), we use the Intraclass Correlation Coefficient (ICC). And the ICC score quantitatively confirm that annotators employed significantly divergent inductive reasoning strategies in this environment.

