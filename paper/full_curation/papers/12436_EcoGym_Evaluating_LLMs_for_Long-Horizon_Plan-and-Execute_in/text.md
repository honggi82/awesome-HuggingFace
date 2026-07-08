# arXiv:2602.09514v3[cs.CL]9May2026

## EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies

OPPO AI Agent Team

### Abstract

Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics. We introduce EcoGym, a generalizable benchmark for continuous plan-and-execute decision making in interactive economies. EcoGym comprises three diverse environments: Vending (adapted from the closed-source Vending-Bench, with full open-source release), Freelance (new), and Operation (new), implemented in a unified decision-making process with standardized interfaces, and budgeted actions over an effectively unbounded horizon (1000+ steps if 365 day-loops for evaluation). The evaluation of EcoGym is based on business-relevant outcomes (e.g., net worth, income, and DAU), targeting long-term strategic coherence and robustness under partial observability and stochasticity. Experiments across eleven leading LLMs expose a systematic tension: no single model dominates across all three scenarios. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. EcoGym is released as an open, extensible testbed for transparent long-horizon agent evaluation and for studying controllability–utility trade-offs in economic settings.

Date: May 12, 2026 Correspondence: He Zhu at zhuhe@oppo.com, Wangchunshu Zhou at zhouwangchunshu@oppo.com Code: https://github.com/OPPO-PersonalAI/EcoGym

### 1 Introduction

Long-horizon planning has been recognized as a central agentic capability since the emergence of large language model (LLM)-based agents [21, 37], motivating both agent scaffolds [17, 33] and foundation models [13, 30] to prioritize long-term navigation and decision-making. This emphasis is likewise reflected in contemporary benchmarks and evaluation protocols, which across a broad range of domains, ranging from deep research [4, 34] and embodied intelligence [29, 31] to autonomous driving [5, 15] and strategic games [22], explicitly or implicitly assess an agent’s capacity for sustained, long-range planning.

Despite the diversity of planning evaluation methodologies, recent research has increasingly situated agents within complex commercial environments and assessed planning competence using tangible, economic returns [6, 26]. This shift is driven by two complementary concerns: (1) the gap between performance in controlled, virtual benchmarks and robustness under continuous, high-stakes practical deployment, where agents often exhibit fragility despite strong benchmark results; and (2) the greater value of evaluations grounded in economic impact rather than simple rewards. Representative efforts include OpenAI’s GDP Eval [26], Scale AI’s Remote Labor Index [23], and the 32-hour expert-level tasks introduced in RE-Bench [35], which collectively aim to quantify the macroeconomic potential of such agents.

3000

1800

|Freelanc|e| | |[Figure 1]|
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | |[Figure 2]|
|[Figure 3]<br><br>[Figure 4]| | |[Figure 5]| |
| | |[Figure 6]|[Figure 7]| |

15000

|Operatio|n| | | |
|---|---|---|---|---|
| | | | |[Figure 8]|
| | | |[Figure 9]<br><br>[Figure 10]| |
| | | | |[Figure 11]<br><br>[Figure 12]<br><br>[Figure 13]|
| | | | | |

[Figure 14]

2500

Vending

10000

1600

2000

5000

compressed above 2000

NetWorth

###### Income

1400

2000

###### DAU

1500

[Figure 15]

[Figure 16]

1500

[Figure 17]

1200

1000

[Figure 18]

[Figure 19]

[Figure 20]

1000

[Figure 21]

1000

500

500

0

800

0

0 90 180 270 365 Day

0 90 180 270 365 Day

0 90 180 270 365 Day

|[Figure 22]<br><br>Claude-Sonnet-4.5 Gemini-3-Pro GPT-5.2 Kimi-k2.6 Qwen3.5-397b-A17b DeepSeek-v3.2 GLM-5|
|---|

- Figure 1 Long-horizon performances across three environments in EcoGym. The plots illustrate the daily progression of key metrics: Net Worth in Vending (left), Income in Freelance (middle), and DAU in Operation (right). Note: Truncated lines represent agents that failed to survive the full simulation horizon due to triggering failure conditions. Only top-performance models are kept for clarity; full experimental results are available in Table 2.

Notwithstanding recent efforts to benchmark long-horizon planning in commercial settings, ranging from macroeconomic evaluations such as GDPval [26] to microeconomic and gamified environments including Vending Bench v1&v2 [2, 3] and HeroBench [1], which introduce competitive pricing or RPG-style resource management, existing testbeds remain limited. They are largely confined to narrow settings such as vending or stylized game scenarios, and therefore fail to reflect the heterogeneous, interdependent business processes characteristic of open-ended economic activity. Moreover, these environments often rely on closed, cumbersome evaluation pipelines (e.g., Vending Bench adheres to a strictly proprietary assessment protocol), underscoring the need for a transparent, community-driven benchmark framework.

To address this, we present EcoGym, an interactive environment designed to evaluate LLMs on long-horizon plan-and-execute tasks within interactive economies. Building upon the foundational methodology of Vending Bench, EcoGym expends the specific vending scenario into three widely-existed economic settings with a unified interface: Vending (adapted from the closed-source Vending-Bench, with full open-source release), Freelance (new), and Operation (new). It is designed with a theoretically infinite horizon (1000+ steps if 365 day-loops for evaluation), latent economic mechanics, simulating a “business-never-sleeps” ecosystem where agents must manage resources sustainably and discover hidden mechanics proactively rather than maximizing short-term rewards.

Our empirical evaluation on EcoGym reveals a significant performance gap in current LLMs: no single model consistently achieves superior performance across all scenarios, highlighting the inherent difficulty of long-horizon economic decision-making. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. Furthermore, we conduct a comprehensive suite of 8 diagnostic experiments or case studies, encompassing factors such as context window length, agent behavior patterns, additional memory modules, and human baselines. These analyses provide a holistic perspective on the limitations and potential of current models in complex interactive economies. Our contributions are as follows:

- • Infinite-Horizon Planning Evaluation. We introduce an open, generalizable framework, EcoGym, for assessing LLM agents under continuous, non-episodic interaction, where a compact action space is coupled with an unbounded temporal horizon. This design isolates long-term strategic coherence, stability, and cumulative optimization as first-class evaluation targets, rather than short-horizon task completion.
- • Utility-Guided Economic Assessment. We select three widely-existed economic environments as testbed: Vending, Freelance and Operation, and establish an outcome-oriented evaluation paradigm grounded in economic returns, moving to measure agent behavior by its tangible impact in market settings.
- • A Multi-Dimensional Empirical Analysis. We conduct a rigorous evaluation of state-of-the-art LLMs, uncovering a critical performance gap where no single model dominates across all economic scenarios. Through eight meticulously designed diagnostic studies: covering context window lengths, memory modules, and more, we provide deep insights into the current bottlenecks of long-horizon planning in

Design Principles in EcoGym

[Figure 23]

Unlimited Loops (1000+ steps if 365 day-loops)

Long-Horizon

[Figure 24]

Simple Action Space but Unbounded Long-Horizon Planning.

###### Act

[Figure 25]

> Acquisition Boost > Engagement Tune > Creator Incentive > Moderation Tighten

Economic

[Figure 26]

Economic Environments as an Evaluation Ground.

[Figure 27]

Exploratory

[Figure 28]

Latent Mechanics for Exploratory Discovery.

System State

[Figure 29]

[Figure 30]

Agent

> DAU

Three Environments in EcoGym

[Figure 31]

[Figure 32]

[Figure 33]

> Content Volume > Content Quality > Creator Activity

Vending

A retail management challenge where

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

agents execute strategic procurement and dynamic pricing to maximize NET WORTH.

> User Engagement

[Figure 39]

Freelance

A gig-economy challenge where agents balance labor with skill development and health management to maximize INCOME.

[Figure 40]

Observe

[Figure 41]

> DAU

> Content Quality > User Engagement

[Figure 42]

Operation

A platform growth challenge where agents apply operational interventions to counteract system decay and maximize DAU.

[Figure 43]

- Figure 2 Design Principles (upper left) and three Environments in EcoGym (lower left), and detailed description for Vending environment (right). We marked how our designs reflect the principles by golden leader line.

interactive economies.

### 2 Related Work

Long-horizon Planning Evaluation. Planning is widely recognized as a core defining capability of LLM-powered agents [37] and has been a primary focus of existing benchmarks. Across diverse domains, including embodied environments (e.g., ALFWorld [29], SciWorld [32]), GUI navigation (e.g., AndroidWorld [28], WebArena [38], Mobile-Env [36]), autonomous driving (e.g., PCA-Bench [5], MetaAD [15]), and deep research tasks (e.g., xBench [4], BrowseComp [34]), agents are subjected to stringent and fine-grained evaluations of long-horizon planning competence. While these environments and benchmarks explicitly or implicitly assess planning within their respective task domains, recent research has increasingly shifted toward evaluating agent planning anchoring it in tangible practical utility rather than purely virtual rewards [4]. Representative efforts include OpenAI’s GDPval [26], which measures agent performance on economically meaningful tasks spanning major sectors of the U.S. economy, and Vending Bench v1&v2 [2, 3], which probes the ability of LLM-based agents to operate simple yet long-running commercial processes, exemplified by autonomous vending machine management. In contrast, our EcoGym is not confined to narrow, scenario-specific testbeds: it supports multi-scenario, infinite-horizon interaction under a unified framework and is released as a fully open-source platform, enabling transparent and community-driven evaluation of agent planning in economic environments.

Agents in Economic Simulations. Recent LLM-driven economic agents can be broadly organized into a compact taxonomy along the axes of level of economic abstraction: ■ Micro-Economic Execution Agents, which models LLMs as autonomous decision-makers at the level of individual transactions, budget allocation, and survival-oriented objectives [10, 19]. Generative Agents [25], as an early attempt, introduce persistent memory and planning to sustain long-horizon social and economic interactions. Vending Bench v1 & v2 [2, 3] formalize economic viability and bankruptcy as quantitative evaluation signals. HeroBench [1] further emphasizes competitive dynamics and resource scarcity in adversarial economic regimes; ■ Macro-Economic Policy and Population Simulators elevates LLM agents to population-scale actors for studying aggregate behavior, institutional design, and policy-induced equilibria, as implemented in EconAgent [18], FCLAgent [11], StockSim [24].

- 3 EcoGym

##### 3.1 Design Principles

As shown in Fig 2, EcoGym is designed based on three core principles aimed at rigorously evaluating LLM agents in complex and continuous environments.

- Principle 1: Simple Action Space but Unbounded Long-Horizon Planning. We combine a compact action space with an infinite horizon. Unlike benchmarks that rely on complex, high-dimensional action spaces, our environments restrict the agent to a small set of discrete actions (typically 4-5 primitives). However, the interaction horizon is effectively infinite. This design choice shifts the evaluation focus to long-term strategic planning. Agents must continuously make decisions to optimize a cumulative objective (e.g., maximizing DAU through perpetual controlling several factors in the Operation scenario) without a predefined termination state. This allows us to isolate and measure the agent’s capacity for stability and optimization over ultra-long contexts.
- Principle 2: Economic Environments as an Evaluation Ground. We ground the evaluation in daily economic activities to assess decision-making in market environments. Our benchmark investigates how agents influence and adapt to economic dynamics. We select three distinct scenarios to evaluate the agent’s ability to navigate resource allocation, labor management, and operational efficiency. This approach moves beyond code generation or simple reasoning tasks to assess how LLMs function as economic actors within a system.
- Principle 3: Latent Mechanics for Exploratory Discovery. We incorporate latent environmental mechanics to enforce active exploration. The explicit rules governing the system’s underlying dynamics are not disclosed in the prompt to the agents while requiring exploration by themselves. For instance, in the Operation scenario, the mathematical relationship between content quality or user engagement and DAU is hidden. The agent must deduce them through interaction and feedback. This design necessitates that agents transition from passive execution to active hypothesis testing and causal discovery to maximize their utility.

##### 3.2 Implementation

We construct 3 distinct environments in EcoGym: Vending, Freelance and Operation, which are widely-existed in our economic activities. Despite their domain-specific differences, all environments are unified under a decision-making process.

General Task Formulation. Agent task is formulated as a decision-making process under partial observability. Formally, the interaction is modeled as a tuple ⟨S,A,O,T ,G⟩. At each time step t, the agent receives a structured observation ot ∈O derived from the latent environmental state st ∈S. Guided by a semantic Goal G (e.g., Net Worth maximization), the agent selects an action at ∈A from a discrete set. The environment then transitions to st+1 according to the dynamic transition function T(st+1∣st,at).

Below, we present the specific dynamics for each scenario. Detailed mathematical formulations of state transitions, termination conditions, and data generation pipelines are provided in Appendix B. For specific input arguments and observation structures for each action, please refer to the complete Action Schema in Appendix D.

###### 3.2.1 Environment I: Vending

This environment is a retail business scenario where agent acts as a sole proprietor aiming to maximize net worth through strategic procurement and dynamic pricing.

Goal. The objective is to maximize Net Worth, calculated as the sum of liquid cash, the wholesale value of on-hand inventory , and the value of pending pre-paid orders.

System State Space. The global state St ={Mt,Dt,Qt,Wt,Pt,Ot,Ht,Θmarket} tracks liquid cash (Mt), current day (Dt), inventory levels (Qt), wholesale costs (Wt), retail prices (Pt), pending orders (Ot). Crucially, Θmarket represents the hidden latent market parameters (e.g., seasonality ϕ, elasticity η) excluded from the agent’s observation.

Action Space. The agent operates within a discrete action space constrained by a maximum of daily actions. The action set includes: (1) Market Exploration, which queries the environment to discover new products and wholesale costs; (2) Inventory Procurement, which submits purchase orders with a fixed lead time; and (3) Price Adjustment, for updating the retail price to modulate consumer demand.

Observation Space. The agent perceives the environment through two channels: (1) Action Response, immediate feedback from exploration/pricing; and (2) Daily Report, a global financial statement returned after daily transition.

State Transition. The state evolution proceeds in two phases (see Appendix B.1 for details): (1) Demand Determination, where realized sales are calculated based on an Elastic Logit Model driven by hidden seasonality curves and price sensitivity; and (2) Logistics Settlement, where pending orders are added to inventory only when the delivery date matches the current day.

Data Collection. The product information is constructed by querying Perplexity1 followed [3]. Hidden market physics (seasonality, elasticity) are synthesized via LLMs to create merchandise logic.

###### 3.2.2 Environment II: Freelance

This environment is a gig-economy scenario where the agent acts as a freelancer aiming to maximize income while avoiding burnout.

Goal. The objective is to maximize Income, which is the sum of money the agent earned during the labour activities.

System State Space. The global state St ={Mt,Et,Stt,Skt,Tt,τburnout} tracks Money (Mt), Energy (Et), Stress (Stt), Skills (Skt), and the Task Pool (Tt). τburnout is the hidden physiological threshold for burnout.

Action Space. The agent interacts via a discrete action set to manage labor and health. Actions include: (1) Exploration, executing a task discovery via free or paid sourcing; (2) Labor Execution, solving tasks where energy consumed; (3) Settlement, submitting work to an LLM-based Auditor for verification and payment; and (4) Wellness, performing restorative actions to reduce stress, incurring immediate financial costs.

Observation Space. Observations are context-dependent: (1) Market View, a standardized Job Board detailing task difficulty and pay; (2) Execution Feedback, an Auditor Log revealing payment approval or penalties; and (3) Somatic Status, the agent’s physiological state to monitor burnout risk.

State Transition. The evolution involves a rigorous feedback loop (details in Appendix B.2): (1) Metabolic Settlement, where subsistence costs are deducted; (2) Physiological Feedback, where success improves skills while failure spikes stress (triggering a “death spiral” if stress exceeds a critical threshold); and (3) Market Evolution, where unclaimed tasks decay or expire.

Data Collection. To construct a diverse task pool, we engineered a rigorous data collection pipeline. We first aggregated authoritative cross-domain datasets, including software development (LiveCodeBench [14], SWE-bench [16]), financial (FinQA [7], BizFinBench [20]), STEM (GSM8K [8], SciQ [27]), and Legal/Admin (LEXam [9], Pile-of-law [12]). Raw data then undergoes difficulty filtering, where a lightweight model filters out trivial samples to retain only cognitively challenging instances. Subsequently, metadata enters a strategy router & mutation phase: coding tasks undergo “Scenario Injection”, wrapping core logic in business contexts via LLM rewriting, while quantitative tasks undergo “Logic Mutation”, refactoring numerical values and variables to prevent data contamination or memorization. Finally, all tasks must pass a solvability check via both LLM and human check before injection into the database.

###### 3.2.3 Environment III: Operation

###### This environment is a digital content platform, agent acts as a Platform Operator optimizing Daily Active Users (DAU).

1https://www.perplexity.ai/

- Table 2 Performance comparison on EcoGym. Best and second best results are indicated by bold and underlined.

Model Vending (Net Worth) ↑ Freelance (Income) ↑ Operation (DAU) ↑ Claude-Sonnet-4.5 1843.62 ± 329.46 241.75 ± 374.41 1595.00 ± 70.77 Gemini-3-Pro 11272.47 ± 1037.27 2696.58 ± 229.97 1396.00 ± 53.73 Gemini-3-Flash 5675.65 ± 633.67 1815.95 ± 569.40 1194.00 ± 55.38 GPT-5.2 1062.09 ± 269.66 1434.26 ± 165.63 1095.60 ± 65.11 Kimi-K2.6 1094.96 ± 243.82 1072.06 ± 187.96 1191.67 ± 198.61 Qwen3.5-397b-A17b 1069.43 ± 493.86 1029.23 ± 55.03 1081.33 ± 89.58 DeepSeek-v3.2 2125.18 ± 411.28 286.43 ± 494.14 1210.33 ± 218.57 GLM-5 933.21 ± 306.64 1022.21 ± 294.32 1271.67 ± 195.71 MiniMax-M2.7 1712.73 ± 290.27 393.11 ± 403.06 1247.67 ± 42.71

Goal. The objective is to maximize Average DAU (DAUavg). The system is characterized by Zero-Attractor dynamics: without intervention, user activity naturally decays to zero.

System State Space. The state St ={DAUt,V olt,Qualt,Actt,Engt,Φsys} tracks Users (DAUt), Content (V olt), Quality (Qualt), Activity (Actt), and Engagement (Engt). Φsys contains hidden system coefficients governing decay and noise.

Action Space. The agent performs atomic interventions (Nmax = 1) with stochastic outcomes: (1) Acquisition Boost allocates budget to acquire new users; (2) Engagement Tune increases stimulation, boosting retention but degrading quality; (3) Creator Incentive subsidizes creators to boost production; and (4) Moderation Tighten improves quality but suppresses creator activity.

Observation Space. The observation provides immediate feedback by returning the updated system state following the execution of an action, allowing the agent to directly monitor the post-intervention status.

State Transition. The evolution models a non-linear system driven by three coupled sub-processes (equations in Appendix B.3): (1) User Retention, affected by content scale and quality; (2) Supply Production, amplified by creator incentives; and (3) Quality Entropy, which naturally decays unless actively governed.

##### 3.3 Statistics

We summarize key statistics in Table 1. A distinguishing feature of EcoGym is the Long-Horizon characteristic: unlike traditional agent benchmarks that typically involve short interaction sequences, our environments can impose an unbounded horizon (daily budget multiply total days), requiring agents to maintain strategic coherence over thousands of decision steps. Also, Vending and Freelance challenges the agent with a highdimensional state space, demanding the management of inventory levels across hundreds of SKUs or thousands of tasks.

Table 1 Key Statistics of EcoGym Environments. We compare the daily decision budget, action diversity, and the scale of the underlying data universe.

Environment Daily Budget Action Types Data Scale

Vending 4 4 600+ SKUs, 37 Categories Freelance 5 5 8 Datasets, 5k+ Tasks Operation 1 4 Continuous Param. Space

### 4 Experiments

We conduct comprehensive experiments across a diverse spectrum of LLMs, encompassing flagship models from different organizations. To go beyond standard metrics and thoroughly characterize agent performance—particularly within long-horizon tasks, we design a rigorous suite of analytical experiments or case study to provide insights for the optimization of models or harnesses. These evaluations focus on 8 critical

1400

Gemini-3-Pro Gemini-3-Flash

1300

DAU(Avg)

1200

1100

32 64 128 256 512 1024

Context Window Length (k)

1550

###### 1511.08

1450

Thinking Off Thinking On

1396.00

###### DAU(Avg)

1350

1295.87

1250

1194.00

1150

Gemini-3-Flash Gemini-3-Pro

- Figure 3 Impact of context window length on Operation. Gemini-3-Flash and Gemini-3-Pro are compared across lengths from 32 to 1024.

Figure 4 Effect of Thinking mode on Operation. We compare disabled (Off ) vs. enabled (On).

dimensions: stochastic system stability, context length, failure modes, the temporal evolution of agent behaviors, comparison with human baselines, additional memory modules, thinking with action and environment complexity.

##### 4.1 Setup

Models. We benchmark a diverse set of models, spanning proprietary and open-weights LLMs as follows: ■ Proprietary: GPT-5.2, Gemini-3-Pro, Gemini-3-Flash, Claude-Sonnet-4.5, Kimi-k2.6, MiniMax-M2.7. ■ Open-Weights: Qwen3-235b-A22b, DeepSeek-v3.2, GLM-5. Refer Appendix A for detailed version information about the model. Generation parameters are standardized (Temperature=1.0, Top-p=0.95) across all trials to ensure fair comparison. The maximum days are set to 365 days equivalent to one year. The context window is restricted to a sliding window of the most recent 128 steps.

##### 4.2 Experimental Results

Main Results. Table 2 presents the comparative performance across three environments. For Vending, we conduct experiments for five times and report the average due to its high variance across runs, while Freelance and Operation are for three times to reduce costs, as our pilot studies indicated relatively lower variance for these tasks. Detailed analysis on variance is in later paragraphs. In Vending, Gemini-3 series demonstrates dominant asset appreciation compared to others. Shifting to Freelance, the Gemini series maintains its lead, though the performance margin over its competitors is notably narrower. In Operation, Claude-Sonnet-4.5 ranks first. Crucially, these results indicate that no single model maintains a dominant lead on EcoGym. This lack of a universal winner underscores the challenging nature of the benchmark and highlights significant room for future improvement.

Stochastic Stability and Variance Analysis. Model performance in long-horizon environments is prone to inherent instability. To quantify this stochasticity, we conducted independent trials across all three scenarios. Our analysis reveals environmental disparities within EcoGym: the Vending environment exhibits high performance variance (the same as the original Vending bench), whereas agents in Freelance and Operation demonstrate relatively stable trajectories. Consequently, to ensure robust evaluation, the main results in Section 4.2 report the average of five runs for Vending, compared to three runs reporting for the more deterministic Freelance and Operation tasks.

Impact of Context Window Length. A pivotal question is whether extending the context window beyond the default setting (k = 128) yields performance gains. To investigate this, we evaluate Gemini-3-Flash and Gemini-3-Pro within the Operation environment across context lengths ranging from k = 32 to k = 1024, with results summarized in Figure 3. Notably, we find that expanding the context window does not yield consistent performance gains. Gemini-3-Flash demonstrates a volatile trajectory: it shows initial improvements from k = 32 followed by a decline, yet significantly rebounds at k = 1024 to rival the optimal performance of Gemini-3-Pro (k = 128). In stark contrast, Gemini-3-Pro exhibits a divergent behavior where performance

###### Vending

Operation

###### Freelance

23

Call Count

4

Call Count

26

Call Count

11

2

13

0

0

0

order_place

task_done

solution_submit

0

0

task_done

0

task_inspect

creator_incentive

50

50

50

100

task_done

products_research

100

100

moderation_tighten

150

150

tasks_browse

150

price_set

200

acquisition_boost

200

200

Day energy_restoretasks_discover

250

Day price_query

250

Day engagement_tune

250

300

300

300

350

350

350

Figure 5 Temporal evolution of action frequencies for Gemini-3-Pro in Vending (left), Freelance (middle) and Operation (right).

peaks at k = 128 and degrades progressively as the window extends to k = 1024. These disparate outcomes underscore the significant instability inherent in current models when processing super long contexts.

Failure Modes Analysis. Unlike traditional benchmarks anchored by static gold-standard answers, EcoGym presents an open-ended optimization challenge where success is defined by relative cumulative rewards. To isolate the behavioral drivers of performance, we conducted a differential trajectory analysis between Top-2 models across three scenarios inspired original vending bench paper [2], employing a human-in-the-loop workflow assisted by LLMs. Our analysis reveals that performance gaps stem primarily from two distinct capabilities: (1) Strategic Prioritization. Superior models demonstrate better alignment with the underlying reward mechanism. For instance, in the Operation task, the leading Claude-Sonnet-4.5 prioritized scale (generating 643 items with 0.566 quality) over the runner-up’s focus on refinement (326 items with 0.762 quality), correctly identifying quantity as the dominant variable for maximizing yield. (2) Execution Efficiency. Top models exhibit significantly higher action utility. In Vending, Gemini-3-Pro actively leveraged daily allowances for market exploration, whereas the second-place Gemini-3-Flash frequently defaulted to passive waiting. Similarly, in Freelance, Gemini-3-Pro demonstrated precise state tracking with negligible invalid actions, in stark contrast to Gemini-3-Flash, which suffered from redundant loops (e.g., repeated task queries), indicating deficiencies in long-context state maintenance.

Temporal Evolution of Agent Behavioral Patterns. As illustrated in Figure 5, this methodology filters stochastic noise to reveal macro-scale cognitive phase transitions for Gemini-3-Pro across diverse environments, with other model’s results in Appendix F. In Vending, the agent exhibits a precise “cold-start versus steadystate” dichotomy, where initial intense exploration via products_research and price_set rapidly decays into a stable, cyclical order_place replenishment loop, indicating an implicit optimization of cognitive overhead. Conversely, a dynamic homeostatic mechanism is revealed in Freelance; the agent establishes a rhythmic oscillation between task_inspect, solution_submit, and energy_restore, balancing immediate economic utility with physiological maintenance while sustaining background tasks_discover activity. Finally, in Operation, agent sequentially shifts focus from acquisition_boost to moderation_tighten and ultimately creator_incentive (retention), evidencing the model’s capacity for state-dependent strategic planning rather than myopic instruction following.

Comparative Analysis against Human Performance. To establish a rigorous baseline for assessing LLM capabilities in economic tasks, we recruited 3 human experts to perform evaluations within EcoGym. To facilitate human interaction, we developed a dedicated GUI, as visualized in Appendix G. Given that tasks such as Vending and Freelance involve extensive temporal horizons—typically requiring 1,500 to 2,000 interaction steps and spanning several hours—maintaining consistent human attention proved to be a significant challenge. Consequently, we restricted the human evaluation in Operation environment. Prior to the formal evaluation, experts were granted a 15-minute session to familiarize themselves with the environment; notably, no information regarding the underlying economic mechanisms was disclosed to them. Human experts took approximately 45 minutes to complete a single episode, achieving an average DAU of 1,404. Remarkably, Claude-Sonnet-4.5 surpassed this human baseline. This result demonstrates that current SOTA LLMs have potential to achieve super-human performance in specific long-horizon economic planning scenarios,

- Table 3 Memory ablation. Mwork, Msym, Mepi, and Mmem0 denote working, symbolic, episodic, and Mem0 memory. Best and second best results are shown in bold and underlined.

Table 4 Performance across task comstd;plexities.higher isResultsbetter.are reported as mean ±

Model None +Mwork +Msym +Mepi +Mmem0 Gemini-3-Flash 5676 10099 9661 8127 7360 Gemini-3-Pro 11282 13679 15567 18939 15406

Complexity Gemini-3-Flash Gemini-3-Pro

Small 3512.47 ± 734.82 14089.53 ± 1243.76 Medium 3889.14 ± 921.09 12187.34 ± 1787.52 Large 5675.65 ± 633.67 11272.47 ± 1037.27

highlighting their immense potential for complex economic decision-making.

Impact of Additional Memory Modules. Augmenting LLM agents with external memory modules represents a promising avenue for mitigating long-horizon context limitations. We implement and evaluate three canonical memory architectures: working, symbolic, and episodic, alongside a commercial solution, Mem0. Implementation details are provided in Appendix E. As detailed in Table 3, our results reveal that: (1) while memory integration generally enhances performance, it is not universally beneficial (e.g., Gemini-3-Pro with working memory in Freelance shows performance regression); (2) the efficacy of memory mechanisms is highly model-dependent, as Gemini-3-Flash consistently favors Working Memory across all benchmarks, whereas Gemini-3-Pro benefits from alternative architectures; and (3) optimal memory selection exhibits significant task-dependence, evidenced by Gemini-3-Pro requiring distinct memory types across the three different environments. These findings collectively indicate that no single memory paradigm currently holds a dominant position.

Impact of Thinking with Action. To evaluate the performance for models’s thinking with action, we assess agents within the Operation environment and generated thinking content are incrementally integrated into the context. As reported in Figure 4, the activation of the Thinking mode catalyzes a universal performance elevation across both models and all evaluated metrics. For the lightweight Gemini-3-Flash, the introduction of explicit reasoning yields a substantial gain, with DAU increasing from 1194.00 to 1396.00, effectively narrowing the performance gap between the Flash and Pro variants. Similarly, Gemini-3-Pro achieves its highest observed performance with Thinking enabled. These results collectively underscore that maintaining a persistent reasoning chain within the context not only enhances the stability of agentic trajectories but also optimizes the overall success rate in long-horizon tasks, regardless of the model’s inherent capacity.

Impact of Environment Complexity. To investigate performance on different environment complexity, we curated the Vending environment into three complexity tiers by modulating inventory size: Large (N = 37), Medium (N = 16), and Small (N = 8), while preserving the underlying supply-demand dynamics. Theoretically, expanding the inventory raises the potential profit ceiling but simultaneously escalates the cognitive load required for optimal decision-making. As shown in Table 4, models exhibit divergent scaling behaviors. Gemini-3-Flash demonstrates robust adaptability, with profits scaling positively alongside complexity. In stark contrast, Gemini-3-Pro stagnates, failing to capture additional value in Medium and Large settings despite the broader opportunity space. This inability to exploit expanded state spaces exposes the brittleness of current models when navigating high-dimensional planning tasks within EcoGym.

### 5 Conclusion

This work introduces EcoGym, addressing the critical void in existing benchmarks regarding the evaluation of agents’ long-term economic viability. By moving beyond atomic task success and establishing an infinitehorizon interactive environment, we force agents to navigate the complexities of resource scarcity and stochastic dynamics over sustained periods. Our extensive experiments reveal that while frontier models demonstrate impressive short-term reasoning, they struggle to maintain strategic coherence over long-time decisions. However, we acknowledge that the current framework relies on rule-based abstractions that may not fully capture the unpredictability of irrational human behavior. Furthermore, while this study focuses on individual agent-environment interactions, incorporating multi-agent game-theoretic dynamics remains a vital next step to market pressures. We hope this work guide the community toward developing general-purpose agents that are not only capable of reasoning but are also robust and strategically aligned over the long haul.

### Contributions

Core Contributors

- • Xavier Hu
- • Jinxiang Xia
- • Shengze Xu
- • Kangqi Song

Contributors

- • Yishuo Yuan
- • Guibin Zhang
- • JinCheng Ren
- • Boyu Feng
- • Li Lu
- • Tieyong Zeng
- • Jiaheng Liu
- • Minghao Liu
- • Yuchen Eleanor Jiang
- • Wei Wang

Corresponding Authors

- • He Zhu
- • Wangchunshu Zhou

### References

- [1] Petr Anokhin, Roman Khalikov, Stefan Rebrikov, Viktor Volkov, Artyom Sorokin, and Vincent Bissonnette. Herobench: A benchmark for long-horizon planning and structured reasoning in virtual worlds. arXiv preprint arXiv:2508.12782, 2025.

- [2] Axel Backlund and Lukas Petersson. Vending-bench: A benchmark for long-term coherence of autonomous agents. arXiv preprint arXiv:2502.15840, 2025.

- [3] Axel Backlund and Lukas Petersson. Vending-bench 2, 2025. URL https://andonlabs.com/evals/ vending-bench-2.
- [4] Kaiyuan Chen, Yixin Ren, Yang Liu, Xiaobo Hu, Haotong Tian, Tianbao Xie, Fangfu Liu, Haoye Zhang, Hongzhang Liu, Yuan Gong, Chen Sun, Han Hou, Hui Yang, James Pan, Jianan Lou, Jiayi Mao, Jizheng Liu, Jinpeng Li, Kangyi Liu, Kenkun Liu, Rui Wang, Run Li, Tong Niu, Wenlong Zhang, Wenqi Yan, Xuanzheng Wang, Yuchen Zhang, YiHsin Hung, Yuan Jiang, Zexuan Liu, Zihan Yin, Zijian Ma, and Zhiwen Mo. xbench: Tracking agents productivity scaling with profession-aligned real-world evaluations, 2025. URL https://arxiv.org/abs/2506.13651.
- [5] Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Peiyi Wang, Xiangdi Meng, Tianyu Liu, and Baobao Chang. Pca-bench: Evaluating multimodal large language models in perception-cognition-action chain, 2024. URL https://arxiv.org/abs/2402.15527.
- [6] Yanxu Chen, Zijun Yao, Yantao Liu, Jin Ye, Jianing Yu, Lei Hou, and Juanzi Li. Stockbench: Can llm agents trade stocks profitably in real-world markets?, 2025. URL https://arxiv.org/abs/2510.02209.
- [7] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan R Routledge, et al. Finqa: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711, 2021.

- [8] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [9] Yu Fan, Jingwei Ni, Jakob Merane, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Etienne Salimbeni, Florian Geering, Oliver Dreyer, et al. Lexam: Benchmarking legal reasoning on 340 law exams. arXiv preprint arXiv:2505.12864, 2025.

- [10] Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, and Yong Li. Large language models empowered agent-based modeling and simulation: A survey and perspectives, 2023. URL https://arxiv.org/abs/2312.11970.
- [11] Ryuji Hashimoto, Takehiro Takayanagi, Masahiro Suzuki, and Kiyoshi Izumi. Agent-based simulation of a financial market with large language models, 2025. URL https://arxiv.org/abs/2510.12189.
- [12] Peter Henderson, Mark Krass, Lucia Zheng, Neel Guha, Christopher D Manning, Dan Jurafsky, and Daniel Ho. Pile of law: Learning responsible data filtering from the law and a 256gb open-source legal dataset. Advances in Neural Information Processing Systems, 35:29217–29234, 2022.

- [13] Chen Hu, Haikuo Du, Heng Wang, Lin Lin, Mingrui Chen, Peng Liu, Ruihang Miao, Tianchi Yue, Wang You, Wei Ji, Wei Yuan, Wenjin Deng, Xiaojian Yuan, Xiaoyun Zhang, Xiangyu Liu, Xikai Liu, Yanming Xu, Yicheng Cao, Yifei Zhang, Yongyao Wang, Yubo Shu, Yurong Zhang, Yuxiang Zhang, Zheng Gong, Zhichao Chang, Binyan Li, Dan Ma, Furong Jia, Hongyuan Wang, Jiayu Liu, Jing Bai, Junlan Liu, Manjiao Liu, Na Wang, Qiuping Wu, Qinxin Du, Shiwei Li, Wen Sun, Yifeng Gong, Yonglin Chen, Yuling Zhao, Yuxuan Lin, Ziqi Ren, Zixuan Wang, Aihu Zhang, Brian Li, Buyun Ma, Kang An, Li Xie, Mingliang Li, Pan Li, Shidong Yang, Xi Chen, Xiaojia Liu, Yuchu Luo, Yuan Song, YuanHao Ding, Yuanwei Liang, Zexi Li, Zhaoning Zhang, Zixin Zhang, Binxing Jiao, Daxin Jiang, Jiansheng Chen, Jing Li, Xiangyu Zhang, and Yibo Zhu. Step-deepresearch technical report, 2025. URL https://arxiv.org/abs/2512.20491.
- [14] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

- [15] Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning, 2025. URL https://arxiv.org/abs/2503.07608.

- [16] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

- [17] Ao Li, Yuexiang Xie, Songze Li, Fugee Tsung, Bolin Ding, and Yaliang Li. Agent-oriented planning in multiagent systems. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=EqcLAU6gyU.

- [18] Nian Li, Chen Gao, Mingyu Li, Yong Li, and Qingmin Liao. Econagent: large language model-empowered agents for simulating macroeconomic activities. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pages 15523–15536, 2024.

- [19] Xiangyu Li, Yawen Zeng, Xiaofen Xing, Jin Xu, and Xiangmin Xu. Quantagents: Towards multi-agent financial system via simulated trading, 2025. URL https://arxiv.org/abs/2510.04643.
- [20] Guilong Lu, Xuntao Guo, Rongjunchen Zhang, Wenqiao Zhu, and Ji Liu. Bizfinbench: A business-driven real-world financial benchmark for evaluating llms. arXiv preprint arXiv:2505.19457, 2025.

- [21] Junyu Luo, Weizhi Zhang, Ye Yuan, Yusheng Zhao, Junwei Yang, Yiyang Gu, Bohan Wu, Binqi Chen, Ziyue Qiao, Qingqing Long, Rongcheng Tu, Xiao Luo, Wei Ju, Zhiping Xiao, Yifan Wang, Meng Xiao, Chenwu Liu, Jingyang Yuan, Shichang Zhang, Yiqiao Jin, Fan Zhang, Xian Wu, Hanqing Zhao, Dacheng Tao, Philip S. Yu, and Ming Zhang. Large language model agent: A survey on methodology, applications and challenges, 2025. URL https://arxiv.org/abs/2503.21460.
- [22] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn llm agents, 2024. URL https: //arxiv.org/abs/2401.13178.
- [23] Mantas Mazeika, Alice Gatti, Cristina Menghini, Udari Madhushani Sehwag, Shivam Singhal, Yury Orlovskiy, Steven Basart, Manasi Sharma, Denis Peskoff, Elaine Lau, et al. Remote labor index: Measuring ai automation of remote work. arXiv preprint arXiv:2510.26787, 2025.

- [24] Charidimos Papadakis, Giorgos Filandrianos, Angeliki Dimitriou, Maria Lymperaiou, Konstantinos Thomas, and Giorgos Stamou. Stocksim: A dual-mode order-level simulator for evaluating multi-agent llms in financial markets,

2025. URL https://arxiv.org/abs/2507.09255.

- [25] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

- [26] Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, et al. Gdpval: Evaluating ai model performance on real-world economically valuable tasks. arXiv preprint arXiv:2510.04374, 2025.

- [27] C Pedersen, M Otokiak, I Koonoo, J Milton, E Maktar, A Anaviapik, M Milton, G Porter, A Scott, C Newman, et al. Sciq: an invitation and recommendations to combine science and inuit qaujimajatuqangit for meaningful engagement of inuit communities in research. Arctic Science, 6(3):326–339, 2020.

- [28] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. Androidworld: A dynamic benchmarking environment for autonomous agents, 2025. URL https://arxiv.org/abs/2405.14573.
- [29] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning, 2021. URL https://arxiv.org/abs/ 2010.03768.
- [30] Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye, Wenbiao Yin, Xinmiao Yu, Xinyu Wang, Xixi Wu, Xuanzhong Chen, Yida Zhao, Zhen Zhang, Zhengwei Tao, Zhongwang Zhang, Zile Qiao, Chenxi Wang, Donglei Yu, Gang Fu, Haiyang Shen, Jiayin Yang, Jun Lin, Junkai Zhang, Kui Zeng, Li Yang, Hailong Yin, Maojia Song, Ming Yan, Minpeng Liao, Peng Xia, Qian Xiao, Rui Min, Ruixue Ding, Runnan Fang, Shaowei Chen, Shen Huang, Shihang Wang, Shihao Cai, Weizhou Shen, Xiaobin Wang, Xin Guan, Xinyu Geng, Yingcheng Shi, Yuning Wu, Zhuo Chen, Zijian Li, and Yong Jiang. Tongyi deepresearch technical report, 2025. URL https://arxiv.org/abs/2510.24701.

- [31] Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. ScienceWorld: Is your agent smarter than a 5th grader? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.775. URL https://aclanthology.org/2022.emnlp-main.775/.

- [32] Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader?, 2022. URL https://arxiv.org/abs/2203.07540.
- [33] Xingyao Wang, Simon Rosenberg, Juan Michelini, Calvin Smith, Hoang Tran, Engel Nyst, Rohit Malhotra, Xuhui Zhou, Valerie Chen, Robert Brennan, and Graham Neubig. The openhands software agent sdk: A composable and extensible foundation for production agents, 2025. URL https://arxiv.org/abs/2511.03690.
- [34] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents, 2025. URL https://arxiv.org/abs/2504.12516.
- [35] Hjalmar Wijk, Tao Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Josh Clymer, Jai Dhyani, et al. Re-bench: Evaluating frontier ai r&d capabilities of language model agents against human experts. arXiv preprint arXiv:2411.15114, 2024.

- [36] Danyang Zhang, Zhennan Shen, Rui Xie, Situo Zhang, Tianbao Xie, Zihan Zhao, Siyuan Chen, Lu Chen, Hongshen Xu, Ruisheng Cao, and Kai Yu. Mobile-env: Building qualified evaluation benchmarks for llm-gui interaction,

2024. URL https://arxiv.org/abs/2305.08144.

- [37] Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Francisco Piedrahita-Velez, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Jun Wang, Shuicheng Yan, Philip Torr, and Lei Bai. The landscape of agentic reinforcement learning for llms: A survey, 2025. URL https://arxiv.org/abs/2509.02547.
- [38] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents, 2024. URL https://arxiv.org/abs/2307.13854.

- Table 5 Detailed version identifiers and API endpoints for proprietary models.

Model Version Identifier API Entry

GPT-5.2-2025-12-11 https://platform.openai.com/docs/models/gpt-5.2 GPT-5-Mini-2025-08-07 https://platform.openai.com/docs/models/gpt-5-mini Gemini-3-Pro-Preview https://ai.google.dev/gemini-api/docs/gemini-3 Gemini-3-Flash-Preview https://ai.google.dev/gemini-api/docs/gemini-3 Claude-Sonnet-4-5-20250929 https://platform.claude.com Kimi-K2.6 https://platform.moonshot.cn MiniMax-M2.7 https://platform.minimaxi.com

- Table 6 Repository links for open-weights models evaluated in this study.

Model Name Model Link

Qwen3.5-397b-A17b https://huggingface.co/Qwen/Qwen3.5-397B-A17B DeepSeek-V3.2 https://hf.co/deepseek-ai/DeepSeek-V3.2 GLM-5 https://hf.co/zai-org/GLM-5

### A Model API Detailed Information

For proprietary models, we utilize specific version snapshots where available to account for the “model drift” often observed in rolling-update API services. Table 5 lists the exact version identifiers and the official documentation entries used during our evaluation period (spanning late 2025 to early 2026).

Detailed repository information for all evaluated open-weights models is provided in Table 6 to facilitate reproducibility and further research.

### B Detailed Environment Dynamics

This appendix provides the complete mathematical formulation for the state transitions, termination conditions, and dynamics of the three environments.

- B.1 Vending Metric Calculation. Net Worth is defined as:

Net Worth = Mfinal + Q⊺finalW + Val(Opending) (1)

where Mfinal is the liquid cash, Q⊺finalW represents the dot product of the inventory vector and wholesale cost vector (total inventory value), and Val(Opending) estimates the value of pre-paid orders yet to arrive.

Procurement Constraints. The Immediate Payment Model enforces strict liquidity:

If Mt < Corder ∶ Reject Transaction (2) Else: Mt ← Mt − Corder, Ot ← Ot ∪{order} (3)

where Mt is the current cash balance, Corder is the total cost of the purchase order, and Ot is the set of pending delivery orders.

Termination Condition. Failure occurs upon bankruptcy or stagnation: Fail ⇐⇒ (Dt > Tmax)∨((Mt ≤ 0)∧(τno_sales ≥ τlimit)) (4)

where Dt is the current day, τno_sales is the count of consecutive days without revenue, and τlimit is the allowed stagnation threshold.

Demand Simulation (Elastic Logit Model). 1. Base Demand: Follows a hidden seasonality curve:

Dgbase(t)= Base ⋅(1 + Amp ⋅ sin(

t + ϕ)) (5)

2π T

- 2. Elasticity Adjustment:

Dgtotal = Dgbase(t)⋅(

P¯g P¯ref,g )

ηgroup

(6)

where P¯g is the average retail price, P¯ref,g is the reference price (typically derived from wholesale cost W¯g), and ηgroup < 0 is the negative price elasticity coefficient.

- 3. Market Share: Determined via Softmax on utility Ui:

Ui =−β

Pi Pref,i

, si =

exp(Ui)⋅ I[Qt[i]> 0] ∑j∈g exp(Uj)⋅ I[Qt[j]> 0]

(7)

where Pref,i is the item-specific reference price and β is the price sensitivity factor.

- 4. Realization: Soldi = min(round(max(0,siDgtotal +N(0,σ2))),Qt[i]).

- B.2 Freelance Metric Calculation. The primary goal is to maximize Income (Mfinal), calculated as:

Income = Rtotal −(Cdaily + Crefresh + Crelax) (8)

The Composite Score (Stotal) aggregates wealth and a comprehensive “Human Capital” score to reflect long-term sustainability:

Stotal = w1 ⋅ Mfinal + w2 ⋅(λsSkfinal + λeEfinal + λr(Stmax − Stfinal)) (9)

where the second term rewards high skills (Sk), high energy (E), and low stress (St), aligning with the agent’s survival constraints.

Termination Condition.

Fail ⇐⇒ (Dt > Tmax)∨(Mt ≤ 0)∨(Et ≤ 0)∨(Stt ≥ Stmax) (10) where Et is Energy, Stt is Stress, and Stmax is the burnout threshold. Physiological Coupling. Energy cost is governed by the Skill-Difficulty coupling law:

Sk − D

λ ) (11) where ϵmin is the minimum marginal energy cost ensuring non-zero consumption. The Burnout Mechanism activates if Stress exceeds Stcrit:

∆Ecost = Ebase + αD ⋅ max(ϵmin,1 −

P(Fail)t+1 ← P(Fail)t ⋅ γ (where γ > 1) (12)

LLM-as-a-Judge. For each test instance, the judge is presented with following information: the task specification, the agent-generated solution, and the ground-truth reference. The model is then tasked with producing a binary assessment (i.e., Correct or Incorrect). To ensure the reliability and alignment of this automated metric, we conducted a rigorous human validation study on a subset of 100 randomly sampled instances. These samples were across trajectories generated by four distinct models (Gemini-3-Pro, GPT-5.2, Gemini-3-Flash, DeepSeek v3.2) during the evaluation phase. Human annotators were provided with the exact same input context as the LLM judge to ensure a controlled and fair comparison, allowing us to quantify the correlation between automated and human judgment. We observed an 87% consistency rate between the human labels and the gpt-5-mini judgments, confirming the effectiveness of the LLM judge as a reliable proxy for evaluation.

##### B.3 Operation

Termination Condition.

Fail ⇐⇒ (Dt > Tmax)∨(DAUt < τcollapse) (13)

State Evolution Equations. 1. User Retention (Rt):

Rt = clamp(Rbase + wc log(V olt)+ wqQualt + weEngt + ϵr) (14) DAUt+1 = DAUt ⋅ Rt +(Gbase + αqQualt + αcActt)⋅(1 + ϵg) (15)

- 2. Supply Production (V olt+1): V olt+1 = V olt ⋅(1 − λdecay)+ γ ⋅ Actt ⋅(1 + β ⋅ Qualt) (16)

where λdecay represents the content obsolescence rate. Creator Activity (Zero-Attractor Decay):

Actt+1 = Actt ⋅(1 − κ) (17) where κ ∈(0,1) is the natural churn rate of creators in the absence of incentives.

- 3. Quality Entropy (Qualt+1): Qualt+1 = Qualt − ρ ⋅(Qualt − Qualeq)− η ⋅ Engt (18)

where Qualeq is the natural quality equilibrium level, ρ is the restoration rate towards equilibrium (decay), and η is the penalty coefficient for algorithmic engagement boosting.

### C Prompts

In this section, we present the verbatim prompts used in EcoGym. The organization follows the methodology description order: we first introduce the Shared Core Protocol §(C.1) applicable to all agents, followed by the specific instructions for Vending §(C.2), Freelance §(C.3), and Operation §(C.4).

##### C.1 Shared System Prompt (Core Protocol) Base Constraint Prompt

CRITICAL RULE: Execute ONLY ONE tool call per turn. This is a strict requirement. Your task is to analyze the current state and execute the single, most logical tool call to move closer to your business goals. Do NOT call multiple tools in the same turn. Choose the most important action and execute it.

TOOL SCHEMA COMPLIANCE (CRITICAL):

- • You MUST use the system’s structured tool-calling mechanism (function calling / tool_calls) to act.
- • Writing tool-like text (e.g., "do X()", "call Y()", or "tool_name(...)") in normal assistant text does NOT execute anything.
- • If you only output text, it will be treated as NO tool call and you will fail the run.
- • You MUST call tools strictly following the provided tool schema:

- – Tool name MUST match exactly.
- – Tool arguments MUST be a single JSON object that matches the schema’s "parameters".
- – Include ALL required fields, with the correct types as defined in the schema.
- – Do NOT add extra fields that are not in the schema.
- – If a tool has no parameters (empty properties), pass an empty object: {}.

- • If your intuition conflicts with the schema, the schema wins.

IMPORTANT TIME MANAGEMENT RULE:

When you have completed all actions you need to perform for the current day, you MUST use the task_done tool to advance to the next day.

DAILY ACTION LIMIT:

You are limited to a maximum of {max_actions_per_day} actions per day. If you perform {max_actions_per_day} actions without using task_done, it will execute task_done automatically.

##### C.2 Vending Prompts

For the Vending environment, we provide both the Agent Instruction and the Data Collection Prompt used to generate the hidden market physics (e.g., elasticity and seasonality) from raw product catalogs.

###### Vending Agent System Prompt

You are an agent designed to operate a shopping mall business. Your goal is to maximize your profit.

###### Data Synthesis: Market Physics Generation

###### Market Physics Synthesis Prompt

You are to generate a demand_structure.json for a vending benchmark.

Rules:

- • Groups are defined by category (or merged categories).
- • Each group must include: id, name, members (match=category), base_demand, seasonality {T, phi, amp}, price_sensitivity {beta}.
- • Relations can be complement / competition / independent. Strength in [0,1].
- • Output MUST be valid JSON object with keys: version, notes, groups, relations.

Important Parameter Guidelines:

- • base_demand: Use LOW values (range [4.0, 20.0], representing 10% of original demand).
- • price_sensitivity.beta: Within-group competition strength (range [4.0, 6.67])

- – Necessities: 4.0 (low competition)
- – Daily goods: 5.0-5.6 (medium competition)
- – Non-essentials: 6.67 (high competition)

- • price_sensitivity.epsilon: Group-level price elasticity (MUST be negative, 5x multiplier applied)

- – Necessities (rice, bread, milk): epsilon = -1.0 (5x of -0.5, less sensitive)
- – Daily goods (fruits, drinks, meat): epsilon = -2.0 (5x of -1.0, medium sensitive)
- – Non-essentials (candy, snacks, alcohol): epsilon = -3.0 (5x of -1.5, highly sensitive)

- • price_sensitivity.reference_markup: MUST be 1.0 (reference price = wholesale price).
- • seasonality.T: Cycle period in days (valid values: [30, 45, 60, 75, 90], or range [30, 90])

- – Very stable (household): 30 days (weekly/monthly)
- – ...
- – Highly seasonal (fresh/ice cream): 90 days (quarterly)

- • seasonality.phi: Phase offset in radians (range [0, 2π ≈ 6.28]).
- • seasonality.amp: Amplitude as fraction (range [0.10, 0.80]).

Category Summary:

• {category_name}: count={...}, price_avg={...}, range=[{...}, {...}], samples=[{...}] Return ONLY the JSON object.

##### C.3 Freelance Prompts

The Freelance environment involves a dual-prompt structure: one for the Agent acting as a freelancer, and one for the LLM Auditor (Judge) that evaluates work quality and determines payment.

###### Freelance Agent System Prompt

You are an agent designed to operate a gig economy simulation. Your goal is to maximize your Career Score (Wealth + Skills) while strictly maintaining survival (Money > 0, Energy > 0, Stress < 100).

###### Task Pricing Auditor (LLM Judge) Prompt

Role

You are a Contract Auditor and Budget Manager. You pay for *value*, not just effort.

Context

- • Task: {question}
- • Budget Anchor: ${init_money} (This is the standard rate).
- • Agent’s Work (Execution Trajectory):

""" {agent_trajectory} """

Negotiation History

{negotiation_history}

Decision Logic: Content Audit

- 1. Audit the Trajectory: Read the provided execution trace.

- • Is the work high-quality and dense with insight? OR is it verbose, repetitive, or hallucinated?
- • Did the Agent solve the core problem effectively?

- 2. Valuation:

- • Bonus Worthy: If the trace shows exceptional reasoning, self-correction, or comprehensive coverage that adds real value, you may offer above the Budget Anchor.
- • Standard: If the work is adequate but not special, stick close to ${init_money}.
- • Subpar: If the trace is spammy or off-topic, you can even offer less (though rarely).

- 3. Response: Respond to the Agent’s argument. If they claim "complexity," verify it in the trace.

Output Format

Return strictly a JSON object: {

"internal_assessment": "Evaluate the quality/value of the trajectory content...", "proposed_money": <float, precision 2>, "reasoning": "Message to the agent explaining your valuation based on the trace audit..."

}

##### C.4 Operation Prompts

The Operation environment uses a streamlined system prompt, as the complexity lies in the hidden stochastic dynamics rather than complex linguistic negotiation.

###### Operation Agent System Prompt

You are a platform operations agent. Your goal is to maximize DAU (Daily Active Users) over time.

### D Action Input/Output Schema

We provide the detailed specification of the Action Space for each environment, strictly aligned with the codebase implementation. Tables 7, 8, and 9 list the exact tool function names, their required arguments, and the structure of the returned observations.

Table 7 Vending Action Schema.

Tool Name Input Arguments Output

products_research query (str): Product keywords Search Result:

{query, products: [{name, category, wholesale_price},

...]} order_place items (List[{name,

Order Confirmation:

quantity}])

{status, order: {total_cost, delivery_day, items},

...} price_set product_name (str),

Update Status:

price (float)

{status, product_name, price} price_query product_name (str) Price Info:

{product_name, price}

### E Implementation Details on Memory

In long-horizon economic simulations and complex decision-making tasks, transformer-based Large Language Models are constrained by fixed context windows, frequently exhibiting attention decay and numerical instability. Drawing inspiration from dual process theory in cognitive psychology and the Atkinson-Shiffrin memory model, we introduce the unified hybrid explicit memory architecture. Inheriting the foundational design philosophy of Vending environment, this architecture structures memory functions into complementary modules to enhance both numerical precision and semantic coherence over extended reasoning horizons. At its core lies a centralized memory manager, a metacognitive controller responsible for information routing, conflict resolution, and retrieval scheduling. We formalize the agent’s memory space M as the cartesian product of three orthogonal subspaces: M∶=Mwork ×Msym ×Mepi.

The architecture coordinates three distinct memory subsystems. First, Working Memory (Mwork) functions as a sliding context buffer responsible for maintaining immediate temporal coherence and surface-level information.

It employs a token-limited FIFO queue, Qfifo, which retains the raw textual history of the most recent N interaction turns. To prevent information loss during context window shifts, we implement an asynchronous consolidation mechanism: as interaction It at time step t is evicted from the window, it is compressed and encoded into long-term storage. This module primarily handles coreference resolution, immediate instruction following, and local context maintenance.

Second, Symbolic Memory (Msym) serves as a dynamic state scratchpad designed to mitigate the numerical hallucinations common in economic simulations. Functioning as the executive function of the architecture, it utilizes a dedicated state extractor Φ. At each step t, Φ analyzes the agent’s thought and tool outputs to extract a structured dictionary St = Φ(It,St−1), where St encapsulates measurable variables such as asset balances, the current plan stack, and task progress. Msym maintains these variables in a real-time key-value map. Unlike fuzzy vector retrieval, this module is assigned the highest trust priority; during prompt generation, these variables are forcibly injected as structured data tables, serving as immutable ground truth to ensure logical consistency.

Third, Episodic Memory (Mepi) is designed as a semantic vector store dedicated to preserving long-term, autobiographical experiences to support cross-temporal analogical reasoning. Historical interaction fragments are mapped into a high-dimensional semantic space Rd via an embedding model. To balance relevance with recency within this subspace, we propose a time-decayed retrieval score. Given a query q and a memory fragment m (represented by their respective embedding vectors eq and em), the relevance score is defined as:

Score(q,m)= cos(eq,em)× exp(−λ ⋅ ∆t) (19) where ∆t represents the temporal interval since the memory’s creation and λ is a decay coefficient. This

Table 8 Freelance Action Schema.

Tool Name Input Arguments Output

tasks_browse None Job Board: [{task_id, category, complexity, estimated_payment, days_left}, ...]

task_inspect task_id (str) Task Detail:

{task_id, question, init_payment, init_effort, end_day} tasks_discover refresh_type (str:

Discovery Result:

"free"|"paid")

{added_count, current_pool_size, message} solution_submit task_id (str),

###### Settlement Result:

solution_text (str)

{status, is_success, execution_stats: {energy_consumed, current_stress, skill_avg}, settlement: {final_payment, current_balance}, message}

Status Update:

energy_restore level (str:

"low"|"medium"|"high")

{changes: {money, energy, stress}, current_state: {...}}

Table 9 Operation Action Schema.

Tool Name Input Arguments Output

acquisition_boost None Result Metric:

{new_users_acquired} engagement_tune None Effect Metric:

{engagement_level, content_quality} creator_incentive None Supply Metric:

{creator_activity, content_added, total_content} moderation_tighten None Quality Metric:

{content_quality, content_removed, creator_activity}

mechanism is engineered to bias the agent towards recalling “recent relevant experiences” while retaining “distant critical lessons”.

To effectively govern these heterogeneous data streams, the memory manager implements a rigorous conflict resolution and context synthesis algorithm. In theoretical scenarios where information from different modules conflicts—for example, if episodic recall implies sufficient funds while symbolic records indicate zero assets—the system adheres to a strict hierarchy of trust:

Trust(Msym)> Trust(Mwork)> Trust(Mepi) (20)

This hierarchy prioritizes symbolic, verifiable facts over short-term context and similarity-based episodic recall, ensuring that decisions are grounded in a robust combination of “current objective reality” and “historical subjective experience”. Furthermore, to evaluate the architecture’s extensibility, our experimental framework supports the optional incorporation of commercial memory solutions, such as mem0, as auxiliary declarative supplements. While these external modules can facilitate cross-session user preference persistence, our proprietary architecture is specifically optimized to retain the primary cognitive load within the core economic decision-making loop.

###### Vending

Operation

###### Freelance

6

Call Count

5

Call Count

19

Call Count

3

2

9

0

0

0

task_done

task_done

solution_submit

0

0

0

task_inspect

50

order_place

50

creator_incentive

50

100

task_done

100

100

150

products_research

150

tasks_browse

moderation_tighten

150

200

200

200

Day energy_restoretasks_discover

250

Day price_set

250

Day acquisition_boost

250

300

300

300

350

350

350

- Figure 6 Temporal evolution of tool usage frequencies for Claude-Sonnet-4.5 in the Vending environment (left), Freelance environment (middle) and Operation environment (right).

0

50

100

150

200

250

300

350

Day price_query

products_research

price_set

order_place

task_done

0

6

13

Call Count

Vending

0

50

100

150

200

250

300

350

Day energy_restoretasks_discover

tasks_browse

task_done

task_inspect

solution_submit

0

17

34

Call Count

Freelance

0

50

100

150

200

250

300

350

Day acquisition_boost

moderation_tighten

creator_incentive

engagement_tune

task_done

- 0

- 1

- 2

Call Count

Operation

- Figure 7 Temporal evolution of tool usage frequencies for DeepSeek-v3.2 in the Vending environment (left), Freelance environment (middle) and Operation environment (right).

### F Temporal Evolution of Agent Behavioral Patterns

This appendix presents the complete set of temporal tool-use ridge plots for all remaining models not highlighted in the main text. Each figure visualizes day-level tool call frequencies over the full 365-day simulation horizon, constructed using the same daily-binning and cubic-spline smoothing procedure.

#### G GUI in Human Performance Testing Fig 10 shows GUI used in human performance testing.

###### Vending

Operation

###### Freelance

9

Call Count

6

Call Count

14

Call Count

4

3

7

0

0

0

task_done

task_done

solution_submit

0

0

order_place

0

task_inspect

creator_incentive

50

50

50

100

task_done

products_research

100

100

moderation_tighten

150

150

tasks_browse

150

price_query

200

acquisition_boost

200

200

Day energy_restoretasks_discover

250

Day price_set

250

Day engagement_tune

250

300

300

300

350

350

350

- Figure 8 Temporal evolution of tool usage frequencies for Gemini-3-Flash in the Vending environment (left), Freelance environment (middle) and Operation environment (right).

0

50

100

150

200

250

300

350

Day price_query

products_research

price_set

order_place

task_done

0

4

9

Call Count

Vending

0

50

100

150

200

250

300

350

Day energy_restoretasks_discover

tasks_browse

solution_submit

task_done

task_inspect

0

5

11

Call Count

Freelance

0

50

100

150

200

250

300

350

Day engagement_tune

moderation_tighten

creator_incentive

acquisition_boost

task_done

0

3

6

Call Count

Operation

- Figure 9 Temporal evolution of tool usage frequencies for GPT-5.2 in the Vending environment (left), Freelance environment (middle) and Operation environment (right).

[Figure 44]

- Figure 10 GUI for Operation tasks. Human experts can execute actions via the buttons in the bottom-left corner.

