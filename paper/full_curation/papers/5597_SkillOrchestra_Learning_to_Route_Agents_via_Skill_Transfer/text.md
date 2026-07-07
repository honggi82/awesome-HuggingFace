## arXiv:2602.19672v1[cs.AI]23Feb2026

# SkillOrchestra: Learning to Route Agents via Skill Transfer

Jiayu Wang1, Yifei Ming2, Zixuan Ke2, Shafiq Joty2, Aws Albarghouthi1, and Frederic Sala1 1University of Wisconsin-Madison, 2Salesforce AI Research

### Abstract

Compound AI systems promise capabilities beyond those of individual models, yet their success depends critically on effective orchestration. Existing routing approaches face two limitations: (1) input-level routers make coarse query-level decisions that ignore evolving task requirements; (2) RL-trained orchestrators are expensive to adapt and often suffer from routing collapse, repeatedly invoking one strong but costly option in multi-turn scenarios. We introduce SkillOrchestra, a framework for skill-aware orchestration. Instead of directly learning a routing policy end-to-end, SkillOrchestra learns fine-grained skills from execution experience and models agent-specific competence and cost under those skills. At deployment, the orchestrator infers the skill demands of the current interaction and selects agents that best satisfy them under an explicit performance-cost trade-off. Extensive experiments across ten benchmarks demonstrate that SkillOrchestra outperforms SoTA RL-based orchestrators by up to 22.5% with 700× and 300× learning cost reduction compared to Router-R1 and ToolOrchestra, respectively. These results show that explicit skill modeling enables scalable, interpretable, and sample-efficient orchestration, offering a principled alternative to data-intensive RL-based approaches. The code is available at: https://github.com/jiayuww/SkillOrchestra.

### 1. Introduction

[Figure 1]

[Figure 2]

Figure 1: Performance-cost tradoffs in multi-turn model routing (left) and agent orchestration (right). SkillOrchestra and SkillOrchestra+ lie on the Pareto frontier, with higher accuracy at lower cost than all baselines.

Modern AI systems are increasingly built as compound agents that coordinate multiple large language models (LLMs) and tools to solve complex, multi-step tasks such as deep research (Gemini, 2024, OpenAI, 2025b) and scientific discovery (Gottweis et al., 2025). Instead of relying on a single model, these systems interleave operations such as web search, code execution, and answer synthesis, dynamically invoking models with different strengths and costs (Ke et al., 2025). In this setting, orchestration, the process deciding what capability is required at each interaction state and which model–tool combination to invoke, is central to

Correspondence: Jiayu Wang: milawang@cs.wisc.edu

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

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

- Figure 2: Comparison of model routing and agent orchestration approaches. (Left) Model routing performs static, query-level model selection without dynamic mode or tool reasoning. (Middle) Direct agent orchestration learns routing end-to-end with implicit capability modeling and is prone to routing collapse. (Right) Skill-aware agent orchestration leverages a reusable Skill Handbook with explicit skill-level capability modeling, enabling balanced agent utilization and extensibility.

both performance and efficiency.

A common form of orchestration is model routing, where a controller selects a model from a model pool (Chen et al., 2024a, Hu et al., 2024, Ong et al., 2025). However, existing routing methods are often ill-suited to modern agentic workloads. Most routers make single-shot, query-level decisions, assuming one model suffices for the entire task. This assumption breaks down in multi-turn interactions, where different states require distinct capabilities. Agentic workflows often interleave operational modes (e.g., web search and coding), each demanding different skills. Routing should therefore operate at the level of fine-grained capability requirements conditioned on the current interaction state, rather than treating the entire query as a single decision unit (Figure 2, left). Recent RL-based orchestration methods (Zhang et al., 2025a, Su et al., 2025) address this by learning sequential routing policies with LLMs. While more flexible, these approaches introduce new challenges: expensive training, limited adaptability to evolving model and tool pools, and a tendency toward what we term routing collapse: the degeneration of the orchestration policy into repeatedly selecting a single option at one or more decision levels (e.g., agent type or backbone model), despite the availability of alternatives with better accuracy-cost trade-offs (Figure 2, middle).

To address these limitations, we introduce SkillOrchestra, a framework for skill-aware orchestration. Rather than directly optimizing a routing policy end-to-end, SkillOrchestra learns a reusable Skill Handbook from execution experience. The handbook encodes (i) mode-level execution insights that guide what operation should be performed at each interaction state, (ii) fine-grained skills that characterize capability requirements within each mode, and (iii) agent profiles that summarize skill-conditioned performance, cost characteristics, and practical usage insights. At deployment, the orchestrator first selects the appropriate operational mode conditioned on the current state, then chooses the agent that best satisfies the required skills under an explicit performance-cost trade-off (Figure 2, right).

This skill-centric perspective brings three systemic advantages. First, it enables state-conditioned, fine-grained orchestration, allowing different models to specialize across capabilities. Second, it promotes stable and balanced routing behavior, mitigating routing collapse seen in RL-tuned orchestrator. Third, it produces transferable orchestration knowledge: the learned Skill Handbook can be reused across different orchestrator backbones and updated model pools, decoupling orchestration knowledge from router parameters.

We evaluate SkillOrchestra in both multi-turn model routing and full agent orchestration settings. As shown in Figure 1, SkillOrchestra and SkillOrchestra+ lie on the Pareto frontier, achieving higher accuracy at lower cost than all baselines. Across ten diverse benchmarks, SkillOrchestra consistently outperforms heuristic, discriminative, and RL-based approaches. For example, SkillOrchestra outperforms SoTA RL-trained orchestrators, achieving up to 22.5% absolute improvement, with 700× and 300× cost reduction compared to Router-R1 (Zhang et al., 2025a) and ToolOrchestra (Su et al., 2025), respectively. Moreover, it exhibits more balanced routing patterns and transfers effectively across orchestrator models without retraining. We summarize our contributions as follows:

❶ Skill-aware orchestration. We propose SkillOrchestra, a new paradigm that structures orchestration decisions around explicit capability abstractions and agent profiles, enabling state-conditioned, performancecost-aware orchestration.

❷ Skill Handbook learning. We introduce a data-efficient framework to discover and refine reusable skills and execution insights from agent traces, while estimating skill-conditioned agent performance and cost.

❸ Granularity-aware skill handbook selection. We show that optimal skill granularity depends on orchestrator capacity, and develop a validation strategy to select orchestrator-specific handbooks that balance expressiveness and decision reliability under performance-cost trade-offs.

❹ Empirical gains and transferability. Extensive experiments across ten benchmarks demonstrate improved accuracy, efficiency, and routing stability over strong RL-tuned baselines, alleviating routing collapse and transferring across orchestrator backbones without retraining.

### 2. Related Works

Model Routing. Model routing aims to select the most appropriate model from a pool to balance performance and inference cost. Early approaches rely on heuristic or cascade strategies (Chen et al., 2024a) that escalate queries based on predicted difficulty or budget constraints (Ding et al., 2024, Šakota et al., 2024). Prior approaches are largely heuristic or discriminative, learning static mappings from query features to model choice (Jiang et al., 2023b) or relying on cascades (Chen et al., 2024a) and difficulty estimation. Instead, a large body of work learns discriminative query-model matching, using similarity-based methods (Hu et al., 2024, Ong et al., 2025), neural classifiers or ensembles (Jiang et al., 2023b, Lu et al., 2024), and graph-based formulations (Feng et al., 2025) to predict which model should answer a query (Chen et al., 2024b, Stripelis

- et al., 2024). Despite their effectiveness, routing decisions for these approaches are typically made once per query using input-level features only, without modeling how model competence differs across intermediate stages. As a result, they struggle to support fine-grained, multi-step orchestration.

RL-based Routing and Orchestration. To enable multi-step decisions, recent work formulates routing as a sequential decision process and trains an LLM-based router using reinforcement learning (Schulman et al., 2017, Shao et al., 2024). Systems such as Router-R1 (Zhang et al., 2025a) and ToolOrchestra (Su et al.,

[Figure 59]

[Figure 60]

[Figure 61]

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

[Figure 77]

[Figure 78]

[Figure 79]

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

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

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

- Figure 3: Overview of SkillOrchestra. (Left) A global Skill Handbook is constructed by discovering and refining reusable skills and execution-level insights from agent traces, while jointly estimating each agent’s skill competence and associated cost. (Middle) An orchestrator-specific handbook is selected via Pareto validation to achieve a principled trade-off between performance and cost. (Right) At deployment, the orchestrator performs mode-aware and skillgrounded agent selection using the selected handbook.

2025) that interleave reasoning and routing, optimizing performance-cost trade-offs via trajectory-level rewards. While more flexible than single-shot routers, RL-based approaches introduce new challenges such as high training cost, poor adaptability to new model pools or tasks, and policy routing collapse, where the router converges to repeatedly invoking a single strong but expensive model. In contrast, we introduce skill as an intermediate abstraction and construct a reusable Skill Handbook that captures mode-conditioned competence patterns. This design enables data-efficient, transferable, and more balanced orchestration without end-to-end RL training.

### 3. Preliminaries

Agent Orchestration. We consider an agentic task environment where a user instruction q ∈ 𝒬 initiates a multi-step reasoning process. The system consists of the following components: [✱] The Orchestrator (𝒪): A central controller responsible for high-level planning and resource allocation. [✱] Operational Modes (Ψ). A set of abstract action modes Ψ = {search,code, . . .} defined at the capability level. At each turn, the orchestrator chooses a mode ψ ∈ Ψ that specifies the type of operation required (e.g., retrieving external information, or code execution).

[✱] Model Pool (ℳ). A set of candidate foundation models ℳ = {m1, . . . , mKM}, which may include general-purpose and specialized LLMs (e.g., GPT-5, Claude, Qwen-3, or domain-specific coder and math

models).

[✱] Tool Pool (𝒯 ). A set of executable tools 𝒯 = {t1, . . . , tKT}, such as web search engines (e.g., Google Search, Tavily Search), code execution environments (e.g., Python), database retrieval systems, or other

external APIs. [✱] Agent Instantiation. An agent is defined as a pair

A = (m,𝒯A), m ∈ ℳ, 𝒯A ⊆ 𝒯 ,

where m is the backbone model and 𝒯A is the subset of tools accessible during execution. Each operational mode ψ ∈ Ψ restricts the allowable tools, inducing a set of valid agents

𝒜ψ = {(m,𝒯A) ∣ m ∈ ℳ, 𝒯A ⊆ 𝒯ψ}, where 𝒯ψ ⊆ 𝒯 denotes the tools relevant to mode ψ.

Task Execution Workflow. Given query q, the system evolves over turns t = 0, . . . , T. Let st denote the system state at turn t, which consists of the original query and the accumulated interaction history up to

that point. At each turn, the orchestrator selects a mode ψt ∈ Ψ (what to do) and an agent At ∈ 𝒜ψt (who executes it), forming the action at = (ψt, At). The selected agent produces an execution trace zt (e.g., search results or generated code), after which the environment returns an observation ot (e.g., tool outputs or execution results), leading to the next state st+1. This interaction induces a trajectory τ = (s0, a0, z0, o0, s1, a1, z1, o1, . . . , sT). An example multi-step workflow is illustrated in Fig. 3 (right).

Problem Formulation. The orchestrator aims to learn a policy π that optimizes performance-cost tradeoffs over trajectories. Formally, we seek to maximize the expected reward R(τ) and minimizing the cumulative execution cost:

T

J(τ) = Eτ∼π [R(τ) − λ

C(At, zt)],

∑

max

π

t=0

where C(At, zt) denotes the cost incurred by the selected agent At when producing trace zt (e.g., token usage and/or latency), and λ is a tradeoff hyperparameter. We factorize the policy as

π(at ∣ st) = πmode(ψt ∣ st) ⋅ πroute(At ∣ st, ψt),

where πmode determines the next operational mode (e.g., Search vs. Coding), and πroute selects the optimal agent At conditioned on the current state and mode.

Under this formulation, traditional model routing (Hu et al., 2024, Chen et al., 2024b) can be viewed as a special case with a single timestep T = 0, a single operational mode Ψ = answer, and no external tools. The objective reduces to maxπroute EA∼πroute(⋅∣q) [R(A, q) − λC(A, z)], where the state s0 = q is the user query and routing consists of choosing one model to generate the final answer in a single step.

Prior work typically instantiates this optimization via RL such as GRPO (Shao et al., 2024) by directly finetuning the orchestrator parameters θ toward the optimal policy (Su et al., 2025). In contrast, SkillOrchestra reframes orchestration as a problem of skill acquisition rather than parameter adaptation. Instead of updating θ, we learn a Skill Handbook ℋ, a reusable experience base that captures (i) mode-level execution insights about what operation to perform at a given interaction state, (ii) fine-grained skills that characterize capability requirements within each mode, and (iii) agent profiles that summarize competence and cost

under those skills (e.g., high-precision arithmetic, symbolic logic coding). Under this view, the optimization shifts from learning a routing policy to identifying the optimal handbook structure:

Eτ∼π(⋅∣ℋ) [J(τ)].

ℋ∗ = argmax

ℋ

By optimizing the Skill Handbook ℋ, we explicitly align abstract task demands with concrete agent capabilities, enabling the orchestrator to reason over the competence landscape of the agent pool even without costly end-to-end RL finetuning.

### 4. SkillOrchestra

SkillOrchestra reframes orchestration as skill-grounded decision making rather than direct policy optimization. Instead of learning a monolithic routing policy, we learn a structured Skill Handbook that captures reusable execution knowledge. During training, the handbook is incrementally constructed and refined from execution traces, including skills, agent profiles, and execution insights. At test time, the orchestrator consults a selected subset of this handbook to guide mode selection and agent routing.

- Definition 4.1 (Skill). A skill is a reusable capability abstraction that specifies the type of competence required to perform a task under an operational mode ψ. Skills form an intermediate layer between high-level modes (e.g., search, code) and individual agents, enabling the system to decouple capability requirements from agent identity.

Formally, a skill σ is represented as

σ ≜ ⟨𝒟,ℐ⟩,

where 𝒟 is a natural-language description of the capability, and ℐ denotes contextual indicators (e.g., keywords, structural patterns, or exemplar queries) that signal when the skill is applicable.

- Definition 4.2 (Agent Profile). An agent profile summarizes an agent’s mode-conditioned competence, cost, and routing characteristics for skill-aware orchestration. For agent A under operational mode ψ, the profile is defined as

𝒫A,ψ = ({ϕA,σ}σ∈Σψ, CˆA(ψ),ℛA,ψ, ΓA),

where ϕA,σ denotes the estimated success probability of agent A on skill σ, CˆA(ψ) is the estimated execution cost (e.g., latency, token usage) under mode ψ, ℛA,ψ encodes mode-conditioned routing signals such as usage constraints or systematic failures, ΓA provides a high-level summary of the agent’s strengths and weaknesses.

- 4.1. Agent Orchestration via Skill Handbook We now describe runtime orchestration using the Skill Handbook (Fig. 3, right).

Skill Handbook. The Skill Handbook ℋ organizes reusable orchestration knowledge at three levels: (i) mode-level execution insights that guide what operation to perform under different interaction states, (ii) a registry of fine-grained skills that capture capability requirements within each mode, and (iii) agent profiles that model skill-conditioned competence, routing signals, and execution cost. It can be viewed as a graph 𝒢ℋ = (𝒱,ℰ), 𝒱 = 𝒱Ψ ∪ 𝒱Σ ∪ 𝒱𝒫 stores mode selection insights, skills, and agent profiles. The edge structure encodes associations between operational modes and relevant skills.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

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

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

- Figure 4: Example instantiation of a learned Skill Handbook. The handbook decouples capability requirements from agent identity through three components: (left) mode-level routing insights, (middle) a hierarchical registry of reusable skills, and (right) agent profiles encoding skill-specific competence estimates and execution cost statistics.

Example (Skill Handbook Instantiation). Figure 4 shows a concrete instantiation of ℋ. For example, under mode ψ = code, the handbook stores mode-level metadata (left) capturing when to code. The skill registry (middle) may include a high-level skill data_processing, which further specializes into subskills such

- as symbolic_logic. Each agent is associated with a profile (right) providing competence estimates over these skills, mode-conditioned routing signals, and execution cost statistics. Together, these components enable structured, skill-grounded agent selection.

[✱] Mode-level metadata 𝒱Ψ. For each operational mode ψ ∈ Ψ, the handbook stores mode-level routing insights ℛψ learned from execution traces, guiding high-level transitions (e.g., when to switch from Search to Code).

[✱] Skill registry 𝒱Σ. The handbook maintains a registry of skills (Definition 4.1), each representing a task-conditioned capability that may be required during execution.

[✱] Agent profiles 𝒱𝒫. Each agent A is associated with an agent profile (Definition 4.2), which stores agent-specific performance estimates over skills, routing insights of this agent, and cost characteristics. Agent profiles are queried during routing but are not indexed by graph edges.

[✱] Mode–skill index ℰ. The graph structure induces a mapping M ∶ Ψ → 2Σ, where Σψ ∶= M(ψ) denotes the set of skills associated with operational mode ψ. This index restricts routing decisions to mode-consistent skills without searching over the full skill space at runtime.

Orchestration with Skill Handbook. At inference time, the orchestrator interacts with the Skill Handbook in a task-conditioned manner. Given a user query q, the system follows a retrieval–execution cycle.

- Step 1: Handbook Selection. The effectiveness of a handbook depends on how well its structural granularity aligns with the reasoning capacity of the target orchestrator. Although the learned handbook ℋ∗ may contain fine-grained skills and detailed routing insights derived from prior experience, not all such structure is equally beneficial for every orchestrator.

Fine-grained skill decompositions require accurate inference of which subskill is active in the current interaction state. While a strong orchestrator may reliably distinguish between subskills such as symbolic_logic and numerical_approximation under code mode, a lower-capacity orchestrator may misidentify the active skill, introducing routing bias and degrading end-to-end performance. For example, in a coding query re-

quiring logical constraint verification, activating numerical_approximation instead of symbolic_logic may route to an agent specialized in numeric computation but suboptimal for symbolic reasoning. Operating

- at a coarser granularity (e.g., using a broader skill such as data_processing) reduces sensitivity to such misidentification and yields more stable routing decisions. Starting from the learned handbook ℋ∗ (Section 4.2), we therefore select an orchestrator-specific subset

ℋbase(𝒪) for orchestrator 𝒪 via Pareto-optimal validation (Section 4.3). This selection determines which skills, agent profiles, and routing metadata are retained, as well as their effective granularity, so as to maximize

end-to-end performance given target orchestrator under a given cost budget. Formally, ℋbase(𝒪) is an induced subgraph of ℋ∗: ℋbase(𝒪) = (𝒱Ψbase ∪ 𝒱Σbase ∪ 𝒱𝒫base,ℰbase), where 𝒱Ψbase contains mode-level routing metadata useful for the orchestrator to select operational modes, 𝒱Σbase contains the skills retained for those modes at the selected granularity, and 𝒱𝒫base contains the corresponding agent profiles. The edge set ℰbase ⊆ ℰ restricts mode–skill associations to the retained nodes. All node attributes, including routing insights, performance estimates, and cost statistics, are inherited from ℋ∗.

At inference time, the orchestrator retrieves ℋbase(𝒪). Optionally, the retrieval operator may augment ℋbase(𝒪) with additional skills whose semantic similarity to the query exceeds a threshold, yielding the final handbook

ℋq used for query q and orchestrator 𝒪: ℋq = ℋbase(𝒪) ∪ ⋃σ∈𝒩

k(q) ({σ} ∪ {𝒫A,ψ ∣ A ∈ 𝒜ψ}), where 𝒩k(q) is the k nearest skills in the embedding space.

- Step 2: Skill-Grounded Agent Routing. Guided by the retrieved handbook ℋq, the orchestrator performs skill-grounded routing through an iterative decision process. An illustration can be found in Figure 3 (right). At each time step t, it decides:

✦ Mode Selection. The mode policy πmode selects the current operational mode ψt based on the interaction state st and the mode-level routing metadata stored in the handbook: ψt ∼ πmode(ψ ∣ st;ℛψ). This decision determines the operational mode to execute next (e.g., Search, Code).

✦ Competence-Aware Agent Routing. Conditioned on the selected mode ψt, the orchestrator identifies a set of relevant skills Σt ⊆ Σψt that are active for the current state. Agent selection is then performed by aggregating competence estimates over this skill set and trading them off against execution cost:

[Eσ∈Σt[ϕA,σ] − λc ⋅ CˆA(ψt)].

A∗t = argmax

A∈𝒜ψt

where ϕA,σ is the performance estimate stored in the agent profile 𝒫A,ψt. In practice, we approximate the expected competence by aggregating the posterior means over the active skill set and optionally incorporating

semantic alignment between the current state and the agent profile:

A∗t = argmax

A∈𝒜ψt

−λc ⋅ CˆA(ψt)

[ ∑

].

αA,σ αA,σ + βA,σ

wt,σ

σ∈Σt

Mode-Specific Cost

Estimated Competence

This ensures that each decision is grounded in task-relevant skills, agent-specific competence estimates, and explicit cost constraints. The full algorithm is provided in Appendix B (Algorithm 1).

###### 4.2. Skill Handbook Learning

We construct and refine the Skill Handbook ℋ from execution traces rather than learning a monolithic routing policy. The procedure iteratively updates the skill registry, agent profiles, and mode-level routing metadata (Figure 3, left).

- Phase 1: Skill Discovery and Profile Construction. We assume an exploratory dataset 𝒟train = {(qi,ℬi)}iN=1, where ℬi = {τi(1), τi(2), . . .} are trajectories obtained by varying the agent choice at specific modes.

For each query and mode ψ, we contrast a successful trajectory τ+ψ with a failed one τ−ψ. Their difference 𝒟diff(τ+ψ ∥ τ−ψ) isolates the missing capability. An LLM-based discoverer abstracts this capability gap into a reusable skill definition σnew, which is added to the registry 𝒱Σ together with its associated mode mapping M.

Agent profiles are then estimated from aggregated outcomes. For each agent A, mode ψ, and skill σ ∈ Σψ, we model success probability as ϕA,σ ∼ Beta(αA,σ, βA,σ), updated via

α(At+,σ1) ← α(At,)σ + ∑ τ∈ℬi

I[A succeeds on σ in τ],

β(At+,σ1) ← β(At),σ + ∑ τ∈ℬi

I[A fails on σ in τ].

Mode-level routing signals (e.g., frequent transitions or systematic failures or recurring recovery patterns) are distilled into reusable mode-selection insights and stored as routing metadata ℛψ.

- Phase 2: Handbook Refinement. To prevent over-fragmentation or redundancy, we periodically refine the skill set using agent profile statistics.

✦ Splitting. A skill σ is marked as a split candidate if agent performance exhibits high variance across its associated queries, indicating multiple underlying capabilities.

✦ Merging. A pair of skills (σi, σj) is marked as a merge candidate when their agent performance profiles are statistically indistinguishable, suggesting redundancy for routing.

Given these candidates, an LLM-based reflector (e.g., GPT-5) reviews the proposed operations and, if appropriate, generates revised skill definitions. Approved refinements update both the skill registry and the associated competence statistics (αA,σ, βA,σ). The final refined handbook ℋ∗ encodes learned skills, agent profiles, and routing metadata, and serves as the reusable knowledge base for inference-time handbook selection (Section 4.1).

###### 4.3. Pareto-Optimal Skill Handbook Selection

This subsection formalizes the handbook selection step introduced in Section 4.1, where an orchestratorspecific subset is chosen to match the reasoning capacity and cost budget of the target orchestrator. An illustration can be found in Figure 3 (middle).

Given the learned handbook ℋ∗ (Section 4.2), our goal is to select, for a target orchestrator 𝒪, a subset ℋ ⊆ ℋ∗ that achieves the best end-to-end performance-cost tradeoff.

Each candidate subset ℋ induces a routing policy πℋ, which produces a trajectory τℋ(q) for a query q. We

Table 1: Experimental results on QA datasets. Bold = best, underline = second best in each column. SkillOrchestra uses the same orchestrator model as baselines. SkillOrchestra+ reports the best performance obtained by switching among different orchestrator models within the same agent pool while using the same learned Skill Handbook.

Method General QA Multi-Hop QA Avg. NQ TriviaQA PopQA HotpotQA 2wiki Musique Bamboogle

Vanilla 9.2 26.0 12.2 14.0 26.6 2.6 4.0 13.5 No Routing SFT 21.2 40.0 16.0 19.8 25.6 5.2 11.2 19.9 RAG 29.8 54.0 36.6 21.6 14.6 7.8 22.4 26.7 CoT (Wei et al., 2022) 12.6 35.8 16.0 16.8 20.8 4.6 22.4 18.4 Search-R1 (Jin et al., 2025) 32.8 51.0 32.4 23.6 27.8 9.0 27.2 29.1 Heuristic & Discriminative Routing Largest LLM 29.6 57.8 35.4 27.8 27.4 10.4 48.0 33.8 Prompt LLM 30.0 58.0 34.0 26.8 26.2 10.8 44.8 32.9 Prompt LLM+ (multi turn) 25.8 50.0 25.6 20.6 24.8 7.8 47.2 28.8 KNN Router (Hu et al., 2024) 26.2 52.8 22.2 22.4 19.6 6.6 36.0 26.5 KNN Router+ (multi turn) 23.6 47.8 23.2 15.4 23.4 7.2 38.4 25.6 MLP Router (Hu et al., 2024) 25.2 46.0 22.2 19.8 21.0 7.2 36.0 25.3 BERT Router (Ong et al., 2025) 23.0 51.6 19.2 21.6 20.6 5.8 31.2 24.7 RouterDC (Chen et al., 2024b) 27.8 59.2 28.2 24.4 21.8 8.0 50.4 31.4 GraphRouter (Feng et al., 2025) 27.6 58.6 28.0 23.4 18.0 7.6 44.8 29.7 FrugalGPT (Chen et al., 2024a) 26.5 56.2 36.2 23.4 26.8 10.3 43.0 31.8 RL-based Routing Router-R1 (Zhang et al., 2025a) 38.8 70.6 38.4 35.2 43.4 13.8 51.2 41.6 Ours

SkillOrchestra 54.2 71.6 42.6 39.0 48.0 18.2 58.4 47.4 SkillOrchestra + 54.8 80.2 48.8 44.2 49.6 20.6 63.2 51.6

evaluate candidate subsets on a held-out validation set 𝒟val and solve:

Eq∼𝒟val ⎡ ⎢⎣

⎤

∣τℋ(q)∣

ℋbase(𝒪) = argmax

R(τℋ(q)) − λ

C(ψt, At)

∑

⎥⎦

.

ℋ⊆ℋ∗

t=0

Here, R(τℋ(q)) ∈ [0,1] denotes task success, and C(ψt, At) is the execution cost at step t. The coefficient λ controls the performance-cost tradeoff. This objective directly evaluates entire trajectories rather than local routing accuracy, ensuring that the selected handbook lies on the Pareto frontier for the target orchestrator.

- 5. Experiments We conduct extensive experiments to answer:

- (RQ1) Effectiveness: Does a learned Skill Handbook improve end-to-end accuracy over heuristic, discriminative, and RL-based methods?
- (RQ2) Efficiency: Does skill-based orchestration yield a better performance-cost trade-off?
- (RQ3) Routing Behavior: Does skill-based orchestration reduce routing collapse and better match model capacity to task difficulty across modes?
- (RQ4) Transferability: Can a Skill Handbook be reused across orchestrators without retraining?

- (RQ5) Component Contribution: How do different components of the Skill Handbook contribute to overall performance and cost efficiency?

###### 5.1. SkillOrchestra for Model Routing

We first evaluate SkillOrchestra in the model routing setting (Chen et al., 2024a, Feng et al., 2025, Zhang

- et al., 2025a), where no external tools or knowledge base are provided. Therefore, the performance gaps directly reflect the quality of model orchestration.

Benchmarks. We consider a diverse suite of knowledge and reasoning-intensive tasks including (1) General QA: Natural Question (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), PopQA (Mallen et al., 2023); (2) Multi-hop QA: HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al., 2020), Musique (Trivedi et al., 2022), and Bamboogle (Press et al., 2023); (3) Math Reasoning: MATH (Hendrycks et al., 2021) and AMC23 (MAA, 2023).

Experimental setup and baselines. We use Qwen2.5-3B (Qwen, 2024) as the orchestrator and adopt the same configuration as Router-R1 for controlled comparison with all routing baselines. Model pool and implementation details are included in Appendix A.1. We compare SkillOrchestra against three categories of methods: (1) No routing: methods that do not dynamically consult different models, including supervised finetuning, RAG as in Zhang et al. (2025a), CoT (Wei et al., 2022), and Search-R1 (Jin et al., 2025); (2) Heuristic & Discriminative routing: methods that select models based on input-level signals or learned classifiers, including Largest LLM, Prompt LLM, Prompt LLM+ (explicit task decomposition+multi-turn), KNN Router (Hu et al., 2024), KNN Router+ (explicit task decomposition and route each subtask to different models which matching query similarity with KNN router), MLP Router (Hu et al., 2024), BERT Router (Ong et al., 2025), RouterDC (Chen et al., 2024b), GraphRouter (Feng et al., 2025), and FrugalGPT (Chen et al., 2024a); (3) RL-based routing: Router-R1 (Zhang et al., 2025a), a strong PPO-trained (Schulman et al., 2017) multi-turn router with 14k samples, which represents the current SoTA in learned end-to-end orchestration.

Observation ❶ SkillOrchestra outperforms all routing baselines, including expensive RL-based methods (RQ1). SkillOrchestra surpasses all baselines on both general and multi-hop QA (Table 1). Compared to Router-R1 (41.6 EM), SkillOrchestra reaches 47.4 (+5.8), and SkillOrchestra+ achieves 51.6 (+10.0). Gains are especially large on multihop tasks such as Musique (13.8 → 18.2 → 20.6) and Bamboogle (51.2 → 58.4 → 63.2). Similar trends hold for math reasoning (Figure 5), with up to +22.5 accuracy over Router-R1 at substantially lower cost. Notably, these gains require only a small fraction of the training data, demonstrating higher data efficiency than RL-based routing.

###### Performance

###### Cost

80

6.5

Router-R1 SkillOrchestra

73.6

| |
|---|

6

| |
|---|

Accuracy(%)

UScents

60

55.8

52.5

3.6

4

40

1.6

2

25.0

0.5

20

0

MATH AMC

MATH AMC

Figure 5: Performance and cost comparison: SkillOrchestra vs. Router-R1. SkillOrchestra achieves up to a 22.5 percentage-point improvement in accuracy while reducing inference cost by ∼ 2.0×.

Observation ❷ SkillOrchestra lies on the Pareto frontier (RQ2). Figure 1 (left) shows that SkillOrchestra and SkillOrchestra+ achieve higher accuracy at lower or comparable cost than all heuristic, discriminative, and RL-based baselines. Importantly, higher per-token price does not necessarily imply higher total inference

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

[Figure 164]

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

Figure 6: Skill-based orchestration mitigates routing collapse and generalizes across orchestrators. (Left) Router-R1 collapses to a single large model (98% Llama3.1-70B), while SkillOrchestra distributes calls according to capability differences. (right) A Skill Handbook learned from Qwen2.5-3B transfers across orchestrator backbones without retraining, consistently improving performance and achieving larger gains with stronger backbones.

cost. Total cost depends jointly on (i) the per-token price of the selected backbone model, (ii) the number of generated tokens, and (iii) the number of routing steps. In practice, some lower per-token models produce substantially longer reasoning chains, leading to higher overall cost. SkillOrchestra explicitly accounts for this trade-off, often selecting capable yet more cost-efficient models (e.g., Mixtral-8×22B) instead of consistently escalating to the most expensive model (LLaMA-3.1-70B). For example, Router-R1 attains 41.6 EM at a high cost (51.8¢), whereas SkillOrchestra achieves higher accuracy (47.4 EM) at a lower cost (38.4¢). SkillOrchestra+ further improves to 51.6 EM at 41.6¢. Router-R1 attains 41.6 EM at a high cost (51.8¢), whereas SkillOrchestra achieves higher accuracy (47.4 EM) at a lower cost (38.4¢). SkillOrchestra+ further improves to 51.6 EM at 41.6¢. Similar advantages appear in math reasoning (Figure 5), where SkillOrchestra improves accuracy while reducing cost by about 2×. These results indicate that skill-aware routing allocates models more efficiently and shortens reasoning chains.

Observation ❸ Skill-based routing alleviates routing collapse seen in RL-based routing (RQ3). To understand the performance and efficiency gap, Figure 6 (left) compares model selection distributions across nine benchmarks. Router-R1 shows clear routing collapse: it selects LLaMA-3.1-70B for 98.02% of all calls, while all other models are almost unused (each ≤0.92%; e.g., Qwen2.5-7B 0.35%, Mistral-7B 0.92%, Mixtral-8×22B 0.04%, Qwen2.5-3B 0.00%). Despite being trained as a multi-model router, its RL policy converges to repeatedly invoking a single large model, limiting specialization and inflating cost. In contrast, SkillOrchestra produces a much more balanced routing pattern: e.g., Mixtral-8×22B 44.53%, Qwen2.5-

- 7B 25.99%, LLaMA-3.1-70B 15.38%, and Qwen2.5-3B 11.50%. This distribution reflects capability-aware specialization, where stronger models are used only when necessary and lighter models handle simpler steps. Importantly, skill-based routing also makes the orchestrator itself more effective. In some cases, the orchestrator can directly answer the query without escalating to a larger model, further reducing unnecessary calls and lowering the total cost. An example in shown in Figure 8.

Observation ❹ The learned skill handbook transfers across orchestrator backbones without retraining (RQ4). We reuse the skill handbook learned from traces where Qwen2.5-3B serves as the orchestrator, and directly apply it to other backbone models without any additional handbook training. Figure 6 (right) shows

performance before and after introducing the same skill handbook, with results averaged over three general QA datasets. The learned handbook consistently improves all tested models. Qwen2.5-3B itself improves from 40.7% to 56.1% (+15.4). When transferred to larger or stronger models, the gains remain substantial: Qwen2.5-7B improves from 35.7% to 60.0% (+24.3), Llama3.1-8B from 35.5% to 58.0% (+22.5), and Mistral-7B from 36.5% to 59.8% (+23.3). Even larger-scale models benefit from such handbook: Mixtral8x22B from 46.5% to 61.3% (+14.8). These results show that the skill handbook captures transferable, model-agnostic orchestration knowledge. Notably, stronger models often achieve the highest absolute performance when paired with the transferred handbook, suggesting that improved backbone capability and structured skill guidance are complementary.

###### 5.2. SkillOrchestra on Agent Orchestration

We next evaluate whether SkillOrchestra extends beyond model routing to full agent orchestration, where the system must coordinate multiple operational modes and tools beyond model selection. We use the same configuration as ToolOrchestra (Su et al., 2025), detailed in the following.

Experimental setup and baselines. We evaluate on FRAMES (Krishna et al., 2024) and consider three operational modes: search (web and local search), code, and answer. Each mode corresponds to a different model pool, detailed in Appendix A.2. The maximum interaction horizon is 50 turns. With Qwen3-

- 8B as the orchestrator, we compare against ToolOrchestra (Su et al., 2025), which trains the orchestrator using GRPO. We also compare against strong proprietary model orchestrators such as GPT-5 (OpenAI, 2025a), Gemini-3-Pro (Google, 2025), or Claude-Opus-4.5 (Anthropic, 2025), while keeping modes, model pools, tools, and execution environments fixed.

Observation ❺ SkillOrchestra achieves better performance-cost trade-offs in full agent orchestration (RQ1, RQ2). Figure 1 (right) shows that SkillOrchestra remains on the Pareto frontier in the more complex agent orchestration setting with multiple tools, operational modes beyond model routing. Our method achieves the highest accuracy (84.3%) while also incurring the lowest total cost ($72.7) among strong learned and proprietarymodel baselines. Compared to the RL-trained ToolOrchestra (76.3%, $92.7), SkillOrchestra improves accuracy by +8.0 points while reducing cost by 21.6%. It also outperforms stronger proprietary orchestrators such as GPT-5 (74.6%, $120.4), Claude Opus 4.5 (77.9%, $758.1), and Gemini 3 Pro (78.9%, $1729.3). These results highlight an important system-level trade-off: while using a stronger model as the orchestrator can improve raw task performance, it often does so at a prohibitive cost due to expensive per-token pricing and long multi-step trajectories. In contrast, SkillOrchestra improves both accuracy and efficiency by coordinating specialized models and tools through explicit skill modeling, rather than relying on a single large model to carry the entire process.

Table 2: Analysis of SkillOrchestra’s Skill Handbook design. HB: Has Handbook, Disc: Skill Discovery, Ref: Handbook Refinement, Sel: Handbook Selection, FG: Fine-grained Skills. Orchestrator: Qwen3-8B.

Setting HB Disc Ref Sel FG Acc % Cost $ No HB ◦ ◦ ◦ ◦ ◦ 71.0 122.9 No Ref + Sel ✔ ✔ ◦ ◦ ✔ 79.0 5.5 No Selection ✔ ✔ ✔ ◦ ✔ 79.3 3.4 No FG Skills ✔ ✔ ✔ ✔ ◦ 80.4 15.1

Full System ✔ ✔ ✔ ✔ ✔ 85.0 9.3

Observation ❻ More skills are not always better; optimal performance-cost trade-offs require refining and selecting skills to match the orchestrator’s capability (RQ1, RQ3, RQ5). To understand the contribution of each component of SkillOrchestra, we conduct a controlled ablation study on 100 randomly

sampled FRAMES tasks. As shown in Table 2, removing the Skill Handbook causes a large drop in accuracy (85.0% → 71.0%) and a sharp increase in cost (9.3 → 122.9), showing that structured skill guidance is crucial for both effectiveness and efficiency. Using discovered skills without handbook refinement and selection, which includes redundant, overlapping or overly broad skills, still achieves reasonable accuracy (79.0%) at low cost (5.5), suggesting that even an unrefined skill set still provides useful routing signals. Enabling refinement further reduces cost (3.4) while maintaining similar accuracy (79.3%), indicating that reorganizing skills by merging redundant ones and splitting indistinguishable ones improves efficiency. Disabling fine-grained skills degrades both accuracy (80.4%) and efficiency (15.1), showing that appropriately detailed skills help the orchestrator make better decisions. Overall, the best performance-cost trade-off is achieved when skills are discovered, reorganized, and selectively applied at a level of detail that the orchestrator can use effectively.

### 6. Conclusion

In this work, we propose SkillOrchestra, an agentic orchestration framework that reframes multi-turn routing as skill-grounded decision making. By learning a Skill Handbook, the orchestrator makes state-aware, competence-aware decisions that explicitly optimize the performance-cost trade-off. Across both model routing and agent orchestration settings, SkillOrchestra achieves superior performance with significantly lower cost compared to competitive baselines. Moreover, the handbook is transferable across orchestrator backbones without retraining, enabling scalable deployment as model pools evolve. We hope this work serves as a springboard for scalable orchestration that improves the performance-cost frontier as agent pools grow and diversify.

### References

Anthropic. Introducing Claude Opus 4.5, 2025. URL https://www.anthropic.com/news/ claude-opus-4-5.

Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024a. ISSN 2835-8856. URL https://openreview.net/forum?id=cSimKw5p6R. Featured Certification.

Shuhao Chen, Weisen Jiang, Baijiong Lin, James Kwok, and Yu Zhang. RouterDC: Query-based router by dual contrastive learning for assembling large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b. URL https://openreview.net/forum?id= 7RQvjayHrM.

Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Ruhle, Laks VS Lakshmanan, and Ahmed Hassan Awadallah. Hybrid llm: Cost-efficient and quality-aware query routing. arXiv preprint arXiv:2404.14618, 2024.

Tao Feng, Yanzhen Shen, and Jiaxuan You. Graphrouter: A graph-based router for LLM selections. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=eU39PDsZtT.

Gemini. Gemini deep research, 2024. URL https://gemini.google/overview/deep-research/.

Team Gemma, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

#### Google. A new era of intelligence with Gemini 3, 2025. URL https://blog.google/ products-and-platforms/products/gemini/gemini-3/.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, Khaled Saab, Dan Popovici, Jacob Blum, Fan Zhang, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Pushmeet Kohli, Yossi Matias, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R D Costa, José R Penadés, Gary Peltz, Yunhan Xu, Annalisa Pawlosky, Alan Karthikesalingam, and Vivek Natarajan. Towards an ai co-scientist. 2025. URL https://arxiv.org/abs/2502.18864.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle,

Aiesha Letman, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783. Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob

Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott, Nuria Bel, and Chengqing Zong, editors, Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.580. URL https://aclanthology.org/2020.coling-main.580/.

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-llm routing system. arXiv preprint arXiv:2403.12031, 2024.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b, 2023a. URL https://arxiv.org/abs/2310.06825.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14165–14178, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.792. URL https://aclanthology.org/2023.acl-long.792/.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan O Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training LLMs to reason and leverage search engines with reinforcement learning. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id= Rwhi91ideu.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10. 18653/v1/P17-1147. URL https://aclanthology.org/P17-1147/.

Zixuan Ke, Fangkai Jiao, Yifei Ming, Xuan-Phi Nguyen, Austin Xu, Do Xuan Long, Minzhi Li, Chengwei Qin, PeiFeng Wang, silvio savarese, Caiming Xiong, and Shafiq Joty. A survey of frontiers in LLM reasoning: Inference scaling, learning to reason, and agentic systems. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=SlsZZ25InC. Survey Certification.

Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation,

#### 2024. URL https://arxiv.org/abs/2409.12941.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.

Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren Zhou. Routing to the expert: Efficient reward-guided ensemble of large language models. In Kevin Duh, Helena Gomez, and Steven Bethard, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1964–1974, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.109. URL https://aclanthology.org/2024.naacl-long.109/.

MAA. American mathematics competitions 2023 problems (amc23), 2023. URL https://maa.org/ student-programs/amc/.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.546. URL https://aclanthology.org/2023.acl-long.546/.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. RouteLLM: Learning to route LLMs from preference data. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=8sSqNntaMr.

OpenAI. Gpt-5 system card, 2025a. URL https://cdn.openai.com/gpt-5-system-card.pdf. OpenAI. Introducing deep research, 2025b. URL https://openai.com/index/

introducing-deep-research/.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5687–5711, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.378. URL https://aclanthology.org/2023.findings-emnlp.378/.

Qwen. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/ blog/qwen2.5/.

Marija Šakota, Maxime Peyrard, and Robert West. Fly-swat or cannon? cost-effective language model choice via meta-modeling. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining, pages 606–615, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Dimitris Stripelis, Zhaozhuo Xu, Zijian Hu, Alay Dilipbhai Shah, Han Jin, Yuhang Yao, Jipeng Zhang, Tong Zhang, Salman Avestimehr, and Chaoyang He. TensorOpera router: A multi-model router for efficient LLM inference. In Franck Dernoncourt, Daniel Preoţiuc-Pietro, and Anastasia Shimorina, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 452–462, Miami, Florida, US, November 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.emnlp-industry.34. URL https://aclanthology.org/2024.emnlp-industry.34/.

Hongjin Su, Shizhe Diao, Ximing Lu, Mingjie Liu, Jiacheng Xu, Xin Dong, Yonggan Fu, Peter Belcak, Hanrong Ye, Hongxu Yin, Yi Dong, Evelina Bakhturina, Tao Yu, Yejin Choi, Jan Kautz, and Pavlo Molchanov. Toolorchestra: Elevating intelligence via efficient model and tool orchestration, 2025. URL https: //arxiv.org/abs/2511.21689.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10: 539–554, 2022. doi: 10.1162/tacl_a_00475. URL https://aclanthology.org/2022.tacl-1.31/.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Haozhen Zhang, Tao Feng, and Jiaxuan You. Router-r1: Teaching llms multi-round routing and aggregation via reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

###### Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025b.

### A. Experimental Details

- A.1. Experimental Details for Model Routing

Implementation Details. We use the same evaluation protocol as Router-R1 for controlled comparison with all routing baselines. We use Qwen2.5-3B (Qwen, 2024) as the orchestrator, and the model pool consists of Qwen2.5-7B (Qwen, 2024), LLaMA-3.1-8B (Grattafiori et al., 2024), LLaMA-3.1-70B (Grattafiori et al., 2024), Mistral-7B (Jiang et al., 2023a), Mixtral-8x22B (Jiang et al., 2024), and Gemma-2-27B (Gemma et al., 2024). Routing operates in two modes: (1) search mode, where the orchestrator selects a model from the pool to perform subtasks (provide knowledge or solve the subtask); and (2) answer mode, where the orchestrator aggregates intermediate results and produce the final answer. We set the max number of turns to 4. We evaluate the performance using Exact Match (EM) and efficiency using total completion cost. SkillOrchestra is trained in a low-data regime: by default, we select k (k < 50) samples from each dataset to train the Skill Handbook and k additional samples for validation and handbook retrieval. We use SkillOrchestra+ to denote the best performance obtained by switching among different orchestrator models within the same agent pool while using the same learned Skill Handbook.

- A.2. Experimental Details for Agent Orchestration

Implementation Details. We follow the same evaluation protocol and experimental setup as ToolOrchestra to ensure a controlled and comparable evaluation. We consider three operational modes: For ψ = search, the allowable tools are Tsearch = {WebSearch, LocalSearch}, where WebSearch uses the Tavily API and LocalSearch uses a FAISS index built with Qwen3-Embedding-8B (Zhang et al., 2025b). The model set is ℳsearch = {GPT-5, GPT-5-mini, Qwen3-32B}. Valid agents are compositions (m, Tsearch) with m ∈ ℳsearch. For ψ = code, the tool set is Tcode = {PythonExec} operating in a sandbox, and ℳcode = {GPT-5, GPT-5-mini, Qwen2.5-Coder-32B}. Valid agents are (m, Tcode) with m ∈ ℳcode. For ψ = answer, no external tools are used (Tanswer = ∅), and ℳanswer = {GPT-5, GPT-5-mini, Llama-3.3-70B-Instruct, Qwen3-32B, Qwen2.5-Math-72B, Qwen2.5-Math-7B}. Valid agents are (m,∅) with m ∈ ℳanswer. The maximum interaction horizon is 50 turns. Final answers are evaluated for accuracy using GPT-5-mini as a judge, and total system cost (USD) is measured.

### B. Skill-Grounded Agent Routing Algorithm Pseudocode

We present an algorithm block for Skill-grounded Agent Routing in Algorithm 1. A concrete illustration can be found in Figure 3 (Deployment).

### C. A Closer Look at Model Selection: SkillOrchestra vs. ToolOrchestra

Skill-grounded routing leads to more efficient tool-model allocation (RQ3). To understand the benefits of SkillOrchestra compared to ToolOrchestra, we also take a closer look at the model selection ratio at each operational mode. We found that the cost reduction of SkillOrchestra comes from smarter allocation of models across different operational modes, rather than simply reducing the number of calls. In search mode, ToolOrchestra routes 99.7% of calls to GPT-5-mini, whereas SkillOrchestra instead uses Qwen3-32B (also

Algorithm 1 Skill-Grounded Agent Routing by Orchestrator 𝒪

Input :State st; query handbook ℋq; cost weight λc Output :Selected mode ψt, agent At, trace zt, observation ot, updated state st+1

▷ Mode selection Select operational mode ψt ∼ πmode(⋅ ∣ st; ℛψ)

▷ Retrieve active skills Retrieve active skills Σt ⊆ Σψt from ℋq

##### ▷ Competence-aware routing foreach A ∈ 𝒜ψt do

▷ posterior-mean competence from estimated stats in the Handbook P̂(A) ← ∑σ∈Σt wt,σ

αA,σ αA,σ + βA,σ

▷ utility = competence - mode-specific cost U(A) ← P̂(A) − λc ⋅ ĈA(ψt)

At ← argmaxA∈𝒜ψt U(A)

▷ Execute + state transition (zt,ot) ← Execute(At,ψt,st) // zt = agent trace, ot = env observation st+1 ← UpdateState(st, ψt, At, zt, ot)

the cheapest) for 100% of search calls, identifying it as sufficiently capable and more cost-efficient for the search task. In answer mode, ToolOrchestra similarly exhibits routing collapse, routing 97.9% of calls to GPT-5. SkillOrchestra distributes answer generation more strategically: GPT-5 is used in 58.4% of calls, with the remainder handled by cheaper or specialized models such as GPT-5-mini (10.0%) and Qwen3-32B or math-expert models. This diversification allows the system to reserve expensive models for truly difficult reasoning steps while offloading simpler synthesis or domain-specific subtasks to more efficient models.

### D. Demonstrations of Skill-Aware Orchestration

We provide full execution traces of the skill-based router in Figures 7–9, along with the orchestration instruction template in Figure 10. The instruction integrates the task query, execution context, and the selected Skill Handbook used for routing decisions.

AMC Example: Skill-Based Router Corrects via Multi-turn Routing

✓ Skill Handbook 2 external model calls (<search>) ✓ Correct

###### Skill Router Instruction

You are a skill-based model router. You are selecting the best model to answer a question by analyzing

a question to identify required skills and their importance related to this question. ## Learned Skill Definitions (from validation) ### Algebra and Functions Symbolic manipulation and equation-solving across rational expressions, logarithms/exponents,

polynomials via Vieta, and trigonometric identities and parameters.

- - Rational/linear manipulation (nested fractions, clearing denominators, simple systems): Evaluate nested or continued fractions, reduce to irreducible form, and set up/solve linear relations from

worded constraints. Examples: Problem 1: Compute 3 + 1/(3 + 1/(3 + 1/3)) as an irreducible fraction and return m+n., Problem 12: Three numbers sum to 96 with linear relations; find |first - second|.

- - Logarithm/exponent identities and metric constraints: Use log laws to simplify expressions and translate distance conditions on the number line into equations in log variables. Examples: Problem 3: Distance between log_6 x and log_6 9 equals twice another distance; find product of solutions., Problem 6: Evaluate (log 5)^3 + (log 20)^3 + (log 8)(log 0.25).
- - Vieta and symmetric sums with parameter shifts: Extract sums/products of roots from polynomial coefficients and evaluate expressions after shifting variables (e.g., edges increased by a constant). Examples: Problem 7: Roots are box dimensions; edges lengthened by 2; compute new volume via Vieta

.

- - Trigonometric identity reduction and parameter interval analysis: Rewrite trig sums/products (e.g

., sin x + sin 2x) and determine parameter ranges guaranteeing multiple solutions or specific solution behaviors. Examples: Problem 9: For a*(sin x + sin 2x) = sin 3x, find all a producing more than one solution and compute p+q+r.

### Geometry and Transformations Spatial reasoning in 2D/3D using vectors, coordinates, isometries, and complex-plane interpretations

of loci and areas.

- - 3D vector/coordinate geometry and dot products: Model regular solids with coordinates, use midpoints and vectors, and compute angles via dot products and norms. Examples: Problem 4: In regular tetrahedron ABCD with M midpoint of AB, find $\cos(\angle CMD)$ as

p/q.

- - Complex-plane geometry: loci, polygons, and area optimization: Interpret complex constraints as geometric loci (segments, disks), compute Minkowski sums (stadium/rounded rectangle areas), and analyze polygons formed by z and 1/z under quadratic relations. Examples: Problem 5: Region from sum of a segment [3, 4i] and a unit disk; find closest integer to

area., Problem 14: For z^2 - cz + 10 = 0, quadrilateral with z1, z2, 1/z1, 1/z2 has maximal area; find $c=\sqrt{m}$.

- - Coordinate geometry with distance constraints in special quadrilaterals: Place figures in coordinates, encode equal-leg/parallel-side constraints, and solve using distance equations to find side ratios. Examples: Problem 13: Isosceles trapezoid with PA=1, PB=2, PC=3, PD=4; find BC/AD.
- - Composition of plane isometries and periodicity: Represent rotations and reflections (matrices or angle-line representations), compose varying-parameter isometries, and determine the least n returning a point to itself. Examples: Problem 10: Find minimal n so $T_1\circ T_2\circ \cdots \circ T_n$ sends (1,0) back to itself.

### Number Theory and Diophantine Analysis Integer-structure problems involving Pell equations, valuations, and divisibility properties of

rational sums.

- - Pell-type equations for figurate-number squares: Convert conditions like triangular numbers being squares into Pell equations, use fundamental solutions/recurrences to generate the next solutions. Examples: Problem 8: Find the fourth triangular number that is also a square; sum its digits.
- - Harmonic denominators vs LCM via p-adic valuations: Analyze reduced denominators of harmonic numbers, compare to LCM(1..n), and count n where strict inequality holds using prime-power valuations and cancellation. Examples: Problem 15: For 1 <= n <= 22, count n with k_n < L_n.

### Combinatorics and Discrete Structures Counting and structural reasoning for pairings and permutations under process constraints.

- - Constrained pairing/matching counts: Model pairings with inequality or dominance constraints, assess feasibility (often via greedy/ordering), and count valid matchings. Examples: Problem 2: Number of ways to pair 1..14 so larger >= 2x smaller in each pair.
- - Permutation process modeling (passes/runs) and counting: Translate left-to-right multi-pass selection procedures into properties like increasing runs or pile counts and enumerate

permutations achieving a given number of passes. Examples: Problem 11: Count orderings of 13 cards that are picked up in exactly two passes.

## Model Performance (learned from validation) ### LLaMA-3.1-70B-Instruct Overall: 15% success (3/20) Skill scores:

- - Pell-type equations for figurate-number squares: 100%
- - Harmonic denominators vs LCM via p-adic valuations: 50%
- - Complex-plane geometry: loci, polygons, and area optimization: 33%
- - Rational/linear manipulation (nested fractions, clearing denominators, simple systems): 0%
- - Constrained pairing/matching counts: 0%

Strengths: geometry.complex_plane_loci_area Weaknesses: geometry.3d_dot_product, algebra.rational_linear_manipulation, math-heavy reasoning in

this sample

### Gemma-2-27B-Instruct Overall: 10% success (2/20) Skill scores:

- - Harmonic denominators vs LCM via p-adic valuations: 50%
- - Rational/linear manipulation (nested fractions, clearing denominators, simple systems): 25%
- - Constrained pairing/matching counts: 0%
- - Logarithm/exponent identities and metric constraints: 0%
- - 3D vector/coordinate geometry and dot products: 0%

Strengths: algebra.rational_linear_manipulation, basic_algebra_word_problems Weaknesses: algebra.logs_and_exponents, quantitative reasoning in this sample, geometry.

complex_plane_loci_area

### Qwen2.5-7B-Instruct Overall: 25% success (5/20) Skill scores:

- - 3D vector/coordinate geometry and dot products: 100%
- - Harmonic denominators vs LCM via p-adic valuations: 100%
- - Complex-plane geometry: loci, polygons, and area optimization: 33%
- - Rational/linear manipulation (nested fractions, clearing denominators, simple systems): 25%
- - Constrained pairing/matching counts: 0%

Strengths: algebra.rational_linear_manipulation, geometry.3d_dot_product (observed), general_backup Weaknesses: algebra.logs_and_exponents, combinatorics.constrained_pairing_matching, algebra.

rational_linear_manipulation

### Mistral-7B-Instruct Overall: 25% success (5/20) Skill scores:

- - Harmonic denominators vs LCM via p-adic valuations: 100%
- - Logarithm/exponent identities and metric constraints: 50%
- - Coordinate geometry with distance constraints in special quadrilaterals: 50%
- - Rational/linear manipulation (nested fractions, clearing denominators, simple systems): 25%
- - Constrained pairing/matching counts: 0%

Strengths: algebra.logs_and_exponents (partial), geometry.coordinate_distance_quadrilaterals, algebra. rational_linear_manipulation Weaknesses: geometry.3d_dot_product, algebra.rational_linear_manipulation, combinatorics. permutation_passes_runs

### Mixtral-8x22B-Instruct Overall: 20% success (4/20) Skill scores:

- - Pell-type equations for figurate-number squares: 100%
- - Rational/linear manipulation (nested fractions, clearing denominators, simple systems): 50%
- - Logarithm/exponent identities and metric constraints: 50%
- - Constrained pairing/matching counts: 0%
- - 3D vector/coordinate geometry and dot products: 0%

Strengths: algebra.logs_and_exponents, algebra.rational_linear_manipulation, general algebraic manipulation Weaknesses: combinatorics.constrained_pairing_matching (no clear advantage), general, geometry. coordinate_distance_quadrilaterals

### LLaMA-3.1-8B-Instruct Overall: 20% success (4/20) Skill scores:

- - Vieta and symmetric sums with parameter shifts: 100%
- - Logarithm/exponent identities and metric constraints: 50%
- - Harmonic denominators vs LCM via p-adic valuations: 50%
- - Rational/linear manipulation (nested fractions, clearing denominators, simple systems): 25%
- - Constrained pairing/matching counts: 0%

Strengths: algebra.logs_and_exponents (partial), algebra.rational_linear_manipulation Weaknesses: algebra.rational_linear_manipulation, geometry.coordinate_distance_quadrilaterals,

combinatorics.permutation_passes_runs ## Cost Tiers (cheapest to most expensive)

- - Cheap: Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, Mistral-7B-Instruct
- - Medium: Gemma-2-27B-Instruct
- - Expensive: LLaMA-3.1-70B-Instruct, Mixtral-8x22B-Instruct ## Task

- 1. First, analyze the question below and identify which skills are needed, along with the percentage/ weight of each skill (how important each skill is for answering this question).

Output your analysis in the following format inside <skill_analysis> tags: <skill_analysis> {

"required_skills": [

{"skill_id": "skill.id", "percentage": 50}, {"skill_id": "skill.id", "percentage": 30}, ...

], "reasoning": "Brief explanation of why these skills are needed"

} </skill_analysis>

The percentages should sum to approximately 100 (they don't need to be exact, but should reflect relative importance).

- 2. After providing the skill analysis, reflect on which model is best suited based on the skills required and model performance data above.
- 3. Route to that model using <search> tags and provide final answer in <answer>...</answer>

Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a

query inside <search> LLM-Name:Your-Query </search>. !!! STRICT FORMAT RULES for <search>: !!!

+ You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct,

Mistral-7B-Instruct, Mixtral-8x22B-Instruct, Gemma-2-27B-Instruct].

+ You MUST replace Your-Query with the EXACT same question as the original question below (DO NOT CHANGE IT).

+ NEVER copy or paste model descriptions into <search>.

+ NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly.

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about:

+ Why external information is needed.

+ Which skills from the catalog are required for this question.

+ Which model is best suited based on the model performance data above.

When you call an LLM, the response will be returned between <information> and </information>. You are encouraged to explore and utilize different LLMs to better understand their respective

strengths and weaknesses.

If you find that no further external knowledge is needed, you can directly provide your final answer to the original question inside <answer> ... </answer>, without additional explanation or illustration.

If you think none of the models listed have the necessary skills to answer this question directly, you can route to the model with the highest overall pass rate of models in the pool to get more information.

Question: How many complex numbers satisfy the equation $z^5=\overline{z}$, where $\overline{z}$ is the conjugate of the complex number $z$?

###### Turn 1 — skill analysis + first external call

<skill_analysis> {

"required_skills": [ {"skill_id": "algebra.logs_and_exponents", "percentage": 70}, {"skill_id": "algebra.rational_linear_manipulation", "percentage": 30}

], "reasoning": "Use polar form; equate magnitudes and angles; solve exponent/angle constraints."

} </skill_analysis> <think> To solve this problem, we need to manipulate the given equation involving complex numbers and their

conjugates. This requires skills in algebra, specifically dealing with logarithms and exponents, as well as rational/linear manipulation.

</think> <search> Mixtral-8x22B-Instruct: How many complex numbers satisfy the equation $z^5=\overline{z}$, where $\

overline{z}$ is the conjugate of the complex number $z$? </search> <information> The equation $z^5=\\overline{z}$ implies that $|z^5|=|\\overline{z}|$, which simplifies to $|z|^5=|z|$

. This implies that $|z|=0$ or $|z|=1$.

- - If $|z|=0$, then $z=0$.
- - If $|z|=1$, then $z=e^{i\\theta}$ for some $\\theta\\in\\mathbb{R}$. Substituting this into the original equation, we get $e^{5i\\theta}=e^{-i\\theta}$. This implies that $5\\theta=-\\theta+2k\\pi$ for some $k\\in\\mathbb{Z}$. Solving for $\\theta$, we get $\\theta=\\frac{k\\pi}{3}$ for some $k\\in\\mathbb{Z}$. Therefore, there are infinitely many complex numbers that satisfy the equation $z^5=\\overline{z}$.

These are $z=0$ and $z=e^{i\\frac{k\\pi}{3}}$ for any $k\\in\\mathbb{Z}$. </information>

###### Turn 2 — second call to resolve counting

<skill_analysis> {

"required_skills": [ {"skill_id": "algebra.logs_and_exponents", "percentage": 60}, {"skill_id": "algebra.rational_linear_manipulation", "percentage": 40}

], "reasoning": "Need correct restriction on k from \theta in [0,2\pi)."

} </skill_analysis>

<think> We must count distinct solutions modulo 2\pi; query a higher-capability model to confirm. </think> <search> LLaMA-3.1-70B-Instruct: How many complex numbers satisfy the equation $z^5=\overline{z}$, where $\

overline{z}$ is the conjugate of the complex number $z$? </search> <information> [...] Substituting $z = re^{i\theta}$ into the equation $z^5=\overline{z}$, we get $(re^{i\theta})^5 = re^{-i\theta}$, i.e., $r^5e^{5i\theta} = re^{-i\theta}$. Equating magnitudes gives $r^5 = r$, so $r = 0$ or $r = 1$. Equating arguments gives $5\theta = -\theta + 2k\pi$, so $6\theta = 2k\pi$ and $\theta = \frac{k\pi}{3}$. Since $\theta \in [0,2\pi)$, we have $0 \le \frac{k\pi}{3} < 2\pi$, so $k=0,1,2,3,4,5$.

- For $r = 0$, there is exactly one solution ($z = 0$).
- For $r = 1$, there are 6 possible values of $\\theta$ corresponding to the 6 possible values of $k$. Therefore, the total number of complex numbers that satisfy the equation is 1 (for $r = 0$) + 6 ( for $r = 1$) = 7.

</information>

###### Turn 3 — final answer

<answer>7</answer>

- Figure 7: Full trace of an AMC example. The router first analyzes the required skills, calls Mixtral-8x22B-Instruct, which derives the correct form but miscounts the solutions; it then routes to LLaMA-3.1-70B-Instruct to correctly restrict θ ∈ [0,2π) and count distinct solutions, producing the correct final answer.

AMC Example: Skill-Based Router Recognizes No External Model Is Needed and Answers Autonomously

✓ Skill Handbook 0 external model calls ✓ Correct

###### Skill Router Instruction

You are a skill-based model router. You are selecting the best model to answer a question by analyzing

a question to identify required skills and their importance related to this question. ## Learned Skill Definitions (from validation) ### Algebraic Modeling and Equation Solving Setting up and solving algebraic relationships, including polynomial constraints, radical systems, and

vector-rate models.

- - Polynomial interpolation from structural constraints: Use given values and leading coefficients to determine polynomial constants or invariant sums (e.g., P(0)+Q(0)) without full reconstruction.

Examples: Problem 1: Quadratic polynomials P and Q with opposite leading coefficients both pass through two points; find P(0)+Q(0).

- - Linearization of symmetric radical systems: Transform symmetric systems with square roots via substitutions, factoring, and controlled squaring to solve for variables or symmetric expressions. Examples: Problem 7: Solve a 3-equation symmetric radical system and compute [(1-x)(1-y)(1-z)]^2.
- - Vector kinematics in flowing media: Model motion with currents/winds by decomposing velocities into components relative to ground and medium; impose geometric/temporal constraints to solve for

unknowns. Examples: Problem 11: Two swimmers in a flowing river head to a common point on the opposite bank; determine the downstream offset D.

### Euclidean Geometry and Transformations

Reasoning about plane and spatial geometry using angle bisectors, tangencies, homothety/inversion, and 3D-to-2D section relations.

- - Circle tangency via homothety/inversion: Exploit homothety or inversion to relate radii, centers, and intersection loci of tangent circles and inscribed figures. Examples: Problem 14: Three equal tangent circles inside a circumcircle around an equilateral triangle; find side of the inner equilateral from circle intersections.
- - Angle bisectors and (ex)center configurations: Use properties of angle bisectors, incenter/ excenters, and their intersections to compute distances in polygons, especially trapezoids and triangles. Examples: Problem 9: In an isosceles trapezoid, angle bisectors meet at P and Q; find PQ given side lengths.
- - Sphere-plane sections and power-of-a-point relations: Relate congruent circular cross-sections of spheres to sphere radii and plane offsets; compute distances between projected centers using tangency and power. Examples: Problem 2: Three tangent spheres cut by a plane in congruent circles with centers A,B,C;

given AB^2, find AC^2.

- - Equal-perimeter splitting lines via reflection: Model equal-perimeter partitions by reflecting across sides to convert perimeter conditions to straight-line distance constraints. Examples: Problem 6: Splitting lines through midpoints M and N in a triangle intersect; deduce side lengths/structure from equal-perimeter property.

### Combinatorics and Discrete Structures Counting and optimization in discrete settings with indicator methods, parity constraints, pattern

avoidance, and digit-product arrangements.

- - Indicator variables and linearity over subsets: Compute sums over families of sets by summing elementwise contributions with indicator variables and symmetry constraints like |A|=|B|. Examples: Problem 4: Evaluate $S_n = \sum |A \cap B|$ over ordered pairs of subsets with equal cardinalities.
- - AP-free (pattern-avoidance) sequence design: Select or count integers under monotonicity while forbidding k-term arithmetic progressions; use modular classes and structural constraints. Examples: Problem 12: Count integer pairs (a,b) so the 8-term increasing sequence contains no 4term arithmetic progression.
- - Parity-constrained arrangements with identical pairs: Count permutations where identical items must be separated by an even/odd number of positions; apply parity classes and inclusionexclusion or structural bijections. Examples: Problem 15: Probability a random arrangement of two each of six colors is 'even' (parity

gap constraint).

- - Digit-based discrete optimization of product ratios: Assign distinct digits to maximize/minimize product differences/ratios under constraints using inequalities, rearrangement, and greedy swaps. Examples: Problem 13: Minimize a ratio of product difference over product using digits 1-9 exactly

once; report m+n.

### Number Theory and Modular Reasoning Reasoning with modular arithmetic, base representations, reduced fractions of repeating decimals, and

roots-of-unity congruences.

- - Repeating decimal reduction via gcd/phi structure: Convert repeating decimals to fractions with denominators of the form 9, 99, 9999..., reduce by gcd, and count distinct reduced numerators leveraging multiplicative structure. Examples: Problem 5: Count distinct numerators obtained from all 4-digit repeating decimals when reduced to lowest terms.
- - Base conversion as digit-constrained Diophantine equations: Translate cross-base digit identities into linear equations with digit bounds; solve for digits satisfying both base systems. Examples: Problem 8: Find three-digit base-10 abc whose base-9 representation is bca_9.
- - Roots-of-unity angle congruences and counting: Interpret complex constants as $e^{i\theta}$; equate powers via modular congruences on arguments and count solutions within given bounds.

Examples: Problem 10: With $w = \frac{\sqrt{3}+i}{2}$ and $z = \frac{-1+i\sqrt{3}}{2}$, count $(r,s)

\le 100$ satisfying $i w^r = z^s$. ## Model Performance (learned from validation) ### LLaMA-3.1-70B-Instruct Overall: 30% success (6/20)

Skill scores:

- - Sphere-plane sections and power-of-a-point relations: 100%
- - Base conversion as digit-constrained Diophantine equations: 100%
- - Parity-constrained arrangements with identical pairs: 100%
- - Circle tangency via homothety/inversion: 50%
- - Indicator variables and linearity over subsets: 50%

Strengths: 3D geometry with spheres/planes and Power of a Point, Number base representation and digitconstraint puzzles, Best overall reliability in this batch Weaknesses: Circle tangency/homothety/inversion geometry, Algebra with polynomial interpolation/ constraint reasoning, Subset-sum/inclusion-exclusion style combinatorics

### Mistral-7B-Instruct Overall: 5% success (1/20) Skill scores:

- - Indicator variables and linearity over subsets: 50%
- - Polynomial interpolation from structural constraints: 0%
- - Sphere-plane sections and power-of-a-point relations: 0%
- - Circle tangency via homothety/inversion: 0%
- - Angle bisectors and (ex)center configurations: 0%

Strengths: Occasional general reasoning success (inconsistent) Weaknesses: Euclidean geometry (especially tangency/homothety/inversion), 3D geometry with spheres/

planes, Polynomial interpolation constraints

### Gemma-2-27B-Instruct Overall: 0% success (0/20) Skill scores:

- - Polynomial interpolation from structural constraints: 0%
- - Sphere-plane sections and power-of-a-point relations: 0%
- - Circle tangency via homothety/inversion: 0%
- - Angle bisectors and (ex)center configurations: 0%
- - Indicator variables and linearity over subsets: 0%

Weaknesses: 3D sphere-plane geometry, Circle tangency/homothety/inversion, Polynomial interpolation constraints

### Mixtral-8x22B-Instruct Overall: 0% success (0/20) Skill scores:

- - Polynomial interpolation from structural constraints: 0%
- - Sphere-plane sections and power-of-a-point relations: 0%
- - Circle tangency via homothety/inversion: 0%
- - Angle bisectors and (ex)center configurations: 0%
- - Indicator variables and linearity over subsets: 0%

Weaknesses: 3D sphere-plane geometry, Circle tangency/homothety/inversion, Polynomial interpolation constraints

### Qwen2.5-7B-Instruct Overall: 0% success (0/20) Skill scores:

- - Polynomial interpolation from structural constraints: 0%
- - Sphere-plane sections and power-of-a-point relations: 0%
- - Circle tangency via homothety/inversion: 0%
- - Angle bisectors and (ex)center configurations: 0%
- - Indicator variables and linearity over subsets: 0%

Weaknesses: Geometry-heavy tasks, Algebraic modeling/constraints, 0% on tested tasks ### LLaMA-3.1-8B-Instruct Overall: 0% success (0/20) Skill scores:

- - Polynomial interpolation from structural constraints: 0%
- - Sphere-plane sections and power-of-a-point relations: 0%
- - Circle tangency via homothety/inversion: 0%
- - Angle bisectors and (ex)center configurations: 0%

- - Indicator variables and linearity over subsets: 0%

Weaknesses: All observed olympiad-style math skills in this batch, All tested math skills in this batch (0% accuracy)

## Cost Tiers (cheapest to most expensive)

- - Cheap: Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, Mistral-7B-Instruct
- - Medium: Gemma-2-27B-Instruct
- - Expensive: LLaMA-3.1-70B-Instruct, Mixtral-8x22B-Instruct ## Task

- 1. First, analyze the question below and identify which skills are needed, along with the percentage/ weight of each skill (how important each skill is for answering this question).

Output your analysis in the following format inside <skill_analysis> tags: <skill_analysis> {

"required_skills": [

{"skill_id": "skill.id", "percentage": 50}, {"skill_id": "skill.id", "percentage": 30}, ...

], "reasoning": "Brief explanation of why these skills are needed"

} </skill_analysis>

The percentages should sum to approximately 100 (they don't need to be exact, but should reflect relative importance).

- 2. After providing the skill analysis, reflect on which model is best suited based on the skills required and model performance data above.
- 3. Route to that model using <search> tags and provide final answer in <answer>...</answer>

Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a

query inside <search> LLM-Name:Your-Query </search>. !!! STRICT FORMAT RULES for <search>: !!!

+ You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, Mistral-7B-Instruct, Mixtral-8x22B-Instruct, Gemma

-2-27B-Instruct].

+ You MUST replace Your-Query with the EXACT same question as the original question below (DO NOT CHANGE IT).

+ NEVER copy or paste model descriptions into <search>.

+ NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly.

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about:

+ Why external information is needed.

+ Which skills from the catalog are required for this question.

+ Which model is best suited based on the model performance data above.

When you call an LLM, the response will be returned between <information> and </information>. You are encouraged to explore and utilize different LLMs to better understand their respective strengths and weaknesses.

If you find that no further external knowledge is needed, you can directly provide your final answer to the original question inside <answer> ... </answer>, without additional explanation or illustration. For example: <answer> Beijing </answer>.

+ Important: You must not output the placeholder text "<answer> and </answer>" alone.

+ You must insert your actual answer between <answer> and </answer>, following the correct format.

+ You must not output the model name or query between <answer> and </answer>. If you think none of the models listed have the necessary skills to answer this question directly, you can route to the model with the highest overall pass rate of models in the pool to get more information.

Question: A $3-4-5$ right triangle is inscribed in circle $A$, and a $5-12-13$ right triangle is inscribed in circle $B$. Find the ratio of the area of circle $A$ to the area of circle $B$. The final answer can be written in the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. What is $m+n$?

###### Turn 1 - direct answer

<think> To solve this problem, we need to find the radii of the circles in which the right triangles are

inscribed. For a right triangle, the hypotenuse is the diameter of the circumscribed circle. Therefore, we can find the radius of each circle by dividing the length of the hypotenuse by 2.

For the $3-4-5$ right triangle, the hypotenuse is 5, so the radius of circle $A$ is $\frac{5}{2}$. For

the $5-12-13$ right triangle, the hypotenuse is 13, so the radius of circle $B$ is $\frac{13}{2} $.

The area of a circle is given by $\pi r^2$. Therefore, the area of circle $A$ is $\pi \left(\frac {5}{2}\right)^2 = \pi \frac{25}{4}$, and the area of circle $B$ is $\pi \left(\frac{13}{2}\right) ^2 = \pi \frac{169}{4}$.

The ratio of the area of circle $A$ to the area of circle $B$ is: \[ \frac{\pi \frac{25}{4}}{\pi \frac{169}{4}} = \frac{25}{169} \] Since 25 and 169 are relatively prime, the ratio is $\frac{25}{169}$. Therefore, $m+n = 25+169 = 194$.

No further information is needed, so I can directly provide the answer. </think> <answer>194</answer>

- Figure 8: Full trace of an AMC example. While the router (Qwen2.5-3B) has access to the Skill Handbook, it autonomously determines that the problem can be solved using its internal capability, refrains from issuing any <search> calls, and produces the correct final answer. This example highlights the flexibility of skill-based routing, where external model invocation is optional rather than mandatory.

PopQA Example: Adaptive Skill-Based Routing Recovers from a Noisy Initial Response

✓ Skill Handbook 3 external calls ✓ Correct

###### Skill Router Instruction

You are a skill-based model router. You are selecting the best model to answer a question by analyzing

a question to identify required skills and their importance related to this question. ## Learned Skill Definitions (from validation) ### Entity Resolution and Attributes Identify and link entities from descriptive phrases and retrieve their attributes or locations.

- - Descriptor-based Entity Linking: Resolve a specific entity from multi-clause descriptors (titles, awards, roles) without the name explicitly given. Examples: Problem 12: Who is the American singer-songwriter, who won an award for Best Female Video at the 2009 MTV Video Music Awards, and wrote a song for the 'AT&T Team USA Soundtrack'?, Problem 1: Who is the English Professional Footballer that is the niece of a former footballer born in the same place as where he plays?
- - Membership/Role Resolution: Identify an entity based on group membership and an additional role or credit.

Examples: Problem 4: What member of the South Korean-Chinese boy group EXO stars in the upcoming movie "The Underdog"?, Problem 9: Maha Kali is an EP by what band from Stromstad that was formed in 1989?

- - Attribute/Location Retrieval: Extract a specific attribute or location associated with an entity from knowledge bases. Examples: Problem 15: Japanese Weekend School of New York has offices in the entertainment complex

in what section of New Rochelle?

### Relational Composition and Constraints Answer questions requiring chaining multiple relations and satisfying constraints across entities.

- - Multi-hop Bridge Reasoning: Combine two or more linked facts (A->B->C) to derive the answer. Examples: Problem 6: The Battle of Cambrai took place during a series of offensives that ended on what date?, Problem 3: Liz Rose has co-written songs with which artist including "White Horse" and "You Belong with Me"?
- - Relational Constraint Satisfaction: Apply constraints such as shared attributes (same birthplace/ place) or time/place filters across relations to identify the correct entity. Examples: Problem 1: Who is the English Professional Footballer that is the niece of a former footballer born in the same place as where he plays?, Problem 9: Maha Kali is an EP by what band from Stromstad that was formed in 1989?
- - Reverse Relation Traversal: Start from works or properties and infer the originating entity via reverse edges (e.g., song -> artist). Examples: Problem 3: Liz Rose has co-written songs with which artist including "White Horse" and " You Belong with Me"?, Problem 10: A Pair of Brown Eyes and Wild Mountain Thyme is based from what

artists song?

### Temporal and Ordinal Reasoning Interpret dates, periods, and ordinal descriptors to resolve time-based queries.

- - Event-Date Alignment: Map events to their dates directly or via higher-level campaigns/series. Examples: Problem 6: The Battle of Cambrai took place during a series of offensives that ended on what date?
- - Ordinal Title/Sequence Disambiguation: Use ordinal descriptors (e.g., third, first) within a known sequence to identify the correct entity and associated dates.

Examples: Problem 5: When did the the Antonine who was known as the third of the five good emperors live?

- - Temporal Qualifier Interpretation: Interpret temporal qualifiers such as 'first', 'upcoming', or year references to locate the relevant time frame. Examples: Problem 4: What member of the South Korean-Chinese boy group EXO stars in the upcoming movie "The Underdog"?, Problem 8: Shirley Breeden won her first Senate term in a narrow upset over the politician who was a member of what branch of the armed forces?

### Comparative, Classification, and Set Reasoning Make comparisons, classify options by properties, and find common attributes across entities.

- - Property-based Option Selection: Choose among given options based on a specific property filter. Examples: Problem 2: Which is a black-and-white film, Flying Padre or Inside Job?
- - Numeric/Quantitative Comparison: Compare numeric attributes (counts, totals) across entities to determine the greater/lesser. Examples: Problem 14: Which player won more Grand Slam titles, Kevin Ullyett or Billie Jean King?
- - Set Intersection/Common Attribute Identification: Find a shared attribute (e.g., profession) between two entities. Examples: Problem 13: What profession does Leonty Magnitsky and Leonid Khachiyan have in common?

### Language Robustness and Causal Inference Handle noisy phrasing and infer causal or thematic relations from text.

- - Noisy/Paraphrase Normalization: Parse ungrammatical or awkward phrasing and map it to a coherent structured query. Examples: Problem 10: A Pair of Brown Eyes and Wild Mountain Thyme is based from what artists song ?, Problem 11: In which song was written by singer-songwriter Taylor Swift and shares the optimistic lyrical message to a song called "Yodel It!"?
- - Causal/Narrative Why-Reasoning: Answer why-questions by identifying causes, motives, or precipitating events in historical narratives. Examples: Problem 7: Why did Rudolf Hess stop serving Hitler in 1941?
- - Thematic/Message Similarity Inference: Infer thematic similarity (e.g., optimistic message) between works when not explicitly linked by facts. Examples: Problem 11: In which song was written by singer-songwriter Taylor Swift and shares the optimistic lyrical message to a song called "Yodel It!"?

## Model Performance (learned from validation) ### LLaMA-3.1-70B-Instruct Overall: 43% success (26/60) Skill scores:

- - Property-based Option Selection: 100%
- - Numeric/Quantitative Comparison: 100%
- - Membership/Role Resolution: 67%
- - Thematic/Message Similarity Inference: 67%
- - Noisy/Paraphrase Normalization: 60%

Strengths: Property-based option selection, Stronger-than-peers multi-hop bridge reasoning, Paraphrase /noise normalization Weaknesses: Descriptor-based entity linking, Relational constraint satisfaction, Relational constraint satisfaction across multiple entities

### Gemma-2-27B-Instruct Overall: 48% success (29/60) Skill scores:

- - Property-based Option Selection: 100%
- - Set Intersection/Common Attribute Identification: 100%
- - Numeric/Quantitative Comparison: 100%
- - Thematic/Message Similarity Inference: 67%
- - Attribute/Location Retrieval: 50%

Strengths: Property-based option selection (top-tier), Paraphrase/noise normalization, Thematic similarity inference Weaknesses: Multi-hop bridge reasoning (mid), Descriptor-based entity linking, Relational constraint satisfaction

### Mistral-7B-Instruct Overall: 30% success (18/60) Skill scores:

- Property-based Option Selection: 100% - Numeric/Quantitative Comparison: 100% - Attribute/Location Retrieval: 39%

- Reverse Relation Traversal: 30% - Multi-hop Bridge Reasoning: 27%

Strengths: Occasional success on simple entity lookups, No demonstrated strengths on measured skills in this set, Cost/latency for very simple, single-hop queries Weaknesses: Multi-hop bridge reasoning (low), Descriptor-based entity linking, Relational constraint satisfaction

### Mixtral-8x22B-Instruct Overall: 45% success (27/60) Skill scores:

- - Property-based Option Selection: 100%
- - Membership/Role Resolution: 100%
- - Set Intersection/Common Attribute Identification: 100%
- - Numeric/Quantitative Comparison: 100%
- - Attribute/Location Retrieval: 44%

Strengths: Multi-hop bridge reasoning (strong among evaluated models), Descriptor-based entity linking (partial strength), Multi-hop bridge reasoning (large-model tier) Weaknesses: Descriptor-based entity linking, Relational constraint satisfaction, Paraphrase/noise normalization

### LLaMA-3.1-8B-Instruct Overall: 25% success (15/60) Skill scores:

- - Property-based Option Selection: 100%
- - Set Intersection/Common Attribute Identification: 67%
- - Numeric/Quantitative Comparison: 50%
- - Event-Date Alignment: 29%

- - Attribute/Location Retrieval: 22%

Strengths: Basic multi-hop (limited), No demonstrated strengths on measured skills in this set, Basic single-hop factoids Weaknesses: Multi-hop bridge reasoning (below larger models), Descriptor-based entity linking, Relational constraint satisfaction

### Qwen2.5-7B-Instruct Overall: 33% success (20/60) Skill scores:

- - Property-based Option Selection: 100%
- - Set Intersection/Common Attribute Identification: 100%
- - Numeric/Quantitative Comparison: 100%
- - Attribute/Location Retrieval: 56%
- - Multi-hop Bridge Reasoning: 21%

Strengths: None clearly demonstrated in this sample, Partial capability in paraphrase/noise normalization, Simple fact lookup with low latency Weaknesses: Multi-hop bridge reasoning (0%), Descriptor-based entity linking, Relational constraint satisfaction

## Cost Tiers (cheapest to most expensive)

- - Cheap: Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, Mistral-7B-Instruct
- - Medium: Gemma-2-27B-Instruct
- - Expensive: LLaMA-3.1-70B-Instruct, Mixtral-8x22B-Instruct ## Task

- 1. First, analyze the question below and identify which skills are needed, along with the percentage/ weight of each skill (how important each skill is for answering this question).

Output your analysis in the following format inside <skill_analysis> tags: <skill_analysis> {

"required_skills": [

{"skill_id": "skill.id", "percentage": 50}, {"skill_id": "skill.id", "percentage": 30}, ...

], "reasoning": "Brief explanation of why these skills are needed"

} </skill_analysis>

- 2. After providing the skill analysis, reflect on which model is best suited based on the skills required and model performance data above.
- 3. Route to that model using <search> tags and provide final answer in <answer>...</answer>

Every time you receive new information, you must first conduct reasoning inside <think> ... </think>. After reasoning, if you find you lack some knowledge, you can call a specialized LLM by writing a

query inside <search> LLM-Name:Your-Query </search>. !!! STRICT FORMAT RULES for <search>: !!!

+ You MUST replace LLM-Name with the EXACT name of a model selected from [Qwen2.5-7B-Instruct, LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, Mistral-7B-Instruct, Mixtral-8x22B-Instruct, Gemma-2-27B-Instruct].

+ You MUST replace Your-Query with the EXACT same question as the original question below (DO NOT CHANGE IT).

+ NEVER copy or paste model descriptions into <search>.

+ NEVER output the placeholder format <search> LLM-Name:Your-Query </search>. Always replace both parts correctly.

Before each LLM call, you MUST explicitly reason inside <think> ... </think> about:

+ Why external information is needed.

+ Which skills from the catalog are required for this question.

+ Which model is best suited based on the model performance data above. When you call an LLM, the response will be returned between <information> and </information>.

If you find that no further external knowledge is needed, you can directly provide your final answer

to the original question inside <answer> ... </answer>, without additional explanation. Question: Who was the composer of It's Going to Take Some Time?

###### Turn 1 — noisy lookup

<think> To answer this question, we need to identify the composer of the song \"It's Going to Take Some Time

\". This requires the skill of Reverse Relation Traversal, which involves starting from a work ( in this case, the song) and inferring the originating entity (the composer). We also need the skill of Attribute/Location Retrieval, as we are looking for a specific attribute (composer) associated with the song.

</think> <skill_analysis> {

"required_skills": [ {"skill_id": "Reverse Relation Traversal", "percentage": 60}, {"skill_id": "Attribute/Location Retrieval", "percentage": 40}

], "reasoning": "We need to start from the song and infer the composer, and then retrieve the composer'

s attribute."

} </skill_analysis>

<search>Qwen2.5-7B-Instruct: Who was the composer of It's Going to Take Some Time?</search> <information> The song "It's Going to Take Some Time" is a popular track from the musical "The Wedding Singer." The composer of this song is Frank Wildhorn. Frank Wildhorn is an American composer known for his work

in musical theater, including other notable works such as "Jekyll & Hyde" and "The Last 5 Years

." He collaborated with lyricist Michael Gore on this particular song. The song was featured in the 1998 film adaptation of "The Wedding Singer," which starred Adam Sandler.

Would you like more information about Frank Wildhorn or the musical "The Wedding Singer"? If not, I can provide insights on the song's significance in the context of the musical. Let me know your next step.

[... noisy repetitive content omitted: the same "let me know" lines repeat many times ...] </information>

###### Turn 2 - uncertain attribution

<think> The Qwen2.5-7B-Instruct model couldn't provide a definitive answer and includes excessive filler

content. I need to consult another model with stronger factual knowledge about music. </think> <search>LLaMA-3.1-70B-Instruct: Who was the composer of It's Going to Take Some Time?</search> <information> "It's Going to Take Some Time" is a song written by Toni Stern and Carole King. It was recorded by

Carole King in 1971. However, I am not sure if she was the composer of the song. </information>

###### Turn 3 — cross verification

<think> LLaMA-3.1-70B identifies Toni Stern and Carole King as the writers, but explicitly expresses

uncertainty about whether Carole King should be treated as the "composer" requested. Cross-verify with another model.

</think> <search>Mixtral-8x22B-Instruct: Who was the composer of It's Going to Take Some Time?</search> <information> The composer of "It's Going to Take Some Time" is Carole King. She co-wrote the song with Toni Stern. The song was first released on her 1971 album "Music" and later became a hit single for The Carpenters

in 1972. </information>

###### Turn 4 — final answer

<answer>Carole King</answer>

- Figure 9: Full trace for a PopQA example. Guided by skill-based analysis, the router detects that the initial model response is incomplete or ambiguous, re-routes to alternative models, and performs cross-model verification before finalizing the answer. This demonstrates robust recovery from intermediate errors.

Agent Orchestration Instruction (Full Specification)

You are a skill-based orchestrator for multi-step question answering. You select the best tool (search|code|answer) and model by analyzing required skills.

## Problem to Solve {{problem}}

## Current Context {{context}}

python code and execution outputs: {{execution_results}}

## Quick Reference: What You Need to Do CRITICAL REQUIREMENT: Before making ANY tool call, you MUST:

- 1. Inside <think>...</think>, analyze required skills and output in <skill_analysis> tags.
- 2. Then choose the appropriate tool with the selected model based on the skill analysis.

The context above may be long - scroll back to see the problem and context, then follow the

instructions below. ## Available Tools and Models (Baseline) Tool: search Models: search-1 ($10/M), search-2 ($2/M), search-3 ($0.8/M) Description: Search for missing information Tool: code | enhance_reasoning Models: reasoner-1 ($10/M), reasoner-2 ($2/M), reasoner-3 ($0.8/M) Description: Write and execute Python code to solve the problem Tool: answer Models: answer-1 ($10/M), answer-2 ($2/M), answer-3 ($0.9/M),

answer-4 ($0.8/M), answer-math-1 ($0.9/M), answer-math-2 ($0.2/M) Description: Extract the final answer if you think you have enough information to answer the problem ## Learned Skill Definitions ### Search Skills

- - search.control.zero_result_guardrail: Detect zero-result searches and prevent escalation to nonsearch stages until several structured reformulations and decomposition attempts have been tried.
- - search.control.stage_adherence: Adhere to the search stage's intended tools and behaviors; avoid misusing the code tool when the task requires query generation and retrieval.
- - search.control.answer_block_on_zero_results: Prevent advancing to the answer stage when searches return zero results; require iterative reformulation attempts first.
- - search.query_planning.multi-hop.list_then_filter.attribute_match: Plan retrieval by first listing entities that satisfy a temporal/role constraint, then filtering them by a specific attribute or name match (e.g., middle name matching a category like Scottish monarchs).
- - search.tool_choice.scope_awareness: Avoid search for well-known facts that can be confidently recalled; reserve search for low-confidence or obscure queries.
- - search.efficiency.tool_call_minimization: Avoids unnecessary tool invocations (search or code) when they do not contribute to retrieval, reducing cost and noise.
- - search.tool_choice.code_for_computation_only: Uses the code tool solely for numerical or dataprocessing steps after retrieval, not for information retrieval or scraping.
- - search.retrieval.robust_authoritative_search: Plan and execute single- and multi-hop web retrieval with entity/event disambiguation and numeric/temporal reasoning while explicitly prioritizing authoritative sources (e.g., Britannica/Wikipedia for general facts, IOC/FIFA/World Athletics for

sports, NOAA/NASA and peer-reviewed literature for science, official corporate and regulator sites for filings). Construct resilient queries using operators and filters (quotes, OR, -, site:,

filetype:, language, date ranges); align dates, ranges, and units; iteratively broaden/narrow via synonyms and result feedback; detect empty/malformed or low-quality-result queries and systematically retry/refine (including minimal viable queries) until at least one high-quality supporting source is obtained or escalate for clarification if recovery fails.

Examples: Compute 10 years after Neil Armstrong's birth, then search nasa.gov with date filters for events on that date; refine queries if initial filters yield no results., Who won the women's marathon at the 2024 Olympics? Prioritize olympics.com or World Athletics results; apply year filters and exact-match athlete names; retry with synonyms or broader terms if initial query is sparse.

- search.strategy.query_formulation_refinement_and_verification: Formulate clear, targeted search queries using key entities and relations; when results are weak or conflicting, rapidly iterate with synonyms, paraphrases, broader/narrower terms, constraints, operators, time filters, and source scoping. Perform lightweight verification searches even when an answer is recalled, corroborate with credible sources, and adjust conclusions based on evidence.

Examples: Need: When was the Kyoto Protocol adopted? Start with 'Kyoto Protocol adopted date'; iterate with 'Kyoto Protocol adoption 1997', add site:unfccc.int and time filters to confirm., Recall an author of a concept; perform a brief search to confirm the name and cite the seminal paper,

revising if results conflict. # Additional skills with performance data available:

- - retrieval.date_lookup
- - retrieval.entity_lookup
- - retrieval.multi_hop
- - search.basic_retrieval
- - search.control.stage_adherence.strict
- - search.cost_effectiveness
- - search.cost_efficiency
- - search.date_lookup
- - search.disambiguation
- - search.entity_lookup
- - search.entity_retrieval
- - search.fact_retrieval
- - search.factual.lookup
- - search.factual_single_hop
- - search.lookup.dates
- - search.lookup.entities
- - search.multi_hop
- - search.multi_hop.light
- - search.multi_hop.simple
- - search.multi_hop_facts
- - search.multi_hop_light
- - search.multihop
- - search.precision.general

- - search.query_planning.multi-hop
- - search.recall.general
- - search.retrieval.cost_efficiency
- - search.retrieval.dates
- - search.retrieval.factoid
- - search.retrieval.facts
- - search.retrieval.multi_hop
- - search.retrieval.multi_hop_light
- - search.retrieval.precision
- - search.retrieval.recall
- - search.retrieval.single_fact
- - search.retrieval.single_hop
- - search.single_hop
- - search.strategy.source_prioritization
- - search.web.fact_lookup
- - search.web.lookup
- - search.web.multi_hop ## Reasoning Skills
- - code.generation.codeblock_integrity: Maintain clean code blocks containing only valid Python, avoiding mixed prose or unrelated text.
- - code.math.date_calculation: Computing differences between dates, handling month/day boundaries and leap years
- - code.data.aggregation: Summing, averaging, grouping, and comparing values across lists or tables
- - code.math.multi_step: Accurate computation across multiple operations or items where mental math is error-prone.\n- code.execution.iteration_budgeting: Limits repeated tool invocations; detects diminishing returns and switches strategy (e.g., back to search/answer) to avoid wasted cost.
- - code.execution.token_economy: Generates concise code and avoids unnecessary large literals or pasted text to minimize token usage and cost.
- - code.stage_adherence: Ensure the code stage produces valid, executable Python only; avoid narrative text or non-code dumps.
- - code.execution.strategy_handoff_on_failure: After detecting that code execution cannot progress (e.g

., needs external data or repeated non-executable outputs), emit a clear handoff signal to search /answer instead of retrying.

- - code.data.tabular_parsing_ops_ranking: Extract structured tables from semi-structured text (lists, textual/markdown/CSV-like tables) and perform tabular filtering, multi-key sorting, stable tieaware ranking (top/bottom N with ties), grouping, and aggregation. Supports producing rank annotations and operating on small tables or tables returned by search.

Examples: Parse this markdown table of product sales, then return the top 10 products by revenue ( include ties) and compute average revenue per category., From this bullet list of shipped orders,

extract fields (order_id, customer, ship_date, total), filter status='shipped', sort by ship_date descending, then group by customer and sum total_spent.

- code.execution.offline_minimal_inline_structures: Decide whether code execution is warranted for a task. When appropriate, produce the smallest self-contained, offline-safe Python snippet. Replace

external I/O with tiny in-memory structures (lists/dicts/DataFrames) to perform lookups and computations. Validate outputs against requested type/format/schema/units and iterate until compliant. When code adds no value or violates constraints, abstain and answer directly while avoiding noisy outputs.

Examples: A request to map a few codes to names without using files or APIs; construct a small inline dict and return the mapped results., Summarize a CSV hosted at a URL in a no-network environment;

create a minimal inline DataFrame with representative rows, compute summary stats offline, and print validated outputs.

- code.datetime.date_difference: Compute accurate differences between dates (years/months/days) with proper inclusivity and leap-year handling, and generate/execute Python datetime code (including parsing) to achieve the same.

Examples: What is the difference between 2020-02-29 and 2021-03-01 in years, months, and days ( inclusive)?, Write Python code to parse two dates and return the year, month, and day deltas, correctly handling leap years.

# Additional skills with performance data available:

- - code.date.arithmetic
- - code.execution.syntax_self_check

- - code.math.arithmetic

## Answer Skills No dedicated skills defined. Use general performance of answer models to select the best model for the answer stage if you think

you have enough information to answer the problem. ## Model Performance (learned from validation) ### Search Models

- #### search-1 (gpt-5) Overall: 55% success (11/20) Skill scores:

- - search.factual_single_hop: 100%
- - search.multi_hop: 100%
- - search.retrieval.multi_hop: 100%
- - search.basic_retrieval: 100%
- - search.retrieval.single_hop: 67%
- - search.tool_choice.scope_awareness: 67%
- - search.factual.lookup: 60%
- - search.fact_retrieval: 60%
- - search.retrieval.cost_efficiency: 50%
- - search.cost_effectiveness: 50%
- - search.cost_efficiency: 40%
- - search.retrieval.precision: 40%
- - search.retrieval.recall: 40%
- - search.control.stage_adherence: 17%
- - search.query_planning.multi-hop: 0%
- - search.control.zero_result_guardrail: 0%
- - search.control.answer_block_on_zero_results: 0%
- - search.control.stage_adherence.strict: 0%
- - search.tool_choice.code_for_computation_only: 0%
- - search.efficiency.tool_call_minimization: 0%
- - search.retrieval.robust_authoritative_search: 0%
- - search.strategy.query_formulation_refinement_and_verification: 0%

Strengths:

- - answered from recall despite zero search results
- - robust multi-hop retrieval
- - maintained stage separation (did not misuse code tool)

Weaknesses:

- - invoked code tool with no effect (unnecessary tool usage)
- - no effective search results and no reformulation
- - issued a search call but returned no results

- #### search-2 (gpt-5-mini)

- Overall: 50% success (10/20) Skill scores:

- - search.cost_efficiency: 100%
- - search.basic_retrieval: 100%
- - search.fact_retrieval: 100%
- - search.multi_hop: 67%
- - search.retrieval.single_fact: 67%
- - search.retrieval.precision: 60%
- - search.retrieval.recall: 60%
- - search.retrieval.multi_hop: 60%
- - search.tool_choice.scope_awareness: 50%
- - search.control.stage_adherence: 20%
- - search.query_planning.multi-hop: 0%
- - search.strategy.source_prioritization: 0%
- - search.control.zero_result_guardrail: 0%
- - search.control.answer_block_on_zero_results: 0%

- - search.control.stage_adherence.strict: 0%
- - search.tool_choice.code_for_computation_only: 0%
- - search.efficiency.tool_call_minimization: 0%
- - search.retrieval.robust_authoritative_search: 0%
- - search.strategy.query_formulation_refinement_and_verification: 0%

Strengths:

- - accurate recall of both anchor years and mental subtraction
- - concise final answer
- - maintained stage separation (did not misuse code tool)

Weaknesses:

- - no effective search results
- - no reformulation after zero results
- - relied on an unverified answer without retrieval

- #### search-3 (Qwen/Qwen3-32B) Overall: 65% success (13/20) Skill scores:

- - search.retrieval.factoid: 100%
- - search.factual_single_hop: 100%
- - search.cost_efficiency: 100%
- - search.date_lookup: 100%
- - search.recall.general: 100%
- - search.precision.general: 100%
- - search.single_hop: 100%
- - search.retrieval.facts: 100%
- - search.retrieval.dates: 100%
- - search.lookup.entities: 100%
- - search.lookup.dates: 100%
- - search.multi_hop.light: 100%
- - search.factual.lookup: 100%
- - search.multi_hop.simple: 100%
- - search.basic_retrieval: 100%
- - search.fact_retrieval: 100%
- - search.disambiguation: 100%
- - search.web.lookup: 100%
- - search.entity_retrieval: 100%
- - search.multi_hop_facts: 100%
- - search.retrieval.single_fact: 100%
- - search.web.fact_lookup: 100%
- - search.web.multi_hop: 100%
- - search.cost_effectiveness: 100%
- - search.tool_choice.scope_awareness: 75%
- - search.retrieval.multi_hop: 67%
- - search.entity_lookup: 67%
- - search.retrieval.single_hop: 67%
- - search.retrieval.multi_hop_light: 67%
- - search.multi_hop_light: 67%
- - search.retrieval.precision: 67%
- - search.retrieval.recall: 67%
- - search.multi_hop: 60%
- - search.multihop: 60%
- - search.efficiency.tool_call_minimization: 50%
- - search.query_planning.multi-hop: 0%
- - search.strategy.source_prioritization: 0%
- - search.control.zero_result_guardrail: 0%
- - search.control.stage_adherence: 0%
- - search.control.answer_block_on_zero_results: 0%
- - search.control.stage_adherence.strict: 0%
- - search.tool_choice.code_for_computation_only: 0%
- - search.retrieval.robust_authoritative_search: 0%
- - search.strategy.query_formulation_refinement_and_verification: 0%

Strengths:

- - answered from recall despite zero search results
- - kept cost low with a single lightweight search attempt and no unnecessary tools
- - low-cost factual retrieval Weaknesses:
- - no effective search query produced results
- - did not reformulate after zero results
- - issued a search call but returned no results ### Reasoning Models

- #### reasoner-1 (gpt-5) Overall: 45% success (10/22) Skill scores:

- - code.math.date_calculation: 60%
- - code.math.arithmetic: 60%
- - code.data.aggregation: 60%
- - code.execution.iteration_budgeting: 8%
- - code.generation.codeblock_integrity: 0%
- - code.execution.token_economy: 0%
- - code.execution.syntax_self_check: 0%
- - code.stage_adherence: 0%
- - code.execution.strategy_handoff_on_failure: 0%
- - code.execution.offline_minimal_inline_structures: 0%
- - code.datetime.date_difference: 0%

Strengths:

- - kept code attempts to a single run (iteration budgeting)
- - lowest orchestration cost among peers
- - eventual handoff to answer after code provided no value

Weaknesses:

- - no executable code produced
- - ineffective search (no useful retrieval)
- - relied on non-code answer without tool-aligned computation/output

- #### reasoner-2 (gpt-5-mini) Overall: 62% success (13/21) Skill scores:

- - code.math.multi_step: 100%
- - code.data.aggregation: 100%
- - code.date.arithmetic: 100%
- - code.execution.iteration_budgeting: 38%
- - code.execution.token_economy: 10%
- - code.generation.codeblock_integrity: 6%
- - code.execution.syntax_self_check: 0%
- - code.stage_adherence: 0%
- - code.execution.strategy_handoff_on_failure: 0%
- - code.execution.offline_minimal_inline_structures: 0%
- - code.datetime.date_difference: 0%

Strengths:

- - arrived at the correct concise numeric answer
- - iterated with multiple tool calls
- - kept Python syntactically valid (no execution errors)

Weaknesses:

- - inserted non-code/prose into a code block
- - no printed/output result from code executions
- - ineffective search (no useful retrieval)

- #### reasoner-3 (Qwen/Qwen2.5-Coder-32B-Instruct)

- Overall: 50% success (11/22) Skill scores:

- - code.math.date_calculation: 100%
- - code.math.arithmetic: 100%
- - code.data.aggregation: 100%

- - code.data.tabular_parsing_ops_ranking: 100%
- - code.execution.strategy_handoff_on_failure: 50%
- - code.execution.iteration_budgeting: 31%
- - code.execution.token_economy: 30%
- - code.generation.codeblock_integrity: 0%
- - code.execution.syntax_self_check: 0%
- - code.stage_adherence: 0%
- - code.execution.offline_minimal_inline_structures: 0%
- - code.datetime.date_difference: 0% Strengths:
- - produced the correct concise numeric answer
- - cost-efficient with a single code call and low token usage
- - date calculations Weaknesses:
- - did not produce or display any code output (no executable/printed result)
- - overkill for simple factual lookups
- - produced no observable output from the code tool

### Answer Models

- #### answer-1 (gpt-5) Overall: 75% success (21/28) Strengths:

- - returned a single exact number
- - used direct recall without retrieval
- - directly recalled the cultural fact

Weaknesses:

- - did not perform the required multi-hop reasoning
- - no verification / disambiguation checks
- - higher cost than answer-3

- #### answer-2 (gpt-5-mini) Overall: 87% success (13/15) Strengths:

- - returned a single exact number
- - direct recall of well-known event
- - concise, exact output

Weaknesses:

- - ineffective query led to no results and no refinement attempt
- - defaulted to an unverified memorized answer
- - did not execute the multi-hop chain

- #### answer-3 (meta-llama/Llama-3.3-70B-Instruct) Overall: 62% success (8/13) Strengths:

- - returned a single exact number
- - low overall overhead
- - immediate direct recall

Weaknesses:

- - performed a search that yielded no results and did not iterate/refine
- - failed to disambiguate role timing
- - did not verify the attribute definition

- #### answer-4 (Qwen/Qwen3-32B) Overall: 47% success (7/15) Strengths:

- - returned a single exact number
- - correct direct recall
- - no tool usage

Weaknesses:

- - higher token usage compared to peers
- - no-hit search without query refinement or alternate strategy

- - no multi-hop reasoning or validation before finalizing

- #### answer-math-1 (Qwen/Qwen2.5-Math-72B-Instruct) Overall: 43% success (6/14) Strengths:

- - returned a single exact number
- - correct direct recall beyond math domain
- - concise exact answer Weaknesses:
- - misused the code tool by emitting non-Python content under the code stage
- - retrieved irrelevant/low-authority content and failed to process it
- - no error handling or validation pipeline

- #### answer-math-2 (Qwen/Qwen2.5-Math-7B-Instruct) Overall: 15% success (2/13) Strengths:

- - attempted search before answering
- - produced the correct content via recall
- - concise single-entity output Weaknesses:
- - produced an incorrect numeric answer
- - did not correctly align the two event dates before computing the difference
- - likely exact-match formatting or evaluator compliance issue

## Detailed Instructions

- STEP 1 - REQUIRED Based on the Problem and Context shown at the top, determine the next stage: (search | code | answer).

- - search: retrieve missing external information needed to answer the problem.
- - code: write and execute Python code to compute or process data.
- - answer: synthesize gathered information into the final answer.

- STEP 2 - REQUIRED FORMAT After deciding the next stage:

- 1. Reason inside <think>...</think>:

- - Why this stage is needed.
- - Which skills are required.
- - The relative importance of each skill.

- 2. Output the structured skill analysis inside <skill_analysis> tags:

<skill_analysis> {

"required_skills": [ {"skill_id": "skill.id", "percentage": 50}, {"skill_id": "skill.id", "percentage": 30}

], "reasoning": "Brief explanation of why these skills are needed."

} </skill_analysis>

- STEP 3 Choose the appropriate tool and model based on the skill analysis.

--## Final Reminders CRITICAL:

- - The <skill_analysis> block is MANDATORY.
- - It MUST appear BEFORE any tool call. IMPORTANT:
- - When calling a tool, you MUST specify the model parameter using the model alias (e.g., \"answer-1\", \"search-1\", \"reasoner-1\").
- - Use exact model names from the Available Models section above.

Now, based on the Problem and Context shown at the top, analyze what should be the next stage (search| code|answer), provide the detailed skill analysis needed for the next stage in the < skill_analysis> tags and then choose an appropriate tool.

- Figure 10: Agent orchestration instruction used for FRAMES, integrating the task query, execution history context, and the selected Skill Handbook to enable skill-aware orchestration.

