# arXiv:2508.12782v2[cs.AI]19Apr2026

## HeroBench: A Benchmark for Long-Horizon Planning and Structured Reasoning in Virtual Worlds

Petr Anokhin1,2 , Roman Khalikov2 , Stefan Rebrikov5,6 , Viktor Volkov1, Artyom Sorokin1 , Vincent Bissonnette7 1AXXX 2Lomonosov Moscow State University 5Higher School of Economics 6Kurchatov Institute 7Independent Researcher Abstract

[Figure 1]

Large language models (LLMs) perform well on step-by-step reasoning benchmarks such as mathematics and code generation, yet their ability to carry out robust long-horizon planning under realistic constraints remains insufficiently evaluated. Existing planning benchmarks often rely on abstract domains or interactive feedback, obscuring end-to-end planning failures and feasibility errors. We introduce HeroBench, a benchmark for evaluating long-horizon, hierarchical planning and structured reasoning in a complex RPG-inspired virtual world. Tasks require models to select numerically feasible equipment, reason over multi-level crafting and resource dependencies, and execute hundreds to thousands of actions as a single end-to-end plan. HeroBench integrates symbolic planning, numeric combat simulation, spatial reasoning, and resource management, while supporting scalable difficulty and adversarial distractors. HeroBench evaluates executable plans through simulation, enabling both success-based and fine-grained progress metrics, as well as detailed failure mode analysis. An evaluation of 25 state-of-the-art LLMs reveals large performance disparities rarely observed in conventional reasoning benchmarks. While reasoning models perform substantially better, no model reliably solves the hardest tasks, highlighting persistent challenges in long-horizon autonomous planning.

Figure 1: A section of the HeroBench virtual environment. A gridbased RPG-inspired world where agents must navigate, gather resources, craft equipment, and defeat enemies. Each location encodes specific environmental elements, forming the foundation for generating complex, structured tasks that challenge long-horizon reasoning and planning abilities of language models.

over extended time horizons. Despite this progress, a growing body of research suggests that LLMs are not inherently capable of planning and often require external mechanisms to validate or supervise their plans [Kambhampati et al., 2024; Valmeekam et al., 2023b].

### 1 Introduction

The rapid advancement of large language models (LLMs) has considerably expanded their applicability beyond traditional natural language processing tasks, establishing them as core components of autonomous agent systems across diverse domains. Today, LLM-driven agents are being developed to perform increasingly complex roles in automation and strategic decision-making. Central to these applications is the ability to perform robust long-term planning and reasoning, especially in scenarios that require the execution of action sequences

Recent reinforcement learning (RL)-based training approaches have enhanced the reasoning capabilities of LLMs, giving rise to large reasoning models (LRM) such as OpenAI o1 and Deepseek R1 [DeepSeek-AI et al., 2025]. These models are capable of producing extended chains of thought and can engage in partial self-verification of their outputs. This has led to notable performance gains, particularly in domains such as mathematics and programming - areas that are

[Figure 2]

- Figure 2: Example of a task in the environment: the agent’s ultimate goal is to defeat the target monster. To achieve this, agent must calculate the optimal gear by considering both its own and the monster’s stats, and acquire all the necessary ingredients.

well-suited to formal verification and commonly included in model training. However, their performance remains suboptimal in tasks that require long-term planning [Valmeekam et al., 2025; Shojaee et al., 2025].

Current claims regarding LLMs’ planning abilities are often based on evaluations using standard algorithmic benchmarks, with tasks like Blocksworld being the primal example [Valmeekam et al., 2023a]. Although such environments are easily scalable, they lack the complexity and variability characteristic of real-world scenarios. Other benchmarks that adopt game-based or real-world settings either fail to provide tasks with sufficient depth to challenge current models, or rely heavily on continuous environmental feedback. This reliance makes it difficult to isolate a model’s planning capabilities and to evaluate performance in domains where robust end-toend planning is essential and failures are unacceptable.

To address these limitations, we introduce HeroBench, a benchmark specifically designed to evaluate long-horizon, hierarchical planning under constraints. Built upon the ArtifactsMMO environment [V. Bissonnette, 2024], originally developed to assess programming skills via script-based gameplay, HeroBench features a classic RPG setting in which agents gather resources, craft equipment, and confront enemies. HeroBench simultaneously satisfies a set of properties that no existing planning benchmark covers in combination making it a strong test setup for LRMs (Table 1). Although HeroBench is framed as a game-based environment, it is designed to evaluate one-shot, end-to-end planning under strict feasibility constraints. Agents must commit to a complete plan before execution, with no opportunity for midcourse correction, closely mirroring domains in which errors

are costly or irreversible. Tasks require hierarchical task decomposition, exact resource and dependency accounting, numeric feasibility reasoning, resistance to distractors, and sustained reasoning over long contexts. Crucially, HeroBench is designed not only to measure planning capability, but to assess the reliability of model behavior across long reasoning horizons.

We evaluated 25 state-of-the-art LLMs and LRMs, including open-source and proprietary models across different families and model sizes, revealing substantial performance differences.

### 2 HeroBench

##### 2.1 Dataset description

The environment displayed in Fig. 1 is a structured RPG-style game with a discrete action space. The world is organized as a grid of 70 locations, each containing specific elements such as resource nodes, workshops, or monster spawns. The environment includes 25 distinct monsters, 17 resource types for crafting, and 208 unique items, including gear and crafting components. All environment data is defined through JSON files.

The dataset consists of tasks of varying difficulty levels, each requiring the player to defeat a specific monster in the game or craft an item. An example prompt is provided in the Appendix C. The player starts with a character of appropriate level and a specific set of equipment. The tasks are divided into two categories: purely crafting tasks, which do not require any combat, and tasks that involve defeating enemies. In the latter case, defeating a monster typically requires the character to craft one or more additional items beforehand. The difficulty of a task is determined by the number of required items and the number of steps involved in crafting them. For example, crafting a simple bronze sword may require only mining and smelting ore, whereas crafting a highlevel item can involve many steps, including obtaining drops from defeated monsters and gathering and refining multiple types of resources (Fig. 2).

##### 2.2 Task Formalization

A task instance induces a deterministic planning problem with state s = ⟨ℓ,I,E,L,H⟩, where ℓ ∈ Z2 is the agent location, I is an inventory multiset, E is an equipment assignment (one item per slot), L are profession skill levels (e.g., mining, woodcutting, smithing), and H denotes derived combat stats (HP, attacks, resistances, damage amplifiers) computed from the base character plus effects of E. We write G(E) for the (slot-wise) set of equipped items implied by E.

Actions are deterministic and have location- and inventorydependent preconditions: move(x,y) updates ℓ; gather yields one unit of the resource available at ℓ; fight(m) is applicable only if monster m is co-located and yields guaranteed drops upon victory; equip/unequip updates E subject to slot constraints. Crafting is captured by craft(i,q), applicable only if (i) ℓ contains the required station σ(i), (ii) L satisfies a skill threshold τ(i), and (iii) I contains the required inputs.

Table 1: Comparison of Benchmarks Across Reasoning and Robustness Criteria Benchmark One-shot Max Act. Feasibility Analysis Hierarch. Numeric Distract Long-cont. LRM

PlanBench ✓ 47 ✗ ✓ ✗ ✗ ✗ ✗ ✓ PlanCraft ✗ 30 ✓ ✓ ✓ ✓ ✓ ✓ ✓ Travel Planner ✗ 30 ✓ ✓ ✓ ✓ ✗ ✓ ✓ Natural Plan ✓ 19 ✓ ✓ ✓ ✓ ✗ ✗ ✗ ScienceWorld ✗ 100 ✓ ✓ ✓ ✓ ✓ ✓ ✓ Robotouille ✓ 82 ✓ ✓✓ ✓ ✓ ✗ ✓ ✓ ALFWorld ✗ 50 ✗ ✓ ✓ ✗ ✓ ✗ ✗ LogiPlan ✓ 300 ✓ ✓ ✗ ✗ ✗ ✓ ✓ HeroBench ✓ 1000+ ✓✓ ✓✓ ✓✓ ✓✓ ✓✓ ✓✓ ✓✓

- Table 2: One-shot: end-to-end planning performed in a single generation; Max Actions: maximum number of environment actions required in the hardest tasks; Feasibility: requirement to satisfy multiple and diverse constraints; Analysis: availability of error analysis tools; Hierarchical: presence of deep hierarchical task structures; Numeric: requirement of numeric reasoning in addition to symbolic planning; Distractors: inclusion of distractor conditions; Long-context: requirement to process long, detailed environment descriptions; LRM: difficulty level appropriate for evaluating state-of-the-art large reasoning models. ✓✓ strong presence, ✓ partial presence, ✗ absence.

Item–ingredient–resource hierarchy. Each craftable item i defines a recipe hyperedge Ri = {(j,nij)}, consuming nij units of ingredient j. Recipe expansion induces a dependency hypergraph/DAG

target item/gear → ingredients

→ crafted sub-ingredients

→ (gather nodes ∪ monster drops).

where leaves are primitive acquisition operators (gather/fight) and internal nodes are craft operations constrained by station locations and skill thresholds. Exact planning requires propagating multiplicities through the hierarchy to compute total required counts.

Embedded numeric feasibility and gear selection. Tasks with a combat goal “defeat monster m” introduce a numeric feasibility constraint under a turn-based, type-specific combat simulator. For any chosen equipment assignment E, the agent computes derived stats H(E) and verifies WIN(m;H(E)). Damage per turn is computed for each non-zero damage type t:

rtm 100

bt 100 − at

dchart = at 1 +

,

rtchar 100

dmt = amt − amt

.

where at is base attack of the attacker in type t, bt is matching damage amplification, and rt is the defender’s resistance in type t (allowing negative values). Total per-turn damage is

t dt, and the simulated fight is feasible iff the monster’s HP reaches 0 before the character’s HP.

Sources of complexity. HeroBench combines: (i) hierarchical resource planning via multi-level recipe expansion with exact counting, (ii) spatial planning to route between resource nodes, monsters, and multiple crafting stations, and (iii) discrete optimization over equipment assignments E (slot-wise) coupled to numeric combat feasibility. Difficulty arises from the tight interaction between combinatorial choices (gear and ordering) and numeric constraints (amplifiers/resistances) that determine whether a plan can satisfy the combat goal.

### 3 Benchmark Task Generation Pipeline

We present a systematic pipeline for constructing benchmark tasks for HeroBench. Each generated instance is a planning problem of the form in Sec. 2.2, with (optional) combat feasibility WIN(m;H(E)) and a crafting/resource dependency DAG induced by recipes.

Monster initialization. Let M = {m1,m2,...,mN} denote the set of all monsters in the game. To generate a new

combat task, we first sample a target monster m ∼ PM(· | dtarget) conditioned on the desired difficulty. Each monster m is associated with a statistics vector m (health, attacks, resistances, etc.) and a combat difficulty level L(m), defined as the minimal character combat level under which m can be defeated under standard conditions.

Combat simulation / feasibility check. For a fixed monster m and an equipment assignment E, combat is simulated in a turn-based manner from the derived stats H(E) and monster stats m. We write the simulator outcome as the feasibility predicate

WIN(m;H(E)) ∈ {0,1}, which returns 1 iff the character defeats m.

Minimal winning equipment search. Let I≤L(m) be the set of equipment items whose level requirement is at most L(m). We search over feasible equipment assignments E using items from I≤L(m) and define a minimal winning equipment assignment E∗ as one that satisfies:

- • WIN(m;H(E∗)) = 1;
- • for any strict ablation of equipped items (i.e., any E′ obtained by unequipping at least one item from E∗ while respecting slot constraints), WIN(m;H(E′)) = 0.

Equivalently, E∗ is minimal in the sense that removing any single equipped item causes failure.

From E∗ we induce the equipped item set G(E∗) and partition it into:

- • Initially equipped items (Ieq): items present in E at episode start,
- • Missing items (Imiss): items that must be acquired/crafted during the episode.

[Figure 3]

- Figure 3: Success rate of LLMs across nine base-task difficulty levels. Solid lines correspond to reasoning (thinking) models, while dashed lines represent standard (non-thinking) variants.

The base task difficulty is d = |Imiss|, and is refined using acquisition and crafting costs.

Crafting and environment analysis. For each item i ∈ Imiss, we traverse the recipe dependency DAG induced by {Ri} (Sec. 2.2) to extract all required materials, intermediate crafted components, relevant monsters (for drops), and required locations (resource nodes and stations). The total task difficulty is:

###### Dtotal = |Imiss| +

###### cost(i),

i∈Imiss

where cost(i) is a crafting/acquisition cost function (App. A). Auxiliary item validation. To enforce robust solution paths, we compute an auxiliary set of valid items Iaux such that:

- • the character equipped with Ieq ∪ Iaux can defeat all non-target monsters present in the scenario;
- • but cannot defeat the target monster m without acquiring all items in Imiss.

Formally, for any monster mj with stats mj present in the scenario,

###### WIN(mj;H(Eeq∪aux)) = I[mj ̸= m],

where Eeq∪aux denotes any equipment assignment consistent with equipping Ieq ∪ Iaux (subject to slot constraints).

Crafting-only tasks follow the same pipeline but omit the combat feasibility steps and directly specify goal items, with difficulty determined solely by traversal of the crafting dependency DAG.

Leveling mechanics. In the base task set, the profession skill-level vector L is initialized to match the highest requirement among items in Imiss. In the extended version, we introduce skill progression: the agent starts at level 1 in all relevant professions. To support skill growth, the environment description includes accessible resource nodes and associated experience rewards, so that crafting high-level items requires planning a sequence of actions that incrementally increases L.

Noise item injection. To increase task complexity and test robustness, we optionally add noise items Inoise: plausible, high-level gear items that appear valid based on stats and level filters, but are impossible to craft because at least one required prerequisite (ingredient/resource/station) is omitted from the environment.

Task Representation Each task is serialized as a structured JSON object, specifying: target monster or craft item name; equipped and missing items; full character state; environment information (dependencies, required monsters, locations, etc.). Prompts for language models are generated from these objects to ensure reproducibility. The final dataset contains 844 tasks with difficulty levels ranging from 2 to 97. Input prompt lengths vary from 1k to 11k tokens. For our experiments, we selected a subset of 180 tasks, divided into 9 difficulty brackets. Leveling and noise mechanics can be incorporated on top of the base tasks. The benchmark is highly accessible: the environment is fully defined in a single prompt, and all scripts needed to score the agent’s plan are provided.

Evaluation The LLM or an agentic system is prompted to generate a sequence of actions in Python that solves the given

Environmental actions

Crafting complexity

Unique milestones

Difficulty

- 1 40 ± 31 8 ± 3 6 ± 3
- 2 122 ± 52 18 ± 3 12 ± 3
- 3 178 ± 53 28 ± 3 17 ± 3
- 4 261 ± 51 38 ± 3 21 ± 3
- 5 304 ± 80 48 ± 3 26 ± 3
- 6 396 ± 70 58 ± 3 29 ± 4
- 7 461 ± 121 68 ± 3 32 ± 4
- 8 552 ± 161 78 ± 3 37 ± 3
- 9 663 ± 143 88 ± 3 39 ± 3
- 10 937 ± 242 88 ± 3 41 ± 3

- Table 3: Difficulty levels in HeroBench. All values are reported as (mean ± SD). Environmental actions denote the number of actions in the environment required to complete the task. Crafting complexity represents the number and depth of crafting steps involved, computed as 2 points for each craftable item and 1 point for each non-craftable ingredient. Unique milestones indicate the number of distinct subtasks that must be completed, such as defeating a monster or obtaining a specific item.

task. The generated sequence is then parsed and executed in the environment, with the resulting simulation logs recorded for analysis.

Two evaluation metrics are used: Success, indicating whether the final goal (crafting the target item or defeating the target monster) is achieved; and Progress score, which reflects partial completion based on valid intermediate actions such as gathering, recycling, defeating required monsters, crafting, and equipping gear.

This dual-metric evaluation enables both binary assessment of task completion and fine-grained measurement of the agent’s progress and problem-solving efficiency. We also provide an evaluation pipeline that offers comprehensive statistics on the types of errors made by the agents. These include mistakes in high level plan decomposition and optimal gear calculation, failures in determining the required amount of resources or appropriate level for item crafting, incorrect usage of provided information such as location coordinates, and improper code formatting in the response. This allows for a more precise assessment of the models’ weaknesses.

Agentic systems We also developed and evaluated handcrafted agentic pipelines on HeroBench. Implementation details are provided in Appendix B.

- 4 Results

We evaluated a wide range of state-of-the-art LLMs on our benchmark, encompassing both standard and reasoning models. Experiments were conducted using local FP16 Qwen-8B and Qwen-32B models, the OpenAI API for the o3 and o4mini, and the OpenRouter API for the remaining models. All hyperparameters were set to default values. The reasoning budget was set to ’high’ for OpenAI models and capped at 40,000 reasoning tokens for the other models, which was not exceeded in our experiments. All models were tested on the base set of tasks, while the top-performing reasoning models were further evaluated on harder difficulty levels incorporating additional mechanics.

Model Success % Score Tokens

Qwen3 8b 0.0 11.5 ± 6.8 2883 ± 1965 Qwen3 32b 1.7 21.9 ± 12.8 2074 ± 1222 GigaChat 2 Max 2.8 21.3 ± 15.4 1190 ± 228 Qwen3 8b (t) 3.9 28.8 ± 15.5 9680 ± 1224 Deepseek-v3 7.2 32.7 ± 17.9 1586 ± 430 Kimi-K2 8.3 29.6 ± 16.4 1309 ± 237 GPT-oss-120b 8.9 27.0 ± 8.7 9372 ± 2959 Magistral-medium 9.4 25.0 ± 18.8 10885 ± 1667 Qwen3 32b (t) 10.0 44.8 ± 17.1 9107 ± 1458 DeepSeek-R1-70B 11.2 27.5 ± 21.2 7448 ± 1029 Qwen3-235b 13.3 34.9 ± 20.5 12006 ± 1746 GPT-4.1-mini 16.1 53.9 ± 17.6 4555 ± 1398 Claude-Sonnet-4 17.2 50.6 ± 21.0 1578 ± 306 Qwen3-235b-2507 24.4 49.4 ± 18.6 11387 ± 2702 Deepseek-R1-0528 21.7 48.7 ± 22.5 10711 ± 2088 Gemini-2.5-flash 26.1 64.8 ± 13.7 11028 ± 4010 GPT-4.1 31.7 73.7 ± 10.3 3518 ± 1202

- o4-mini 35.0 56.1 ± 23.5 21993 ± 8181 GPT-5-mini 35.0 59.8 ± 22.5 14126 ± 4169 Claude-Sonnet-4 (t) 44.4 73.8 ± 16.9 16397 ± 4313
- o3 60.6 84.6 ± 8.5 13897 ± 5250 Gemini-2.5-pro 62.9 86.6 ± 10.4 12935 ± 4295 GPT-5 83.9 95.0 ± 3.3 17851 ± 7149 Grok-4 91.7 95.3 ± 3.3 15470 ± 5838

Table 4: Mean performance of all evaluated models across nine base task difficulty levels in HeroBench. Columns show success rate (%), score (mean ± SD), and tokens (mean ± SD). SD is computed across the nine difficulty-level averages for each model. Thinkingenabled variants are denoted by (t).

##### 4.1 Base tasks

The success rate over different difficulties is shown in Fig. 3 and mean metrics over all difficulties are presented in table 4. Reasoning models consistently outperform standard models across all levels of task difficulty. However, the accuracy of most LRMs declines as complexity increases. In contrast to conventional mathematical and coding benchmarks, where open-source models approach the performance of leading proprietary ones, our evaluation reveals substantial variability in model performance. Notably, Grok 4 achieved the highest scores and exhibited the least performance degradation as task difficulty increased, clearly outperforming other models at higher difficulty levels.

Among non-reasoning models, GPT-4.1 demonstrated the best performance, outperforming several open-source reasoning models and achieving a success rate close to that of o4mini, while even surpassing it in score. Its success rate was nearly double that of Claude Sonnet-4 (non-thinking), although it used more than twice the number of tokens to solve the tasks. Overall, GPT-4.1 and Claude Sonnet-4 (nonthinking) exhibited the best performance in terms of success per tokens spent.

We tested the Qwen3-235b-a22 and Qwen3-235b-a222507 models following a switch from the widely used GRPO [Shao et al., 2024] algorithm to GSPO [Zheng et al., 2025], which is better suited for RL training of MoE architectures. This change resulted in a significant improvement in the models’ planning capabilities, though still not sufficient to match

Model Errors (mean ± SD) Failure Types (% of all tasks) High-level Execution Failed Only Gear Gear+Exec Only Exec Invalid Output

Qwen3 8b 3.32 ± 2.23 2.59 ± 0.49 100.0 8.9 67.8 10.6 12.8 Qwen3 32b 3.62 ± 2.33 6.44 ± 5.86 98.3 5.0 77.2 7.2 7.2 GigaChat 2 Max 3.89 ± 2.49 1.93 ± 1.11 97.2 17.2 61.1 5.6 11.7 Qwen3 8b (think) 3.52 ± 2.31 2.69 ± 0.88 96.1 5.6 78.9 7.8 2.8 Deepseek-v3 3.69 ± 2.38 1.39 ± 0.63 92.8 16.7 68.3 5.0 1.7 Kimi-K2 3.78 ± 2.52 1.34 ± 0.71 91.7 17.8 53.9 1.1 18.3 GPT-oss-120b (think) 3.30 ± 2.22 4.90 ± 5.95 91.9 12.2 34.4 3.9 40.6 Magistral-medium-2506 (think) 3.10 ± 2.48 2.14 ± 2.10 90.6 16.1 32.8 2.8 36.7 Qwen3 32b (think) 3.21 ± 2.18 4.89 ± 4.09 90.0 9.4 70.6 5.6 3.3 DeepSeek-R1-70B (think) 3.25 ± 2.64 1.86 ± 0.88 86.7 11.1 41.7 6.7 26.7 Qwen3-235b-a22 (think) 2.79 ± 2.40 2.79 ± 1.91 87.8 9.4 45.0 2.8 27.2 GPT-4.1-mini 2.77 ± 2.42 1.91 ± 1.79 83.9 7.2 55.0 6.7 11.1 Claude-Sonnet-4 3.41 ± 2.59 4.93 ± 5.01 82.8 0.6 60.6 3.3 16.1 Qwen3-235b-a22-2507 (think) 2.76 ± 2.41 1.89 ± 1.15 77.8 13.9 36.7 2.2 22.2 Deepseek-R1-0528 (think) 2.73 ± 2.46 1.38 ± 1.39 78.3 19.4 39.4 4.4 12.2 Gemini-2.5-flash (think) 1.29 ± 1.27 1.56 ± 1.66 73.9 17.8 30.6 7.8 10.6 GPT-4.1 1.50 ± 1.39 0.78 ± 0.89 68.3 26.1 32.2 3.3 2.2

- o4-mini (think) 2.12 ± 2.11 0.44 ± 0.30 65.0 22.8 22.2 3.9 14.4 GPT-5-mini (think) 1.86 ± 2.07 0.37 ± 0.32 65.0 29.4 15.6 2.8 15.0 Claude-sonnet-4 (think) 1.58 ± 1.58 0.34 ± 0.21 55.6 24.4 21.7 3.9 3.3

- o3 (think) 0.82 ± 0.98 0.18 ± 0.17 39.4 21.7 7.8 7.8 1.1 Gemini-2.5-pro (think) 0.83 ± 1.09 0.10 ± 0.12 37.8 22.2 7.8 1.1 1.7 GPT-5 (think) 0.19 ± 0.28 0.08 ± 0.07 16.1 8.3 1.1 5.6 0.6 Grok-4 (think) 0.11 ± 0.16 0.02 ± 0.04 8.3 2.8 1.7 0.6 2.8

- Table 5: Breakdown of failure types across models. First two columns show the mean ± SD number of errors per task for tasks with valid generated code. The first column reports the average number of items missing in the high-level plan, while the second reports the average number of low-level execution mistakes. The remaining columns show the percentage of tasks that failed due to: gear selection only, gear selection plus execution errors, execution-only errors, or invalid code.

the performance of proprietary models.

Additional charts, including token usage efficiency, are presented in Appendix D.

Results analysis To understand the weaknesses and failure modes of various models, we provide a script for comprehensive analytics of model’s performance on the tasks. It scores how many tasks failed due to errors in the high-level plan for selecting optimal gear, how many items were incorrectly chosen for the optimal outcome, how many mistakes the model made in executing the plan (e.g., incorrect amounts of resources, misusing environmental information, redundant steps, etc.), and how many plans failed due to incorrectly formatted output.

An analysis of the failure modes in Table 5 reveals a clear bottleneck in high-level planning relative to low-level execution across model architectures. Although top-tier reasoning models such as Gemini 2.5 Pro, Grok-4, and GPT-

- 5 exhibit substantially fewer execution errors, demonstrating their ability to reliably interpret and use contextual information, they continue to struggle with high-level search for optimal gear configurations. This challenge, which involves complex numeric reasoning and constraint satisfaction, indicates that determining optimal strategies remains more difficult than correctly implementing them. In contrast, smaller and open-source models fail more pervasively, often making simultaneous errors in both gear selection and execution. Other models, such as GPT-oss-120b, are further hindered by severe instruction-following instability, with 40.6% of tasks

failing due to invalid output generation. Overall, the results show that while enhanced reasoning capabilities significantly reduce syntax and execution errors, the demands of longhorizon, constraint-heavy planning remain a persistent challenge even for state-of-the-art systems.

##### 4.2 Leveling + distractor noise mechanics

Table 6 presents the results for level 10 difficulty tasks (level 9, plus leveling mechanics and distractor noise items), evaluated on top-performing models. Grok-4 demonstrates a significant lead over all other models, with only Gemini 2.5 Pro and GPT-5 managing to solve a subset of the hardest tasks. Notably, Grok-4’s performance remains consistently high across all difficulty levels, showing drop in success rate with addition of leveling and score with addition of noise items. The performance of GPT-5 decreases with the addition of leveling mechanics but remains unaffected by the inclusion of additional noise items. Notably, GPT-5 and Grok-4 are the only two models that substantially increase their reasoning length as task difficulty rises. The results of Grok-4 also show that the tasks are solvable within the 20-35k output tokens. The details of Grok-4’s architecture and training remain undisclosed, though its impressive performance may be attributed to the reported large-scale reinforcement learning applied during post-training.

###### Model

|Base<br><br>|Leveling|Leveling+Noise|
|---|---|---|
|Succ (%) Score Tokens|Succ (%) Score Tokens<br><br>|Succ (%) Score Tokens|

o3 5 66.2 ± 32.1 20688 ± 2791 0 26.6 ± 28.4 22606 ± 2788 0 15.9 ± 12.0 23562 ± 3996 Claude-Sonnet-4 10 42.6 ± 36.3 21366 ± 6036 0 25.6 ± 19.0 24588 ± 6651 0 21.9 ± 14.2 25404 ± 5697 Gemini-2.5-pro 25 66.1 ± 26.6 18636 ± 3835 10 32.7 ± 26.4 20047 ± 3141 5 36.0 ± 28.5 21741 ± 3127 GPT-5 55 90.6 ± 16.5 28052 ± 3776 15 62.3 ± 32.6 31704 ± 3656 20 59.9 ± 34.2 36052 ± 4196 Grok-4 80 95.5 ± 14.2 22850 ± 4587 65 92.9 ± 16.5 28361 ± 5953 65 78.8 ± 31.8 33305 ± 6672

- Table 6: Evaluation of five leading reasoning models under increased task complexity (difficulty 10). Results are shown for three conditions: Base (standard level 9 tasks), Leveling (requires skill progression before crafting), and Leveling+Noise (adds adversarial distractor items). Metrics include success rate, progress score (mean ± SD), and token usage (mean ± SD).

|Model<br><br>|Diff|k<br><br>|pass@k<br><br>|Mean Win|
|---|---|---|---|---|
|Qwen3-8B Qwen3-8B (t) Qwen3-32B Qwen3-32B (t)|1<br><br>1<br><br>2<br><br><br>2<br><br><br>|200 10 200 10|45.0% 65.0% 30.0% 75.0%<br><br>|11.8% 30.5% 0.6% 20.0%|

- Table 7: Performance of thinking (t) and non-thinking Qwen3 Models using pass@k metric. Results suggest that the RLVR approach noticeably improves the results and may be task-dependent.

in natural language across trip planning, meeting scheduling, and calendar coordination tasks, and TravelPlanner [Xie

- et al., 2024] extends this setting with large-scale data, tool use, and complex real-world constraints. LogiPlan [Cai et al., 2025] targets logical and relational planning by requiring models to generate and verify structured dependency graphs. ROBOTOUILLE [Gonzalez-Pumariega et al., 2025] introduces asynchronous planning with time delays and longhorizon dependencies in a cooking domain. Plancraft [Dagan
- et al., 2025] presents a multimodal Minecraft-based benchmark for evaluating hierarchical planning, resource management, and feasibility recognition, including unsolvable tasks.

##### 4.3 Pass@k metric

While recent findings suggest that reinforcement learning with verifiable rewards (RLVR) may not consistently improve over the base model’s pass@k performance when k is sufficiently large [Yue et al., 2025], our results indicate that this conclusion may be task-dependent. In the context of our planning benchmark, particularly at difficulty levels 1 and 2, we observed that, even after 200 attempts, the base models Qwen3-8B and Qwen3-32B were unable to match the performance of their reasoning-enabled counterparts, which achieved higher pass rates with just 10 attempts Table 7. Since these lower-difficulty tasks do not demand long reasoning chains, as evidenced by the success of other nonreasoning models on them,this gap suggests that RLVR can provide tangible benefits in planning scenarios where structured reasoning is essential.

### 6 Conclusions

This work introduced HeroBench, a benchmark designed to stress-test long-horizon planning and structured reasoning in LLMs and LRMs under tightly constrained conditions. Unlike prior planning benchmarks that emphasize abstract symbolic domains, lack difficulty and depth or rely on continuous interaction and feedback, HeroBench evaluates one-shot, end-to-end planning in a rich and complex environment with many interdependent rules, constraints, and dependencies, making it well suited to challenging the latest generation of models.

Our extensive evaluation of LLMs across a range of sizes and model families shows that RL-based post-training enables substantial gains over base models, yet performance still degrades as task horizon, dependency depth, and numeric feasibility constraints increase. Even the strongest reasoning models fail to reliably solve the hardest tasks, despite producing long and detailed reasoning traces, while smaller and open-source models exhibit very low reliability even on comparatively simple tasks.

### 5 Related work

Planning is a fundamental capability for LLMs and LLMbased agents, and its reliable evaluation requires benchmarks that capture multi-step reasoning, state transitions, constraints, and long-horizon decision making [Yehudai et al., 2025; Wei et al., 2025; Li et al., 2025] A variety of benchmarks have been proposed to evaluate the planning capabilities of LLM-based agents across symbolic, embodied, and real-world domains. PlanBench [Valmeekam et al., 2023a] evaluates classical planning by testing reasoning about actions and state transitions beyond commonsense recall. ALFWorld [Shridhar et al., 2021] studies multistep household planning by aligning abstract language plans with embodied execution, while ScienceWorld [Wang et al., 2022] focuses on procedural scientific planning through interactive experiments rather than question answering. Natural Plan [Zheng et al., 2024] evaluates realistic planning

Overall, HeroBench provides a challenging and transparent testbed that exposes limitations of current LLMs in longhorizon, constraint-heavy planning, offering a clear target for future advances in autonomous reasoning systems.

### 7 Future work

Although the current set of tasks is designed to evaluate existing systems without overwhelming them with excessive complexity, the environment natively supports multi-agent dynamics, seamless transition to open-ended play, and integration of visual modalities. Future extensions may include

multi-agent manipulation, collaboration and competition dynamics, additional in-game mechanics, stochasticity in the tasks and a natural extension of HeroBench into a RL environment.

### References

[Cai et al., 2025] Yanan Cai, Ahmed Salem, Besmira Nushi, and Mark Russinovich. Logiplan: A structured benchmark for logical planning and relational reasoning in llms, 2025.

[Dagan et al., 2025] Gautier Dagan, Frank Keller, and Alex Lascarides. Plancraft: an evaluation dataset for planning with LLM agents, 2025.

[DeepSeek-AI et al., 2025] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang,

Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseekr1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.

[Gonzalez-Pumariega et al., 2025] Gonzalo GonzalezPumariega, Leong Su Yean, Neha Sunkara, and Sanjiban Choudhury. Robotouille: An asynchronous planning benchmark for llm agents, 2025.

[Kambhampati et al., 2024] Subbarao Kambhampati, Karthik Valmeekam, Lin Guan, Mudit Verma, Kaya Stechly, Siddhant Bhambri, Lucas Paul Saldyt, and Anil B Murthy. Position: LLMs can’t plan, but can help planning in LLM-modulo frameworks. In Forty-first International Conference on Machine Learning, 2024.

[Li et al., 2025] Haoming Li, Zhaoliang Chen, Jonathan Zhang, and Fei Liu. Planet: A collection of benchmarks for evaluating llms’ planning capabilities, 2025.

[Shao et al., 2024] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.

[Shojaee et al., 2025] Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity, 2025.

[Shridhar et al., 2021] Mohit Shridhar, Xingdi Yuan, MarcAlexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning, 2021.

[Tan et al., 2024] John Chong Min Tan, Prince Saroj, Bharat Runwal, Hardik Maheshwari, Brian Lim Yi Sheng, Richard Cottrill, Alankrit Chona, Ambuj Kumar, and Mehul Motani. TaskGen: A task-based, memory-infused agentic framework using StrictJSON, 2024.

[V. Bissonnette, 2024] V. Bissonnette. Artifactsmmo. https: //www.artifactsmmo.com/, 2024.

- [Valmeekam et al., 2023a] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 38975–38987. Curran Associates, Inc., 2023.
- [Valmeekam et al., 2023b] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models - a critical investigation. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 75993–76005. Curran Associates, Inc., 2023.

[Valmeekam et al., 2025] Karthik Valmeekam, Kaya Stechly, Atharva Gundawar, and Subbarao Kambhampati. A systematic evaluation of the planning and scheduling

abilities of the reasoning model o1. Transactions on Machine Learning Research, 2025.

[Wang et al., 2022] Ruoyao Wang, Peter Jansen, MarcAlexandre Côté, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader?, 2022.

[Wei et al., 2025] Hui Wei, Zihao Zhang, Shenghua He, Tian Xia, Shijia Pan, and Fei Liu. Plangenllms: A modern survey of llm planning capabilities, 2025.

[Xie et al., 2024] Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. TravelPlanner: A benchmark for real-world planning with language agents, 2024.

[Yehudai et al., 2025] Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, and Michal Shmueli-Scheuer. Survey on evaluation of llmbased agents, 2025.

[Yue et al., 2025] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025.

- [Zheng et al., 2024] Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, and Denny Zhou. NATURAL PLAN: Benchmarking LLMs on natural language planning, 2024.
- [Zheng et al., 2025] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025.

### A Model Inference Configuration

All experiments using Openrouter API were conducted using default hyperparameters: temperature = 1.0, top_p = 1.0, top_k = 0, frequency_penalty = 0.0, presence_penalty = 0.0, repetition_penalty = 1.0, min_p = 0.0, top_a = 0.0.

The Qwen 3 models were run with the same hyperparameters on a system equipped with 2× A100 GPUs (80GB VRAM each).

### B Multi-Agent systems architecture

A-1 represents the first version of the multi-agent system developed for HeroBench. The A-1 architecture employs a pair of agents: a decomposer/action agent and a critic agent. This agent pair is used to perform two-level task decomposition. The architecture of A-1 is shown in Fig. 4A.

At the first level, the task is divided into high-level subtasks, which are used to form a high-level plan. Once the plan is verified by the critic agent, it is passed to the second stage of decomposition. In this stage, each item in the high-level plan is further broken down into basic subtasks, resulting in low-level steps (actions). The collection of all actions constitutes the final executable plan, which is carried out in the environment through calls to the corresponding environment API functions. To prevent infinite loops, the system imposes limits on the number of allowed decomposition attempts. For decomposition, the agent relies on the current environment state and automatically generated descriptions of the tasks or subtasks target objects.

A-2 is a more complex multi-agent system designed to evaluate a deeper, linear hierarchy of cooperating agents. The main idea is to allow agents to solve simple, one-bite tasks individually while assisting one another. This system builds upon the TaskGen agentic framework [Tan et al., 2024], extending the initial A-1 architecture (Fig. 4B).

Several new agents were introduced in A-2. First, a curriculum agent formulates high-level plans based on the global task and the current state of the characters. An optional fight analytic agent estimates the outcomes of combat encounters, taking into account character statuses, equipment, and potential crafted items. Additional subagents were also integrated into the decomposer agent: a map expert and a craft expert, both capable of acquiring world knowledge and answering the decomposer’s queries. Finally, an action agent was added to generate the executable Python code to complete the task.

Both A-1 and A-2 are also capable of solving open-ended tasks by incorporating feedback from the environment.

B.1 Multi-Agent systems performance

To evaluate the multi-agent systems, tasks with difficulty levels 2 and 3 were selected. GPT-4.1-mini was chosen as the baseline model due to its accessibility, high speed, and reasonably strong performance. The experimental results (Table 8) show that the A-1 multi-agent system achieved a higher success rate than the baseline model. However, the performance of A-2 was lower than that of the baseline, presumably due to its more complex architecture and prompt overengineering. Smaller models were unable to effectively process the context provided by the subagents and exhibited hallucinations during plan and subtask generation. These findings indirectly suggest that while multi-agent systems can better maintain problem-solving capabilities at higher task complexities, they require very careful design - particularly with regard to task complexity and prompt size.

|Agent/Model|Difficulty 2<br><br>|Difficulty 3|
|---|---|---|
|A-1<br><br>|65%<br><br>|60%|
|A-2|35%|10%|
|GPT-4.1-mini|45%|15%|

Table 8: A-1, A-2 and GPT-4.1-mini success rate comparison. The results indicate that the simple decomposer-critic loop for the small models outperformed a modified decomposition architecture with one-bite tasks.

#### C Prompt Example This section contains example prompt from HeroBench dataset.

Prompt Example

I will provide you with the information about game environment and a task you need to accomplish.

Rules for items: Weapons have attack stats that will provide the damage type and damage value, for

example {'name': 'attack_air', 'value': 8} means air damage type and 8 damage per turn. Armor and jewelry can have hp

stat, that will add hp number to the players hp {'name': 'hp', 'value': 20}, damage amplifications stat in % {'name': 'dmg_fire', 'value': 3} and resistance stat in % {'name': 'res_air', 'value': 3}. The damage amplification will increase the attack of your current weapon by the stated

value if you have matching attack type weapon. Rules for Fighting Monsters: To fight a monster you need to be on the same location as monster and perform fight

action. Combat follows a turn-based system where you attack first, followed by the monster. There are different damage types: fire, earth, water, and air. If a character or monster has resistance matching the attacker's damage type, the damage is either reduced or amplified based on

the resistance percentage. The calculations of damage follow these formulas: character_damage = base_attack + dmg_boost - res_reduction monster_damage = base_attack - res_reduction res_reduction = base_attack * (resist_percentage / 100) dmg_boost = base_attack * (dmg_percentage / 100)

The damage is calculated from every non 0 attack type the monster or character has. After each turn, the health points of the character or monster decrease by the calculated damage. The

fight is automatic and continues until the health points of one participant drop to 0. The items acquired from monsters drop after you defeat them in a fight. Assume that drop

rates from monsters are 100%. Your health and monsters health is restored to full after each fight.

Rules for crafting: Each gathering action provides you one resource. To craft an item you should be on the appropriate location like crafting station for the

item you want to craft. Your crafting level should not be less than required. You can perform the following actions: move(character_name: str, x: int, y: int) fight(character_name: str) equip(character_name: str, slot: str, item_name: str, quantity: int = 1) unequip(character_name: str, slot: str, quantity: int = 1) Item slot. Allowed values: 'weapon', 'shield', 'helmet', 'body_armor', 'leg_armor', '

boots',

'ring1', 'ring2', 'amulet', 'artifact1', 'artifact2', 'artifact3', 'consumable1 ' or

'consumable2'.

gather(character_name: str) craft(character_name: str, item_name: str, quantity: int) Use codes of items in functions like 'copper_dagger'. You can only use these functions

and for loops in your answer, nothing else. Do not include parameter names when calling functions. If there is an item in the slot, you should unequip it before equipping a different item

. {'Monsters': [{'name': 'Lich', 'code': 'lich', 'level': 30, 'hp': 1500, 'attack_fire': 60, 'attack_earth': 60, 'attack_water': 0, 'attack_air': 0, 'res_fire': 24, 'res_earth': 24,

'res_water': 18, 'res_air': 18, 'min_gold': 0, 'max_gold': 15, 'drops': [{'code': ' life_crystal',

'rate': 3000, 'min_quantity': 1, 'max_quantity': 1}, {'code': 'lich_crown', 'rate': 800, 'min_quantity': 1, 'max_quantity': 1}]}, {'name': 'Ogre', 'code': 'ogre', 'level': 20, '

hp': 680, 'attack_fire': 0, 'attack_earth': 80, 'attack_water': 0, 'attack_air': 0, 'res_fire':

-20, 'res_earth': 30, 'res_water': 0, 'res_air': 0, 'min_gold': 0, 'max_gold': 3, 'drops': [{'code': 'ogre_eye', 'rate': 20, 'min_quantity': 1, 'max_quantity': 1}, {'code': 'ogre_skin', ' rate': 20, 'min_quantity': 1, 'max_quantity': 1}]}], 'Craftable items': [{'name': 'Dreadful Ring', 'code': 'dreadful_ring', 'level': 20, 'type': 'ring', 'subtype': '', 'description': '', 'effects ': [{'name': 'hp', 'value': 20}, {'name': 'dmg_earth', 'value': 11}, {'name': 'dmg_water', 'value': 11}], 'craft': {'skill': 'jewelrycrafting', 'level': 20, 'items': [{'code': 'steel', 'quantity

': 7}, {'code': 'ogre_eye', 'quantity': 4}, {'code': 'ogre_skin', 'quantity': 3}, {'code': 'jasper_crystal', 'quantity': 1}], 'quantity': 1}}, {'name': 'Steel', 'code': 'steel', '

level': 20, 'type': 'resource', 'subtype': 'alloy', 'description': '', 'effects': [], 'craft': {' skill': 'mining', 'level': 20, 'items': [{'code': 'iron_ore', 'quantity': 3}, {'code': 'coal', ' quantity': 5}], 'quantity': 1}}], 'Resources': [{'name': 'Iron Ore', 'code': 'iron_ore', 'level': 10, 'type': 'resource', 'subtype': 'mining', 'description': '', 'effects': [], 'craft': None, ' sources': ['iron_rocks']}, {'name': 'Coal', 'code': 'coal', 'level': 20, 'type': 'resource', ' subtype': 'mining', 'description': '', 'effects': [], 'craft': None, 'sources': ['coal_rocks']}, {'name': 'Ogre Eye', 'code': 'ogre_eye', 'level': 20, 'type': 'resource', 'subtype': 'mob', ' description': '', 'effects': [], 'craft': None}, {'name': 'Ogre Skin', 'code': 'ogre_skin', 'level': 20, 'type': 'resource', 'subtype': 'mob', 'description': '', 'effects': [], 'craft': None}, {'name': 'Jasper Crystal', 'code': 'jasper_crystal', 'level': 15, 'type': 'resource', 'subtype': 'mining ', 'description': '', 'effects': [], 'craft': None, 'sources': ['jasper_rocks']}], ' Locations': [{'name': 'Forest', 'skin': 'forest_ironore2', 'x': 1, 'y': 7, 'content': {'type': ' resource', 'code': 'iron_rocks'}}, {'name': 'Forest', 'skin': 'forest_coal1', 'x': 1, 'y': 6, ' content': {'type': 'resource', 'code': 'coal_rocks'}}, {'name': 'Forest', 'skin': 'forest_ogre1', 'x': -5, 'y': -5, 'content': {'type': 'monster', 'code': 'ogre'}}, {'name': 'Forest', 'skin': ' forest_ogre2',

- 'x': -5, 'y': -4, 'content': {'type': 'monster', 'code': 'ogre'}}, {'name': 'Forest', ' skin':

'forest_2', 'x': 0, 'y': 6, 'content': {'type': 'resource', 'code': 'jasper_rocks'}}, {' name': 'Graveyard', 'skin': 'forest_skeleton5', 'x': 9, 'y': 7, 'content': {'type': 'monster', 'code': 'lich'}}, {'name': 'Forest (Forge)', 'skin': 'forest_miningstation1', 'x': 1, 'y': 5, ' content': {'type': 'workshop', 'code': 'mining'}}, {'name': 'City', 'skin': ' forest_jewelrycrafting1', 'x': 1,

- 'y': 3, 'content': {'type': 'workshop', 'code': 'jewelrycrafting'}}], 'Items stats': []}

Character Stats: {'name': 'Hero', 'skin': 'men1', 'level': 30, 'xp': 0, 'max_xp': 150, 'mining_level':

20, 'mining_xp': 0, 'mining_max_xp': 150, 'woodcutting_level': 1, 'woodcutting_xp': 0, 'woodcutting_max_xp': 150, 'fishing_level': 1, 'fishing_xp': 0, 'fishing_max_xp': 150, 'weaponcrafting_level': 1, 'weaponcrafting_xp': 0, 'weaponcrafting_max_xp': 150, 'gearcrafting_level': 1, 'gearcrafting_xp': 0, 'gearcrafting_max_xp': 150, '

jewelrycrafting_level': 20, 'jewelrycrafting_xp': 0, 'jewelrycrafting_max_xp': 150, 'cooking_level': 1, ' cooking_xp': 0, 'cooking_max_xp': 150, 'hp': 965, 'attack_fire': 0, 'attack_earth': 20, 'attack_water': 60, 'attack_air': 0, 'dmg_fire': 25, 'dmg_earth': 11, 'dmg_water': 103, 'dmg_air': 5, '

res_fire': 19, 'res_earth': 17, 'res_water': 24, 'res_air': 17, 'x': 0, 'y': 0, 'weapon_slot': 'greater_dreadful_staff', 'shield_slot': 'steel_shield', 'helmet_slot': 'gold_helm', 'body_armor_slot': 'lizard_skin_armor', 'leg_armor_slot': 'obsidian_legs_armor', '

boots_slot': 'gold_boots', 'ring1_slot': 'dreadful_ring', 'ring2_slot': '', 'amulet_slot': ' sapphire_amulet', 'artifact1_slot': '', 'artifact2_slot': '', 'artifact3_slot': '', 'consumable1_slot': '', 'consumable1_slot_quantity': 0, 'consumable2_slot': '', 'consumable2_slot_quantity': 0, 'inventory':

[]} Character stats include bonuses from equipped gear.

Your task is to kill 1 Lich. Make sure you can defeat it. You may need to craft and equip several items to do it. Make sure you have all resources

for crafting. Do not make assumptions, calculate everything. Think and write a sequence of actions to do it. End your answer with the following: 'Final answer: <python code to solve the task>'

### D Additional Charts

[Figure 4]

###### Figure 4: Two agent architectures were evaluated in the benchmark. A: A-1, operates in two phases: in the first phase, it generates a high-level plan; in the second, it decomposes this plan into executable actions. B: Agentic system A-2, is a modification of A-1 based on the idea of assigning each agent a single, one-bite task.

[Figure 5]

###### Figure 5: Score of LLMs across nine base-task difficulty levels. Solid lines correspond to reasoning-enabled (thinking) models, while dashed lines represent standard (non-thinking) variants.

[Figure 6]

###### Figure 6: Token use of LLMs across nine base-task difficulty levels. Solid lines correspond to reasoning-enabled (thinking) models, while dashed lines represent standard (non-thinking) variants.

[Figure 7]

###### Figure 7: Mean score of LLMs over nine base-task difficulty levels. Solid bars correspond to reasoning-enabled (thinking) models, while hatched bars represent standard (non-thinking) variants.

[Figure 8]

###### Figure 8: Mean success rate of LLMs over nine base-task difficulty levels. Solid bars correspond to reasoning-enabled (thinking) models, while hatched bars represent standard (non-thinking) variants.

[Figure 9]

###### Figure 9: Mean score per token of LLMs over nine base-task difficulty levels. Solid bars correspond to reasoning-enabled (thinking) models, while hatched bars represent standard (non-thinking) variants.

[Figure 10]

###### Figure 10: Mean success rate per token of LLMs over nine base-task difficulty levels. Solid bars correspond to reasoning-enabled (thinking) models, while hatched bars represent standard (non-thinking) variants.

