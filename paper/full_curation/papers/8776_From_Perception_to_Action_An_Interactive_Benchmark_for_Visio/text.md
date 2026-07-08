## From Perception to Action: An Interactive Benchmark for Vision Reasoning

Yuhao Wu*1 Maojia Song*1 Yihuai Lan*2 Lei Wang2 Zhiqiang Hu1 Yao Xiao1 Heng Zhou3 Weihua Zheng1 Dylan Raharja1 Soujanya Poria4 Roy Ka-Wei Lee1

# arXiv:2602.21015v1[cs.CV]24Feb2026

### Abstract

Understanding the physical structure is essential for real-world applications such as embodied agents, interactive design, and long-horizon manipulation. Yet, prevailing Vision–Language Model (VLM) evaluations still center on structureagnostic, single-turn setups (e.g., VQA), which fail to assess agents’ ability to reason about how geometry, contact, and support relations jointly constrain what actions are possible in a dynamic environment. To address this gap, we introduce the Causal Hierarchy of Actions and Interactions (CHAIN) benchmark, an interactive 3D, physics-driven testbed designed to evaluate whether models can understand, plan, and execute structured action sequences grounded in physical constraints. CHAIN shifts evaluation from passive perception to active problem solving, spanning tasks such as interlocking mechanical puzzles and causal-chain manipulation. We conduct a comprehensive study of state-ofthe-art VLMs and diffusion-based models under unified interactive settings. Our results show that top-performing models still struggle to internalize physical structure and causal constraints, often failing to produce reliable long-horizon plans and cannot robustly translate perceived structure into effective actions. The project is available at https://social-ai-studio.

github.io/CHAIN/.

### 1. Introduction

Real-world physical problem solving takes place in interactive settings involving rich visual feedback and physical

*Equal contribution 1Singapore University of Technology and Design (SUTD), Singapore 2Singapore Management University (SMU), Singapore 3University of Science and Technology of China (USTC), China 4Nanyang Technological University (NTU), Singapore. Correspondence to: Lei Wang <lei.wang.2019@phdcs.smu.edu.sg>, Roy Ka-Wei Lee <roy lee@sutd.edu.sg>.

Preprint. February 25, 2026.

[Figure 1]

Figure 1. Static vs. Interactive Evaluation of Physical Struction Reasoning. (a) Traditional VQA relies on passive observation of an image. (b) Our paradigm requires multi-step interaction, enabling procedural evaluation of planning and structural understanding.

constraints, where objects must be manipulated, configurations explored, and outcomes assessed under feasibility constraints. Success in such settings depends on tightly coupling perception, action, and iterative feedback over multiple steps, rather than on static scene understanding alone (Kim et al., 2024; Bai et al., 2025b). In these scenarios, the central challenge shifts from recognizing what is present to anticipating what actions are feasible, as feasibility is jointly determined by geometric structure, interfacial contacts, and multi-body support relations (Cheng et al., 2025; Li et al., 2024; Gu et al., 2023).

Given these requirements, an important question is whether existing evaluation protocols adequately capture such interactive, constraint-driven reasoning. However, current evaluations for Vision–Language Model (VLM) remain largely centered on static, single-turn formats such as Visual Question Answering (VQA), where performance is measured

by the correctness of a final textual answer (Agrawal et al., 2016; Lu et al., 2022; He et al., 2024; Wang et al., 2025; Chen et al., 2024; Lyu et al., 2024; Wang et al., 2024a; Guo et al., 2024; Xu et al.). While effective for diagnosing recognition and factual grounding, these benchmarks—by remaining largely static and single-turn—leave untested the agent’s ability to plan and adapt actions as constraints evolve, including whether it can anticipate how early choices will constrain (or preserve) the feasible action space later on; as a result, they tend to underestimate the difficulty of physical reasoning in interactive scenarios (Cheng et al., 2025). In particular, they do not evaluate whether an agent can reason about how early actions constrainor preserve future feasible action spaces. Alongside VLMs, diffusion-based models have been explored as another paradigm for reasoning and planning. However, existing evaluations of such approaches predominantly focus on simplified 2D environments that sidestep the challenges introduced by 3D geometry, contact constraints, and support relations (Taufeeque et al., 2025; Yang et al., 2025; Luo et al., 2025). As a result, it remains unclear whether either modern VLMs or diffusion models can reliably solve structured 3D physical problems whose feasibility is governed by hidden geometric constraints and contact-driven dependencies.

To address this gap, we introduce CHAIN (Causal Hierarchy of Actions and Interactions), an interactive 3D benchmark designed to evaluate whether models can understand, plan, and execute structured action sequences grounded in physical constraints. CHAIN reorients evaluation from passive perception to active problem solving in a physics–engine-driven environment1, where models must iteratively observe the environment, select feasible interactions, and revise their plans based on intermediate outcomes. CHAIN comprises task families that stress complementary aspects of structured physical reasoning: (1) Interlocking mechanical puzzles, which probe constraintaware reasoning over tightly coupled 3D objects, including contact-rich dependencies, non-intuitive feasible motions, and hidden geometric constraints; (2) 3D stacking and packing, which assess maintaining global feasibility over long horizons and stability reasoning under gravity, including support relations, balance constraints, and feasible assembly orders for stable configurations. Cross two families, these tasks emphasize structure-sensitive decision making: correct actions depend on understanding how geometry and constraints restrict what can be done next.

We conduct a comprehensive empirical study of state-ofthe-art VLMs and diffusion-based models under a unified interactive setting. Our results indicate that current models have difficulty internalizing physical structure and constraint

1Our physics environment is implemented with Unity (Unity Technologies, 2023; Juliani et al., 2020).

dependencies, often fail to maintain coherent multi-step strategies, and do not robustly translate perceived structure into effective actions over extended interactions. Overall, CHAIN provides a challenging benchmark and reproducible baselines to catalyze progress toward physically grounded, structure-aware interactive agents.

Our contributions are twofold:

- 1. We introduce and open-source CHAIN, an interactive 3D, physics-driven benchmark that shifts evaluation from static VQA to closed-loop physical problem solving. CHAIN comprises 109 distinct interactive levels with clear difficulty separation, designed to test whether agents can infer and exploit physical structureincluding hidden geometric constraints, contact-induced dependencies, and multi-body support relations—to select feasible actions over multi-step interaction.
- 2. We conduct a unified empirical study of state-of-theart VLMs and diffusion-based image-to-video models under the same interactive protocol and exposing persistent limitations in physical-structure-aware reasoning. In particular, current models often fail to convert perceived structure into valid action sequences as constraints tighten, struggling with feasibility reasoning grounded in 3D geometry, contact constraints, and support relations over long-horizon interactions.

### 2. CHAIN Benchmark

This section introduces our interactive 3D benchmark for rigorous evaluation of VLMs’ physical reasoning capabilities. We first outline the task families and their targeted capabilities (Section 2.1), then present the end-to-end pipeline for building controllable, reproducible environments (Section 2.2), and define metrics for correctness and efficiency under realistic interaction cost constraints (Section 2.3)

#### 2.1. Task Overview

Puzzle (Interlocking Mechanical Structures). The puzzle task family drawing inspiration from classical interlocking designs, including Kongming locks, Lu Ban locks, interlocking cubes, and burr puzzles. In these tasks, the agent is required to assemble or disassemble multi-piece structures mortise-and-tenon2 through fine-grained manipulation actions (e.g., pick, rotate, insert), where progress critically depends on executing steps in the correct order. The key challenge lies in reasoning about mechanical constraints, such as kinematic feasibility, collision avoidance, force directions, and the dependencies between components, making these tasks inherently long-horizon and causally structured.

2Mortise-and-tenon refers to a traditional interlocking joint structure that connects parts without nails or adhesives, widely used in classical Chinese craftsmanship.

Our puzzle suite consists of 32 task instances organized into three difficulty levels: 10 easy, 12 medium, and 10 hard, with difficulty naturally increasing from small sixpiece locks to complex designs involving more than thirty interlocking parts. Task success is determined by rule-based verification of the final configuration, requiring an exact match to the target state. Overall, these puzzles closely resemble real-world mechanical assembly and maintenance scenarios, where robust performance demands precise structural understanding, planning, and reliable execution of physically grounded actions.

Stacking (3D Spatial Packing). The stacking family focuses on spatial packing problems, where the agent must place multiple blocks with diverse geometric shapes into a fixed container (e.g., a cube or rectangular prism). At each step, the agent performs fine-grained manipulation operations (e.g., select, rotate, and place blocks) to incrementally construct a valid packing arrangement. These tasks primarily evaluate three-dimensional spatial reasoning, requiring the agent to jointly reason about shape compatibility, orientation constraints, and how early placement decisions progressively restrict the remaining free space.

Our stacking suite comprises 77 task instances, spanning 10 easy, 20 medium, and 47 hard task instances. The difficulty of the task is jointly determined by the container size, the number of objects and the complexity of the block shape. Importantly, the stacking environment is generated programmatically (Appendix A.5), enabling systematic control over container dimensions, block shapes, and object counts. As a result, stacking instances can be expanded to near-unlimited complexity, with problem difficulty scaling naturally as the geometry becomes more irregular and the feasible solution space becomes more constrained. Task success is determined by rule-based validation of the final configuration, requiring exact volume coverage without overlap or gaps, making this benchmark a scalable testbed to evaluate long-horizon spatial planning and constraint satisfaction3.

#### 2.2. Construction Pipeline

As illustrated in Figure 2, we build our interactive benchmark using a three-step pipeline that ensures: (i) the puzzles are intrinsically multi-step and require solutions with sequential causal dependencies; (ii) the environments are controllable, reproducible, and of moderate difficulty; and (iii) the evaluation faithfully captures both correctness and efficiency, going beyond simple success metrics.

3We present several examples in Appendix A.3.

- Step 1: Finding Suitable Puzzles. We begin by sourcing candidate puzzles from Puzzlemaster4 and filtering them using three criteria. First, we assess chain dependency. We retain only puzzles that require sequential reasoning and ordered action execution, where intermediate steps must be performed in a correct order and early actions causally influence the outcomes of later ones. Second, we evaluate feasibility. This includes whether a puzzle can be reliably modelled with a controllable action space and stable state transitions, and whether its difficulty lies within a regime that is both simulatable and informative for model evaluation. Third, we conduct a human difficulty check. Human experts5 play each candidate puzzle while we record completion time and assign difficulty labels accordingly. Based on solve time, puzzles are categorized as easy (< 5 minutes), medium (5–15 minutes), or hard (> 15 minutes). Stacking tasks are automatically generated by code rather than curated from external sources; we nevertheless use the same human solve-time protocol to assign difficulty tiers.
- Step 2: Building Physical Environments. To ensure experimental consistency, we unify the collected puzzles, spanning diverse structures and dynamics, into standardized interactive environments. We implement these environments using two complementary toolchains: Unity and a lightweight 3D Python engine. Unity is used for puzzles with complex interlocking mechanics (e.g., Kongming and Lu Ban locks), where precise control over kinematic constraints and contact interactions is required. In contrast, stacking-based spatial packing tasks, which involve simpler physical dynamics, are implemented in 3D Python for greater development efficiency. To provide a uniform interaction interface across tasks, we adopt a color-hinted control scheme: each object is assigned a distinct color, and the color–object mapping is exposed to the agent as additional metadata. The agent specifies objects by color when selecting and manipulating pieces. This design avoids introducing an additional action controller (as in VLA-style setups) that could confound the evaluation. Finally, we provide multi-view observations, allowing agents to inspect the scene from different viewpoints and reducing failures due to occlusion.

We evaluate each instance under a closed-loop protocol that treats the benchmark as an interactive platform, rather than a single-turn question answering setting. For each evaluation episode, we select one task instance from either the puzzle suite or the stacking suite and initialize the

- 4https://www.puzzlemaster.ca/
- 5We recruit two human experts, who are administrators of the corresponding puzzle interest groups in the forum. Each expert independently solves every candidate puzzle; we record their completion times and use the average solve time to assign the difficulty label.

[Figure 2]

Figure 2. Benchmark construction pipeline. We illustrate the end-to-end process for building our benchmark, including problem sourcing, document collection and filtering, concept annotation, regime construction, and final evaluation setup. The pipeline is designed to ensure controlled difficulty, minimize parametric leakage, and enable fine-grained analysis of reasoning and retrieval behaviors.

corresponding environment to its predefined initial state (including object poses, constraints, and the fixed action set). The agent then interacts with the environment over multiple steps. At step t, it receives (i) a task instruction specifying the goal state, (ii) a compact interaction history summarizing prior observations and actions, and (iii) the current multi-view visual observations rendered by the environment. Conditioned on these inputs, the agent selects an action from the predefined action space; the simulator executes the action to update the environment state and returns new observations for step t+1. This perception– action loop continues until the agent solves the instance or a predefined step budget is reached. After the episode ends, we compute evaluation metrics from the full trajectory, including both the final success signal and the recorded interaction trace. Beyond task success, we also measure how efficiently the agent achieves the goal, which motivates the metric design described below.

#### 2.3. Metric

We evaluate agents from three complementary perspectives: (i) task success, (ii) plan efficiency, and (iv) cost efficiency. Together, they measure not only whether the agent solves the task, but also how efficiently it executes actions and how expensive the interaction is under realistic billing.

Notation. We evaluate N tasks (indexed by i). Each task yields a binary outcome si ∈ 0,1 (1 if solved, 0 otherwise). During interaction, the agent executes atomic code actions, and we denote the number of executed actions by stepsi ∈ N. For each task, we also compute a task-specific minimal plan length oi ∈ N (i.e., the shortest valid solution plan under the environment rules). For best-of-K evaluation, we sample K independent runs ai,jKj=1 per task. We use ⊮(·) to denote the indicator function.

- (1) Task success. Task success measures how often an agent correctly completes a task per our evaluation protocol.

Pass@1 (single-attempt success rate) reports the fraction of tasks that are solved in a single run:

Pass@1 =

1 N

N

i=1

⊮si = 1 (1)

where we evaluate N task instances indexed by i, and si ∈ 0,1 indicates whether the agent successfully solves task i in its only attempt. Each instance therefore contributes 1 if solved and 0 otherwise, and Pass@1 is simply the mean success indicator across the benchmark.

- (2) Plan efficiency (conditioned on success). Task success rate alone can mask inefficient behaviors such as redundant actions, detours, or trial-and-error. To disentangle efficiency from task difficulty, we evaluate plan efficiency

only on solved tasks. Let S = {i : si = 1} denote the set of solved tasks.

Average Steps measures the length of successful executions (mean number of actions over solved tasks):

1 |S| i∈S

AvgStepssolved =

stepsi (2)

For instance, if a task has an optimal length oi = 3 but the agent solves it using 6 steps, it contributes 6 to

AvgStepssolved. Distance-to-Optimal quantifies avoidable overhead by counting extra steps beyond the minimal plan:

1 |S| i∈S

Dist2Opt =

max 0, stepsi − oi (3)

This compares each successful trajectory against the taskspecific optimum, thereby separating inefficiency from inherent task difficulty. For example, solving a 3-step puzzle in 5 steps adds 2 to Dist2Opt.

Normalized Distance provides a scale-free measure by normalizing the overhead by oi, making tasks with different optimal lengths comparable:

max(0, stepsi − oi) max(1, oi)

1 |S| i∈S

NormDist =

(4)

For example, a 10-step optimal plan solved in 15 steps yields 0.5, whereas a 3-step plan solved in 4 steps yields 13.

(3) Token and cost efficiency. Beyond success and action efficiency, we measure how cost-effective an agent is to deploy in terms of both tokens and dollars. We count all billable tokens under the target tokenizer, including prompts, tool schemas, tool inputs/outputs, and model replies, aggregated over all interaction rounds. Let tokens in(i) and tokens out(i) denote the total input and output tokens used on task i, and let si ∈ {0,1} indicate whether task i is solved. To summarize token efficiency, we report Solved/1M Tokens:

N i=1 si

N i=1 tokens in(i) + tokens out(i) × 106

Solved/Tokens =

(5)

This metric rewards agents that solve more tasks while consuming fewer tokens, capturing differences in interaction length and verbosity. To translate token usage into monetary cost, we use provider prices pin and pout (USD per 1K tokens) and define the cost of task i as

pin · tokens in(i) + pout · tokens out(i) 1000

Cost(i) =

(6) We then report Solved/(1 USD):

N i=1 si

Solved/USD =

N i=1 Cost(i)

(7)

In practice, these measures quantify the deployment tradeoff between lightweight agents and larger models: similar task success can correspond to very different token footprints and therefore substantially different costs.

### 3. Experiments

#### 3.1. Experimental Setup

We evaluate a diverse set of state-of-the-art models under a unified decoding and interaction protocol to ensure fair comparison in the interactive physical reasoning setting of CHAIN. Concretely, our evaluated pool includes 1) closed-source models: GPT5.2(Singh et al., 2025), Openaio3(OpenAI et al., 2024), Claude-Opus-4.5(Anthropic,

2025a), Claude-Sonnet-4.5(Anthropic, 2025b), Gemini-3Pro (prev.)(Google DeepMind, 2025b), Gemini-3-Flash (prev)(Google DeepMind, 2025a), Seed-1.6-Flash and Seed-1.6(Seed, 2025); 2) open-source models: Qwen3VL-235B-Ins, Qwen3-VL-30B-A3B-Ins, Qwen3-VL-235BThinking, Qwen3-VL-30B-A3B-Thinking, Qwen3-VL-8BThinking, Qwen3-VL-4B-Thinking (Bai et al., 2025a), GLM-

- 4.6V (Team et al., 2026) and Kimi-k2.5(Team et al., 2025). (details provided in Appendix A.2, spanning both generalpurpose VLMs and recent strong open/closed models.

We use identical generation hyperparameters across all models: temperature is fixed to 0.6, and nucleus sampling is set to top-p = 0.95. For each task instance, the model interacts with the environment through the same action API, and all atomic actions are executed by the system proxy to factor out low-level control and isolate operation-level decision making. We additionally cap the interaction budget at 30 − 60 steps per instance, which is substantially larger than the optimal solution length for all tasks, ensuring that failures are not driven by insufficient rollout horizon. Finally, we set the trajectory history window to 5 (i.e., the model conditions on only the most recent five interaction turns when making each decision).

3.2. Main Experiment

Table 1 reports the performance of a broad set of closedand open-source models on CHAIN across all task families and difficulty levels. Overall, closed-source models dominat the leaderboard. GPT-5.2 achieves the best aggregate result (Pass@1=22.9, Succ Task=25) and also attains the highest Stacking score (31.2). Among open-source models, Kimik2.5 is the strongest overall (Pass@1=13.8, Succ Task=15). Despite steady gains, CHAIN remains challenging: the 3D puzzle is a major bottleneck for all evaluated VLMs. In particular, Puzzle success is consistently far below Stacking (most models reach only ≈ 0.0–3.1 on Puzzle, versus 10.4–31.2 on Stacking). This gap suggests that even stateof-the-art models struggle with the structural inference and constrained, sequential manipulation required by the hardest human-designed 3D puzzles. Qualitatively, many models fail early and stall around the second level, indicating that failures are driven less by minor execution noise and more by difficulty in identifying valid intermediate constraints.

Higher success does not always mean better efficiency or lower cost. Stronger models often backtrack and revise their plans, which increases AvgSteps/Dist2Opt/NormDist and raises Solved/Tokens and Solved/USD. For instance, GPT-

- 5.2 is relatively expensive per solved task ($1.3/level,)). In contrast, lightweight “flash” models can be extremely cheap but achieve much lower success. To visualize the cost–quality trade-off, Fig. 3 plots per-solved cost and token usage on the x-axis against overall success on the y-axis;

Task Success Plan Efficiency (Solved) Token & Cost

Models

Pass@1 ↑ Succ.Task ↑ Puzzle ↑ Stacking ↑ AvgSteps ↓ Dist2Opt ↓ N.Dist ↓ Solved/Tokens ↑ Solved/USD ↑ Closed-source Models

GPT-5.2 22.9 25 3.1 31.2 9.6 1.43 0.50 10.1 0.10 GPT-5-mini 11.0 12 3.1 14.3 6.8 1.08 0.57 19.3 0.19 OpenAI-o3 10.1 11 3.1 13.0 6.1 0.81 0.37 26.1 0.26 Claude-Opus-4.5 15.6 17 3.1 20.8 8.0 1.42 0.63 17.8 0.17 Claude-Sonnet-4.5 13.8 15 3.1 18.2 7.1 1.02 0.51 29.0 0.29 Gemini-3-Pro 19.3 21 3.1 26.0 10.4 2.38 1.29 4.63 0.04 Gemini-3-Flash 11.9 13 3.1 15.6 6.1 0.84 0.46 36.9 0.36 Seed-1.6 10.1 11 3.1 13.0 5.5 0.73 0.39 23.9 0.23 Seed-1.6-Flash 7.3 8 0.0 10.4 4.5 0.23 0.07 32.4 0.32

Open-source Models

Kimi-k2.5 13.8 15 0.0 19.5 7.0 0.71 0.19 6.2 0.06 GLM-4.6V 7.3 8 0.0 10.4 6.5 0.92 0.39 22.6 0.22 Qwen3-VL-235B-A22B-Thk. 9.2 10 3.1 11.7 3.8 0.33 0.31 40.7 0.40 Qwen3-VL-30B-A3B-Thk. 10.1 11 0.0 14.3 5.5 0.40 0.11 18.8 0.18 Qwen3-VL-8B-Thinking 8.3 9 0.0 11.7 3.8 0.03 0.01 19.8 0.19 Qwen3-VL-235B-A22B-Inst 8.3 9 3.1 10.4 3.7 0.13 0.11 66.8 0.66 Qwen3-VL-30B-A3B-Inst 3.7 4 3.1 3.9 4.8 1.30 1.14 51.4 0.51

- Table 1. Main model evaluation results. We report overall task success on the full benchmark (Pass@1 (All)) and subset success on Puzzle & Stacking. Plan-efficiency metrics (AvgSteps, Dist2Opt, NormDist; lower = better) are computed only for solved tasks. Token and monetary costs (Solved/Tokens, Solved/USD) are normalized by successful tasks.

points closer to the upper-left represent better value. Notably, in easy regimes, extremely small models are not always the most cost-effective choice: low success can inflate cost-per-success, whereas a moderately stronger model may solve more reliably with fewer failed attempts, yielding a better effective return despite higher per-call expense.

Finally, we analyze representative failure cases to pinpoint what most limits progress on each task family. For Puzzle, the primary obstacle is weak perception and utilization of physical structure: even when partial internal structure is revealed as a reference, many models still cannot reliably identify the first critical move that unlocks the puzzle. As a result, they resort to extensive trial-and-error over candidate blocks, leading to largely random exploration with little constraint-guided progress. For Stacking, two factors dominate failures. First is object-set complexity and coupling: as illustrated in Fig. 6, while easy instances can be solved with locally reasonable placements, mid and hard levels require tightly coupled packing decisions to maximize space utilization. Second is global spatial planning: most models greedily place “easy” items early, only to later discover that the remaining free volume is fragmented, forcing costly removals and replanning. Stronger models such as GPT-5.2 and Gemini-Pro exhibit more deliberate lookahead, which helps avoid these dead-ends and explains a substantial portion of their advantage on difficult stacking regimes. For a specific discussion of the factors contributing to success and failure, please refer to Appendix A.4.

[Figure 3]

Figure 3. Cost and token efficiency with solved tasks comparison between models

#### 3.3. Catastrophic Failure of World Models

To further extend physical interaction into the setting of world models, we use our puzzle subtask to assess world models for producing step-by-step disassembly process under explicit physical constraints. Given a reference image of a Luban lock, models are instructed to disassemble all pieces while respecting contact, interlocking, and collision rules (see the detailed prompt in Table A.6). We evaluate state-of-the-art video generation models (SORA 2(Li et al., 2025), WAN 2.6(Wan et al., 2025), VEO 3.1(Wiedemer et al., 2025), KLING 2.6(Kuaishou Technology, 2025), and HUNYUANVIDEO 1.5(Team, 2025)) on two configurations: Level 1 (two beams) and Level 2 (six beams). Despite the highly specified prompts, none of the models successfully completes the disassembly; instead, they exhibit systematic, catastrophic failures with severe hallucinations shown in Figure 4, which become more pronounced as structural complexity increases.

First, some models exhibit superficial instruction-

[Figure 4]

Figure 4. Qualitative results on the Luban puzzle subtask across world models. Top: Level 1 (two beams). Bottom: Level 2 (six beams). All models fail to produce a physically valid disassembly, either violating interlocking constraints or hallucinating (e.g., structural corruption and object insertion/removal), with failures worsening at higher complexity.

Puzzle Acc ↑ Stacking Acc ↑ Easy Mid Hard Easy Mid Hard

Models

GPT5.2 10.0 0.0 0.0 100.0 55.0 6.3 Gemini-3-Pro 10.0 0.0 0.0 90.0 40.0 6.3 Sonnet-4.5 10.0 0.0 0.0 100.0 20.0 0.0

- Table 2. Difficulty-stratified accuracy (Acc) on CHAIN. We report Easy/Mid/Hard tiers for both Puzzle and Stacking.

following with physics violations. For example, SORA 2 and WAN 2.6 often extract a target beam via direct translation, even when such motion is physically infeasible due to interlocking constraints. It further degrades in Level 2, where these models increasingly fail to follow the prescribed step-by-step procedure and instead perform random or under-specified actions that remain physically invalid.

Second, other models undergo a more severe representational collapse, losing consistency of both object structure and object identity. In Level 1, VEO 3.1, KLING 2.6, and HUNYUANVIDEO 1.5 frequently generate corrupted configurations, including distorted geometry and spurious components. In Level 2, this behaviour escalates into full hallucination: models may add, remove, or merge beams, or transform the puzzle into an unrecognisable structure. Such failures indicate a breakdown of object permanence and constraint consistency under multi-part coupling.

Overall, these results demonstrate that although modern world models can generate visually plausible motion or simple physical event transitions, they remain fundamentally unreliable for structured, constraint-driven interaction. In particular, tasks requiring multi-step manipulation grounded in object-centric reasoning and physical feasibility remain beyond the capabilities of current models, highlighting an important direction for future world model research.

- 3.4. Impact of Difficulty Stratification

Difficulty in Stacking increases with both the number of blocks and the complexity of support relations, making success increasingly dependent on long-horizon planning. This is reflected in Table 2: Stacking-Easy is essentially solved by

Interactive (%) ↑ One-shot (%) ↑ ∆ Puzzle Stacking All Puzzle Stacking All

Models

GPT5.2 3.1 31.2 22.9 0.0 9.1 7.1 -15.8 Sonnet-4.5 3.1 18.2 13.8 0.0 10.3 8.1 -5.7 Gemini-3-Pro 3.1 26.0 19.3 0.0 9.1 7.1 -12.2

Table 3. Interactive vs. one-shot solving on CHAIN. In one-shot, the model receives only a single fixed-view image and must output a complete (or near-complete) solution with minimal or no intermediate feedback. ∆ All Avg denotes the performance gap between interactive and one-shot settings (Interactive − One-shot).

top models (e.g., GPT5.2 and Claude-Sonnet-4.5 at 100.0%), but the gap opens on Stacking-Mid (e.g., GPT5.2 at 55.0% vs. Claude-Sonnet-4.5 at 20.0%) and performance collapses on Stacking-Hard (best results only 6.3%).

In contrast, Puzzle is consistently harder across tiers: even on Puzzle-Easy models peak at only 10.0%, and PuzzleMid/Hard remains at 0.0%. This suggests the dominant bottleneck is not a smooth easy-to-hard degradation, but the fundamentally higher difficulty of 3D interlocking/structurecentric reasoning itself—requiring inference of hidden blocking constraints and feasible multi-step disentanglement trajectories from partial observations6.

#### 3.5. One-shot Solving without Interaction

A key motivation for interactive evaluation is to distinguish offline planning from closed-loop adaptation. In physical problem solving, interaction can compensate for imperfect priors: a model may probe the environment, observe feasibility constraints, and revise its plan accordingly. To isolate how much success on CHAIN depends on such feedback, we construct a one-shot setting where the model receives only a single fixed-view image and must output a complete solution with minimal or no intermediate feedback. We then compare one-shot accuracy with the interactive setting; the gap ∆ (Interactive − One-shot) quantifies the benefit brought by interaction.

6A more detailed description of the difficulty-level definitions can be found in Appendix A.3

horizon interactive planning, and more reliable verifier-style signals appear necessary for consistent gains.

Strategy Puzzle(%) ↑ Stacking(%) ↑ All(%) ↑ ∆ vs. Avg@4 ↑ Avg@4 3.1 15.5 9.3 –

- Pass@1 3.1 15.6 9.4 +0.1

- Pass@2 3.1 19.4 11.2 +1.9 Pass@4 3.1 19.4 11.2 +1.9

### 4. Related Work

VLM judge 3.1 18.1 10.3 +1.3 Reward model 3.1 16.8 9.9 +0.6

Reasoning of Vision-Language Models. While VLMs effectively align visual perception with language reasoning, they remain largely confined to static scenes and singlestep inference with limited temporal dynamics (Sarch et al., 2025). Embodied extensions of these agents (Liu et al., 2024) still primarily rely on instantaneous observations, frequently failing in multi-round physical interactions (Guo et al., 2024). Despite recent advances in multi-stage supervision and reinforcement learning (Xu et al.; Guo et al.,

Table 4. Comparison of multi-sample selection and reward-model (RM) reranking on CHAIN by kimi-k2.5. All strategies use the same base generator and differ only in candidate selection. ∆ is computed w.r.t. Avg@4 on All Avg.

Table 3 shows that one-shot performance is uniformly lower, which further highlights gains of interactions. This indicates that CHAIN cannot be reliably solved by pre-computed reasoning from a single view. For Puzzle, one-shot accuracy collapses to 0.0% for all evaluated models, while interactive accuracy reaches 3.1%, suggesting that even modest success requires iterative constraint discovery rather than a fully inferred disassembly plan from the initial observation. For Stacking, interaction is even more critical: GPT5.2 drops from 31.2% (interactive) to 9.1% (one-shot), Gemini-3-Pro shows the same drop (26.0% → 9.1%), and Claude-Sonnet-

- 2025; Liu et al., 2025; Kang et al., 2025), these strategies are mostly validated on static benchmarks. CHAIN addresses this by targeting underexamined causal and interactive reasoning over multi-step physical processes.

Physical Benchmarks. Physical reasoning evaluation has evolved from visual plausibility in simplified settings (Riochet et al., 2018; Rajani et al., 2020) to synthetic primitives in controlled environments (Yi et al., 2019; Wang et al., 2024b; Zheng et al., 2024; Tung et al., 2023). While many benchmarks emphasize commonsense QA (Lu et al., 2022; He et al., 2024; Wang et al., 2025) or broader perceptioncentered tasks (Chow et al., 2025), they remain largely static. CHAIN shifts this paradigm toward dynamic, interactiondriven evaluation of multi-step causal processes.

3D Structure Perception. Foundational work in 3D structure perception (Chen et al., 2024; Lyu et al., 2024; Wang et al., 2024a) has enabled models to reason about depth and relative positioning via multi-view or point-cloud representations (Zhao et al., 2025; Lu et al., 2025; Ma et al.,

- 2026; Wang et al., 2026). However, these efforts primarily prioritize static scene reconstruction or snapshot reasoning. Consequently, existing benchmarks rarely assess how spatial configurations evolve through interaction or how actions induce causal changes over time.

- 4.5 decreases from 18.2% to 10.3%. Aggregated over all tasks, the All Avg accuracy decreases sharply in one-shot. Overall, these consistent gaps support that CHAIN genuinely evaluates closed-loop physical-structure reasoning under evolving feasibility constraints, rather than one-shot recognition or static plan synthesis.

#### 3.6. Reward Models vs. Verifier-based Checking

RLVR indicates verification greatly aids domains with deterministic checking (e.g., math/code). We question if this applies to interactive physical reasoning—where success relies on long action sequences and errors may emerge late. Supervision is tougher here: reward models often favor locally plausible moves, whereas verifier-style checks are more conservative but environment-based.

We therefore compare RM-based selection(Ong et al., 2025) with verifier-based checking for choosing candidate actions/plans (e.g., execution-grounded validity/consistency), keeping the generator fixed, and testing lightweight strategies like reranking and pairwise judging and using Kimik2.5 as action model (Table 4).

### 5. Conclusion

We introduced CHAIN , an interactive 3D benchmark that shifts the evaluation of vision-language models from passive perception to closed-loop, multi-step physical reasoning. Across interlocking puzzles and 3D stacking/packing, models must repeatedly observe outcomes, choose feasible interactions, and revise plans as geometry, contacts, and support relations restrict what can be done next. Our evaluation of state-of-the-art VLMs and diffusion-based image-to-video models shows a clear pattern: performance degrades sharply as structural constraints tighten, and many models fail to maintain coherent multi-step strategies even when they can

Results show Pass@1 and Avg@4 are nearly identical overall, implying limited sampling variance. Multi-sample pass selection helps but saturates, while Pass@4 adds no further gain. RM reranking yields smaller improvements (+0.6), whereas a stronger VLM pairwise judge does better (+1.3) but still trails Pass@2. Beam search (batch size=2) with Reward also cannot match Pass@2, suggesting the bottleneck is selection-signal quality rather than decoding compute. Overall, current vision RMs offer limited leverage for long-

perceive the scene correctly. The most striking failure occurs on interlocking mechanical puzzles, where hidden geometric constraints cause near-total collapse; stacking tasks further expose brittle stability and long-horizon feasibility reasoning. These results highlight a persistent gap between “seeing” and “acting”: current models rarely internalize how early actions reshape the future feasible action space.

### Impact Statements

This paper presents work whose goal is to advance machine learning by enabling more rigorous evaluation of interactive, physics-grounded vision reasoning through an open benchmark and reproducible baselines. There are many potential societal consequences of improved interactive perceptionand-action systems, but we do not anticipate any direct negative impacts from releasing this evaluation benchmark beyond standard concerns for open research artifacts.

### References

Agrawal, A., Lu, J., Antol, S., Mitchell, M., Zitnick, C. L., Batra, D., and Parikh, D. Vqa: Visual question answering, 2016. URL https://arxiv.org/abs/1505. 00468.

Anthropic. Claude opus 4.5 system card. Technical report, Anthropic, November 2025a. URL https://www.anthropic.com/ claude-opus-4-5-system-card.

Anthropic. Claude sonnet 4.5 system card. Technical report, Anthropic, October 2025b. URL https://www.anthropic.com/ claude-sonnet-4-5-system-card.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025b. URL https://arxiv.org/abs/2502.13923.

Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., and Xia, F. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455–14465, June 2024.

Cheng, Z., Tu, Y., Li, R., Dai, S., Hu, J., Hu, S., Li, J., Shi, Y., Yu, T., Chen, W., Shi, L., and Sun, M. Embodiedeval: Evaluate multimodal llms as embodied agents, 2025. URL https://arxiv.org/abs/2501.11858.

Chow, W., Mao, J., Li, B., Seita, D., Guizilini, V., and Wang, Y. Physbench: Benchmarking and enhancing visionlanguage models for physical world understanding, 2025. URL https://arxiv.org/abs/2501.16411.

Google DeepMind. Gemini 3 flash: Model card. Technical report, Google DeepMind, December 2025a. URL https://storage.googleapis. com/deepmind-media/Model-Cards/ Gemini-3-Flash-Model-Card.pdf.

Google DeepMind. Gemini 3 pro: Model card. Technical report, Google DeepMind, November 2025b. URL https://storage.googleapis. com/deepmind-media/Model-Cards/ Gemini-3-Pro-Model-Card.pdf.

Gu, J., Xiang, F., Li, X., Ling, Z., Liu, X., Mu, T., Tang, Y., Tao, S., Wei, X., Yao, Y., Yuan, X., Xie, P., Huang, Z., Chen, R., and Su, H. Maniskill2: A unified benchmark for generalizable manipulation skills, 2023. URL https:

//arxiv.org/abs/2302.04659.

Guo, D., Xiang, Y., Zhao, S., Zhu, X., Tomizuka, M., Ding, M., and Zhan, W. Phygrasp: Generalizing robotic grasping with physics-informed large multimodal models, 2024. URL https://arxiv.org/abs/2402.

16836.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. arXiv, 2024.

Juliani, A., Berges, V.-P., Teng, E., Cohen, A., Harper, J., Elion, C., Goy, C., Gao, Y., Henry, H., Mattar, M., and Lange, D. Unity: A general platform for intelligent agents, 2020. URL https://arxiv.org/ abs/1809.02627.

Kang, L., Song, X., Zhou, H., Qin, Y., Yang, J., Liu, X., Torr, P., Bai, L., and Yin, Z. Viki-r: Coordinating embodied multi-agent cooperation via reinforcement learning. arXiv preprint arXiv:2506.09049, 2025.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., and Finn, C. Openvla: An open-source vision-language-action model, 2024. URL https://arxiv.org/abs/2406.09246.

Kuaishou Technology. Kling ai 2.6: Next-generation video generation model. https://klingai.com/, dec

2025. Accessed: 2026-01-28.

Li, C., Zhang, R., Wong, J., Gokmen, C., Srivastava, S., Mart´ın-Mart´ın, R., Wang, C., Levine, G., Ai, W., Martinez, B., Yin, H., Lingelbach, M., Hwang, M., Hiranaka, A., Garlanka, S., Aydin, A., Lee, S., Sun, J., Anvari, M., Sharma, M., Bansal, D., Hunter, S., Kim, K.-Y., Lou, A., Matthews, C. R., Villa-Renteria, I., Tang,

J. H., Tang, C., Xia, F., Li, Y., Savarese, S., Gweon, H., Liu, C. K., Wu, J., and Fei-Fei, L. Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation, 2024. URL https://arxiv.org/abs/2403.09227.

Li, H., Okhonko, D., Verma, A., Zhang, E., Wang, R., Luhman, T., Luhman, E., Wallace, B., Mintun, E., Chang, M., et al. Sora 2. https://openai.com/ zh-Hans-CN/index/sora-2/, sep 2025. Full author list includes the OpenAI Sora 2 Research and Product teams. Accessed: 2026-01-28.

Liu, F., Fang, K., Abbeel, P., and Levine, S. Moka: Openvocabulary robotic manipulation through mark-based visual prompting. arXiv, 2024.

Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., and Wang, J. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.

Lu, M., Xu, R., Fang, Y., Zhang, W., Yu, Y., Srivastava, G., Zhuang, Y., Elhoseiny, M., Fleming, C., Yang, C., Tu, Z., Xie, Y., Xiao, G., Wang, H., Jin, D., Shi, W., and Wang, X. Scaling agentic reinforcement learning for tool-integrated reasoning in vlms, 2025. URL https:

//arxiv.org/abs/2511.19773.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Luo, Y., Zhao, X., Lin, B., Zhu, L., Tang, L., Liu, Y., Chen, Y.-C., Qian, S., Wang, X., and You, Y. V-reasonbench: Toward unified reasoning benchmark suite for video generation models, 2025. URL https://arxiv.org/ abs/2511.16668.

Lyu, R., Wang, T., Lin, J., Yang, S., Mao, X., Chen, Y., Xu, R., Huang, H., Zhu, C., Lin, D., et al. Mmscan: A multi-modal 3d scene dataset with hierarchical grounded language annotations. arXiv, 2024.

Ma, W., Wang, C., Yuan, R., Chen, H., Dai, N., Zhou, S. K., Yang, Y., Yuille, A., and Chen, J. Causalspatial: A benchmark for object-centric causal spatial reasoning, 2026. URL https://arxiv.org/abs/2601.13304.

Ong, B., Pala, T. D., Toh, V., Tjhi, W. C., and Poria, S. Training vision-language process reward models for testtime scaling in multimodal reasoning: Key insights and lessons learned, 2025. URL https://arxiv.org/ abs/2509.23250.

OpenAI, :, Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., Iftimie, A., Karpenko, A., Passos,

- A. T., Neitz, A., Prokofiev, A., Wei, A., Tam, A., Bennett,

- A., Kumar, A., Saraiva, A., Vallone, A., Duberstein, A., Kondrich, A., Mishchenko, A., Applebaum, A., Jiang, A., Nair, A., Zoph, B., Ghorbani, B., Rossen, B., Sokolowsky,
- B., Barak, B., McGrew, B., Minaiev, B., Hao, B., Baker,

B., Houghton, B., McKinzie, B., Eastman, B., Lugaresi, C., Bassin, C., Hudson, C., Li, C. M., de Bourcy, C., Voss,

- C., Shen, C., Zhang, C., Koch, C., Orsinger, C., Hesse, C., Fischer, C., Chan, C., Roberts, D., Kappler, D., Levy,
- D., Selsam, D., Dohan, D., Farhi, D., Mely, D., Robinson,

- D., Tsipras, D., Li, D., Oprica, D., Freeman, E., Zhang,
- E., Wong, E., Proehl, E., Cheung, E., Mitchell, E., Wallace, E., Ritter, E., Mays, E., Wang, F., Such, F. P., Raso,
- F., Leoni, F., Tsimpourlas, F., Song, F., von Lohmann, F., Sulit, F., Salmon, G., Parascandolo, G., Chabot, G., Zhao, G., Brockman, G., Leclerc, G., Salman, H., Bao,

- H., Sheng, H., Andrin, H., Bagherinezhad, H., Ren, H., Lightman, H., Chung, H. W., Kivlichan, I., O’Connell,
- I., Osband, I., Gilaberte, I. C., Akkaya, I., Kostrikov, I., Sutskever, I., Kofman, I., Pachocki, J., Lennon, J., Wei,
- J., Harb, J., Twore, J., Feng, J., Yu, J., Weng, J., Tang, J., Yu, J., Candela, J. Q., Palermo, J., Parish, J., Heidecke,

- J., Hallman, J., Rizzo, J., Gordon, J., Uesato, J., Ward,

- J., Huizinga, J., Wang, J., Chen, K., Xiao, K., Singhal,
- K., Nguyen, K., Cobbe, K., Shi, K., Wood, K., Rimbach,

- K., Gu-Lemberg, K., Liu, K., Lu, K., Stone, K., Yu, K., Ahmad, L., Yang, L., Liu, L., Maksin, L., Ho, L., Fedus,
- L., Weng, L., Li, L., McCallum, L., Held, L., Kuhn, L., Kondraciuk, L., Kaiser, L., Metz, L., Boyd, M., Trebacz,
- M., Joglekar, M., Chen, M., Tintor, M., Meyer, M., Jones, M., Kaufer, M., Schwarzer, M., Shah, M., Yatbaz, M., Guan, M. Y., Xu, M., Yan, M., Glaese, M., Chen, M., Lampe, M., Malek, M., Wang, M., Fradin, M., McClay,

- M., Pavlov, M., Wang, M., Wang, M., Murati, M., Bavarian, M., Rohaninejad, M., McAleese, N., Chowdhury,
- N., Chowdhury, N., Ryder, N., Tezak, N., Brown, N., Nachum, O., Boiko, O., Murk, O., Watkins, O., Chao, P., Ashbourne, P., Izmailov, P., Zhokhov, P., Dias, R., Arora, R., Lin, R., Lopes, R. G., Gaon, R., Miyara, R., Leike, R., Hwang, R., Garg, R., Brown, R., James, R., Shu, R., Cheu, R., Greene, R., Jain, S., Altman, S., Toizer, S., Toyer, S., Miserendino, S., Agarwal, S., Hernandez, S., Baker, S., McKinney, S., Yan, S., Zhao, S., Hu, S., Santurkar, S., Chaudhuri, S. R., Zhang, S., Fu, S., Papay, S., Lin, S., Balaji, S., Sanjeev, S., Sidor, S., Broda, T., Clark, A., Wang, T., Gordon, T., Sanders, T., Patwardhan, T., Sottiaux, T., Degry, T., Dimson, T., Zheng, T., Garipov, T., Stasi, T., Bansal, T., Creech, T., Peterson, T., Eloundou, T., Qi, V., Kosaraju, V., Monaco, V., Pong, V., Fomenko, V., Zheng, W., Zhou, W., McCabe, W., Zaremba, W., Dubois, Y., Lu, Y., Chen, Y., Cha, Y., Bai, Y., He, Y., Zhang, Y., Wang, Y.,

Shao, Z., and Li, Z. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720.

Rajani, N. F., Zhang, R., Tan, Y. C., Zheng, S., Weiss, J., Vyas, A., Gupta, A., Xiong, C., Socher, R., and Radev, D. Esprit: Explaining solutions to physical reasoning tasks. arXiv, 2020.

Riochet, R., Castro, M. Y., Bernard, M., Lerer, A., Fergus, R., Izard, V., and Dupoux, E. Intphys: A framework and benchmark for visual intuitive physics reasoning. arXiv, 2018.

Sarch, G. H., Saha, S., Khandelwal, N., Jain, A., Tarr, M. J., Kumar, A., and Fragkiadaki, K. Grounded reinforcement learning for visual reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=1amnhVRQ3l.

Seed, B. Seed1. 6 tech introduction. Accessed on September, 28:2025, 2025.

Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., Nathan, A., Luo, A., Helyar, A., Madry, A., Efremov, A., Spyra, A., Baker-Whitcomb, A., Beutel, A., Karpenko, A., Makelov, A., Neitz, A., Wei, A., Barr, A., Kirchmeyer, A., Ivanov, A., Christakis, A., Gillespie, A., Tam, A., Bennett, A., Wan, A., Huang, A., Sandjideh, A. M., Yang, A., Kumar, A., Saraiva, A., Vallone, A., Gheorghe, A., Garcia, A. G., Braunstein, A., Liu, A., Schmidt, A., Mereskin, A., Mishchenko, A., Applebaum, A., Rogerson, A., Rajan, A., Wei, A., Kotha, A., Srivastava, A., Agrawal, A., Vijayvergiya, A., Tyra, A., Nair,

- A., Nayak, A., Eggers, B., Ji, B., Hoover, B., Chen, B., Chen, B., Barak, B., Minaiev, B., Hao, B., Baker, B., Lightcap, B., McKinzie, B., Wang, B., Quinn, B., Fioca,
- B., Hsu, B., Yang, B., Yu, B., Zhang, B., Brenner, B., Zetino, C. R., Raymond, C., Lugaresi, C., Paz, C., Hudson, C., Whitney, C., Li, C., Chen, C., Cole, C., Voss,
- C., Ding, C., Shen, C., Huang, C., Colby, C., Hallacy, C., Koch, C., Lu, C., Kaplan, C., Kim, C., Minott-Henriques, C., Frey, C., Yu, C., Czarnecki, C., Reid, C., Wei, C., Decareaux, C., Scheau, C., Zhang, C., Forbes, C., Tang,
- D., Goldberg, D., Roberts, D., Palmie, D., Kappler, D., Levine, D., Wright, D., Leo, D., Lin, D., Robinson, D., Grabb, D., Chen, D., Lim, D., Salama, D., Bhattacharjee,

- D., Tsipras, D., Li, D., Yu, D., Strouse, D., Williams, D., Hunn, D., Bayes, E., Arbus, E., Akyurek, E., Le, E. Y., Widmann, E., Yani, E., Proehl, E., Sert, E., Cheung, E., Schwartz, E., Han, E., Jiang, E., Mitchell, E., Sigler, E., Wallace, E., Ritter, E., Kavanaugh, E., Mays, E., Nikishin,
- E., Li, F., Such, F. P., de Avila Belbute Peres, F., Raso,
- F., Bekerman, F., Tsimpourlas, F., Chantzis, F., Song, F., Zhang, F., Raila, G., McGrath, G., Briggs, G., Yang, G.,

- Parascandolo, G., Chabot, G., Kim, G., Zhao, G., Valiant,
- G., Leclerc, G., Salman, H., Wang, H., Sheng, H., Jiang,
- H., Wang, H., Jin, H., Sikchi, H., Schmidt, H., Aspegren,

- H., Chen, H., Qiu, H., Lightman, H., Covert, I., Kivlichan,
- I., Silber, I., Sohl, I., Hammoud, I., Clavera, I., Lan, I., Akkaya, I., Kostrikov, I., Kofman, I., Etinger, I., Singal,

I., Hehir, J., Huh, J., Pan, J., Wilczynski, J., Pachocki, J., Lee, J., Quinn, J., Kiros, J., Kalra, J., Samaroo, J., Wang, J., Wolfe, J., Chen, J., Wang, J., Harb, J., Han, J., Wang,

- J., Zhao, J., Chen, J., Yang, J., Tworek, J., Chand, J., Landon, J., Liang, J., Lin, J., Liu, J., Wang, J., Tang, J., Yin,

- J., Jang, J., Morris, J., Flynn, J., Ferstad, J., Heidecke, J., Fishbein, J., Hallman, J., Grant, J., Chien, J., Gordon, J., Park, J., Liss, J., Kraaijeveld, J., Guay, J., Mo, J., Lawson,

- J., McGrath, J., Vendrow, J., Jiao, J., Lee, J., Steele, J., Wang, J., Mao, J., Chen, K., Hayashi, K., Xiao, K., Salahi,
- K., Wu, K., Sekhri, K., Sharma, K., Singhal, K., Li, K., Nguyen, K., Gu-Lemberg, K., King, K., Liu, K., Stone,

K., Yu, K., Ying, K., Georgiev, K., Lim, K., Tirumala, K., Miller, K., Ahmad, L., Lv, L., Clare, L., Fauconnet, L., Itow, L., Yang, L., Romaniuk, L., Anise, L., Byron, L., Pathak, L., Maksin, L., Lo, L., Ho, L., Jing, L., Wu,

- L., Xiong, L., Mamitsuka, L., Yang, L., McCallum, L., Held, L., Bourgeois, L., Engstrom, L., Kuhn, L., Feuvrier, L., Zhang, L., Switzer, L., Kondraciuk, L., Kaiser, L., Joglekar, M., Singh, M., Shah, M., Stratta, M., Williams,
- M., Chen, M., Sun, M., Cayton, M., Li, M., Zhang, M., Aljubeh, M., Nichols, M., Haines, M., Schwarzer, M., Gupta, M., Shah, M., Huang, M., Dong, M., Wang, M., Glaese, M., Carroll, M., Lampe, M., Malek, M., Sharman, M., Zhang, M., Wang, M., Pokrass, M., Florian,

- M., Pavlov, M., Wang, M., Chen, M., Wang, M., Feng,

- M., Bavarian, M., Lin, M., Abdool, M., Rohaninejad, M., Soto, N., Staudacher, N., LaFontaine, N., Marwell, N., Liu, N., Preston, N., Turley, N., Ansman, N., Blades, N., Pancha, N., Mikhaylin, N., Felix, N., Handa, N., Rai, N., Keskar, N., Brown, N., Nachum, O., Boiko, O., Murk, O., Watkins, O., Gleeson, O., Mishkin, P., Lesiewicz, P., Baltescu, P., Belov, P., Zhokhov, P., Pronin, P., Guo, P., Thacker, P., Liu, Q., Yuan, Q., Liu, Q., Dias, R., Puckett,

- R., Arora, R., Mullapudi, R. T., Gaon, R., Miyara, R., Song, R., Aggarwal, R., Marsan, R., Yemiru, R., Xiong,

- R., Kshirsagar, R., Nuttall, R., Tsiupa, R., Eldan, R., Wang, R., James, R., Ziv, R., Shu, R., Nigmatullin, R., Jain, S., Talaie, S., Altman, S., Arnesen, S., Toizer, S., Toyer, S., Miserendino, S., Agarwal, S., Yoo, S., Heon, S., Ethersmith, S., Grove, S., Taylor, S., Bubeck, S., Banesiu,
- S., Amdo, S., Zhao, S., Wu, S., Santurkar, S., Zhao, S., Chaudhuri, S. R., Krishnaswamy, S., Shuaiqi, Xia, Cheng,

- S., Anadkat, S., Fishman, S. P., Tobin, S., Fu, S., Jain, S., Mei, S., Egoian, S., Kim, S., Golden, S., Mah, S., Lin, S., Imm, S., Sharpe, S., Yadlowsky, S., Choudhry,

- S., Eum, S., Sanjeev, S., Khan, T., Stramer, T., Wang,
- T., Xin, T., Gogineni, T., Christianson, T., Sanders, T.,

Patwardhan, T., Degry, T., Shadwell, T., Fu, T., Gao, T., Garipov, T., Sriskandarajah, T., Sherbakov, T., Kaftan, T., Hiratsuka, T., Wang, T., Song, T., Zhao, T., Peterson, T., Kharitonov, V., Chernova, V., Kosaraju, V., Kuo,

- V., Pong, V., Verma, V., Petrov, V., Jiang, W., Zhang,
- W., Zhou, W., Xie, W., Zhan, W., McCabe, W., DePue, W., Ellsworth, W., Bain, W., Thompson, W., Chen, X., Qi, X., Xiang, X., Shi, X., Dubois, Y., Yu, Y., Khakbaz, Y., Wu, Y., Qian, Y., Lee, Y. T., Chen, Y., Zhang, Y., Xiong, Y., Tian, Y., Cha, Y., Bai, Y., Yang, Y., Yuan, Y., Li, Y., Zhang, Y., Yang, Y., Jin, Y., Jiang, Y., Wang, Y., Wang, Y., Liu, Y., Stubenvoll, Z., Dou, Z., Wu, Z., and Wang, Z. Openai gpt-5 system card, 2025. URL https://arxiv.org/abs/2601.03267.

Taufeeque, M., Quirke, P., Li, M., Cundy, C., Tucker, A. D., Gleave, A., and Garriga-Alonso, A. Planning in a recurrent neural network that plays sokoban, 2025. URL https://arxiv.org/abs/2407.15421.

Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., Chen, Z., Cui, J., Ding, H., Dong, M., Du, A., Du, C., Du, D., Du, Y., Fan, Y., Feng, Y., Fu, K., Gao, B., Gao, H., Gao, P., Gao, T., Gu, X., Guan, L., Guo, H., Guo, J., Hu, H., Hao, X., He, T., He, W., He, W., Hong, C., Hu, Y., Hu, Z., Huang, W., Huang, Z., Huang, Z., Jiang, T., Jiang, Z., Jin, X., Kang, Y., Lai, G., Li, C., Li, F., Li, H., Li, M., Li, W., Li, Y., Li, Y., Li, Z., Li, Z., Lin, H., Lin, X., Lin, Z., Liu, C., Liu, C., Liu, H., Liu, J., Liu, J., Liu, L., Liu, S., Liu, T. Y., Liu, T., Liu, W., Liu, Y., Liu, Y., Liu, Y., Liu, Y., Liu, Z., Lu, E., Lu, L., Ma, S., Ma, X., Ma, Y., Mao, S., Mei, J., Men, X., Miao, Y., Pan, S., Peng, Y., Qin, R., Qu, B., Shang, Z., Shi, L., Shi, S., Song, F., Su, J., Su, Z., Sun, X., Sung, F., Tang, H., Tao, J., Teng, Q., Wang, C., Wang, D., Wang, F., Wang, H., Wang, J., Wang, J., Wang, J., Wang, S., Wang, S., Wang, Y., Wang, Y., Wang, Y., Wang, Y., Wang,

- Y., Wang, Z., Wang, Z., Wang, Z., Wei, C., Wei, Q., Wu,

- W., Wu, X., Wu, Y., Xiao, C., Xie, X., Xiong, W., Xu, B., Xu, J., Xu, J., Xu, L. H., Xu, L., Xu, S., Xu, W., Xu,
- X., Xu, Y., Xu, Z., Yan, J., Yan, Y., Yang, X., Yang, Y., Yang, Z., Yang, Z., Yang, Z., Yao, H., Yao, X., Ye, W., Ye,

Z., Yin, B., Yu, L., Yuan, E., Yuan, H., Yuan, M., Zhan, H., Zhang, D., Zhang, H., Zhang, W., Zhang, X., Zhang,

- Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang, Y., Zhang,
- Y., Zhang, Z., Zhao, H., Zhao, Y., Zheng, H., Zheng, S., Zhou, J., Zhou, X., Zhou, Z., Zhu, Z., Zhuang, W., and Zu, X. Kimi k2: Open agentic intelligence, 2025. URL https://arxiv.org/abs/2507.20534.

Team, T. H. F. M. Hunyuanvideo 1.5 technical report, 2025. URL https://arxiv.org/abs/2511.18870.

Team, V., Hong, W., Yu, W., Gu, X., Wang, G., Gan, G., Tang, H., Cheng, J., Qi, J., Ji, J., Pan, L., Duan, S., Wang, W., Wang, Y., Cheng, Y., He, Z., Su, Z., Yang, Z., Pan, Z.,

Zeng, A., Wang, B., Chen, B., Shi, B., Pang, C., Zhang, C., Yin, D., Yang, F., Chen, G., Li, H., Zhu, J., Chen, J., Xu, J., Xu, J., Chen, J., Lin, J., Chen, J., Wang, J., Chen, J., Lei, L., Gong, L., Pan, L., Liu, M., Xu, M., Zhang, M., Zheng, Q., Lyu, R., Tu, S., Yang, S., Meng, S., Zhong, S., Huang, S., Zhao, S., Xue, S., Zhang, T., Luo, T., Hao, T., Tong, T., Jia, W., Li, W., Liu, X., Zhang,

- X., Lyu, X., Zhang, X., Fan, X., Huang, X., Xue, Y., Wang, Y., Wang, Y., Wang, Y., An, Y., Du, Y., Huang,
- Y., Niu, Y., Shi, Y., Wang, Y., Wang, Y., Yue, Y., Li, Y., Liu, Y., Zhang, Y., Wang, Y., Zhang, Y., Xue, Z., Du, Z., Hou, Z., Wang, Z., Zhang, P., Liu, D., Xu, B., Li, J., Huang, M., Dong, Y., and Tang, J. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2026. URL https://arxiv.org/abs/2507.01006.

Tung, F., Ding, M., Chen, Z., Bear, D. M., Gan, C., Tenenbaum, J. B., Yamins, D. L. K., Fan, J., and Smith, K. A. Physion++: Evaluating physical scene understanding that requires online inference of different physical properties. arXiv, 2023.

Unity Technologies. Unity, 2023. URL https://unity. com/. Game development platform.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.-F., and Liu, Z. Wan: Open and advanced large-scale video generative models, 2025. URL https:

//arxiv.org/abs/2503.20314.

Wang, T., Mao, X., Zhu, C., Xu, R., Lyu, R., Li, P., Chen, X., Zhang, W., Chen, K., Xue, T., Liu, X., Lu, C., Lin, D., and Pang, J. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In Conference on Computer Vision and Pattern Recognition, 2024a.

- Wang, X., Ma, W., Wang, A., Chen, S., Kortylewski, A., and Yuille, A. Compositional 4d dynamic scenes understanding with physics priors for video question answering. arXiv, 2024b.
- Wang, Y., Wang, Y., Mao, J., Niu, L., Zhi, W., Zhai, Z.X., Zhang, Z., Wang, C., He, Z., Li, L.-J., Chang, K.W., Zhu, S.-C., and Zhu, Y. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. In International Conference on

Learning Representations (ICLR), 2025. URL https: //openreview.net/forum?id=Q6a9W6kzv5.

Wang, Z., Zhang, J., Ge, J., Lian, L., Fu, L., Dunlap, L., Goldberg, K., Wang, X., Stoica, I., Chan, D. M., Min, S., and Gonzalez, J. E. Visgym: Diverse, customizable, scalable environments for multimodal agents. arXiv preprint arXiv:2601.16973, 2026. URL https: //arxiv.org/abs/2601.16973.

Wiedemer, T., Li, Y., Vicol, P., Gu, S. S., Matarese, N., Swersky, K., Kim, B., Jaini, P., and Geirhos, R. Video models are zero-shot learners and reasoners, 2025. URL https://arxiv.org/abs/2509.20328.

Xu, G., Jin, P., Hao, L., Song, Y., Sun, L., and Yuan, L. Llava-cot: Let vision language models reason step-bystep, 2024. URL https://arxiv. org/abs/2411.10440.

Yang, C., Wan, H., Peng, Y., Cheng, X., Yu, Z., Zhang, J., Yu, J., Yu, X., Zheng, X., Zhou, D., and Wu, C. Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks, 2025. URL https:

//arxiv.org/abs/2511.15065.

Yi, K., Gan, C., Li, Y., Kohli, P., Wu, J., Torralba, A., and Tenenbaum, J. B. Clevrer: Collision events for video representation and reasoning. arXiv, 2019.

Zhao, Q., Lu, Y., Kim, M. J., Fu, Z., Zhang, Z., Wu, Y., Li, Z., Ma, Q., Han, S., Finn, C., Handa, A., Liu, M.-Y., Xiang, D., Wetzstein, G., and Lin, T.-Y. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1702–1713, 2025. URL https://api.semanticscholar.

org/CorpusID:277435005.

Zheng, Z., Yan, X., Chen, Z., Wang, J., Lim, Q. Z. E., Tenenbaum, J. B., and Gan, C. Contphy: Continuum physical concept learning and reasoning from videos. In International Conference on Machine Learning. PMLR, 2024.

Model Specs Release Date Context (tokens) Model Size Closed-source Models

Models (abbr.)

GPT5.2 2025-12-11 400,000 – o3 (optional) 2025-04-16 200,000 – Opus4.5 (optional) 2025-11-24 200,000 – Sonnet4.5 2025-09-29 1,000,000 – Gemini3-Pro (preview) 2025-11-18 1,048,576 – Gemini3-Flash (preview) 2025-12-17 1,000,000 – Seed1.6-Flash (optional) 2025-12-23 262,144 – Seed1.6 2025-12-23 262,144 –

Open-source Models

Qwen3VL-235B (A22B, think) 2025-09-23 262,144 235B (A22B) Qwen3VL-30B (A3B, think) 2025-10-06 131,072 30B (A3B) Qwen3VL-8B (think) 2025-10-14 256,000 8B Qwen3VL-2B (think) 2025-10-21 256,000 2B GLM4.6V (optional) 2025-12-08 131,072 106B (A12B) MiMo-v2-Flash (optional) 2025-12-14 262,144 309B (A15B) Kimi-VL-A3B (think, opt.) 2025-04-10 131,072 – (A2.8B)

Table 5. Model release time, context window, and parameter scale for the evaluated model pool. For MoE models, AxxB denotes active parameters per forward pass when available. Unknown values are marked as --.

### A. Appendix

#### A.1. Limitations

We acknowledge two main limitations of the current CHAIN benchmark. (1) Limited scale of interactive environments, especially for interlocking puzzles. Our benchmark currently contains a finite set of Puzzle and Stacking instances (32 and 77, respectively). While stacking instances are programmatically generated and can be scaled to near-unlimited complexity, high-fidelity interlocking puzzles require careful environment engineering to faithfully capture kinematic feasibility and contact-rich constraints, and we implement them in Unity for precise control of interlocking mechanics. As a result, each new puzzle environment incurs substantial manual modeling and debugging effort, leading to significant time and cost overhead, which currently limits benchmark scale.

(2) Current evaluation mainly reports Pass@1 due to high multi-step interaction cost. We primarily report Pass@1 (single-attempt success) as our main success metric. This choice is partly driven by the evaluation cost of closed-loop interaction: each episode involves a non-trivial number of rounds, and we cap the interaction budget at 30–60 steps per instance. We recognize that interactive tasks can exhibit run-to-run variability; however, our preliminary analysis in Section 3.6 suggests that multi-sample evaluation (e.g., Avg@4) and Pass@1 show consistent trends, indicating that the variance is not dominant in practice. Looking forward, once sufficient API budget is available, we will report more robust best-of-K results (e.g., Pass@4) following the standard multi-run protocol.

#### A.2. Model list

Evaluated Models. We evaluate a broad pool of state-of-the-art multimodal LLMs, covering both closed-source APIs and open-source checkpoints. Specifically, our closed-source set includes GPT5.2, o3, Opus4.5, Sonnet4.5, Gemini3-Pro (preview), Gemini3-Flash (preview), and Seed1.6 (with Seed1.6-Flash as an optional lightweight variant). Our open-source set consists of Qwen3VL-235B (A22B, think), Qwen3VL-30B (A3B, think), Qwen3VL-8B (think), Qwen3VL-2B (think), GLM4.6V, MiMo-v2-Flash, and Kimi-VL-A3B (think). Table 5 summarizes each model’s release date, maximum context window, and parameter scale (when disclosed). For MoE models, we additionally report the active parameter count per forward pass in the form of AxxB. Models whose parameter sizes are not publicly released are marked as --.

#### A.3. Task Examples

This section showcases some cases of puzzles and stacking. As shown in Fig. 5, we divide the puzzle task into three difficulty levels according to (i) the number of pieces and (ii) the complexity of interlocks:

- • Easy Level: the object consists of a small number of pieces with simple interfaces and weak interlocking. Each piece has a distinctive placement, and the disassembly can be completed with few steps and low risk of dead-ends.
- • Middle Level: the object contains more pieces and more complex contact surfaces. Interlocking features become more frequent, requiring more careful selection of the moving direction and intermediate ordering.
- • Hard Level: the object consists of many pieces with strong mutual interlocks (e.g., multiple pieces jointly constrain removal), often necessitating long-horizon planning and reasoning over constraint propagation across the entire disassembly.

As shown in Fig. 6, we categorize stacking difficulty primarily by the container size and the number and shape diversity of pieces, which jointly determine the combinatorial search space:

- • Easy Level: a small container (e.g., 2×2×3) with relatively few pieces and limited shape complexity. Feasible placements are constrained and the solution space is small.
- • Middle Level: a medium container (e.g., 2×3×3) with increased spatial freedom and more placement permutations. The agent must reason more carefully about partial fills to avoid creating unreachable cavities.
- • Hard Level: a large container (e.g., 3×3×4) and a larger set of pieces. This setting introduces substantial ambiguity in placement order and orientation, making the task highly combinatorial and sensitive to early decisions.

#### A.4. Case Analysis of Stacking Task

Fig. 7 and Fig. 8 demonstrate a successful and a failed trajectory of the stacking task, respectively. In the successful case (Fig. 7), the agent follows a structure-first strategy. It starts by placing compact blocks to form a stable, flat base (Steps 1–2), and then stacks the upper layers layer by layer (Step 3). After the base is expanded (Step 4), subsequent placements (Steps 5–7) are used to systematically fill the remaining volume without creating unsupported overhangs or isolated cavities. This ordering preserves sufficient free space for later insertions and keeps the target shape “fillable” at every intermediate state, enabling the final completion.

In contrast, the failure case (Fig. 8) is mainly caused by an early irreversible commitment. The agent places a tall vertical column at the beginning (Step 1), before establishing a complete foundation. This choice constrains where later pieces can go, forcing the following placements to wrap around the column (Steps 2–5). Even though the agent attempts to roll back some actions and try new placements (Steps 6–8), the partially built structure encloses awkward residual space and reduces the clearance needed to insert the remaining piece. Although each intermediate placement is locally valid, the resulting configuration becomes a dead-end: the remaining piece cannot fit due to geometric mismatch, so the episode ends incomplete.

#### A.5. Polycube Stacking Puzzle Generation Pipeline

This appendix describes how we generate stable, solvable, and de-duplicated 3D polycube stacking/assembly puzzles inside an a × b × c voxel box. The generator follows a sample–verify design: we first sample candidate partitions into connected pieces, then apply a sequence of hard validity checks (exact-cover solvability, structural constraints, and (optional) linear assembly feasibility). We also enforce strong de-duplication by canonicalizing each piece shape up to rigid rotations and hashing the resulting multiset.

- A.5.1. OBJECTIVES AND CORE CONSTRAINTS Given a box B = {0,...,a−1} × {0,...,b−1} × {0,...,c−1}, we generate a set of pieces P = {P1,...,PK} such that:

- 1. Exact cover (solvable packing). There exists a placement of all pieces (allowing 24 rigid rotations and translations within the box) that covers each voxel in B exactly once.

[Figure 5]

[Figure 6]

[Figure 7]

(a) Easy Level (b) Middle Level (c) Hard Level

- Figure 5. Examples of different levels of puzzle task.

[Figure 8]

(a) Easy Level (b) Middle Level (c) Hard Level

[Figure 9]

[Figure 10]

- Figure 6. Examples of different levels of stacking task.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

(a) Step 1 (b) Step 2 (c) Step 3 (d) Step 4

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

(e) Step 5 (f) Step 6 (g) Step 7 (h) Step 8 Figure 7. A case of successfully solving the stacking task.

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

(a) Step 1 (b) Step 2 (c) Step 3 (d) Step 4

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

(e) Step 5 (f) Step 6 (g) Step 7 (h) Step 8 Figure 8. A case of failed to solve the stacking task.

- 2. Connectivity. Each piece is 6-neighbor connected.
- 3. Piece size bounds. Each piece satisfies |Pi| ∈ [min piece,max piece cells].

- 4. Structural rule: no 2 × 3 full face. No piece contains a filled 2 × 3 rectangle in any axis-aligned plane (XY/YZ/XZ), which avoids large flat plates that often reduce interaction richness.
- 5. No isolated piece. In the final packed solution, each piece must touch at least one other piece via 6-neighbor adjacency (preventing “floating” blocks).
- 6. (Optional) Linear assembly. If enabled, the packed solution must admit a collision-free linear disassembly/assembly sequence where each step removes one piece by translating it along one of {±x,±y,±z} without intersecting remaining pieces.

- A.5.2. CANDIDATE PARTITION SAMPLING BY DIFFICULTY We support three difficulty modes by varying how candidate pieces are sampled before verification:

Easy (regular cuboids). We sample a multiset of axis-aligned cuboids whose volumes sum to abc and whose dimensions satisfy an “easy-shape” rule: at least two dimensions are equal (e.g., 1×1×2, 1×2×2, 2×2×1). We then verify whether these cuboids can be packed (with rotations) to exactly fill the box via the same exact-cover solver as other modes.

Mid (flat connected growth). We grow connected pieces by seeded BFS-style expansion, but constrain each piece to a plane by fixing one coordinate (e.g., x = x0). This produces “sheet-like” structures while remaining connected and bounded in size.

Hard (free connected growth + uniqueness). We grow connected pieces without planar constraints and further enforce that all pieces have distinct shapes up to rigid rotations. If duplicates occur, we attempt to improve uniqueness via limited voxel transfers between adjacent pieces while maintaining connectivity; otherwise we reject the sample. We also use solver search effort (visited nodes) as a difficulty proxy and enforce a mode-specific threshold.

- A.5.3. EXACT-COVER SOLVABILITY VIA DLX

We validate packability by reducing the problem to an exact cover instance. Each candidate placement of piece i corresponds to a row that covers: (i) the voxels occupied by the placement, and (ii) a “use piece i” column that enforces each piece is used exactly once. We solve the resulting sparse exact-cover matrix using Algorithm X with dancing links (DLX), augmented

- Algorithm 1 Linear Assembly Sequence Extraction (via Greedy Disassembly)

Require: Placed pieces {Ci}Ki=1, where Ci ⊆ B is the set of voxels occupied by piece i Ensure: Assembly order π and removal directions {dt}, or FAIL

- 1: R ← {1,2,...,K} ▷ Indices of remaining pieces
- 2: removal ← [] ▷ Disassembly sequence as a list of (i,d)
- 3: while R ̸= ∅ do
- 4: U ← j∈R Cj ▷ All voxels occupied by remaining pieces
- 5: found ← false
- 6: for i ∈ R do
- 7: O ← U \ Ci ▷ Voxels occupied by other pieces
- 8: for d ∈ {+⃗x,−⃗x,+⃗y,−⃗y,+z⃗,−z⃗} do
- 9: if REMOVABLE(Ci,O,d) then
- 10: Append (i,d) to removal
- 11: R ← R \ {i}
- 12: found ← true
- 13: break
- 14: end if
- 15: end for
- 16: if found then
- 17: break
- 18: end if
- 19: end for
- 20: if not found then
- 21: return FAIL
- 22: end if
- 23: end while
- 24: π ← REVERSE(removal)
- 25: return π

with: (1) caching all placements per canonical shape and box size, (2) an anchor piece (the piece with fewest placements) constrained to cover the origin voxel to reduce symmetries, and (3) an MRV-style column choice heuristic.

- A.5.4. PHYSICAL FEASIBILITY AND LINEAR ASSEMBLY EXTRACTION

After DLX returns a packed solution (a concrete set of voxels for each placed piece), we apply two physical-feasibility checks.

No isolated piece. We reject solutions where any piece has no 6-neighbor contact with other pieces. This avoids degenerate instances that are technically solvable but physically uninformative for assembly reasoning.

Linear disassembly/assembly sequence. If linear assembly is required, we attempt to find a sequence in which pieces can be removed one-by-one by a pure translation along one of the six axis directions, without colliding with remaining pieces. A piece P is removable along direction d if translating each voxel of P by one unit along d does not enter any voxel occupied by other pieces that are still present (within box bounds). We greedily remove any currently removable piece; if we can remove all pieces, reversing this removal order yields a valid assembly sequence.

- Algorithm 1 gives the extraction procedure used in our pipeline.

The predicate REMOVABLE checks whether translating Ci by one voxel step along d causes any voxel of the translated set to intersect O (within bounds). If no direction works for any remaining piece at some iteration, the instance does not admit a linear assembly and is rejected when linear assembly is required.

- Algorithm 2 Batch Generation with Staged Sampling, Verification and De-duplication

Require: Set of box sizes D, Per-size configuration (max pieces, difficulty thresholds), Target number of puzzles per size/mode T

Ensure: A dataset of accepted puzzles,

Summary metadata (counts, rejection reasons, generation statistics)

- 1: for all (a,b,c) ∈ D do
- 2: for each mode ∈ {easy,mid,hard} do
- 3: for t = 1 to T do
- 4: gen ← GENERATEONESTAGED(a,b,c,mode,cfg)
- 5: if gen = Fail then
- 6: continue
- 7: end if
- 8: s ← PUZZLESIGNATURE(gen.pieces)
- 9: if ¬ RESERVESIGNATUREATOMICALLY(s) then
- 10: continue
- 11: end if
- 12: SAVEPUZZLE(gen,s) ▷ Store as JSON + visual renders in per-puzzle folder
- 13: end for
- 14: end for
- 15: end for
- 16: return (dataset,metadata)

- A.5.5. STRONG DE-DUPLICATION BY SHAPE MULTISET CANONICALIZATION

To avoid repeated puzzle templates, we de-duplicate instances by canonicalizing each piece shape up to rigid rotation. For each piece Pi, we: (i) translate it to a normalized origin (minimum coordinate at zero), then (ii) enumerate its 24 rotation-equivalent shapes, and (iii) select the lexicographically smallest voxel list as its canonical representation. We then sort the list of canonical piece representations to form a multiset signature for the puzzle and hash it (e.g., SHA1). In multi-processing, we reserve signatures with an atomic marker file to ensure thread-safe de-duplication.

- A.5.6. END-TO-END BATCH GENERATION AND OUTPUTS

For each box size (a,b,c) and difficulty mode (easy/mid/hard), we generate a fixed number of puzzles by repeatedly invoking the staged sampler with verification and de-duplication. Each accepted puzzle is stored in its own folder with: (i) a JSON manifest (pieces, packed solution, assembly order, difficulty stats, signature), and (ii) rendered 3D voxel visualizations of the full assembly and individual pieces.

- Algorithm 2 summarizes the full pipeline.

Staged min-piece fallback. GenerateOneStaged implements a staged attempt schedule: it first searches with a stricter minimum piece size (e.g., min piece = 4) and, if unsuccessful within budget, relaxes to min piece = 3 to improve yield for small volumes or tight constraints. Each sampled partition is accepted only if it passes the full chain of checks: (i) structural rules, (ii) DLX solvability, (iii) no isolated piece, (iv) (optional) linear assembly, and (v) difficulty threshold (hard).

- A.6. Prompt details Fig. 9 shows the content of our world model prompt.

Figure 9. World-model prompt for generating physically valid disassembly videos of Kongming/Luban locks (interlocking burr puzzles). Constraints include rigidity, collision avoidance, continuous motion, and axis-aligned sliding.

Generate a video showing the disassembly of a Kongming/Luban lock (interlocking burr puzzle) from the exact configuration in the provided reference image. The video must be physically valid and obey these constraints:

- • All parts are rigid wooden blocks (no bending or deformation).
- • No interpenetration: contacts and collisions are respected; parts never pass through each other.
- • No teleportation: motion is continuous over time.
- • Each piece moves only by axis-aligned sliding along its allowed rail; no rotation until it is fully freed.
- • Use a realistic unlocking sequence: remove the key piece first, then remove any newly unblocked pieces until fully separated.
- • Match the reference geometry and part count exactly; do not add, remove, resize, or alter pieces.

