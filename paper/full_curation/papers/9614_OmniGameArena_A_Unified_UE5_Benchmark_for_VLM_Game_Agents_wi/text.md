# arXiv:2606.09826v1[cs.CV]8Jun2026

## OmniGameArena: A Unified UE5 Benchmark for VLM Game Agents with Improvement Dynamics

##### Mingxian Lin1, Shengju Qian2, ‡, Yuqi Liu3, Yi-Hua Huang1, Yiyu Wang2, Wei Huang1, Yitang Li4, Fan Zhang3, Zeyu Hu2, Lingting Zhu2, Xin Wang2, Xiaojuan Qi1, †

1The University of Hong Kong, 2LIGHTSPEED, 3The Chinese University of Hong Kong, 4Tsinghua University

‡Project Leader †Corresponding Author Project Page: https://mxlin043.github.io/OmniGameArena/

##### Abstract

report a single first-attempt score per (agent, game) pair, leaving invisible the trajectory by which an agent improves under repeated interaction with the same task. They also lean heavily toward singleagent Solo play, while adversarial (PvP) and cooperative (Coop) regimes remain underrepresented even though they probe distinct capabilities such as opponent modeling, role assignment, and recovery from a teammate’s mistakes. Whether an agent can adapt under repeated reflection, and whether it can do so in adversarial or cooperative settings, therefore remains largely unmeasured.

Vision-language model (VLM) agents are increasingly deployed in interactive game environments. Yet game benchmarks for VLM agents typically report a single first-attempt score per (agent, game) pair, focus on singleagent Solo play, and lack unified protocols for evaluating heterogeneous agent classes (commercial VLMs, open-weight VLMs, and specialized game policies) on the same footing. We address these gaps with OmniGameArena, a real-time benchmark of twelve newly built Unreal Engine 5 games spanning Solo (7), PvP (3), and Coop (2) with unified action interfaces, and the Improvement Dynamics Curve (IDC), an agentic-reflection harness in which a tool-using reflector LLM autonomously refines a bounded skill prompt across multiple rounds. Beyond cold-start leaderboard scores, IDC exposes two additional observables for each (agent, game) pair: how the score evolves across reflection rounds, and how the learned skill behaves on held-out task variants. We report these observables for twelve VLM agents on the cold-start leaderboard and four top agents under IDC.

We address both with OmniGameArena, a realtime benchmark of twelve newly built Unreal Engine 5 games spanning Solo, PvP and Coop, and the Improvement Dynamics Curve (IDC), an agenticreflection harness built on top of it. The twelve games are authored for this benchmark rather than reused from public titles, lowering the risk of pretraining leakage, and share unified action interfaces (keyboard-mouse, gamepad) so that commercial VLMs, open-weight VLMs, and specialized game policies can all be evaluated under matched environment conditions. The IDC harness runs each (agent, game) instance for multiple rounds: the agent plays K episodes under a current skill prompt, after which a reflector LLM inspects the trajectories through tool-use, deciding on its own what to read and when to stop, before refining the skill for the next round. We report both the perround score sequence (the IDC of that instance) and a transfer score on held-out task variants.

##### 1 Introduction

Foundation models are increasingly evaluated by how they act, not only by what they answer, and games are a natural stress test for this shift (Wang et al., 2023; Tan et al., 2024; Paglieri et al., 2024): an agent must read a changing visual scene, choose actions under time pressure, plan across delayed rewards, and adapt when the environment resists. Game benchmarks now span text-only worlds, 2D grid suites, and 3D open environments built on existing commercial titles, and have driven rapid progress in vision-language game agents (Tan et al., 2025; Magne et al., 2026; Wang et al., 2025b).

Across twelve agents on the cold-start leaderboard, no single VLM dominates, and commercial agents hold a wide gap over open-weight VLMs and specialized policies. Among the four top agents that we run through IDC, all four improve over their cold-start baseline through reflection, yet peak performance is typically reached mid-curve rather than at the final round. Most notably, origintask improvement and held-out variant transfer can

Yet current benchmarks rarely measure two properties that matter for deploying these agents. Most

### Omni Game Arena

12 UE5 Games

###### Solo PvP Coop

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

……

Obstacle Run Monster Shooting Scene Escape Last Stand Sky Duel Crystal Guard Midline Clash Shared-floor Delivery Handoff Cooperation

Real-time Harness

###### Real-time Execution (Loop)

###### Observation Agent

###### Adapters

###### UE5 Env Score / Logs

time

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

...

Commercial VLMs

Prompt -> Keys

UE5 Env

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Inference (Agent)

[Figure 15]

Leaderboard

Open-source VLMs

Prompt -> Pad

Action Chunk (Adapter)

Keyboard-Mouse Policies

Keys -> Keys

Trajectory Logs

Control Applied (Env)

Gamepad Policies

Pad -> Pad

Game world keeps ticking

What We Report

[Figure 16]

[Figure 17]

- Figure 1: OmniGameArena at a glance. Twelve newly built UE5 games span Solo (7), PvP (3), and Coop (2) regimes (top). Heterogeneous agents (commercial VLMs, open-weight VLMs, keyboard-mouse policies, and gamepad policies) connect to the same real-time UE5 environment through documented adapters (middle). Evaluation reports the cold-start leaderboard and the Improvement Dynamics Curve (IDC) under multi-round reflection (bottom).

diverge in our experiments; this divergence is hidden by single-round leaderboard scores and is a central observable IDC exposes.

To summarize, our contributions are threefold: (i) OmniGameArena, a twelve-game UE5 benchmark spanning Solo, PvP, and Coop with unified action interfaces and game instances built specifically for this benchmark; (ii) the IDC harness, an agentic-reflection framework whose autonomous tool-use reflector refines a bounded skill prompt across R rounds, with persistent memory and best-skill rollback; and (iii) an empirical study across twelve agents showing that leadership rotates across games and that origin-task gain does not by itself predict held-out variant transfer.

##### 2 Related Work

Benchmarks in game environments. Interactive games have served as AI testbeds since the rise of reinforcement learning and now anchor evaluations of Large Language Models (LLMs) and VisionLanguage Models (VLMs). Early LLM evaluations were text-only (Huang et al., 2024; Wu et al., 2023; Hu et al., 2024), effective for logical reasoning but lacking visual grounding. 2D-grid suites such as BALROG (Paglieri et al., 2024) and LVLMPlayground (Wang et al., 2025a) added spatial and

multimodal demands, and more recent suites built on Minecraft or general visual benchmarks (VMAGE (Zheng et al., 2025), Cradle (Tan et al., 2024), VideoGameBench (Zhang et al., 2025)) extended evaluation into 3D open worlds with longhorizon planning from pixels. Beyond game playing, related benchmarks probe embodied reasoning and action in complex 3D environments (Lin et al.,

- 2025; Zhu et al., 2026) and unified reasoning across video-generation models (Luo et al., 2025), but neither targets multi-regime, real-time game interaction. Two limitations of these benchmarks motivate our work: most reuse existing commercial titles, leaving them exposed to pre-training contamination; and few cover Solo, PvP, and Coop regimes in a single real-time environment. OmniGameArena addresses both with twelve newly built UE5 games that span all three interaction regimes. Game-playing LLM and VLM agents. Early LLM game agents operated in text-only environments (Hausknecht et al., 2020; Tsai et al., 2023) and 2D grid worlds (Feng et al., 2023; Küttler et al., 2020). Voyager (Wang et al., 2023) and MineDojo (Fan et al., 2022) extended LLM agents to 3D Minecraft, but the heavy per-game engineering they require limits cross-game generality. The current VLM-agent generation (Li et al., 2025; Bai et al.,
- 2026; Wang et al., 2025b; Tan et al., 2025; Magne

- Figure 2: Radar charts of the 12 OmniGameArena games across seven capability dimensions. The abbreviations are: VP = Visual Perception, SN = Spatial Navigation, RT = Reaction, MEM = Memory, PLN = Planning, ADV = Adversarial, and COOP = Cooperation. Each dimension is scored from 0 to 3.

et al., 2026) controls GUI or keyboard-mouse interfaces directly across diverse 3D worlds: GameTARS (Wang et al., 2025b) is pre-trained on over 500B tokens of multimodal gameplay; NitroGen (Magne et al., 2026) on 40,000 hours of gameplay video across 1,000+ titles; and Lumine (Tan et al.,

- 2025) executes hours-long real-time missions in

3D environments. These agents motivate, but also outpace, the standardized cross-game evaluation infrastructure we provide.

Reflection and Self-improvement. Most gameagent benchmarks report a single-shot score, obscuring whether and how fast an agent improves with repeated interaction. Reflection-based methods address this without weight updates by accumulating natural-language summaries of past experience. Reflexion (Shinn et al., 2023) converts episode feedback into verbal self-critiques; SelfRefine (Madaan et al., 2023) iteratively rewrites model outputs; ExpeL (Zhao et al., 2024) extracts task-level insights; Voyager’s skill library (Wang et al., 2023) accumulates reusable code skills inside Minecraft; and GameVerse (Zhang et al., 2026) reports a single with-vs-without reflection comparison per game. Our work aligns with the recently articulated heuristic learning paradigm (Weng,

- 2026), which views LLM-driven self-improvement

- as a learning process operating on explicit artifacts (prompts, code, memory) rather than weights. Our IDC harness instantiates this paradigm in three concrete ways: (i) reflection runs for multiple rounds, producing a full score trajectory rather than a before-after comparison; (ii) the reflector is itself an LLM with tool-use that decides what to inspect rather than executing a fixed-template script; and

(iii) we test the resulting skill on held-out task variants per game, revealing differences in skill style that single-number metrics hide.

##### 3 OmniGameArena

We introduce OmniGameArena, a suite of twelve custom Unreal Engine (UE5) games spanning Solo, PvP, and Coop regimes (Section 3.1) to systematically evaluate distinct capability axes of vision-based game agents using robust, continuous progress metrics. Furthermore, to address the critical challenge of evaluation contamination, we detail proactive data avoidance strategies and rigorous empirical analysis (Section 3.2) to ensure the integrity and novelty of our benchmark.

###### 3.1 Game Suite and Evaluation Metrics

The OmniGameArena suite comprises twelve visually rich and physically complex environments developed in Unreal Engine 5. Each game is purposefully designed to isolate and test specific subsets of embodied capabilities, ranging from solitary spatial reasoning to complex multi-agent cooperation, as shown in Figure 2. Game progress is uniformly normalized to a continuous scale of [0,1], providing a consistent metric across highly diverse tasks. We provide a brief description and its evaluation protocol for each game, as demonstrated in Table 1.

###### 3.2 Contamination Avoidance

We adopt a proactive approach to mitigate data contamination during the benchmark design phase. We begin by conducting a web-exposure audit to search for exact game names, task phrases, rule descriptions, and scoring events prior to release, ensuring

Game Description Evaluation Solo ObstacleRun3D A 3D parkour game where the agent navigates to a finish

xa−xstart xfinish−xstart , where xa: agent pos., xstart: start, xfinish: target

line while avoiding physical obstacles.

xa−xstart xfinish−xstart , where xa: agent pos., xstart: start, xfinish: target

ObstacleRun2D A 2D side-scrolling platformer where the agent must reach the end of a linear level.

tsurvive

LastStand A platform survival game where the agent must avoid hazards and falling off.

Tmax , where tsurvive: time survived, Tmax: max duration

De Htotal , where De: effective damage, Htotal: total enemy health

MonsterShoot A survival shooting game to locate and eliminate hostile entities while avoiding damage.

SceneEscape A scene-based puzzle game requiring the completion of NPC-assigned tasks to escape.

n N , where n: completed tasks, N: total tasks

CueChase A third-person exploration game to locate and activate hidden triggers across the map.

k K , where k: activated triggers, K: total triggers

vdelivered

SoloCraft A logistics game where the agent collects, prepares, and delivers items to fulfill orders.

Vtarget , where vdelivered: fulfilled value, Vtarget: target value

PvP SkyDuel A direct 1v1 combat game where the agent must engage

hself Hmax , where hself: agent remaining health, Hmax: agent max health

and defeat an opponent.

howncore Hmax , where howncore: own core health, Hmax: core max health

CrystalGuard A symmetric attack-and-defend game to destroy the opponent’s crystal core while protecting one’s own.

sa Starget , where sa: agent score, Starget: target score

MidlineClash A competitive logistics game where two agents race to fulfill resource orders in a shared environment.

Cooperation SharedFloor A symmetric cooperative game where agents share a

vdelivered

Vteam , where vdelivered: fulfilled value, Vteam: team target value

space and capabilities to fulfill delivery orders.

vdelivered

HandoffRun An asymmetric cooperative game requiring agents to pass items across restricted areas based on distinct roles.

Vteam , where vdelivered: fulfilled value, Vteam: team target value

Table 1: Summary of 12 Interactive Games.

Benchmark Games(#) Recog.(%)↓ Mech. (%)↓

BALROG 6 66.7 100.0 LMGame-Bench 6 100.0 100.0 ORAK 12 100.0 100.0 OmniGameArena 12 0.0 50.0

Table 2: Contamination analysis when given screenshots. Recog.: Percentage of games recognized; Mech.: Percentage of games where underlying mechanics were successfully described.

these specific elements are strictly excluded from the benchmark. To guarantee environment novelty, we construct entirely new games using Unreal Engine 5 (UE5). For visual content, we utilize a mixture of bespoke and off-the-shelf UE5 marketplace assets. However, the combination of assets, the design of the level geometry, the execution order of scripts, and the criteria for success are uniquely designed, ensuring the evaluation scenarios cannot be memorized from pre-training data.

Contamination Analysis. To empirically verify the effectiveness of our avoidance strategies, we conduct a contamination analysis focusing on visual novelty and rule leakage. First, we provide a representative model (e.g., Gemini) with screen-

shots of games to assess whether it can recognize the game name, confirming the visual novelty of the tasks. Second, we evaluate whether the model can successfully describe the underlying mechanics of the games based purely on visual inputs, which tests for the leakage of memorized game rules. As shown in Table 2, we compare existing benchmarks (Paglieri et al., 2024; Park et al., 2025; Hu et al., 2025) against our proposed OmniGameArena across these two tests. The results clearly demonstrate that the games within the existing benchmarks are highly recognizable, allowing the model to retrieve their mechanics directly from memory. In contrast, OmniGameArena exhibits a 0.0% recognition rate and significantly reduced mechanics leakage (50.0%). These results suggest that OmniGameArena substantially mitigates the risk of pre-training contamination, reducing the influence of memorized priors on agent evaluation.

##### 4 Game Agent Harness

OmniGameArena specifies the games; the harness specifies how an agent plays them. The harness has two layers: a per-episode loop (§4.1) that drives

###### Experience Acquisition Module

###### Reflection Module

Persistent Module

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Current Skills

K Episodes Explore: inspect / search / review trajectories

- 1

- 2

- 3

- 4

| |
|---|

###### Experience Notebook

[Figure 22]

[Figure 23]

Task insights and knowledge

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Diagnose: compare causes / explain cases

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Validated Skills Discovered from experience

[Figure 35]

#### ……

[Figure 36]

[Figure 37]

[Figure 38]

Validate: test / verify / confirm skills

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

###### Curve

[Figure 45]

Distill: summarize / refine / store knowledge

[Figure 46]

Historical performance

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Figure 3: Overview of the Improvement Dynamics Curve (IDC) harness. The experience acquisition module (left) runs K episodes under the current skill. The reflection module (center) reads both the new trajectories and the persistent state (notebook and prior skill), then runs four autonomous stages (Explore, Diagnose, Validate, Distill) to produce a refined skill. The persistent module (right) stores the experience notebook, validated skills, and per-round score curve, which seed the next round.

any agent during cold-start runs, and a reflective outer loop (§4.2) whose round-level scores form the Improvement Dynamics Curve studied in §5.3.

Let πθ denote a VLM with frozen weights θ. At round r the agent is conditioned on a skill prompt mr and acts according to at ∼ πθ(at | ot,Ht,mr); the skill mr is fixed throughout the round.

- 4.1 Per-Episode Loop OmniGameArena exposes a Gym-like interface. At step t the harness receives observation ot = (It,wt,ht,τt), where It is the RGB frame at resolution wt × ht and τt is its capture timestamp. The agent emits an action at (a chunked keyboardmouse action for VLMs), which the engine exe-

cutes before returning ot+1; the loop terminates on done.

Bounded visual-action history. For VLM agents, the harness maintains a sliding window of the last L observation-response pairs:

Ht = [(It−L,yt−L),...,(It−1,yt−1)],

where yi is the raw VLM response at step i. At step t, the VLM is prompted with system instructions, an optional skill prompt m, the history Ht, and the current frame It, for up to L + 1 images per call. The current frame is appended to history only after yt is returned.

- 4.2 Improvement Dynamics Curve

Experience acquisition. Each round contains K episodes, yielding trajectories τr = {τr,1,...,τr,K} and round score

K

1 K

Sr =

s(τr,k).

k=1

Round r = 0 uses an empty skill (m0 = ∅) and serves as the cold-start baseline.

Reflection. After round r, the reflector R refines the skill autonomously, reading the new trajectories together with the persistent state (notebook and prior skill) and deciding for itself which content to inspect, how many tool calls to spend, and when to terminate. The refinement proceeds in four stages. The Explore stage exposes the round’s per-episode trajectories through sandboxed readonly tools (list_dir, read_text, read_image, grep); the reflector chooses what to inspect rather than executing a fixed script. The Diagnose stage commits an explicit list of failure modes via submit_diagnosis, separating causes from prescriptions. The Validate stage proposes an updated skill and calls validate_skill, an independent LLM judge that rejects proposals which memorize map content, or contradict the diagnosis; the reflector iterates up to five times before committing. The Distill stage finalizes the accepted skill mr+1 and, optionally, edits the notebook with durable observations. The combination of agentic autonomy within a small, fixed tool surface (read, diagnose, judge, write) is what keeps the framework lightweight without sacrificing functional completeness.

IDC wraps the per-episode loop with a reflective outer loop organized into three modules (Figure 3): an experience acquisition module that runs K episodes under the current skill, a reflection module that converts the resulting trajectories into a refined skill, and a persistent module that carries state across rounds. We deliberately keep the loop lightweight yet functionally complete: a small fixed tool surface gives the reflector full agentic autonomy without prescribing an inspection script.

Agent ObstacleRun2D ObstacleRun3D LastStand MonsterShoot SceneEscape CueChase SoloCraft

Claude Opus 4.7 (Anthropic, 2026b) 0.220±0.218 0.094±0.040 0.308±0.146 0.400±0.087 0.460±0.152 0.380±0.192 0.160±0.028 Claude Opus 4.6 (Anthropic, 2026a) 0.338±0.003 0.172±0.094 0.147±0.044 0.362±0.079 0.540±0.134 0.840±0.102 0.228±0.023 Claude Sonnet 4.6 (Anthropic, 2026c) 0.075±0.002 0.200±0.138 0.144±0.050 0.166±0.048 0.440±0.207 0.440±0.215 0.124±0.033 GPT-5.5 (OpenAI, 2026b) 0.473±0.121 0.133±0.007 0.416±0.257 0.464±0.064 0.720±0.370 0.580±0.098 0.252±0.023 GPT-5.4 (OpenAI, 2026a) 0.122±0.070 0.089±0.037 0.148±0.070 0.326±0.090 0.680±0.045 0.300±0.063 0.084±0.017 Gemini 3.1 Pro Preview (Google, 2026b) 0.102±0.059 0.165±0.112 0.230±0.090 0.710±0.138 0.660±0.336 0.600±0.322 0.148±0.100 Gemini 3.1 Flash-Lite Preview (Google, 2026b) 0.278±0.086 0.097±0.044 0.122±0.030 0.182±0.083 0.300±0.158 0.440±0.080 0.036±0.022 Kimi K2.5 (Moonshot AI, 2026) 0.109±0.056 0.075±0.031 0.232±0.113 0.290±0.075 0.220±0.130 0.140±0.102 0.064±0.043

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Qwen3.5-397B-A17B (Qwen Team, 2026) 0.114±0.012 0.112±0.043 0.106±0.017 0.072±0.025 0.200±0.258 0.040±0.049 0.000±0.000 Qwen3.5-122B-A10B (Qwen Team, 2026) 0.092±0.028 0.034±0.032 0.093±0.019 0.024±0.019 0.060±0.080 0.040±0.049 0.013±0.023

[Figure 61]

NitroGen (gamepad) (Magne et al., 2026) 0.034±0.042 0.065±0.046 0.100±0.029 0.014±0.014 0.120±0.040 0.020±0.040 0.000±0.000 Open-P2P (kbd-mouse) (Yue et al., 2026) 0.056±0.040 0.023±0.045 0.063±0.034 0.000±0.000 0.100±0.082 0.000±0.000 0.000±0.000

[Figure 62]

[Figure 63]

Table 3: Solo cold-start scores. Top 3 per column: red > orange > yellow .

Persistent module. Three artifacts carry state across rounds. The experience notebook is a factual log written by the reflector for its own future use (e.g., “round r episode k step t: event”), capped

- at 2000 tokens, edited rather than appended, and never shown to the player. The validated skills

comprise the skill prompt mr used in the current round (player-visible, capped at 1200 tokens of cueto-response heuristics) together with the best-skill cache mr∗ used for rollback. The curve is the score sequence [S0,S1,...,Sr] accumulated so far.

Best-skill rollback. Reflection is not monotone. When a round’s score drops sharply below the best seen so far (Sr+1 < α · Sr∗, α = 0.5), the harness resets the next round’s starting prompt to mr∗ and resumes reflection from there, guarding against catastrophic skill drift.

The curve. Iterating for R rounds yields [S0,S1,...,SR], the Improvement Dynamics Curve of the (agent, game) pair. Two models with the same final SR can produce very different curves (early vs. late convergence, monotone vs. oscillating); the curve, rather than any single round, is the measurement object in §5.3.

##### 5 Experiments

We report results in three blocks. §5.1 fixes the evaluation protocol, the set of agents, and the scoring conventions used throughout. §5.2 reports the main cold-start leaderboard, in which every agent plays every game once with no prior trajectories and no provided skill, spanning the seven Solo, three PvP, and two Coop games. §5.3 relaxes the cold-start constraint via the Improvement Dynamics Curve (IDC), an agentic self-reflection protocol in which the same model alternates between playing the game and rewriting its own skill (a naturallanguage summary of game-specific strategy) for R rounds. We use IDC to characterize (i) how

each model improves on the original task, and (ii) whether the learned skill transfers to three held-out task variants per game.

###### 5.1 Experimental Settings

Agents. We evaluate twelve agents grouped into three classes so their strengths and limitations are not averaged away. (a) Commercial VLMs (API-only): three Claude models (Opus 4.7 (Anthropic, 2026b), Opus 4.6 (Anthropic, 2026a), Sonnet 4.6 (Anthropic, 2026c)), two OpenAI models (GPT-5.5 (OpenAI, 2026b) and GPT-5.4 (OpenAI, 2026a)), two Gemini models (3.1 Pro Preview and 3.1 Flash-Lite Preview (Google, 2026b)), and Kimi K2.5 (Moonshot AI, 2026). (b) Openweight VLMs: two Qwen3.5 mixture-of-experts checkpoints, 397B-A17B and 122B-A10B (Qwen Team, 2026), served locally behind an OpenAIcompatible endpoint. (c) Specialized game policies: NITROGEN (Magne et al., 2026), which consumes a single frame and emits a chunk of 18 gamepad actions (21-dim each), and OPENP2P (Yue et al., 2026), which consumes a 200frame keyboard-mouse history and emits one action per frame. Both run with the native real-time protocols from their original papers. All agents are evaluated on the same OmniGameArena real-time environment with matched configuration. VLMs use OmniGameArena’s chunked keyboard-mouse adapter, while the specialized policies are routed through their native interfaces unchanged, so each system is evaluated at the operating point it was designed for.

Evaluation protocol. Each (agent, game) cell is evaluated over N=5 episodes; PvP cells additionally cover every pairwise matchup in the agent pool. OmniGameArena natively runs in real time, but commercial VLM API calls suffer from network jitter that is orthogonal to model capability. We

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

- Figure 4: PvP win rates of Player 1 (row) against Player 2 (column) per game over all pairings.

therefore evaluate under two clock modes that both pause the environment during inference: Paused Decision Quality (PDQ) freezes the environment for the full inference call and treats decision time as free, isolating pure decision quality; LatencyControlled Real-Time (LCRT) additionally idles for the server-reported inference time before each action is applied, charging agents for on-device latency while excluding network round-trip noise. Main-table results use PDQ and LCRT is reported in Appendix A.

###### 5.2 Cold-start Leaderboard

Solo games. Table 3 reports the seven Solo games. Three patterns stand out. (1) No single model dominates. Leadership rotates across games: GPT-5.5 leads four of seven games (ObstacleRun2D, LastStand, SceneEscape, SoloCraft), Claude Opus 4.6 wins CueChase by a wide margin (0.840 vs. next 0.580), and Gemini 3.1 Pro leads MonsterShoot (0.710 vs. next 0.464). (2) Newer is not always better. Claude Opus 4.6 outperforms Opus 4.7 on five of seven Solo games, and GPT-5.4 exceeds GPT-5.5 on SceneEscape, indicating that capability ranking is task-specific rather than monotone in release order. (3) Open-weight VLMs and specialized policies fail to transfer. Both Qwen3.5 MoE checkpoints score below 0.15 on every game and exactly 0.00 on several, while NITROGEN (gamepad) and OPEN-P2P (keyboardmouse) collapse to near zero on all but a handful of games. This confirms that OmniGameArena’s task diversity lies well outside the training distribution of policies optimized for narrow single-game.

PvP games. Figure 4 reports Player 1 win rates over all pairings (diagonal omitted). SkyDuel and CrystalGuard show a clean dominance hierarchy that tracks the Solo leaderboard: GPT-5.5 and Gemini 3.1 Pro win against nearly every opponent, while both Qwen3.5 variants lose nearly every matchup. MidlineClash is non-transitive: Kimi K2.5 beats Claude Opus 4.6 in all five Player 1 matches (1.00) even though Claude is the

Agent SharedFloor HandoffRun

Claude Opus 4.7 (Anthropic, 2026b) 0.136±0.033 0.040±0.040 Claude Opus 4.6 (Anthropic, 2026a) 0.152±0.030 0.064±0.036 Claude Sonnet 4.6 (Anthropic, 2026c) 0.148±0.064 0.008±0.018 GPT-5.5 (OpenAI, 2026b) 0.368±0.036 0.184±0.022 GPT-5.4 (OpenAI, 2026a) 0.068±0.023 0.060±0.024 Gemini 3.1 Pro Preview (Google, 2026b) 0.336±0.043 0.136±0.043 Gemini 3.1 Flash-Lite Preview (Google, 2026a) 0.052±0.036 0.020±0.023 Kimi K2.5 (Moonshot AI, 2026) 0.072±0.023 0.008±0.018

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Qwen3.5-397B-A17B (Qwen Team, 2026) 0.000±0.000 0.000±0.000 Qwen3.5-122B-A10B (Qwen Team, 2026) 0.008±0.018 0.000±0.000

[Figure 77]

Table 4: Coop team scores on two cooperative games, where two copies of the same model must coordinate to complete a shared task.

stronger Solo agent, and Claude wins decisively only against Qwen3.5. This indicates that MidlineClash rewards game-specific tactics that do not align with the Solo capability ranking.

Coop games. Table 4 reports team scores when two copies of the same model cooperate. The leaderboard mirrors Solo: GPT-5.5 leads both games, Gemini 3.1 Pro is a close second, and Claude Opus 4.6 follows. Two findings extend beyond Solo. First, the gap between commercial and open-weight VLMs widens: both Qwen3.5 checkpoints score exactly 0.000 on both games, failing to complete a single shared order or handoff. Second, even the strongest model reaches only 0.368 on SharedFloor and 0.184 on HandoffRun, leaving substantial headroom and indicating that LLM-LLM coordination is an unsolved capability gap rather than a saturated benchmark dimension.

###### 5.3 Improvement Dynamics Curve

Setup. We run IDC on LASTSTAND (Solo) and SHAREDFLOOR (Coop) for complementary skill coverage: LASTSTAND stresses reactive control as tiles progressively fall away, while SHAREDFLOOR combines explicit task rules with multiagent coordination. We exclude PvP to keep IDC gains attributable to the reflector rather than to opponent behavior. Four top-performing agents from the cold-start leaderboard participate: Claude Opus 4.6, Claude Opus 4.7, GPT-5.5, and Gemini 3.1 Pro. Each agent completes R=10 rounds of K=5 episodes under PDQ; Round 0 reuses the cold-start baseline from §5.2. The best skill from each (model, game) run is then reapplied to three held-out task variants per game.

###### 5.3.1 Improvement on the original task

Figure 5 reports per-round mean score. Through multi-round agentic reflection, every model on both games discovers skills that improve over its

LastStand SharedFloor Agent origin var1 var2 var3 origin var1 var2 var3

- Claude Opus 4.6 +0.641 (+437%) +0.279 (+175%) −0.168 (−72%) +0.011 (+6%) +0.060 (+39%) +0.036 (+56%) +0.036 (+38%) +0.028 (+30%)

[Figure 78]

- Claude Opus 4.7 +0.620 (+201%) −0.097 (−45%) −0.266 (−76%) −0.044 (−21%) +0.068 (+50%) +0.040 (+43%) +0.020 (+17%) +0.024 (+22%) GPT-5.5 +0.540 (+130%) +0.292 (+57%) +0.422 (+79%) +0.012 (+6%) +0.020 (+5%) +0.034 (+13%) +0.016 (+6%) +0.020 (+10%) Gemini 3.1 Pro Preview +0.701 (+305%) +0.266 (+166%) −0.062 (−36%) +0.017 (+12%) +0.076 (+23%) +0.040 (+17%) +0.016 (+6%) +0.020 (+9%)

[Figure 79]

[Figure 80]

[Figure 81]

Table 5: Transfer of IDC best skill across the origin split and three unseen variants. Each cell shows the gain Vbest − V0 followed by the %-change relative to V0. V0 is the no-skill baseline on that split; Vbest applies the skill that gave the highest mean score during the 10-round run.

###### LastStand

###### SharedFloor

1.00

0.45

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |

0.75

0.36

0.50

0.28

0.25

0.19

0

0.10

R0 R2 R4 R6 R8 R10

R0 R2 R4 R6 R8 R10

round

round

###### Claude 4.6 Claude 4.7 GPT-5.5 Gemini 3.1

- Figure 5: IDC curves: per-round mean episode score across 10 reflection rounds for four agents on LastStand (left) and SharedFloor (right). Horizontal lines mark each agent’s R0 (no-skill PDQ baseline) for reference; deviations above the line indicate skill-driven improvement. Each round aggregates 5 episodes per agent.

Round 0 baseline. On LastStand, best-round gains range from +0.54 to +0.70 in survival fraction (+130% to +437% relative). On SharedFloor, bestround teams complete 1 to 4 additional orders per episode, a substantial improvement in coordination throughput. However, peak performance is typically reached mid-curve rather than at Round 10: on LastStand the two Opus models lose 0.40 to 0.52 between best and final round, while GPT-5.5 and Gemini retain almost all gains. This best-vsfinal gap justifies the best-skill rollback mechanism described in §4.2.

###### 5.3.2 Transfer to held-out variants

Table 5 reports the gain of each best skill on the origin split and three held-out variants per game. SharedFloor transfers universally (16/16 positive across variants). Gains range from +6% to +56%, corresponding to roughly 1 to 2 additional completed orders per episode under the best skill. The skills encode coordination heuristics rather than spatial memory: agents are guided to observe the teammate’s position and currently held item before committing to an order (to avoid both players picking up the same item), to discard duplicates at the trash bin when this happens, to keep distance from the teammate’s workstation to prevent crowding, and to maintain a visible division of labor across stations. Because the vari-

ants only change workbench and item placements while preserving coordination rules, these behavioral heuristics transfer without modification. LastStand transfer depends on skill style, not origin gain magnitude. The origin drops tiles one at a time, so three of four models converge to a “find a safe tile and stay” policy that exploits the single-tile structure. VAR1 (different seed, same mechanics) leaves the structure intact and transfers positively for three of four models. VAR2 (cluster drops of multiple connected tiles) removes the safe pocket the skills relied on: both Opus models collapse (−72% and −76%), while GPT-5.5 gains +79%. Skill inspection (Appendix C) explains the divergence: the Opus and Gemini skills converge on movement-minimizing policies (e.g., “stand still unless your tile is red”, “never chain forward steps”), which are optimal under single-tile drops but maladaptive when a cluster wipes out the safe pocket. GPT-5.5’s skill instead encodes a “move briefly, then reassess” loop that adapts to either mechanic. VAR3 (tiles that track the player) eliminates static safety entirely; transfer drops to small or negative gains for all four models. GPT-5.5 is the only model with positive transfer on all three variants yet has the smallest origin gain (+130%); Opus 4.7 gains +201% on origin but transfers negatively on every variant. This dissociation between origin gain and transferability is the central finding of the variant experiment.

##### 6 Conclusion

We introduced OmniGameArena, a benchmark of twelve newly built UE5 real-time games spanning Solo, PvP, and Coop, and the Improvement Dynamics Curve (IDC), an agentic-reflection harness that produces multi-round self-improvement trajectories. Beyond single-round leaderboard scores, the IDC exposes two additional observables for each (agent, game) pair: how the score evolves across reflection rounds, and how the learned skill behaves on held-out task variants.

##### Limitations

IDC scope. Due to compute constraints, our IDC experiments cover only two environments (LASTSTAND and SHAREDFLOOR), each with three held-out variants, and four agents from the cold-start leaderboard. Scaling IDC to additional games, variants, and models is an extension.

Single-skill format. Our reflector maintains a single bounded skill prompt that is replaced each round, rather than a growing library of skills in the style of Voyager. Library-based extensions are orthogonal to the round-by-round refinement.

Shared model for player and reflector. Each agent uses the same underlying model as both player and reflector. Whether asymmetric setups (e.g., a smaller player paired with a stronger reflector) yield different improvement is untested.

##### References

- Anthropic. 2026a. Introducing Claude Opus 4.6. https://www.anthropic.com/news/

- claude-opus-4-6.

Anthropic. 2026b. Introducing Claude Opus 4.7. https://www.anthropic.com/news/

- claude-opus-4-7.

- Anthropic. 2026c. Introducing Claude Sonnet 4.6. https://www.anthropic.com/news/ claude-sonnet-4-6.

Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar, and Spencer Whitehead. 2026. Webgym: Scaling training environments for visual web agents with realistic tasks. arXiv preprint arXiv:2601.02439.

Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. 2022. Minedojo: Building open-ended embodied agents with internet-scale knowledge. volume 35, pages 18343–18362.

Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, and Jun Wang. 2023. Chessgpt: Bridging policy learning and language modeling. Advances in Neural Information Processing Systems, 36:7216–7262.

- Google. 2026a. Introducing Gemini 3.1 Flash-Lite. https://blog.google/innovation-and-ai/ models-and-research/gemini-models/ gemini-3-1-flash-lite/.
- Google. 2026b. Introducing Gemini 3.1 Pro. https://blog.google/innovation-and-ai/ models-and-research/gemini-models/ gemini-3-1-pro/.

Matthew Hausknecht, Prithviraj Ammanabrolu, MarcAlexandre Côté, and Xingdi Yuan. 2020. Interactive fiction games: A colossal adventure. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7903–7910.

Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P Xing, Ion Stoica, Tajana Rosing, Haojian Jin, and Hao Zhang. 2025. lmgame-bench: How good are llms at playing games? arXiv preprint arXiv:2505.15146.

Lanxiang Hu, Qiyu Li, Anze Xie, Nan Jiang, Ion Stoica, Haojian Jin, and Hao Zhang. 2024. Gamearena: Evaluating llm reasoning through live computer games. arXiv preprint arXiv:2412.06394.

Jen-tse Huang, Eric John Li, Man Ho Lam, Tian Liang, Wenxuan Wang, Youliang Yuan, Wenxiang Jiao, Xing Wang, Zhaopeng Tu, and Michael R Lyu. 2024. How far are we on the decision-making of llms? evaluating llms’ gaming ability in multi-agent environments. arXiv preprint arXiv:2403.11807.

Heinrich Küttler, Nantas Nardelli, Alexander Miller, Roberta Raileanu, Marco Selvatici, Edward Grefenstette, and Tim Rocktäschel. 2020. The nethack learning environment. Advances in Neural Information Processing Systems, 33:7671–7684.

Muyao Li, Zihao Wang, Kaichen He, Xiaojian Ma, and Yitao Liang. 2025. Jarvis-vla: Post-training largescale vision language models to play visual games with keyboards and mouse. In Findings of the Association for Computational Linguistics: ACL 2025, pages 17878–17899.

Mingxian Lin, Wei Huang, Yitang Li, Chengjie Jiang, Kui Wu, Fangwei Zhong, Shengju Qian, Xin Wang, and Xiaojuan Qi. 2025. Embrace-3k: Embodied reasoning and action in complex environments. arXiv preprint arXiv:2507.10548.

Yang Luo, Xuanlei Zhao, Baijiong Lin, Lingting Zhu, Liyao Tang, Yuqi Liu, Ying-Cong Chen, Shengju Qian, Xin Wang, and Yang You. 2025. Vreasonbench: Toward unified reasoning benchmark suite for video generation models. arXiv preprint arXiv:2511.16668.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, and 1 others. 2023. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594.

Loïc Magne, Anas Awadalla, Guanzhi Wang, Yinzhen Xu, Joshua Belofsky, Fengyuan Hu, Joohwan Kim, Ludwig Schmidt, Georgia Gkioxari, Jan Kautz, and 1 others. 2026. Nitrogen: An open foundation model for generalist gaming agents. arXiv preprint arXiv:2601.02427.

Moonshot AI. 2026. Kimi k2.5: Visual agentic intelligence. https://www.kimi.com/blog/kimi-k2-5.

- OpenAI. 2026a. Introducing GPT-5.4. https://

- openai.com/index/introducing-gpt-5-4/.

OpenAI. 2026b. Introducing GPT-5.5. https://

- openai.com/index/introducing-gpt-5-5/.

Davide Paglieri, Bartłomiej Cupiał, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Łukasz Kuci´nski, Lerrel Pinto, Rob Fergus, and 1 others. 2024. Balrog: Benchmarking agentic llm and vlm reasoning on games. arXiv preprint arXiv:2411.13543.

Dongmin Park, Minkyu Kim, Beongjun Choi, Junhyuck Kim, Keon Lee, Jonghyun Lee, Inkyu Park, ByeongUk Lee, Jaeyoung Hwang, Jaewoo Ahn, and 1 others. 2025. Orak: A foundational benchmark for training and evaluating llm agents on diverse video games. arXiv preprint arXiv:2506.03610.

Qwen Team. 2026. Qwen3.5: A Native Multimodal Foundation Model for Efficiency. https://qwen. ai/blog?id=qwen3.5.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Weihao Tan, Ziluo Ding, Wentao Zhang, Boyu Li, Bohan Zhou, Junpeng Yue, Haochong Xia, Jiechuan Jiang, Longtao Zheng, Xinrun Xu, and 1 others. 2024. Towards general computer control: A multimodal agent for red dead redemption ii as a case study. arXiv preprint arXiv:2403.03186, 1(2).

Weihao Tan, Xiangyang Li, Yunhao Fang, Heyuan Yao, Shi Yan, Hao Luo, Tenglong Ao, Huihui Li, Hongbin Ren, Bairen Yi, and 1 others. 2025. Lumine: An open recipe for building generalist agents in 3d open worlds. arXiv preprint arXiv:2511.08892.

Chen Feng Tsai, Xiaochen Zhou, Sierra S Liu, Jing Li, Mo Yu, and Hongyuan Mei. 2023. Can large language models play text games well? current state-of-the-art and open questions. arXiv preprint arXiv:2304.02868.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models, 2023. URL https://arxiv. org/abs/2305.16291, 2(11).

Xinyu Wang, Bohan Zhuang, and Qi Wu. 2025a. Are large vision language models good game players? arXiv preprint arXiv:2503.02358.

Zihao Wang, Xujing Li, Yining Ye, Junjie Fang, Haoming Wang, Longxiang Liu, Shihao Liang, Junting Lu, Zhiyong Wu, Jiazhan Feng, and 1 others. 2025b. Game-tars: Pretrained foundation models for scalable generalist multimodal game agents. arXiv preprint arXiv:2510.23691.

Jiayi Weng. 2026. Learning beyond gradients. https://trinkle23897.github.io/ learning-beyond-gradients/. Blog post.

Yue Wu, Xuan Tang, Tom M Mitchell, and Yuanzhi Li.

2023. Smartplay: A benchmark for llms as intelligent agents. arXiv preprint arXiv:2310.01557.

Yuguang Yue, Irakli Salia, Samuel Hunt, Chris Green, Wenzhe Shi, and Jonathan J Hunt. 2026. Scaling behavior cloning improves causal reasoning: An open model for real-time video game playing. arXiv preprint arXiv:2601.04575.

Alex L Zhang, Thomas L Griffiths, Karthik R Narasimhan, and Ofir Press. 2025. Videogamebench: Can vision-language models complete popular video games? arXiv preprint arXiv:2505.18134.

Kuan Zhang, Dongchen Liu, Qiyue Zhao, Jinkun Hou, Xinran Zhang, Qinlei Xie, Miao Liu, and Yiming Li. 2026. Gameverse: Can vision-language models learn from video-based reflection? arXiv preprint arXiv:2603.06656.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642.

Xiangxi Zheng, Linjie Li, Zhengyuan Yang, Ping Yu, Alex Jinpeng Wang, Rui Yan, Yuan Yao, and Lijuan Wang. 2025. V-mage: A game evaluation framework for assessing vision-centric capabilities in multimodal large language models. arXiv preprint arXiv:2504.06148.

Lingting Zhu, Shengju Qian, Haidi Fan, Jiayu Dong, Zhenchao Jin, Siwei Zhou, Gen Dong, Xin Wang, and Lequan Yu. 2026. Assetformer: Modular 3d assets generation with autoregressive transformer. arXiv preprint arXiv:2602.12100.

##### A Latency-Controlled Real-Time Eval

We further evaluate a subset of agents under a Latency-Controlled Real-Time (LCRT) protocol. In PDQ, the simulator is paused while the model is thinking, and the returned action is executed immediately from the observed state. In LCRT, the simulator is likewise paused during the wall-clock model call, but the model’s measured decision latency is then injected back into the game timeline before the action is executed. Thus an action predicted from observation ot is applied after approximately ∆t seconds of simulated game time, where ∆t is the model’s decision latency. LCRT requires a reliable estimate of pure model inference time. We therefore report LCRT only for the four models whose backends expose usable model-side timing signals in our implementation: Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.5, and GPT-5.4. Other agents are excluded because their available timings fold in client-side overheads such as request queuing, network transfer, retries, or wrapper latency; injecting such end-to-end wall-clock measurements would confound model latency with infrastructure latency and make the comparison unfair.

Solo Games. We focus the LCRT analysis on tasks where latency is expected to affect either the evolving game state or the available time budget, and accordingly report LastStand and MonsterShoot as dynamic real-time tasks and SoloCraft as a timebudgeted interaction task. As shown in Table 6, the three tasks reveal distinct sensitivity patterns. MonsterShoot is the most latency-sensitive: all four models drop, most steeply GPT-5.5 (∆ = −0.408), consistent with its need for continuous aiming and target tracking. LastStand behaves differently: its optimal policy is nearly stationary, waiting on a safe tile and moving only when one’s own tile is about to fall, so latency is not necessarily harmful here, and taking fewer, later actions can even avoid fatal missteps. This may explain why Claude Opus

- 4.6 and GPT-5.4 even improve under LCRT and Claude Sonnet 4.6 stays essentially flat, while GPT-
- 5.5 is the only model that degrades. SoloCraft is not a reactive-control task in the same sense; here latency mainly consumes the episode time budget and reduces interaction throughput, an effect we quantify below. PvP Games. Figure 6 reports Player 1 win rates on MidlineClash under LCRT for the four commercial VLMs. They differ from the PDQ heatmap (Figure 4), but the difference reflects the change

Agent LastStand MonsterShoot SoloCraft Claude Opus 4.6 (Anthropic, 2026a)

0.248±0.126

0.180±0.027

0.080±0.018

[Figure 82]

∆ +0.101

∆ -0.182

∆ -0.148

0.154±0.058

0.116±0.033

0.072±0.010

Claude Sonnet 4.6 (Anthropic, 2026c)

[Figure 83]

∆ +0.010

∆ -0.050

∆ -0.052

0.324±0.166

0.056±0.023

0.052±0.010

GPT-5.5 (OpenAI, 2026b)

[Figure 84]

∆ -0.092

∆ -0.408

∆ -0.200

0.253±0.094

0.138±0.107

0.040±0.036

GPT-5.4 (OpenAI, 2026a)

[Figure 85]

∆ +0.105

∆ -0.188

∆ -0.044

Table 6: LCRT results on selected Solo games. Each cell reports mean with standard deviation as subscript over N=5 episodes, with ∆ denoting the change relative to the corresponding PDQ result.

###### MidlineClash

1.00

|N/A|0.80|0.20|0.40|
|---|---|---|---|
|0.20|N/A|0.00|0.60|
|0.20|0.80|N/A|0.80|
|0.00|0.00|0.20|N/A|

[Figure 86]

Opus4.6

0.80

Sonnet4.6

0.60

Score

0.40

GPT-5.5

0.20

GPT-5.4

0.00

Opus4.6 Sonnet4.6 GPT-5.5 GPT-5.4

Figure 6: PvP win rates of Player 1 (row) against Player 2 (column) on MidlineClash under latency control setting.

of clock mode rather than a change in model ability. Charging decision latency against the game clock leaves far less game time per move, so each player completes only about 18 actions per game under LCRT against the ∼42-action PDQ budget, and matches compress into low-scoring, frequently drawn games. The effect is clearest in the GPT-5.5vs-Opus 4.6 pairing, the one matchup both protocols share: GPT-5.5 swept it 10–0 under PDQ by wide margins (e.g. 8–0, 10–4, 11–1), whereas under LCRT the same pairing yields 5 GPT-5.5 wins, 3 Opus wins, and 2 draws, with narrow scorelines such as 1–0, 2–2, and 0–0 and the pairing’s average per-player score falling from 0.121 to 0.016. GPT5.5 still wins the pairing and still posts the highest average Player-1 win rate (0.60, versus 0.07 for the weakest, GPT-5.4), even though it carries by far the largest inference latency (∼16 s of pure model time against ∼4–7 s for the others). Because both sides pay the same delay, latency in symmetric play mainly compresses margins and amplifies singlegame randomness rather than re-ranking the agents, a much milder effect than on the throughput tasks below.

Cooperative Game. We further report the cooperative task SharedFloor, in which two instances of the same model coordinate to complete shared or-

Agent SharedFloor Claude Opus 4.6 (Anthropic, 2026a)

0.048±0.016

[Figure 87]

∆ -0.104

0.028±0.016

Claude Sonnet 4.6 (Anthropic, 2026c)

[Figure 88]

∆ -0.120

0.048±0.020

GPT-5.5 (OpenAI, 2026b)

[Figure 89]

∆ -0.320

0.024±0.020

GPT-5.4 (OpenAI, 2026a)

[Figure 90]

∆ -0.044

Table 7: LCRT results on the cooperative game SharedFloor. Each cell reports mean with standard deviation as subscript over N=5 episodes, with ∆ denoting the change relative to the corresponding PDQ result.

ders before a fixed match deadline (Table 7). Every model degrades under LCRT, and the loss scales with the action budget: because LCRT charges each model’s decision latency against the match clock, the slowest agent completes the fewest interactions. GPT-5.5, the slowest, takes the largest absolute drop, from 0.368 under PDQ to 0.048 (∆ = −0.320); but having started far ahead, it still ties Opus 4.6 for the best LCRT score rather than collapsing. GPT-5.4, the fastest, retains the most actions and changes the least (∆ = −0.044). We quantify this action-budget account next.

Action budget versus per-action efficiency. To see why scores fall on the throughput tasks, we measure for each agent both its actions per episode and its score per action under the two protocols on SoloCraft (Table 8) and SharedFloor (Table 9). The action budget shrinks monotonically with model speed: GPT-5.5, the slowest, completes the fewest actions, only about 8 of its 43 PDQ actions per episode on SoloCraft and 14 versus 84 on SharedFloor, while the faster agents retain far more. The drop is overwhelmingly a budget effect: score per action is largely preserved between PDQ and LCRT, most clearly for GPT-5.5 (essentially unchanged on SoloCraft, only mildly lower on SharedFloor), so each agent’s score falls roughly in proportion to its smaller action count, e.g. GPT-5.5 on SoloCraft drops 0.252 → 0.052 for 43 → 8 actions. This account is specific to interaction-throughput tasks, where score accumulates with the number of useful interactions before a fixed deadline; it does not extend to the dynamic tasks, where latency acts through reaction timing rather than throughput and is not even monotone, helping in LastStand, whose near-stationary optimal policy rewards fewer and later actions, but hurting in MonsterShoot through missed and mist-

Actions / episode Score / action Agent PDQ LCRT PDQ LCRT

Claude Opus 4.6 (Anthropic, 2026a) 43 17 0.0053 0.0047 Claude Sonnet 4.6 (Anthropic, 2026c) 43 19 0.0029 0.0038 GPT-5.5 (OpenAI, 2026b) 43 8 0.0059 0.0065 GPT-5.4 (OpenAI, 2026a) 43 28 0.0020 0.0014

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

- Table 8: SoloCraft action budget. Actions per episode and normalized score per action under PDQ vs. LCRT. Score per action is nearly unchanged, so each agent’s LCRT drop (Table 6) is almost entirely its smaller action budget.

Actions / episode Score / action Agent PDQ LCRT PDQ LCRT

[Figure 95]

Claude Opus 4.6 (Anthropic, 2026a) 84 33 0.0018 0.0015 Claude Sonnet 4.6 (Anthropic, 2026c) 84 36 0.0018 0.0008 GPT-5.5 (OpenAI, 2026b) 84 14 0.0044 0.0034 GPT-5.4 (OpenAI, 2026a) 84 55 0.0008 0.0004

[Figure 96]

[Figure 97]

[Figure 98]

- Table 9: SharedFloor action budget (actions and team score summed over both players; score normalized). The LCRT drop tracks the shrinking action budget: GPT5.5, the slowest, completes by far the fewest actions, yet keeps the highest score per action under both protocols, so its loss is a budget effect rather than a per-action collapse.

imed shots. Together these regimes show that latency is a first-class evaluation axis whose effect is task-dependent, taxing single-agent throughput, compressing rather than re-ranking symmetric play, and even helping near-stationary survival, so PDQ margins do not transfer directly to real-time deployment.

##### B Qualitative Comparison with IDC

IDC skill induction improves the agents’ behavior across both environments. Without IDC, the agents tend to make unstable or inefficient decisions: in the survival-oriented task, the agent fails to consistently maintain a safe position, while in the cooperative task, the agents show weaker coordination and complete fewer objectives. After IDC, the learned behaviors become more effective and task-aligned. The agent in the survival task maintains safer positions and achieves a higher score, and the agents in the cooperative task exhibit better coordination and obtain a higher team score. (See Figure 8)

##### C Skill Inspection

This appendix lists the best measured skill prompt for each game–model pair in the IDC comparison. It explains the divergence discussed in the main text: LastStand prompts converge on conservative

[Figure 99]

Figure 7: Qualitative comparison on Last Stand using GPT-5.5.

[Figure 100]

Figure 8: Qualitative comparison on Shared Floor using Gemini-3.1-Pro.

tile-survival behavior, while SharedFloor prompts emphasize cooperative division of labor, station alignment, and order-refresh handling. The scope here is limited to LastStand and SharedFloor; ObstacleRun3D is intentionally excluded.

##### D Visualization

Visualization results are shown in Figures 9–20. For each game, we visualize representative trajectories from different models. Each row corresponds to one model or matchup, with five sampled frames illustrating the progression of the episode.

- LastStand claude-opus-4-6

This is a survival game: you stand on a tiled platform where tiles progressively turn red (warning) then fall away. Your only goal is to stay alive as long as possible; score increases linearly with survival time.

Your default action is absolute stillness: no movement inputs. Stay still for as many consecutive steps as possible. Every movement risks walking into a gap; stillness on a safe tile carries zero risk. Commit to stillness within the first few steps once you find any safe gray tile.

Camera rotation is completely safe and costs nothing. Use camera turns to scan your surroundings without moving your character. Proactively rotate the camera to check all directions, especially once holes start appearing on multiple sides.

The only reason to move is if the tile directly under your character has turned red. Before moving, explicitly verify: “Is my tile red?” If your tile is gray, do not move, regardless of how many red tiles or gaps you see nearby. Adjacent red tiles are not your problem until they become gaps, and even then you only move if your own tile is threatened.

Moving into a gap is always fatal and always worse than staying on a tile that might turn red soon. A red tile gives you at least one full step of warning before it falls. Treat unnecessary movement as more dangerous than a nearby red tile.

When you must move, use the absolute minimum directional inputs: one or two total across all sub-steps. Never use three or more movement inputs in a single step. Overshoot into gaps is the primary killer; a tiny step is always safer than a large one.

Stay near the center of the largest connected cluster of safe gray tiles. Once you reach a reasonable central position, stop immediately and commit to staying there indefinitely until your own tile turns red.

The first few steps are critical: red tiles will appear around the platform but most are not near you. Resist the urge to relocate; if your current tile is gray, stay on it.

Never jump. Jumping adds unpredictability and frequently causes fatal overshoot. Accept being on a smaller island rather than risking a jump.

Table 10: Best skill prompt for LastStand using claude-opus-4-6.

- LastStand claude-opus-4-7

The score grows monotonically each tick you remain alive, so time alive equals score. Almost every move risks landing on a red tile or in a gap.

Never chain forward steps. Repeated w ; w ; w over multiple sub-steps is the strongest predictor of falling; each repeat compounds blindness to what just appeared underfoot.

Avoid jumps unless a gap is unavoidable. Space/jump landings are harder to read than walks; only use when no adjacent solid tile exists.

Table 11: Best skill prompt for LastStand using claude-opus-4-7.

LastStand gemini-3.1-pro-preview

The goal is to survive as long as possible on a platform where tiles turn red and fall away. If you are standing on a safe, grey tile, do not move. Output empty actions to stay in place and survive. Only move when the tile you are standing on turns red. When your tile turns red, you must visually identify a solid grey block to move to. Empty spaces showing the blue sky below are holes; moving into them will cause you to fall and die. Look carefully at the adjacent spaces: forward, backward, left, right. Only move in a direction if you clearly see a solid grey tile there. Do not guess or move blindly. Use minimal steps to reach the adjacent safe tile. Do not use long sequences of movement as you will overshoot and fall into a hole.

###### Table 12: Best skill prompt for LastStand using gemini-3.1-pro-preview.

###### LastStand gpt-5.5

Your score mainly improves by staying alive, so prefer stable survival over exploring distant floor. Keep the view tilted enough toward the floor to judge which tiles are intact, warning-colored, or already gone. If your current tile turns warning-colored, leave promptly for the nearest connected intact tile; if warning tiles are only nearby, first verify whether standing still is still safe. Favor short repositioning pulses followed by a pause to reassess, because long continuous movement across fragmented floor caused many early deaths. Aim to finish each move near the center of the largest connected intact patch, not on edges, seams, narrow bridges, or corners beside voids. When already centered on a safe small island, hold position until a direct warning appears rather than wandering for marginally larger space. Treat narrow bridges and isolated transfers as high risk: cross only when the current area is becoming unsafe, then stop and recenter on the wider patch. Avoid jumping unless there is no connected walking escape from a collapsing or isolated tile.

Table 13: Best skill prompt for LastStand using gpt-5.5.

- SharedFloor claude-opus-4-6

This is a cooperative factory game. You and your teammate share the same arena with parts to pick up, a WorkBench to upgrade them, an Order Counter to submit them, and a Bin to discard unwanted items. The team score is the sum of both players’ individual scores.

Divide labor immediately: at the start, one player should handle the item closest to them while the other targets a different item. If you see your teammate already moving toward or carrying an item, choose a different one.

Avoid duplication above all else. Before upgrading any item, check if your teammate is carrying or upgrading the same item type. If you cannot confirm they are doing something different, do the basic 1-point version of that order instead. Getting stranded with an unmatched High-grade item wastes 7–10 steps discarding and is the single biggest score killer.

When both a basic order and a High-grade order exist for the same item type, split them: one player does basic, one does High-grade. If you are unsure which your teammate chose, default to basic; it is faster and risk-free.

Only upgrade when you are confident your teammate is not upgrading the same type and at least 20 seconds remain. The upgrade cycle takes 8–10 steps; a failed or duplicated upgrade wastes the entire investment.

Never pick up an item that does not match any current order. Check the orders display before grabbing anything. A High-grade item cannot fulfill a basic order of the same type.

When submitting at the Order Counter, you must be precisely positioned for the “Submit” prompt to work. If pressing interact does not register, nudge your position in a distinctly different direction and retry; do not spam interact from the same spot repeatedly.

After upgrading at the WorkBench, verify the label above your character says “High-grade” before leaving. If it still shows the basic item name, interact again.

With fewer than about 15 seconds remaining, only attempt basic 1-point orders that do not require upgrading. After submitting an order, immediately check what new orders appeared and plan your next target before moving.

###### Table 14: Best skill prompt for SharedFloor using claude-opus-4-6.

- SharedFloor claude-opus-4-7

This is a 2-player coop kitchen-style game with a shared floor. The bottom HUD lists exactly two open orders at any time: one usually labeled High-grade (2), the other a base item (1). High-grade variants must be upgraded at the WorkBench before submission and pay double. Orders rotate the moment a successful submit lands, not on a timer. The round is hard time-limited. Stations include WorkBench, Order Counter, and a Bin for discards.

Use these as heuristics; do not follow them as a script.

Open by proximity, then commit. At step 0, look at spawns and pick whichever order’s part you are closer to; the partner takes the other. Do not reverse that assignment in the first several steps because of a guess that the partner might also be going there; abandoning costs more steps than the conflict would. Only switch if you actually see the partner already at that spawn with a PickUp prompt or visibly carrying that item.

Trust the on-screen prompt as ground truth. A visible PickUp / Upgrade / Submit / Bin prompt authoritatively tells you both that you are in range and whether you are or are not carrying the relevant item. The floating green name above your character is unreliable; the prompt is not.

Prompt visible → interact, do not reposition. When an interact prompt is on screen, prioritize the interact that frame over more movement; extra movement that frame can carry you back out of range.

Adjacency, not approach. Pickup / Submit / Upgrade triggers on a specific tile, not on raw proximity. If you keep advancing toward a target and no prompt appears, you are misaligned on the perpendicular axis; slide sideways rather than pushing further forward.

Score-credit can lag a step or two behind a successful submit. If you just pressed interact at the counter and the team_score has not ticked yet, do not mash interact more times in place; the credit usually arrives next observation. If it still has not after a couple of steps, you were misaligned: slide sideways and re-approach.

Re-read the order list every time the score ticks. After any successful submit, yours or the partner’s, the orders rotate. Before committing your next move, look at the HUD orders again; if what you are carrying or heading for no longer appears there, redirect immediately.

Bin off dead weight. If orders rotated while you were carrying and what you hold matches neither current order, dump at the Bin and re-grab; do not tour the map with a useless item.

Single re-read on carry-state flip. If your stated inventory has flipped between consecutive frames, pause movement briefly and resolve from the prompt: PickUp showing means you are empty; Upgrade or Submit showing means you are carrying.

Specialization beats symmetry. One player owning the full High-grade loop, pickup → WorkBench → Submit, while the partner runs the base-item loop is a known high-score pattern, not a coordination failure.

Diversify at the workbench. Do not both queue the same part type at the WorkBench; while one upgrades, the other should be staging the other order’s part or running a base submit.

No idle near the counter. Empty-handed after a submit, immediately pick the next part the current orders need; idle frames are pure loss.

Endgame triage. When the round timer is visibly near zero and you are holding a base item, go straight to the Order Counter; do not detour to the WorkBench.

###### Table 15: Best skill prompt for SharedFloor using claude-opus-4-7.

SharedFloor gemini-3.1-pro-preview Strict division of labor: never work on the same order as your teammate. Check the “Current orders” list and observe your teammate’s held item and movement direction. If they are moving toward a dispenser, assume they are claiming that item’s order and choose a different one.

Preventing the polite dance: if you and your teammate accidentally target the same dispenser, the player who is physically closer should claim it. If you decide to yield, you must commit to a different order and never switch back, even if your teammate also yielded.

Inferring High-grade intent: High-grade orders require taking a base item to the WorkBench to upgrade it. If your teammate is carrying a base item and moving toward the WorkBench, they are claiming the High-grade order for that item.

Verify held item before upgrading: you must physically pick up a base item from its dispenser before you can upgrade it. Do not go to the WorkBench and press interact if your character is empty-handed.

Discarding obsolete items: you can only hold one item at a time. If you are holding an item that is no longer needed, immediately go to the Bin to discard it so you can pick up a new item.

Navigating the Order Counter: the Order Counter is a solid obstacle. If you are stuck walking into it, use perpendicular movement to navigate around it before continuing to your destination.

Combine movement and interaction: do not stop completely to interact with dispensers, the WorkBench, or the Order Counter. Instead, include the interact action in the same step as your final approach movement to save time and ensure you are close enough to trigger the interaction.

###### Table 16: Best skill prompt for SharedFloor using gemini-3.1-pro-preview.

SharedFloor gpt-5.5 At each decision, read the active orders and choose the closest unfinished order you can complete now, not the item you were already planning around. Split labor: when your teammate is clearly carrying or working an order, take a different visible order, a normal cash-in, or a nonblocking support route. Start an upgrade only when you are visibly holding the matching raw item and can picture the whole workbench-tocounter return; otherwise cash a normal order. At any station, use one deliberate interaction, then check for a changed held item, score, prompt, or teammate score before repeating. If an interaction fails, adjust position or reassess whether you are holding the right item instead of standing still or pressing again from the same spot. After your teammate scores, immediately refresh the order list and treat your carried duplicate as suspect; repurpose it nearby or drop it rather than taking a long disposal trip. Once you carry a matching finished item, deliver it at the counter, clear the interaction area, then refresh orders and abandon duplicates if your teammate scores first. Near the end, score only items already held near the counter or already returning from an upgrade; ignore fresh pickups unless they are essentially a submission already.

###### Table 17: Best skill prompt for SharedFloor using gpt-5.5.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Claude Sonnet 4.6

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Gemini 3.1 Flash-Lite Preview

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Gemini 3.1 Pro Preview

- GPT-5.4

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

- GPT-5.5

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

Kimi-K2.5

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Qwen3.5-122B-A10B

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Qwen3.5-397B-A17B

- Figure 9: Visualization results for cue_chase. Each row shows one model, with five sampled frames from the corresponding trajectory.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Gemini 3.1 Flash-Lite Preview

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Gemini 3.1 Pro Preview

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

GPT-5.4

- GPT-5.4 Mini

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

- GPT-5.5

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Kimi-K2.5

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Qwen3.5-122B-A10B

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Qwen3.5-397B-A17B

- Figure 10: Visualization results for last_stand. Each row shows one model, with five sampled frames from the corresponding trajectory.

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Claude Sonnet 4.6

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Gemini 3.1 Flash-Lite Preview

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Gemini 3.1 Pro Preview

- GPT-5.4

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

- GPT-5.5

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Kimi-K2.5

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

Qwen3.5-122B-A10B

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Qwen3.5-397B-A17B

- Figure 11: Visualization results for monster_shoot. Each row shows one model, with five sampled frames from the corresponding trajectory.

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Gemini 3.1 Flash-Lite Preview

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Gemini 3.1 Pro Preview

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

GPT-5.4

- GPT-5.4 Mini

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

- GPT-5.5

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Kimi-K2.5

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Qwen3.5-122B-A10B

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Qwen3.5-397B-A17B

- Figure 12: Visualization results for obstacle_run_2d. Each row shows one model, with five sampled frames from the corresponding trajectory.

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Gemini 3.1 Flash-Lite Preview

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Gemini 3.1 Pro Preview

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

GPT-5.4

- GPT-5.4 Mini

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

- GPT-5.5

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

Kimi-K2.5

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

Qwen3.5-122B-A10B

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

Qwen3.5-397B-A17B

- Figure 13: Visualization results for obstacle_run_3d. Each row shows one model, with five sampled frames from the corresponding trajectory.

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Claude Sonnet 4.6

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

Gemini 3.1 Flash-Lite Preview

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

Gemini 3.1 Pro Preview

- GPT-5.4

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

- GPT-5.5

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Kimi-K2.5

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

Qwen3.5-122B-A10B

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Qwen3.5-397B-A17B

- Figure 14: Visualization results for scene_escape. Each row shows one model, with five sampled frames from the corresponding trajectory.

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

Claude Sonnet 4.6

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Gemini 3.1 Flash-Lite Preview

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

Gemini 3.1 Pro Preview

- GPT-5.4

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

- GPT-5.5

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

Kimi-K2.5

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

Qwen3.5-122B-A10B

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

Qwen3.5-397B-A17B

- Figure 15: Visualization results for solo_craft. Each row shows one model, with five sampled frames from the corresponding trajectory.

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

Claude Sonnet 4.6

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

Gemini 3.1 Flash-Lite Preview

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

Gemini 3.1 Pro Preview

- GPT-5.4

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

- GPT-5.5

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

Kimi-K2.5

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Qwen3.5-122B-A10B

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

Qwen3.5-397B-A17B

- Figure 16: Visualization results for handoff_run. Each row shows one cooperative model pair, with five sampled frames from the corresponding episode.

- Claude Opus 4.6

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

- Claude Opus 4.7

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

Claude Sonnet 4.6

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

Gemini 3.1 Flash-Lite Preview

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

Gemini 3.1 Pro Preview

- GPT-5.4

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

- GPT-5.5

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

Kimi-K2.5

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

Qwen3.5-122B-A10B

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

Qwen3.5-397B-A17B

- Figure 17: Visualization results for shared_floor. Each row shows one cooperative model pair, with five sampled frames from the corresponding episode.

GPT-5.5 vs Claude Opus 4.6

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

Kimi-K2.5 vs Claude Opus 4.6

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

- Figure 18: Visualization results for crystal_guard. Each row shows one representative PvP matchup, with five sampled frames from the corresponding match.

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

Claude Opus 4.6 vs Kimi-K2.5

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

Claude Opus 4.6 vs Gemini 3.1 Pro Preview

- Figure 19: Visualization results for midline_clash. Each row shows one representative PvP matchup, with five sampled frames from the corresponding match.

Claude Opus 4.6 vs Kimi-K2.5

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

Claude Opus 4.6 vs GPT-5.5

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

- Figure 20: Visualization results for sky_duel. Each row shows one representative PvP matchup, with five sampled frames from the corresponding match.

