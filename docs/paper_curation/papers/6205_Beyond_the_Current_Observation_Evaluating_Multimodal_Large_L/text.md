## arXiv:2606.19338v1[cs.CV]17Jun2026

[Figure 1]

2026-6-18

# Beyond the Current Observation: Evaluating Multimodal Large Language Models in Controllable Non-Markov Games

#### Shengyuan Ding*1,2,3, Xilin Wei*1, Xinyu Fang*4, Haodong Duan†5, Dahua Lin3,5, Jiaqi Wang†2 and Yuhang Zang†3

1Fudan University, 2Shanghai Innovation Institute, 3Shanghai Artificial Intelligence Laboratory, 4Zhejiang University, 5The Chinese University of Hong Kong

Deploying multimodal foundation models as closed-loop policies increasingly requires conditioning actions on observations that are no longer visible. However, existing benchmarks either expose the full state, conflate hidden-state reconstruction with other agent skills, or test recall only after an episode has ended. We introduce RNG-Bench (Reconstructive Non-Markov Games), a benchmark suite designed to isolate a base model’s ability to reconstruct past observations and act on them during multi-step interaction. RNG-Bench includes two complementary games: Matching Pairs, where card identities briefly revealed at specific locations must later be recalled, and 3D Maze, where egocentric views must be integrated into a spatial map. Both games are evaluated under a unified harness with three controlled difficulty axes: grid size, visual pattern, and observation modality. The benchmark further introduces a head-to-head duel protocol to control for instance-level variance and a Memory Gap metric that disentangles forgetting from poor action selection. The hardest configurations require contexts of roughly 128K tokens and 350 image inputs per episode, and remain far from saturated by frontier MLLMs. Memory Gap analysis shows that most residual errors stem from forgetting earlier observations rather than from suboptimal decision making. Finally, fine-tuning Qwen3.5-9B on optimal-policy rollouts and filtered model demonstrations improves performance on RNG-Bench and transfers to existing benchmarks without degrading general multimodal capability.

### 1. Introduction

In long-horizon interaction, the correct action often depends on observations from several turns earlier rather than on the current view. A single recall error can change subsequent observations and compound throughout the episode. We refer to this regime as Non-Markov: the current observation alone is insufficient for optimal action, so the model must infer the relevant hidden state from its history before acting. As multimodal models are deployed in closed-loop settings such as embodied control and multi-turn tool use, this ability is becoming critical alongside reasoning over visible inputs.

Fig. 1(a) makes the contrast concrete: in Markov games such as Go or chess the visible board determines the next move, whereas in Matching Pairs two boards with the same visible state can require different actions when their histories differ, so the visible state is not a sufficient statistic. This places two demands on the model: it must retain identities and locations across many turns and re-bind them to fading visual evidence, and it must tolerate that any recall error is causal, altering the next observation rather than only the final score.

Existing benchmarks fall into three families, none of which isolates this regime (Tab. 1). Fully observed games such as Go and chess [68] and strategic-reasoning suites [12, 20] reward search and planning over a state that is already visible and do not require recall of earlier observations. Agent and multi-environment suites [8, 24, 27, 41, 44, 59–61, 80] do include hidden information, but bundle it with exploration, rule discovery, and free-form action, so the four confounds named above remain entangled with memory at the level of episode outcomes. Long-context and memory benchmarks [4, 22, 26, 40, 53, 78] do isolate recall, but

*Equal contribution. †Corresponding authors.

(b) RNG Test Suite

(a) Markov vs. Non-Markov Markov

###### Matching Pairs

3D Maze

Current state is sufficient for Action

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Tic-Tac-Toe Go Chess

1. Mainly evaluate model reasoning ability

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

State N State N+1

State N+2

Walking Without Map

2. Require only current-state understanding

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Reconstruct state from History first, then Act

Non-Markov

###### State M State M+1 State M+2 Walking With a Minimap

Action: aB

Facing the same Current State

3

- 2

1. Evaluate memory for action ability

Flip two cards each turn and remove matched pairs

Match Failed

Navigate an egocentric 3D maze from local views

♥

♣

3

3

♥

♥

A B C

Filp Back

3

2. Require reconstructing hidden state from history

♥

- a
- b

aC Modality Pattern MemGap

Scale

[Figure 18]

3

3

- 3

8

♥

♥

Match Failed

♥

♠

External memory helps both, more strongly for cards

State tracking is sensitive to visual patterns

Performance drops quickly as belief state size grows

Text works best; 2D/3D visual tracking is harder

3. Faulty recall leads to wrong actions

History Info is different So Action differs

Filp Back

Action: aA

- Figure 1: (a) Markov games are determined by the current state, while RNG-Bench is non-Markov. (b) Two environments, Matching Pairs and 3D Maze, evaluated along three controlled axes (scale, visual pattern, and observation modality) with a Memory Gap diagnostic that isolates forgetting from action selection.

Benchmark Domain Eval MM Closed-loop NM-Focus Multi-pl. Scal.-Diff. Max Ctx Max #Img

- (i) Fully-visible game benchmarks GameBench [12] 9 strategic board / card games Agent ∼ ✓ ✗ ✓ ✗ 6† 1 GTBench [20] 10 game-theory games Agent ✗ ✓ ✗ ✓ ✗ 8† 0

- (ii) Agent / multi-environment suites with hidden information (bundled with other skills) AgentBench [44] Multi-task agent suite (8 envs) Agent ✗ ✓ ✗ ✗ ✗ 12† 0 SmartPlay [80] 6 text games (RPS, Bandit, Hanoi, ...) Agent ✗ ✓ ✗ ✗ ✓ 6† 0 AvalonBench [41] Social deduction (hidden roles) Agent ✗ ✓ ✗ ✓ ✗ 3.5† 0 BALROG [60] 6 RL game envs (incl. POMDPs: NetHack, ...) Base ✓ ✓ ✗ ✗ ∼ 16† 1 LMGame-Bench [27] 6 video games + scaffolds Both ✓ ✓ ✗ ✗ ✗ 20 1 GameWorld [59] 34 browser games, 170 tasks Agent ✓ ✓ ✗ ✗ ∼ 8† 3 MACHIAVELLI [61] 134 text-adventure games Base ✗ ✓ ✗ ✗ ✗ 2† 0 clembench [8] Dialogue games Base ✗ ✓ ✗ ✓ ✓ 6† 0 TextArena [24] 100+ text games (TrueSkill rating) Base ✗ ✓ ✗ ✓ ∼ 32† 0

- (iii) Long-context & memory benchmarks (game-based) EMemBench [40] Text + visual games (episodic-memory QA) Both ✓ ✗ ✗ ✗ ✗ 20† 4 RNG-Bench (ours) Non-Markov games (2D card + 3D maze) Base ✓ ✓ ✓ ✓ ✓ 128 350

- Table 1: RNG-Bench vs. prior benchmarks, grouped as in §1: (i) fully-visible games, (ii) agent suites that mix hidden information with other skills, (iii) game-based memory benchmarks. Eval: raw model (Base) vs. wrapped harness (Agent). MM: multimodal observations. Closed-loop: per-step action vs. post-hoc QA. NM-Focus: non-Markov recall as the central axis. Multi-pl.: duel or multi-agent protocol. Scal.-Diff.: controllable difficulty knobs. Max Ctx and Max #Img: max per-prompt tokens and images. ∼ = partial; † = estimated from code.

probe it post-hoc: the model reads a trajectory and answers a question once, a remember-to-answer setting in which a recall error has no effect on subsequent inputs. Our regime is instead remember-to-act, where each recall feeds back into the next observation.

We introduce RNG-Bench (Reconstructive Non-Markov Games), a benchmark designed to isolate rememberto-act along controlled axes, instantiated as two complementary closed-loop games. Matching Pairs is a card game that each symbol is revealed for a single turn and must later be recalled by location, isolating static, categorical hidden state. 3D Maze is an egocentric navigation task that corridors leave the field of view as the player moves and must be reassembled into a map, isolating dynamic, spatial hidden state. Both games run in a closed loop: the model issues one action per turn and a faulty recall becomes a wrong move that reshapes the remaining observations, in contrast to benchmarks that score a single post-hoc answer. We vary difficulty along controlled axes: grid or map size, visual pattern, and observation modality (text or image), while rule understanding is held fixed by in-prompt rules and action formatting by a strict parser, so a drop along any axis is attributable to the axis rather than to a parallel confound. To analyze residual failures, we compare each model against an oracle condition that injects the true hidden state at every step. The score gap between

the two is our Memory Gap, separating forgetting from decision making given the correct state.

RNG-Bench leaves substantial headroom for current models. On image Matching Pairs at 10×10, GPT5.4 [57] matches 62.3% of pairs and Qwen3.5-397B [63] reaches 25.3%, and across 16 head-to-head duels Gemini-3.1-Pro [23] wins every matchup. For reference, an optimal policy uses roughly 60% fewer moves per matched pair than the strongest model (3.24 vs. 8.01). On 3D Maze at 13×13 (mean optimal path 60 steps), Gemini-3.1-Pro obtains the best result with 50.0% SR and 49.7% GS, while GPT-5.4 and Seed-2.0-Lite each reach 20.0% SR, Kimi-K2.5 reaches 10.0% SR, and Qwen3.5-397B reaches 0.0% SR, a ranking that diverges from Matching Pairs and points to a distinct hidden-state demand. The Memory Gap is consistent with forgetting accounting for a large share of the residual error rather than decision making given the correct state, and we further show that supervised fine-tuning on simulator rollouts from RNG-Bench narrows this gap and transfers to external memory and spatial benchmarks (Sec. 5).

Contributions: (1) We release RNG-Bench, two games (Matching Pairs and 3D Maze) under a unified closed-loop harness with a duel protocol. (2) We evaluate leading multimodal models along three controlled axes (grid or map size, visual pattern, and observation modality) and introduce a Memory Gap metric that attributes failures to forgetting earlier observations. (3) We construct training data from optimal-policy rollouts and filtered model demonstrations; fine-tuning Qwen3.5-9B improves RNG-Bench performance and transfers to existing benchmarks [40, 65] without regressing on general capability.

### 2. Related Work

Game Benchmarks. Game benchmarks evaluate reasoning, planning, and multimodal action in interactive settings. AgentBench covers diverse agent tasks [44], while GameBench, BALROG, and GameWorld use games to test strategic reasoning and multimodal game play [12, 59, 60]. However, these broad benchmarks make memory-specific failures hard to isolate, as errors may come from perception, rule understanding, exploration, planning, or action formatting. In contrast, our environments make hidden state explicit and controllable, allowing Matching Pairs and 3D Maze to test whether models can recover information that is invisible and use it for later actions.

Long-Context and Retrieval Benchmarks. Long-context benchmarks study how well models use information distributed across extended inputs. LongBench, L-Eval, M4LE, and Ada-LEval provide broad long-context evaluation suites [3, 4, 33, 73], while Lost in the Middle, RULER, NoLiMa, HELMET, and LongBench v2 probe position sensitivity, literal-match shortcuts, and realistic long-context reasoning [5, 26, 43, 54, 85]. Retrieval and reading benchmarks such as MS MARCO, TriviaQA, KILT, and BEIR test whether models can locate relevant evidence from available sources [6, 30, 34, 62, 72, 83], and MMNeedle extends this to long visual contexts [74]. These show that placing evidence in a context window does not guarantee robust use of it. Our setting differs because long context is produced by interaction rather than given as a fixed input, and key evidence may be seen only once before it becomes useful later.

### 3. Benchmark Design

#### 3.1. Problem Formulation

We model each benchmark instance as a Partially Observable Markov Decision Process (POMDP) (𝒮,𝒪,𝒜,𝑇,𝑍,𝑅) [31]. Here, 𝒮, 𝒪, and 𝒜 denote the state, observation, and action spaces. The transition function 𝑇 specifies how the state changes after an action, the observation function 𝑍 specifies what the agent can observe from the current state, and 𝑅 gives the reward. At step 𝑡, the agent receives the current observation 𝑜𝑡 and the in-context episode history ℎ𝑡 = (𝑜1,𝑎1,...,𝑜𝑡−1,𝑎𝑡−1,𝑜𝑡), then selects action 𝑎𝑡. We evaluate models as history-based policies 𝜋(𝑎𝑡 | ℎ𝑡) that directly use the raw in-context history, with no external belief module by default.

The POMDP framework covers both fully observed games (when 𝑍 preserves all task-relevant state) and partially observed ones where history matters. A game is Markov when the current observation suffices for optimal play: 𝜋*(𝑎𝑡 | 𝑜𝑡) = 𝜋*(𝑎𝑡 | ℎ𝑡) , ∀ℎ𝑡. Fully observed board games such as chess and Go, when encoded with all rule-relevant variables, are typical examples. A game is non-Markov when the current observation alone is not sufficient to determine the optimal action. Equivalently, two different histories can lead to the

##### RNGBench: Two Non-Markov Game Settings

[Figure 19]

Reconstructed Internal Belief State

[Figure 20]

[Figure 21]

 + 

Next New Observation

[Figure 22]

Episode History

Take Action

###### Both environments isolate memory for action under controlled settings.

##### 3D Maze

###### Matching Pairs

Dynamic, Spatial Hidden State

Static, Discrete Hidden State

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Action Hidden State

###### Action HiddenState

[Figure 29]

###### Optimal Play

[Figure 30]

###### Optimal Play

###### Observation

###### Observation

Pos 1 Pos 2 Pos 3

[Figure 31]

[Figure 32]

- 1）Move: Forward Until Intersection
- 2）Turn: Choose Turn Left Or Right

One Turn Two Flip: Not matching, Automatic Flip Back.

[Figure 33]

[Figure 34]

[Figure 35]

 Reconstruct map and pose.

 Exploit known pairs.

[Figure 36]

[Figure 37]

 Avoid revisits and wall hits.

 Explore Only when no match is known.

[Figure 38]

 Move toward goal or unexplored frontier.

[Figure 39]

Matching, They will be removed from board.

Have Ego-View of Maze

 Avoid redundant re-flips.

[Figure 40]

Different Position, Different orientations， Different View

Global maze layout, visited cells, position, and orientation

Current Revealed Card Identities (And Pos)

Locations and Identities of Observed Pairs

[Figure 41]

###### External Memory

Pattern Variants Pattern Variants

###### Duel Mode

###### Modality

###### Size Variants

[Figure 42]

Same Board, Share Observation One Fail Another Chance!

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

P1 P2

Length and width Texture change arbitrarily

No Map: Internal Memory. MiniMap: See visited Path.

Poster Color Texture Plain

Text 2D Patch 3D View

Noise Poker

[Figure 55]

[Figure 56]

###### Eval Metrics Score (%)

###### Eval Metrics Game Score

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

###### Explore(%)

###### Duel Win(%) / Elo

###### Resp / Score

###### Eff(%)

###### PF(%) / IA(%)

###### Wall Hits(%)

Fraction of matched Pairs

Measures success, efficiency, and exploration

Responses per matched pair

Optimal path / actual path

Ratio of maze cells visited

Action Parse Failure Invalid Action

Special In Duel Mode, Pairwise win-based rating

Wall collisions

- Figure 2: Two complementary environments for in-context state tracking. Matching Pairs tests static identity-location memory, while 3D Maze tests dynamic map construction from egocentric observations. Both use simple rules, scalable grids, and controllable visual settings to isolate belief-state tracking from other sources of difficulty.

same current observation but require different actions. Let 𝒜*(ℎ𝑡) denote the set of optimal actions under history ℎ𝑡. Then a non-Markov game satisfies:

∃ℎ𝑡,ℎ˜𝑡 s.t. 𝑍(𝑠𝑡) = 𝑍(˜𝑠𝑡), 𝒜*(ℎ𝑡) ̸= 𝒜*(ℎ˜𝑡).

(1)

To act well in such games, the agent must build a internal belief state 𝑏𝑡 = 𝑓(ℎ𝑡) from its interaction history. This state is an internal summary of the hidden, task-relevant information that is no longer directly visible. If the internal belief state is accurate, the agent can act as if the relevant state were observable. If it is inaccurate or incomplete, the agent may revisit known states, repeat ineffective actions, ignore useful past observations, and make locally plausible but globally wrong decisions.

Our benchmark is designed to test this ability: whether a model can maintain an accurate internal belief state within its in-context episode history and use that belief to choose the next action. We refer to this ability as in-context state tracking for action.

#### 3.2. Why These Environments

We choose Matching Pairs and 3D Maze because they are simple but diagnostic POMDP instances: the optimal action depends on information that appeared earlier in the episode but is no longer visible, so the model must reconstruct a belief state from its in-context history. The two games stress different hidden states (Fig. 2). Matching Pairs focuses on static, discrete, factual state: card identities and positions are fixed, but each card is only briefly revealed, so the agent must remember which identity appeared at which location. 3D Maze focuses on dynamic, spatial, structural state: position, orientation, visited cells, and local topology must be updated incrementally from egocentric views.

The two environments span three observation modalities (text, 2D image, and 3D rendering). The rules are simple, so failures reflect belief-state tracking rather than rule misunderstanding. Difficulty scales with grid size, increasing the hidden-state load without changing the task definition. Full trajectory logs expose fine-grained diagnostics (repeated flips, revisited cells, wall collisions), and targeted interventions (oracle state injection, scratchpads, minimap) can selectively remove the memory requirement to localize the bottleneck. Both environments test a remember-to-act ability: the model must use reconstructed belief state for immediate

action, not merely recall information after the episode ends (remember-to-answer).

#### 3.3. Environment Construction

Matching Pairs. A rectangular grid of size 𝑅×𝐶 is populated with 𝑅×2𝐶 card pairs, each sharing a visual identity, all initially face-down. At each turn, the agent flips two cards. Matched cards are removed, while

unmatched cards are turned back over. The game ends when all pairs are matched or a fixed response budget is exhausted. The hidden state is the set of previously revealed but currently hidden identity-location bindings. An optimal agent should use these bindings to find known pairs and avoid redundant re-flips.

We systematically vary board size (configurable to arbitrary 𝑅×𝐶), observation modality (image with controlled token count vs. text), visual pattern (e.g., ASCII glyphs, poker suits, textures, noise patterns, ...), action feedback (explicit flip result vs. no feedback), CoT prompting (allowed vs. direct action only), and response budget to isolate the contribution of each factor.

3D Maze. The agent navigates from the top-left start to the bottom-right goal in a procedurally generated grid maze. It default receives only an egocentric first-person rendering and the dialogue history, with no top-down map. The hidden state consists of the maze topology, visited cells, current position, and facing direction, all must be maintained incrementally from local views. The action space is move_forward, turn_left and turn_right.

Mazes are generated with a loop rate of 0.15 (the fraction of extra openings added to the spanning tree), which introduces cycles that make simple wall-following less reliable. Each configuration is evaluated over five seeds with a step budget of max(80, 4×𝐿*), where 𝐿* is the shortest-path length. We vary maze size (5×5 to 15×15), minimap availability (converting the task to approximately Markov), ask-output prompting (externalizing the spatial belief), and history window (3/5/10 turns vs. full history). These settings let us test how spatial belief tracking changes with scale, external memory, explicit belief reporting, and context length.

#### 3.4. Duel Protocol for Matching Pairs

We introduce a duel protocol for Matching Pairs to compare models under the same hidden-state structure. Two models play on the same board with identical rules and take turns to flip cards. Each player observes the cards revealed by both itself and its opponent, but does not observe the opponent’s reasoning. A successful match grants an extra turn, while a non-match passes the turn to the opponent. The player that removes more pairs wins.

Duel mode offers three advantages over single-agent evaluation. First, it controls for board randomness because both models face the same card layout. Second, it tests whether a model can use information revealed by the opponent’s flips as part of its own belief state. Third, it gives a robustness ranking that complements single-agent scores. To control for first-mover bias, each pair of competitors plays the same board twice with swapped turn order. The aggregated duel result is used as a complementary robustness check for belief-state tracking under shared observations.

#### 3.5. Evaluation Metrics

Each environment has its own primary completion metric, supplemented by trajectory-level diagnostics that capture how failures occur. For Matching Pairs, we report Score%, the fraction of pairs matched in each game, and Resp./Score, the average number of responses needed per matched pair, where lower is better. Parse failures and invalid actions are reported as additional diagnostics. For 3D Maze, we report Success Rate (SR), the fraction of episodes that reach the goal within the budget, and Efficiency, defined as 𝐿*/𝐿actual over successful episodes, where 𝐿* is the shortest-path length and 𝐿actual is the executed path length. We also report Explore, the ratio of visited cells, and Walls, the number of wall collisions. The primary scalar metric is Game Score (GS):

SR + SR×Eff + (1−SR)×Explore 2

GS =

, (2)

GS rewards successful completion, gives an efficiency bonus for successful episodes, and assigns partial credit for exploration when the agent fails to reach the goal. Its value lies in [0,1].

Matching Pairs 10×10 3D Maze 13×13

Model Name

PF%↓ IA%↓ Resp./Score↓ Score%↑ SR%↑ Explore%↑ Walls↓ Eff.%↑ GS%↑

GPT-5.4 0.0 4.3 8.01 62.3 20.0 32.3 3.2 75.7 30.5 Gemini-3.1-Pro 0.4 2.5 10.00 50.0 50.0 36.4 0.1 62.5 49.7 Seed-2.0-Lite 1.2 4.3 11.57 43.2 20.0 19.4 16.6 38.9 21.7 Kimi-K2.5 1.8 2.8 13.16 38.0 10.0 17.9 7.1 61.1 16.1 Qwen3.5-397B 0.0 3.0 19.74 25.3 0.0 21.0 9.9 0.0 10.5

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

- Table 2: Main results on the two environments (single-player setting). We focus on two metrics: Score, the fraction of matched pairs on Matching Pairs, and GS, the aggregate 3D Maze score combining success rate, efficiency, and exploration. The remaining columns support analysis: PF/IA means parse-failure and invalid-action rates; Resp./Score reports responses per matched pair; SR, Explore, Walls, and Eff refers to success rate, exploration coverage, wall collisions, and path efficiency.

Model Name Win% W T L Score% ELO

[Figure 66]

Gemini-3.1-Pro 100.0 16 0 0 36.5 1803 GPT-5.4 50.0 7 2 7 25.3 1492 Qwen3.5-397B 46.7 7 1 8 18.0 1476 Kimi-K2.5 37.5 5 2 9 18.0 1423 Seed-2.0-Lite 15.6 2 1 13 12.3 1306

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

- Table 3: Main results on Matching Pairs (duel setting). Each model plays 16 games against the other four, aggregating both player orders over two board seeds.

Memory Gap. To separate belief-state tracking from action selection, we define an oracle condition that provides the true hidden state 𝑠𝑡 into the prompt at each step. Let 𝑆(𝑚) and 𝑆*(𝑚) denote the score of model 𝑚 under the normal and oracle conditions, respectively. Depending on the setting, 𝑆 can be SR, Score%, or Efficiency. The Memory Gap is defined as

MemoryGap(𝑚) = (︂1 −

)︂ × 100%. (3)

𝑆(𝑚) 𝑆*(𝑚)

A large Memory Gap points to internal belief state reconstruction as the main bottleneck, while a small gap suggests errors in action selection, rule understanding, or perception.

### 4. Experiments

All models are evaluated under a unified harness built on top of VLMEvalKit [19], ensuring identical prompts, parsing, and metric computation across models. We first report main results, followed by diagnostic analyses (hidden-state scale, external memory, observation modality, and action-feedback text) to pinpoint where belief-state tracking degrades. Additional analyses appear in the appendix.

#### 4.1. Main Results

Single-player setting. Tab. 2 reports the main results on both environments. Matching Pairs uses a 10×10 board (50 pairs) with image observations and the noise card theme. GPT-5.4 leads at 62.3% with the lowest response cost per matched pair (8.01). Gemini-3.1-Pro follows at 50.0%, and Qwen3.5-397B reaches 25.3%. Parse failures (PF%) and invalid actions (IA%) stay below 5% across models, so the gaps are not explained by output-format compliance.

The ranking changes on 3D Maze. We evaluate on 13×13 mazes with no minimap and a mean optimal path length of 60.0 steps. Gemini-3.1-Pro obtains the highest success rate (50.0%) and game score (49.7%),

###### Matching Pairs

###### 3D Maze

90.6%

100

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |0.<br><br>|7%<br><br>| |
| | | | | | | | | | | | |

| | | |66.|7| | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | |.7|
| | | | | | |19| |
| | | | | | | | |
| | | | | | | | |

GameScore%

75

Explore%

40

Score%

50

50

20

25

0

0

0

5×57×79×911×1113×1315×15

4×44×66×66×88×88×1010×1010×1212×1212×14

Maze Size

Board Size

- Figure 3: Hidden-state scale sweep. Left: Matching Pairs Score% from 4×4 to 12×14. Right: 3D Maze Game Score and Explore% from 5×5 to 15×15. Performance drops sharply in both environments as the hidden state grows, pointing to belief-state maintenance rather than rule comprehension as the bottleneck.

###### Matching Pairs

###### 3D Maze

Baseline

+MemMap

Baseline

+Minimap

| |
|---|

| |
|---|

GameScore%

78.7 80.3

50

80

Score%

40.2

40

35.6

60

30

43.3

23.8 24.6

MG 46.1

MG 51.3

38.3

40

MG 40.8

MG 30.9

20

20

10

0

0

Qwen3.5-397B Kimi-K2.5

Qwen3.5-397B Kimi-K2.5

Figure 4: External-memory intervention. Matching Pairs and 3D Maze with and without a memory map or minimap. Red arrows show MemGap in Eq. 3. External memory doubles Matching Pairs but recovers a smaller share on 3D Maze, indicating spatial navigation couples hidden-state tracking with action planning.

whereas GPT-5.4 reaches 20.0% SR and 30.5% GS despite leading on Matching Pairs. Seed-2.0-Lite matches GPT-5.4’s SR but trails in GS, while Kimi-K2.5 and Qwen3.5-397B remain lower. This suggests that the two tasks stress different forms of hidden-state tracking: identity retention and pairwise retrieval in Matching Pairs, versus spatial belief updating and route planning in 3D Maze.

Duel setting. The duel mode is a complementary evaluation for Matching Pairs: two models alternate turns on the same board, and each player’s flips expose card identities the opponent can exploit. Unlike the single-player setting, this tests whether a model can track both its own flips and the cards revealed by another model.

Tab. 3 reports head-to-head results across five models. Gemini-3.1-Pro wins all 16 games and tops the Elo ranking. GPT-5.4 and Qwen3.5-397B post similar win rates (50% and 47%), but GPT-5.4 averages more matched pairs per game (10.1 vs. 7.2) and a slightly higher Elo. Kimi-K2.5 [71] wins 38% of games, while Seed-2.0-Lite [7] wins 16%. The duel ranking partially differs from the single-player order. GPT-5.4 drops from first to second and Gemini-3.1-Pro rises from second to first. Because Win rate (%) in Tab. 3 aggregates over swapped player orders, the shift is unlikely to be a first- or second-mover bias. Instead, Gemini appears better at using card identities revealed by the opponent and converting them into consecutive matching turns. This strategic advantage helps explain its top duel performance and suggests stronger perception and retention of image-based information over long interaction histories.

#### 4.2. Diagnostic Analysis

Performance drops sharply as the hidden state grows. We test whether performance degrades as the hidden state grows while task rules stay fixed. In Matching Pairs, the hidden state scales with board size since the model must retain more card identities and locations; in 3D Maze, it scales with maze size and path complexity, requiring a larger spatial belief state over a longer action history. Fig. 3 shows a clear scale effect for Qwen3.5-397B in both environments: Matching Pairs Score% drops from 90.6% on 4×4 to 0.7% on 12×12, while 3D Maze Game Score peaks at 7×7 and then declines from 9×9 onward, reaching 0.197 at 15×15. The parallel drop in Explore% indicates that larger mazes degrade not only task success but also state-space coverage. The model can follow the rules at small scale, but its belief-state maintenance becomes unreliable as the latent state grows.

External memory recovers most of the gap on Matching Pairs but only part of it on 3D Maze. We test whether providing explicit external memory repairs hidden-state tracking. In Matching Pairs the intervention is a memory map of known cards; in 3D Maze it is a minimap with the visited-state summary. Fig. 4 reports both, annotated with our MemGap metric (Eq. 3), which quantifies the residual deficit after external memory is supplied. On Matching Pairs, Qwen3.5-397B and Kimi-K2.5 roughly double their Score% with the memory map, yielding MemGap values of 51.3 and 46.1. On 3D Maze the minimap closes a smaller share of the gap (MemGap 40.8 and 30.9). Kimi-K2.5’s MemGap is uniformly lower than Qwen3.5-397B’s, indicating that external memory helps Kimi-K2.5 less relative to its baseline. These results localize the Matching Pairs bottleneck primarily to hidden-state maintenance, while pointing to additional limitations beyond memory access in 3D Maze.

Matching Pairs 3D Maze

Model Name

Modality Score%↑ Modality Eff.%↑ Explore%↑ GS%↑

Text 100.0 Text Symbolic 67.2 41.0 70.9 Image-ASCII 75.8 2D Local Patch 27.5 18.0 20.1 Image-Noise 38.3 3D Scene 21.9 29.0 23.7

Qwen3.5-397B

[Figure 71]

Text 100.0 Text Symbolic 74.9 38.0 60.0 Image-ASCII 72.5 2D Local Patch 87.5 17.0 25.7 Image-Noise 43.3 3D Scene 62.5 24.0 39.7

Kimi-K2.5

[Figure 72]

- Table 4: Modality ablation. Matching Pairs compares symbolic text, ASCII-style, and noise-pattern image cards; 3D Maze compares text-symbolic, 2D local patches, and 3D first-person views. Text-only modality dominate, indicating that visual recognition limits hidden-state tracking.

Board

[Figure 73]

GPT-5.4 Qwen3.5-397B

[Figure 74]

w/Act. w/oAct. Δrel (%) w/Act. w/oAct. Δrel (%)

8×10 69.6 15.0 −78.4 29.4 9.0 −69.4 10×10 62.3 15.3 −75.4 25.3 6.3 −75.1

- Table 5: Effect of removing the model’s action history in Matching Pairs. “w/Act.” keeps the action trace in context; “w/oAct.” keeps only the board images.

Matching Pairs 3D Maze

Model Name

Score%↑ Resp./Score↓ SR%↑ GS%↑

Qwen3.5-9B 0.0 – 0.0 1.5 with opt32k 14.6 14.7 0.0 5.0 with rmix32k 29.5 6.8 10.0 16.3

[Figure 75]

Table 6: Held-out scale evaluation of fine-tuned Qwen3.5-9B. Evaluation sizes are strictly larger than the training data pool.

Visual observations, not just memory length, drive the bottleneck. We disentangle visual perception from history-to-state tracking by varying the observation modality while holding the hidden state fixed. Matching Pairs compares symbolic text, ASCII-style image cards, and noise-pattern image cards; 3D Maze compares text-symbolic, 2D local patches, and 3D first-person views. Tab. 4 reports both environments side by side. Both Qwen3.5-397B and Kimi-K2.5 solve Matching Pairs perfectly under text but fall to 38.3% and 43.3% under noise-pattern images. The 3D Maze shows the same ordering, with text-symbolic Game Scores far above the 2D patch and 3D scene settings. That text-only configurations remain strong while image settings collapse indicates that visual recognition, not history length alone, limits hidden-state tracking in these models. Per-pair duel results on Matching Pairs are reported in Appendix. B

Removing the action-feedback text collapses Matching Pairs to near-chance. We test whether the model’s own action history is necessary when the rendered board already carries every visual change. Tab. 5 strips the model’s previous actions from the conversation history on Matching Pairs (image-noise), keeping only the sequence of board images; the model must re-infer which cells it flipped from the visual diff between consecutive boards. The effect is consistent across a leading closed-source and a leading open-source model: GPT-5.4’s Score% falls by roughly 75% on both 8×10 and 10×10 (69.6→15.0, 62.3→15.3), and Qwen3.5-397B falls by about 70–75% from a lower base. Parse failures and invalid actions remain at zero, so the collapse is not an output-format issue. Board images are in principle sufficient to recover each flip, yet neither model can close the loop from observation to belief update without an explicit textual record of its own actions. The action trace functions as a load-bearing channel for belief-state tracking rather than redundant decoration.

More Analysis. Additional ablations covering text-vs.-image duel observations, visual-pattern distinctiveness, the 3D Maze minimap, ask-output map externalization, and bounded-history (Markov-control) settings, together with trajectory visualizations, appear in Appendix B.

### 5. Training with Non-Markov Trajectories

Because RNG-Bench is driven by two simulators rather than a fixed test set, we can roll out fresh trajectories with known optimal actions and use them as supervision. This section asks whether SFT on such rollouts teaches a smaller MLLM to act on prior observations, and is organized as: a data recipe (Sec. 5), generalization to held-out board and maze sizes (Tab. 6), and transfer to external memory and spatial benchmarks (Tab. 7).

Setup. We fine-tune Qwen3.5-9B with supervised fine-tuning on action tokens, masking the loss over observation tokens. Training trajectories are drawn from Matching Pairs boards of size 2×4 to 8×8 and 3D mazes of size 5×5 to 9×9, while all evaluation episodes use strictly larger sizes and disjoint seeds, so no training instance recurs at test time.

Data construction. The optimal pool is rule-based: a hand-coded oracle solves each instance in closed form, so trajectories are generated without any model, the pool scales with the number of sampled episodes, and we use 32K. The rollout pool is harvested by running larger MLLMs (Qwen3.5-397B, Kimi-K2.5) on RNG-Bench episodes and keeping only trajectories that solve the task. The correctness filter discards most rollouts, and we cap this pool at 6K. Because 6K alone is insufficient for standalone SFT in our setting, we use the rollout pool as an augmentation rather than a replacement: we compare opt32k (32K optimal) against rmix32k (26K optimal plus the 6K rollouts) at a fixed budget of 32K trajectories, so any gap isolates what the rollout component adds over an equivalently sized optimal-only baseline.

Held-out scale generalization. Tab. 6 reports Qwen3.5-9B on board and maze sizes strictly larger than the training pool. The optimal pool alone (opt32k) lifts Matching Pairs from 0.0 to 14.6 and 3D Maze from 1.5 to 5.0, indicating that the supervision transfers to unseen sizes. Adding 6K rollouts (rmix32k) further raises Score% to 29.5, halves response cost per matched pair, and yields the only non-zero maze SR. A plausible explanation is that the oracle pool, being mistake-free, lacks recovery states that an imperfect policy must reach; the rollout pool supplies such states.

Benchmark Metric Baseline SFT Δ Memory- and spatial-reasoning benchmarks

EMeMBench Visual DIF_50 49.5 54.7 +5.2 VGRPBench Macro Perception Acc 24.9 29.5 +4.6 MMSIBench_circular Overall 7.4 9.7 +2.3 ViewSpatialBench Overall 41.9 43.4 +1.5

Group mean 30.9 34.3 +3.4 General multimodal benchmarks

OCRBench Final Score Norm 85.9 89.2 +3.3 MMBench_DEV_EN_V11 Overall 85.8 86.8 +1.0 NaturalBenchDataset Acc 79.7 80.2 +0.5 AI2D_TEST Overall 84.6 85.0 +0.4 CV-Bench-3D Overall 92.7 93.1 +0.4 ERQA Overall 40.3 40.5 +0.2 MMStar Overall 73.3 72.7 -0.6 RealWorldQA Overall 76.6 75.6 -1.0

Group mean 77.4 77.9 +0.5

- Table 7: Results on external benchmarks. Scores for Qwen3.5-9B before (Baseline) and after (SFT) finetuning on game environment rollouts. Benchmarks are split into a memory/spatial-reasoning group and a general group. ∆ is SFT minus baseline.

External-benchmark transfer. We evaluate the same fine-tuned Qwen3.5-9B on an external suite split into a targeted group of memory and spatial-reasoning benchmarks and a general group covering broader multimodal abilities (Tab. 7). All four targeted benchmarks improve, with a group-mean gain of +3.4 and the largest deltas on EMeMBench (+5.2). On the general group the mean shifts by +0.5. The pattern is consistent with fine-tuning lifting the targeted capabilities without large regressions on general multimodal performance.

### 6. Conclusion

We introduced RNG-Bench, a controllable benchmark for non-Markov games that isolates in-context beliefstate tracking from rule understanding and perception. Across leading MLLMs, performance collapses as the latent state grows, image observations drive the bottleneck more than history length, and stripping the action trace alone reduces Matching Pairs to near-chance. Analyses localize the failure to belief-state maintenance and provide a testbed for interactive MLLMs.

### Limitations

RNG-Bench focuses on two environments (Matching Pairs and 3D Maze) chosen for their controllable hidden state; broader coverage of game genres, model families, and visual styles is left to future work. In the image settings, hidden-state tracking is observed through the model’s perceptual interface, and the Memory Gap metric is intended as a practical diagnostic under our oracle interface rather than a standalone causal decomposition. Our fine-tuning study is a feasibility demonstration on a single base model family.

### Ethics Statement

This work studies hidden-state tracking in controlled game environments. All visual observations are synthetically generated by our code. We do not collect human-subject data or include any personal or sensitive information, and the generated images do not depict real people or real-world copyrighted content.

### References

- [1] Qingyao Ai, Yichen Tang, Changyue Wang, Jianming Long, Weihang Su, and Yiqun Liu. Memorybench: A benchmark for memory and continual learning in llm systems. arXiv preprint arXiv:2510.17281, 2025.
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.
- [3] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14388–14411, 2024.
- [4] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119–3137, 2024.
- [5] Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3509–3532, 2025.
- [6] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. Ms marco: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268, 2016.
- [7] ByteDance Seed. Seed-2.0-Lite-260428: Omni-modal understanding across video, image, audio, and text, April 2026. URL https://seed.bytedance.com/en/seed2.
- [8] Kranti Chalamalasetti, Jana Götze, Sherzod Hakimov, Brielen Madureira, Philipp Sadler, and David Schlangen. clembench: Using game play to evaluate chat-optimized language models as conversational agents. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 11174–11219, 2023.
- [9] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. SpatialVLM: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [10] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, and Jiaqi Wang. ShareGPT4Video: Improving video understanding and generation with better captions. Advances in Neural Information Processing Systems, 37:19472–19495, 2024.

- [11] Yang Chen, Minghao Liu, Yufan Shen, Yunwen Li, Tianyuan Huang, Xinyu Fang, Tianyu Zheng, Wenxuan Huang, Cheng Yang, Daocheng Fu, et al. Iwr-bench: Can lvlms reconstruct interactive webpage from a user interaction video? arXiv preprint arXiv:2509.24709, 2025.
- [12] Anthony Costarelli, Mat Allen, Roman Hauksson, Grace Sodunke, Suhas Hariharan, Carlson Cheng, Wenjie Li, and Arjun Yadav. Gamebench: Evaluating strategic reasoning abilities of llm agents. arXiv preprint arXiv:2406.06613, 2024.
- [13] Xuanlang Dai, Yujie Zhou, Long Xing, Jiazi Bu, Xilin Wei, Yuhong Liu, Beichen Zhang, Kai Chen, and Yuhang Zang. EndoCoT: Scaling endogenous chain-of-thought reasoning in diffusion models. arXiv preprint arXiv:2603.12252, 2026.
- [14] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36, 2023.
- [15] Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. MM-IFEngine: Towards multimodal instruction following. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.
- [16] Shengyuan Ding, Xinyu Fang, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiangyu Zhao, Haodong Duan, Xiaoyi Dong, Jiaqi Liang, et al. ARM-Thinker: Reinforcing multimodal generative reward models with agentic tool use and visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.
- [17] Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Yang JingYi, Penghui Yang, Zhixiong Zhang, Xilin Wei, Xinyu Fang, et al. WildClawBench: A benchmark for real-world, long-horizon agent evaluation. arXiv preprint arXiv:2605.10912, 2026.
- [18] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, and Jiaqi Wang. InternLMXComposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.
- [19] Haodong Duan, Xinyu Fang, Junming Yang, Xiangyu Zhao, Yuxuan Qiao, Mo Li, Amit Agarwal, Zhe Chen, Lin Chen, Yuan Liu, Yubo Ma, Hailong Sun, Yifan Zhang, Shiyin Lu, Tack Hwa Wong, Weiyun Wang, Peiheng Zhou, Xiaozhe Li, Chaoyou Fu, Junbo Cui, Jixuan Chen, Enxin Song, Song Mao, Shengyuan Ding, Tianhao Liang, Zicheng Zhang, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. VLMEvalKit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 11198–11201, 2024.
- [20] Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias Stengel-Eskin, Mohit Bansal, Tianlong Chen, and Kaidi Xu. GTBench: Uncovering the strategic reasoning limitations of LLMs via game-theoretic evaluations. arXiv preprint arXiv:2402.12348, 2024.
- [21] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbenchvideo: A long-form multi-shot benchmark for holistic video understanding. Advances in Neural Information Processing Systems, 37:89098–89124, 2024.
- [22] Xinyu Fang, Zhijian Chen, Kai Lan, Lixin Ma, Shengyuan Ding, Yingji Liang, Xiangyu Zhao, Farong Wen, Zicheng Zhang, Guofeng Zhang, Haodong Duan, Kai Chen, and Dahua Lin. Creation-mmbench: Assessing context-aware creative intelligence in mllms. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 447–456, October 2025.
- [23] Google DeepMind. Gemini 3.1 Pro model card, February 2026. URL https://deepmind.google/ models/model-cards/gemini-3-1-pro/.
- [24] Leon Guertler, Bobby Cheng, Simon Yu, Bo Liu, Leshem Choshen, and Cheston Tan. TextArena. arXiv preprint arXiv:2504.11442, 2025.

- [25] Daya Guo, Dejian Yang, Haowei Zhang, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [26] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.
- [27] Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P Xing, Ion Stoica, Tajana Rosing, Haojian Jin, and Hao Zhang. lmgame-bench: How good are llms at playing games? arXiv preprint arXiv:2505.15146, 2025.
- [28] Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in llm agents via incremental multi-turn interactions. arXiv preprint arXiv:2507.05257, 2025.
- [29] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [30] Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.
- [31] Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1–2):99–134, 1998.
- [32] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems, volume 35, pages 22199–22213, 2022.
- [33] Wai-Chung Kwan, Xingshan Zeng, Yufei Wang, Yusen Sun, Liangyou Li, Yuxin Jiang, Lifeng Shang, Qun Liu, and Kam-Fai Wong. M4le: A multi-ability multi-range multi-task multi-domain long-context evaluation benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15568–15592, 2024.
- [34] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.
- [35] Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, and Dahua Lin. Beyond fixed: Trainingfree variable-length denoising for diffusion large language models. arXiv preprint arXiv:2508.00819, 2025.
- [36] Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, and Dahua Lin. Visual self-refine: A pixel-guided paradigm for accurate chart parsing. arXiv preprint arXiv:2602.16455, 2026.
- [37] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. VideoChat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [38] Xiaozhe Li, Jixuan Chen, Xinyu Fang, Shengyuan Ding, Haodong Duan, Qingwen Liu, and Kai Chen. OPT-Bench: Evaluating LLM agent on large-scale search spaces optimization problems. arXiv preprint arXiv:2506.10764, 2025.
- [39] Xiaozhe Li, Xinyu Fang, Shengyuan Ding, Linyang Li, Haodong Duan, Qingwen Liu, and Kai Chen. NP-Engine: Empowering optimization reasoning in large language models with verifiable synthetic NP problems. arXiv preprint arXiv:2510.16476, 2025.
- [40] Xinze Li, Ziyue Zhu, Siyuan Liu, Yubo Ma, Yuhang Zang, Yixin Cao, and Aixin Sun. Emembench: Interactive benchmarking of episodic memory for vlm agents. arXiv preprint arXiv:2601.16690, 2026.

- [41] Jonathan Light, Min Cai, Sheng Shen, and Ziniu Hu. AvalonBench: Evaluating LLMs playing the game of Avalon. arXiv preprint arXiv:2310.05036, 2023.
- [42] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, 2023.
- [43] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024.
- [44] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, 2024.
- [45] Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. Spatial-SSRL: Enhancing spatial understanding via self-supervised reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.
- [46] Zihan Liu, Zhikang Niu, Qiuyang Xiao, Zhisheng Zheng, Ruoqi Yuan, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Jianze Liang, Xie Chen, Leilei Sun, Dahua Lin, and Jiaqi Wang. STAR-Bench: Probing deep spatiotemporal reasoning as audio 4d intelligence. In International Conference on Learning Representations (ICLR), 2026.
- [47] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, and Jiaqi Wang. MMDU: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for LVLMs. Advances in Neural Information Processing Systems, 37: 8698–8733, 2024.
- [48] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-RFT: Visual reinforcement fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2034–2044, 2025.
- [49] Ziyu Liu, Yuhang Zang, Shengyuan Ding, Yuhang Cao, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. SPARK: Synergistic policy and reward co-evolving framework. arXiv preprint arXiv:2509.22624, 2025.
- [50] Ziyu Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan, Conghui He, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. MIA-DPO: Multi-image augmented direct preference optimization for large vision-language models. In International Conference on Learning Representations (ICLR), 2025.
- [51] Ziyu Liu, Shengyuan Ding, Xinyu Fang, Xuanlang Dai, Penghui Yang, Jianze Liang, Jiaqi Wang, Kai Chen, Dahua Lin, and Yuhang Zang. Visual-ERM: Reward modeling for visual equivalence. arXiv preprint arXiv:2603.13224, 2026.
- [52] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems, 37:95963–96010, 2024.
- [53] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, 2024.
- [54] Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun Yoon, and Hinrich Schütze. Nolima: Long-context evaluation beyond literal matching. In International Conference on Machine Learning, 2025.
- [55] Shen Nie, Fengqi Zhu, Zebin You, et al. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

- [56] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114, 2021.
- [57] OpenAI. Introducing gpt-5.4, March 2026. URL https://openai.com/index/ introducing-gpt-5-4/.
- [58] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [59] Mingyu Ouyang, Siyuan Hu, Kevin Qinghong Lin, Hwee Tou Ng, and Mike Zheng Shou. Gameworld: Towards standardized and verifiable evaluation of multimodal game agents. arXiv preprint arXiv:2604.07429, 2026.
- [60] Davide Paglieri, Bartlomiej Cupial, Samuel Coward, Ulyana Piterbarg, Maciej Wolczyk, Akbir Khan, Eduardo Pignatelli, Lukasz Kucinski, Lerrel Pinto, Rob Fergus, Jakob Nicolaus Foerster, Jack ParkerHolder, and Tim Rocktaschel. Balrog: Benchmarking agentic llm and vlm reasoning on games. arXiv preprint arXiv:2411.13543, 2024.
- [61] Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li, Steven Basart, Thomas Woodside, Jonathan Ng, Hanlin Zhang, Scott Emmons, and Dan Hendrycks. Do the rewards justify the means? Measuring trade-offs between rewards and ethical behavior in the MACHIAVELLI benchmark. In International Conference on Machine Learning, 2023.
- [62] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. Kilt: a benchmark for knowledge intensive language tasks. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2523–2544, 2021.
- [63] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5.
- [64] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- [65] Yufan Ren, Konstantinos Tertikas, Shalini Maiti, Junlin Han, Tong Zhang, Sabine Süsstrunk, and Filippos Kokkinos. Vgrp-bench: Visual grid reasoning puzzle benchmark for large vision-language models. arXiv preprint arXiv:2503.23064, 2025.
- [66] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2023.
- [67] Alina Shutova, Alexandra Olenina, Ivan Vinogradov, and Anton Sinitsin. Evaluating memory structure in llm agents. arXiv preprint arXiv:2602.11243, 2026.
- [68] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419):1140–1144, 2018.
- [69] Zeyi Sun, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Tong Wu, Dahua Lin, and Jiaqi Wang. SEAgent: Self-evolving computer use agent with autonomous learning from experience. arXiv preprint arXiv:2508.04700, 2025.
- [70] Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, and Zhenhua Dong. Membench: Towards more comprehensive evaluation on the memory of llm-based agents. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19336–19352, 2025.

- [71] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S. H. Cai, Yuan Cao, Y. Charles, H. S. Che, Cheng Chen, Guanduo Chen, Huarong Chen, Jia Chen, Jiahao Chen, Jianlong Chen, Jun Chen, Kefan Chen, Liang Chen, Ruijue Chen, Xinhao Chen, Yanru Chen, Yanxu Chen, Yicun Chen, Yimin Chen, Yingjiang Chen, Yuankun Chen, Yujie Chen, Yutian Chen, Zhirong Chen, Ziwei Chen, Dazhi Cheng, Minghan Chu, Jialei Cui, Jiaqi Deng, Muxi Diao, Hao Ding, Mengfan Dong, Mengnan Dong, Yuxin Dong, Yuhao Dong, Angang Du, Chenzhuang Du, Dikang Du, Lingxiao Du, Yulun Du, Yu Fan, Shengjun Fang, Qiulin Feng, Yichen Feng, Garimugai Fu, Kelin Fu, Hongcheng Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Chengyang Gong, Xiaochen Gong, Zhuoma Gongque, Qizheng Gu, Xinran Gu, Yicheng Gu, Longyu Guan, Yuanying Guo, Xiaoru Hao, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Jiaxi Hu, Yangyang Hu, Zhenxing Hu, Ke Huang, Ruiyuan Huang, Weixiao Huang, Zhiqi Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yu Jing, Guokun Lai, Aidi Li, C. Li, Cheng Li, Fang Li, Guanghe Li, Guanyu Li, Haitao Li, Haoyang Li, Jia Li, Jingwei Li, Junxiong Li, Lincan Li, Mo Li, Weihong Li, Wentao Li, Xinhang Li, Xinhao Li, Yang Li, Yanhao Li, Yiwei Li, Yuxiao Li, Zhaowei Li, Zheming Li, Weilong Liao, Jiawei Lin, Xiaohan Lin, Zhishan Lin, Zichao Lin, Cheng Liu, Chenyu Liu, Hongzhang Liu, Liang Liu, Shaowei Liu, Shudong Liu, Shuran Liu, Tianwei Liu, Tianyu Liu, Weizhou Liu, Xiangyan Liu, Yangyang Liu, Yanming Liu, Yibo Liu, Yuanxin Liu, Yue Liu, Zhengying Liu, Zhongnuo Liu, Enzhe Lu, Haoyu Lu, Zhiyuan Lu, Junyu Luo, Tongxu Luo, Yashuo Luo, Long Ma, Yingwei Ma, Shaoguang Mao, Yuan Mei, Xin Men, Fanqing Meng, Zhiyong Meng, Yibo Miao, Minqing Ni, Kun Ouyang, Siyuan Pan, Bo Pang, Yuchao Qian, Ruoyu Qin, Zeyu Qin, Jiezhong Qiu, Bowen Qu, Zeyu Shang, Youbo Shao, Tianxiao Shen, Zhennan Shen, Juanfeng Shi, Lidong Shi, Shengyuan Shi, Feifan Song, Pengwei Song, Tianhui Song, Xiaoxi Song, Hongjin Su, Jianlin Su, Zhaochen Su, Lin Sui, Jinsong Sun, Junyao Sun, Tongyu Sun, Flood Sung, Yunpeng Tai, Chuning Tang, Heyi Tang, Xiaojuan Tang, Zhengyang Tang, Jiawen Tao, Shiyuan Teng, Chaoran Tian, Pengfei Tian, Ao Wang, Bowen Wang, Chensi Wang, Chuang Wang, Congcong Wang, Dingkun Wang, Dinglu Wang, Dongliang Wang, Feng Wang, Hailong Wang, Haiming Wang, Hengzhi Wang, Huaqing Wang, Hui Wang, Jiahao Wang, Jinhong Wang, Jiuzheng Wang, Kaixin Wang, Linian Wang, Qibin Wang, Shengjie Wang, Shuyi Wang, Si Wang, Wei Wang, Xiaochen Wang, Xinyuan Wang, Yao Wang, Yejie Wang, Yipu Wang, Yiqin Wang, Yucheng Wang, Yuzhi Wang, Zhaoji Wang, Zhaowei Wang, Zhengtao Wang, Zhexu Wang, Zihan Wang, Zizhe Wang, Chu Wei, Ming Wei, Chuan Wen, Zichen Wen, Chengjie Wu, Haoning Wu, Junyan Wu, Rucong Wu, Wenhao Wu, Yuefeng Wu, Yuhao Wu, Yuxin Wu, Zijian Wu, Chenjun Xiao, Jin Xie, Xiaotong Xie, Yuchong Xie, Yifei Xin, Bowei Xing, Boyu Xu, Jianfan Xu, Jing Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinbo Xu, Xinran Xu, Yangchuan Xu, Yichang Xu, Yuemeng Xu, Zelai Xu, Ziyao Xu, Junjie Yan, Yuzi Yan, Guangyao Yang, Hao Yang, Junwei Yang, Kai Yang, Ningyuan Yang, Ruihan Yang, Xiaofei Yang, Xinlong Yang, Ying Yang, Yi Yang, Yi Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Dan Ye, Wenjie Ye, Zhuorui Ye, Bohong Yin, Chengzhen Yu, Longhui Yu, Tao Yu, Tianxiang Yu, Enming Yuan, Mengjie Yuan, Xiaokun Yuan, Yang Yue, Weihao Zeng, Dunyuan Zha, Haobing Zhan, Dehao Zhang, Hao Zhang, Jin Zhang, Puqi Zhang, Qiao Zhang, Rui Zhang, Xiaobin Zhang, Y. Zhang, Yadong Zhang, Yangkun Zhang, Yichi Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yushun Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Chenguang Zhao, Feifan Zhao, Jinxiang Zhao, Shuai Zhao, Xiangyu Zhao, Yikai Zhao, Zijia Zhao, Huabin Zheng, Ruihan Zheng, Shaojie Zheng, Tengyang Zheng, Junfeng Zhong, Longguang Zhong, Weiming Zhong, M. Zhou, Runjie Zhou, Xinyu Zhou, Zaida Zhou, Jinguo Zhu, Liya Zhu, Xinhao Zhu, Yuxuan Zhu, Zhen Zhu, Jingze Zhuang, Weiyu Zhuang, Ying Zou, and Xinxing Zu. Kimi k2.5: Visual agentic intelligence, 2026. URL https://arxiv.org/abs/2602.02276.
- [72] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.
- [73] Chonghua Wang, Haodong Duan, Songyang Zhang, Dahua Lin, and Kai Chen. Ada-leval: Evaluating long-context llms with length-adaptable benchmarks. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3712–3724, 2024.
- [74] Hengyi Wang, Haizhou Shi, Shiwei Tan, Weiyi Qin, Wenyuan Wang, Tunyu Zhang, Akshay Nambi, Tanuja Ganu, and Hao Wang. Multimodal needle in a haystack: Benchmarking long-context capability of multimodal large language models. In Proceedings of the 2025 Conference of the Nations of the Americas

- Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3221–3241, 2025.
- [75] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837, 2022.
- [76] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Jian Tong, Haodong Duan, Qipeng Guo, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. VideoRoPE: What makes for good video rotary position embedding? In Proceedings of the 42nd International Conference on Machine Learning, pages 66118–66136, 2025.
- [77] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. SIM-CoT: Supervised implicit chain-of-thought. In International Conference on Learning Representations, 2026.
- [78] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813, 2024.
- [79] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37: 28828–28857, 2024.
- [80] Yue Wu, Xuan Tang, Tom M. Mitchell, and Yuanzhi Li. SmartPlay: A benchmark for LLMs as intelligent agents. In International Conference on Learning Representations, 2024.
- [81] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [82] Penghui Yang, Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Yibin Wang, Yujie Zhou, Jiazi Bu, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, and Dahua Lin. CapRL++: Unified reinforcement learning with verifiable rewards for dense image and video captioning. arXiv preprint arXiv:2606.09393, 2026.
- [83] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.
- [84] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.
- [85] Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. Helmet: How to evaluate long-context language models effectively and thoroughly. In International Conference on Learning Representations, 2025.
- [86] Tianyu Yu, Yuan Yao, Haoye Zhang, et al. RLHF-V: Towards trustworthy MLLMs via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [87] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [88] Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Ziyu Liu, Shengyuan Ding, Shenxi Wu, Yubo Ma, Haodong Duan, Wenwei Zhang, Kai Chen, Dahua Lin, and Jiaqi Wang. InternLM-XComposer2.5-Reward: A simple yet effective multi-modal reward model. In Findings of the Association for Computational Linguistics: ACL 2025, 2025.

- [89] Beichen Zhang, Kun Zhou, Xilin Wei, Xin Zhao, Jing Sha, Shijin Wang, and Ji-Rong Wen. Evaluating and improving tool-augmented computation-intensive math reasoning. Advances in Neural Information Processing Systems, 36:23570–23589, 2023.
- [90] Beichen Zhang, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Think visually, reason textually: Vision-language synergy in ARC. arXiv preprint arXiv:2511.15703, 2025.
- [91] Beichen Zhang, Yuhong Liu, Jinsong Li, Yuhang Zang, Jiaqi Wang, and Dahua Lin. ETCHR: Editing to clarify and harness reasoning. arXiv preprint arXiv:2605.23897, 2026.
- [92] Pan Zhang, Xiaoyi Dong, Yuhang Cao, Yuhang Zang, Rui Qian, Xilin Wei, Lin Chen, Yifei Li, Junbo Niu, Shuangrui Ding, Qipeng Guo, Haodong Duan, Xin Chen, Han Lv, Zheng Nie, Min Zhang, Bin Wang, Wenwei Zhang, Xinyue Zhang, Jiaye Ge, Wei Li, Jingwen Li, Zhongying Tu, Conghui He, Xingcheng Zhang, Kai Chen, Yu Qiao, Dahua Lin, and Jiaqi Wang. InternLM-XComposer2.5-OmniLive: A comprehensive multimodal system for long-term streaming video and audio interactions. arXiv preprint arXiv:2412.09596, 2024.
- [93] Zhixiong Zhang, Shuangrui Ding, Xiaoyi Dong, Songxin He, Jianfan Lin, Junsong Tang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. SeC: Advancing complex video object segmentation via progressive concept construction. In International Conference on Learning Representations (ICLR), 2026.
- [94] Zhixiong Zhang, Yizhuo Li, Shuangrui Ding, Yuhang Zang, Shengyuan Ding, Long Xing, Yibin Wang, Qiaosheng Zhang, and Jiaqi Wang. SetCon: Towards open-ended referring segmentation via set-level concept prediction. arXiv preprint arXiv:2605.20110, 2026.
- [95] Zicheng Zhang, Junying Wang, Farong Wen, Yijin Guo, Xiangyu Zhao, Xinyu Fang, Shengyuan Ding, Xuemei Zhou, Guangtao Zhai, et al. Large multimodal models evaluation: A survey. Science China Information Sciences, 68(12):221301, 2025.
- [96] Xiangyu Zhao, Shengyuan Ding, Zicheng Zhang, Haian Huang, Maosong Cao, Weiyun Wang, Jiaqi Wang, Xinyu Fang, Wenhai Wang, Guangtao Zhai, Haodong Duan, Hua Yang, and Kai Chen. OmniAlign-V: Towards enhanced alignment of MLLMs with human preference. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), pages 18490–18515, 2025.
- [97] Xiangyu Zhao, Peiyuan Zhang, Junming Lin, Tianhao Liang, Yuchen Duan, Shengyuan Ding, Changyao Tian, Yuhang Zang, Junchi Yan, and Xue Yang. Trust your critic: Robust reward modeling and reinforcement learning for faithful image editing and generation. arXiv preprint arXiv:2603.12247, 2026.
- [98] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.
- [99] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024.
- [100] Yicheng Zou, Dongsheng Zhu, Lin Zhu, Tong Zhu, Yunhua Zhou, Peiheng Zhou, Xinyu Zhou, Dongzhan Zhou, Zhiwang Zhou, Shengyuan Ding, et al. Intern-S1-Pro: Scientific multimodal foundation model at trillion scale. arXiv preprint arXiv:2603.25040, 2026.

### A. More Related Work

Memory Benchmarks. Memory benchmarks target the ability to retain, update, and organize information across conversations or sessions. LoCoMo and LongMemEval evaluate long-term conversational memory through question answering over dialogue histories [53, 78]. MemoryBench, MemBench, and MemoryAgentBench examine memory systems under continual feedback or multi-turn information accumulation [1, 28, 70]. StructMemEval studies whether agents can organize memories into useful structures [67]. EMemBench is especially close in spirit: it constructs trajectory-grounded questions from text and visual game interactions to evaluate episodic memory in VLM agents [40]. We share this focus on episodic and interaction-derived memory but shift the evaluation target from answering retrospective questions to choosing prospective actions. An agent may recall a past observation in isolation yet still fail to integrate it into the latent state needed for the next move.

Multimodal Understanding and Vision-Language Models. Our benchmark requires models to ground visual observations across multiple turns, a capability rooted in the rapid progress of vision-language modeling. Early few-shot multimodal learners such as Flamingo [2] demonstrated that large-scale pre-training can bridge vision and language, and visual instruction tuning [42] further scaled this paradigm to open-ended multimodal conversations. Recent vision-language systems have extended to richer interaction settings including free-form text-image composition, long-context streaming, and scientific reasoning [18, 92, 100]. On the video side, chatcentric and long-context video understanding has advanced rapidly [10, 11, 21, 37, 79], with parallel progress on complex video object segmentation [93] and on probing fine-grained spatio-temporal perception [46], and temporal positional encoding designs have been shown to be critical for reasoning over video frames [76]. Multi-turn, multi-image dialog understanding—the setting closest to our sequential observation regime—has been studied in several recent benchmarks and preference-tuning datasets [47, 50, 52], and fine-grained visual grounding has been pushed toward open-ended referring segmentation [94] and pixel-level chart parsing [36].

Reasoning, Planning, and Agent Evaluation. Chain-of-thought (CoT) prompting [32, 75] has become the standard approach for eliciting multi-step reasoning in LLMs. More recent work pushes reasoning into latent or non-autoregressive spaces [13, 35, 55, 56, 77], stabilizing implicit reasoning through step-level supervision or extending it to diffusion language models with variable-length denoising on structured generation tasks.

- A complementary line elicits visual reasoning by “thinking with images” and exploiting vision–language synergy [29, 90, 91]. On the tool-augmented side, language models have been shown to learn tool invocation autonomously [66, 89], and external computation can compensate for internal reasoning limitations—a theme echoed by our external-memory interventions. For agent evaluation, the synergy between reasoning and acting [84] has motivated a growing suite of benchmarks spanning web navigation [14, 99], general LLM-asagent tasks [44], computer-use environments [81], and real-world long-horizon agent evaluation [17], with agents that self-evolve by learning from their own interaction experience [69]. Closely related, optimizationoriented benchmarks evaluate LLM agents over large search spaces and verifiable synthetic problems [38, 39], providing controllable difficulty in a different domain from our games.

Evaluation, Alignment, and Reward Modeling of MLLMs. The rapid progress of multimodal models has been accompanied by a growing evaluation and alignment ecosystem [95]. Open-source toolkits and broad benchmarks standardize large-scale evaluation across models and tasks [19, 87], while dedicated benchmarks probe specific capabilities such as multimodal instruction following [15, 98] and human-preference alignment [96]. Building on reinforcement learning from human feedback and preference optimization [58, 64] and, more recently, reinforcement learning with verifiable rewards [25], a parallel line develops multimodal reward models and RL recipes to strengthen reasoning and alignment [16, 48, 49, 51, 86, 88, 97]; verifiable rewards have further been applied to spatial understanding and dense captioning [9, 45, 82]. Our study is complementary: rather than measuring instruction following, preference alignment, or static capability, we isolate whether a model can maintain and act on hidden state across a long interactive history, and our fine-tuning study connects to this line by supervising on simulator rollouts with verifiable outcomes.

- B. More Analysis

Scale-sweep raw data for Fig. 3. Tabs. 9 and 10 report the per-size numbers underlying the scale-sweep plots in the main text. On Matching Pairs, Score% declines from 90.6% (4×4) to 0.7% (12×12), while response cost per matched pair rises from 4.59 to 720. On 3D Maze, Game Score peaks at 7×7 (66.7%) and drops from 9×9

Matching Pairs (8×10) 3D Maze (9×9)

Model Name

Baseline↑ +Memory Map↑ MemGap↓ Baseline↑ +Minimap↑ MemGap↓

Qwen3.5-397B 38.3 78.7 51.3 23.8 40.2 40.8 Kimi-K2.5 43.3 80.3 46.1 24.6 35.6 30.9

[Figure 76]

[Figure 77]

- Table 8: External-memory intervention on both environments. Matching Pairs reports Score%, 3D Maze reports GS% (both higher is better). Matching Pairs baseline numbers are ported from the Image-Noise rows of Tab. 4; 3D Maze baseline is the no-minimap setting. +Memory Map: an environment-provided table of all previously revealed (position, identity) pairs. +Minimap: an environment-provided top-down map of visited cells. MemGap: 1 − 𝑆/𝑆* with 𝑆 = Baseline and 𝑆* = intervention score (Eq. 3); values near 1 localize the bottleneck to belief-state tracking.

Board Pairs Score% ↑ Resp./Score ↓

4×4 8 90.6 4.59 4×6 12 56.2 8.89 6×6 18 33.3 15.00 6×8 24 32.3 15.48 8×8 32 32.8 15.24 8×10 40 23.8 21.05 10×10 50 12.5 40.00 10×12 60 12.1 41.38 12×12 72 0.7 720.00 12×14 84 3.6 140.00

- Table 9: Ablation on Matching Pairs board size. Results are for Qwen3.5-397B in single-player actionfeedback image mode with the textures theme and max_resp_per_pair=5, averaged over four seeds. Resp./Score reports the response cost per matched pair.

Maze Cells GS% ↑ Eff.% ↑ Explore% ↑

5×5 25 62.6 46.1 42 7×7 49 66.7 55.4 45 9×9 81 23.8 21.9 29 11×11 121 21.8 37.7 20 13×13 169 22.3 51.2 18 15×15 225 19.7 36.6 15

Table 10: Ablation on 3D Maze size. Results are for Qwen3.5-397B in the no-minimap setting, averaged over five seeds. Cells is the maze grid size. GS%, Eff.%, and Explore% are defined in Tab. 2.

onward, paralleled by declining Explore%.

External-memory intervention raw data for Fig. 4. Tab. 8 provides the exact baseline and intervention scores behind the bar chart in the main text. On Matching Pairs (8×10), the memory map roughly doubles Score% for both Qwen3.5-397B (38.3→78.7) and Kimi-K2.5 (43.3→80.3). On 3D Maze (9×9), the minimap yields smaller gains (Qwen: 23.8→40.2; Kimi: 24.6→35.6), consistent with additional bottlenecks beyond memory access.

Oracle interface contents and MemGap reading guide. The Matching Pairs memory map lists, at each step, the (identity, position) pairs that have already been revealed in the current episode; positions not yet flipped are omitted. The 3D Maze minimap encodes the set of visited cells, the agent’s current cell and orientation, and the wall segments observed from those cells; unvisited cells, unseen walls, and the goal location are not included. The minimap therefore reduces but does not eliminate non-Markov structure, since the agent must still plan exploration of the unobserved region. We report MemGap (Eq. 3) only when the oracle score 𝑆*(𝑚) is non-trivially above the baseline, and read its values as practical contributions of externalized state under our specific interface rather than exact decompositions of failure modes.

Text observations dominate image observations in within-model duels, except for Seed. Tab. 11 pits a text-observation player against an image-observation player from the same model family on 8×8 and 8×10 boards, with both player orders merged over three seeds. The text side wins 100% of games for Kimi-K2.5 and Qwen3.5-397B with gaps of +22.7 to +35.7 pairs, while Seed-2.0 is the exception: the image player wins on 8×8 (−3.3) and the text player wins only narrowly on 8×10 (+1.8). One possible explanation is that Seed-2.0’s vision encoder retains card identities more faithfully than those of Kimi and Qwen, reducing the modality gap that other models suffer from. This suggests that the text–image performance gap is not an inherent property of the task but depends on how well each model’s visual pipeline preserves identity information across turns. Tab. 12 aggregates the text-observation duels into the same ranking format as the main image-mode duel table for cross-modality comparison.

Text Player Image Player Outcome

Model Pair Board

Model Name Score Win% Model Name Score Win% Gap Winner

Kimi text vs. image 8×8 Kimi 27.8 100.0 Kimi 2.8 0.0 +25.0 Text Qwen3.5 text vs. image 8×8 Qwen3.5 27.7 100.0 Qwen3.5 2.7 0.0 +25.0 Text Seed text vs. image 8×8 Seed 7.2 33.3 Seed 10.5 66.7 -3.3 Image

[Figure 78]

[Figure 79]

[Figure 80]

Qwen3.5 text vs. image 8×10 Qwen3.5 37.8 100.0 Qwen3.5 2.2 0.0 +35.7 Text Kimi text vs. image 8×10 Kimi 26.7 100.0 Kimi 4.0 0.0 +22.7 Text Seed text vs. image 8×10 Seed 8.0 33.3 Seed 6.2 50.0 +1.8 Text

[Figure 81]

[Figure 82]

[Figure 83]

- Table 11: Ablation on text versus image observations in two-player Matching Pairs. Each row compares a text-observation player against an image-observation player under the same model family, merging both player orders over three seeds. Gap is Text score minus Image score. Missing runs are shown as –.

Model Name Elo W T L Score% Rank

[Figure 84]

Qwen3.5-397B 1070 16 1 7 58.2 1 Kimi-K2.5 1035 14 2 8 56.0 2 Seed-2.0-Lite 895 4 1 19 33.1 3

[Figure 85]

[Figure 86]

- Table 12: Rank list for two-player Matching Pairs duels in text mode. Results aggregate completed pairwise matchups across both player orders, 8×8 and 8×10 boards, and three seeds (24 games per model). Metrics follow Tab. 3.

Model Name Theme Score%↑

[Figure 87]

Qwen3.5-397B

ASCII 75.8 Abstract 56.7 Similar-Colors 54.2 Textures 44.2 Noise 38.3 Poker 20.0

[Figure 88]

Kimi-K2.5

ASCII 72.5 Similar-Colors 70.8 Textures 50.8 Noise 43.3 Abstract 41.4 Poker 30.1

Model Name Size Baseline Color Tag

SR Eff.% Explore% GS% SR Eff.% Explore% GS%

[Figure 89]

Kimi-K2.5

7×7 1/5 66.7 44.0 34.3 4/5 39.9 47.0 60.7 9×9 2/5 62.5 24.0 39.7 1/5 100.0 19.0 27.6

11×11 2/5 68.1 25.0 41.1 2/5 33.5 24.0 33.9

[Figure 90]

Qwen3.5-397B

7×7 4/5 55.4 47.0 66.9 4/5 30.0 51.0 57.1 9×9 1/5 21.9 29.0 23.8 2/5 84.3 24.0 44.1

11×11 1/5 37.7 20.0 21.8 2/5 32.2 27.0 34.5

- Table 13: Visual-pattern ablation. Left: Matching Pairs theme sweep (with action feedback), varying cardidentifier distinctiveness while keeping rules fixed. Right: 3D Maze baseline (uniform walls) vs. Color Tag (wall segments painted with distinct colors as visual landmarks). Identifier distinctiveness strongly affects Matching Pairs, while wall color tags do not consistently help 3D Maze.

Visual identifier distinctiveness strongly affects Matching Pairs, but wall color tags do not consistently help 3D Maze. We test whether failures arise only from long-horizon memory or also from unstable visual identity binding. Matching Pairs (8×10) varies card-pattern distinctiveness across six themes; 3D Maze compares uniform walls against color-tagged walls. Tab. 13 reports both. On Matching Pairs, Qwen3.5-397B drops from 75.8% (ASCII) to 20.0% (Poker), and Kimi-K2.5 from 72.5% to 30.1%, showing that less distinctive identifiers compound the memory load. On 3D Maze the color-tag intervention is mixed: Kimi-K2.5 improves at 7×7 (1/5→4/5) but degrades at 9×9 (2/5→1/5), and Qwen3.5-397B gains modestly at 9×9 and 11×11. Hidden-state tracking is therefore coupled with perceptual binding for card identification, while wall-level visual landmarks alone fail to stabilize spatial belief updates. The asymmetry between the two environments is informative: in Matching Pairs, each card identity must be discriminated from dozens of visually similar alternatives, so pattern distinctiveness directly affects recognition accuracy. In 3D Maze, wall colors provide only coarse spatial cues that do not resolve the core difficulty of maintaining a global position estimate from local egocentric views. This distinction suggests that visual interventions are most effective when they target the specific perceptual bottleneck of the task.

The minimap helps mostly at intermediate and large scales, and its effect is model-dependent. Tab. 14 reports the minimap condition across 5×5 to 15×15 mazes with exploration and collision statistics. Seed-2.0-

Model Name 5×5 7×7 9×9 11×11 13×13 15×15

Eff.% Explore% GS% Eff.% Explore% GS% Eff.% Explore% GS% Eff.% Explore% GS% Eff.% Explore% GS% Eff.% Explore% GS%

Seed-2.0-Lite 36.9 46.0 59.4 31.6 40.0 29.2 39.0 26.0 35.6 0.0 16.0 8.0 0.0 16.0 8.0 0.0 10.0 5.0 Kimi-K2.5 61.8 43.0 69.0 41.5 36.0 39.1 41.2 32.0 37.8 36.5 26.0 35.1 0.0 19.0 9.5 0.0 14.0 7.0 Qwen3.5-397B 58.8 44.0 79.4 33.5 46.0 58.0 54.7 31.0 40.2 34.4 28.0 45.9 33.6 23.0 22.6 0.0 13.0 6.5 Seed-2.0-Pro 56.6 48.0 78.3 69.9 43.0 59.6 70.6 34.0 58.0 63.0 30.0 54.9 75.9 20.0 25.6 51.1 22.0 36.8

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

- Table 14: Maze-size sweep with minimap. Eff.%, Explore%, and GS% are defined in Tab. 2; GS% (aggregate) is rightmost in each size block.

Model Name Standard Ask-Output

Eff.% Explore% GS% Eff.% Explore% Traj-Match% GS%

[Figure 95]

Kimi-K2.5 62.5 23.0 39.4 100.0 28.0 68.2 31.2 Qwen3.5-397B 21.9 29.0 23.8 100.0 19.0 82.7 27.6 Seed-2.0-Lite 26.2 30.0 24.6 49.9 35.0 66.3 63.5 Seed-2.0-Pro 60.2 32.0 54.5 67.2 31.0 87.2 56.4

[Figure 96]

[Figure 97]

[Figure 98]

- Table 15: Ask-output ablation on 9×9 mazes. The ask-output setting explicitly asks the model to emit its internal wall map at every step; Traj-Match measures trajectory-map agreement.

Pro benefits the most, reaching 3/5 success at both 9×9 and 11×11 and recovering 2/5 at 15×15. At 13×13, Qwen3.5-397B and Seed-2.0-Pro each complete one minimap run, while Seed-2.0-Lite and Kimi-K2.5 fail on all seeds. The Walls column reveals a second pattern: Kimi-K2.5 keeps collisions low even when it fails, showing that safer local control does not guarantee global completion. This dissociation between local safety and global success highlights two distinct failure modes in 3D Maze: models can fail either because they misread the immediate 3D scene (leading to wall collisions) or because they lose track of the global spatial layout (leading to loops and revisits). Kimi-K2.5 predominantly exhibits the latter failure, maintaining accurate local perception while lacking a coherent spatial map.

Ask-output prompting helps Seed-2.0-Lite substantially but does not improve overall completion for Kimi or Qwen. Tab. 15 reports results when the agent is prompted to explicitly output its internal spatial map at each step. Seed-2.0-Lite rises from 1/5 to 4/5, while Kimi-K2.5 and Qwen3.5-397B show no gain in completion. Their successful ask-output runs are near-optimal (Eff = 1.000) when they work, but occur no more frequently than in the standard setting. Seed-2.0-Pro maintains 3/5 success in both conditions with slightly higher efficiency under ask-output. The Traj-Match column reveals that map-output quality varies widely: Seed-2.0-Pro reaches 87.2% and Qwen3.5-397B achieves 82.7%, while Seed-2.0-Lite and Kimi-K2.5 score lower (66.3% and 68.2%). Explicit map externalization thus benefits models that can maintain accurate spatial representations. The gap between Traj-Match accuracy and task completion suggests that merely producing a map is not enough: the model must also use that map to plan. A model with high Traj-Match but low success rate accurately records where it has been yet fails to translate that record into effective route planning, consistent with a decision-selection contribution in addition to memory access.

Limiting context does not affect all models equally, and more history does not uniformly help. Tab. 16 compares the standard full-history setting with history windows of 3, 5, and 10 turns on the extended 9×9 experiment. Seed-2.0-Pro achieves its best success (3/5) under full history and drops to 2/5 across all three windows, consistent with exploiting long-range dependencies. Qwen3.5-397B shows the opposite trend: only 1/5 with full history but 2/5 under all windowed conditions. Very long dialogue history may impair its decisions through attention dilution or accumulated misinformation. Kimi-K2.5 shows a milder version of this pattern (2/5 under full history, dropping to 1/5 at windows of 5 and 10), and Seed-2.0-Lite remains unstable across all conditions (1–2/5). The history–performance relationship is model-dependent: some models benefit from richer context while others are hindered by it.

### C. Visualization Cases

The following trajectory visualizations illustrate three common failure modes observed across our experiments: (i) spatial drift, where the model’s stated position gradually diverges from its actual location; (ii) local oscillation, where the model revisits the same small region without making global progress; and (iii) reasoning–action

###### Model Name Full History Window = 3 Window = 5 Window = 10

Eff.% Explore% GS% Eff.% Explore% GS% Eff.% Explore% GS% Eff.% Explore% GS%

Kimi-K2.5 62.5 23.0 39.4 66.1 26.0 41.0 93.3 28.0 30.5 100.0 25.0 30.0 Qwen3.5-397B 21.9 29.0 23.8 56.3 30.0 40.3 26.6 31.0 34.6 58.9 33.0 41.7 Seed-2.0-Lite 26.2 30.0 24.6 82.4 34.0 31.8 25.8 33.0 35.1 73.7 23.0 26.6 Seed-2.0-Pro 60.2 32.0 54.5 42.6 21.0 34.8 64.5 30.0 41.9 64.6 31.0 42.2

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

- Table 16: History-window ablation on 9×9 mazes without minimap. Full history is compared against windows retaining only the most recent 3, 5, or 10 turns.

Kimi-K2.5 (Failure) Steps: 236 / 236 (failed, Eff: 0.00)

Seed-2.0 (Success) Steps: 53 (Optimal: 25, Eff: 0.47)

[Figure 103]

[Figure 104]

1 Step 1 2 Step 47 3 Step 92 1 Step 1 2 Step 79 3 Step 236

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

- Figure 5: Success rate with minimap across maze sizes, corresponding to Tab. 14.

mismatch, where the model’s verbal reasoning is spatially coherent but its chosen action contradicts its own stated plan. Each figure contrasts a successful and a failed run on the same maze to highlight where the reasoning diverges.

Successful runs maintain spatially grounded reasoning; failed runs drift into confusion despite the minimap. Fig. 5 contrasts two 3D Maze trajectories under the minimap condition. Seed-2.0 reaches the goal in 53 steps (optimal 25, Eff 0.47); its reasoning anchors each move to explicit coordinates and headings (“at (3,6) facing South”, “straight corridor to (10,10)”), so the trajectory (green) advances steadily toward the goal. Kimi-K2.5 exhausts the 236-step budget on the same map without reaching the goal. Its reasoning still cites the minimap but loses spatial closure (“hit a wall, the explored area has expanded significantly, but still can’t find a way”) and oscillates around the lower-right region (red trajectory). Providing an external minimap does not by itself fix hidden-state tracking: the model must still translate the minimap into a consistent position belief. Failures appear as repeated wall collisions and incoherent direction choices rather than as missing information. This case demonstrates that the minimap’s value depends on the model’s ability to ground its position within the provided map. When this grounding fails, the minimap becomes decorative: the model cites it in its reasoning but cannot use it to correct accumulated spatial errors.

Kimi-K2.5 (Failure) Steps: 96 / 96 (failed, Eff: 0.00)

Seed-2.0 (Success) Steps: 28 (Optimal: 24, Eff: 0.86)

[Figure 109]

[Figure 110]

1 Step 1 2 Step 9 3 Step 28 1 Step 1 2 Step 25 3 Step 73

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

- Figure 6: Baseline 3D Maze trajectories (no minimap, uniform walls). Seed-2.0 (left, blue) reaches the goal in 28 steps (optimal 24, Eff 0.86) with spatially grounded reasoning. Kimi-K2.5 (right, red) exhausts the 96-step budget, revisiting cells and oscillating despite recognizing dead ends.

Without any external aid, strong spatial grounding separates success from failure. Fig. 6 shows a 7×7 baseline run (no minimap, uniform walls). Seed-2.0 maintains coordinate-level awareness throughout (“moving South increases my y-coordinate, bringing me closer to the goal”) and reaches the goal near-optimally. KimiK2.5 correctly identifies dead ends (“I need a completely different approach, I’ve been going in circles”) but lacks the spatial map to act on this realization, resulting in repeated revisits. The contrast reveals that recognizing failure is not the bottleneck: Kimi-K2.5 diagnoses the problem in natural language but cannot translate that diagnosis into a corrective spatial plan. Without a persistent internal map, each “different approach” amounts to a random direction change rather than a systematic exploration strategy.

Color-tagged walls can help spatial disambiguation but do not eliminate failures. Fig. 7 compares trajectories on the same 7×7 maze under two visual conditions. With color tags, Seed-2.0 anchors its reasoning to wall colors and junction landmarks, reaching the goal in 53 steps. Under uniform walls, the same model loses spatial orientation after step 26 and oscillates in the upper-right region until timeout. The color tags provide perceptual anchors that stabilize position tracking, but as shown in Tab. 13, this advantage does not generalize across all models or maze sizes. Across the three visualization cases, a consistent pattern emerges: successful models maintain an explicit, coordinate-level spatial representation that they update after each action, while failed models rely on qualitative spatial language (“keep going”, “try a different direction”) that does not accumulate into a coherent map. The key differentiator is not whether the model can perceive the current view, but whether it can integrate that view with its history to form a stable belief about global position.

Baseline (Failure) Steps: 100 / 100 (failed, Eff: 0.00)

Color Tag (Success) Steps: 53 (Optimal: 25, Eff: 0.47)

[Figure 115]

[Figure 116]

1 Step 1 2 Step 16 3 Step 47 1 Step 1 2 Step 37 3 Step 99

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

- Figure 7: Color Tag vs. Baseline on 3D Maze. Left (blue): color-tagged walls help Seed-2.0 navigate a 7×7 maze in 53 steps (Eff 0.47). Right (red): the same model fails under uniform walls, exhausting the 100-step budget with repeated oscillations.

0 20 40 60 80 100 120

Round

0

10

20

30

40

50

CumulativeMatchedPairs

Single-Player on 10×10 Noise (Same Board, Seed 0)

GPT-5.4 (31/50)

Gemini-3.1-Pro (16/50)

Maximum (50)

[Figure 121]

GPT-5.4 R40 Gemini R40

[Figure 122]

[Figure 123]

GPT-5.4 R100 Gemini R100

[Figure 124]

- Figure 8: Single-player Matching Pairs trajectory. GPT-5.4 and Gemini-3.1-Pro are evaluated on the same 10×10 noise board with seed 0. The curve reports cumulative matched pairs over 120 rounds, and the snapshots show board states at selected rounds.

Gemini-3.1-Pro vs GPT-5.4 on 8×10 Textures (Same Board, Seed 0)

###### Gemini Goes First

###### GPT-5.4 Goes First

20.0

Gemini-3.1-Pro (1st)

GPT-5.4 (1st)

18 pairs

GPT-5.4 (2nd)

Gemini-3.1-Pro (2nd)

17.5

CumulativeMatchedPairs

15.0

15 pairs

12.5

10.0

9 pairs

7.5

6 pairs

5.0

2.5

0.0

0 20 40 60 80

0 20 40 60 80

Round

Round

- Figure 9: Duel Matching Pairs trajectory. Gemini-3.1-Pro and GPT-5.4 are compared on the same 8×10 texture board with seed 0 under both player orders. The two panels test whether the outcome is mainly caused by first-move advantage.

### D. Matching Pairs Case Studies

Single-player trajectory. Fig. 8 compares GPT-5.4 and Gemini-3.1-Pro on the same 10×10 noise board. GPT-5.4 finishes with 31 of 50 pairs, while Gemini-3.1-Pro reaches 16 of 50. Beyond the final score, the trajectory reveals a clear difference in how observations are converted into matches. Gemini-3.1-Pro shows several plateau phases, where continued flips do not lead to a score. This indicates a weaker ability to maintain and reuse card-location bindings after they disappear from the current observation. GPT-5.4 instead shows a sharp improvement between rounds 80 and 100. This late jump suggests that earlier observations have been integrated into a more useful belief state, which then supports a concentrated phase of successful matching. The snapshots are consistent with this pattern, as GPT-5.4 clears a larger region of the board by round 100 while Gemini-3.1-Pro leaves more cells unresolved. This case highlights that the benchmark tests not only visual observation but also whether the model can transform past observations into a stable hidden-state estimate for later action.

Duel trajectory. Fig. 9 compares the two models on the same 8×10 texture board under both player orders. When Gemini-3.1-Pro moves first, it finishes with 15 pairs, while GPT-5.4 reaches 9. When GPT-5.4 moves first, Gemini-3.1-Pro still finishes with 18 pairs, while GPT-5.4 reaches 6. This suggests that the duel result is not mainly driven by first-move advantage. Gemini-3.1-Pro gains more matches in the late game and appears to use earlier observations more effectively. The duel setting therefore provides a stricter test of internal belief state reconstruction, because each model must remember both its own observations and the cards revealed by the opponent.

### E. Potential Risks

The main risk of this work is possible over-interpretation of benchmark results as a complete measure of model intelligence or real-world reliability. To reduce this risk, we report the task settings, evaluation metrics, and limitations of our benchmark, and we frame the results as evidence about hidden-state tracking under controlled conditions. The benchmark is intended for research evaluation and diagnostic analysis, rather than for direct deployment decisions in safety-critical applications.

### F. SFT Training Details

All Qwen3.5-9B fine-tuning runs in §5 use the same recipe; only the training data composition changes across the opt32k and rmix32k configurations.

Finetuning scope. We perform full fine-tuning of the language model with the vision tower and multimodal projector frozen, using the qwen3_5_nothink chat template.

Optimization. AdamW with learning rate 1e−5, cosine schedule, warmup ratio 0.1, 1 epoch, and an effective batch size of 128 (per-device batch size 1 with gradient accumulation 16 across 8 GPUs).

Sequence and visual budget. Cutoff length 28,160 tokens for opt32k and 29,000 tokens for rmix32k; per-image maximum 65,536 pixels with patch cropping enabled.

Data composition. opt32k mixes 16k Matching Pairs and 16k 3D Maze optimal-policy rollouts (∼32k records total); rmix32k replaces each 16k optimal block with a 16k rollout-plus-optimal mixture from the same environment. The two datasets are concatenated with the concat mix strategy.

Reproducibility. Random seed and other unspecified hyperparameters follow LLaMA-Factory defaults. Each configuration is trained with a single seed; downstream evaluation aggregates over 5 environment seeds per configuration (see §3.5).

