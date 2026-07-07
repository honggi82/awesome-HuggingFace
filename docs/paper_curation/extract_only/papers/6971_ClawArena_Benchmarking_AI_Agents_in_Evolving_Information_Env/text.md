# arXiv:2604.04202v2[cs.LG]16May2026

[Figure 1]

## ClawArena: Benchmarking AI Agents in Evolving Information Environments

Haonian Ji1,∗Kaiwen Xiong1∗, Siwei Han1, Peng Xia1, Shi Qiu1, Yiyang Zhou1, Jiaqi Liu1, Jinlong Li1, Bingzhou Li1, Zeyu Zheng3, Cihang Xie2, Huaxiu Yao1 1UNC-Chapel Hill 2University of California, Santa Cruz 3University of California, Berkeley {haonianj, xkaiwen, huaxiu}@cs.unc.edu

### Abstract

AI agents deployed as persistent assistants must maintain correct beliefs as their information environment evolves. In practice, evidence is scattered across heterogeneous sources that often contradict one another, new information can invalidate earlier conclusions, and user preferences surface through corrections rather than explicit instructions. Existing benchmarks largely assume static, single-authority settings and do not evaluate whether agents can keep up with this complexity. We introduce CLAWARENA, a benchmark for evaluating AI agents in evolving information environments. Each scenario maintains a complete hidden ground truth while exposing the agent only to noisy, partial, and sometimes contradictory traces across multi-channel sessions, workspace files, and staged updates. Evaluation is organized around three coupled challenges: multi-source conflict reasoning, dynamic belief revision, and implicit personalization, whose interactions yield a 14-category question taxonomy. Two question formats, multi-choice (set-selection) and shell-based executable checks, test both reasoning and workspace grounding. CLAWARENA comprises 12 multi-turn scenarios spanning 337 evaluation rounds with 45 dynamic updates, evaluated across five agent frameworks and 18 language models from proprietary, community-accessible, and self-hosted sources. To capture behavioral consistency beyond correctness, we introduce the Composite Reliability Score (CRS), which combines Task Completion Rate with a Robustness term that rewards sustained success streaks and penalizes clustered failures. Experiments show that model capability accounts for a 29-point CRS range across models while framework design accounts for up to a 24-point range, that MetaClaw’s skill overlay reliably improves Robustness without degrading accuracy, and that belief revision difficulty is determined by update design strategy rather than update volume. Code is available at https://github.com/aiming-lab/ClawArena.

### 1 Introduction

Maintaining a correct view of ongoing work is critical for AI agents deployed as persistent assistants, yet real information environments make this difficult: evidence is scattered across heterogeneous sources (chat histories, workspace files, monitoring logs) that often contradict one another, for instance when multiple sources report conflicting timelines for the same incident. These environments create three coupled challenges. The agent must judge source reliability rather than naively aggregate (multi-source conflict reasoning). New evidence arrives over time and can invalidate previously correct conclusions, so the agent must revise rather than accumulate (dynamic belief revision). User preferences are rarely stated explicitly and surface through corrections and interaction patterns, so the agent must learn and apply them without reminders (implicit personalization). An agent that handles conflicts but fails to revise, or revises but ignores format preferences, still produces unreliable output.

∗Equal contribution.

[Figure 2]

Figure 1: Overview of CLAWARENA. An evolving information environment with conflicting and progressively updated evidence is structured along three coupled evaluation dimensions, instantiated through two complementary question formats, and summarized by a composite reliability score that jointly reflects accuracy and behavioural consistency.

Existing evaluations test fragments of this problem but not the full setting. Task-oriented agent benchmarks typically provide a single authoritative environment (Jimenez et al., 2024; Liu et al., 2024; Zhou et al., 2024; Xie et al., 2024b; Mialon et al., 2024). Long-context and multi-hop QA benchmarks test retrieval and composition over static evidence (Yang et al., 2018; Trivedi et al., 2022; Bai et al., 2024; Hsieh et al., 2024). Memory benchmarks emphasize long-horizon recall but usually without explicit source conflict or silent preference retention (Maharana et al., 2024; Zhang et al., 2018). Taken together, existing benchmarks assume a static, single-authority information environment, leaving open the question of whether agents can maintain correct and up-to-date beliefs as their information environment evolves.

We introduce CLAWARENA, a benchmark for AI agents in evolving information environments (Figure 1). Each scenario combines multi-channel session histories, workspace files, staged update packages, and a four-stage personalization protocol ending in silent-exam rounds with no preference reminders. Each scenario maintains a complete hidden ground truth, while the agent observes only noisy, partial, sometimes contradictory traces; correctness is verified against the ground truth rather than any single observable source. Observable materials are grounded in real-world empirical distributions covering message timing, contact frequency, and information overload. The three dimensions and their interactions yield a 14-category taxonomy that blocks one-dimensional scoring, and two question formats (multi-choice set-selection and shell-based executable checks) test reasoning and workspace grounding. The benchmark comprises 12 multi-turn scenarios spanning 337 evaluation rounds with 45 dynamic updates, used in full for all reported experiments.

Beyond the benchmark itself we contribute: (i) a reproducible three-stage construction pipeline with real-world empirical grounding; (ii) a 14-category diagnostic taxonomy enabling per-dimension and per-interaction failure analysis; (iii) the Composite Reliability Score (CRS), with a Robustness decomposition into Success Cohesion and Failure Dispersion, capturing behavioral consistency beyond raw accuracy; and (iv) experiments across five frameworks and 18 language models from proprietary, community-accessible, and self-hosted sources, showing that model capability dominates framework design in CRS range, that MetaClaw’s skill overlay improves Robustness without degrading accuracy, and that belief revision difficulty is governed by update design rather than update volume.

### 2 Related Work

Agent Benchmarks. Task-oriented benchmarks such as SWE-bench (Jimenez et al., 2024), AgentBench (Liu et al., 2024), WebArena (Zhou et al., 2024), OSWorld (Xie et al., 2024b), and GAIA (Mialon et al., 2024) evaluate tool use, planning, and execution in largely singleauthority environments and do not test adjudication across conflicting sources. Long-context and multi-hop QA benchmarks such as HotpotQA (Yang et al., 2018), MuSiQue (Trivedi

- Table 1: Comparison with harness-native agent benchmarks. ✓= supported, ✗= unsupported, = partial (mechanism present but weakened, e.g., third-party messages exposed only as plain files rather than channel-tagged sessions, or preferences encoded as static persona rather than learned in silent rounds). MSC: agent must analyse user–third-party conversations distinguished by channel. DU: environment is overwritten between user turns (intra-loop tool-return dynamics do not count). MU: user re-engages with new queries across rounds. Pref.: implicit user preferences applied in silent-exam rounds.

Benchmark Task sourcing Exec. mode MSC DU MU Pref. Verification Frmw. Scale (Q/Scen.)

ClawBench Manual pool Live web ✗ ✗ ✗ ✗ rule+llm 8 283/144 sites Claw-Eval Curated from upstream Sandbox + mock ✗ ✗ ✓ ✗ rule+llm 1 300/9 cats Claw-Eval-Live Live signals (quarterly) Mock services ✗ ✗ ✗ ✗ rule+llm 1 105/17 fam. ClawMark Manual + AI synthesis Sandboxed services ✓ ✓ ✓ ✗ rule-based 1 100/13 scen. ClawsBench Expert-designed Mock services ✓ ✗ ✗ ✗ rule-based 4 44/5 svc. MetaClawBench Synthetic Workspace sim ✓ ✓ rule-based 1 346/30 days PinchBench Manual (real-world) Real (OpenClaw) ✗ ✗ ✗ ✗ rule+llm 1 23/8 cats QwenClawBench Empirical (claimed) Real (Docker) ✗ ✗ ✗ rule+llm 1 100/8 dom. WildClawBench Manual (in the wild) Real (OpenClaw) ✓ ✗ ✗ rule+llm 1 60/6 cats ZClawBench Manual + synthetic Real + part. mock ✗ ✗ ✗ ✗ rule+llm 1 116/6 cats

ClawArena (Ours) Empirical synthesis Multi-channel sim ✓ ✓ ✓ ✓ rule-based 5 337/12 scen.

et al., 2022), LongBench (Bai et al., 2024), and RULER (Hsieh et al., 2024) stress retrieval and composition over static evidence; ConflictQA (Xie et al., 2024a) adds conflicting claims, but evidence remains fixed at inference time, so none evaluates revision after new evidence arrives. Memory and persona benchmarks such as LoCoMo (Maharana et al., 2024) and PersonaChat (Zhang et al., 2018) capture long-horizon recall and user-model consistency but do not jointly require cross-source conflict resolution, dynamic updates, and silent preference retention. ClawArena targets this joint setting.

Harness-native Benchmarks. A second emerging line of work evaluates models inside concrete agent harnesses (OpenClaw, Claude Code, Codex, Gemini CLI, etc.) rather than

- as standalone reasoners. These include browser harnesses (CLAWBENCH (Zhang et al., 2026)), real or sandboxed productivity stacks (PINCHBENCH (PinchBench Contributors, 2026), QWENCLAWBENCH (Qwen Team & Alibaba Data, 2026), WILDCLAWBENCH (Ding et al., 2026), ZCLAWBENCH (Z.AI, 2026)), mock-service suites (CLAWSBENCH (Li et al., 2026b), CLAW-EVAL (Ye et al., 2026), CLAW-EVAL-LIVE (Li et al., 2026a)), workspace simulations (METACLAWBENCH (Xia et al., 2026b)), and multi-day coworker scenarios (CLAWMARK (Meng et al., 2026)). Table 1 contrasts them across the four design axes that drive evolving information environments, together with task sourcing, execution mode, verification method, and the number of frameworks evaluated. Most prior work covers
- at most two of these axes; the closest entry, CLAWMARK, satisfies three but omits implicit personalization. CLAWARENA additionally instantiates a four-stage silent-exam preference protocol and reports across five distinct frameworks rather than one, making it the only benchmark to satisfy all four design axes simultaneously.

LLM Agents. Modern LLMs enable agents beyond single-turn QA (Comanici et al., 2025; OpenAI, 2025; Yao et al., 2022; Shinn et al., 2023); reasoning-focused models handle complex multi-step tasks (Zhao et al., 2023; Tang et al., 2025; Ouyang et al., 2025; Chhikara et al., 2025; Liu et al., 2026b; Xia et al., 2026a;b; Liu et al., 2026a); tool-augmented agents interact with APIs, interpreters, and file systems (Team et al., 2025; Qin et al., 2024; Feng et al., 2025; Xia et al., 2025; Liu et al., 2025; Jin et al., 2025); and recent work explores multi-agent coordination (Wu et al., 2024; Li et al., 2023), domain-specific systems (Ji et al., 2025; Han et al., 2026), and long-horizon planning. Most such architectures are evaluated in episodic settings that reset between tasks. ClawArena instead targets the persistent-assistant regime where beliefs accumulate across sessions and must be revised as the environment evolves.

### 3 ClawArena

#### 3.1 Overview

Each ClawArena scenario simulates a realistic information environment an AI agent must navigate. A scenario comprises multi-channel session histories (5–7 channels such as Slack,

email, and WeChat, 200–400 messages), workspace files (4–8 documents including monitoring logs, sprint notes, audit reports), staged update packages that inject new evidence over time, and a four-stage personalization protocol ending in silent-exam rounds. Each scenario maintains a complete hidden ground truth against which answers are verified, while the agent observes only noisy, partial, sometimes contradictory traces.

For example, a startup outage scenario presents conflicting recovery timelines across a direct message, monitoring export, and sprint notes; the agent must synthesize conflicts, revise as later updates arrive, and present results in the user’s preferred format.

CLAWARENA comprises 12 scenarios spanning diverse professional contexts (retail analytics, finance, healthcare, information security, HR, education, research integrity, and others) with 337 evaluation rounds and 45 dynamic updates in total; representative scenarios are catalogued in Appendix A and the full statistical breakdown is reported in Appendix E. Two formats test complementary capabilities: multi-choice (set-selection) over 7–9 candidate statements, and shell-based executable checks verifying whether claims about workspace evidence hold in the actual files.

- Table 2: Question taxonomy of CLAWARENA. Each row shows a non-empty subset of {MS, DU, P} and a representative task.

Tag Dimensions Representative question Single dimension

MS Multi-Source Identify which statements are supported across a direct message, a group chat, and a log export. DU Dynamic Update Update the conclusion after a new audit file or appended message arrives. P Personalization Recall or apply the learned output style without an explicit reminder.

Pairwise interaction MS×DU MS × DU Reassess a cross-source claim after the evidence state changes.

MS×P MS × P Resolve source conflict while using the learned reporting style. DU×P DU × P Apply preferences to a task that emerges only after an update.

Three-way interaction

All MS × DU × P Synthesize updated cross-source evidence and present it in the

learned style.

#### 3.2 Evaluation dimensions and taxonomy

A persistent assistant’s information environment has three structural traits causing distinct failure modes: information spans multiple conflicting sources, the evidence base evolves, and user expectations form from prior interactions rather than upfront statements. ClawArena frames evaluation around three matching dimensions, each overlooked by existing benchmarks:

Multi-source conflict reasoning (MS). Evidence is distributed across heterogeneous sources that may contradict one another (e.g., a DM claiming a four-minute outage versus a monitoring log recording 47 minutes). The agent must judge source reliability; accuracy is measured as exact-set match on cross-source conflict questions.

Dynamic belief revision (DU). New evidence can invalidate previously correct conclusions (e.g., a later audit reveals an incomplete fix). The agent must explicitly revise; the revision rate measures how often it updates its answer after contradictory evidence arrives.

Implicit personalization (P). User preferences emerge via corrections and interaction patterns (e.g., users reformat bullet outputs into tables). The agent must learn and apply these preferences without reminders; compliance is scored only in silent-exam rounds.

The three dimensions and their pairwise and three-way interactions yield seven combination categories (Table 2), each split into recall (can the agent retrieve the evidence?) and reasoning

(can it draw the correct conclusion?) variants for 14 categories. This blocks scoring well by solving one dimension in isolation.

#### 3.3 Scenario design

[Figure 3]

Each scenario is a six-layer specification. Layer 0 holds the hidden ground truth (objective timeline, contradiction map, answer provenance). Layers 1–4 are agentvisible: workspace files, session histories, evaluation questions, and staged update packages. Layer 5 is an internal generation guide for formatting and noise controls. Layer 0 is never shown, so verification is reliable while the observable layers behave like noisy reflections of the same reality.

Multi-source conflict design. ClawArena distinguishes integration (assembling a coherent picture from scattered evidence) from conflict reasoning (recognizing disagreement and deciding which sources dominate). Each scenario embeds four canonical evidence relations: factual conflicts (C1), authority conflicts (C2, undocumented or contradicted approvals), nonconflict slots (C3, genuine agreement), and temporal/process conflicts (C4). C3 blocks scoring well by treating every disagreement as a contradiction.

Staged update design. Updates arrive in stages: early rounds expose plausible but incomplete narratives, while later rounds inject contradictions and authoritative confirmations. Updates are subjective (appended messages shifting credibility) or objective (file modifications altering the factual record). Agents are scored on whether they revise once contradictions arrive, not on pre-emptive skepticism.

Figure 2: CLAWARENA construction pipeline. Real-world distributions and character profiles feed a three-stage bootstrap, producing the 12-scenario benchmark organized into six layers and refined through three validation passes.

Personalization protocol. Personalization proceeds through four stages: (1) calibration with natural hints (“put that in a table”); (2) feedback via corrections;

(3) session-implicit, expressed only through interaction patterns; and (4) silent-exam with no reminders; only stage-4 rounds are scored. Preferences span output format, artifact naming, document structure, analytical style, and communication tone.

#### 3.4 Construction pipeline

A key challenge is producing scenarios at scale while preserving causal coherence across layers and realistic behavioral patterns. We use a three-stage pipeline combined with empirical distribution constraints (Figure 2).

- Stage 1: Seed construction. The first batch was authored by hand with cross-validation. For example, the startup outage scenario was iteratively refined until all four contradiction types were present, every answer required at least two sources, each update changed at least one previously correct answer, and all keys were unambiguous.

- Stage 2: Meta-specification induction. From the seeds we distilled a meta-specification encoding structural invariants: narrative patterns, contradiction-type ratios, bias-phrase insertion rules, and update-question binding constraints (e.g., exactly one C3 slot per scenario to prevent over-flagging; at least one update must flip a previously asked answer). This parallels constraining large-scale generation with high-quality exemplars (Wang et al., 2023; Bai et al., 2022).
- Stage 3: Batch generation with real-world grounding. Over 200 published empirical distributions (email volume (Radicati Group, 2024), commit patterns (GitHub, 2024), messaging activity (Golder & Macy, 2011), social network structure (Dunbar, 1992; Onnela et al., 2007)) constrain character profiles and generation along three authenticity axes. Workspace: documents follow domain-specific conventions and resemble system exports. Session: message timing follows scenario-specific diurnal curves, contact frequency follows a four-tier Dunbar-layer weighting (Dunbar, 1998) where intimate contacts appear roughly 100× more often than peripheral ones, and 30–50% of messages are irrelevant noise. Causal: all observable materials derive from a single Layer 0, preserving causal connections rather than independently fabricated text.
- Stage 4: Validation. Every scenario is validated at three levels. Structural: directory layout, question schema, file existence, session alternation, update integrity. Semantic: contradiction coverage, answer-key consistency, and linkage between observable traces and Layer 0 provenance. Control: bias-phrase placement and non-conflict-slot consistency. During development, these checks caught 37 specification errors before any evaluation began, so any observed failure reflects agent behavior rather than hidden scenario inconsistency.
- Stage 5: Refinement. The released 12-scenario benchmark is a refined subset of an earlier larger candidate pool: scenarios that did not pass all three validation passes were removed, ambiguous answer keys were rewritten, contradiction-type coverage was tightened, and the multi-choice / executable-check ratio was rebalanced per scenario. The result is a smaller but structurally cleaner evaluation surface in which every retained scenario satisfies the full set of design constraints rather than approximating them at scale.

#### 3.5 Scoring

Question-level scoring. Multi-choice questions present 7–9 candidates; the agent selects a subset via a \bbox{...} token. Credit is awarded only on exact-match; partial diagnostics (IoU, precision, recall, F1) are recorded but never contribute to the score, so selecting all options is not rewarded. Executable checks are binary pass/fail: a sandboxed shell command verifies whether a workspace claim holds in the actual files.

Composite reliability metrics. Alongside the Task Completion Rate (TCR, per-round mean score macro-averaged across scenarios, decomposed into MC and EC sub-scores), we report:

Success Cohesion (SC) = (S − k)/(N − 1), where S correct rounds form k consecutive success runs over N rounds; SC = 1 for one unbroken streak, SC = 0 for pass/fail alternation even at 50% TCR.

Failure Dispersion (FD) = 1 − (Sf − kf)/(N − 1) for Sf failures in kf runs; high FD indicates no prolonged collapse.

Robustness = SC × FD; the multiplicative form penalizes collapse on either axis. Composite Reliability Score = (TCR + Robustness)/2. Derivations and worked examples are in Appendix D.

### 4 Experiments

#### 4.1 Setup

Frameworks. We evaluate five AI agent frameworks spanning a range of design complexity: OpenClaw (enterprise-grade, TypeScript), Claude Code (Anthropic’s official CLI), NanoBot (minimalist, Python), PicoClaw (lightweight, Go), and MetaClaw (skill-driven self-evolving

framework (Xia et al., 2026b), built on top of OpenClaw as its executor). All are deployed as persistent assistants receiving unified conversational input on the same workspace. Architectural details are provided in Appendix B.

Models. We evaluate 18 models spanning proprietary APIs, community-accessible providers, and self-hosted deployments. Proprietary models include GPT-5.5, GPT-5.4, GPT-5.2, GPT-5.1 (OpenAI), Kimi-K2.5 (Moonshot AI), and Claude Opus-4.7, Claude Sonnet4.6, and Claude Haiku-4.5 (Anthropic). Community-accessible models via third-party APIs include GLM-5.1, GLM-4.7-Flash, Qwen3.6-Plus, DeepSeek-V4-Pro, Gemini-3.1-Pro, Mimo-V2.5-Pro, and Ling-2.6. Self-hosted models via local vLLM include Qwen3.6-35B, Qwen3.6-27B, and Gemma-4-31B. Cross-model comparisons fix the framework to OpenClaw except for the three Anthropic models, which are incompatible with OpenClaw and are therefore reported separately under Claude Code; cross-framework comparisons are conducted under GPT-5.1.

Evaluation protocol. Each scenario consists of a sequence of rounds, and each round is a full multi-turn agent interaction whose API-call count varies from a single call to several dozen depending on tool use and reasoning depth, making evaluation far costlier than single-inference benchmarks. All reported experiments use the complete CLAWARENA benchmark (12 scenarios, 337 rounds, 45 dynamic updates).

#### 4.2 Cross-model comparison

To isolate the effect of model capability, we fix the framework to OpenClaw and evaluate 15 models across the full benchmark (12 scenarios, 337 rounds), partitioned into proprietary API models and open-weight models. Table 3 shows results sorted by CRS within each group. We additionally report three Anthropic models (Claude Opus-4.7, Claude Sonnet-4.6, and Claude Haiku-4.5) in a separate provider-native block because they are incompatible with OpenClaw and can only be exercised through Anthropic’s Claude Code harness; their numbers are therefore not directly comparable to the OpenClaw-hosted rows above and serve as reference points across the current Anthropic model tier.

Among proprietary models (all OpenClaw), GPT-5.5 leads on CRS (68.28) and TCR (78.34%), with GPT-5.1 second (63.28). The proprietary range is 14.10 points, driven by the gap between the two GPT-5.x generations at the top and the remaining models clustered between 54 and 59 CRS. The provider-native block reports the Anthropic tier under Claude Code: Opus-4.7 (CRS 66.31, TCR 76.13, EC 80.58), Sonnet-4.6 (CRS 62.16, TCR 73.36, EC 77.69), and Haiku-4.5 (CRS 60.93, TCR 72.29, EC 75.62). These rows are not directly comparable to the OpenClaw rows above since the harness differs, but the size-ordered hierarchy is preserved across all three metrics, and EC values consistently track the stronger proprietary tier, suggesting that workspace grounding remains the relative strength of the Claude family.

Among open-weight models, Gemma-4-31B leads on CRS (63.80) and dominates MC across the table (81.05%), showing strong multi-choice reasoning is attainable at open-weight scale; GLM-5.1 ranks second (60.63). The open-weight range (24.98 points) is wider, reflecting greater architectural diversity, with the smaller Qwen3.6-27B posting the highest FD across the open-weight block (93.12) despite a mid-pack CRS, while GLM-4.7-Flash anchors the bottom (38.82 CRS). Gemma-4-31B comes within 4.5 CRS of GPT-5.5 but its EC trails by 6.2 points (73.14 vs. 79.34), indicating workspace grounding is the primary remaining gap.

MC and EC are only moderately correlated: DeepSeek-V4-Pro reaches 70.66 EC despite 57.89 MC, while Ling-2.6 shows the reverse (66.32 MC, 50.83 EC), so workspace grounding and reasoning are partially independent. The model-induced CRS range (29.46 points) exceeds the largest framework-induced range below (23.78 points under GPT-5.5), confirming that model capability dominates framework design.

#### 4.3 Cross-framework comparison

To isolate the effect of framework design, we evaluate four frameworks under three representative models (GPT-5.1, GPT-5.5, Kimi-K2.5) spanning two providers, examining whether framework-induced patterns generalize across model families.

- Table 3: Cross-model comparison on 12 scenarios under OpenClaw. Models are partitioned into proprietary (top) and open-weight (middle), each sorted by CRS. The bottom block reports three Anthropic models under their provider-native Claude Code harness (incompatible with OpenClaw); their numbers are not directly comparable with the OpenClaw rows above. Bold marks the best per column within each of the first two groups; underline marks the runner-up. † denotes self-hosted vLLM deployment; all others use provider APIs.

Model

TCR Robustness

CRS Avg MC EC Overall SC FD

Proprietary GPT-5.5 78.34 75.79 79.34 58.22 61.24 95.06 68.28

- GPT-5.1 70.33 75.79 68.18 56.24 58.96 95.37 63.28 GPT-5.4 71.22 71.58 71.07 46.77 51.51 90.78 58.99 Gemini-3.1-Pro 69.57 66.32 71.07 45.62 50.54 90.23 57.59 Qwen3.6-Plus 67.06 71.58 65.29 43.29 47.89 90.38 55.17

- GPT-5.2 65.88 61.05 67.77 42.49 47.21 90.01 54.18 Open-weight

Gemma-4-31B† 75.37 81.05 73.14 52.22 56.76 91.90 63.80 GLM-5.1 72.70 72.63 72.73 48.57 52.74 92.07 60.63 Mimo-V2.5-Pro 71.45 66.32 73.55 47.85 52.23 91.62 59.65 Kimi-K2.5 69.44 60.00 73.14 45.08 48.86 90.02 57.24 Qwen3.6-27B† 66.63 65.26 68.60 45.07 48.40 93.12 55.85 DeepSeek-V4-Pro 66.89 57.89 70.66 43.60 48.56 89.82 55.25 Qwen3.6-35B-A3B† 60.24 51.58 63.64 37.49 42.17 88.93 48.86 Ling-2.6 55.05 66.32 50.83 33.09 37.62 87.94 44.07 GLM-4.7-Flash 54.10 42.11 57.02 23.54 30.55 77.05 38.82

Provider-native (Claude Code, not comparable to rows above)

Claude Opus-4.7 76.13 65.26 80.58 57.47 60.06 94.06 66.31 Claude Sonnet-4.6 73.36 63.16 77.69 52.40 54.80 93.02 62.16 Claude Haiku-4.5 72.29 64.21 75.62 50.61 54.74 90.54 60.93

- Table 4: Cross-framework comparison. Three models are evaluated across all four frameworks. Bold marks the best per column within each model block; underline marks the runner-up.

TCR Robustness

Model Framework

CRS Avg MC EC Overall SC FD

OpenClaw 78.34 75.79 79.34 58.22 61.24 95.06 68.28 Claude Code 59.31 67.37 55.79 30.30 36.27 81.84 44.50 PicoClaw 77.15 74.74 78.10 56.66 59.49 94.74 66.87 NanoBot 73.00 72.63 73.14 50.98 55.24 90.76 61.88

GPT-5.5

OpenClaw 70.33 75.79 68.18 56.24 58.96 95.37 63.28 Claude Code 55.19 65.26 51.24 39.95 43.66 91.54 47.57 PicoClaw 71.51 73.68 70.66 56.86 59.90 94.90 64.18 NanoBot 67.95 73.68 65.70 51.80 55.09 94.04 59.88

GPT-5.1

OpenClaw 69.44 60.00 73.14 45.08 48.86 90.02 57.24 Claude Code 66.77 56.84 70.66 52.73 56.26 93.74 59.75 PicoClaw 65.58 65.26 65.70 49.64 53.72 92.41 57.61 NanoBot 45.99 49.47 44.63 29.18 32.97 88.50 37.58

Kimi-K2.5

Under GPT-5.1, PicoClaw leads on CRS (64.18), with OpenClaw (63.28) close behind. OpenClaw retains the best MC (75.79), while PicoClaw leads on EC (70.66). The GPT-5.1 framework range is 16.61 points, narrower than the 29.46-point model range above.

The GPT-5.5 block shows a substantially larger framework swing across the same four frameworks: OpenClaw leads at 68.28 CRS with PicoClaw close behind (66.87), NanoBot

- Table 5: Skill-driven self-evolution ablation. Each pair shares the same model and base OpenClaw configuration; the +MetaClaw variant adds skill injection. ∆CRS is the change relative to the baseline.

TCR Robustness

Model Config

CRS Avg MC EC SC FD

Baseline 78.34 75.79 79.34 61.24 95.06 68.28

GPT-5.5

+MetaClaw skills 78.86 75.79 80.58 61.90 95.19 68.89 (+0.61) GPT-5.1

Baseline 70.33 75.79 68.18 58.96 95.37 63.28

+MetaClaw skills 70.62 74.74 69.01 59.05 96.22 63.72 (+0.44) GLM-5.1

Baseline 72.70 72.63 72.73 52.74 92.07 60.63

+MetaClaw skills 73.29 69.47 74.79 54.65 92.19 61.82 (+1.19) Qwen3.6-Plus

Baseline 67.06 71.58 65.29 47.89 90.38 55.17

+MetaClaw skills 67.06 68.42 66.53 48.41 90.72 55.50 (+0.33)

trails at 61.88, and Claude Code collapses to 44.50, a 23.78-point intra-model spread. Both GPT-5.x configurations bottom out under Claude Code (47.57 for GPT-5.1, 44.50 for GPT-5.5), with the gap widening rather than narrowing as the model improves. This stands in sharp contrast to Kimi-K2.5, where Claude Code is in fact the strongest framework (59.75 CRS), suggesting that the OpenAI GPT family interacts poorly with the Anthropic Claude Code harness, whose tool-binding and prompt scaffold appear tuned to Claude-style instruction following.

The Kimi-K2.5 block shows framework effects generalize across families but differ in magnitude: Claude Code achieves the best CRS for Kimi-K2.5 (59.75), suggesting its workspacenative design compensates for structured-response instability elsewhere, while NanoBot lags (37.58), a drop more than twice the analogous NanoBot/PicoClaw gap under GPT-5.1. Framework robustness interacts non-linearly with model reliability.

#### 4.4 Skill-driven self-evolution

To isolate skill-based self-evolution, we fix model and framework and compare each model against its MetaClaw-enabled variant (Table 5, four matched pairs).

Across the four pairs, MetaClaw improves CRS by 0.33–1.19 without degrading TCR. The primary mechanism is Robustness: SC and FD both rise in every pair, so skill injection reduces isolated failures and suppresses long failure streaks rather than changing the raw correct-answer rate. The largest gain occurs for GLM-5.1 (+1.19 CRS, +1.79 Robustness) whose baseline Robustness is least consistent; the smallest (Qwen3.6-Plus, +0.33 CRS) on the most stable baseline. The strongest baseline (GPT-5.5, 68.28 CRS) still benefits (+0.61), confirming the overlay is not redundant for already-capable models. MetaClaw is a reliable but modest enhancer that narrows the gap to stronger baselines through consistency, not accuracy.

#### 4.5 Error analysis

We characterize agent failures with ten per-option case studies covering the MS, DU, P, and exec_check dimensions; scenario-specific evidence and round-level diagnostics are deferred to Appendix G (Figures 4–8). Six recurring failure modes emerge.

Compound claims are omitted while primary facts are captured. MS-R errors concentrate on options that require conjoining two clauses of the same source or promoting a crossstatement relation to a first-class claim. When such compound options appear, exact-match rates collapse even though every atomic fact is correctly recalled; the ceiling is therefore set by willingness to aggregate, not by evidence availability.

Belief revision fails asymmetrically across update packages. DU-R failures cluster on specific update installments rather than accumulating monotonically with update count. The decisive factor is the bridging inference each update demands (e.g., distinguishing partial from full restitution); updates that merely extend prior evidence are handled reliably while those that force a re-interpretation of earlier claims are not.

Framework choice does not compensate for model-level output-format gaps. When the same model fails a strict-schema exec_check task under one framework, it fails under the alternate framework with identical schema-compliance errors, even as other models clear the same task under the first framework. Framework contribution is substantive for in-context reasoning discipline but limited for literal structural output.

Compound structural constraints define a capability ceiling. A small cluster of exec_check rounds attains 0/8 pass rates across all eight configurations. These rounds share a common profile: correct factual content plus a conjunction of structural constraints (timezone-aware timestamps, typed JSON fields with provenance arrays, enum-restricted labels, exact toplevel key names). The underlying facts are usually identified; the failure is compound output-format compliance that MC metrics cannot detect.

Implicit-preference compliance decays as checkers tighten. Models that pass a lenient mid-scenario preference check often fall off once a later round layers additional literal constraints onto the same convention. Passing a softer preference check does not predict compliance with a stricter successor, indicating brittle rather than internalized preference tracking.

MC and EC skills remain loosely correlated. Set-selection leaders do not dominate exec_check, and models with sporadic MC extraction failures can still succeed on scriptgeneration rounds when the expected schema is listed verbatim. Workspace grounding draws on file parsing, shell construction, and literal-key discipline, which are only weakly predicted by multi-choice reasoning accuracy, supporting the two-format design of CLAWARENA. Performance variation across domains exceeds 60% for every model tested; the Chinese-language enterprise scenario disproportionately favors models with stronger multilingual training, and the hardest scenario keeps every configuration below 30% CRS.

### 5 Conclusion

We introduced CLAWARENA, a benchmark for AI agents in evolving information environments. Across five frameworks and 18 models, model capability dominates framework design in CRS range (29 vs. 24 points), MetaClaw’s skill overlay improves Robustness without degrading accuracy across all four tested families, belief revision difficulty is governed by update specificity rather than volume, and per-option diagnostics expose failure modes that aggregate scores conceal. A natural extension moves beyond staged updates toward live, unconstrained environments where agents formulate their own queries against real-time sources.

### References

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Constitutional AI: Harmlessness from AI feedback. arXiv preprint arXiv:2212.08073, 2022.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Liao, Yanzhou Chen, Sijie Liu, Wenxuan Ji, Yiduo Liu, Haonan Wen, Xiuqi Liu, et al. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2024. URL https://arxiv.org/ abs/2308.14508.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Jingyi Yang, Penghui Yang, Zhixiong Zhang, Xilin Wei, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang. Wildclawbench, 2026. URL https://github.com/ InternLM/WildClawBench.

Robin I. M. Dunbar. Neocortex size as a constraint on group size in primates. Journal of Human Evolution, 22(6):469–493, 1992.

Robin I. M. Dunbar. The social brain hypothesis. Evolutionary Anthropology, 6(5):178–190, 1998.

Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms. arXiv preprint arXiv:2504.11536, 2025.

GitHub. Octoverse: The state of open source. Technical report, GitHub, Inc., 2024. URL https://octoverse.github.com/.

Scott A. Golder and Michael W. Macy. Diurnal and seasonal mood vary with work, sleep, and daylength across diverse cultures. Science, 333(6051):1878–1881, 2011.

Siwei Han, Haonian Ji, Siyang Xin, Juanquan Shi, Shi Qiu, Xinyu Ye, Peng Xia, Jiaqi Liu, Zhaorun Chen, Yiyang Zhou, Linjie Li, Lijuan Wang, and Huaxiu Yao. Paper2figure: A multi-agent collaborative system for figure generation towards academic research paper. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.

Cheng-Yu Hsieh, Shih-Hung Li, Chih-Kuan Yeh, Hadi Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Ruler: What’s the real context size of your long-context language models? In Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=kIoBbc76Sy.

Haonian Ji, Shi Qiu, Siyang Xin, Siwei Han, Zhaorun Chen, Dake Zhang, Hongyi Wang, and Huaxiu Yao. From EduVisBench to EduVisAgent: A benchmark and multi-agent framework for reasoning-driven pedagogical visualization. arXiv preprint arXiv:2505.16832, 2025. URL https://arxiv.org/abs/2505.16832.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=VTF8yNQM66.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Chenxin Li, Zhengyang Tang, Mingxin Huang, Yunlong Lin, Shijue Huang, Shengyuan Liu, Bowen Ye, Rang Li, Lei Li, Benyou Wang, and Yixuan Yuan. Claw-eval-live: A live agent benchmark for evolving real-world workflows, 2026a. URL https://arxiv.org/ abs/2604.28139.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for" mind" exploration of large language model society. Advances in neural information processing systems, 36:51991–52008, 2023.

Xiangyi Li, Kyoung Whan Choe, Yimin Liu, Xiaokun Chen, Chujun Tao, Bingran You, Wenbo Chen, Zonglin Di, Jiankai Sun, Shenghan Zheng, Jiajun Bao, Yuanli Wang, Weixiang Yan, Yiyuan Li, and Han chung Lee. Clawsbench: Evaluating capability and safety of llm productivity agents in simulated workspaces, 2026b. URL https://arxiv.org/abs/2604.

05172.

Jiaqi Liu, Kaiwen Xiong, Peng Xia, Yiyang Zhou, Haonian Ji, Lu Feng, Siwei Han, Mingyu Ding, and Huaxiu Yao. Agent0-vl: Exploring self-evolving agent for tool-integrated vision-language reasoning. arXiv preprint arXiv:2511.19900, 2025.

Jiaqi Liu, Zipeng Ling, Shi Qiu, Yanqing Liu, Siwei Han, Peng Xia, Haoqin Tu, Zeyu Zheng, Cihang Xie, Charles Fleming, Mingyu Ding, and Huaxiu Yao. Omni-simplemem: Autoresearch-guided discovery of lifelong multimodal agent memory. arXiv preprint arXiv:2604.01007, 2026a.

Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. Simplemem: Efficient lifelong memory for llm agents. arXiv preprint arXiv:2601.02553, 2026b.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2024. URL https://arxiv.org/abs/2308.03688.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753, 2024. URL https://arxiv.org/abs/2402.17753.

Fanqing Meng, Lingxiao Du, Zijian Wu, Guanzheng Chen, Xiangyan Liu, Jiaqi Liao, Chonghe Jiang, Zhenglin Wan, Jiawei Gu, Pengfei Zhou, Rui Huang, Ziqi Zhao, Shengyuan Ding, Ailing Yu, Bo Peng, Bowei Xia, Hao Sun, Haotian Liang, Ji Xie, Jiajun Chen, Jiajun Song, Liu Yang, Ming Xu, Qionglin Qiu, Runhao Fu, Shengfang Zhai, Shijian Wang, Tengfei Ma, Tianyi Wu, Weiyang Jin, Yan Wang, Yang Dai, Yao Lai, Youwei Shu, Yue Liu, Yunzhuo Hao, Yuwei Niu, Jinkai Huang, Jiayuan Zhuo, Zhennan Shen, Linyu Wu, Hannah Yao, Charles Chen, Cihang Xie, Yuyin Zhou, Jiaheng Zhang, Zeyu Zheng, Mengkang Hu, and Michael Qizhe Shieh. Clawmark: A livingworld benchmark for multi-turn, multi-day, multimodal coworker agents, 2026. URL https://arxiv.org/abs/2604.23781.

Grégoire Mialon, Roberto Dessì, María Lomeli, Julen Etxaniz, Liam Magee, Federico Belotti, Philip Quirke, Antoine Soubrier Lopez, Luca Beurer-Kellner, Riccardo Pucella, et al. Gaia: A benchmark for general ai assistants. arXiv preprint arXiv:2311.12983, 2024. URL https://arxiv.org/abs/2311.12983.

Jukka-Pekka Onnela, Jari Saramäki, Juhani Hyvönen, György Szabó, David Lazer, Kimmo Kaski, János Kertész, and Albert-László Barabási. Structure and tie strengths in mobile communication networks. Proceedings of the National Academy of Sciences, 104(18):7332– 7336, 2007.

OpenAI. Introducing o3 and o4-mini. Blog post, https://openai.com/index/ introducing-o3-and-o4-mini/, 2025.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025.

PinchBench Contributors. PinchBench: A leaderboard for OpenClaw coding agents. https: //pinchbench.com/, 2026. Accessed 2026-05-09.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. In The Twelfth International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2307.16789.

Qwen Team and Alibaba Data. QwenClawBench: Real-user-distribution benchmark for OpenClaw agents, April 2026. URL https://github.com/SKYLENAGE-AI/QwenClawBench.

Radicati Group. Email statistics report, 2024–2028. Technical report, The Radicati Group, Inc., 2024.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, et al. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229, 2025.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. In Transactions of the Association for Computational Linguistics, volume 10, pp. 539–554, 2022. URL https://aclanthology. org/2022.tacl-1.31/.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pp. 13484–13508, 2023.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024.

Peng Xia, Kaide Zeng, Jiaqi Liu, Can Qin, Fang Wu, Yiyang Zhou, Caiming Xiong, and Huaxiu Yao. Agent0: Unleashing self-evolving agents from zero data via tool-integrated reasoning. arXiv preprint arXiv:2511.16043, 2025.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026a.

Peng Xia, Jianwen Chen, Xinyu Yang, Haoqin Tu, Jiaqi Liu, Kaiwen Xiong, Siwei Han, Shi Qiu, Haonian Ji, Yuyin Zhou, et al. Metaclaw: Just talk–an agent that meta-learns and evolves in the wild. arXiv preprint arXiv:2603.17187, 2026b.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. arXiv preprint arXiv:2305.13300, 2024a. URL https://arxiv.org/abs/2305.13300.

Tianbao Xie, Shuyin Zou, Zhaoyang Shi, Jieren Li, Qian Chen, Xun He, Quzhe Huang, Xiaotian Wu, Hao Yue, Yu Zhong, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024b. URL https://arxiv.org/abs/2404.07972.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multihop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018. URL https://aclanthology.org/D18-1259/.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Bowen Ye, Rang Li, Qibin Yang, Yuanxin Liu, Linli Yao, Hanglong Lv, Zhihui Xie, Chenxin An, Lei Li, Lingpeng Kong, Qi Liu, Zhifang Sui, and Tong Yang. Claw-eval: Towards trustworthy evaluation of autonomous agents, 2026. URL https://arxiv.org/abs/2604. 06132.

Z.AI. ZClawBench: A hybrid three-level evaluation benchmark for OpenClaw agents. https://huggingface.co/datasets/zai-org/ZClawBench, 2026. Accessed 2026-05-09.

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018. URL https://aclanthology.org/P18-1205/.

Yuxuan Zhang, Yubo Wang, Yipeng Zhu, Penghui Du, Junwen Miao, Xuan Lu, Wendong Xu, Yunzhuo Hao, Songcheng Cai, Xiaochen Wang, Huaisong Zhang, Xian Wu, Yi Lu, Minyi Lei, Kai Zou, Huifeng Yin, Ping Nie, Liang Chen, Dongfu Jiang, Wenhu Chen, and Kelsey R. Allen. Clawbench: Can ai agents complete everyday online tasks?, 2026. URL https://arxiv.org/abs/2604.08523.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. arXiv preprint arXiv:2308.10144, 2023. URL https://arxiv.org/abs/2308.10144.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2024. URL https: //arxiv.org/abs/2307.13854.

### Appendix

- A Data Sample Overview 15
- B Framework Details 15
- C Implementation Details 16
- D Scoring Details 17
- E Benchmark Statistics 18

- E.1 Overall scale . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- E.2 Token composition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- E.3 Question statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- E.4 Staged updates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- E.5 Per-scenario breakdown . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20

- F Meta-Specification Templates 20

- F.1 Template 1: Execution Guide . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- F.2 Template 2: Narrative Bible (Layer 0) . . . . . . . . . . . . . . . . . . . . . . . 21
- F.3 Template 3: Evidence Emission Map . . . . . . . . . . . . . . . . . . . . . . . 21
- F.4 Template 4: Workspace Specification (Layer 1) . . . . . . . . . . . . . . . . . 22
- F.5 Template 5: Session Specification (Layer 2) . . . . . . . . . . . . . . . . . . . 22
- F.6 Template 6: Evaluation Specification (Layer 3) . . . . . . . . . . . . . . . . . 22
- F.7 Template 7: Dynamic Update Specification (Layer 4) . . . . . . . . . . . . . . 23

- G Per-Option Case Studies 24

### A Data Sample Overview

Figure 3 illustrates the surface form of CLAWARENA scenarios across the diverse professional contexts represented in the benchmark. Each tile shows the workspace, multi-channel session sources, the type of evaluation question posed, and a representative evidence chain through which the agent must navigate to recover the hidden ground truth. The gallery is intended as a qualitative orientation to the data style of CLAWARENA; the structural framework, evaluation dimensions, and scoring summarized in Figure 1 are formalized in Section 3.

### B Framework Details

We evaluate five AI agent frameworks with representative design differences, all deployed

- as persistent assistants receiving unified conversational input and operating on the same workspace environment.

OpenClaw (TypeScript, Agent Client Protocol architecture) provides enterprise-grade session management and tool binding. It supports multi-channel session routing, structured memory with semantic retrieval, and configurable tool pipelines.

[Figure 4]

- Figure 3: Cross-domain data sample gallery from CLAWARENA. Each tile presents one scenario in a distinct professional context, highlighting the workspace, session sources, evaluation question, and an evidence chain that traverses conflicting and time-evolving observations. The center logo reflects the benchmark’s adversarial spirit: agents must “claw” through conflicting evidence to reach the ground truth.

MetaClaw (skill-driven self-evolving framework) maintains a growing repository of procedural skills distilled from prior failure trajectories; these skills are retrieved and injected into the agent’s prompt on each round without modifying model weights. In this work it is built on top of OpenClaw as its executor, so that differences in downstream behaviour isolate the effect of skill injection over the same tool-binding and memory stack.

Claude Code (Anthropic’s official CLI) is optimized for code and file workspace operations with terminal-native interaction. It uses a single long-context window and relies on the model’s in-context reasoning rather than external memory modules.

NanoBot (Python, ∼1,700 lines) follows a minimalist design with a MEMORY.md + HISTORY.md dual-layer memory. Its simplicity makes it a useful lower-bound baseline for framework complexity.

PicoClaw (Go, <10MB runtime memory) uses an event-driven architecture with dateorganized memory. It prioritizes low resource consumption over retrieval sophistication.

### C Implementation Details

Model providers. Table 6 lists the provider routing for every model evaluated in Section 4. Proprietary and community-accessible models are served through their respective commercial APIs; self-hosted models are deployed locally via vLLM. Claude Haiku-4.5 is the sole model that cannot be served into OpenClaw’s tool-binding stack, and is instead exercised through Anthropic’s provider-native Claude Code harness using a Claude OAuth credential. No third-party API endpoint is used for this model.

Self-hosted vLLM deployment. The two open-weight models evaluated locally are served with vLLM on 4×NVIDIA RTX 6000 Ada Generation GPUs (48GB each) under tensor parallelism of 4. Both share a maximum sequence length of 204,800 tokens and a GPU memory utilization target of 0.92. Model-specific parser and quantization settings are summarized in Table 7.

Table 6: Provider routing for all evaluated models.

Provider Access Models

OpenAI Codex API GPT-5.5, GPT-5.4, GPT-5.2 Azure OpenAI Azure endpoint GPT-5.1, Kimi-K2.5 Anthropic Claude OAuth Claude Opus-4.7, Claude Sonnet-4.6, Claude Haiku-4.5 OpenRouter OpenRouter API GLM-5.1 (z-ai), GLM-4.7-Flash (z-ai), Qwen3.6-Plus

(qwen), DeepSeek-V4-Pro (deepseek), Gemini-3.1-Pro (google), Mimo-V2.5-Pro (xiaomi), Ling-2.6 (inclusionai)

Self-hosted vLLM Local (4×GPU) Qwen3.6-35B-A3B, Qwen3.6-27B, Gemma-4-31B-it

- Table 7: vLLM serving configuration for self-hosted models. TP denotes tensor parallelism; max-model-len is the served context length; tool-call-parser and reasoning-parser select the vLLM output-format adapters.

Model TP max-model-len tool-call-parser reasoning-parser KV cache

Qwen3.6-35B-A3B 4 204,800 qwen3_coder qwen3 bf16 Gemma-4-31B-it 4 204,800 gemma4 gemma4 fp8

### D Scoring Details

Multi-choice (set-selection). Each question presents 7–9 candidate statements; the agent must select the correct subset by enclosing its answer in a \bbox{} token. Credit is awarded only when the selected set matches the gold set exactly (exact-match). Partial diagnostics (IoU, precision, recall, and F1) are recorded per round for analysis but do not contribute to any reported score, ensuring that selecting all options to guarantee a non-zero partial credit is never rewarded.

Executable checks. Executable-check questions are binary pass/fail: a sandboxed shell command tests whether a specific workspace-level claim (e.g., that a file contains a particular entry, or that two documents report consistent timestamps) holds in the actual files. Executable-check scores are independent of multi-choice scores and test workspace grounding rather than reasoning.

Composite Reliability Score (CRS). Let a configuration produce a sequence of N round scores s1, . . . , sN ∈ {0,1}. Define S = ∑i si (correct rounds), k = number of maximal consecutive success runs, Sf = N − S (failed rounds), and kf = number of maximal consecutive failure runs. The four aggregate metrics are:

TCR = S/N. Mean correctness across all rounds.

SC (Success Cohesion) = (S − k)/(N − 1) if N > 1, else 0. Measures how much correct performance is concentrated into long unbroken runs. A configuration that passes all rounds in one streak achieves SC = 1; one that alternates pass/fail achieves SC = 0 even at 50% TCR.

FD (Failure Dispersion) = 1 − (Sf − kf)/(N − 1) if N > 1, else 1. Measures how evenly failures are spread across the sequence. High FD indicates no prolonged collapse; a single long failure run drives FD toward 0.

Robustness = SC × FD. Multiplicative form ensures that a collapse on either axis (many isolated successes or one long failure streak) substantially reduces the score.

CRS = (TCR + Robustness)/2. Gives equal weight to raw correctness and behavioral consistency. All metrics are macro-averaged across scenarios before CRS is computed.

Worked example. For N = 5 with scores [1,0,1,0,1]: S = 3, k = 3, Sf = 2, kf = 2. Then SC = (3 − 3)/4 = 0, FD = 1 − (2 − 2)/4 = 1, Robustness = 0, TCR = 0.6, CRS = 0.3.

Although TCR is 60%, the alternating pattern yields zero Robustness, pulling CRS well below TCR. For scores [1,1,1,0,0]: S = 3, k = 1, Sf = 2, kf = 1. SC = (3 − 1)/4 = 0.5, FD = 1 − (2 − 1)/4 = 0.75, Robustness = 0.375, CRS = 0.488. The same TCR but coherent success and contained failure yields a substantially higher CRS.

### E Benchmark Statistics

This section characterizes CLAWARENA in full statistical detail, covering token composition, question shape, staged-update structure, and per-scenario breakdowns referenced throughout Section 4. Cross-model comparisons fix the framework to OpenClaw; cross-framework comparisons fix the model to GPT-5.1. Each round is a full multi-turn agent interaction whose API-call count varies from a single call to several dozen depending on tool use and reasoning depth.

Tokenization. Token counts are produced with OpenAI’s cl100k_base byte-pair-encoding tokenizer (the encoding used by GPT-4/GPT-4o and by tiktoken as a cross-model reference). For plain artefacts (workspace files, question strings, feedback strings, markdown transcripts) the raw UTF-8 bytes are encoded directly. For jsonl session files (whose on-disk layout differs sharply across frameworks) we first parse the transcript into a normalized sequence of {role, content} messages, concatenating text parts and flattening tool-call arguments, and then re-render the sequence into the ChatML template (<|im_start|>role\n content<|im_end|>) before tokenization. This parse-then-render step approximates the prompt payload that the inference backend assembles from the transcript, rather than charging the model for the per-record JSON braces, keys (role, content, timestamp, ...) and metadata that each framework happens to persist on disk. Markdown transcripts (used by Claude Code and NanoBot for per-turn history) are already in near-prompt form and are counted verbatim. The same tokenizer is applied uniformly across all frameworks and artefact types, so the per-framework differences reported in Table 8 reflect genuine differences in session-format verbosity (tool-call encoding, per-turn metadata volume, history containerization) rather than measurement artefacts.

#### E.1 Overall scale

CLAWARENA spans 12 scenarios and 337 rounds. Of these, 45 rounds (13.4%) expose at least one staged update and 24 rounds (7.1%) carry an implicit user preference. Perscenario round counts range from 24 to 30 (mean 28.1), giving balanced coverage across the represented professional contexts. Under the OpenClaw layout, the cumulative input across the benchmark totals 4,739,550 tokens, dominated by workspace content; the other three frameworks fall within a 0.4% band of this figure (see Table 8).

#### E.2 Token composition

Table 8 decomposes the total token budget by source under each of the four compatible frameworks. Workspace content (baseline files plus staged workspace updates) accounts for 90.1–90.3% of tokens across frameworks, reflecting the benchmark’s focus on file-grounded reasoning rather than long dialogue histories. Session-side updates contribute a further 2.6% of tokens, with the three jsonl-based frameworks (Claude Code, NanoBot, PicoClaw) carrying marginally larger session-update budgets than OpenClaw because their tool-call argument payloads survive the parse-and-render step intact, whereas Claude Code and NanoBot transcribe per-turn history as markdown and PicoClaw as jsonl into the workspace message-log tree. The entire authored conversational surface (questions, feedback, explicit preferences) consumes only 3.5% of tokens yet drives every scored interaction.

#### E.3 Question statistics

The 337 rounds partition into 242 executable-check (exec_check, 71.8%) and 95 multi-choice (multi_choice, 28.2%) rounds. All multi-choice rounds are multi-answer. Table 9 summarizes the shape of multi-choice prompts and the feature coverage of executable-check prompts.

- Table 8: Token distribution across input sources in CLAWARENA, reported independently for each of the four compatible frameworks. All counts use the cl100k_base tokenizer; jsonl session files are parsed into a normalized message sequence and rendered into the ChatML template before counting, while markdown transcripts and all non-session artefacts are counted verbatim (see Appendix E narrative).

Source

OpenClaw Claude Code NanoBot PicoClaw Tokens % Tokens % Tokens % Tokens %

Main session 10,457 0.221 4,170 0.088 7,976 0.168 7,989 0.168 History sessions 158,090 3.335 167,139 3.531 167,139 3.519 159,359 3.361 Workspace (base) 1,483,834 31.307 1,472,224 31.103 1,484,686 31.260 1,484,566 31.305 Workspace update 2,797,037 59.015 2,797,133 59.095 2,797,037 58.891 2,797,037 58.983 Session update 122,271 2.580 124,800 2.637 124,800 2.628 125,343 2.643 Questions 101,881 2.149 101,881 2.152 101,881 2.145 101,881 2.148 Feedback 64,271 1.356 64,271 1.358 64,271 1.353 64,271 1.355 Preferences 1,709 0.036 1,709 0.036 1,709 0.036 1,709 0.036

Total 4,739,550 100.000 4,733,327 100.000 4,749,499 100.000 4,742,155 100.000

- Table 9: Multi-choice prompt shape and executable-check feature coverage in CLAWARENA. MC statistics are computed over all 95 multi-choice rounds; EC statistics are computed over all 242 executable-check rounds.

Dimension Metric Mean Min Max Multi-choice Options per round 6.38 5 10

Correct answers per round 4.46 2 8

Executable-check Sandbox timeout (s) 40.2 30 60 expect_exit coverage (%) 100.0 — expect_stdout coverage (%) 0.4 — —

Every multi-choice round requires selecting a subset rather than a single option, eliminating single-answer degeneracy; the mean 4.46-of-6.38 selection ratio makes random guessing expensively uninformative. Executable-check rounds universally assert a specific exit status against a bounded timeout, with expect_stdout used sparingly (1/242) so that pass/fail outcomes hinge on workspace state rather than output string formatting. Implicit preferences appear in 24 rounds (7.1%), concentrated in P-R-category scenarios.

E.4 Staged updates

Across the 45 update-bearing rounds there are 83 update events touching 25 distinct workspace files. Table 10 breaks down the update events by target scope and the filelevel actions by type.

- Table 10: Staged-update distribution. Target reports the 83 update events; Action reports the 25 file-level update operations triggered by those events.

Facet Category Count % Target (events) Workspace-targeted 42 50.6

Session-targeted 37 44.6 Group-targeted 4 4.8

Action (files) append 14 56.0 new 11 44.0

The roughly balanced split between workspace- and session-targeted updates exercises both belief-revision pathways tested by DU-R: modifying or extending evidence in the filesystem versus introducing new turns into the conversation history. File-level actions are similarly balanced between append (extending existing artefacts) and new (introducing previously

unseen files), ensuring that neither pure accumulation nor pure discovery dominates the revision workload. Per-round file mutations average 0.30, with a maximum of four files modified in a single round.

#### E.5 Per-scenario breakdown

- Table 11 lists round counts, question-type mix, preference and update coverage, update and workspace file counts, and total tokens for each of the 12 scenarios under the OpenClaw layout. All scenarios fall within a narrow 24–30 round band, but token totals vary by nearly two orders of magnitude due to one outlier scenario (hil_s1) whose staged workspace updates inject the bulk of the benchmark’s workspace tokens.

- Table 11: Per-scenario breakdown under the OpenClaw layout. Columns: rounds, multichoice rounds (MC), executable-check rounds (EC), rounds with a preference (w/Pref), rounds exposing updates (w/Upd), total update events (Upd), files touched by updates (UpdF), baseline workspace files (WSF, excluding framework-injected session transcripts), and total tokens.

Scenario Rounds MC EC w/Pref w/Upd Upd UpdF WSF Tokens hil_s1 24 8 16 3 4 8 25 24 4,344,165 hil_c7 28 8 20 2 3 6 9 12 44,210 hil_d3 30 8 22 2 4 8 10 12 44,137 hil_e4 24 7 17 2 3 6 6 11 40,112 hil_g4 27 8 19 2 3 7 8 10 37,429 hil_g1 30 8 22 3 4 7 8 10 36,077 hil_h3 27 8 19 2 4 6 7 10 35,038 hil_j1 30 8 22 2 4 5 6 8 34,823

- hil_f3 30 8 22 2 4 6 7 11 34,061 hil_i2 30 8 22 2 4 7 7 10 30,940
- hil_g3 30 8 22 0 4 6 7 7 30,474 hil_f7 27 8 19 2 4 11 7 10 28,084

hil_s1 alone accounts for 91.7% of benchmark tokens, driven chiefly by its staged workspace updates (25 files) atop an already-large 24-file baseline; the remaining 11 scenarios collectively consume 395,385 tokens (8.3%) and share a compact 7–12-file workspace baseline. This bimodal token distribution is intentional: it preserves a representative workspace-heavy stress scenario while keeping the aggregate evaluation cost tractable. Baseline workspace file counts (WSF) are computed from the OpenClaw layout after excluding framework-injected session-equivalent directories; the three non-native-multi-session frameworks (Claude Code, NanoBot, PicoClaw) instead transcribe per-turn history into message_logs/ and sessions/ subtrees, which are classified as session content rather than workspace baseline and so do not inflate WSF. hil_f7 carries the densest update schedule (11 update events in 27 rounds) with 7 update-touched files atop a 10-file baseline, isolating the effect of high-frequency session-side belief revision. The multi-choice and executable-check quotas per scenario are nearly uniform (MC ∈ {7,8}, EC ∈ {16,22}), preventing any single scenario from biasing the exec_check-dominated question distribution reported in the previous subsection.

### F Meta-Specification Templates

Each ClawArena scenario is produced from a meta-specification that encodes structural invariants across seven template documents: the execution guide, narrative bible (Layer 0), evidence emission map, and Layers 1–4. Below we present all seven templates (Section F.1).

#### F.1 Template 1: Execution Guide

The execution guide is the entry point for scenario construction, linking all six layers and defining the build workflow.

###### Execution Guide Template (GUIDE.md)

- 1. Task Overview Task ID / Domain / Core evaluation goals / Final output target
- 2. Spec File Index

- layer0-narrative.md truth baseline read first
- layer1-workspace.md workspace plan read second
- layer2-sessions.md session plan read third
- layer3-eval.md round plan read fourth
- layer4-dynamic.md update plan read fifth

- 3. Role and Session Table Each row maps a character role to a communication channel, session filename, and whether it appears in the initial release or a later update.
- 4. Contradiction and Bias Quick-Reference Lists every contradiction (C1–C4) and bias (B1–B2) with the round where it first becomes visible and the round where reversal evidence arrives.
- 5. Eight-Step Execution Workflow Step 0: Read all layers, generate UUIDs, freeze filenames. Step 1: Create fixed agent files + initial workspace files. Step 2: Write session intermediate JSON files. Step 3: Build .jsonl session files from intermediate JSON. Step 4: Create update source files under updates/. Step 5: Write questions.json. Step 6: Register sessions and update metadata. Step 7: Append to all_tests.json. Step 8: Run structural, semantic, and control validation.
- 6. Mandatory Checks Data text is English; questions.json is a single group object; updatecreated sessions carry a channel field; initial sessions are registered in sessions.json.

- Table 12: Execution guide template, linking all six specification layers and defining the build workflow.

- F.2 Template 2: Narrative Bible (Layer 0)

- Layer 0 defines the hidden ground truth that is never shown to the evaluated system. It structures the objective timeline, role-level truth gaps, contradiction map, bias design, and evaluation traps.

Narrative Bible Template (Layer 0)

- 1. Scene Summary Task ID / Domain / Time span / Main protagonist / Core benchmark factors (MS, DU, P).
- 2. Objective Timeline Each row: Time | Objective event | What actually happened | Who knew at that time.
- 3. Role-Level Truth vs Self-Narrative For each character: Objective position, Public narrative, Private narrative, and Why the gap exists.
- 4. Contradiction Map Each row: ID | Description | Source A claim + location | Source B claim + location | Objective truth | Visible rounds | Reversal. Exactly one slot (C3) is NON-CONFLICT.
- 5. Agent Historical Bias Design Each row: Bias ID | Session and phase | Exact verbatim phrase | Why misled | Reversal trigger.
- 6. Eval Trap Table Each row: Trap ID | Related contradiction(s) | Related bias(es) | Round(s) | What shallow agents miss.
- 7. Writer Constraints Only introduce listed contradictions. Every key judgment needs evidence in ≥2 independent sources. Timestamps internally consistent. Bias phrases verbatim.

Table 13: Narrative bible template (Layer 0), defining the hidden ground truth never shown to the evaluated system.

- F.3 Template 3: Evidence Emission Map

The evidence emission map translates each objective event into observable traces across multiple channels, ensuring that no single source contains the full truth.

###### Evidence Emission Map Template

- 1. Event-Level Map Each row: Event ID | Objective truth | Official workspace evidence | Private DM evidence | Group-session evidence | Update-only evidence | What remains hidden early.
- 2. Source Responsibility Map For each source type: what it is allowed to establish vs. what it should not fully settle, ensuring information fragmentation.
- 3. Contradiction Seeding Each row: Contradiction ID | Source A claim | Source B claim | Source of truth | Earliest visible round.
- 4. Agent Bias Hooks Each row: Bias ID | Session and loop | Why reasonable at that point | Exact phrase to embed.

- Table 14: Evidence emission map template, translating objective events into multi-channel observable traces.

- F.4 Template 4: Workspace Specification (Layer 1)

- Layer 1 specifies the workspace files visible to the agent, including fixed agent configuration files and scenario-specific documents with their timing and noise design.

Workspace Specification Template (Layer 1)

- 1. Fixed Agent Files Five standard files bootstrapping agent behavior: AGENTS.md (startup behavior), IDENTITY.md (agent identity), SOUL.md (working principles: cautious attribution, evidence-first reasoning), USER.md (participants and channels), TOOLS.md (available tools and rules).
- 2. Scenario-Specific Files Each row: File | Type | Initial or update | Key facts carried | Token estimate.
- 3. File Timing Summary Each row: File | First visible round | Why delayed or immediate.
- 4. Near-Signal Noise Design For each noise file: why it looks relevant, and why it should not settle the core contradiction.
- 5. Total Workspace Estimate Initial workspace tokens, update-added tokens, and balance notes.

Table 15: Workspace specification template (Layer 1), defining all files visible to the agent with timing and noise controls.

F.5 Template 5: Session Specification (Layer 2)

- Layer 2 specifies all session histories (main session and history sessions across DMs and group channels) with per-loop signal/noise labels and phase structure.

Session Specification Template (Layer 2)

- 1. Main Session Channel main; Loop 0 user message provides scene background and full history-session roster; Loop 0 assistant reply must state it will inspect workspace and use session-history tools.
- 2. History Session Roster Each row: Session name | Channel | DM/Group | Session ID placeholder | Phase count | Token estimate.
- 3. Per-Session Design For each session: meta (channel, DM/Group, participants, placeholder), then per-phase loop entries. Each loop specifies: signal/noise label, user message, agent tool calls, agent reply, and contradiction or bias effect.
- 4. Session Rules History sessions may use read and light exec. History sessions should not use session-listing tools. Group session user text includes full channel prefix; DM text stays plain.

Table 16: Session specification template (Layer 2), defining multi-channel session histories with per-loop signal/noise structure.

F.6 Template 6: Evaluation Specification (Layer 3)

- Layer 3 specifies the evaluation rounds, reversal matrix, and personalization scoring.

###### Evaluation Specification Template (Layer 3)

- 1. Round Inventory Each row: Round | Question type | Main skill tested | Depends on update? | Reversal?
- 2. Round Specs For each round: type, question goal, evidence required, correct answer logic, and shallow failure mode.
- 3. Reversal Matrix Each row: Earlier round | Later round | What changed | Why the earlier answer should be revised.
- 4. Personalization Scoring Notes Each row: Round | Preference in scope | What should change in the correct answer.
- 5. Evidence Coverage Check Every correct option has a named evidence source. At least one round asks about epistemic limits. At least one round asks about revision after new information.

- Table 17: Evaluation specification template (Layer 3), defining rounds, reversals, and personalization scoring.

- F.7 Template 7: Dynamic Update Specification (Layer 4)

- Layer 4 specifies the staged updates that inject new evidence over time, including action lists and runtime checks.

Dynamic Update Specification Template (Layer 4)

- 1. Update Summary Each row: Update ID | Trigger round | Goal | New sessions? | New workspace files?
- 2. Action Lists Per update: a JSON array of actions, each specifying type (workspace/session), action (new/append), path, source, and optionally channel for new sessions.
- 3. Source File Notes Each row: Source file | Update | Type | What it reveals | Must match existing layer section.
- 4. Runtime Checks new session actions include channel. Initial and appended session filenames are consistent. Update-introduced facts directly support the intended reversal.

- Table 18: Dynamic update specification template (Layer 4), defining staged evidence injection and runtime validation.

### G Per-Option Case Studies

Figures 4–8 present ten per-option case studies drawn from CLAWARENA’s 12 scenarios, covering interaction categories MS-R, DU-R, P-R, and exec_check across the security, clinical, HR, and e-commerce domains. Each case pairs the question statement and ground-truth subset with per-configuration selections, extracted failure patterns, and a key insight.

[Figure 5]

- Figure 4: Case 1 (MS-R, NexaFlow API breach, q7 + q25): both GPT-5.1 configurations omit the top-level 2,340-record confirmation in q7 despite correctly absorbing five other log-derived facts; q25 exhibits five distinct failure modes across models, with only GPT-5.1 (OC), Mimo V2.5, and Ling-2.6 achieving a perfect reliability ranking. Case 2 (exec_check, hil_g4/q3 and hil_i2/q4): GPT-5.1 under both OpenClaw and Claude Code fails both tasks with identical schema-compliance errors, while six OpenClaw-hosted models pass hil_i2/q4 at a 6/6 rate—isolating the failure to the model rather than the framework.

[Figure 6]

- Figure 5: Case 3 (MS-R, research integrity, q1 + q2): q1 is the only round in the benchmark with a 0% exact-match across all eight models, because the correct options are compound relational statements (e.g., IRB-precedes-extraction). Case 4 (DU-R, authority-influenced revision, q5 + q17): a single IRB pre-registration option is missed by the majority in both rounds, while all eight configurations correctly decline the overtly false “endorsed complaint” and “recommended retraction” options—failure here is one of omission, not capitulation to authority.

[Figure 7]

- Figure 6: Case 5 (P-R + exec_check, wrongful-termination report, q18 + q26): GPT-5.1 under both frameworks fails both rounds by omitting the YYYY-MM-DD_ filename prefix; Mimo V2.5 and Gemini-3.1 pass the lenient midterm check but fall off once q26 layers three literal ISO dates and ≥5 section headings. Case 6 (exec_check, NexaFlow GDPR bundle, q15 + q23

+ q26): all three questions achieve 0/8 pass rates; the shared failure mode is compound output-format compliance (timezone-aware timestamps, typed JSON fields with provenance citation, enum-constrained priority labels), defining a current structural-output ceiling.

[Figure 8]

- Figure 7: Case 7 (DU-R, 618 GPU fraud, q7 + q13 + q25): failure clusters are update-specific rather than cumulative—four models conflate the CNY 32,000 partial refund with full-price restitution on q13, while Ling-2.6 and DeepSeek-V4 show isolated answer-extraction failures when the response format drifts from their usual \bbox{} emission. Case 8 (exec_check, JSON schema adherence, hil_g4/q3 + hil_f7/q21 + hil_f7/q22): pass rates track schema documentation style—only 2/8 pass hil_g4/q3 whose key is described in natural language, versus 6/8 on hil_f7/q21 where the four required keys are listed verbatim.

[Figure 9]

- Figure 8: Case 9 (MS-R + DU-R, wrongful termination, q13 + q27): four models miss the option where Sun Wei admits the termination conversation lacked an email record—a claim that requires conjoining two elements of the same reply; in q27 seven of eight achieve perfect synthesis and only DeepSeek-V4 drops the two-part PIP-violation statement. Case 10 (exec_check + MS-R, pipeline authorship and final synthesis, hil_i2/q8 + hil_c7/q28): only GPT-5.1 (OC) passes the M2-contrast write-up (1/8); on the final synthesis GPT-5.1 (CC), Mimo V2.5, and GLM-5.1 miss the same forward-looking remediation option, while DeepSeek-V4 produces no extractable answer.

