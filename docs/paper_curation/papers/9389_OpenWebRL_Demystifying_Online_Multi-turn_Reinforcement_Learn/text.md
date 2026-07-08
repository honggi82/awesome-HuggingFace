# arXiv:2606.02031v2[cs.LG]4Jun2026

## OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents

Rui Yang1∗†, Qianhui Wu2∗‡, Yuxi Chen1, Hao Bai1, Wenlin Yao2, Hao Cheng2 Baolin Peng2, Huan Zhang1, Tong Zhang1, Jianfeng Gao2

[Figure 1]

[Figure 2]

1UIUC, 2Microsoft https://openwebrl.github.io

### Abstract

Building capable visual web agents requires long-horizon reasoning, precise grounding, and robust interaction with dynamic real-world websites. Despite rapid progress, the strongest systems remain largely proprietary, while open agents still depend heavily on supervised post-training over large collections of curated web trajectories. This dependence creates a major scalability bottleneck: high-quality demonstrations are expensive to collect, and static datasets offer limited coverage of the diverse, ever-changing open web. Although online reinforcement learning (RL) has shown promise for text-based agents, its potential for training visual web agents directly on live websites remains largely underexplored. In this paper, we introduce OpenWebRL, an open framework for training visual web agents with online multi-turn RL on real websites. OpenWebRL covers the full training pipeline, including scalable live-browser infrastructure, supervised initialization, multimodal context management, trajectory-level success judging, and efficient multi-turn policy optimization. Using this framework, we train OpenWebRL-4B, which establishes a new open-source state of the art on challenging live-web benchmarks. With only 0.4K initialization trajectories and 2.2K open-ended RL training tasks, OpenWebRL-4B achieves 67.0% success on Online-Mind2Web and 64.0% on DeepShop, outperforming prior open agents of similar or larger scale and remaining competitive with proprietary systems including OpenAI CUA and Gemini CUA. Beyond strong benchmark performance, we systematically study the key design choices that make online RL effective for visual web agents, and analyze how RL improves agentic reasoning. Overall, our work offers a practical path toward building more capable, reproducible, and cost-efficient open web agents. We will release our training data, models, and code to support future research.

| | | | |
|---|---|---|---|
| |OpenWebRL-4B-SFT<br><br>OpenWebRL-4B|Qwen3-VL-2|GPT-5 SoM<br><br>Gemini CUA<br><br>35B-A22B|
| |Fara-7B<br><br>MolmoWeb-4B|UI-TARS-1.5-7B<br><br>GLM-4.1V-9B<br><br>MolmoWeb-8B|OpenAI CUA|
| | | | |
| | | | |
| | | | |
| | | | |

- 0

10

20

30

40

50

60

70

80

Avg.SuccessRate(%)

Average Performance

GPT-5SoMOpenAICUAGeminiCUAUI-TARS-1.5-7BGLM-4.1V-9BMolmoWeb-4BFara-7BQwen3-VL-235B-A22BMolmoWeb-8BOpenWebRL-4B-SFTOpenWebRL-4B

0

10

20

30

40

50

60

70

SuccessRate(%)

57.7 58.3 57.3

31.3

33.9 34.1

31.3

35.3

63.7

47.0

67.0

Online-Mind2Web

GPT-5SoMOpenAICUAGeminiCUAUI-TARS-1.5-7BGLM-4.1V-9BMolmoWeb-4BFara-7BQwen3-VL-235B-A22BMolmoWeb-8BOpenWebRL-4B-SFTOpenWebRL-4B

0

10

20

30

40

50

60

70

49.1

24.7

62.0

11.6

32.0

26.2

35.6

42.3

56.7

48.7

64.0

DeepShop

GPT-5SoMOpenAICUAGeminiCUAUI-TARS-1.5-7BGLM-4.1V-9BMolmoWeb-4BFara-7BQwen3-VL-235B-A22BMolmoWeb-8BOpenWebRL-4B-SFTOpenWebRL-4B

0

20

40

60

80

100

90.6

70.9

88.6

66.4 66.8

73.5 75.2

78.2

66.4

60.2

74.1

WebVoyager

Figure 1 Performance comparison on online web benchmarks, including Online-Mind2Web (2025.04) [50], DeepShop (2025.06) [30], and WebVoyager (2024.01) [18]. As WebVoyager was constructed earlier than the other live benchmarks, temporal changes in websites may have a larger effect on direct comparability with previously reported results.

- 1 Introduction

4B 7B 8B 9B 235B

Model Size

Web agents powered by large vision-language models (VLMs) [1, 6, 39, 38, 41] have made rapid progress on realistic browser tasks, including product search, information extraction, and multi-step

∗ Equal contribution. ‡ Corresponding Author. † Part of the work was done during internship at Microsoft. Emails: ry21@illinois.edu, qianhuiwu@microsoft.com 1

navigation [12, 18, 19, 50, 63, 49, 14, 25]. However, the strongest open agents still lag behind proprietary systems in long-horizon planning and adaptation to dynamic websites. At the same time, proprietary models, closed training data and pipelines, and expensive API-based evaluation create substantial barriers for open research. As a result, there remains a widening gap between what closed systems can achieve and what the open research community can study, reproduce, and build upon.

A central bottleneck is the lack of scalable data and training frameworks. Leading open agents such as MolmoWeb [17] rely on supervised post-training over hundreds of thousands of curated trajectories, e.g., 278K trajectories, which are expensive to collect and inherently limited by static coverage [3, 17, 20]. Online reinforcement learning offers a principled alternative: instead of imitating fixed demonstrations, agents can improve by interacting with real websites and learning from task outcomes. Yet applying online RL to visual agents on the open web introduces challenges that are qualitatively different from those in controlled environments: web pages are dynamic and non-stationary, live browser interaction is slow and brittle, and open-ended tasks often lack reliable rule-based verification. Prior RL work has largely sidestepped these challenges by focusing on text-only agents in simulated or self-hosted environments [45, 61, 46]. The closest efforts on live websites, such as PAE [64] and WebGym [4], remain limited in their RL formulation and still rely on costly proprietary evaluation. More broadly, the design choices that make online RL effective for visual web agents remain underexplored.

We introduce OpenWebRL, a fully open framework for training visual web agents through direct interaction with live websites. Rather than treating online RL as a black-box recipe, OpenWebRL exposes and systematically studies the key factors behind effective open-web learning. The framework is built on a robust browser infrastructure for large-scale parallel rollout collection [33], and consists of three components: (1) a supervised warm start using only 0.4K trajectories, placing the policy in a productive exploration regime before online training; (2) an agent harness with multi-tool action execution, textual environment feedback, and multimodal context management to make online agentic RL more efficient and reliable; and (3) a multimodal multi-turn GRPO objective with trajectory-level judging, using either a GPT-based or a distilled 8B judge that matches proprietary performance while reducing evaluation cost by roughly $545.5 per experiment. Built with this framework, OpenWebRL-4B establishes a new state of the art among open visual web agents across three challenging live web benchmarks. With only a 4B backbone, 0.4K warm-start trajectories, and 2.2K open-web RL training tasks, OpenWebRL-4B achieves success rates of 74.1% on WebVoyager [18], 67.0% on Online-Mind2Web [50], and 64.0% on DeepShop [30]. These results substantially outperform prior open agents, including FARA-7B [3], MolmoWeb-8B [17], and even Qwen3-VL-235B-A22B-Thinking on the evaluated benchmarks, while remaining competitive with several proprietary systems such as GPT-5, OpenAI CUA, and Gemini CUA. Beyond benchmark performance, we systematically analyze the key design choices that make online RL effective for visual web agents and study how RL shapes agentic reasoning during training.

Our contributions are threefold. First, we introduce OpenWebRL, a fully open framework for end-to-end online RL of visual web agents on live websites. Second, we develop a practical multimodal multi-turn RL recipe that combines robust browser infrastructure, trajectory-level judging, and efficient context management, making open-web training effective for compact models. Third, we release a strong 4B-scale open agent together with a detailed empirical study of the ingredients that make online RL successful for visual web interaction. We hope OpenWebRL provides a practical foundation for future research on capable, reproducible, and cost-efficient open web agents.

### 2 Related Work

LLM/VLM-based Web Agents. Recent advances in LLM/VLM-based web agents have been driven by three complementary threads: stronger foundation models [6, 39, 10, 51, 21], agent frameworks [15, 52, 44], and web- or GUI-specific post-training [49, 35, 11, 48, 47, 41, 14, 7, 65]. More recent systems further enhance agent capabilities through curated demonstrations, synthetic data, and supervised fine-tuning tailored to realistic browser interaction [25, 3, 17, 20]. In parallel, evaluation has broadened from curated, static web benchmarks toward more diverse and realistic open-web settings [32, 18, 50, 30, 40, 22, 57, 9, 4].

Reinforcement Learning for Multimodal Agents. Deploying VLMs as agents in visually grounded environ-

- Table 1 Summary of existing multi-turn web agents training pipelines. Simulated web refers to environments such as WebShop; self-hosted web refers to more realistic offline web stacks such as WebArena. Green ✓ / red

✗ indicate whether weights, data, and code were publicly released.

Paper Method Observation Train Env Reward Weights Data Code

WebRL [34] Online curriculum RL Text Self-hosted web Trained judge ✓ ✓ ✓ AgentRL [61] SFT+GRPO Text Simulated web Rule-based ✗ ✓ ✓ PAE [64] Online Filtered BC Multimodal Open web Prompted judge ✓ ✓ ✓ RAGEN [45] StarPO Text Simulated web Rule-based ✗ ✓ ✓ WebAgent-R1 [46] SFT+Multi-turn GRPO Text Self-hosted web Rule-based ✗ ✓ ✓ UI-TARS-2 [41] Multi-turn PPO Multimodal Computer-use sandbox Mixed judges ✗ ✗ ✗ WebGym [4] Online Filtered BC Multimodal Open web Prompted judge ✗ ✓ ✓

WebSTAR [20] Step-filtered SFT Multimodal — Prompted judge ✗ ✓ ✓ GUI-Libra [55] Step-wise GRPO Multimodal — Rule-based ✓ ✓ ✓ ScaleCUA [25] SFT Multimodal — — ✓ ✓ ✓ Fara-7B [3] SFT Multimodal — — ✓ ✗ ✗ MolmoWeb [17] SFT Multimodal — — ✓ ✓ ✓

OpenWebRL (Ours) SFT + MM-GRPO Multimodal Open web Mixed judges ✓ ✓ ✓

ments requires moving beyond static understanding toward long-horizon decision-making[53, 43]. To bridge the gap, recent research adopts a two-stage SFT-then-RL paradigm[8, 59, 60]. The success of outcome-based RL for language reasoning [16] has spurred growing efforts to extend RLVR to VLMs and interactive agents. Early work applies this paradigm to static tasks such as perception, grounding, and visual question answering [26, 24, 37, 42, 28, 29, 55], while more recent methods target multi-turn and agentic settings [34, 64, 5, 4, 45, 61, 46, 41]. However, most of this work operates in simulated, self-hosted, or narrowly scoped environments. In contrast, we study fully open online RL for compact visual web agents trained and evaluated on live websites.

### 3 Preliminaries

Problem Formulation. We formulate multimodal web-agent training as a POMDP M = (S,O,A,T ,R).

Each task q specifies a start URL and instruction. At step t, the agent observes ot = (xt,It), where xt contains textual browser information (URL, tab info, environment feedback) and It is a screenshot. Given interaction history ht = (q,o0,a0,...,at−1,ot), policy πθ generates a response yt containing reasoning and a structured browser action, after which the environment transitions to st+1 ∼ T (st+1 | st,at) and yields ot+1. An episode τ = {(ht,yt,at,ot+1)}Tt=0−1 terminates when the agent calls done, the step budget is exhausted, or an environment failure occurs. Task success is evaluated only after the full interaction via rule-based checks or a judge model.

Multi-turn GRPO. GRPO [36] is a critic-free policy optimization method that replaces a learned value function with group-relative advantages. In multi-turn settings, the sampled unit is a full trajectory rather than a single response. For each task q, the policy samples a group of trajectories {τi}Gi=1, where each trajectory receives a trajectory-level reward R(τi). The group-relative advantage is Ai = R(τi)−mean({R(τj)}

G j=1)

std({R(τj)}Gj=1) . Multi-turn GRPO [46] propagates this trajectory-level signal to the action tokens generated across all turns in τi. Denoting by yi,t,k the k-th token in the response at turn t of trajectory τi, the optimization objective becomes

1 G

LMT-GRPO(θ) = −

G

i=1

Ti−1

1 Ti

t=0

|yi,t|

1 |yi,t|

min(ρi,t,k(θ)Ai, clip(ρi,t,k(θ),1 − ϵ,1 + ϵ)Ai),

k=1

where ρi,t,k(θ) is the token-level importance ratio at turn t.

### 4 OpenWebRL: An Online RL Training Framework for Visual Web Agents

We train multimodal web agents with an end-to-end online RL pipeline on the open web, starting from a general-purpose VLM. Our goal is not only to improve visual web-agent performance, but also to

systematically study the key components of online RL for open-web navigation. Since online RL is computationally expensive, we focus our main experiments on small VLMs, such as Qwen3-VL-4B [6]. However, general-purpose small VLMs often lack the web-specific knowledge needed to operate in dynamic browser environments. We therefore warm-start the policy with supervised finetuning on a small set of trajectories collected using a stronger open-source model. Starting from the SFT policy, we iteratively collect online rollouts in live browser environments and update the model with group-relative policy optimization. In the following subsections, we describe the data preparation, rollout system, reward design, and optimization objective.

#### 4.1 Agent Harness

Training and evaluating web agents on live websites introduces substantial environment noise, including dynamic page updates, pop-ups, redirects, bot detection, blocking, and transient network failures. To make open-web rollouts more reliable, we build a fault-tolerant browser environment based on Orchard Env [33] with navigation retries, timeout handling, and structured failure attribution, separating unstable website behavior from model behavior and making failures diagnosable during large-scale training. For efficiency, we run parallel browser instances that asynchronously sample independent trajectories, each maintaining its own page state and interaction history. Implementation details are in Appendix A.1.

Based on the environment infrastructure, we employ a generic multi-turn ReAct-style tool-calling agent [56] to avoid conflating scaffold-specific effects with differences in data or training recipes, and more importantly, let the same agent paradigm generalize beyond GUI navigation to any tool-using domain. Within this framework, the only browser-specific design choices are how observations are organized, what tools the agent can call, and how multi-step interaction history is maintained.

Observation and Environment Feedback. The environment provides a multimodal observation consisting of the current screenshot, active URL, viewport dimensions, and tab metadata. These signals allow the agent to track browser state across heterogeneous websites with diverse layouts, dynamic content, and multi-tab workflows. In addition, each action returns a concise textual environment-feedback message extracted from DOM-tree changes between consecutive interaction steps. These messages summarize the execution outcome and observable state changes, such as successful page navigation, new-tab creation, typed-text mismatches, or failed scrolling attempts. This lightweight feedback makes web interaction more observable, allowing the agent to distinguish successful actions from silent failures or unexpected browser behavior commonly encountered on live websites. More details are provided in Appendix A.2.

Action Space and Multi-tool-call Interface. The agent is equipped with a structured action space A of 13 atomic browser tools spanning pointer management (click, hover, drag), keyboard input (write, press keys), page navigation (scroll, goto url, go back, wait), tab management (new tab, switch tab, close tab), and termination via done(response), which is the sole mechanism for successfully ending an episode and emitting the final user-facing answer. At each step, the policy response is expected to consist of one reasoning block followed by one or more tool call blocks. The environment parses the tool calls and executes the corresponding browser actions sequentially before returning per-call feedback and the next screenshot. This multi-tool-call interface improves rollout efficiency: short deterministic interaction chains—e.g., focusing a search box, entering a query, and pressing Enter—can be completed within a single model step, eliminating unnecessary model-environment round trips that would otherwise dominate live-web rollouts. Please refer to Table 6 for more details.

Context Management. Long-horizon web interaction creates a fundamental context-management challenge for multimodal RL agents [23]. Each rollout step may contain a full-page screenshot, textual metadata, tool calls, environment feedback, and model reasoning traces. Retaining all screenshots quickly becomes impractical: a 30-step trajectory can exceed the context budget even for 64k-token models. However, preserving every historical screenshot is often unnecessary. Human users do not repeatedly inspect every previous browser state; instead, they rely primarily on recent visual observations together with memory of prior actions, outcomes, and task progress. We adopt the same principle by separating visual grounding from long-term memory. Recent screenshots are retained explicitly

for perception, while older interaction history is compressed into textual state information, including environment feedback and the agent’s own reasoning traces.

Specifically, at turn t, the policy receives the system instruction s, the task query q, the full sequence of previous model responses {yj}j<t, the full sequence of environment feedback strings, and the current browser observation. Each observation is represented as ot = (xt,It), where xt contains lightweight textual metadata such as the active tab, URL, and previous-step feedback, while It denotes the rendered screenshot. To bound multimodal context cost, we retain only the most recent K screenshots: It = (Imax(0,t−K+1),...,It). Empirically, retaining only the current screenshot (K = 1) already achieves strong performance while substantially reducing training cost (Section 5.1). Let yj denote the model response at turn j which includes both reasoning content and tool calls, the resulting policy context is: ht = (s,q,o0,y0,o1,y1,...,ot−1,yt−1,ot,It).

A key design choice is that environment feedback is always preserved, even after its corresponding screenshot has been discarded. These feedback strings provide the only explicit execution signal for browser actions, indicating whether interactions succeeded, failed, triggered navigation, opened new tabs, or produced unexpected behavior that may not be visually obvious from the screenshot alone. This becomes particularly important under our multi-tool-call interface, where a single model step may execute several browser actions sequentially and the subsequent turn must identify which sub-actions succeeded and which require recovery or retry.

Unlike prior GUI-agent approaches that compress history into short action summaries [49, 55] or executable action code [17], we retain the agent’s full historical reasoning traces as part of the context. We find that these reasoning traces naturally serve as compact textual memory, capturing prior visual observations, action intent, intermediate conclusions, and task progress without requiring all past screenshots to remain in context.

#### 4.2 Data Preparation and Supervised Fine-tuning

We construct the training corpus from WebGym [4] through a task-filtering, teacher-rollout, and trajectory-curation pipeline. Starting from 292K raw task instances, we remove tasks that overlap with evaluation benchmarks, subtasks decomposed from parent intents, tasks from long-tail or unstable websites, and near-duplicate intents. To identify near duplicates, we embed task intents with Qwen3-Embedding-8B and apply greedy similarity-based deduplication with a predefined threshold. We use a threshold of 0.99 to construct the SFT candidate pool, resulting in 15,601 filtered seed tasks. For the RL task pool, we apply the same filtering pipeline with a stricter threshold of 0.95 to further reduce semantic redundancy, retaining approximately 2.2K tasks spanning diverse real-world websites for RL.

To obtain demonstrations for supervised warm-starting, we rollout a strong open-source teacher model, Qwen3-VL-235B-A22B-Thinking [6], using the agent harness described in Section 4.1. For each filtered seed task, we sample four independent teacher trajectories and use GPT-4.1 to judge task success based on the final answer, interaction history, and screenshot trajectory. This yields a pool of successful demonstrations with multiple attempts per task, providing both high-quality trajectories and useful signals about task difficulty and rollout diversity. For SFT, we intentionally curate a small, high-quality subset instead of imitating all successful teacher trajectories. The goal is to provide the small model with sufficient web-interaction competence for productive exploration, while avoiding excessive imitation that may limit the effectiveness of subsequent online RL. By default, we select successful trajectories from the PAE-WebVoyager subset, which densely covers popular real-world websites. For each task group, we retain the shortest successful trajectory. When multiple trajectories have the same length, we use the shorter total response length as a tie-breaker. We also cap the number of tasks per website to improve domain diversity. This produces our default SFT set of 412 trajectories spanning 70 websites. The same curation rule can be applied to the InSTA-v3 subset to construct a larger SFT set, but in practice we find that the 412-trajectory set is already sufficient for supervised warm-starting small models.

We train small models, such as Qwen3-VL-4B-Thinking, with multi-turn behavioral cloning on the curated trajectories. Each assistant turn is treated as a training target conditioned on the serialized

interaction history, including screenshots, reasoning traces, tool calls, and environment feedback. Following standard multi-turn agent-training practice, we apply the loss only to the target assistant response and mask historical context and environment observations.

#### 4.3 Reward Design and Judge Model

We design the reward to capture both format correctness and task success, with details provided in Appendix A.4. The reward consists of two components. (1) Format reward. First, a format reward checks whether the response follows the required browser-agent protocol: rfmt(yt) ∈ {0,1}. A response receives format credit only if it contains the required thinking termination tag and a successfully parsed tool call. (2) Task-success reward. For completed trajectories, we use a VLM-as-a-judge reward to evaluate whether the task has been successfully completed. The judge takes as input the task instruction, the final response, recent screenshots, and the trajectory history, including tool calls and environment feedback. Examples of judge inputs and outputs are provided in Appendix F.2. The final reward is assigned at the trajectory level. A trajectory receives a positive reward only when it both follows the required response format and is judged successful. It receives a negative reward if it terminates due to repeated format errors, and receives zero reward otherwise. This design encourages the agent to produce valid executable actions while optimizing for actual task completion.

We use GPT-4.1 as the default judge during training. However, relying on a proprietary judge can be costly and may limit accessibility for the research community. In our experiments, a typical training run requires 43.2K judge API calls, costing approximately $545.5. To reduce this cost and make the pipeline more accessible, we distill an 8B judge model from 12.5K diverse online rollouts and GPT-4.1 judge labels. The distilled judge is trained to predict both the judging rationale and the final success evaluation. We evaluate the effectiveness of learned judge models in Section 5.4.

#### 4.4 Multimodal Multi-turn GRPO

Since rewards are assigned only at the trajectory level, we extend GRPO to optimize all assistant responses across a rollout, assigning the same group-relative advantage to every turn in the trajectory. For a trajectory τi with Ti assistant turns, we construct {(hi,t,yi,t)}T

i−1

t=0 , where hi,t is the managed multimodal context and yi,t is the assistant response at turn t. We use the same context-management procedure for rollout and optimization, and apply the loss only to assistant response tokens.

For each task q, we sample a group of G trajectories from the current policy: {τi}Gi=1, τi ∼ πθ

. Let Ri = R(τi) be the trajectory-level reward. Following group-relative policy optimization, we compute a normalized group-relative advantage: Ai = R

old

σG+ϵ , where µG = G1 Gj=1 Rj and σG is the reward standard deviation within the group. The resulting advantage Ai is assigned to all response tokens from all turns of trajectory τi. We optimize the clipped multi-turn GRPO objective:

i−µG

1 G

LMM-GRPO(θ) = −

G

i=1

Ti−1

k mi,t,k min (ρi,t,k(θ)Ai, clip (ρi,t,k(θ), 1 − ϵlow, 1 + ϵhigh) Ai) max( k mi,t,k, 1)

,

t=0

where ρi,t,k(θ) = ππθ(yi,t,k|hi,t,yi,t,<k)

θold(yi,t,k|hi,t,yi,t,<k) is the importance ratio for token k in turn t of trajectory i, and mi,t,k masks out non-assistant tokens. Note that we do not include the 1/Ti normalization factor at the trajectory level, as doing so would downweight longer trajectories and weaken the learning signal for harder tasks that require more interaction steps.

We use asymmetric clipping with ϵlow = 0.2 and ϵhigh = 0.28, and adopt trajectory-level dynamic sampling from [58], discarding task groups whose trajectories all receive identical rewards (e.g., all-zero or all-one). This filtering removes groups dominated by environment failures or trivial tasks, while creating a natural curriculum that focuses optimization on tasks that are solvable but not yet reliably solved by the current policy. We omit KL and entropy regularization to keep the training signal focused on trajectory-level reward.

### 5 Experiments

Training Settings. We use Qwen3-VL-4B-Thinking [6] as the main base model to keep online RL training computationally feasible. We first warm-start the model with SFT via LlamaFactory[62] for 3 epochs to obtain OpenWebRL-4B-SFT. Starting from this SFT checkpoint, we further apply our MM-GRPO algorithm in online web environments, using a curated 2.2K training set described in Section 4.2. The main RL training runs for 90 iterations, where each iteration consists of online rollouts followed by MM-GRPO updates, requiring about 300 B200 GPU hours in total. Across training, we collect approximately 54K online trajectories. Since 30-step rollouts are substantially slower, we train OpenWebRL-4B in two stages: 90 iterations with a maximum of 15 rollout steps, followed by 50 iterations with a maximum of 30 rollout steps. We also report the results of a 4B variant trained with our distilled 8B judge model and an 8B variant (OpenWebRL-8B) trained with the same pipeline. More training details are provided in Appendix A.5.

Benchmarks. We evaluate on three challenging live web benchmarks: WebVoyager [18], covering open-domain navigation across popular websites (using the FARA-curated 595-task version [3]); OnlineMind2Web [50], comprising 300 long-horizon tasks across 136 websites; and DeepShop [30], targeting realistic shopping tasks with multi-constraint product selection. We follow each benchmark’s standard evaluation protocol. Full details are in Appendix B.

Baselines. We compare our models with both proprietary and open-source baselines. The proprietary baselines include GPT-5, Gemini-3-Flash, o3, OpenAI CUA, and Gemini CUA. The open-source baselines include Holo1-7B [2], UI-TARS-1.5-7B [35], GLM-4.1V-9B-Thinking [21], FARA-7B [3], MolmoWeb-4B/8B [17], Qwen3-VL-4B-Thinking, and Qwen3-VL-235B-A22B-Thinking [6].

Evaluation Metrics. Our primary metric is the official success rate, following the standard evaluation protocol used in prior work [3, 17]. This protocol relies on Browser-Use Stealth Browsers1, a managed cloud-browser service that provides CAPTCHA solving and stable browser sessions, reducing failures caused by website blocking and session instability during live-web evaluation. However, Browser-Use Service introduces a paid third-party dependency, which increases evaluation cost and makes exact reproduction less accessible for the academic community. To improve transparency, we also report

“success rate w/o aborted tasks”—the success rate excluding tasks with non-agent failures such as blocked pages or disconnected browser sessions, providing an estimate of agent performance without using Browser-Use Service.

#### 5.1 Main Results

OpenWebRL establishes a new open-source state of the art and is competitive with proprietary web agents.

- Table 2 reports official success rates on three live-web benchmarks. With only a 4B backbone, a 30-step evaluation budget, and a much smaller training set than prior open-source systems, OpenWebRL-4B achieves an average success rate of 68.4%, substantially outperforming existing open-source agents. The gains of OpenWebRL-4B are particularly pronounced on long-horizon benchmarks, i.e., OnlineMind2Web and DeepShop, as indicated by Appendix C.1. OpenWebRL-4B outperforms FARA-7B by

+32.9 and +37.8 points, and surpasses MolmoWeb-8B by +31.7 and +21.7 points on the two benchmarks, respectively. Scaling the backbone to 8B further improves the average success rate to 68.7% under the same 30-step evaluation budget. Increasing the evaluation budget to 50 steps yields a slightly higher average success rate of 69.2% for OpenWebRL-8B, suggesting that the model can generalize to longer interaction horizons despite being trained with 30-step SFT and RL budgets. However, the 50-step setting also incurs substantially higher computation and wall-clock cost, so we use 30 steps as the default evaluation setting. Notably, both OpenWebRL-4B and OpenWebRL-8B are highly competitive with proprietary systems. On Online-Mind2Web and DeepShop, they surpass several closed-source agents, including GPT-5, Gemini-3-Flash, GPT-4o, o3, and the OpenAI/Gemini computer-use agents. These results show that effective online RL can substantially improve open-source VLMs for live-web interaction, enabling compact 4B–8B models to compete with much larger proprietary systems on challenging live-web tasks.

1https://browser-use.com/stealth-browsers

Table 2 Official success rates (%) across three open-web benchmarks. * marks numbers reported in FARA [3]; † marks numbers reported in MolmoWeb [17].

Model Name # Steps # Tasks WebVoyager Online-Mind2Web DeepShop Average Proprietary Models

GPT-5 (Axtree)† 30 – 70.6 41.9 40.7 51.1 Gemini-3-flash (Axtree)† 30 – 74.4 34.8 45.1 51.4 Gemini-3-flash (Axtree)† 100 – 85.6 44.8 55.3 61.9

- GPT-4o (SoM)* 100 – 65.1 34.6 16.0 38.6 o3 (SoM)* 100 – 79.3 55.4 49.7 61.5

- GPT-5 (SoM)* 100 – 90.6 57.7 49.1 65.8 OpenAI computer-use-preview* 100 – 70.9 58.3 24.7 51.3 Gemini computer-use-preview† 100 – 88.6 57.3 62.0 69.3 Open-Source Models

Holo1-7B† 30 >15.6k 55.4 – – – UI-TARS-1.5-7B* 100 – 66.4 31.3 11.6 36.4 GLM-4.1V-9B-Thinking* 100 – 66.8 33.9 32.0 44.2 Fara-7B* 100 >123.2k 73.5 34.1 26.2 44.6 MolmoWeb-4B† 100 >278.5k 75.2 31.3 35.6 47.4 MolmoWeb-8B† 100 >278.5k 78.2 35.3 42.3 51.9 Qwen3-VL-4B-Thinking 30 – 52.6 32.0 33.3 39.3 Qwen3-VL-8B-Thinking 30 – 61.3 38.7 44.0 48.0 Qwen3-VL-235B-A22B-Thinking 30 – 66.4 63.7 56.7 62.3

Ours: 4B backbone

OpenWebRL-4B-SFT 30 0.4k 60.2 47.0 48.7 52.0 OpenWebRL-4B 30 2.2k 74.1 67.0 64.0 68.4 OpenWebRL-4B w/ OpenWebRL-Judge-8B 30 2.2k 68.9 67.3 68.7 68.3

Ours: 8B backbone

OpenWebRL-8B-SFT 30 0.9k 66.2 54.0 50.0 56.7 OpenWebRL-8B 30 2.2k 73.8 67.0 65.3 68.7 OpenWebRL-8B 50 2.2k 74.6 69.7 63.3 69.2

Both SFT and MM-GRPO contribute to the final gains. Table 2 shows that supervised fine-tuning provides a strong warm start for both model scales, while MM-GRPO contributes even larger improvements in the subsequent online RL stage. For the 4B backbone, SFT improves the average success rate from 39.3% to 52.0%, while MM-GRPO further increases it to 68.4%, yielding gains of +16.4 points over SFT and +29.1 points over the base model. We observe a similar trend for the 8B backbone: SFT improves the average success rate from 48.0% to 56.7%, and MM-GRPO further raises it to 68.7%, giving gains of +12.0 points over SFT and +20.7 points over the original backbone. The improvements are consistent across all three benchmarks, suggesting that SFT provides the interaction competence needed for exploration, while online RL is the key driver of the final performance gains.

#### 5.2 Learning Dynamics of MM-GRPO

SFT initialization leads to faster and more effective RL. Figure 2 compares MM-GRPO training initialized from the base model (Qwen3-VL-4B-Thinking) and from the SFT checkpoint on Online-Mind2Web benchmark. Although both runs eventually reach similar training rewards and follow a similar evaluation trend, the SFT-initialized run maintains a clear ∼10% advantage in evaluation success rate throughout training. This gap suggests that supervised warm-starting does more than simply accelerate optimization: it places the policy in a better region of the behavior space, enabling more effective online exploration. Overall, MM-GRPO produces stable improvements from both initializations, and the SFT-initialized policy continues to improve even after 80 iterations.

The benefit of SFT initialization is especially pronounced on harder tasks. Figure 2(d) further breaks down performance by task difficulty. MM-GRPO from the SFT checkpoint improves success across all

Training Reward

Avg. Response Length

Eval Success Rate w/o Aborted Tasks

Δ Success Rate by Difficulty

+22.3

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | |M M<br><br>|M-GRPO M-GRPO|from Base from SFT| |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | |M M<br><br>|M-GRPO M-GRPO|from Base from SFT| |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | |M M<br><br>|M-GRPO M-GRPO|from Base from SFT| |

55

MM-GRPO from Base

50

| |
|---|

Avg.ResponseLength

TrainingReward(%)

ΔSuccessRate(pp)

20

MM-GRPO from SFT

SuccessRate(%)

400

50

+15.8

40

15

+14.0

45

350

+12.0

10

30

40

+6.3

300

35

5

20

+2.3

30

250

0

0 20 40 60 80

0 20 40 60 80

0 20 40 60 80

Easy Medium Hard

Iteration

Iteration

Iteration

Difficulty

- Figure 2 Comparison of MM-GRPO training from SFT and base-model initializations. (a) Average training reward and (b) average response length measured on training rollouts. (c) Evaluation success rate (excluding aborted tasks) and (d) success rate improvement across difficulty splits measured on Online-Mind2Web.

Mean P90

0

4K

8K

12K

16K

20K

Tokenspertrajectory

10.9K

18.1K

8.8K

17.5K

7.9K

15.0K

Avg steps: 14.0 (Iter 0) 12.7 (Iter 5) 8.9 (Iter 80)

History summarization

Blocker diagnosis

Retry-plan loop

Condition proof

0

5

10

15

20

25

30

Presencerate(%)

14.5 14.2

15.3

7.1

18.4

16.4

17.8

7.1

21.4

23.7 23.4

12.1

Nonproxy

History summarization

Blocker diagnosis

Retry-plan loop

Condition proof

0

100

200

300

400

500

600

Avg.tokensperstep

282

332

273

317

294 277

323

288

326

310

325

542

440

535

411

(a) Trajectory length (b) Proxy presence rate (c) Avg. step-level response length

| |
|---|

Iter 0

| |
|---|

Iter 5

| |
|---|

Iter 80

- Figure 3 Diagnosing response-length growth during MM-GRPO. (a) Trajectory length, measured by the mean and P90 (90th percentile), together with the average number of interaction steps. (b) Step-level proxy presence rates. (c) Average step-level response length conditioned on proxy type, including no-proxy steps.

difficulty splits, with the largest gain on hard tasks (+22.3 points). In contrast, RL from the base model improves only marginally on hard tasks (+2.3 points), with most of its gains concentrated on the easy and medium splits. Together, these results validate the effectiveness and necessity of warm-starting online RL from a SFT model rather than training directly from the base model.

MM-GRPO drives targeted rather than uniform reasoning expansion. As shown in Figure 2 (b), the model produces longer responses during MM-GRPO training. We further analyze how response length changes in Figure 3. Interestingly, the overall trajectory length does not increase. Instead, the average number of interaction steps decreases from 14.0 at iteration 0 to 8.9 at iteration 80, while the average and 90th-percentile trajectory lengths drop from 10.9K and 18.1K tokens to 7.9K and 15.0K tokens, respectively. To better understand the source of the increased response length, we apply lexical proxy detection to model outputs on Online-Mind2Web to identify several step-level reasoning patterns (see Appendix D for details). As shown in Figure 3(b) and (c), MM-GRPO increases both the frequency and the conditional length of several recurring reasoning patterns, including history summarization, blocker diagnosis, retry-plan reasoning, and condition-proof reasoning. For instance, the step-level presence rate rises from 14.5% to 21.4% for history summarization, from 14.2% to 23.7% for blocker diagnosis. The average response length for proxy-bearing steps also grows substantially, from 332 to 542 tokens for history summarization, from 273 to 440 tokens for blocker diagnosis. By comparison, non-proxy steps remain much more stable, increasing only from 282 to 325 tokens on average. These results suggest that response-length growth during RL is not a uniform expansion across all responses; instead, the model allocates additional verbosity and reasoning patterns selectively to important steps.

#### 5.3 Test-time Scaling

We further study whether visual web agents can benefit from additional test-time computation. Specifically, we evaluate pass@k, where each task is attempted with k independent rollouts and is counted as successful if at least one rollout succeeds. We use k ∈ {1,2,3,4} and keep the inference

###### pass@k · WebVoyager

###### pass@k · OnlineMind2Web

###### pass@k · DeepShop

100

90

90

90

80

80

80

70

pass@k(%)

pass@k(%)

pass@k(%)

70

60

70

60

50

50

60

Base Model (Qwen3-VL-4B-Thinking)

Base Model (Qwen3-VL-4B-Thinking)

Base Model (Qwen3-VL-4B-Thinking)

40

40

OpenWebRL-4B-SFT

OpenWebRL-4B-SFT

OpenWebRL-4B-SFT

OpenWebRL-4B

OpenWebRL-4B

OpenWebRL-4B

30

50

30

1 2 3 4

1 2 3 4

1 2 3 4

k

k

k

Figure 4 Pass@k performance of OpenWebRL-4B and baselines on three online benchmarks.

Figure 5 Training and evaluation curve for RL with different judges.

Table 3 Judge model evaluation results. Oracle is labeled by GPT-4.1. All numbers are percentages.

Training Reward

Eval Success Rate w/o Aborted Tasks

Model Accuracy Precision Recall F1

| |GPT-4.<br><br>Qwen3|1 as judge -VL-8B as|judge| | |
|---|---|---|---|---|---|
| |OpenW|ebRL-Judge|-8B| | |
| | | | | | |
| | | | | | |
| | | | | | |

| |GPT-4|.1 as judge| | | |
|---|---|---|---|---|---|
| |Qwen3 OpenW<br><br>|-VL-8B as ju ebRL-Judge|dge<br><br>-8B| | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

55

o4-mini 76.0 95.2 61.7 74.9 gpt-4o 85.6 83.6 93.4 88.3 WebJudge-7B [50] 71.6 67.9 96.9 79.8 Qwen3-VL-4B-Instruct 76.8 75.7 89.0 81.8 Qwen3-VL-8B-Instruct 80.6 78.6 91.7 84.7 Qwen3-VL-32B-Instruct 85.8 87.4 88.3 87.8

TrainingReward(%)

60

SuccessRate(%)

50

45

50

40

40

35

30

OpenWebRL-Judge-4B 85.2 86.1 89.7 87.8 OpenWebRL-Judge-8B 89.8 89.5 94.8 92.1

30

0 20 40 60 80

0 20 40 60 80

Iteration

Iteration

setup fixed across models, with a maximum rollout length of 30 steps. Figure 4 compares the base model (Qwen3-VL-4B-Thinking), OpenWebRL-4B-SFT, and OpenWebRL-4B on WebVoyager, OnlineMind2Web, and DeepShop. The results show that live-web agent performance can be scaled by sampling multiple independent attempts. However, OpenWebRL-4B exhibits a consistently stronger pass@k curve than both the base and SFT models across all three benchmarks. This indicates that online RL improves not only single-rollout success, but also the probability that repeated attempts discover a successful trajectory. Notably, OpenWebRL-4B achieves over 90% pass@4 official success rate across the three live-web benchmarks, substantially outperforming MolmoWeb-8B [17], which reports around 60% pass@4 success rate on Online-Mind2Web. The strong pass@k scaling suggests that OpenWebRL-4B learns not only more accurate individual actions, but also a richer distribution of viable interaction strategies that can be exploited through multiple test-time attempts.

#### 5.4 Evaluation of OpenWebRL-Judge

The distilled OpenWebRL-Judge-8B provides effective training signals for stable online RL. As shown in Table 2, OpenWebRL-4B w/ OpenWebRL-Judge-8B achieves strong performance among open-source agents on both Online-Mind2Web (67.3) and DeepShop (68.7). Its overall average score (68.3) is nearly identical to that of the GPT-4.1-Judge variant (68.4), despite relying entirely on a distilled open-source reward model during RL training, substantially reducing both the cost and dependency associated with proprietary API calls. We further compare RL training dynamics in Figure 5. Both the training reward and the evaluation success rate w/o aborted tasks on Online-Mind2Web show stable optimization trends that closely match those obtained with GPT-4.1 supervision. In contrast, using Qwen3-VL-8B as the judge model leads to clear reward hacking behavior [13, 54, 31], producing higher training rewards but substantially lower evaluation success rates. These results further demonstrate that OpenWebRL-Judge-8B provides reliable and effective reward signals for stable online RL training.

OpenWebRL-Judge-8B closely matches GPT-4.1 on held-out trajectory evaluation. To evaluate the judge model independently from RL training, we construct a held-out set of 500 trajectory rollouts collected from different training stages and annotate them with GPT-4.1. We then compare several judge models by measuring their agreement with these GPT-4.1 annotations. As shown in Table 3, OpenWebRLJudge-8B achieves the strongest alignment with GPT-4.1, reaching 89.8% accuracy and a 92.1% F1

Table 4 Ablation study on rollout length and context management strategies for online RL.

System WebVoyager Online-Mind2Web DeepShop OpenWebRL-4B 74.1 67.0 64.0 Rollout length ablations

RL w/ 30-Step Rollout Only 66.7 (↓ -7.4) 65.4 (↓ -1.6) 63.3 (↓ -0.7) RL w/ 15-Step Rollout Only 70.1 (↓ -4.0) 65.0 (↓ -2.0) 63.3 (↓ -0.7) RL w/ 10-Step Rollout Only 70.6 (↓ -3.5) 60.7 (↓ -6.3) 57.3 (↓ -6.7)

Context management ablations with 15-step rollouts

w/ Recent Two Screenshots 68.2 (↓ -1.9) 65.3 (↑ +0.3) 59.3 (↓ -4.0) w/o Textual Environment Feedback 64.9 (↓ -5.2) 57.0 (↓ -8.0) 56.7 (↓ -6.6) w/o Historical Reasoning 55.5 (↓ -14.6) 41.3 (↓ -23.7) 54.7 (↓ -8.6)

score. It outperforms strong baselines, including WebJudge-7B [50], Qwen3-VL-32B, and GPT-4o. While some baselines achieve strong precision or recall individually, their overall accuracy and F1 scores remain lower, suggesting that our distilled judge captures web-task success signals more reliably than both general-purpose VLM judges and prior web-specific judge models in this setting.

#### 5.5 Ablations on Rollout and Context Management

Here we ablate key design choices in OpenWebRL-4B: (i) the rollout-length curriculum used in OpenWebRL-4B, where we first run online RL with a 15-step horizon and then continue with a 30-step horizon; and (ii) the three context management strategies, including textual environment feedback, historical reasoning, and the number of recent screenshots retained in context. To ensure a controlled comparison, all variants are initialized from the same SFT checkpoint and use the identical RL training recipe, with only the ablated factor changed. For cost efficiency, the context-management ablations are trained using only 15-step rollouts and are compared against the “RL w/ 15-Step Rollout Only” baseline. Table 4 reports the results.

Rollout-length curriculum. OpenWebRL-4B outperforms all fixed-budget RL variants. Training with only 30-step rollouts leads to performance drops of 7.4, 1.6, and 0.7 points on WebVoyager, Online-Mind2Web, and DeepShop, respectively. Using only 15-step rollouts performs better than starting directly with 30-step rollouts, but still underperforms the full curriculum by 4.0, 2.0, and 0.7 points. Moreover, using an overly short horizon, such as 10-step rollouts, substantially hurts performance on Online-Mind2Web and DeepShop (−6.3 and −6.7 points), both of which require longer interaction horizons to complete tasks successfully. These results suggest that a medium-horizon training stage helps stabilize early exploration, while longer-horizon rollouts improve the policy’s ability to handle tasks requiring extended interactions. Overall, combining 15-step and 30-step training stages is more effective than using a fixed rollout budget throughout training.

Textual environment feedback provides lightweight but informative signals. Removing this feedback reduces RL performance by 5.2, 8.0, and 6.6 points relative to the 15-step rollout baseline. This implies that the feedback provides explicit information about action execution that is not always inferable from the screenshot alone. Without it, the policy is less reliable in deciding whether to retry, recover from errors, or adjust strategy, particularly in longer-horizon and more failure-prone settings such as Online-Mind2Web.

Historical reasoning provides critical contextual signals. Removing historical reasoning yields the largest degradation among all ablations, reducing performance by 14.6, 23.7, and 8.6 points relative to the 15-step rollout baseline. We conjecture that these reasoning traces can serve as a compact memory of the high-level plan, interpretations of past observations, attempted actions, and their outcomes. Without this information, the model must effectively replan from scratch at each step and loses access to accumulated context, which is particularly detrimental in long-horizon web navigation.

Keeping more recent screenshots does not consistently help. By default, the model retains only the most recent screenshot in context. Increasing this to the two most recent screenshots does not yield consistent

Training Reward

Eval Success Rate w/o Aborted Tasks

50

55

TrainingReward(%)

SuccessRate(%)

50

40

45

30

40

MM-GRPO from Base

MM-GRPO from Base

MM-GRPO from 0.4K SFT ep1

MM-GRPO from 0.4K SFT ep1

35

20

- MM-GRPO from 0.4K SFT ep3

- MM-GRPO from 1.9K SFT ep3

- MM-GRPO from 0.4K SFT ep3

- MM-GRPO from 1.9K SFT ep3

30

0 20 40 60 80

0 20 40 60 80

Iteration

Iteration

###### Figure 6 Comparison of MM-GRPO training under different supervised warm-start initializations.

    % %   

((   )          )     ( (

 (  % %    - ( 

          ))      -      (      ) )

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

 (  % %   - (     

     ))         

((   )

    % %   

   (    %

   (    %

 )   )  %

 )   )  %

(a) (b) (c) (d)

Figure 7 Ablation study on (a)(b) dynamic sampling (DS) and (c)(d) PPO epoch for MM-GRPO.

gains, changing performance by −1.9, +0.3, and −4.0 points. This suggests that historical reasoning already captures most useful information from earlier visual observations, while additional screenshots mainly increase context length and add visual token overhead. Beyond the latest frame, visual history provides limited benefit and may even degrade performance. It also significantly increases training cost, extending runtime from approximately 240 GPU hours to 400 GPU hours.

#### 5.6 Ablations on MM-GRPO Designs

Online RL benefits most from a balanced SFT warm start. We first study how supervised warm-starting affects subsequent online RL. Figure 6 compares MM-GRPO training from four initial 4B policies: the base model without SFT, a lightweight SFT model trained on 0.4K trajectories for 1 epoch, our default SFT model trained on the same 0.4K trajectories for 3 epochs, and a larger-data SFT model trained on 1.9K trajectories for 3 epochs. The 1.9K trajectories are collected with the same teacher rollout pipeline, but retain a broader and less aggressively curated task distribution from WebGym. The results show that supervised initialization is crucial, i.e., all SFT initializations outperform the base model initialization, but stronger warm start does not necessarily lead to better online RL. The 1.9K/3-epoch initialization underperforms the default setting after online RL. Although it starts from a relatively strong policy, its improvement quickly saturates, and its final evaluation performance remains below that of the 0.4K/3-epoch initialization. We hypothesize that heavier imitation training, especially on a less curated task distribution, may reduce policy plasticity or bias the policy toward behaviors that are less amenable to online refinement. In contrast, the default initialization gives the policy enough competence to explore effectively while still allowing RL to further adapt it through online trial and error.

Trajectory dynamic sampling improves training stability and optimization efficiency. As shown in Figure 7(a) and (b), removing trajectory-level dynamic sampling leads to noticeably less stable training dynamics in both training reward and evaluation success rate. With dynamic sampling, training collects a fixed number of effective groups by filtering out zero-reward-variance groups, whereas the variant without dynamic sampling always collects a fixed number of groups regardless of reward variance. As a result, the effective batch size can vary significantly across iterations without dynamic sampling, leading to

noisier policy updates and less stable optimization. In contrast, dynamic sampling stabilizes training and enables faster performance improvement in MM-GRPO.

Moderate PPO epochs provide a balance between data efficiency and stable optimization. PPO epochs control how many times the policy is updated on the same rollout batch before collecting new trajectories, creating a trade-off between data efficiency and off-policyness. As shown in Figure 7(c) and (d), using only one PPO epoch leads to slower optimization and worse evaluation performance, suggesting insufficient updates for each rollout batch. Increasing the PPO epochs to 2 substantially improves both training reward and evaluation success rate, achieving the best overall performance. However, increasing the PPO epochs to 3 results in less stable evaluation behavior: although training rewards remain high, the evaluation success rate peaks early and later declines. This suggests that excessively reusing the same rollout data increases off-policyness and can lead to over-optimization, while a moderate number of PPO epochs better balances sample efficiency and stable online RL training.

- 6 Additional Analysis We provide additional diagnostics in the appendix to provide further insight into the main results.

- • Appendix C.1 analyzes benchmark difficulty and shows that Online-Mind2Web and DeepShop require longer trajectories and have lower success rates than WebVoyager, making the gains of OpenWebRL-4B on these benchmarks especially informative.
- • Appendix E further analyzes 100 failed trajectories from OpenWebRL-4B. Access and environment issues account for the largest share of failures (51%), followed by reasoning and constraint-tracking limitations (27%), visual grounding and interaction errors (13%), and task definition or judge issues (9%). These results suggest that future progress will require not only stronger policies, but also more robust open-web agent infrastructure.

- 7 Conclusion

We introduced OpenWebRL, an open framework for training visual web agents with online multi-turn RL on live websites. Built on this framework, OpenWebRL-4B achieves strong performance on challenging live-web benchmarks, demonstrating that effective web-agent training can emerge from relatively modest supervised initialization and scalable online interaction, rather than relying on large-scale demonstration datasets. Through systematic analysis, we identify several key ingredients that make online RL effective for visual web agents, including supervised warm start, textual environment feedback, historical reasoning for long-horizon memory, rollout-length curricula, and reliable trajectory-level success judging. We hope OpenWebRL serves as an actionable and reproducible foundation for future research on open, cost-efficient, and scalable web-agent training, and helps narrow the gap between open and proprietary systems for long-horizon interaction on the live web.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Mathieu Andreux, Breno Baldas Skuk, Hamza Benchekroun, Emilien Bire´, Antoine Bonnet, Riaz Bordie, Nathan Bout, Matthias Brunel, Pierre-Louis Cedoz, Antoine Chassang, et al. Surfer-h meets holo1: Cost-efficient web agent powered by open weights. arXiv preprint arXiv:2506.02865, 2025.
- [3] Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, and Andrew Zhao. Fara-7B: An efficient agentic model for computer use. arXiv preprint arXiv:2511.19663,

- 2025.

- [4] Hao Bai, Alexey Taymanov, Tong Zhang, Aviral Kumar, and Spencer Whitehead. WebGym: Scaling training environments for visual web agents with realistic tasks. arXiv preprint arXiv:2601.02439,

- 2026.

- [5] Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, and Aviral Kumar. DigiRL: Training in-the-wild device-control agents with autonomous reinforcement learning. In Advances in Neural Information Processing Systems, volume 37, 2024.
- [6] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [7] Hyungjoo Chae, Namyoung Kim, Kai Ong, Minju Gwak, Gwanwoo Song, Jihoon Kim, Sunghwan Kim, Dongha Lee, and Jinyoung Yeo. Web agents with world models: Learning and leveraging environment dynamics in web navigation. In International Conference on Learning Representations, volume 2025, pages 63707–63738, 2025.
- [8] Hanyang Chen, Mark Zhao, Rui Yang, Qinwei Ma, Ke Yang, Jiarui Yao, Kangrui Wang, Hao Bai, Zhenhailong Wang, Rui Pan, et al. Era: Transforming vlms into embodied agents via embodied prior learning and online reinforcement learning. arXiv preprint arXiv:2510.12693, 2025.
- [9] Yuxi Chen, Haoyu Zhai, Chenkai Wang, Rui Yang, Lingming Zhang, Gang Wang, and Huan Zhang. Captcha solving for native gui agents: Automated reasoning-action data generation and self-corrective training. arXiv preprint arXiv:2603.23559, 2026.
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024.
- [11] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, 2024.
- [12] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.
- [13] Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR, 2023.
- [14] Boyu Gou, Demi Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. In International Conference on Learning Representations, volume 2025, pages 30851–30883, 2025.
- [15] Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, et al. Is your llm secretly a world model of the internet? model-based planning for web agents. arXiv preprint arXiv:2411.06559, 2024.
- [16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.
- [17] Tanmay Gupta, Piper Wolters, Zixian Ma, Peter Sushko, Rock Yuren Pang, Diego Llanes, Yue Yang, Taira Anderson, Boyuan Zheng, Zhongzheng Ren, et al. Molmoweb: Open visual web agent and open data for the open web. arXiv preprint arXiv:2604.08516, 2026.
- [18] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. In

- Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6864–6890, 2024.
- [19] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Hongming Zhang, Tianqing Fang, Zhenzhong Lan, and Dong Yu. Openwebvoyager: Building multimodal web agents via iterative real-world exploration, feedback and optimization. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27545–27564, 2025.
- [20] Yifei He, Pranit Chawla, Yaser Souri, Subhojit Som, and Xia Song. Scalable data synthesis for computer use agents with step-level filtering. arXiv preprint arXiv:2512.10962, 2025.
- [21] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv e-prints, pages arXiv–2507, 2025.
- [22] Yining Hong, Rui Sun, Bingxuan Li, Xingcheng Yao, Maxine Wu, Alexander Chien, Da Yin, Ying Nian Wu, Zhecan Wang, and Kai-Wei Chang. Embodied web agents: Bridging physical-digital realms for integrated agent intelligence. Advances in Neural Information Processing Systems, 38, 2026.
- [23] Wei-Chieh Huang, Weizhi Zhang, Yueqing Liang, Yuanchen Bei, Yankai Chen, Tao Feng, Xinyu Pan, Zhen Tan, Yu Wang, Tianxin Wei, et al. Rethinking memory mechanisms of foundation agents in the second half: A survey. arXiv preprint arXiv:2602.06052, 2026.
- [24] Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [25] Zhaoyang Liu, JingJing Xie, Zichen Ding, Zehao Li, Bowen Yang, Zhenyu Wu, Xuehui Wang, Qiushi Sun, Shi Liu, Weiyun Wang, et al. Scalecua: Scaling open-source computer use agents with cross-platform data. arXiv preprint arXiv:2509.15221, 2025.
- [26] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2034–2044, 2025.
- [27] Xing Han Lu`, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stan´czak, Peter Shaw, Christopher J Pal, and Siva Reddy. Agentrewardbench: Evaluating automatic evaluations of web agent trajectories. arXiv preprint arXiv:2504.08942, 2025.
- [28] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Pengxiang Zhao, Guangyi Liu, et al. Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 17608–17616, 2026.
- [29] Run Luo, Lu Wang, Wanwei He, Longze Chen, Jiaming Li, and Xiaobo Xia. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.
- [30] Yougang Lyu, Xiaoyu Zhang, Lingyong Yan, Maarten de Rijke, Zhaochun Ren, and Xiuying Chen. Deepshop: A benchmark for deep research shopping agents. arXiv preprint arXiv:2506.02839, 2025.
- [31] Yuchun Miao, Sen Zhang, Liang Ding, Rong Bao, Lefei Zhang, and Dacheng Tao. Inform: Mitigating reward hacking in rlhf via information-theoretic reward modeling. Advances in Neural Information Processing Systems, 37:134387–134429, 2024.
- [32] Yichen Pan, Dehan Kong, Sida Zhou, Cheng Cui, Yifei Leng, Bing Jiang, Hangyu Liu, Yanyi Shang, Shuyan Zhou, Tongshuang Wu, et al. Webcanvas: Benchmarking web agents in online environments. arXiv preprint arXiv:2406.12373, 2024.

- [33] Baolin Peng, Wenlin Yao, Qianhui Wu, Hao Cheng, Xiao Yu, Rui Yang, Tao Ge, Alessandrio Sordoni, Xingdi Yuan, Yelong Shen, et al. Orchard: An open-source agentic modeling framework. arXiv preprint arXiv:2605.15040, 2026.
- [34] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. In International Conference on Learning Representations, volume 2025, pages 79791–79821, 2025.
- [35] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.
- [36] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [37] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.
- [38] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- [39] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.
- [40] Brandon Trabucco, Gunnar Sigurdsson, Robinson Piramuthu, and Ruslan Salakhutdinov. Insta: Towards internet-scale training for agents. arXiv preprint arXiv:2502.06776, 2025.
- [41] Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, et al. Ui-tars-2 technical report: Advancing gui agent with multi-turn reinforcement learning. arXiv preprint arXiv:2509.02544, 2025.
- [42] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. Advances in Neural Information Processing Systems, 38:30865–30891, 2026.
- [43] Kangrui Wang, Pingyue Zhang, Zihan Wang, Yaning Gao, Linjie Li, Qineng Wang, Hanyang Chen, Yiping Lu, Zhengyuan Yang, Lijuan Wang, et al. Vagen: Reinforcing world model reasoning for multi-turn vlm agents. Advances in Neural Information Processing Systems, 38:172871–172933, 2026.
- [44] Zhaoyang Wang, Qianhui Wu, Xuchao Zhang, Chaoyun Zhang, Wenlin Yao, Fazle Elahi Faisal, Baolin Peng, Si Qin, Suman Nath, Qingwei Lin, et al. Webxskill: Skill learning for autonomous web agents. arXiv preprint arXiv:2604.13318, 2026.
- [45] Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025.
- [46] Zhepei Wei, Wenlin Yao, Yao Liu, Weizhi Zhang, Qin Lu, Liang Qiu, Changlong Yu, Puyang Xu, Chao Zhang, Bing Yin, et al. Webagent-r1: Training web agents via end-to-end multi-turn reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7920–7939, 2025.
- [47] Qianhui Wu, Kanzhi Cheng, Rui Yang, Chaoyun Zhang, Jianwei Yang, Huiqiang Jiang, Jian Mu, Baolin Peng, Bo Qiao, Reuben Tan, et al. Gui-actor: Coordinate-free visual grounding for gui agents. arXiv preprint arXiv:2506.03143, 2025.

- [48] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: Foundation action model for generalist gui agents. In International Conference on Learning Representations, volume 2025, pages 5090–5108, 2025.
- [49] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.
- [50] Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? assessing the current state of web agents. In Second Conference on Language Modeling, 2025.
- [51] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the computer vision and pattern recognition conference, pages 14203–14214, 2025.
- [52] Ke Yang, Yao Liu, Sapana Chaudhary, Rasool Fakoor, Pratik A Chaudhari, George Karypis, and Huzefa Rangwala. Agentoccam: A simple yet strong baseline for llm-based web agents. In International Conference on Learning Representations, volume 2025, pages 97533–97565, 2025.
- [53] Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, et al. Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents. arXiv preprint arXiv:2502.09560, 2025.
- [54] Rui Yang, Ruomeng Ding, Yong Lin, Huan Zhang, and Tong Zhang. Regularizing hidden states enables learning generalizable reward model for llms. Advances in Neural Information Processing Systems, 37:62279–62309, 2024.
- [55] Rui Yang, Qianhui Wu, Zhaoyang Wang, Hanyang Chen, Ke Yang, Hao Cheng, Huaxiu Yao, Baoling Peng, Huan Zhang, Jianfeng Gao, and Tong Zhang. GUI-Libra: Training native GUI agents to reason and act with action-aware supervision and partially verifiable RL, 2026.
- [56] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.
- [57] Kuai Yu, Naicheng Yu, Han Wang, Rui Yang, and Huan Zhang. How do visual attributes influence web agents? a comprehensive evaluation of user interface design factors. arXiv preprint arXiv:2601.21961, 2026.
- [58] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [59] Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, and Sergey Levine. Fine-tuning large vision-language models as decision-making agents via reinforcement learning, 2024.
- [60] Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, and Daniel Kang. Beat: Visual backdoor attacks on vlm-based embodied agents via contrastive trigger learning, 2026.
- [61] Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou, Zehan Qi, Hanyu Lai, Yifan Xu, Rui Lu, Hongning Wang, Jie Tang, and Yuxiao Dong. AgentRL: Scaling agentic reinforcement learning with a multi-turn, multi-task framework, 2025.
- [62] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Yixin Cao, Yang Feng, and Deyi Xiong, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics

- (Volume 3: System Demonstrations), pages 400–410, Bangkok, Thailand, August 2024. Association for Computational Linguistics.
- [63] Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 414–431, 2025.
- [64] Yifei Zhou, Qianlan Yang, Kaixiang Lin, Min Bai, Xiong Zhou, Yu-Xiong Wang, Sergey Levine, and Erran Li. Proposer-agent-evaluator (PAE): Autonomous skill discovery for foundation model internet agents. In International Conference on Machine Learning, 2025.
- [65] Yuchen Zhuang, Di Jin, Jiaao Chen, Wenqi Shi, Hanrui Wang, and Chao Zhang. Workforceagent-r1: Incentivizing reasoning capability in llm-based web agents via reinforcement learning. In Findings of the Association for Computational Linguistics: EACL 2026, pages 34–49, 2026.

### A Implementation Details

#### A.1 Robust Web Environment Infra

To support browser-agent training on live websites, we implement a robust web environment that explicitly handles the instability of the open web. Unlike self-hosted benchmark sites, live websites exhibit frequent layout changes, asynchronous loading, pop-ups, redirects, anti-automation defenses, network errors, and non-deterministic page states.

Sandboxed Execution. The environment is built on Playwright with a Chromium backend and supports sandboxed execution by default. Each rollout runs inside an isolated Kubernetes sandbox [33] with its own browser environment server, preventing crashes, stale cookies, memory leaks, and site-specific side effects from contaminating other trajectories.

Timeouts and Recovery. The browser environment uses separate timeouts for initialization, interaction, and screenshot capture. Initial navigation is retried multiple times to handle common live-site failures such as HTTP/2 errors, connection resets, slow scripts, and transient bot-detection behavior.

Failure Attribution and Diagnostics. The rollout layer separates model failures from environment and infrastructure failures using explicit termination reasons, including task completion, maximum-step exhaustion, generation length limits, formatting errors, environment step errors, sandbox failures, and initialization failures. The system further records diagnostics such as server latency, memory usage, uptime, request counts, and tracebacks, enabling post-hoc analysis of whether a low reward reflects model behavior or external factors such as blocking, network instability, or server failure. Network-related initialization errors can also update a host blacklist, allowing future training to avoid websites that are repeatedly unreachable or automation-hostile.

#### A.2 Textual Environment Feedback

Browser agents operate inside an instrumented web environment, which exposes state signals that are not directly visible from screenshots. These signals include the DOM element under an action coordinate, the currently focused input field, the active URL, tab state, and lightweight accessibility-tree changes. If the agent relies only on visual input, it often needs multiple historical screenshots to infer whether the current state has changed after an action. For example, after a click, a screenshot alone may not clearly indicate whether the intended element was selected, whether an input field was focused, or whether navigation was blocked. Similarly, after a scroll, the agent may not know whether the page actually moved or has already reached the boundary. Without such feedback, the agent can easily repeat the same failed click or continue scrolling in the current state. To provide this information more efficiently, we augment visual observations with lightweight textual environment feedback.

For each parsed tool call, the environment generates feedback by comparing browser states before and after action execution. Before executing an action, it records lightweight state variables such as the active page URL, the number of open tabs, the current scroll position, the focused element, and, when applicable, the DOM element located at the action coordinates. The environment then executes the corresponding Playwright operation and applies action-specific rule logic to convert the observed state change into a concise textual message. When an action contains multiple tool calls, the environment processes them sequentially and returns the feedback as a list, with each entry corresponding to one executed tool call.

Several representative feedback rules are summarized below and in Table 5. For click, the feedback uses the clicked coordinate, the DOM element returned by document.elementFromPoint, changes in the page URL, and changes in the number of open tabs. For write, the feedback reports the currently focused element and compares the intended text with the actual value in the field after typing. For scroll, the feedback compares pre- and post-action scroll offsets to identify successful scrolling, boundary conditions, or no-op actions. For press keys, the environment reports the normalized key sequence and whether the action triggers a URL change.

Table 5 Examples of action-level environment feedback returned by the browser environment. Feedback exposes both execution status and observable browser-state changes.

Action Feedback signal Example feedback click Target element, location, navigation or

Succeed: ‘click‘ on <button> "Search" at (512, 86) executed. Page navigated to ...

new-tab effect

click No-op detection Succeed: ‘click‘ on <div> at (211,

335) executed. Note: no visible navigation or new tab detected.

write Focused element and typed content Succeed: ‘write‘ typed "Alpine Ridge" into <input> role=combobox "Search".

write Actual value mismatch Note: the field’s actual value is "New York, NY", which differs from the typed text.

scroll Scroll direction, amount, and boundary detection

Succeed: ‘scroll‘ down by 50% executed. Note: page scroll position did not change and may be at a boundary.

press keys Normalized keys and navigation effect Succeed: ‘press keys‘ [‘Enter‘] executed. Page navigated to ...

goto url Direct navigation success or failure Failed: ‘goto url‘ execution failed: Page.goto: net::ERR HTTP2 PROTOCOL ERROR ...

tab ops New, switched, or closed tab state Succeed: ‘switch tab‘ switched to tab 1 (https://example.com).

done Task termination with final response Succeed: task marked as done. Response: ...

Execution exceptions are also caught and converted into explicit failure messages rather than silently terminating the rollout. Successful actions return concise messages describing both execution status and observable browser-state changes. The resulting feedback is appended to the next observation, giving the policy a compact signal for detecting failed clicks, blocked navigation, unintended tab changes, rejected inputs, boundary scrolling, and other common open-web interaction issues.

#### A.3 Action Space

We define a compact browser action space consisting of 13 atomic tools, as summarized in Table 6. These tools cover the core operations needed for open-web interaction, including pointer control, keyboard input, page navigation, tab management, and task termination. Each action is represented as a structured tool call with explicit arguments, which makes the policy output easy to parse and execute in the browser environment.

#### A.4 Reward Design

Our browser RL reward is a hybrid of deterministic rule-based verification and LLM-as-a-judge evaluation. The deterministic rules serve two purposes: they enforce the interaction protocol required by the browser environment, and they avoid unnecessary judge calls for trajectories that are clearly invalid or incomplete. The judge model is only used for completed trajectories that pass these basic validity

Table 6 Browser action space: 13 atomic tools grouped by family.

###### Category Tool Argument Description

Mouse-click at a screen pixel; supports single/double click and left/right/middle button.

click x, y, button, click type

Pointer Mgmt.

hover x, y Move the cursor to a pixel to reveal tooltips or open dropdowns.

drag x1, y1, x2, y2 Drag-and-drop from a start pixel to an end pixel. Keyboard Mgmt.

write text Clear the focused input and type a string. press keys keys Press keys sequentially or as a hotkey combo.

Scroll the page or element by a viewport fraction.

scroll direction, amount

Page Nav.

goto url url Navigate the current tab to a given URL. go back — Navigate back in the browser history. wait seconds Pause for N seconds to allow the page to settle.

new tab — Open a new blank browser tab. switch tab index Switch to the tab with the given 0-based index. close tab — Close the current tab.

Tab Mgmt.

Termination done answer End the episode and emit the final answer.

checks. For each generated trajectory τi, we compute three reward-related quantities: a format score F(τi), a judge score J(τi), and the final training reward R(τi).

Rule-based format verification. Before invoking the judge model, we verify that the agent response follows the required browser-action format. In our response format, a turn is considered valid only if it contains the expected closing thinking tag and a parseable tool call. This ensures that the model emits an executable browser action rather than free-form text.

Rule-based status filtering. The judge model is not called for trajectories that do not reach a completed browser state, where agent outputs ’done’ by itself. If the rollout terminates because of malformed tool calls, generation aborts, context-length truncation, environment errors, or exhausting the maximum number of browser steps, the judge score is set to zero by rule:

J(τi) = 0. Malformed format termination is treated as a special failure case and receives a negative final reward:

R(τi) = −1 if the trajectory terminates due to repeated format errors. This penalizes trajectories that cannot produce executable browser actions.

Even for completed trajectories, the judge model is skipped if no final answer can be extracted. The final answer must come from the terminal done tool call. If the final answer is missing, we set J(τi) = 0 without querying the judge.

VLM-as-a-judge evaluation. For completed trajectories with a valid final answer, we query a judge model to determine task success. The judge input contains the original task instruction, the agent’s final answer, three recent screenshots from the trajectory, and an explicit action history summarizing the agent’s tool calls and environment feedback. The judge is instructed to return a verdict of either SUCCESS or NOT SUCCESS.

The judge output is mapped back to a binary score by a deterministic parser:

 

1, if the judge output contains SUCCESS, 0, if the judge output contains NOT SUCCESS, 0, if the judge output cannot be parsed.

J(τi) =



The parser checks for NOT SUCCESS before SUCCESS to avoid mistaking a negative verdict for a positive one.

Final reward composition. The final reward is a gated combination of the rule-based checks and the judge score:

 

−1, if the trajectory terminates due to repeated format errors, 0, if F(τi) = 0, J(τi), otherwise.

R(τi) =



Thus, a trajectory can receive reward 1 only if it both follows the required browser-action format and is judged successful.

Invalid or unreliable samples. If reward evaluation fails because of judge timeout or infrastructurerelated rollout failures, the sample can be marked for removal from optimization. Such samples have their loss mask zeroed, so they do not contribute gradients. This keeps reward-model or environment-system failures from introducing noisy policy updates.

#### A.5 Training Details

SFT. We train Qwen3-VL-4B-Thinking for 3 epochs with peak learning rate 10−5 under a cosine schedule with a 10% linear warmup. Each optimizer step uses a per-device batch of 2 with 8-step gradient accumulation, giving a per-worker effective batch of 16 and a global batch of 128 across 8 data-parallel workers. For the 8B model, we use the same training setup, with the only difference being the SFT dataset: we add 500 trajectories from the InSTA-v3 subset, resulting in 912 trajectories in total.

MM-GRPO. The policy is initialized from the SFT checkpoint. During training, each task is executed interactively in a live browser environment. Each rollout state comprises the current observation, recent screenshot context, and environment feedback in browser tool-call format. Rewards combine format validity with an LLM-as-a-judge success signal, where the judge receives the full action history and recent trajectory screenshots as input. To ensure stable training, we apply bounded rollout execution with explicit timeouts governing model generation, browser actions, sandbox acquisition, and task-level rollout completion. Training runs for 90 iterations and requires approximately 300 B200 GPU hours in total. We employ trajectory-level dynamic sampling, collecting rollouts until 48 effective groups of size 5 are obtained. Key training hyperparameters for our OpenWebRL-4B are summarized in Table 7. Regarding the 8B model, we use a lower learning rate of 5 × 10−7.

#### A.6 Evaluation Details

Each evaluation task is executed as an interactive browser trajectory with a maximum of 30 steps. We serve our model using the SGLang framework. For official success rate evaluation, we follow the judge protocol of prior work and use Browser-Use Stealth Browsers. Based on the finding of [17] that stochastic sampling outperforms deterministic sampling for web agents, we adopt stochastic decoding with the following hyperparameters: temperature 0.6, top-p 0.95, top-k 20, max response length 4096 tokens, and repetition penalty 1.0. For success rate evaluation without aborted tasks, we use deterministic decoding with temperature 0.0. All evaluation settings and decoding parameters are summarized in Table 8.

### B Benchmark Details

We evaluate on three online benchmarks:

• WebVoyager [18] is a 643-task open-domain benchmark spanning 15 popular websites. We use the version curated by FARA [3], which removes infeasible tasks and updates those with outdated

Table 7 Training hyperparameters for browser-agent reinforcement learning. Values summarize the Qwen3-VL 4B browser training configuration in examples/browser; ranges indicate script variants.

###### Hyperparameter Value

Number of training iterations 90 Max rollout steps 15, 30 Context screenshots per turn 1 Judge screenshots per trajectory 3 Judge model GPT-4.1 or our trained 8B judge model Effective rollout query number per iteration 48 Group size per prompt 5 PPO epochs 2 Maximum response length 1024 Maximum context length 32768 RL algorithm MM-GRPO KL coefficient 0.0 Entropy coefficient 0.0 PPO clipping range low 0.2, high 0.28 Global batch size for optimization 256 Optimizer Adam Learning rate 1 × 10−6, constant schedule Weight decay 0.1 Adam betas (0.9, 0.98) Training backend Megatron Tensor parallel size 4 Micro-batch size 1 Attention backend FlashAttention Browser viewport 1280 × 1000, DPR 1 Coordinate normalization scale 1000 Sampling temperature 0.8 Rollout engine SGLang Sandbox mode Kubernetes sandbox Sandbox CPU and memory 1 CPU, 4 GiB Max concurrent sandboxes 80–100 during training Inference step timeout 30 s Rollout task timeout 600 s Sandbox acquire timeout 600 s Browser step request timeout 45 s Browser step retries 2 Environment exit timeout 15 s Rollout abort wait timeout 30 s Rollout health check Interval 30 s, timeout 30 s, first wait 180 s

information from the original dataset, resulting in 595 tasks in total2. For evaluation, we follow the canonical WebVoyager protocol3 and use GPT-4o as the judge, consistent with FARA.

• Online-Mind2Web [50] comprises 300 longer-horizon tasks drawn from 136 popular websites across diverse domains. Compared to WebVoyager, these tasks typically require more extended multistep reasoning and interaction sequences. For evaluation, we adopt the OSU AgentTrek protocol4,

- 2https://github.com/microsoft/fara/blob/44908264c810d3806365c6aab63a43c2d52a8057/webeval/data/webvoyager/

WebVoyager data 08312025.jsonl#L4

- 3https://github.com/MinorJerry/WebVoyager/blob/main/evaluation/auto eval.py

- 4https://github.com/OSU-NLP-Group/Online-Mind2Web/blob/main/src/methods/agenttrek eval.py

Table 8 Evaluation hyperparameters for browser-agent evaluation in examples/browser.

###### Hyperparameter Value

Max evaluation rollout steps 30 Sandbox mode Kubernetes sandbox Sandbox CPU and memory 1 CPU, 4 GiB Parallel browser sandboxes 16 Context screenshots per turn 1 Judge screenshots per trajectory 3 Judge model GPT-4.1 for our evaluation success rate w/o aborted tasks;

officially recommended models for reporting the official score Judge timeout 120 s Generation temperature 0.6 for the official-score evaluation and 0.0 for evaluation success rate w/o aborted tasks Top-p 0.95 for the official-score evaluation and 1.0 for evaluation success rate w/o aborted tasks Top-k 20 for the official-score evaluation and 1 for evaluation success rate w/o aborted tasks Maximum response length 4096 tokens Maximum context length 32768 tokens Task timeout 600 s Inference step timeout 120 s Rollout engine SGLang SGLang dtype bfloat16 SGLang context length 32768 tokens SGLang tensor parallel size 2 SGLang data parallel size 4

following the standard setup used in prior work with o4-mini as the judge.

• DeepShop [30] is a 150-task benchmark focused on online shopping scenarios, where agents must search, compare, and select products under realistic constraints. Task success is determined using Molmo-Web’s structured-output evaluation protocol [17] with GPT-4o as the judge, which programmatically verifies the correctness of the final response5.

### C Additional Experiment Results

#### C.1 Task Difficulty Understanding

To characterize task difficulty and the number of steps required for task completion, we analyze Qwen3VL-235B-Thinking’s results across three online benchmarks. As shown in Figure 8, Online-Mind2Web and DeepShop exhibit lower success rates and require more steps to complete successfully (11.5 and 14.2 on average, respectively) compared to WebVoyager (8.0). Note that the step counts reported in Figure 8 are averaged over all trajectories, including failed ones. These results confirm that Online-Mind2Web and DeepShop are more challenging benchmarks, demanding longer action sequences and yielding lower success rates. The stronger performance of OpenWebRL-4B on precisely these harder benchmarks (Table 2) therefore provides stronger evidence for the effectiveness of our training framework.

#### C.2 Results on AgentRewardBench

To further evaluate the generalizability and effectiveness of our judge, we report results on AgentRewardBench [27]. AgentRewardBench contains 1,302 trajectories from five widely used web-agent benchmarks: WebArena (WA), VisualWebArena (VWA), AssistantBench (AB), WorkArena (Work), and WorkArena++ (Wk++). As shown in Table 9, OpenWebRL-Judge-8B achieves the best overall

5https://github.com/allenai/molmoweb/blob/main/benchmarks/judges/deepshop judge.py

Online-Mind2Web

DeepShop

WebVoyager

45

90

100

(success rate 63.7% · average step 16.7)

(success rate 56.7% · average step 19.0)

40

80

(success rate 66.4% · average step 12.1)

25

succeeded ( =8.3) failed ( =19.8)

14 succeeded ( =14.2)

80

succeeded ( =11.5) failed ( =25.9)

failed ( =25.2)

| |
|---|

| |
|---|

12

20

60

10

#tasks

#tasks

#tasks

15

8

40

6

10

20

4

5

2

0

0 5 10 15 20 25 30

0

0

0 5 10 15 20 25 30

0 5 10 15 20 25 30

total steps

total steps

total steps

Figure 8 Trajectory length (total steps) distribution of Qwen3-VL-235B-Thinking.

F1 and Recall, while maintaining competitive Precision across benchmarks. OpenWebRL-Judge-4B also performs strongly, achieving the best Precision on WebArena and WorkArena.

It is worth noting that AgentRewardBench is built on a different agent stack, introducing distribution shift in both action representation and available environment feedback. In particular, AgentRewardBench contains action types outside our action space, making some action conversions approximate. Its trajectories also lack the per-step environment feedback used by our judges, leaving only partial signals such as URL changes, propagated errors, and accessibility-tree fragments. We believe these results should be viewed as a lower-bound estimate of our performance under this out-of-distribution setting.

- Table 9 Fine-grained evaluation results on AgentRewardBench. We report overall Precision, Recall, and F1, as well as per-benchmark Precision across AssistantBench (AB), VisualWebArena (VWA), WebArena (WA), WorkArena (Work), and WorkArena++ (Wk++). Rows marked with † are taken from AgentRewardBench [27] of its simple judge with final screenshot, and rows marked with ∗ are taken from Online-Mind2Web [50].

Overall Per-Benchmark Precision Precision Recall F1 AB VWA WA Work Wk++

Judge

Claude 3.7 S.† 69.4 76.3 72.7 71.4 64.8 69.3 85.3 66.7 GPT-4o† 68.1 80.3 73.7 77.8 60.7 69.9 93.8 59.6 GPT-4o Mini† 64.5 78.3 70.8 80.0 57.4 66.9 90.3 54.8 WebJudge (GPT-4o)∗ 73.7 71.2 72.4 66.7 69.8 72.6 92.3 75.0 WebJudge-7B∗ 75.7 58.0 65.6 80.0 66.7 77.5 100.0 70.0 OpenWebRL-Judge-4B 73.2 67.0 70.0 75.0 62.2 78.0 92.6 70.0 OpenWebRL-Judge-8B 72.8 72.5 72.7 80.0 67.0 73.5 82.9 74.2

#### C.3 Comparison with Online Filtered Behavior Cloning Baseline

Figure 9 compares MM-GRPO with an online filtered behavior cloning (BC) baseline. The BC baseline uses the same online training setup as MM-GRPO on top of the same SFT checkpoint, but only keeps successful trajectories for supervised updates, uses a fixed rollout query batch size, and does not apply dynamic sampling. Although online filtered BC initially achieves competitive training rewards, its evaluation success rate without aborted tasks steadily declines throughout training, eventually dropping below 30%. In contrast, MM-GRPO maintains stable optimization and continuously improves evaluation performance, reaching over 50% evaluation success rate, indicating that simply imitating filtered online trajectories is insufficient for robust policy improvement in open-web environments. These results highlight the importance of reinforcement learning updates, rather than pure supervised imitation, for effective online adaptation and long-horizon web interaction.

Training Reward

Eval Success Rate w/o Aborted Tasks

| |MM-|GRPO| | | |
|---|---|---|---|---|---|
| |Onlin|e Filtered|BC| | |
| | | | | | |
| | | | | | |
| | | | | | |

|MM-|GRPO| | | |
|---|---|---|---|---|
|Onlin|e Filtered|BC| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

60

50

TrainingReward(%)

SuccessRate(%)

45

50

40

40

35

30

30

25

0 20 40 60 80

0 20 40 60 80

Iteration

Iteration

Figure 9 Comparison with online filtered behavior cloning baseline.

- Table 10 Lexical proxies used to analyze response-length growth. The proxies are regex-based, interpretable, and not mutually exclusive.

###### Proxy What it measures Example phrase families Motivation

History summarization

The agent lists previous attempts, actions, failures, or exhausted sources.

“I’ve tried”, “we already checked”, “previous attempt”,

Long responses often contain detailed inventories of prior failures, especially near termination.

“so far”, “multiple approaches”, “all sources failed”.

Blocker diagnosis The agent diagnoses CAPTCHA, verification, access denial, or automationrelated blockers.

“CAPTCHA”, “Cloudflare”,

Many long-tail failures arise from web security or anti-automation barriers.

“403 forbidden”, “verification required”, “anti-bot”, “automated access”, “cannot solve verification”.

Retry-plan loop The agent considers repeated alternatives, direct navigation, or “one more” attempts.

“one final approach”, “try a different search”, “alternative source”, “go back”, “search directly”, “let me reconsider”.

Captures cases where the agent continues planning retries instead of terminating or taking a concise action.

Condition proof The agent checks task constraints one by one or explicitly verifies that concrete constraints are satisfied.

“meets the requirements”, “all criteria met”, “first condition”,

Verbose successful or partially successful cases often explicitly verify each user constraint.

“under $50”, “within 5 miles”, “gluten-free”.

### D Proxy-Based Analysis of Response Length

To better understand why response length increases across training iterations, we define four lightweight lexical proxies over agent responses. These proxies are not intended to be exhaustive semantic labels; instead, they provide interpretable, automatically computable indicators of common reasoning patterns observed in long responses. We apply them to assistant turns by matching precision-oriented regular expression families in the reasoning text before the first tool call. These proxies are deliberately simple and transparent. A response may match multiple proxies, and the categories are not mutually exclusive. For example, a response can both summarize failed attempts and diagnose a CAPTCHA blocker. We therefore use these proxies as descriptive measurements rather than ground-truth semantic labels. We summarize those proxies in Table 10.

All regular expressions are evaluated case-insensitively. A step is marked as proxy-bearing if it matches any pattern from one of the four proxy families. We also separately track responses with none of the four proxies as a baseline. This no-proxy baseline shows smaller length growth than proxy-bearing responses, suggesting that these four patterns account for a meaningful portion of the observed per-step verbosity increase.

In our analysis, the proxies already appear in early iterations, indicating that later training does not introduce entirely new reasoning modes. Instead, later iterations tend to amplify these patterns when they occur. In particular, history summarization and retry-plan loops are associated with longer responses and heavier response-length tails. Blocker diagnosis captures common failure states caused by CAPTCHA or access restrictions, while condition proof explains why even valid or successful responses may become more verbose: the agent increasingly justifies its answer by checking task requirements explicitly.

Exact matching patterns. For reproducibility, we include the exact patched proxy definitions used in the analysis below. We use a compact style to keep long regular expressions readable.

|HISTORY_SUMMARIZATION = [ r"\b(?:i(?:’ve| have)|we(?:’ve| have))\s+(?:already\s+)?" r"(?:tried|attempted|checked|searched|looked at|visited|used)\b", r"\b(?:i|we)\s+already\s+" r"(?:tried|attempted|checked|searched|looked at|visited|used)\b", r"\b(?:i|we)\s+had\s+(?:already\s+)?" r"(?:tried|attempted|checked|searched|looked at|visited|used)\b", r"\b(?:previous|earlier|last)\s+" r"(?:attempt|search|approach|step|site|page)\b", r"\b(?:so far|up to this point),?\s+(?:i|we)\s+(?:have|’ve|had)\b", r"\b(?:tried|attempted|checked|searched|looked at|visited|used)\b" r"[^.!?\n]{0,80}\b(?:multiple|several)\s+" r"(?:attempts|approaches|searches|sites|websites|sources)\b", r"\b(?:multiple|several)\s+" r"(?:attempts|approaches|searches|sites|websites|sources)\b" r"[^.!?\n]{0,80}\b(?:failed|blocked|led to|resulted in|did not|" r"didn’t|couldn’t|cannot)\b", r"\b(?:all|every)\s+(?:attempt|approach|site|source|path|option)s?\s+" r"(?:has|have)\s+(?:failed|led to|resulted in)\b", r"\b(?:exhausted|tried)\s+(?:all|every|multiple|several)\s+" r"(?:options|approaches|paths|sources|sites)\b",<br><br>]<br><br>|
|---|

Listing 1 History summarization proxy patterns.

|BLOCKER_DIAGNOSIS = [ r"\b(?:captcha|recaptcha|cloudflare|access denied|403\s+forbidden|" r"robot check)\b", r"\baccess\s+(?:is\s+)?forbidden\b", r"\b(?:blocked|blocking|unable to access|cannot access|can’t access|" r"not accessible)\b", r"\b(?:(?:security|bot|human)\s+verification|verification\s+" r"(?:challenge|required|page|screen|check))\b", r"\b(?:security check|security measure|anti[- ]bot|anti[- ]automation)\b", r"\b(?:automated access|automated behavior|bot detection|" r"detected automated)\b", r"\b(?:requires?|needs?)\s+human\s+" r"(?:interaction|verification|input)\b", r"\b(?:cannot|can’t|unable to)\s+(?:interact with|solve|complete)\s+" r"(?:the\s+)?(?:captcha|recaptcha|verification)\b",<br><br>]<br><br>|
|---|

Listing 2 Blocker diagnosis proxy patterns.

RETRY_PLAN_LOOP = [ r"\b(?:one|a)\s+(?:final|last)\s+(?:approach|attempt|try|option)\b", r"\b(?:try|attempt|use|take)\s+(?:a\s+)?" r"(?:different|another|alternative|new)\s+" r"(?:approach|search|query|site|source|path|method)\b", r"\b(?:need|needs|needed|should|could|might|may|will|would|try|use|" r"take|switch to|look for|search for)\b[^.!?\n]{0,50}\b" r"(?:different|alternative|another|new)\s+" r"(?:approach|search|query|site|source|path|method)\b", r"\b(?:different|alternative|another|new)\s+" r"(?:approach|search|query|site|source|path|method)\b" r"[^.!?\n]{0,50}\b(?:is needed|may work|might work|could work|" r"should work|would help|to try|to search|to use)\b", r"\b(?:go back|return)\s+(?:and|to)\s+(?:try|search|look|navigate)\b", r"\b(?:navigate directly|search directly|try searching directly|" r"search more broadly)\b", r"\b(?:let me|i should)\s+" r"(?:reconsider|try again|try another|switch to|use a different)\b", r"\b(?:rather than|instead of)\s+[^.!?\n]{0,80}\b"

r"(?:i should|i’ll|i will|let me)\s+(?:try|search|navigate|use)\b", ]

Listing 3 Retry-plan loop proxy patterns.

|PROOF_VERB = ( r"\b(?:meets?|matches?|satisf(?:y|ies|ied)|fulfills?|" r"qualif(?:y|ies|ied)|confirms?|verif(?:y|ies|ied)|fits?|" r"complies with|addresses)\b"<br><br>) REQ_WORD = (<br><br>r"(?:criteria|requirements|conditions|constraints|criterion|" r"requirement|condition|constraint)"<br><br>) REQ_MET = PROOF_VERB + r"[^.!?\n]{0,80}\b" + REQ_WORD + r"\b" ALL_MET = (<br><br>r"\b(?:all|each|every)\s+(?:the\s+)?" r"(?:criteria|requirements|conditions|constraints)\b" r"[^.!?\n]{0,80}\b(?:met|satisfied|fulfilled|addressed|checked)\b"<br><br>) ORDINAL_REQ = (<br><br>r"\b(?:(?:first|second|third|fourth|fifth|sixth|seventh|eighth|" r"ninth|tenth|final|last)|\d+(?:st|nd|rd|th)?)\s+\b"<br><br>+ REQ_WORD + r"\b"<br><br>) VALUE_PATTERNS = [<br><br>r"\b(?:under|less than|below)\s+\$?[0-9][0-9,]*(?:\.[0-9]+)?\b", r"\b(?:at least|at most|no more than|fewer than|greater than)\s+" r"[0-9][0-9,]*(?:\.[0-9]+)?\b", r"\bwithin\s+[0-9][0-9,]*(?:\.[0-9]+)?\s*" r"(?:miles?|mi|km|kilometers?)\b", r"\b(?:before|after)\s+[0-9]{1,2}:[0-9]{2}\b", r"\b(?:model years?|years?)\s+[0-9]{4}\s*(?:-|to)\s*[0-9]{4}\b",<br><br>] DOMAIN_PATTERNS = [<br><br>r"\bgood with kids\b", r"\bgood with cats\b", r"\bgluten[- ]free\b", r"\bnut[- ]free\b", r"\bshort[- ]haired\b", r"\bdepart(?:ing|s)? from\s+[A-Z][A-Za-z .-]+", r"\blocated near\s+[A-Z0-9][A-Za-z0-9 .-]+",<br><br>] def has_condition_proof(step):<br><br>if re.search(REQ_MET, step, flags=re.I): return True if re.search(ALL_MET, step, flags=re.I): return True if re.search(ORDINAL_REQ, step, flags=re.I): return True<br><br>constraints = VALUE_PATTERNS + DOMAIN_PATTERNS for sent in re.split(r"[.!?\n]+", step):<br><br>has_proof = re.search(PROOF_VERB, sent, flags=re.I) n_constraints = sum(<br><br>bool(re.search(pat, sent, flags=re.I)) for pat in constraints<br><br>) has_value = any(<br><br>re.search(pat, sent, flags=re.I) for pat in VALUE_PATTERNS<br><br>) if has_proof and has_value:<br><br>return True if has_proof and n_constraints >= 2:<br><br>return True return False<br><br>|
|---|

Listing 4 Condition proof proxy patterns and composite rules.

def proxy_labels(step): labels = set() if any(re.search(p, step, flags=re.I) for p in HISTORY_SUMMARIZATION):

labels.add("history_summarization") if any(re.search(p, step, flags=re.I) for p in BLOCKER_DIAGNOSIS): labels.add("blocker_diagnosis") if any(re.search(p, step, flags=re.I) for p in RETRY_PLAN_LOOP):

labels.add("retry_plan_loop") if has_condition_proof(step):

labels.add("condition_proof") return labels

def is_non_proxy(step): return len(proxy_labels(step)) == 0

Listing 5 Final proxy and non-proxy labels.

Limitations. Because the proxies are regex-based, they may miss paraphrases or include mild false positives. The retry-plan proxy is intentionally broad enough to recover context-anchored alternativeapproach language, but this also means that some benign planning statements may be counted. The condition-proof proxy is more conservative: generic mentions of constraints are not sufficient unless they match explicit requirement-satisfaction patterns or co-occur with concrete constraints and a proof verb in the same sentence. Nevertheless, the proxy definitions are interpretable, easy to reproduce, and useful for identifying high-level mechanisms behind response-length growth.

### E Error Analysis

We manually inspect 100 failed trajectories sampled from evaluation runs of multiple OpenWebRL checkpoints on Online-Mind2Web without the Browser-Use Stealth Browser service, and categorize them into four failure types. Figure 10 summarizes the distribution.

Access and environment issues (51%). The largest category stems from live-web instability, including page loading failures, access restrictions, and CAPTCHA blockers. In these cases, the agent typically follows a reasonable strategy, but environmental obstacles prevent the trajectory from reaching a verifiable terminal state.

Reasoning and knowledge limitations (27%). The second major category reflects model-side deficiencies in planning and constraint tracking. In long-horizon tasks—particularly shopping or search tasks involving multiple simultaneous requirements such as price, color, rating, size, and product type—the agent may satisfy most constraints while overlooking one or more. A smaller share of failures arises when the task presupposes background or domain-specific knowledge that the model does not reliably possess.

Visual grounding and interaction errors (13%). Another common failure mode is inaccurate interaction with the page. The agent may click a nearby but incorrect element, miss a small dropdown or pagination control, or fail to notice that a filter has not been applied. These errors are more likely when multiple actions are executed before the next observation, making it harder for the agent to diagnose intermediate failures.

Task definition and evaluation issues (9%). The remaining failures stem from ambiguous or underspecified task instructions, with a small fraction attributed to judge errors—cases where the agent completes the task correctly but the automated judge incorrectly marks it as incomplete.

Overall, this error analysis indicates that further progress requires greater robustness to open-web instability, more reliable long-horizon constraint maintenance, improved intermediate-state monitoring, and stronger recovery mechanisms for failed interactions.

[Figure 3]

Network

Connection (43%)

Access and Environment Issues (51%)

CAPTCHAs (8%)

Task Definition Issue (3%)

Reasoning and Knowledge Limitations (27%)

Evaluation Issue (6%)

Perception Error (13%)

Intermediate

Reasoning Error (19%)

State Not Observed (5%)

Repet

Grounding Error (8%)

Lack Knowledge (6%)

ition (2%)

Figure 10 Distribution of failure modes based on a manual inspection of 100 failed trajectories.

### F Prompt Templates

This section presents prompt examples used in our pipeline. For rollout examples, we show one user turn and the corresponding assistant turn in the exact serialized conversation format consumed by the model. For judge examples, we provide the full template together with one concrete example constructed from a real trajectory.

#### F.1 Agent Input and Output Example Rollout Input: System Prompt

<|im start|>system You are a GUI agent designed to operate in an iterative loop to automate browser tasks.

# GUI Agent Policy

- As an autonomous GUI agent operating on the Web Browser platform, your primary function is to analyze screen captures and perform appropriate UI actions to complete assigned tasks.

## Core Responsibilities

You can perform web browser interactions including mouse interactions, keyboard interactions , navigation, tab management, task completion, and waiting.

## Input Information

- At each step, you will receive the following information:

- 1. Action History
- 2. User Request
- 3. Observation, including tab info, screenshot, and optional A11y Tree

## Output Requirements

Your output must include one <think> block and one or more <tool call> blocks. The response must follow this exact structure: <think> ... </think> <tool call> {"name": ..., "arguments": ...} </tool call>

## Guidelines

Analyze the current browser state, reflect on prior actions, assess progress toward the goal , plan the next step, and emit only valid executable tool calls. When appropriate, multiple sequential tool calls may be emitted in one response. The final done.response must be plain text only, with no LaTeX-style markup.

# Tools

... <|im end|>

##### Example Rollout Input: User Prompt

<|im start|>user <tool response> Succeed: `click` on <rect> at (237, 310) executed. Note: no visible navigation or new tab detected. Succeed: `click` on <button> type=button "Apply" at (393, 557) executed. Page navigated to https://www.healthgrades.com/usearch.

<observation> screen size: 1280 x 1000 Current tab: 0 Available tabs:

- Tab 0 (active): https://www.healthgrades.com - Healthgrades > Universal Search > Find a Doctor screenshot: <|vision start|><|image pad|><|vision end|> </observation> </tool response><|im end|>

[Figure 4]

Screenshot attached in the raw rollout input above

##### Example Rollout Output

<|im start|>assistant <think> The search has successfully found dentists with 5-star ratings near Quincy, WA (zip code 98011 area). There are exactly 2 results shown, both with 5.0 ratings. The first dentist, Dr. Jay Magnuson, is located at a distance of 0.2 miles, while the second, Dr. Paul Kwon, is 0.6 miles away.

The user asked for the closest 5-star rated dentist to zip code 98011. Based on the search results, Dr. Jay Magnuson is the closest at 0.2 miles. This completes the task of finding the closest 5-star rated dentist.

I should now provide the final answer with the name and location of the closest 5-star rated dentist.

</think> <tool call> {"name": "done", "arguments": {"response": "The closest 5-star rated dentist to zip code 98011 is Dr. Jay Magnuson, DDS, located at 201 H St SE, Quincy, WA 98848, which is 0.2 miles away. He has a 5.0 rating based on 1 review."}} </tool call><|im end|>

#### F.2 OpenWebRL-Judge Input and Output Prompt Template for OpenWebRL-Judge As an evaluator, you will be presented with four primary components to assist you in your role:

- 1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out.
- 2. Agent Action History: This is a chronological summary of the agent’s observed actions across steps. Use it to understand what the agent tried to do, but do not treat it as ground truth if it conflicts with the screenshots.
- 3. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. Each screenshot will be annotated with an inferred step index in text.
- 4. Result Response: This is a textual response obtained after the execution of the web task.

### TASK: {} ### Agent Action History: {} ### Result Response: {} ### {num} screenshots from the trajectory are attached below with inferred step indices.

## Guidelines

- • You DO NOT NEED to interact with web pages or perform actions.
- • You SHOULD use the screenshots as the strongest evidence about the actual page state.
- • You SHOULD use the action history to judge whether the agent followed the instruction and whether the final response is supported by what happened on screen.
- • If the action history conflicts with screenshots, trust the screenshots.

- • NOTE that the instruction may involve more than one task. Failing to complete either task should be considered unsuccessful.
- • NOTE that the final response may contradict the screenshots; in that case the screenshots prevail. If the final response contains information not visible in the screenshots, you may still consider it only if it is consistent with the screenshots and action history.
- • You should first explain your reasoning with explicit reference to the instruction, action history, screenshots, and final response. Then provide a definitive verdict as either SUCCESS or NOT SUCCESS.

## Response Format:

Return the reasoning first, then the verdict. Reasoning: Explain how you arrived at your final evaluation with explicit reference to the instruction, action history, screenshots, and final response.

Verdict: SUCCESS or NOT SUCCESS

Your verdict:

OpenWebRL-Judge Example Input As an evaluator, you will be presented with four primary components to assist you in your role. ### TASK: Find the closest 5-star rated dentist to zip code 98011.

### Agent Action History:

- Step 1: action=[click({"point 2d": [307, 630]}); write({"message": "dentist"}); press keys({"keys": ["enter"]})] environment feedback: click: Succeed: click on search input executed; write: typed "dentist"; press keys navigated to Healthgrades search results for dentists in Quincy, WA.

- Step 2: action=[click({"point 2d": [150, 207]})] environment feedback: clicked the Rating filter with no navigation.

- Step 3: action=[click({"point 2d": [185, 310]}); click({"point 2d": [307, 557]})] environment feedback: selected the 5-star filter and clicked Apply, navigating to filtered 5-star dentist results in Quincy, WA.

- Step 4: action=[done({"response": "The closest 5-star rated dentist to zip code 98011 is Dr. Jay Magnuson, DDS, located at 201 H St SE, Quincy, WA 98848, which is 0.2 miles away. He has a 5.0 rating based on 1 review."})] environment feedback: task marked as done with the final response above.

### Result Response: The closest 5-star rated dentist to zip code 98011 is Dr. Jay Magnuson, DDS, located at 201 H St SE, Quincy, WA 98848, which is 0.2 miles away. He has a 5.0 rating based on 1 review.

### 3 screenshots from the trajectory are attached below with inferred step indices.

[Figure 5]

[Figure 6]

[Figure 7]

Screenshot 1 Screenshot 2 Screenshot 3 ## Guidelines ...

## Response Format ...

Your verdict:

##### OpenWebRL-Judge Example Output

Reasoning: The task instruction is to find the closest 5-star rated dentist to zip code 98011. The action history shows that the agent searched for dentist, opened the Rating filter, selected 5 stars, and returned Dr. Jay Magnuson, DDS as the answer. However, the screenshots show that the search location remained Quincy, WA rather than zip code 98011. Screenshot 2 shows dentists near Quincy, WA after the initial search, and Screenshot 3 confirms that the final 5-star results are also in Quincy, WA, with Dr. Jay Magnuson at 0.2 miles and Dr. Paul Kwon at 0.6 miles from that Quincy search center. The final response repeats Quincy, WA and does not provide evidence that the result is near zip code 98011. Since the instruction explicitly requires a dentist near 98011 and the observed trajectory instead stays in Quincy, WA, the task was not completed successfully.

Verdict: NOT SUCCESS

### G Broader Impacts

OpenWebRL releases an open framework, training data, and a 4B-parameter model for training visual web agents with online multi-turn RL. We discuss the main societal considerations below.

Positive impacts. The strongest visual web agents today are proprietary, with training recipes and weights closed. By releasing our full pipeline, training tasks, and OpenWebRL-4B, we lower the barrier for academic and independent researchers to study and build on online RL for visual agents. Showing that competitive performance is achievable with modest initialization data also reduces reliance on expensive human demonstrations, making this line of research more accessible to resource-constrained groups. Capable open web agents further serve as building blocks for assistive applications that help users who struggle with conventional web interfaces.

Potential negative impacts. More capable visual web agents lower the technical barrier for automated scraping in violation of site terms of service, circumvention of bot-detection systems, and large-scale generation of inauthentic accounts or engagement. Multi-turn agents can also compound errors over long horizons, which is particularly concerning when deployed in settings involving purchases, form submissions, or other state-changing actions. Online RL on live websites additionally places traffic load on third-party servers.

Mitigations. We trained on benchmark environments and task distributions designed for agent research, and rate-limited interactions during data collection. OpenWebRL-4B is released at a scale where capabilities remain bounded relative to frontier proprietary systems, limiting the marginal uplift our release provides to a sophisticated misuse actor. We provide model and dataset cards documenting intended research use and encourage downstream users to respect site terms of service and applicable laws, and to apply human-in-the-loop confirmation or domain restrictions when deploying agents in user-facing settings. We believe open release with clear documentation better supports community work on agent safety, bot-detection defenses, and misuse evaluation than a closed release would.

### H Additional Case Studies

Figure 11 and Figure 12 show two representative long-horizon shopping tasks in which the agent must satisfy multiple user constraints through extended web interaction. In Figure 11, the agent searches for a top-selling men’s Birkenstock clog in brown and size 10–10.5. In Figure 12, it searches for the cheapest black leather sofa with at least three seats and a rating of at least four stars.

These trajectories highlight several agent capabilities. First, the agent decomposes the high-level request into concrete subgoals, including navigating to the relevant product category, applying filters, inspecting candidate products, selecting product options, and locating the final purchase control. Second, the agent exhibits reflective behavior by monitoring whether an action achieves the intended state and revising its strategy when it does not; for example, it switches from ineffective filtering or sorting operations to direct product inspection, and returns to the results page after identifying an unsuitable candidate. Third, the agent maintains task-relevant information across many steps, preserving constraints such as color, size, material, rating, seat count, and price while evaluating later pages. This persistent state tracking allows the agent to reject partially matching products and complete the task only after verifying that all required attributes are satisfied.

###### 1

###### 2

###### 3

4

|[Figure 8]<br><br>[Figure 9]|
|---|

|[Figure 10]<br><br>[Figure 11]|
|---|

|[Figure 12]<br><br>[Figure 13]|
|---|

|[Figure 14]<br><br>[Figure 15]|
|---|

Reasoning:The user wantstoadd the most top-selling Birkenstock's men clogs in brown, size

Reasoning: The user wantstoadd the most top-selling Birkenstock's men clogs in brown, size

Reasoning: The user wantstoadd the most top-selling Birkenstock's men clogs in brown, size

Reasoning: The user wantstoadd the most top-selling Birkenstock's men clogs in brown, size

10-10.5 to the cart. ... To find men's clogs

10-10.5 to the cart. ... I'm now on the men's

10-10.5 to the cart. ... I'm now on the men's clogs

10-10.5 to the cart. ... I'm now on the Boston

specifically, I should: 1. ... First navigate to the men's section 2. ... Select the correct size (10-10.5)

department page which shows different categories of men's footwear. ... I need to: 1. ... Click on "MEN'S CLOGS" to view clog-specific products ...

page that shows different clog collections ... To find the top-selling brown clogs, I need to: 1. ... Either click on one of these collections and then look for

Collection page which shows 159 products, but I can only see the top row of products and they appear to be a limited selection (at least three

5. ... Add to cart I see there's a "SHOP MEN'S" link

which would be a good starting point. ... I should

This will take me to a page that likely shows

"best sellers" sorting 2. ... Use the filter options to

visible). ... To find the actual best-selling items

click on the "SHOP MEN'S" link to navigate to the men's department page.

different clog models with filtering options such as sort by (best sellers), color filters, and size filters.

narrow down to brown clogs and then sort by best sellers The BOSTON clog appears to be a classic Birkenstock style… Action:{"name":"click","arguments": {"point_2d": [134,658]}}

(which would typically require sorting by best sellers), I should use the "SORT BY" dropdown that's visible on the page. Action:{"name":"click","arguments": {"point_2d": [903,410]}}

Action:{"name":"click","arguments":{"point_2d": [513,613]}}

Action:{"name":"click","arguments": {"point_2d": [308,530]}}

5

###### 6

###### 7

###### 8

|[Figure 16]<br><br>[Figure 17]|
|---|

|[Figure 18]<br><br>[Figure 19]|
|---|

|[Figure 20]<br><br>[Figure 21]|
|---|

|[Figure 22]<br><br>[Figure 23]|
|---|

Reasoning: The previousactionsuccessfully

Reasoning: I'mnowonthe product detail page for the Boston Soft Footbed in Suede Leather. ... To find the brown color option, I need to select the appropriate color first. ... Looking at the color thumbnails, I can see a light brown/tan color and a darker brown color available for selection. ... My

Reasoning: The previousattempttosort by

Reasoning: The previousaction successfully

clicked on the SORT BY dropdown, and now I can see the available sorting options including "Best Sellers". ... I should select "Best Sellers" from the

"Best Sellers" did not work because it seems the

changed the color to a brown variant, as I can now

see the product displaying a rich brown suede clog.

website doesn't have a "Best Sellers" option … I need to consider alternative approaches: 1. ... Use the FILTER button to apply filters that might

... The product has been filtered to a brown color option, so now I need to proceed with selecting the

dropdown to reorganize the product list so that the

top-selling items appear first. ... The next step is to click on the "Best Sellers" option in the dropdown menu that's currently visible.

size. ... Since the size information isn't fully visible

highlight popular items 2. ... Click on one of the

next steps should be: 1. ... Click on a brown color

and I can't see the actual size buttons/fields clearly, scrolling down slightly will help me better see the size selection area …

product types and then look for sorting or best seller tags 3. ... To check if this is a brown clog (as required in the user's request) and to verify its size

option (there appears to be a medium brown color in the second row of color thumbnails) …

availability, I should click on the product. ...

Action:{"name":"scroll","arguments": {"direction": "down","amount": 0.2,"point_2d": [750,500]}}

Action:{"name":"click","arguments": {"point_2d": [903,230]}}

Action:{"name":"click","arguments": {"point_2d": [823,553]}}

Action:{"name":"click","arguments":

{"point_2d": [185,670]}}

###### 11

###### 10

###### 12

###### 9

|[Figure 24]<br><br>[Figure 25]|
|---|

|[Figure 26]<br><br>[Figure 27]|
|---|

|[Figure 28]<br><br>[Figure 29]|
|---|

|[Figure 30]<br><br>[Figure 31]|
|---|

Reasoning: The size selectionhasbeen successfully updated to show "10-10.5” ... I've now found the specific product that matches what the user is looking for: - Birkenstock men's clogs Brown color (Mink brown) - Size 10-10.5 The next logical step is to add this item to the cart. ...

Reasoning: The scroll revealed the size selection area clearly. …Since the user is looking for men's clogs, I need to select a size from the

Reasoning: ...The productpage showsall the correct specifications that match what the user is

Reasoning: The scroll down actionhas revealed the size selection area clearly ... However,

looking for: - Birkenstock men's clogs - Brown color

I still don't see the "Add to Cart" button, which is

men's size chart. ... I should click on the "10-10.5"

(Mink brown) - Size 10-10.5 The item is displayed prominently in the center of the page, and the color and size selections are visible. ... However, I still don't see the "Add to Cart" button, which is typically located either near the product pricing or at the bottom of the product details section.

likely further down the page. ... The user's requirements have been met for product selection: …. highlighted in the size chart I've successfully navigated to the correct product ... The next logical step would be to scroll down further to locate the "Add to Cart" button.

size option under the Men's sizes. ... Once I select this size, the website should update to show that the item is available in this size ... The next action

However, I don't see a visible "Add to Cart" button

in the current view. ... It's likely further down the page or it may require a size/selection confirmation first. Action:{"name":"scroll","arguments":

should be to click on the "10-10.5" size option in

the Men's size chart. Action:{"name":"click","arguments": {"point_2d": [939,863]}}

Action:{"name":"scroll","arguments": {"direction": "down","amount": 0.35,"point_2d": [500,500]}}

Action:{"name":"scroll","arguments":

{"direction": "down","amount": 0.25,"point_2d":

{"direction": "up","amount": 0.3,"point_2d": [500,500]}}

[500,500]}}

###### 13

###### 14

|[Figure 32]<br><br>[Figure 33]|
|---|

|[Figure 34]<br><br>[Figure 35]<br><br>|
|---|

Reasoning: he scroll downaction has

Reasoning: The clickonthe "ADDTO CART"

successfully revealed the purchasing options. ... I can now see the "ADD TO CART" button clearly visible on the page, and the product is set to the

button was successful ... The right side of the screen displays a popup with details of the added item ... This confirms that the specific Birkenstock

correct size ... The "ADD TO CART" button is

men's clog in brown (Mink color), size 10-10.5, has

prominently displayed and ready to be clicked to add the item to the cart. ... I've now completed all the prior steps successfully: 1. ... Navigated to the

been added to the cart as requested. ... The task has been completed successfully. ... The user's requirement to "Add the most top-selling

Birkenstock website 2. ... Selected the brown color

Birkenstock's men clogs in brown, size 10-10.5 to

(Mink brown suede) …

my cart" has been fulfilled.

Action:{"name":"click","arguments":

Action:{"name":"done","arguments":

{"point_2d": [804,596]}}

{"response": <success_message>}}

- Figure 11 Trajectory Example: Add the most top-selling Birkenstock’s men clogs in brown, size 10-10.5 to my cart.

###### 1

###### 2

###### 3

4

|[Figure 36]<br><br>[Figure 37]|
|---|

|[Figure 38]<br><br>[Figure 39]|
|---|

|[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]|
|---|

|[Figure 43]<br><br>[Figure 44]|
|---|

Reasoning: The search bar isnowactive and

Reasoning:The currentpage showsthe IKEA global homepage with a cookie consent dialog. ... I need to accept cookies to remove this dialog or navigate to a specific country's IKEA website. ... I should first accept cookies to clear the dialog and have full access to the page. ... I'll click the

Reasoning: The cookie consentdialoghas been successfully removed, and now I have full access to the IKEA homepage. ... I can see the main page with a prominent yellow "Go shopping" button on the right side. ... To search for a specific item (a black leather sofa with at least 3 seats and

Reasoning: The previousactionsuccessfully navigated from the IKEA global homepage to the IKEA US/EN storefront ... I'm now on the main IKEA product page that shows various furniture categories and a large promotional image of a white sectional sofa. ... To efficiently search for a

displaying a dropdown of suggestions ... The cursor is ready for input in the search field ... I need to search for "sofa" to find all sofa products, after which I can apply filters for color (black), material

(leather), seating count (at least 3), and rating (4+

stars). ... Using the general term "sofa" will give me the most comprehensive list of sofa options to work with ... The next logical step is to type "sofa" into the active search bar. Action:{"name":"write","arguments": {"message": "sofa"}},{"name": "press_keys",

"Accept" button which is located at the bottom left

4+ stars), I should: 1. ... First navigate to the

specific product with filters, I should either: 1. ...

of the cookie consent dialog. ... Once I accept cookies, the dialog should disappear … Action:{"name":"click","arguments": {"point_2d": [84,944]}}

shopping section by clicking the "Go shopping" button ... it's likely to take me to a page with a search bar … Action:{"name":"click","arguments":{"point_2d": [800,445]}}

Use the search bar to directly search for "sofa" and then apply filters 2. ...

Action:{"name":"click","arguments": {"point_2d": [663,93]}}

"arguments": {"keys": ["enter"]}}

5

###### 6

###### 7

###### 8

|[Figure 45]<br><br>[Figure 46]|
|---|

|[Figure 47]<br><br>[Figure 48]|
|---|

|[Figure 49]<br><br>[Figure 50]|
|---|

|[Figure 51]<br><br>[Figure 52]|
|---|

Reasoning:The previousaction successfully opened the Color filter section, and now I can see

Reasoning: The search for "sofa" has successfully returned 751 results ... I'm now in the product listing section of IKEA where I can begin applying filters to narrow down to the specific criteria . ... Looking at the left sidebar, I can see

Reasoning: The previousaction successfully applied the 3-seat filter, narrowing down the search results from 751 sofas to 86 sofas with exactly

Reasoning: The previousactionsuccessfully opened the "Number of seats" filter section, but it

various color options with their corresponding sofa

seems the checkboxes next to "3 seats" have not

models. ... This is exactly what I needed ... I need to select the black color option to filter the results to only black sofas. ... This will help me meet the user's requirement of "black" while also reducing

three seating positions. ... I need to continue

been selected ... I need to select the checkbox next to "3 seats" to filter the results accordingly. ... Once this filter is applied, the page should refresh to show only 3-seat sofas, which will make it easier to

filtering to find the cheapest black leather sofa with at least four stars. ... The next steps should be: 1. ... Look for a way to filter by color (black) 2. ... Look for a way to filter by material/finish (leather) 3. ... Look

several potential filtering options that will help me

narrow down the results … I'll click on the \"Number of seats\" filter to apply a minimum of 3 seats, then continue narrowing down with color and material filters.", Action:{"name":"click","arguments":

the number of options so I can more easily apply

then apply other filters such as color (black),

additional filters for leather material and customer rating.

for a way to filter by customer rating (stars) … Action:{"name":"click","arguments":

material (leather), and customer rating (4+ stars). Action:{"name":"click","arguments": {"point_2d": [207,607]}}

Action:{"name":"click","arguments":{"point_2d": [87,708]}}

{"point_2d": [127,539]}}

{"point_2d": [127,494]}}

###### 10

###### 12

###### 11

###### 9

|[Figure 53]<br><br>[Figure 54]|
|---|

|[Figure 55]<br><br>[Figure 56]|
|---|

|[Figure 57]<br><br>[Figure 58]|
|---|

|[Figure 59]<br><br>[Figure 60]|
|---|

Reasoning :The hover actionover the KIVIK Sofa … didn't surface any customer star ratings as previously hoped. ... The MANNARP at $429 might be a candidate if it has a leather-like finish and sufficient stars, but I need to verify. ... Given the

Reasoning The previousaction successfully

Reasoning: I've successfullynavigated to the

Reasoning: I'mbackonthe search results

applied the black color filter ... I now need to

MANNARP 3-seat sofa product page, but it seems this is not a leather finish sofa despite meeting some of our criteria. ... I'm now on a product-

page with 40 black 3-seat sofas ... I need to continue looking for a leather finish. ... I need to use the search filters more effectively. ... More importantly, there's a \"Customer rating\" filter that

continue filtering to find the specific type: a black leather sofa with at least four stars. ... The next steps should be: 1. ... Apply a leather material filter

specific page rather than search results, which

current state, my next action should be to click on

2. ... Look for a way to sort by customer rating … I

means I need to go back to continue shopping for a different sofa that better matches all criteria (black, leather, 3+ seats, 4+ stars). ... I should return to the search results where I can select other sofas …

will help me find sofas with at least four stars. … I

the MANNRAP product to verify if it meets our criteria (black, 3 seats, leather finish, at least 4 stars, and lowest price).

should hover over the product images or names to

expect that after clicking on the Customer rating filter, I'll be able to see a more detailed rating distribution across the product listings …

see if detailed star ratings become visible …

Action:{"name":"goto_url","arguments": {"url": "https://www.ikea.com/us/en/search/?q=sofa&filte

Action:{"name": "click","arguments": {"point_2d": [127,684]}}

Action:{"name":"click","arguments":{"point_2d": [870,727]}}

Action:{"name":"hover","arguments": {"point_2d": [344,584]}}

rs=f-colors%3A10139%2Cf-seats%3A38768"}}

###### 15

13

###### 14

###### 16

|[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]|
|---|

|[Figure 64]<br><br>[Figure 65]|
|---|

|[Figure 66]<br><br>[Figure 67]|
|---|

|[Figure 68]<br><br>[Figure 69]|
|---|

Reasoning: The previousactionsuccessfully

Reasoning: The previousactionsuccessfully

Reasoning: The hover over the Price filter didn't reveal any additional options ... I still need to identify the cheapest option that has a leather

Reasoning: I'mnowonthe KIVIK Sofa product page, but this appears to be a white/black version of the sofa rather than a black sofa. ... This product

applied both the 4-star and 5-star customer rating filters ... now I can see products that better match the user's criteria in terms of ratings. ... The KIVIK

opened the Customer rating filter section, allowing me to see the distribution of rated sofas across different star categories. ... I should click on the checkbox next to the 4-star and 5-star ratings to

finish. ... I can't determine which one is the

doesn't match all the criteria - specifically, it's not

Sofa at $839.00 appears to be the most affordable

cheapest leather sofa as the results aren't sorted by price. … I should click on the KIVIK Sofa that costs $839.00 (Tallmyra white/black) to verify its material composition and ensure it fully meets all

black and likely doesn't have a leather finish (it appears to be a fabric cover based on the product type and images). ... I need to go back to the search results to continue looking for a black leather 3-

option visible, but I need to confirm it has a leather finish. … I should click on the KIVIK Sofa that costs $839.00 (Tallmyra white/black) to verify its material composition and ensure it fully meets all the user's

select products with at least four stars. This will

further narrow down the search results to only high-rated sofas, bringing me closer to meeting all the user's requirements simultaneously.

the user's criteria … Action:{"name":"click","arguments":

seat sofa that meets all criteria. ... Action:{"name":"goto_url","arguments": {"url":

criteria …

Action:{"name":"click","arguments": {"point_2d": [207,805]}},{"name": "click", "arguments": {"point_2d": [207,751]}}

Action:{"name":"hover","arguments": {"point_2d”: [134,439]}}

"https://www.ikea.com/us/en/search/?q=sofa&filte rs=f-colors%3A10139%2Cf- ratings%3A4%7C5%2 Cf-seats%3A38768"}}

{"point_2d": [521,740]}}

17 19

###### 18

|[Figure 70]<br><br>[Figure 71]|
|---|

|[Figure 72]<br><br>[Figure 73]|
|---|

|[Figure 74]<br><br>[Figure 75]|
|---|

Reasoning: I've found a product that matches all the user's criteria perfectly! ... The FRIHETEN

Reasoning: I'mbackonthe search results page ... On the far right, I can see the FRIHETEN Sleeper sofa which is black and has a 4-star rating

Reasoning: The actionwas successful -the FRIHETEN Sleeper sofa has been added to the shopping bag as confirmed by the popup message on the right side of the screen. ... The popup clearly states "FRIHETEN was added to your shopping bag" along with the price of $999.00 ... This verifies that

Sleeper sofa is a black sofa with a leather finish (it

shows as "leather" in the product type), has at least three seats, and has an impressive 4-star rating ... I've successfully identified the sofa that matches all the user's requirements. ... The next logical step

... Looking more carefully at the other options, I can

see that most of these appear to be standard black sofas. ... The next action should be to click on the FRIHETEN Sleeper sofa product to verify its specifications, particularly the material

the cheapest black leather sofa with at least three

is to add this sofa to the cart so the user can

seats and a leather finish that meets the four-star requirement has now been added to the cart. ...

proceed with purchase if they wish. ... Action:{"name":"click","arguments": {"point_2d": [839,801]}}

composition, to determine if it qualifies as a true

leather sofa.",

Action:{"name":"done","arguments": {"response": <success_message>}}

Action:{"name":"click","arguments":

{"point_2d": [870,727]}}

37

- Figure 12 Trajectory Example: Add the cheapest black sofa with at least three seats, a leather finish, and at least four stars to my cart.

