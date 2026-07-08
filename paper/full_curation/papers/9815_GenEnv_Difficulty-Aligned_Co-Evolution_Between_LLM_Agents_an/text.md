# arXiv:2512.19682v2[cs.CL]23Dec2025

[Figure 1]

2025-12-24

## GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators

##### Jiacheng Guo1*, Ling Yang1*†, Peter Chen2*, Qixin Xiao3*, Yinjie Wang4, Xinzhe Juan3, Jiahao Qiu1, Ke Shen, Mengdi Wang1†

1Princeton University 2Columbia University 3University of Michigan 4University of Chicago *Equal Contribution †Corresponding Authors

Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a Data-Evolving Paradigm: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent’s “zone of proximal development”. This process is guided by a simple but effective 𝛼-Curriculum Reward, which aligns task difficulty with the agent’s current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to +40.3% over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3× less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities. Our codes are available at https://github.com/Gen-Verse/GenEnv.

GenEnv (7B) vs. Baselines

BFCL

Higher Accuracy with Less Data

###### (a) (b)

48%

+2.0 vs Gemini 3.3× using 3.3× less data

API Bank

+34.8

45.8%

46%

+17.5

FinalValidationScore

###### 43.8%

43.2% 43.4%

44%

ALFWorld

+40.3

42%

40.8%

40%

+8.0

38%

+2.3

Bamboogle

36%

(1.8× data)

(3.3× data)

RandomEnv Static Aug Gemini 1.8× Gemini 3.3× GenEnv (Ours)

TravelPlanner

GPT-OSS 20B

ReSearch SearchR1

Qwen 2.5 7B

GenEnv (Ours)

Qwen 3 14B

ToRL

Figure 1 | GenEnv’s cross-benchmark gains and data efficiency. (a) We compare GenEnv (7B) against representative baselines (Qwen2.5-7B, ReSearch, SearchR1, ToRL) and larger open models (e.g., Qwen3-14B, GPT-OSS-20B). Blue callouts report the absolute improvement of GenEnv over Qwen2.5-7B on each benchmark. (b) Validation data-efficiency comparison on BFCL: GenEnv surpasses RandomEnv and Static Augmentation, and outperforms Gemini-based offline augmentation even with 3.3× more synthetic data. Together, the figure shows that difficulty-aligned adaptive simulation can outperform stronger static augmentation baselines under comparable training settings.

TRADITIONAL “MODEL-EVOLVING ON STATIC DATA”

Trained Agent LLM

Static Training (Imitation)

High-Cost

Real-World Environment

###### Static Expert Dataset

Inefficient Training & Mismatched Difficulty

Interaction

(𝛑𝐀𝐠𝐞𝐧𝐭)

[Figure 2]

[Figure 3]

[Figure 4]

|High Data Cost Static Curriculum Poor Generalization<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]|
|---|

GENENV CO-EVOLUTIONARY LEARNING

Expert Steps

| |Environment LLM<br><br>(𝛑Env)| |
|---|---|---|
| |[Figure 8]| |

Agent LLM (𝛑𝐀𝐠𝐞𝐧𝐭)

###### Generate Adaptive Tasks

Updates

Evolved Agent

|𝐑𝐀𝐠𝐞𝐧𝐭 → 𝐔𝐩𝐝𝐚𝐭𝐞 𝛉𝐀𝐠𝐞𝐧𝐭|
|---|

[Figure 9]

[Figure 10]

&

[Figure 11]

|𝐑𝐄𝐧𝐯 → 𝐔𝐩𝐝𝐚𝐭𝐞 𝛉𝐄𝐧𝐯|
|---|

Simulator

Next Iteration

|Low-Cost Simulation Adaptive Curriculum Improved Efficiency<br><br>[Figure 12]<br><br>[Figure 13]<br><br>[Figure 14]|
|---|

- Figure 2 | A comparison between the traditional training paradigm and our proposed GenEnv framework. The traditional approach (top) relies on high-cost interaction with the real world to create a static dataset, leading to inefficient training and poor generalization. GenEnv (bottom) creates a co-evolutionary loop where an Environment LLM generates adaptive tasks for the Agent LLM, enabling low-cost simulation, an adaptive curriculum, and improved efficiency.

### 1. Introduction

Training capable Large Language Model (LLM) agents for complex, interactive tasks like web navigation or tool use is constrained by a significant bottleneck: the high cost of data collection through real-world interaction (Ning et al., 2025; Shinn et al., 2023; Wang et al., 2025a,b; Yao et al., 2023). Each step an agent takes in a real-world environment can be slow, expensive, and difficult to parallelize. For instance, a web agent that navigates an e-commerce site may fail when a button’s label changes from “Add to Cart” to “Add to Basket” (Gur et al., 2023), but discovering such failure modes requires extensive and costly real-world exploration. This fragility highlights a key limitation in how these agents are commonly trained.

A central driver of this issue is the reliance on static, pre-collected datasets of expert trajectories (Levine et al., 2020; Pomerleau, 1991; Samadi et al., 2024). Such datasets, no matter how large, represent a fixed snapshot of the world and struggle to capture the wide range of variations an agent will encounter in open-ended environments (Levine et al., 2020). Increasing the dataset size alone does not resolve this limitation: the bottleneck often lies not just in data volume, but in the static and costly nature of its collection and its inability to adapt as the agent improves.

The challenge of insufficient and static data has led to significant interest in synthetic data generation. However, despite progress, these methods often produce a large but ultimately static corpus that can fail to adapt to the agent’s evolving requirements (Ding et al., 2024; Ye et al., 2024). This can result in inefficient training that still does not effectively target the agent’s specific weaknesses. The high cost of interaction and data collection remains a core problem.

To address this, we approach the problem differently by proposing GenEnv, a framework built on

leveraging an LLM as a scalable environment simulator to reduce interaction costs. Instead of relying on slow and expensive real-world feedback, our framework trains the agent almost entirely within a simulated environment that can generate diverse and relevant training scenarios at a substantially lower cost. As illustrated in Figure 2, this contrasts with traditional methods that evolve a model on static data. In our approach, a generative environment model is trained to produce an adaptive curriculum of tasks, creating challenges tailored to the agent’s performance. This leads to our primary research question: Can an LLM-based environment simulator provide a scalable, low-cost alternative to real-world interaction for effectively training capable agents?

In this simulation-centric process, the agent learns to overcome challenges generated by the simulator. The agent’s performance provides a natural reward signal that guides the simulator’s curriculum generation, allowing both to improve in a self-contained training loop. Throughout the paper, we use “agent” and Agent Policy 𝜋agent interchangeably, and “environment simulator” and Environment Policy 𝜋env interchangeably. As previewed in Figure 1, GenEnv delivers consistent gains over strong 7B baselines across five benchmarks and achieves higher accuracy than Gemini-based offline augmentation while using less synthetic data, highlighting the advantage of difficulty-aligned, adaptive simulation over static scaling of data. Our contributions are:

- • The Data-Evolving Paradigm: We propose a co-evolutionary framework where the training data distribution adapts dynamically to the agent’s learning progress, breaking the reliance on static corpora.
- • Difficulty-Aligned Simulation: We introduce the 𝛼-Curriculum Reward, a mechanism that rewards the simulator for generating tasks within the agent’s target success zone (akin to the “zone of proximal development” (Vygotsky, 1978)), ensuring an efficient automated curriculum.
- • Data Efficiency: On our benchmarks, GenEnv matches or surpasses Gemini 2.5 Pro-based static augmentation pipelines while using 3.3× less synthetic data, suggesting that an adaptive simulator can be more valuable than simply scaling the teacher model.

### 2. GenEnv: Difficulty-Aligned Co-Evolution

GenEnv views agent training as a two-player curriculum game rather than a single-player optimization problem. We maintain two policies: an Agent Policy 𝜋agent (the agent) and an Environment Policy 𝜋env (the environment simulator). Unlike standard RL where the environment is fixed, GenEnv enables both to co-evolve:

- • 𝜋agent learns to solve tasks sampled from the current simulator.
- • 𝜋env is rewarded for generating tasks whose difficulty is aligned with the agent’s current capability—targeting the “zone of proximal development” where learning is most effective (Vygotsky, 1978).

###### 2.1. Data-Evolving Paradigm: From Static Corpora to Adaptive Simulation

Standard training minimizes a loss L(𝜃) over a static distribution Dstatic, where 𝜃 denotes the parameters of the agent. In contrast, GenEnv implements a Data-Evolving Paradigm. The training data D𝑡 is generated on-the-fly by 𝜋env, conditioned on the agent’s historical performance. This creates a feedback loop (Figure 3) where the simulator seeks not to defeat the agent, but to find its “breaking points” to facilitate learning.

GENENV TRAINING LOOP (ALGORITHM 1) Iterative Process

- STAGE 1: Data Generation & Agent Execution
- STAGE 2: Reward Calculation

[Figure 15]

Generated Tasks

Decompose

[Figure 16]

Expert Trajectories

[Figure 17]

Trajectories into

(s, a) steps

[Figure 18]

[Figure 19]

Environment LLM

Agent LLM

STAGE 3: Model Updates

Update 𝝅agent

[Figure 20]

Calculate Agent Rewards

|GRPO: maximize E[𝐑agent] on 𝐄𝐭|
|---|

[Figure 21]

𝐑agent (Compare a’ and a) → Metric: Exact Match / sim(a′,a)

Update 𝝅env

Calculate Environment Rewards

[Figure 22]

[Figure 23]

|RWR: weight ∝ exp(𝝀 · 𝐑𝛂(𝐩)ෝ )|
|---|

𝐑𝛂(𝐩)ෝ (Success Rate 𝒑ෝ vs Target 𝛂) → Difficulty Alignment

- Figure 3 | The GenEnv Co-Evolutionary Loop. (1) The Environment Policy generates tasks. (2) The Agent Policy attempts them. (3) The environment reward (difficulty alignment) updates the simulator, while the agent reward (task success) updates the agent.

- 2.2. Rewards: Agent vs. Environment (with explicit equation references)

- 2.2.1. Agent Task Reward (𝑅agent)

Each environment-generated task induces a target action/trajectory 𝑎 (e.g., a sequence of tool calls or a final answer), and the agent produces a prediction 𝑎′. We distinguish a structured action space Astruct (e.g., executable API calls) from free-form answers (e.g., natural language). For structured actions we can rely on exact execution; for unstructured ones we use a soft similarity score. We define the agent reward (used to update 𝜋agent) as:

𝑅agent(𝑎′, 𝑎) = 𝕀(𝑎′ = 𝑎) · 𝕀(𝑎 ∈ Astruct) + sim(𝑎′, 𝑎) · 𝕀(𝑎 ∉ Astruct), (1)

where sim(𝑎′, 𝑎) ∈ [0, 1] is task-dependent (e.g., normalized token-F1 or embedding similarity). In all benchmarks we scale 𝑅agent into [0, 1].

- 2.2.2. Environment Difficulty-Alignment Reward (𝑅env)

The core innovation for 𝜋env is a difficulty-aligned reward that targets a success-rate band around a desired 𝛼 ∈ (0, 1) (we use 𝛼 = 0.5). For each generated batch of 𝑛 task variations, after the agent attempts them we compute the empirical success rate:

𝑘 𝑛

ˆ𝑝 =

, (2)

where 𝑘 is the number of successes under 𝑅agent (Eq. equation 1). We then assign the environment reward (used to update 𝜋env):

𝑅env(ˆ𝑝) = exp −𝛽(ˆ𝑝 − 𝛼)2 , (3)

where 𝛽 > 0 controls sharpness. This bell-shaped reward peaks when the agent’s performance matches 𝛼, discouraging tasks that are already mastered (ˆ𝑝 → 1) or hopeless (ˆ𝑝 → 0). We additionally apply

a difficulty filter: task batches with |ˆ𝑝 − 𝛼| > 𝑘min (we use 𝑘min = 0.1) are excluded from environment updates to prevent overfitting to transient spikes.

###### 2.3. Data Structures and How New Training Data Is Produced

A recurring confusion in co-evolution papers is where the training data actually comes from. We therefore make the data flow explicit.

What the environment generates. At epoch 𝑡, the environment policy 𝜋env generates a task batch T𝑡. Concretely, T𝑡 contains 𝑛 task instances (often multiple variations of the same seed), where each instance includes: (i) a task prompt/context (including tool specs, constraints, and goal), (ii) an evaluation specification (e.g., executable checker / exact-match target / reference answer), and optionally (iii) structured “ground truth” target action 𝑎 when the benchmark provides it (e.g., tool-call arguments).

What the agent produces. The agent interacts with each task instance and yields an interaction trace (rollout) that we denote by an element 𝑒 ∈ E𝑡. Each trace records at minimum:

𝑒 = (task, trajectory, 𝑎′, 𝑟),

where trajectory can include intermediate reasoning text and tool calls, 𝑎′ is the final output/action, and 𝑟 = 𝑅agent(𝑎′, 𝑎) is computed via Eq. equation 1 (or its benchmark-specific instantiation).

Two growing datasets: one for the agent, one for the environment. We maintain two pools that grow online:

- • Agent training pool Dtrain: stores valid interaction traces from E𝑡. A trace is “valid” if it is well-formed and evaluable for the benchmark (e.g., tool calls parse and execute; outputs follow

required schema; checker runs without error). We append these valid tuples to Dtrain so the agent can (i) learn from fresh on-policy experiences and (ii) retain mastery of earlier curricula by continuing to sample from the accumulated pool.

- • Environment SFT pool Denv: stores environment generations used to train 𝜋env via RWR. Each record is a supervised pair of the form

(env-conditioning context → generated task instance),

where the conditioning context includes the seed prompt plus any summary signals (e.g., recent success statistics) that 𝜋env conditions on. We weight each record by a monotone function of the environment reward in Eq. equation 3 (e.g., ∝ exp(𝜆𝑅env(ˆ𝑝)), where 𝜆 = 1.0 is a temperature hyperparameter).

How this produces a “data-evolving” training set. At every epoch, both T𝑡 and E𝑡 are newly generated; thus Dtrain is not a fixed offline corpus but an evolving mixture of (a) base data, (b) previously collected valid traces, and (c) newly collected on-policy traces. Meanwhile, Denv evolves toward generating tasks whose empirical success rate stays near 𝛼 (Eq. equation 3), which in turn shifts the difficulty distribution of future T𝑡+1. This closes the loop: new data is produced as a byproduct of interaction, and is then explicitly aggregated into training pools for subsequent updates.

###### 2.4. Two-Player Curriculum RL: Optimization Loop

- Player 1 Update (Agent). The Agent Policy 𝜋agent is updated to maximize 𝔼[𝑅agent] (Eq. equation 1). In our experiments, we instantiate this using Group Relative Policy Optimization (GRPO) (Shao et al., 2024).
- Player 2 Update (Environment). The Environment Policy 𝜋env is updated to maximize 𝔼[𝑅env] (Eq. equation 3). We implement this via Reward-Weighted Regression (RWR): from each environmentgenerated batch, we compute ˆ𝑝 (Eq. equation 2) from agent rollouts and assign 𝑅env(ˆ𝑝). We then construct a weighted SFT set of environment generations and fine-tune 𝜋env toward higher-reward generations. For stability, we regularize updates with a KL penalty to the initial simulator and cap per-step updates by a maximum KL threshold. Algorithm 1 GenEnv Co-Evolutionary Loop (with explicit equation references)

Initialize: Agent 𝜋agent, Environment 𝜋env, Agent pool Dtrain, Env pool Denv. for epoch 𝑡 = 1, . . . ,𝑇 do

▷ Phase 1: Online Generation & Interaction Environment generates a task batch T𝑡 ∼ 𝜋env(·). Agent 𝜋agent rolls out on T𝑡 to obtain traces E𝑡 and per-trajectory agent rewards 𝑅agent via

equation 1. Compute batch success ˆ𝑝 via equation 2 and assign environment reward 𝑅env(ˆ𝑝) via equation 3.

- ▷ Phase 2: Dual Update (Two Players, Two Objectives) Update agent 𝜋agent via GRPO to maximize 𝔼[𝑅agent] (equation 1). Filter out batches with |ˆ𝑝 − 𝛼| > 𝑘min for environment updates. Build weighted env SFT set D˜env𝑡 from T𝑡 with weights ∝ exp(𝜆𝑅env(ˆ𝑝)) (equation 3). Update environment 𝜋env via RWR on D˜env𝑡 to maximize 𝔼[𝑅env] (equation 3).
- ▷ Phase 3: Aggregation (How New Data Enters Training)

Extract valid traces from E𝑡 (e.g., parseable/executable/checker-passed) and append to agent pool:

Dtrain ← Dtrain ∪ Valid(E𝑡).

Append weighted environment generations to env pool:

Denv ← Denv ∪ D˜env𝑡 . end for

- 3. Theoretical Analysis of Difficulty-Aligned GenEnv

In this section we provide a simple theoretical analysis of the difficulty-aligned co-evolution mechanism in GenEnv. Our goal is not to fully characterize the dynamics of large LLMs, but to clarify why (1) tasks whose success rate is close to the target band 𝛼 carry the strongest learning signal for the Agent Policy 𝜋agent, and (2) the 𝛼-Curriculum Reward 𝑅env provides a statistically consistent signal for the Environment Policy 𝜋env to rank task types by how well their difficulty matches the current agent.

###### 3.1. Intermediate Difficulty Maximizes Agent Learning Signal

We first consider a stylized bandit setting in which the Agent Policy 𝜋agent interacts with a single environment-generated task type 𝜏 (e.g., a family of API-calling problems of similar difficulty). The outcome of each attempt is a scalar reward 𝑟 ∈ {0, 1}, where 𝑟 = 1 denotes success on the task and 𝑟 = 0 denotes failure.1 For a fixed Agent Policy parameterization 𝜃, let 𝑝(𝜏) = Pr(𝑟 = 1 | 𝜏, 𝜃) denote

1The analysis extends to 𝑅agent ∈ [0, 1] by rescaling; we use the binary case for clarity.

the success probability on task type 𝜏. The Agent Policy is updated with a REINFORCE-style estimator 𝑔(𝜏, 𝑟) = (𝑟 − 𝑏(𝜏))∇𝜃 log𝜋𝜃(𝑎 | 𝜏), (4)

where 𝑎 is the sampled action (e.g., a rollout trajectory of tool-calling sequence) and 𝑏(𝜏) is a baseline (e.g., an estimate of the expected reward on 𝜏). The quantity 𝔼[∥𝑔(𝜏, 𝑟)∥2] can be viewed as measuring how strong the stochastic gradient signal is for this task type.

Given the trust-region KL constraint and the gradient-clipping bias in GRPO-style policy updates, it is expected that the squared norm of the score function remains within a trust-region bound. This behavior has been discussed in recent analyses of one-step policy updates (e.g., Chen et al. (2025c, Theorem 3.2)) as well as in the literature on trust-region policy optimization (Schulman et al., 2015). Accordingly, we establish our theoretical analysis under the reasonable assumption that the squared norm of the score function ∇𝜃 log𝜋𝜃(𝑎 | 𝜏) does not vary too dramatically when conditioned on the binary outcome 𝑟.

Assumption 1 (Bounded score variation). For a fixed task type 𝜏, there exist constants 0 < 𝑐min ≤ 𝑐max < ∞ such that

𝑐min ≤ 𝔼 ∇𝜃 log𝜋𝜃(𝑎 | 𝜏) 2 𝑟 ≤ 𝑐max for both 𝑟 = 0 and 𝑟 = 1.

We take the baseline to be the on-task expected reward, 𝑏(𝜏) = 𝔼[𝑟 | 𝜏] = 𝑝(𝜏), which is the variance minimizer in the standard REINFORCE analysis. Under these conditions we obtain the following result.

Proposition 1 (Intermediate difficulty maximizes gradient signal). Suppose Assumption 1 holds and the baseline is chosen as 𝑏(𝜏) = 𝑝(𝜏). Then there exist positive constants 𝐶min and 𝐶max, independent of 𝑝(𝜏), such that

𝐶min 𝑝(𝜏) 1 − 𝑝(𝜏) ≤ 𝔼 ∥𝑔(𝜏, 𝑟)∥2 ≤ 𝐶max 𝑝(𝜏) 1 − 𝑝(𝜏) . (5)

In particular, up to constant factors, the expected squared gradient norm is proportional to 𝑝(𝜏) 1− 𝑝(𝜏) , which is maximized when 𝑝(𝜏) = 1/2, i.e., for tasks of intermediate difficulty.

Proof sketch. With 𝑟 ∈ {0, 1} and 𝑏(𝜏) = 𝑝(𝜏), we have 𝔼[(𝑟 − 𝑝(𝜏))2 | 𝜏] = Var(𝑟 | 𝜏) := 𝑝(𝜏)(1− 𝑝(𝜏)). Using the law of total expectation and Assumption 1, we can factor out the variation coming from the score function up to multiplicative constants, which yields equation 5. The function 𝑝(1−𝑝) is a concave quadratic on [0, 1] with a unique maximum at 𝑝 = 1/2. A full proof is given in Appendix A.1. □

Remark 1 (12-Curriculum reward). Considering the case that 𝛼 = 12, we have the identity 𝑝(1 − 𝑝) = 1

2. Thus, maximizing the variance term 𝑝(1 − 𝑝) is exactly equivalent to minimizing the

- 4 − 𝑝 − 12

squared distance to the target success rate 𝛼 = 12. In GenEnv, the Environment Policy does not observe the true success probability 𝑝(𝜏) but only an empirical estimate ˆ𝑝(𝜏) from a finite number of rollouts.

The 𝛼-Curriculum Reward takes the form 𝑅env(ˆ𝑝(𝜏)) = exp − 𝛽(ˆ𝑝(𝜏) − 𝛼)2 , (6)

which is a monotone transformation of −(ˆ𝑝(𝜏) − 𝛼)2 and therefore encourages the simulator to propose tasks whose empirical success rate stays close to 𝛼. Proposition 1 then suggests that, in expectation, this aligns the simulator with task types that provide the strongest learning signal for 𝜋agent.

- 3.2. Ranking Consistency of the 𝛼-Curriculum Reward

We next show that, despite relying on noisy empirical success rates, the 𝛼-Curriculum Reward provides a statistically consistent signal for ranking task types by how well their difficulty matches the target band. The argument is based on standard concentration inequalities.

Consider two task types 𝜏1 and 𝜏2. For a fixed Agent Policy 𝜋agent, let 𝑝𝑖 = 𝑝(𝜏𝑖) denote the true success probability on 𝜏𝑖, and define their distances to the target band 𝛼 as

Δ𝑖 = |𝑝𝑖 − 𝛼|, 𝑖 ∈ {1, 2}. (7)

Without loss of generality, assume Δ1 < Δ2, i.e., 𝜏1 is closer to the target difficulty than 𝜏2. For each 𝜏𝑖 we run 𝑛𝑖 independent rollouts and compute the empirical success rate ˆ𝑝𝑖 = 𝑘𝑖/𝑛𝑖, where 𝑘𝑖 is the number of successes. The Environment Policy receives the reward

𝑅env(ˆ𝑝𝑖) = exp − 𝛽(ˆ𝑝𝑖 − 𝛼)2 . (8)

Since the exponential is monotone, ranking tasks by 𝑅env is equivalent to ranking them by their squared distance (ˆ𝑝𝑖 − 𝛼)2 to the target band.

The following theorem shows that the mis-ranking probability decays exponentially in the minimum number of rollouts. Theorem 1 (Ranking consistency of 𝑅env). Let 𝑛 = min{𝑛1, 𝑛2} and Δ1 < Δ2 as above. Define 𝛿 = (Δ2 − Δ1)/3 > 0. Then

Pr 𝑅env(ˆ𝑝1) ≤ 𝑅env(ˆ𝑝2) ≤ 4exp −

2 9(Δ2 − Δ1)2 𝑛 . (9) In particular, as 𝑛 → ∞ the reward ranking is consistent: tasks whose true success probability lies closer to the target band 𝛼 receive higher 𝛼-Curriculum Reward with probability approaching 1 at an exponential rate.

Proof sketch. Because the exponential is monotone, 𝑅env(ˆ𝑝1) ≤ 𝑅env(ˆ𝑝2) is equivalent to |ˆ𝑝1 − 𝛼| ≥ |ˆ𝑝2−𝛼|. We show that if both empirical estimates ˆ𝑝𝑖 lie within 𝛿 of their true means 𝑝𝑖, then necessarily |ˆ𝑝1 −𝛼| < |ˆ𝑝2 −𝛼| and hence 𝑅env(ˆ𝑝1) > 𝑅env(ˆ𝑝2). Thus a mis-ranking can only occur when at least one empirical mean deviates from its expectation by more than 𝛿, which can be bounded using Hoeffding’s inequality for Bernoulli random variables. A detailed proof is provided in Appendix A.1. □

Implications for GenEnv. Theorem 1 shows that, even though the Environment Policy only observes noisy empirical success rates ˆ𝑝𝑖 derived from a finite number of rollouts, the 𝛼-Curriculum Reward is a statistically consistent proxy for task difficulty. As we increase the rollout budget per task type, the environment LLM can more reliably identify and up-weight task families whose difficulty lies in the target zone of proximal development. This provides a formal justification for the empirical convergence behaviour observed in Figure 7, where the agent’s success rate on simulated tasks concentrates around a band centered at 𝛼 = 0.5. Together, Proposition 1 and Theorem 1 clarify why aligning the simulator with an intermediate success rate both maximizes learning signal for the agent and yields a stable, difficulty-calibrated curriculum.

- 4. Experiments

- 4.1. Experimental Setup

Backbone Models. Unless otherwise specified, all 7B agents and simulators are initialized from Qwen2.5-7B-Instruct (Yang and Qwen Team, 2024). For large-scale baselines in Table 1, we include

Table 1 | Main Results. Comparison on five benchmarks. Models are grouped by size: Large Scale Models (> 10B) are sorted by size descending, followed by 7B Models. Bold numbers indicate the best performance within each group. GenEnv (7B) significantly outperforms other 7B baselines and even surpasses the average performance of several 72B/405B models on this suite.

Model ALFWorld BFCL API-Bank Bamboogle TravelPlanner Average Large Scale Models

Llama 3.1 405B 65.3 5.5 74.4 77.6 16.5 47.9 GPT-OSS 120B 60.4 21.9 53.6 29.6 14.7 36.0 Qwen 2.5 72B 63.5 35.3 54.9 69.6 20.5 48.8 Llama 3.1 70B 60.1 13.4 64.3 76.8 17.6 46.4 Qwen 3 32B 52.3 33.8 63.8 71.2 22.5 48.7 GPT-OSS 20B 53.6 24.4 41.2 33.6 14.9 33.5 Qwen 3 14B 37.8 29.4 66.7 76.0 14.7 44.9

7B Models

ReSearch 18.7 5.0 65.3 68.0 16.4 34.7 SearchR1 16.1 5.0 63.3 67.2 16.1 33.5 Qwen 2.5 7B 14.2 7.0 61.6 68.0 14.3 33.0 ToRL 8.0 0.0 54.1 34.4 14.8 22.3 GenEnv (Ours) 54.5 41.8 79.1 76.0 16.6 53.6

Llama 3.1 models (Grattafiori et al., 2024), GPT-OSS open-weight models (OpenAI, 2025), and Qwen3 models (Yang and Qwen Team, 2025). These models are evaluated using the same tool-calling interface and prompt templates as our 7B baselines for fairness.

Benchmarks. We evaluate across 5 diverse benchmarks that span tool use, embodied interaction, and real-world planning. API-Bank (Li et al., 2023) measures function-calling and tool-augmented reasoning, and Bamboogle is a compositional multi-hop QA benchmark built on top of the framework from Press et al. (2023); for both, we follow the evaluation protocols from ToRL (Li et al., 2025b). ALFWorld (Shridhar et al., 2021) aligns textual instructions with embodied environments; we utilize the official validation set, where multi-turn tasks are decomposed into single steps for evaluation. BFCL follows the Berkeley Function-Calling Leaderboard setup (Patil et al., 2025); specifically, we evaluate on the long-context subset treating each turn independently. TravelPlanner (Xie et al.,

- 2024) captures end-to-end planning and tool use in realistic travel scenarios; we report the average of four metrics: CS Micro (%), CS Macro (%), HD Micro (%), and HD Macro (%). The Average column in Table 1 is the unweighted mean of these per-benchmark success metrics.

Baselines & Variants. We compare against standard instructed models (Qwen2.5-7B-Instruct) and specialized search-and-planning agents, including ReSearch, SearchR1, and ToRL (Jin et al., 2025; Li et al., 2025b). These methods represent strong model-evolving pipelines that either improve search policies or alignment rewards on largely static datasets. To strictly evaluate our Data-Evolving contribution, we define:

- • GenEnv-Random: The simulator generates new tasks every epoch but is not trained via 𝑅env. This isolates the effect of dynamic data vs. aligned curriculum.
- • GenEnv-Static: The simulator generates a large batch of synthetic data once before training.
- • Gemini-Offline (2x / 3.3x): High-quality synthetic data generated offline by Gemini 2.5 Pro (approx. 1.76x and 3.27x the training set size). This represents a strong “teacher-distillation” baseline.

(a)

(b)

Training Score Mean

Validation Score

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.7

0.4

0.6

0.5

ValidationScore

0.3

ScoreMean

0.4

0.2

0.3

0.2

0.1

0.1

0.0

0.0

0 20 40 60 80 100

Init 1 2 3 4 5 6

Training Step

Epoch

(c)

###### (d)

Agent Response Length

Per-Epoch Average Score

0.517 0.524

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

250

0.505

0.488

0.5

225

ResponseLength(tokens)

0.425

200

0.4

AverageScore

175

0.3

150

125

0.2

0.138

100

0.1

75

50

0.0

0 20 40 60 80 100

1 2 3 4 5 6

Training Step

Epoch

- Figure 4 | Training dynamics of GenEnv. From left to right: (a) training step-wise reward (critic/score/mean); (b) validation score across epochs; (c) batch-level ground-truth accuracy; and (d) per-epoch average reward. The curves show that GenEnv trains stably without reward collapse or divergence, with both reward and accuracy improving smoothly over time.

Training Configuration. For the Agent Policy (𝜋agent), we train for 10 epochs using GRPO with batch size 64 and maximum sequence length 9,000 tokens (prompt + response). The Environment Policy (𝜋env) is updated at the same epoch frequency via RWR with batch size 64. We use the same optimizer family (AdamW) for both policies, with learning rates detailed in Appendix A.2. All methods are trained on the same base dataset, and for GenEnv variants the additional data comes solely from the simulator.

Evaluation Protocol. All models—including large models—are evaluated in a unified tool-calling framework. We use identical system prompts, tool specifications, and decoding settings for all models on a given benchmark. We do not attach additional multi-agent orchestration or human-in-the-loop corrections to any method, to focus the comparison on training and data regimes.

Summary of Results. Across all benchmarks, GenEnv improves the 7B base agent significantly, with gains up to +40.3% on ALFWorld and +20.4% on API-Bank compared to strong baselines.

(a)

(b)

(c)

Task Description Length

Agent Response Length

Agent Success Rate

7000

0.7

|582|8| | | | | |
|---|---|---|---|---|---|---|
| |572|4 570|8 567|9 567|1 565|7|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | |20|4|
| | | | | | | |
| |+49%|16<br><br>|0<br><br>17|4<br><br>18|7| |
|13|7<br><br>14|7| | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | |0.4|9<br><br>0.5|1 0.5|2 0.5|2|
|Targ|et α 0.4|3| | | | |
| | | | | | | |
| | | | | | | |
|0.1|4| | | | | |
| | | | | | | |

220

0.6

6000

ResponseLength(tokens)

200

PromptLength(tokens)

0.5

GTAccuracy

180

5000

0.4

160

0.3

4000

140

0.2

3000

120

0.1

2000

100

0.0

1 2 3 4 5 6

1 2 3 4 5 6

1 2 3 4 5 6

Epoch

Epoch

Epoch

- Figure 5 | Emergent curriculum in GenEnv. Across training epochs, the environment simulator gradually increases task complexity (a), reflected by longer task descriptions; the agent correspondingly produces longer reasoning chains (b) as it learns to solve harder tasks; and its success rate (c) remains within a controlled band despite rising difficulty. Together these curves show that GenEnv induces an emergent curriculum in which task difficulty and agent capability co-evolve in a stable manner.

###### 4.2. RQ1: Does GenEnv Improve Downstream Task Performance?

- Table 1 presents the main comparison. GenEnv consistently outperforms both general-purpose models and specialized RL agents across all five benchmarks. Notably, on ALFWorld, which requires long-horizon planning, GenEnv achieves 54.5% accuracy compared to 14.2% for the base model, demonstrating the power of the simulator in generating diverse embodied scenarios that are costly to collect in the real world. On API-Bank and BFCL, GenEnv achieves 79.1% and 41.8% success respectively, markedly improving over other 7B baselines that rely on static data or non-adaptive exploration.

Beyond absolute performance, GenEnv also closes much of the gap to substantially larger models. The average score of GenEnv (53.6) is competitive with—and in many cases exceeds—that of 14B–72B models that do not benefit from a difficulty-aligned simulator. This supports our central claim that how data is generated and aligned with the agent matters as much as, or more than, simply scaling model size or collecting larger static datasets. In the remainder of the section, we investigate how the co-evolutionary process shapes the curriculum, data efficiency, and difficulty calibration of the environment.

We also verify that the co-evolutionary training process itself is well behaved. Figure 4 plots the training dynamics of GenEnv: the per-step GRPO reward and batch-level ground-truth accuracy both increase steadily, and the validation score improves monotonically before saturating, without signs of reward hacking or instability. This suggests that our difficulty-aligned simulator can be optimized jointly with the agent using standard policy gradients, without introducing pathological oscillations during training.

###### 4.3. RQ2: Does GenEnv Learn to Tackle Harder Tasks Over Time?

We next investigate whether the environment simulator actually learns a curriculum, rather than simply generating random variations. To this end, we use the average length of the agent’s required response as a proxy for reasoning complexity and task difficulty, and track how it evolves throughout training. Figure 5 plots this average response length together with the agent’s success rate on simulated tasks across epochs.

Finding. The required response length increases from 137 to 204 tokens (+49%) by epoch 6. This confirms that 𝜋env learns to generate progressively more complex reasoning challenges as 𝜋agent becomes more capable, creating an emergent curriculum without manual design. At the same time, the agent’s success rate does not collapse; instead, it grows in tandem with task complexity, suggesting that the simulator is adapting difficulty in a controlled way rather than simply making tasks arbitrarily harder.

This pattern is consistent with our design of the 𝛼-Curriculum Reward: as the agent improves on a given family of tasks, their success rate on that family moves away from the target band around 𝛼, reducing its contribution to 𝑅env and encouraging the simulator to propose harder variations. The observed increase in response length indicates that these harder tasks require more extensive multi-step reasoning and tool use, rather than superficial changes to surface form. Qualitatively, we observe that later tasks tend to involve more complex compositions of tools and deeper chains of intermediate subgoals.

Finally, the fact that curriculum emerges without any hand-specified difficulty schedule supports our broader view of GenEnv as a data-evolving system: the environment learns where the agent’s “breaking points” are and adapts task generation accordingly. This contrasts with conventional curriculum learning, which typically relies on fixed heuristics or manually designed difficulty levels. Here, difficulty is inferred directly from the agent’s behaviour via 𝑅env(ˆ𝑝).

###### 4.4. RQ3: Is GenEnv More Data-Efficient Than Gemini-Based Augmentation?

Validation Score Comparison

+12.3% vs RandomEnv

+6.0% vs Static Aug

50%

40%

ValidationTestScore

30%

20%

10%

RandomEnv

Stable improvement zone

Static Aug

GenEnv (Ours)

0%

Init 1 2 3 4 5 6

Epoch

(a) Method comparison. GenEnv outperforms both static Gemini-based augmentation and the GenEnvRandom variant, showing that how data is generated and aligned with the agent matters more than simply adding more offline synthetic data.

Validation Score: Data Efficiency Comparison

| | | | |+4.6% with 3.3×|better less data| | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | |RandomEnv (40.8 Gemini 2× (43.4% Gemini 3.3× (43. GenEnv (Ours) (4|%) ) 8%) 5.8%)|

50%

40%

ValidationTestScore

30%

20%

10%

0%

Init 1 2 3 4 5 6

Epoch

(b) Data efficiency. GenEnv achieves higher validation performance while using substantially fewer synthetic samples than Gemini-based offline augmentation, indicating that difficulty-aligned, on-policy simulation provides more learning signal per example than untargeted teacher-generated data.

###### Figure 6 | Static vs. difficulty-aligned simulation.

- A key question is whether GenEnv is simply benefiting from “more data” or from better-targeted data. We compare GenEnv (which generates data on-the-fly) against offline augmentation using Gemini 2.5 Pro, under comparable training budgets. Gemini-Offline (2x / 3.3x) corresponds to large static corpora generated before training, while GenEnv continuously adapts task generation as the agent evolves.

Finding. As shown in Figure 6b, GenEnv (using 1x original data + dynamic simulation) reaches a validation score of 0.458. This outperforms Gemini-Offline (3.3x) (0.438), which uses ≈3.3x more synthetic data generated by a much stronger model. This confirms that targeted, difficulty-aligned

data generation can be structurally superior to massive but untargeted augmentation. Furthermore, GenEnv outperforms GenEnv-Random by 12.3%, indicating that the 𝑅env optimization is critical.

These results highlight two distinct effects. First, merely adding more synthetic trajectories from a powerful teacher model quickly encounters diminishing returns: once the static dataset ceases to match the agent’s current weaknesses, additional examples provide limited new learning signal. Second, the comparison between GenEnv and GenEnv-Random controls for the presence of a simulator: both generate trajectories online, but only GenEnv trains the simulator to target the 𝛼 band. The performance gap between these two variants isolates the benefit of difficulty alignment itself, rather than just the benefit of having a generative environment.

From a practical standpoint, these findings suggest a shift in how we invest computational and annotation budget. Instead of paying for ever larger static datasets created by stronger teachers, it may be more effective to invest in a moderately sized simulator that co-evolves with the student agent. This is especially attractive in domains where collecting real trajectories is expensive or slow, as the simulator can keep generating fresh, on-policy data without requiring repeated human involvement.

###### 4.5. RQ4: Does the Environment Reward Produce Well-Calibrated Difficulty?

Success Rate Convergence to Target α

| | | | | | | |
|---|---|---|---|---|---|---|
|Enters α-z at epoch|one 2| | | |Converged: 52|.4%|
|Targe|t α = 50%| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | |Gen Targ<br><br>| |
|---|
|Env et Zone (α=40%|-60%)|

70%

60%

AgentSuccessRate

50%

40%

30%

20%

10%

0%

1 2 3 4 5 6

Epoch

- Figure 7 | Difficulty calibration via the 𝛼-Curriculum Reward. As training progresses, the agent’s success rate on simulator-generated tasks converges to the target difficulty band (centered at 𝛼 = 0.5), demonstrating that the environment policy reliably adapts task difficulty to match the agent’s current

capability. This empirically verifies the theoretical ranking consistency of 𝑅env and shows that the simulator self-calibrates to maintain tasks in the zone of proximal development.

Finally, we verify if the 𝛼-Curriculum Reward successfully calibrates task difficulty. During training, we track the agent’s success rate on simulated tasks generated by 𝜋env and examine whether it converges to the intended band around 𝛼. Figure 7 summarizes this trajectory over training epochs.

Finding. Figure 7 shows the agent’s success rate on generated tasks during training. Starting from 0.138, the success rate converges towards a band centered at the target difficulty 𝛼 = 0.5, remaining within a range of approximately [0.4, 0.6] for most of training. This suggests that 𝜋env is actively optimizing for the “zone of proximal development,” avoiding the collapse into trivial or impossible tasks that plagues random generation.

This behaviour is precisely what our theoretical analysis predicts. Theorem 1 shows that, given

###### (a)

###### (b)

Fully Solved Tasks Over Training

Unsolved Tasks Over Training

+3.5% more solved per batch

3000%

RandomEnv

RandomEnv

6000%

GenEnv (Ours)

GenEnv (Ours)

2500%

5000%

FullySolvedTasks(perbatch)

UnsolvedTasks(perbatch)

4000%

2000%

3000%

20.3% fewer unsolved tasks

1500%

2000%

1000%

1000%

500%

0%

1 2 3 4 5 6

1 2 3 4 5 6

Epoch

Epoch

- Figure 8 | Problem-solving behavior during training. (a) GenEnv consistently increases the proportion of fully solved tasks per batch, surpassing the RandomEnv variant; (b) the rate of unsolved tasks decreases substantially faster under GenEnv. These trends show that difficulty-aligned simulation not only improves average performance but also accelerates the elimination of failure modes compared to unguided task generation.

enough rollouts, 𝑅env(ˆ𝑝) provides a consistent ranking signal that favours task types with success probabilities closest to 𝛼. Empirically, we see this mechanism in action: early in training, when most tasks are either too hard or too easy, the simulator updates quickly reshape the distribution towards intermediate difficulty. Later, updates become smaller as the success rate stabilizes near the target band, leading to a self-calibrated curriculum.

We also observe that the calibrated difficulty band coexists with improved downstream performance on real benchmarks. That is, the simulator does not merely “keep the agent at 50% success” on synthetic tasks; rather, it continually moves the frontier of what intermediate difficulty means as the agent learns. This reinforces our view of GenEnv as a genuinely co-evolving system, in which both the agent and the task distribution adapt in lockstep.

### 5. Related Work

###### 5.1. Large Language Model Agents

Recent advancements have demonstrated the ability of LLMs to function as autonomous agents. Pioneering works like ReAct, Reflexion, and Voyager have shown that combining chain-of-thought reasoning with action generation and memory evaluation enables agents to tackle complex tasks (Guo et al., 2025; Shinn et al., 2023; Wang et al., 2023; Yao et al., 2023; Zou et al., 2025). More recent efforts such as KnowAgent and MemBench further refine planning and memory capabilities in agentic settings (Tan et al., 2025; Zhu et al., 2024). Others, such as Toolformer and WebGPT, have focused on augmenting LLMs with external tools and web-browsing capabilities, expanding their operational scope (Nakano et al., 2021; Schick et al., 2023). While these models showcase strong performance, their training paradigms primarily rely on imitation learning from static, pre-collected datasets of expert trajectories (Nakano et al., 2021; Wang et al., 2023). This dependency on expert data forms a significant bottleneck, limiting the agent’s ability to explore and discover strategies beyond the provided demonstrations (Shinn et al., 2023). Our work addresses this limitation by

creating a dynamic learning environment that does not solely depend on a fixed dataset.

###### 5.2. Trajectory Synthesis for Agent Training

Recent efforts have focused on generating synthetic trajectories to address the limitations of fixed expert data for training LLM agents (Yu et al., 2025). Some methods focus on offline synthetic data generation, creating novel trajectories to increase diversity and coverage where expert data is sparse (Ding et al., 2024; Ye et al., 2024). Other approaches leverage LLMs to generate self-reflective trajectories, incorporating reflections and corrections to learn from errors (Chen, 2025), or to provide stepwise guidance from a teacher model toward correcting mistakes (Chen et al., 2025b). For web agents, scalable synthesis pipelines use web tutorials or exploration-driven methods to produce large-scale synthetic datasets of multimodal trajectories (Pahuja et al., 2025; Yuan et al., 2024). Other works introduce iterative self-training for reflection (Wang et al., 2025a; Yuan et al., 2025), fine-tuning on massive interaction trajectories (Zhang et al., 2025a), step-level calibration (Luo et al.,

- 2025), and simulators for online exploration to generate high-quality feedback-driven data (Hoang et al., 2025).

Beyond these, several recent frameworks further advance autonomous and adaptive trajectory synthesis. Wang et al. (2025c) propose a scalable LLM-based digital environment that models user-interface transitions as structured trajectories, combining simulation and targeted scaling to expose agents to increasingly complex states. Zhao et al. (2025) build upon exploration-driven task generation, using a two-stage pipeline that first explores application environments and then synthesizes executable trajectories, yielding tens of thousands of realistic multimodal interaction traces. In the data-science domain, Zhang et al. (2025b) integrate curriculum-based agentic training with a data-grounded trajectory synthesis framework to produce high-fidelity analytical workflows, allowing smaller models to outperform larger workflow-based agents. Complementarily, Liu et al. (2025) challenge the assumption that more data is always better—demonstrating that strategically curated and high-quality trajectories can induce stronger agentic reasoning from only a handful of demonstrations. Chen et al. (2025a) also demonstrates that noisy preference data could be utilized into improve agent alignment. Finally, Sun et al. (2025) introduce an implicit-feedback paradigm where agents learn from their own early interactions before formal reinforcement learning, leveraging self-reflection and world-modeling to bootstrap generalization from suboptimal actions.

Taken together, these developments indicate a trend toward adaptive, self-improving trajectory synthesis: instead of relying on static or expert-curated data, agents increasingly generate, evaluate, and refine their own experiences—closing the loop between simulation, exploration, and learning. This motivates our approach, which leverages a dynamic environment simulator to generate these adaptive experiences on-demand, directly addressing the limitations of static datasets.

###### 5.3. Environment Simulation

Simulators have long been a cornerstone of reinforcement learning, especially in domains such as robotics where interacting with the real world is costly (Todorov et al., 2012). In the context of LLM agents, environment simulation has emerged as a key mechanism for generating training data and evaluating agent capabilities. Wang et al. (2025c) demonstrate how an LLM-powered digital world simulator can generate structured user-interface states and transitions; its targeted scaling strategy produces diverse, high-impact tasks and yields agents that rival those trained on real UIs. Complementary simulation frameworks move beyond UI tasks: Zhou et al. (2025) introduce a scalable closed-loop simulator that samples multi-step tasks from a tool–relationship graph, simulates interactions with configurable user and environment archetypes, and evaluates procedural alignment

and success. Experiments reveal that environment reliability and user archetypes are dominant factors in agent performance. Beyond task-centric environments, Li et al. (2025a) integrate LLM-driven agents with a realistic societal environment and a large-scale simulation engine, generating social lives for over ten thousand agents and millions of interactions among agents and their surroundings. These works highlight the importance of faithful and diverse environment simulation in creating high-quality training data and benchmarks.

Our environment LLM diverges from traditional simulators: rather than predicting state transitions or user responses for a fixed task, it generates entire tasks and goals conditioned on the agent’s recent performance, with an explicit objective to match a target difficulty band. This casts environment design itself as a learnable policy with its own reward signal.

### 6. Conclusion

We presented GenEnv, a framework that shifts agent training from a static, model-evolving process to a dynamic, data-evolving game. By establishing a difficulty-aligned co-evolutionary loop between an Agent Policy and an Environment Policy, GenEnv achieves superior performance and data efficiency on a diverse suite of agent benchmarks. Our results suggest that future agent training systems should move beyond larger static datasets toward adaptive, self-calibrating simulation environments. Beyond the particular instantiation studied here, we believe that difficulty-aligned simulators can serve as a general recipe for training robust LLM agents in domains where real-world exploration is costly or risky.

### References

P. Chen, X. Chen, W. Yin, and T. Lin. Compo: Preference alignment via comparison oracles. In NeurIPS, 2025a.

P. Chen, X. Li, Z. Li, X. Chen, and T. Lin. Stepwise guided policy optimization: Coloring your incorrect

reasoning in grpo. In The 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025, 2025b. P. Chen, X. Li, Z. Li, W. Yin, X. Chen, and T. Lin. Exploration vs exploitation: Rethinking rlvr through

clipping, entropy, and spurious reward. arXiv preprint arXiv:2512.16912, 2025c.

- Y. Chen. Training LLM-based agents with synthetic self-reflected trajectories and partial masking. arXiv preprint arXiv:2505.20023, 2025.

B. Ding, C. Qin, R. Liu, L. Bing, S. Joty, Q. Li, and C. Xiao. Data augmentation using large language models: Data perspectives, learning paradigms and challenges. arXiv preprint arXiv:2403.02990, 2024.

- A. Grattafiori et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

J. Guo, Z. Li, J. Qiu, Y. Wu, and M. Wang. On the role of preference variance in preference optimization. arXiv preprint arXiv:2510.13022, 2025.

- I. Gur, O. Huang, A. Kim, A. Anderson, G. Fast, N. Kushman, B. Zitkovich, D. Aiken, M. G. Luong, et al. Understanding HTML with large language models. arXiv preprint arXiv:2210.03945, 2023.

T. Q. Hoang, K.-H. Huang, S. Kokane, J. Zhang, Z. Liu, M. Zhu, J. Grigsby, T. Lan, M. S. Ryoo, C.-S. Wu, S. Heinecke, H. Wang, S. Savarese, C. Xiong, and J. C. Niebles. LAM SIMULATOR: Advancing data generation for large action model training via online exploration and trajectory feedback. Findings of the Association for Computational Linguistics: ACL 2025, 2025.

B. Jin, H. Zeng, Z. Yue, D. Wang, H. Zamani, and J. Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025. URL https://arxiv.org/abs/2503.09516.

S. Levine, A. Kumar, G. Tucker, and J. Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems, 2020.

C. Li, R. Wang, X. Zhao, B. Xu, F. Liu, R. Yang, and X. Zhang. Agentsociety: Large-scale simulation of LLM-driven generative agents advances understanding of human behaviors and society. arXiv preprint arXiv:2502.08691, 2025a. URL https://arxiv.org/abs/2502.08691.

- M. Li, Y. Chen, Z. Zhang, et al. A comprehensive benchmark for tool-augmented LLMs. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. URL https://aclanthology.org/2023.emnlp-main.187. API-Bank benchmark.

Z. Li, T. Zou, et al. ToRL: Scaling tool-integrated reinforcement learning. arXiv preprint

arXiv:2503.23383, 2025b. URL https://arxiv.org/abs/2503.23383.

Z. Liu, H. Yuan, Q. Xu, X. Jin, and H. Ren. LIMI: Less is more for agency. arXiv preprint arXiv:2509.17567, 2025.

- J. Luo et al. STeCa: Step-level trajectory calibration for LLM agent learning. arXiv preprint arXiv:2503.21460, 2025.

- R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders,

- X. Jiang, K. Cobbe, T. Eloundou, G. Krueger, K. Button, M. Knight, B. Chess, and J. Schulman. Webgpt: Browser-assisted question-answering with human feedback, 2021.

L. Ning, Z. Liang, Z. Jiang, H. Qu, Y. Ding, W. Fan, X.-y. Wei, S. Lin, H. Liu, P. S. Yu, and Q. Li. A survey of webagents: Towards next-generation AI agents for web automation with large foundation models. arXiv preprint arXiv:2503.23350, 2025.

OpenAI. GPT-OSS: Open-weight models for reasoning and agents. https://openai.com/index/ introducing-gpt-oss/, 2025. OpenAI blog post.

V. Pahuja, Y. Lu, C. Rosset, B. Gou, A. Mitra, S. Whitehead, Y. Su, and A. H. Awadallah. Explorer: Scaling exploration-driven web trajectory synthesis for multimodal web agents. Findings of the Association for Computational Linguistics: ACL 2025, 2025.

- S. G. Patil, Z. Zhang, X. Li, et al. The berkeley function-calling leaderboard (BFCL). In International Conference on Machine Learning (ICML), 2025. URL https://gorilla.cs.berkeley.edu/blogs/ 8_berkeley_function_calling_leaderboard.html. Berkeley Function-Calling Leaderboard.

D. A. Pomerleau. Efficient training of artificial neural networks for autonomous navigation. Neural Computation, 3(1):88–97, 1991.

O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, and M. Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, 2023. URL https://arxiv.org/abs/2210.03350.

A. Samadi, S. K. S. Ghasemipour, V. Caggiano, M. Dangelmaier, and S. Jha. Good data is all imitation learning needs. arXiv preprint arXiv:2409.17605, 2024.

- T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

J. Schulman, S. Levine, P. Abbeel, M. Jordan, and P. Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

Z. Shao et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300. Introduces GRPO (Group Relative Policy Optimization).

- N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS),

#### 2023. URL https://arxiv.org/abs/2303.11366.

M. Shridhar, X. Yuan, M.-A. Côté, Y. Bisk, A. Trischler, and M. Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations (ICLR), 2021. URL https://arxiv.org/abs/2010.03768.

J. Sun, Y. Ren, H. Wang, Y. Tang, J. Liu, and Z. Yuan. Agent learning via early experience. arXiv preprint arXiv:2510.08558, 2025.

H. Tan, Z. Zhang, C. Ma, X. Chen, Q. Dai, and Z. Dong. MemBench: Towards more comprehensive evaluation on the memory of LLM-based agents. In Findings of the Association for Computational Linguistics: ACL 2025, 2025. URL https://aclanthology.org/2025.findings-acl.989.

- E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5026–5033, 2012.

L. S. Vygotsky. Mind in Society: The Development of Higher Psychological Processes. Harvard University Press, Cambridge, MA, 1978.

G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. Voyager: An open-ended embodied agent with large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.16291.

- Y. Wang, L. Yang, Y. Tian, K. Shen, and M. Wang. Co-evolving LLM coder and unit tester via reinforcement learning. arXiv preprint arXiv:2506.03136, 2025a.

Y. Wang, L. Yang, Y. Tian, K. Shen, and M. Wang. Cure: Co-evolving coders and unit testers via reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b.

- Y. Wang, D. Yin, Y. Cui, R. Zheng, Z. Li, et al. UI-Simulator: LLMs as scalable, general-purpose simulators for evolving digital agent training. arXiv preprint arXiv:2510.14969, 2025c. URL https://arxiv.org/abs/2510.14969.

J. Xie, K. Zhang, J. Chen, T. Zhu, R. Lou, Y. Tian, Y. Xiao, and Y. Su. Travelplanner: A benchmark for real-world planning with language agents. In International Conference on Machine Learning (ICML),

2024. URL https://arxiv.org/abs/2402.01622.

A. Yang and Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024. URL

https://arxiv.org/abs/2412.15115.

A. Yang and Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL

https://arxiv.org/abs/2505.09388.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR),

2023. URL https://arxiv.org/abs/2210.03629.

J. Ye, X. Gao, K. Zhang, G. Gao, Y. Li, and X. Liu. LLM-DA: Data augmentation via large language models for few-shot named entity recognition. arXiv preprint arXiv:2402.14568, 2024.

- Z. Yu, L. Yang, J. Zou, S. Yan, and M. Wang. Demystifying reinforcement learning in agentic reasoning. arXiv preprint arXiv:2510.11701, 2025.

S. Yuan, Z. Chen, Z. Xi, J. Ye, Z. Du, and J. Chen. Agent-R: Training language model agents to reflect via iterative self-training. arXiv preprint arXiv:2501.11425, 2025.

- S. Yuan et al. Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials. arXiv preprint, 2024.

J. Zhang et al. Agentbank: Towards generalized LLM agents via fine-tuning on 50000+ interaction trajectories. arXiv preprint, 2025a.

R. Zhang, H. Li, J. Liu, M. Chen, and X. Wang. Deepanalyze: Agentic large language models for autonomous data science. arXiv preprint arXiv:2510.16872, 2025b.

- T. Zhao, Q. Huang, H. Liu, J. He, F. Peng, P. Li, and Y. Liu. Scaling synthetic task generation for agents via exploration. arXiv preprint arXiv:2509.25047, 2025.

J. Zhou, Y. Lee, K. Huang, Y. Zhao, R. Chen, and H. Zhang. Faithful simulation of user–agent– environment interactions for reliable evaluation of LLM agents. In Proceedings of the 2025 International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/ pdf/b590327f99064f537038a6254f0d993d2671ad29.pdf. OpenReview preprint.

Y. Zhu, S. Qiao, Y. Ou, S. Deng, N. Zhang, et al. Knowagent: Knowledge-augmented planning for LLM-based agents. arXiv preprint arXiv:2403.03101, 2024. URL https://arxiv.org/abs/ 2403.03101.

J. Zou, X. Yang, R. Qiu, G. Li, K. Tieu, P. Lu, K. Shen, H. Tong, Y. Choi, J. He, et al. Latent collaboration in multi-agent systems. arXiv preprint arXiv:2511.20639, 2025.

### A. Appendix

- A.1. Proofs for Section 3 In this appendix we provide detailed proofs for the theoretical results stated in Section 3.

###### A.1.1. Proof of Proposition 1

Recall that for a fixed task type 𝜏 and parameter vector 𝜃, the Agent Policy update uses the REINFORCEstyle estimator

𝑔(𝜏, 𝑟) = (𝑟 − 𝑏(𝜏))∇𝜃 log𝜋𝜃(𝑎 | 𝜏),

and we choose the baseline to be 𝑏(𝜏) = 𝔼[𝑟 | 𝜏] = 𝑝(𝜏). Let 𝑆 = ∇𝜃 log𝜋𝜃(𝑎 | 𝜏) for brevity. Conditioned on 𝜏, the reward 𝑟 is Bernoulli with success probability 𝑝(𝜏), so 𝔼[𝑟 | 𝜏] = 𝑝(𝜏) and Var(𝑟 | 𝜏) = 𝑝(𝜏)(1 − 𝑝(𝜏)).

We have

𝔼 ∥𝑔(𝜏, 𝑟)∥2 | 𝜏 = 𝔼 (𝑟 − 𝑝(𝜏))2 ∥𝑆∥2 | 𝜏 (10)

= ∑︁

Pr(𝑟 | 𝜏) (𝑟 − 𝑝(𝜏))2 𝔼 ∥𝑆∥2 | 𝜏, 𝑟 . (11)

𝑟∈{0,1}

By Assumption 1, for both 𝑟 = 0 and 𝑟 = 1 we have

𝑐min ≤ 𝔼 ∥𝑆∥2 | 𝜏, 𝑟 ≤ 𝑐max. Therefore

𝔼 ∥𝑔(𝜏, 𝑟)∥2 | 𝜏 ≥ 𝑐min ∑︁

Pr(𝑟 | 𝜏)(𝑟 − 𝑝(𝜏))2 (12)

𝑟∈{0,1}

= 𝑐min 𝔼 (𝑟 − 𝑝(𝜏))2 | 𝜏 (13) = 𝑐min Var(𝑟 | 𝜏) (14) = 𝑐min 𝑝(𝜏) 1 − 𝑝(𝜏) , (15)

and similarly

𝔼 ∥𝑔(𝜏, 𝑟)∥2 | 𝜏 ≤ 𝑐max ∑︁

Pr(𝑟 | 𝜏)(𝑟 − 𝑝(𝜏))2 (16)

𝑟∈{0,1}

= 𝑐max 𝑝(𝜏) 1 − 𝑝(𝜏) . (17)

Taking 𝐶min = 𝑐min and 𝐶max = 𝑐max yields the bounds in Equation equation 5. Since 𝑝(1 − 𝑝) is a concave quadratic on [0, 1] with a unique maximum at 𝑝 = 1/2, the expected squared gradient norm is maximized (up to constant factors) for tasks with 𝑝(𝜏) = 1/2, i.e., tasks of intermediate difficulty. This proves Proposition 1.

###### A.1.2. Proof of Theorem 1

We restate the setting for clarity. For 𝑖 ∈ {1, 2}, the true success probability on task type 𝜏𝑖 is 𝑝𝑖, and we define Δ𝑖 = |𝑝𝑖 − 𝛼| with Δ1 < Δ2. From 𝑛𝑖 independent rollouts we obtain the empirical success rate ˆ𝑝𝑖 = 𝑘𝑖/𝑛𝑖, where 𝑘𝑖 is the number of successes. The 𝛼-Curriculum Reward is

𝑅env(ˆ𝑝𝑖) = exp − 𝛽(ˆ𝑝𝑖 − 𝛼)2 .

Since the exponential function is strictly monotone decreasing in (ˆ𝑝𝑖 − 𝛼)2, we have 𝑅env(ˆ𝑝1) ≤ 𝑅env(ˆ𝑝2) ⇐⇒ |ˆ𝑝1 − 𝛼| ≥ |ˆ𝑝2 − 𝛼|.

Thus the event that the Environment Policy mis-ranks the two tasks (i.e., gives 𝜏1 no larger reward than 𝜏2) is exactly the event

Emis = { |ˆ𝑝1 − 𝛼| ≥ |ˆ𝑝2 − 𝛼| }. Let 𝑛 = min{𝑛1, 𝑛2} and define

Δ2 − Δ1 3

> 0. Consider the event

𝛿 =

Egood = |ˆ𝑝1 − 𝑝1| ≤ 𝛿 and |ˆ𝑝2 − 𝑝2| ≤ 𝛿 .

We claim that on Egood we must have |ˆ𝑝1 − 𝛼| < |ˆ𝑝2 − 𝛼|, and hence 𝑅env(ˆ𝑝1) > 𝑅env(ˆ𝑝2). Indeed, by the triangle inequality,

- |ˆ𝑝1 − 𝛼| ≤ |𝑝1 − 𝛼| + |ˆ𝑝1 − 𝑝1| ≤ Δ1 + 𝛿, (18)
- |ˆ𝑝2 − 𝛼| ≥ |𝑝2 − 𝛼| − |ˆ𝑝2 − 𝑝2| ≥ Δ2 − 𝛿. (19)

By the choice of 𝛿, we have

2Δ1 + Δ2 3

Δ1 + 2Δ2 3

Δ2 − Δ1 3

Δ2 − Δ1 3

=

, Δ2 − 𝛿 = Δ2 −

=

Δ1 + 𝛿 = Δ1 +

,

and since Δ1 < Δ2, it follows that

2Δ1 + Δ2 3

Δ1 + 2Δ2 3

= Δ2 − 𝛿. Therefore,

Δ1 + 𝛿 =

<

|ˆ𝑝1 − 𝛼| ≤ Δ1 + 𝛿 < Δ2 − 𝛿 ≤ |ˆ𝑝2 − 𝛼| on Egood, which implies 𝑅env(ˆ𝑝1) > 𝑅env(ˆ𝑝2). Consequently, Emis can only occur on the complement event Egood𝑐 , and hence

Pr(Emis) ≤ Pr(Egood𝑐 ) ≤ Pr |ˆ𝑝1 − 𝑝1| > 𝛿 + Pr |ˆ𝑝2 − 𝑝2| > 𝛿 , where the last inequality is a union bound.

For each 𝑖 ∈ {1, 2}, ˆ𝑝𝑖 is the empirical mean of 𝑛𝑖 i.i.d. Bernoulli random variables with mean 𝑝𝑖. By Hoeffding’s inequality,

Pr |ˆ𝑝𝑖 − 𝑝𝑖| > 𝛿 ≤ 2exp(−2𝑛𝑖𝛿2) ≤ 2exp(−2𝑛𝛿2), where 𝑛 = min{𝑛1, 𝑛2}. Therefore

Pr(Emis) ≤ 4exp(−2𝑛𝛿2) (20)

2

Δ2 − Δ1 3

= 4exp −2𝑛

(21)

2 9(Δ2 − Δ1)2 𝑛 , (22)

= 4exp −

which establishes the desired exponential bound and completes the proof of Theorem 1.

- Table 2 | Hyperparameters for GenEnv training.

###### Parameter Agent Policy (𝜋agent) Environment Policy (𝜋env)

Optimizer (AdamW) Learning Rate 1 × 10−6 5 × 10−7 Batch Size 64 64

Policy Optimization Method GRPO Reward-Weighted Update (RWR) Total Epochs 10 10 Target Difficulty 𝛼 N/A 0.5 Difficulty Filter 𝑘min N/A 0.1 RWR Temperature 𝜆 N/A 1.0

###### A.2. Hyperparameter Details

Table 2 lists the key hyperparameters used in our experiments. Note that for the Agent (𝜋agent), we employ GRPO, while the Environment (𝜋env) uses Reward-Weighted Regression (RWR).

###### A.3. Environment Baseline Implementation Details Here we detail the specific implementation of the environment variants used in Section 4:

- • GenEnv-Random: We use the same base model (Qwen2.5-7B-Instruct) for the environment. The autoenv.disable_env_training flag is set to True. It generates 4 variations per prompt per epoch dynamically, but the model weights are never updated based on 𝑅env(ˆ𝑝).
- • GenEnv-Static: We use the generate_static_augmentation.py script to pre-generate 5 variations for each of the 544 original training samples, resulting in a fixed dataset of 3,264 samples. The agent is trained using standard PPO for 10 epochs.
- • Gemini-Offline Baselines: We prompted Gemini 2.5 Pro to generate variations of the training data. Due to API constraints and filtering, the Gemini-2x setting resulted in 957 samples (≈1.76x) and Gemini-4x resulted in 1,777 samples (≈3.27x). These represent high-quality, but static, external data augmentation.

