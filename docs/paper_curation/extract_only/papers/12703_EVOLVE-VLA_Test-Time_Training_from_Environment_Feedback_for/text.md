# arXiv:2512.14666v1[cs.RO]16Dec2025

## EVOLVE-VLA: Test-Time Training from Environment Feedback for Vision-Language-Action Models

Zechen Bai, Chen Gao, Mike Zheng Shou*

Show Lab, National University of Singapore

https://showlab.github.io/EVOLVE-VLA

### Abstract

Achieving truly adaptive embodied intelligence requires agents that learn not just by imitating static demonstrations, but by continuously improving through environmental interaction, which is akin to how humans master skills through practice. Vision-Language-Action (VLA) models have advanced robotic manipulation by leveraging large language models, yet remain fundamentally limited by Supervised Finetuning (SFT): requiring hundreds of demonstrations per task, rigidly memorizing trajectories, and failing to adapt when deployment conditions deviate from training. We introduce EVOLVE-VLA, a test-time training framework enabling VLAs to continuously adapt through environment interaction with minimal or zero task-specific demonstrations. The key technical challenge is replacing oracle reward signals (unavailable at test time) with autonomous feedback. We address this through a learned progress estimator providing dense feedback, and critically, we design our framework to “tame” this inherently noisy signal via two mechanisms: (1) an accumulative progress estimation mechanism smoothing noisy point-wise estimates, and (2) a progressive horizon extension strategy enabling gradual policy evolution. EVOLVE-VLA achieves substantial gains: +8.6% on long-horizon tasks, +22.0% in 1-shot learning, and enables cross-task generalization—achieving 20.8% success on unseen tasks without task-specific demonstrations training (vs. 0% for pure SFT). Qualitative analysis reveals emergent capabilities absent in demonstrations, including error recovery and novel strategies. This work represents a critical step toward VLAs that truly learn and adapt, moving beyond static imitation toward continuous self-improvements.

#### 1. Introduction

How do humans develop manipulation skills? We do not simply watch an expert perform a task once and then flawlessly replicate it. Instead, we learn through practice: attempting the task repeatedly, making mistakes, receiving feedback from the environment, and gradually refining our movements through continued experience. This process of learning by doing, rather than merely learning by watching, is fundamental to how intelligent agents acquire robust and adaptable capabilities in the real world.

Embodied intelligence, the integration of AI technology with robots, has seen remarkable progress in recent years. Propelled by the capabilities of Large Language Models (LLMs), control policies are rapidly evolving beyond traditional methods toward general Vision-Language-

*Corresponding Author

(a) Supervised Fine-Tuning (b) Test-time Training

###### Numerous Demonstrations Required

😊 Few Demonstration for Pre-training 💪 Robust and Generalizable

[Figure 1]

[Figure 2]

😤 😵 Pure Imitation -> Poor Generalization

[Figure 3]

[Figure 4]

|[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]| | |[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]| |
|---|---|---|---|---|

action

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

###### …

Environment

[Figure 28]

[Figure 29]

Reward Model

Action-wise Behavior Cloning

reward

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

VLA Model

VLA Model

|[Figure 34]<br><br>[Figure 35]|
|---|

|[Figure 36]<br><br>[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Fail to pick Still go to basket (w/o TTT)

|[Figure 42]<br><br>[Figure 43]|
|---|

|[Figure 44]<br><br>[Figure 45]|
|---|

[Figure 46]

Put into basket (w/ TTT)

React to pick again (w/ TTT)

(c) Improvement with TTT (d) Performance Comparison (e) Evolving React Ability with TTT

- Figure 1 | (a) Traditional supervised finetuning paradigm for training VLA model requires numerous demonstration data and risks rigidly cloning trajectories in the training data. (b) Our test-time training framework requires only few demonstrations (even none) for pre-training and can evolve in the deployment environment. In (c) and (d), we show that our method achieves performance gains across various number of demonstrations and different task suites. Notably, for the first time, we observe zero-shot cross-task generation with the test-time-training framework. (e) illustrates the ability of recovering from mistakes evolved during TTT.

Action (VLA) models [3, 11, 12], which process multimodal inputs to produce a sequence of actions for completing a given task. By leveraging the rich semantic priors from LLMs, VLAs demonstrate impressive contextual understanding compared to their predecessors. However, despite these advances, current VLA training remains fundamentally misaligned with the human learning principle described above: they are trained exclusively through Supervised Fine-Tuning (SFT) on fixed demonstration datasets, learning to imitate expert behavior but lacking mechanism to improve through environmental interaction.

This paradigm of static imitation learning entails two fundamental limitations. (1) High labor cost. As shown in Fig. 1(a), adapting VLA models to new tasks requires collecting hundreds of demonstrations for supervised fine-tuning (SFT). This cost multiplies linearly with tasks, making it infeasible to scale VLAs to truly general-purpose robots. (2) Brittle memorization. VLAs optimized through behavior cloning merely imitate demonstrations and struggle to generalize beyond training distribution. They lack the ability to recover from execution deviations, where a single misstep often leads to complete task failure. These limitations represent a fundamental misalignment with how adaptive intelligence should operate. We believe that enabling continuous learning from deployment experience is essential for achieving truly general-purpose vision-language-action models.

In this work, we propose EVOLVE-VLA (Efficient VLA Online Learning Via Experience), a test-time training framework that fundamentally shifts how VLAs learn and adapt. As illustrated in Fig. 1(b), instead of requiring hundreds of expert demonstrations, our method needs only minimal supervision, a few demonstrations or even none, for lightweight initialization

via SFT. The key innovation lies in what happens after this initial pre-training: rather than freezing the policy, we deploy it directly in the target environment where it continues to learn autonomously through active interaction. The VLA explores the environment, receives feedback, and refines its behavior via online reinforcement learning, mirroring the trial-and-error process through which humans develop manipulation skills.

This paradigm shift addresses both limitations: (1) it dramatically reduces labor costs by replacing extensive demonstrations with autonomous learning, and (2) it enables genuine adaptation rather than memorization, producing policies that recover from errors and discover novel strategies. For example, Fig. 1(e) shows our model developing error correction capabilities absent from training demonstrations. Beyond improving seen tasks, this approach enables cross-task generalization through self-directed exploration.

While prior works like SimpleVLA-RL [13] have explored RL for VLA models, they rely on oracle reward functions (e.g., binary success signals) unavailable at test time. The central challenge of practical TTT is replacing the oracle with autonomous feedback. We introduce a learned progress estimator as reward, with the policy optimized via GRPO [21]. Unlike sparse success signals, progress-based rewards provide dense, continuous feedback crucial for sample-efficient learning. However, practical progress estimators are inherently noisy [18, 19, 22, 30], and errors accumulated over long horizons can mislead the policy.

Our core technical challenge is therefore not to build a perfect estimator, but to successfully “tame” this noisy reward signal to make learning possible. To achieve this, we introduce two key technical contributions. First, we design an accumulative progress estimation mechanism with interval-based sampling, which aggregates and smooths noisy point-wise estimates into a stable, reliable signal. Second, we propose a progressive horizon extension strategy that optimizes the policy with progressively increasing exploration horizon, making the model more resilient to estimation errors by allowing it to first master simpler sub-tasks. This combined approach not only mitigates the impact of estimation noise but also allows the VLA to effectively utilize the dense, albeit imperfect, reward.

Our framework enables VLA models to perform test-time training using self-generated environmental feedback without oracle rewards. We validate EVOLVE-VLA on the LIBERO benchmark, achieving substantial gains: +8.6% on long-horizon tasks, +22.0% in 1-shot learning, and cross-task transfer (0% → 20.8% on unseen tasks through autonomous adaptation). Qualitative analysis reveals emergent capabilities absent from demonstrations, including error recovery and novel strategies. These results validate that test-time training represents a paradigm shift toward adaptive embodied agents—a critical step toward truly general-purpose VLA systems. Our contributions include:

- • We propose EVOLVE-VLA, a test-time training framework that enables VLAs to continuously adapt through autonomous interaction, addressing the brittleness and scalability limitations of static SFT.
- • We tackle the central challenge of absence of oracle rewards by introducing a learned progress estimator. Critically, we develop techniques to “tame” inherently noisy reward signals, making practical test-time training feasible.
- • We introduce two key innovations: (1) an accumulative progress estimation mechanism that smooths noisy estimates into stable signals, and (2) a progressive horizon extension strategy enabling gradual policy evolution, proving effective for long-horizon tasks.
- • We demonstrate strong results: +8.6% on long-horizon tasks, +22.0% in 1-shot learning, and pioneering zero-shot cross-task generalization (0% → 20.8%) through test-time adaptation alone. Our analysis reveals emergent skills like error recovery arising from autonomous exploration.

#### 2. Related Work

Vision-Language-Action Models. Recent advances in Vision-Language-Action (VLA) models [3, 7, 11, 12, 20, 26, 27, 31] aim to equip embodied agents with the ability to perceive, reason, and act upon multimodal inputs. Early works like RT [4] and Octo [23] investigate how to connect the power of large models with the interactive nature of embodied environments, paving the way toward generalist robot manipulation. OpenVLA [12] presents an open-source VLA model fine-tuned across multiple manipulation tasks, aiming to standardize evaluation and promote reproducible research. OpenVLA-OFT [11] further proposes parallel decoding, action chunking, and a continuous action representation to improve performance. 𝜋0 [3] introduces a VLA flow model by a continuous flow-based architecture. The approach demonstrates strong generalization across diverse robot manipulation tasks and sets a new direction for flow-based embodied reasoning.

Some works focus on improving the efficiency of VLA model. TinyVLA [27] designs a lightweight VLA for robotic manipulation, which employs parameter sharing and distillation to retain performance under limited data. Recent works [2, 9, 28] also investigate how to involve tactile modality in VLA models. However, previous methods rely heavily on imitation learning with numerous manual-collected data, leading to labor-cost and poor generalization models, especially when meeting the new tasks and environments.

RL Fine-Tuning for VLA Models. With the recent advances of RL post-training in LLMs [5, 24] and MLLMs [15, 25], some studies have begun to explore RL post-training for VLA models. For example, iRe-VLA [10] explores how online RL can enhance pretrained VLA models by allowing continual improvement through interaction. VLA-RL [17] introduces a trajectorylevel RL formulation for VLA training. OctoNav [8] investigates how GRPO-like RL training can improve VLA reasoning ability in embodied navigation. SimpleVLA-RL [13] and 𝜋𝑅𝐿 [6] explore RL fine-tuning for autoregressive and flow-based VLAs, respectively. RL4VLA [16] systematically studies different RL policies and the impact of RL fine-tuning across diverse visual, semantic, and execution dimensions. Although these works have explored RL posttraining strategies for VLA models, they still assume access to Ground-Truth (GT) information during the RL training phase, such as whether a trajectory succeeds or fails. However, at test time, such GT supervision signals are unavailable. To address this, we propose a test-time training framework that enables the model to adapt without relying on GT feedback.

Concurrent Work: 𝜋∗0.6. Concurrent to our work, Physical Intelligence recently released 𝜋∗0.6 [1], a vision-language-action model that learns from autonomous experience using their Recap method (RL with Experience & Corrections via Advantage-conditioned Policies). Our work shares a similar motivation and spirit with 𝜋∗0.6 in addressing a fundamental limitation of VLA models trained purely on demonstration data: the inability to handle compounding errors and improve from deployment experience.

The concurrent emergence of both works from academia and industry highlights a growing recognition that experience-based reinforcement learning is essential for VLA models to move beyond behavior cloning. Both approaches demonstrate that achieving reliable and robust performance requires learning from the robot’s own experience rather than solely imitating expert demonstrations. We submitted EVOLVE-VLAbefore the release of 𝜋∗0.6, representing pioneering academic work in this direction. To foster further research and democratize access to this paradigm, we commit to releasing our full training and inference codebase upon publication.

[Figure 47]

[Figure 48]

Rollout Generation

Task Progress Estimation

[Figure 49]

[Figure 50]

Environment

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

😿Inaccurate long-horizon comparison

…

Progress: 42%

[Figure 55]

[Figure 56]

[Figure 57]

###### Timeline

User Instruction

[Figure 58]

[Figure 59]

[Figure 60]

Δ_milestone Ini al Milestone

…

Δ_check

Progress: 87%

“Turn on the stove and put the moka …”

[Figure 61]

💪 Against Nearest Milestone

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

🔥

…

VLA

Progress: 61%

###### Progress Accumula on

GRPO Update Progress as Dense Reward

+9%

0% 23% 40%

Exploration Horizon

𝑣 = 𝑣 + 100 − 𝑣 × 𝑐 /100

[Figure 66]

Progressive Horizon Extension

Training Progress

- Figure 2 | Framework overview. In the test-time training phase, a VLA model interacts with the environment and generates diverse rollout trajectories. A task progress estimation module assign a progress for each rollout, which will be utilized as the reward for GRPO optimization. Task progress estimation employs an accumulative strategy that produces clean, stable and smooth reward. The training strategy undergoes a progressive horizon extension schedule, enabling a learning curriculum.
- 3. Method

###### 3.1. Task Definition

We formulate the robotic manipulation task as a Markov Decision Process (MDP) M = (S, A, 𝑃, R, 𝛾), where S is the state space, A is the action space, 𝑃 represents transition dynamics, R is the reward function, and 𝛾 ∈ [0,1) is the discount factor. At timestep 𝑡, the state 𝑠𝑡 = (𝑜vis𝑡 , 𝑜prop𝑡 , 𝑙task) consists of visual observation 𝑜vis𝑡 , proprioceptive state 𝑜prop𝑡 , and task instruction 𝑙task.

A VLA policy 𝜋𝜃 : S → Δ(A) maps states to action distributions. Following modern VLA architectures [3, 12], we adopt action tokenization where continuous robot actions 𝑎𝑡 ∈ R𝑑 are discretized into tokens. The policy autoregressively generates action token sequences a = (𝑎1, . . . , 𝑎𝑇) with probability 𝜋𝜃(a | 𝑠𝑡) = 𝑇𝑘=1 𝜋𝜃(𝑎𝑘 | 𝑠𝑡, 𝑎<𝑘). A trajectory 𝜏 = {(𝑠0, 𝑎0), . . . , (𝑠𝐻, 𝑎𝐻)} is generated through closed-loop interaction: the policy outputs actions, the environment transitions based on physical dynamics, and updated observations feed back into the policy until task completion or maximum horizon 𝐻.

###### 3.2. Test-time Training Framework

During deployment, a VLA model pretrained via SFT on expert demonstrations encounters novel scenarios that differ from its training distribution. Traditional SFT models, which learn purely through imitation, lack the mechanism to adapt to these out-of-distribution states. Our goal is to enable the VLA to continue learning at test-time by leveraging online interaction with the environment.

Test-time training (TTT) requires two key components: (1) the ability to actively interact with the environment to generate diverse rollouts, and (2) a feedback signal to evaluate and improve these rollouts. We achieve this through online reinforcement learning, where the policy is iteratively refined based on rewards obtained from environment interaction. Fig. 2 shows the overview of our TTT framework.

###### 3.2.1. Online Reinforcement Learning

Interactive Rollout Generation. For a given task, we generate multiple diverse trajectories by sampling from the policy’s action token distribution with temperature 𝑇 > 1. Specifically, starting from initial state 𝑠0, at each timestep 𝑡, the policy outputs action token probabilities and samples an action 𝑎𝑡 from the distribution. This action is executed in the environment, producing a new state 𝑠𝑡+1. This closed-loop interaction continues until the estimated task progress exceeds a threshold (indicating completion) or the maximum horizon 𝐻max is reached, yielding a trajectory 𝜏𝑖 = {(𝑠0, 𝑎0), . . . , (𝑠𝐻, 𝑎𝐻)}. By sampling 𝐺 trajectories {𝜏𝑖}𝐺𝑖=1 with different random seeds, we explore diverse solution strategies.

Environment Feedback. Each trajectory receives a reward 𝑅𝑖 that evaluates its quality. This reward signal, which we detail in §3.2.2, serves as the supervisory feedback guiding policy improvement. Unlike SFT which only learns from successful demonstrations, the reward signal provides differential feedback, distinguishing better trajectories from worse ones and enabling the model to discover and reinforce effective behaviors through trial and error.

Policy Update. We employ Group Relative Policy Optimization (GRPO) [21] to update the policy. GRPO normalizes trajectory rewards within each batch to compute advantages and applies PPO-style clipping for stable updates, without requiring a separate value network.

###### 3.2.2. Task Progress Estimation

A critical challenge for test-time training is the absence of oracle reward signals (e.g., groundtruth success indicators from simulators) that are available during training in simulator but unavailable at deployment. We address this by learning a reward function based on task progress: an estimate of how much of the task has been completed.

Task Progress as Reward Function. Progress-based rewards offer several advantages over binary success signals. First, they are dense: progress can be estimated at any point during execution, providing continuous feedback even for failed attempts. This density is crucial for sample-efficient learning, especially in long-horizon tasks where successful rollouts may be rare initially. Second, progress is a more general, grounded concept than task-specific metrics or black-box reward scores, making it applicable across diverse manipulation tasks.

Task Progress as Termination Condition. Beyond providing rewards, task progress estimation also determines when to terminate rollouts. When estimated progress exceeds a predefined threshold, the rollout stops as the task is deemed complete; otherwise, execution continues until maximum horizon 𝐻max. This dual-purpose usage imposes stringent requirements on the estimator: it must be (1) computationally efficient, as it is queried frequently (every Δcheck steps) to detect completion in real-time, and (2) temporally smooth and consistent, as erratic estimates can cause premature termination (stopping promising trajectories early) or delayed termination (wasting computation on completed tasks). While noisy rewards can be mitigated through averaging during policy learning, a single erroneous termination decision can truncate an entire trajectory. Therefore, stabilizing the progress signal is essential not just for learning efficiency, but for correct rollout execution.

Vanilla Progress Estimation. We employ a foundation critic model, VLAC [29], which takes two images and task instruction as input and output a critic value. A positive value indicates how much the second image progresses the task compared to the first image. A negative value vice versa. Specifically, given a trajectory 𝜏 = {(𝑠0, 𝑎0), . . . , (𝑠𝐻, 𝑎𝐻)}, we compute the reward as 𝑅𝑖 = Critic(𝑜0, 𝑜𝐻, 𝑙task), where 𝑜0 and 𝑜𝐻 are the initial and final observations of the trajectory, and 𝑙task is the task instruction. The estimated reward is then normalized to [0,1] to serve as the trajectory reward 𝑅𝑖 for GRPO.

###### 3.3. Accumulative Progress Estimation

While the progress critic provides dense feedback, we observe that it can be noisy and inconsistent, especially for long-horizon tasks involving multiple sub-goals. A single frame-pair comparison may be misled by superficial visual changes or fail to capture intermediate progress. As discussed in §3.2.2, this noisy estimation can negatively affect both reward feedback and rollout termination.

To address these challenges, we introduce an accumulative progress estimation mechanism. Our key insight is inspired by a slow-fast philosophy: instead of comparing the final state to the very beginning (which becomes unreliable for long trajectories), we maintain milestone frames at regular intervals and compute progress incrementally.

Interval-Based Milestone Sampling. We define a sampling interval Δmilestone (e.g., 64

timesteps). During rollout, we maintain a list of milestone frames Fmilestone = { 𝑓0, 𝑓Δmilestone, 𝑓2Δmilestone, . . .} that captures the trajectory’s evolution at a coarse granularity. These milestones serve as refer-

ence points for measuring progress.

Incremental Progress Computation. At a finer granularity (every Δcheck steps, where Δcheck < Δmilestone), we query the critic to estimate progress relative to the most recent milestone. Specifically, at timestep 𝑡, we compute:

𝑐𝑡 = Critic( 𝑓⌊𝑡/Δmilestone⌋·Δmilestone, 𝑜𝑡), (1)

where 𝑐𝑡 ∈ [−100,100] represents the incremental progress from the last milestone to the current state. When 𝑡 reaches a new milestone (𝑡 mod Δmilestone = 0), we append 𝑜𝑡 to Fmilestone and store 𝑐𝑡 in the critic history.

Accumulative Value Aggregation. Given a sequence of incremental critic values

{𝑐Δmilestone, 𝑐2Δmilestone, . . . , 𝑐𝑘Δmilestone} collected at milestones, we accumulate them into a progress value 𝑣𝑡 ∈ [0,100] that estimates task completion percentage:

𝑣𝑖 = 𝑣𝑖−1 + (100 − 𝑣𝑖−1) · 𝑐𝑖/100, 𝑣0 = 0, (2)

where 𝑖 indexes the milestones. This recursive formulation applies a diminishing returns principle: positive progress advances the value toward 100 by a fraction of the remaining distance, while negative critics decrease the value proportionally. Critically, adjustments scale with (100 − 𝑣𝑖−1) (the remaining gap to completion) prevents both overshooting from overly optimistic critics and catastrophic collapse from pessimistic ones.

The full mechanism is shown in Algorithm 1. It effectively smooths the noisy critic: by comparing to recent milestones rather than the distant initial state, we reduce the impact of long-term drift; by applying proportional adjustments rather than raw critic values, we create a more stable learning signal; and by accumulating progress incrementally with diminishing returns, we smooth out local fluctuations. Such a smoothed reward provides more reliable feedback for the reinforcement optimization.

In addition, this mechanism is also computationally efficient. Recall that since the progress need to called frequently for determining rollout termination, at timestep 𝑇, a naive multi-frame approach would require 𝑇 − 1 critic calls to evaluate all pairwise comparisons, whereas our method requires only a single call—comparing the current frame to the nearest milestone.

###### 3.4. Progressive Horizon Extension

Long-horizon tasks present a fundamental challenge for test-time training: early in training, the policy is far from proficient and successful task completion is rare, making credit assignment

Algorithm 1: Accumulative Progress Estimation

Input: Critic model, Δmilestone, Δcheck, 𝜏threshold Output: Trajectory reward 𝑅accum(𝜏) Initialize milestone frames Fmilestone ← [𝑜0]; Initialize critic history C ← []; Initialize progress values V ← [0]; Initialize 𝑣current ← 0; for 𝑡 = 1 to 𝐻max do

Execute action 𝑎𝑡, observe 𝑜𝑡; if 𝑡 mod Δcheck = 0 then

𝑘 ← ⌊𝑡/Δmilestone⌋ ; // Nearest milestone 𝑓ref ← Fmilestone[𝑘] ; // milestone frame 𝑐𝑡 ← Critic( 𝑓ref,𝑜𝑡) ; // Compute incremental progress if 𝑡 mod Δmilestone = 0 then

Append 𝑜𝑡 to Fmilestone ; // New milestone Append 𝑐𝑡 to C ; // Store critic value // Accumulate progress value with diminishing returns

𝑣current ← 𝑣current + (100 − 𝑣current) · 𝑐𝑡/100; Append 𝑣current to V;

##### // Termination check

if 𝑣current/100 > 𝜏threshold then

##### break ; // Task deemed complete // Use accumulated progress as reward

𝑅accum(𝜏) ← 𝑣current/100; return 𝑅accum(𝜏)

difficult with noisy reward signals. Simply allowing free exploration until the maximum horizon 𝐻max leads to low-quality trajectories that provide weak learning signals. Even with our accumulative progress estimation, optimizing over very long horizons from the start can lead to unstable learning dynamics.

To address this, we adopt a progressive horizon extension strategy. We divide the training process into stages, where each stage operates with a maximum rollout horizon 𝐻max. As training progresses through stages, we gradually increase 𝐻max, allowing the policy to first master shorter sub-goals before tackling the complete task. In early stages, the agent focuses on immediate objectives and fundamental manipulation behaviors where the reward signal is cleaner and more direct. As the horizon extends in later stages, the policy learns to chain these behaviors together and reason over longer temporal dependencies, ultimately optimizing complete task execution.

This schedule provides several benefits. First, shorter horizons naturally reduce the accumulation of noise in progress estimation, as fewer milestone comparisons are needed. Second, early success on simpler sub-goals provides positive learning signals that would be absent when optimizing full-length trajectories from scratch. Third, the staged progression allows the policy to build compositional skills, where early stages establish robust primitives, while later stages learn to orchestrate them.

Importantly, progressive learning and accumulative progress estimation are complementary mechanisms. The progressive curriculum addresses temporal credit assignment, i.e., determining when and what to learn, by controlling the optimization scope. Accumulative estimation addresses noisy rewards, i.e., stabilizing the feedback signal, by aggregating incremental progress. Together, they enable robust test-time training on long-horizon manipulation tasks where both challenges are present.

- Table 1 | Main results of different VLA models on LIBERO.

LIBERO Spatial Object Goal Long Avg Octo 78.9 85.7 84.6 51.1 75.1 OpenVLA 84.7 88.4 79.2 53.7 76.5 Nora 92.2 95.4 89.4 74.6 87.9 𝜋0 + FAST 96.4 96.8 88.6 60.2 85.5 𝜋0 96.8 98.8 95.8 85.2 94.2 UniVLA 96.5 96.8 95.6 92.0 95.2 VLA-RL 90.2 91.8 82.2 59.8 81.0 SimpleVLA† 94.3 90.5 92.3 87.7 91.2 OpenVLA-OFT 91.3 90.1 89.8 85.8 89.2 EVOLVE-VLA 95.4 97.4 95.8 94.4 95.8

Model

Δ +4.1 +7.3 +6.0 +8.6 +6.5

#### 4. Experiments

Benchmark. We evaluate our method on the LIBERO benchmark [14], a widely used simulation benchmark for lifelong learning in robotic manipulation. LIBERO focuses on language-guided manipulation tasks across diverse object types, task specifications, and environments. It consists of four task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. Each suite contains 10 tasks, with 50 expert demonstrations per task. We report the average Success Rate (SR) across 50 trials for each task, following the evaluation protocol in previous work [12, 13].

Base Model. We apply our method to OpenVLA-OFT [11], a state-of-the-art autoregressive VLA model that achieves high performance and inference efficiency. Following prior work [13], we adopt the action chunking and parallel decoding designs, while disabling the continuous action regression head, i.e., use discrete action tokens instead. This would enable action generation compatible with the optimization of reinforcement learning.

Reward Model. For test-time training, we employ a foundation critic model VLAC [29] as the progress estimator. VLAC takes two images and a language instruction as input and outputs a critic value indicating how much the second image represents progress toward task completion compared to the first image. This foundation model has been pre-trained on largescale robotic manipulation datasets, demonstrating its ability to estimate task progress across diverse tasks and environments.

###### 4.1. Main Results

Tab. 1 presents our main results on the LIBERO benchmark, comparing our TTT framework against state-of-the-art VLA models. We apply TTT to the OpenVLA-OFT model (pre-trained with full trajectory demonstrations), enabling it to continue learning during deployment.

Significant Performance Gains. Our TTT framework achieves substantial improvements across all four LIBERO task suites. On average, we observe a +6.5% absolute gain in success rate, elevating the baseline from 89.2% to 95.8%. The improvements are consistent across diverse task types: +4.1% on LIBERO-Spatial, +7.3% on LIBERO-Object, +6.0% on LIBERO-Goal, and most notably, +8.6% on LIBERO-Long. The substantial gain on LIBERO-Long is particularly significant, as this suite contains the most challenging long-horizon tasks with complex multistep procedures. With TTT, our method achieves 95.8% average success rate, surpassing models like 𝜋0 (94.2%) and matching UniVLA (95.2%), demonstrating that test-time adaptation can be

- Table 2 | Test-time training in low data regime, i.e., one demonstration for pre-training.

LIBERO Spatial Object Goal Long Avg

Model

OpenVLA-OFT 65.1 40.1 57.2 15.1 43.6 EVOLVE-VLA 73.4 70.0 64.7 37.1 61.3

Δ +8.3 +29.9 +7.5 +22.0 +17.7

as effective as collecting and training on large amounts of additional demonstration data.

Challenge of Naive Reward Modeling. We also compare with SimpleVLA, which initially employs the binary outcome reward from the simulator, i.e., oracle reward. We then replace the oracle reward with our progress estimator, and use a simple threshold-based approach to convert progress estimates into binary outcome rewards. This version of SimpleVLA achieves only 87.7% on LIBERO-Long, a modest +1.9% improvement over the SFT-only baseline (85.8%). The limited gain highlights a critical challenge: directly using a noisy progress estimator to generate binary rewards for online RL is insufficient. In contrast, our accumulative progress estimation mechanism that smooths noisy signals and provides dense, stable feedback achieves 94.4% (+8.6%), demonstrating the importance of properly “taming” the reward model.

###### 4.2. TTT Under Low Data Regimes

A key motivation for TTT is to reduce the labor cost of collecting extensive demonstration data. To evaluate TTT’s effectiveness in low-data scenarios, we experiment with a more challenging setting: only one demonstration per task for SFT pre-training1, followed by test-time training.

As shown in Tab. 2, the 1-shot SFT baseline (OpenVLA-OFT) achieves only 43.6% average success rate, indicating that a single demonstration is insufficient for learning robust manipulation policies. However, applying our TTT framework yields substantial improvements, achieving 61.3% average success rate, which is a remarkable +17.7% absolute gain. The improvements are consistent across all task suites: +8.3% on LIBERO-Spatial, +29.9% on LIBERO-Object, +7.5% on LIBERO-Goal, and +22.0% on LIBERO-Long. These gains validate our core claim: test-time training can effectively alleviate the data collection burden by enabling learning from self-generated experiences rather than relying solely on extensive expert demonstrations.

###### 4.3. Toward Zero-Shot Cross-Task Generalization

An intriguing capability enabled by our TTT framework is cross-task generalization through online learning. To explore this, we conduct a preliminary experiment: we take a VLA model pre-trained exclusively on LIBERO-Long tasks (50 demonstrations per task) and directly deploy it on LIBERO-Object tasks without fine-tuning on task-specific demonstrations.

When deployed directly, the LIBERO-Long pre-trained policy achieves 0% success rate on LIBERO-Object, as expected. Although, conceptually, the two task suites may share some common motion primitives, the behavior cloning paradigm strongly hinders generalization. Remarkably, however, by applying our TTT framework with progress-based feedback, the policy adapts purely through autonomous exploration and reaches 20.8% success rate on LIBEROObject. While this performance remains modest compared to task-specific SFT baselines (which achieve 40.1% with a single demonstration and 96.6% with 50 demonstrations), the ability to break 0 success rate without finetuning on task-specific human demonstrations represents a

1All 1-trajectory SFT models are reused from the SimpleVLA-RL released checkpoints [13].

- Table 3 | Ablation study on accumulative progress estimation and and temporal sampling efficiency on LIBERO-Long task suite.

Method Sampling Reward Calls F-Score SR (%)

SFT - - - 85.8 Baseline (2 frames) - 32 0.04 88.3

Accumulative (4 frames) Uniform 96 0.09 90.1 Accumulative (8 frames) Uniform 224 0.17 89.3 Accumulative (Ours) Interval 32 0.20 91.3

qualitatively different capability. To the best of our knowledge, no prior VLA training method has demonstrated such cross-task transfer through test-time adaptation alone. This preliminary result suggests that TTT, when paired with a foundation-level progress estimator like VLAC, can enable VLAs to generalize across task distributions through self-directed learning.

###### 4.4. Ablation Studies

Accumulative Progress Estimation. Tab. 3 validates the effectiveness and efficiency of our accumulative progress estimation mechanism with various frame sampling strategies. F-Score is computed based on a balanced validation set (100 success cases, 100 failure cases) assessing task progress estimation performance. The baseline that directly uses 2-frame critic values without accumulation achieves 88.3% success rate with 32 reward calls but suffers from low F-score (0.04), indicating unreliable progress estimation. When incorporating accumulative progress estimation, the sampling strategy for millstone frames matters. The uniform sampling variants improve F-score over baseline but at the cost of significantly more reward calls (96 and 224 respectively), with diminishing or even negative returns in success rate, suggesting that naive dense sampling introduces noise without proper temporal structure. In contrast, our method achieves the best performance (91.3% SR, 0.20 F-score) while maintaining computational efficiency with only 32 reward calls, demonstrating that interval-based sampling (with a sliding Δ𝑐ℎ𝑒𝑐𝑘) combined with accumulative aggregation is both more effective and more efficient than naive uniform approaches.

Progressive Horizon Extension. Tab. 4 demonstrates the importance of progressive horizon extension for long-horizon tasks. Starting from the SFT baseline (85.8%), we examine three TTT variants. First, using binary outcome rewards (thresholding the progress estimator) yields only 87.7% (+1.9%), confirming that converting dense progress into sparse signals loses valuable learning information. Second, applying dense rewards from our accumulative progress estimator without progressive horizon achieves 91.3% (+5.5%), showing the benefit of dense feedback. Finally, adding progressive horizon extension, i.e., gradually increasing the maximum rollout length during training, reaches 94.4% (+8.6%), providing an additional 3.1% gain. This validates our strategy: by initially constraining exploration to shorter horizons and progressively extending them, the policy learns more stable sub-task skills before tackling full-length trajectories, making it more resilient to estimation errors in long-horizon tasks.

###### 4.5. Qualitative Analysis

To gain deeper insights into how test-time training shapes policy behavior, we analyze representative rollout trajectories after TTT in Fig. 3. First, the policy develops error recovery capabilities: when initial grasp attempts fail, the SFT-only policy usually continue with the pre-programmed motion and fails, whereas after TTT the policy autonomously re-attempts grasping (top row).

- Table 4 | Ablation study on Progressive Horizon Extension with LIBERO-Long task suite.

###### Method Milestones SR (%)

Baseline (SFT only) - 85.8 + Binary Outcome - 87.7 + Dense Reward (Vanilla Critic) - 91.3 + Progressive Horizon ✓ 94.4

Second, the policy adapts to pick an object but accidentally changed the object state, then it adjusts its motion to fit the new config rather than rigidly following memorized patterns (middle row). Third, the policy discovers alternative manipulation strategies not present in demonstrations. For instance, grasping a pot by its body instead of the handle (bottom row). These improvements indicate that progress-based feedback enables the policy to generalize beyond trajectory-level imitation to goal-oriented manipulation and explore diverse solutions.

EVOLVE-VLA	can recover	from	unsuccessful grasp.

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

EVOLVE-VLA can	handle object state change caused	by accidental touch.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

EVOLVE-VLA grows novel manipulation	pattern.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

###### Figure 3 | Qualitative example of policy behavior with TTT.

Despite the improvements, we observe failure cases that reveal a fundamental challenge: misalignment between the environment’s rule-based success criterion and the semantic task completion assessed by our progress estimator. This mismatch manifests in two ways as shown in Fig. 4. First, in some cases the policy brings the scene very close to the goal state, leading the progress estimator to assign high rewards (near-completion signal), yet the environment’s coordinate-based rules still judge the task as unsuccessful. This creates a form of “reward hacking” where the policy optimizes for high progress scores without meeting the strict environmental criteria. Second, the opposite occurs: the environment judges tasks as successful based on coordinate rules despite semantic incompleteness. For instance, in Fig. 4, a book placement task where the environment reports success because the book’s coordinates satisfy the spatial constraints, yet semantically the book is not properly placed inside the shelf. These

###### misalignments highlight the inherent difficulty in aligning rule-based simulation criteria with semantic task understanding, suggesting that future work should explore improved calibration between progress estimators and environment oracles.

Task: put both the alphabet soup and the tomato sauce in the basket.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Fail

Reward Hacking

Progress 97%

[Figure 84]

Task: turn on the stove and put the moka pot on it.

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Success

Flawed Environment Criteria

Progress 88%

[Figure 90]

- Figure 4 | Example of environment success criterion mismatch.
- 5. Future Work

Our work demonstrates that test-time training for VLA models is feasible by addressing noisy progress estimation through accumulative estimation and progressive horizon extension. This foundational step enables VLAs to learn from experience rather than merely imitating demonstrations, unlocking several promising directions for future research. First, developing more robust reward models would significantly enhance the framework’s capabilities. While our accumulative mechanism effectively handles noisy estimates, future reward models with better semantic alignment to environment success criteria could reduce the mismatch between progress estimation and rule-based success signals. Additionally, improving zero-shot capability would eliminate the need for in-context examples, enabling truly zero-shot cross-task generalization. While our current approach shows promising cross-task transfer (LIBERO-Long → LIBEROObject), the reward model still benefits from task-specific context; future reward models trained on more diverse manipulation data, with better generalization capability could enable seamless adaptation to entirely novel tasks without any task-specific examples, even for the reward model.

Second, extending test-time training to real-world robotic deployment presents both opportunities and challenges. The long training times required for online RL can be prohibitive in physical environments, where data collection is inherently slower than simulation. Future work can explore techniques to accelerate real-world training, such as sim-to-real transfer for reward models, parallel robot deployment for distributed data collection, or more sample-efficient online learning algorithms. Equally important is ensuring safety during exploration: the uncontrolled policy behavior in early training stages could damage the robot or environment. Developing safety mechanisms—such as action constraints, safety critics, or human oversight protocols—would be crucial for enabling safe autonomous learning in physical environments. Third, exploring more sophisticated exploration strategies and curriculum designs could further improve sample efficiency and enable adaptation to even more complex, long-horizon manipulation tasks.

#### 6. Conclusion

We introduced EVOLVE-VLA, a test-time training framework that enables VLA models to continuously adapt through environment interaction, addressing the fundamental limitations of static SFT. Inspired by how humans develop manipulation skills through practice and trial-and-error, our approach shifts VLAs from rigid trajectory memorization toward genuine adaptive learning. By replacing impractical oracle rewards with a learned progress estimator and introducing two key technical contributions: (1) accumulative progress estimation and (2) progressive horizon extension, we demonstrate that VLAs can effectively learn from inherently noisy, self-generated feedback signals. Our experiments on the LIBERO benchmark validate this approach, achieving +8.6% on long-horizon tasks, +22.0% in 1-shot learning, and enabling crosstask generalization (0% → 20.8%) without task-specific demonstration training. Beyond these quantitative gains, we observe emergent capabilities like error recovery that arise purely from autonomous exploration. We believe this work represents an essential step on the path toward truly general-purpose VLA systems that can continuously learn and improve in real-world deployment.

#### References

- [1] A. Amin et al. 𝜋∗0.6: A VLA that Learns from Experience. https://www.physicalintelligence. company/blog/pistar06. Physical Intelligence Blog, November 2025. 2025.
- [2] J. Bi et al. “Vla-touch: Enhancing vision-language-action models with dual-level tactile feedback”. In: arXiv preprint arXiv:2507.17294 (2025).
- [3] K. Black et al. “pi0: A Vision-Language-Action Flow Model for General Robot Control”. In: arXiv preprint arXiv:2410.24164 (2024).
- [4] A. Brohan et al. “Rt-1: Robotics transformer for real-world control at scale”. In: arXiv preprint arXiv:2212.06817 (2022).
- [5] T. Brown et al. “Language models are few-shot learners”. In: NeurIPS (2020).
- [6] K. Chen et al. “pi RL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models”. In: arXiv preprint arXiv:2510.25889 (2025).
- [7] P. Ding et al. “Quar-vla: Vision-language-action model for quadruped robots”. In: European Conference on Computer Vision. Springer. 2024, pp. 352–367.
- [8] C. Gao et al. “OctoNav: Towards Generalist Embodied Navigation”. In: arXiv preprint arXiv:2506.09839 (2025).
- [9] H. Guo et al. “OmniVLA: Unifiying Multi-Sensor Perception for Physically-Grounded Multimodal VLA”. In: arXiv preprint arXiv:2511.01210 (2025).
- [10] Y. Guo et al. “Improving Vision-Language-Action Model with Online Reinforcement Learning”. In: arXiv preprint arXiv:2501.16664 (2025).
- [11] M. J. Kim, C. Finn, and P. Liang. “Fine-tuning vision-language-action models: Optimizing speed and success”. In: arXiv preprint arXiv:2502.19645 (2025).
- [12] M. J. Kim et al. “Openvla: An open-source vision-language-action model”. In: arXiv preprint arXiv:2406.09246 (2024).
- [13] H. Li et al. “Simplevla-rl: Scaling vla training via reinforcement learning”. In: arXiv preprint arXiv:2509.09674 (2025).
- [14] B. Liu et al. “Libero: Benchmarking knowledge transfer for lifelong robot learning”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 44776–44791.

- [15] H. Liu et al. “Visual instruction tuning”. In: Advances in neural information processing systems 36 (2023), pp. 34892–34916.
- [16] J. Liu et al. “What Can RL Bring to VLA Generalization? An Empirical Study”. In: arXiv preprint arXiv:2505.19789 (2025).
- [17] G. Lu et al. “VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning”. In: arXiv preprint arXiv:2505.18719 (2025).
- [18] Y. J. Ma et al. “Liv: Language-image representations and rewards for robotic control”. In: International Conference on Machine Learning. PMLR. 2023, pp. 23301–23320.
- [19] Y. J. Ma et al. “Vision language models are in-context value learners”. In: The Thirteenth International Conference on Learning Representations. 2024.
- [20] D. Qu et al. “Spatialvla: Exploring spatial representations for visual-language-action model”. In: arXiv preprint arXiv:2501.15830 (2025).
- [21] Z. Shao et al. “Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024”. In: URL https://arxiv. org/abs/2402.03300 2.3 (2024), p. 5.
- [22] S. Sontakke et al. “Roboclip: One demonstration is enough to learn robot policies”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 55681–55693.
- [23] O. M. Team et al. “Octo: An open-source generalist robot policy”. In: arXiv preprint arXiv:2405.12213 (2024).
- [24] H. Touvron et al. “LLaMA: Open and efficient foundation language models”. In: arXiv preprint arXiv:2302.13971 (2023).
- [25] P. Wang et al. “Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution”. In: arXiv preprint arXiv:2409.12191 (2024).
- [26] J. Wen et al. “DiffusionVLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression”. In: Forty-second International Conference on Machine Learning. 2025.
- [27] J. Wen et al. “Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation”. In: IEEE Robotics and Automation Letters (2025).
- [28] S. Yang et al. “BiTLA: A Bimanual Tactile-Language-Action Model for Contact-Rich Robotic Manipulation”. In: Proceedings of the 1st International Workshop on Multi-Sensorial Media and Applications. 2025, pp. 12–17.
- [29] S. Zhai et al. “A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning”. In: arXiv preprint arXiv:2509.15937 (2025).
- [30] J. Zhang et al. “ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations”. In: arXiv preprint arXiv:2505.10911 (2025).
- [31] Q. Zhao et al. “Cot-vla: Visual chain-of-thought reasoning for vision-language-action models”. In: Proceedings of the Computer Vision and Pattern Recognition Conference. 2025, pp. 1702–1713.

