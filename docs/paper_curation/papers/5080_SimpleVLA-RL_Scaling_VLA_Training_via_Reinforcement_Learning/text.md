SimpleVLA-RL Technical Report 2025-09-12

# SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning

###### Haozhan Li*,1, Yuxin Zuo*,1, Jiale Yu*,1, Yuhao Zhang*,3, Zhaohui Yang3, Kaiyan Zhang1, Xuekai Zhu1, Yuchen Zhang4, Tianxing Chen5, Ganqu Cui2, Dehui Wang3, Dingxiang Luo3, Yuchen Fan3, Youbang Sun1, Jia Zeng2, Jiangmiao Pang2, Shanghang Zhang4, Yu Wang1, Yao Mu2,3, Bowen Zhou†,1,2 and Ning Ding†,1,2

1Tsinghua University, 2Shanghai AI Lab, 3Shanghai Jiao Tong University, 4Peking University, 5The University of Hong Kong ∗Equal Contributions, †Corresponding Authors

zhan72426@gmail.com, dingning@mail.tsinghua.edu.cn, PRIME-RL/SimpleVLA-RL

## arXiv:2509.09674v1[cs.RO]11Sep2025

Abstract | Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms 𝜋0 on RoboTwin 1.0&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon “pushcut” during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process.

Data Scarcity

###### Simulation & Real-World Tasks

|SimpleVLA-RL SFT-only #" RDT<br><br>|
|---|

91.7

[Figure 1]

LIBERO

###### RoboTwin2.0

RoboTwin1.0

[Figure 2]

[Figure 3]

Full-Trajectory SFT: 86.5

99.1 96.9

###### 100

94.2

[Figure 4]

[Figure 5]

80

SuccessRate(%)

###### SimpleVLA-RL

70.4

68.8

+74.4 (↑430.1%)

Real World Tasks

60

49.2

70 60

SuccessRate(%)

91.0

60

40

39 24

One-Trajectory SFT: 17.3

48.9

39.8

20

38.3

20

14 10

###### 10

4

Task: Move Can Pot

0

Avg

Task1 Task2 Task3 Task4

Full

One Full

Full

[Figure 6]

[Figure 7]

###### Generalization

New Pattern Discovery: Pushcut Phenomenon

Grasp

SFT

###### UnseenTaskSR(%)

###### SFT Model After SimpleVLA-RL

[Figure 8]

[Figure 9]

[Figure 10]

Task: Move Can Pot

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

###### Spatial Object Goal

Push

Grasp

RL

SFT

Seen Task SR(%) Seen Task SR(%) Seen Task SR(%)

[Figure 15]

[Figure 16]

RL enables the policy to acquire novel actions never seen before. RL provides strong, multi-dimensional generalization capabilities!

Push

RL

Figure 1 | Overview of SimpleVLA-RL. SimpleVLA-RL is an efficient RL framework for VLA that improves long-horizon planning under data scarcity, outperforms SFT in simulation and real-world tasks, reveals a “pushcut” new-action phenomenon, and strengthens spatial/object/goal generalization.

###### Contents

- 1 Introduction 3
- 2 Preliminaries 4

- 2.1 RL Formulation for LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 RL Formulation for VLAs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 Group Relative Policy Optimization . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 3 SimpleVLA-RL 6

- 3.1 Interactive VLA Rollout . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.2 Outcome Reward Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.3 Exploration Enhancements . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.4 Training Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 4 Experiments 9

- 4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 4.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 5 Analysis 12

- 5.1 Overcoming Data Scarcity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 5.2 Generalization Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 5.3 Real-World Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 6 Discussions 15

- 6.1 “Pushcut”: Emergence of New Patterns through RL . . . . . . . . . . . . . . . . . . . 15
- 6.2 Failure Modes of SimpleVLA-RL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 7 Related Works 17

- 7.1 Reinforcement Learning for Large Language Models . . . . . . . . . . . . . . . . . . . 17
- 7.2 Vision Language Action Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 8 Conclusion 18

###### 1. Introduction

Vision-Language-Action (VLA) models have emerged as a promising paradigm for enabling robots to solve diverse and challenging manipulation tasks in physical environments [Firoozi et al., 2025]. While these models demonstrate considerable potential, their development entails substantial complexity, as they necessitate the unification of visual perception, language understanding, and action generation within a unified framework [Kim et al., 2024; Zhong et al., 2025]. To this end, previous works have typically adopted a two-stage training strategy: first conducting large-scale pretraining on abundant multimodal data with human manipulation videos [Sapkota et al., 2025], image-text pairs, and heterogeneous robot datasets [O’Neill et al., 2024], followed by supervised fine-tuning (SFT) on extensive high-quality robot trajectories to enhance task-specific capabilities.

This paradigm has achieved substantial progress, particularly through large-scale pretraining. However, as model training scales further, especially when requiring scaled SFT for downstream task optimization, several critical challenges have emerged: 1) Data Scarcity: Scaling SFT necessitates a substantially larger amount of human-operated robot trajectories, which remain both scarce and prohibitively expensive, posing significant challenges to scalability. The collection of robotic trajectories demands meticulously designed experimental scenarios, diverse manipulation objects, and skilled operators, which severely constrains both data scale and diversity [Mandlekar et al., 2021; Bu et al., 2025a; Gao et al., 2024; Team et al., 2025a]. 2) Poor Generalization: Generalization remains a crucial bottleneck for VLA, particularly on compositional, long-horizon, or real-world tasks involving distribution shift. This challenge is particularly evident because the current SFT of VLAs typically relies on limited, scene- and task-specific data. Consequently, when VLA models encounter unseen tasks, environments, or objects, their performance inevitably degrades [Liu et al., 2025a].

Recent advances in Large Reasoning Models (LRMs), such as DeepSeek-R1 [Guo et al., 2025a], have demonstrated that reinforcement learning (RL) can drive remarkable progress even when relying solely on outcome rewards. These findings highlight the crucial role of RL in enhancing models’ ability to perform step-by-step chain-of-thought (CoT) reasoning [Team et al., 2025b; Zeng et al., 2025; Yang et al., 2025]. This naturally raises a question: Can RL also strengthen VLA models’ capacity to generate accurate actions step by step? This insight motivates us to explore reproducing the success of DeepSeek-R1 in VLA models, which may also help overcome the aforementioned two challenges of SFT. However, applying RL to VLAs actually presents several unique challenges. Traditional RL methods in robotic tasks typically rely on hand-crafted process rewards, which severely limit scalability [Ibarz et al., 2021; Kroemer et al., 2021; Ma et al., 2023]. Moreover, unlike the rollout of LLMs, VLAs require multi-round interactions with the environment. It is not only slower but also substantially more costly.

We introduce SimpleVLA-RL, an effective RL framework for VLA models. Building upon Volcano Engine Reinforcement Learning for LLMs (veRL)1 [Sheng et al., 2024], a general-purpose RL framework for LLMs, we enable end-to-end online rule-based RL for VLA models through the implementation of VLA-specific interactive trajectory sampling and loss computation. To further support scalable RL for VLA models, we extend veRL with parallel multi-environment rendering for faster sampling, and adapt it into an integrated training–inference–rendering framework. Surprisingly, we observe that the policy discovers previously unseen patterns during RL training, extending beyond those encountered in supervised data. We refer to this novel phenomenon as “pushcut”. Our main contributions include:

• Efficient online RL framework for VLA: We develop an efficient end-to-end VLA online RL framework based on veRL that enables stable, sample-efficient training, optimized for rendering parallelization and distributed training & inference.

1https://github.com/volcengine/verl

- • SoTA performance: We incorporate exploration-enhancing strategies, yielding consistent performance improvements of 10–15%. Moreover, SimpleVLA-RL surpasses multiple SoTA baselines on both LIBERO and RoboTwin 1.0 & 2.0.
- • Data efficiency and generalization: With only a single demonstration per task, RL boosts LIBERO-Long success rates from 17.1% to 91.7%, and significantly outperforms SFT in spatial, object, and task generalization.
- • Real-world deployment capability: Simulation-trained policies transfer effectively to realworld, achieving strong sim-to-real performance gains without requiring any real robot data.

###### 2. Preliminaries

To provide an intuitive illustration of the existing gap when extending RL methodologies from LLMs to the VLA domain, we formalize RL for both LLMs and VLA models, presenting their state representations, action spaces, reward functions, and environments in this section.

- 2.1. RL Formulation for LLMs State (𝑠𝑡): At step 𝑡, the state 𝑠𝑡 comprises the input prompt and previously generated tokens:

𝑠𝑡 = (𝑥prompt, 𝑦1, 𝑦2, . . . , 𝑦𝑡−1), (1) where 𝑥prompt denotes the initial prompt and 𝑦𝑡 denotes the 𝑡-th generated token.

Action (𝑎𝑡): An action corresponds to selecting the next token from the vocabulary V. At each step, the policy outputs a probability distribution over tokens, and the action token is selected via random sampling. Formally, the action is defined as:

𝑎𝑡 = 𝑦𝑡 ∈ V, where 𝑦𝑡 ∼ 𝜋𝜃(·|𝑠𝑡) = softmax ( 𝑓𝜃(𝑠𝑡)/𝑇) , (2)

where 𝑓𝜃(𝑠𝑡) ∈ ℝ|V| represents the LLM logit outputs and 𝑇 is the temperature parameter controlling the randomness of sampling.

Environment: The environment provides reward signals upon sequence completion. In rule-based settings, binary rewards are assigned based on the correctness. Alternatively, learned reward models or human feedback systems provide continuous rewards based on criteria such as helpfulness, harmlessness, or task alignment. The reward is computed as follows:

𝑟(𝜏) =

1, if 𝜏 satisfies correctness criteria 0, otherwise

, or 𝑟(𝜏) = 𝑅𝜙(𝜏) ∈ [0, 1], (3)

where 𝑅𝜙 is a learned reward model and 𝜏 = (𝑥prompt, 𝑦1, 𝑦2, . . . , 𝑦𝑇seq) represents the complete generated sequence of length 𝑇seq.

Rollout: Given an input prompt 𝑥prompt, the LLM auto-regressively generates a sequence by sampling tokens from 𝜋𝜃(𝑦𝑡|𝑠𝑡) until termination, without intermediate environmental feedback. With a nonzero temperature 𝑇, the policy can produce diverse rollouts that explore different solution paths.

- 2.2. RL Formulation for VLAs

State (𝑠𝑡): The state consists of multimodal observations including visual input (RGB images, depth maps, or point clouds), proprioceptive information (joint angles, end-effector pose), and language instructions of the tasks. Formally, the state is defined as:

𝑠𝑡 = (𝑜vis𝑡 , 𝑜prop𝑡 , 𝑙task), (4)

where 𝑜vis𝑡 is multimodal observations, 𝑜prop𝑡 is proprioceptive information, and 𝑙task is language instructions.

Action (𝑎𝑡): Actions are control commands in the robot action space, typically end-effector deltas or joint angle targets, where 𝑎𝑡 ∈ ℝ𝑑 (e.g., 𝑑 = 7 for 6-DoF pose plus gripper position). Most VLA policies generate actions through either a diffusion-based action expert or a discrete action tokenizer. The action is defined as follows:

𝑎𝑡 = 𝐷𝑒𝑐𝑜𝑑𝑒𝑟(ℎ𝜃(𝑠𝑡)), 𝐷𝑒𝑐𝑜𝑑𝑒𝑟 ∈ {Diffusion Expert, Action Tokenizer}, 𝑎𝑡 ∈ ℝ𝑑, (5)

where ℎ𝜃(𝑠𝑡) represents the hidden state of 𝑠𝑡 in the VLA model, and 𝐷𝑒𝑐𝑜𝑑𝑒𝑟 is the action decoder. Environment: The environment represents the physical world or simulation where the robot operates. It provides state transitions 𝑠𝑡+1 = Env(𝑠𝑡, 𝑎𝑡) and reward signals:

𝑟𝑡 = 𝛼 · 𝐼success + (1 − 𝛼) · ∑︁

1, if task success 0, otherwise

𝑤𝑖 · 𝜙𝑖(𝑠𝑡, 𝑎𝑡), 𝛼 ∈ [0, 1], 𝐼success =

, (6)

𝑖

where 𝜙𝑖(𝑠𝑡, 𝑎𝑡) represents process rewards (e.g. distance to goal), 𝑤𝑖 are weights, and 𝛼 balances outcome and process rewards.

Rollout: VLA models generate trajectories through iterative interaction with the environment. At each timestep, the policy 𝜋𝜃 takes the current state 𝑠𝑡 as input and outputs an action chunk (𝑎𝑡, 𝑎𝑡+1, . . . , 𝑎𝑡+𝑘−1) of length 𝑘. The robot executes these actions sequentially and the environment produces updated states based on physical dynamics. After execution, the model takes the new state 𝑠𝑡+𝑘 as input and generates the next action chunk. This process continues until task completion or maximum episode length, producing a complete trajectory 𝜏 = ((𝑠0, 𝑎0), (𝑠1, 𝑎1), . . . , (𝑠𝑇, 𝑎𝑇)) through interactive sampling.

###### 2.3. Group Relative Policy Optimization

Group Relative Policy Optimization (GRPO) [Shao et al., 2024] is an RL method that eliminates the value function by computing advantages through group-relative normalization. Given an initial state 𝑠0, the behavior policy 𝜋𝜃old generates 𝐺 trajectories {𝜏𝑖}𝐺𝑖=1. The GRPO objective employs PPO-style clipping with KL regularization to constrain policy updates:

###### ∑︁𝐺

∑︁|𝜏𝑖|

1 |𝜏𝑖|

1

min 𝑟𝑖,𝑡(𝜃)𝐴ˆ𝑖, clip(𝑟𝑖,𝑡(𝜃), 1 − 𝜖, 1 + 𝜖)𝐴ˆ𝑖

𝐽GRPO(𝜃) = 𝔼𝑠0∼D,{𝜏

𝑖}∼𝜋𝜃old

𝐺

𝑖=1

𝑡=1

(7)

− 𝛽𝐷KL(𝜋𝜃||𝜋ref) ,

where the importance sampling ratio 𝑟𝑖,𝑡(𝜃) and the normalized advantage 𝐴ˆ𝑖 are defined as:

𝑅𝑖 − mean({𝑅𝑖}𝐺𝑖=1) std({𝑅𝑖}𝐺𝑖=1)

𝜋𝜃(𝑎𝑖,𝑡|𝑠𝑖,𝑡) 𝜋𝜃old(𝑎𝑖,𝑡|𝑠𝑖,𝑡)

, 𝐴ˆ𝑖 =

. (8)

𝑟𝑖,𝑡(𝜃) =

Here 𝑅𝑖 denotes the total reward of the 𝑖-th trajectory, 𝜖 > 0 is the PPO clipping parameter that limits the policy ratio, and 𝛽 > 0 is the coefficient controlling the strength of KL regularization with respect

to the reference policy 𝜋ref.

##### SFT

###### Limited Offline Trajectories

[Figure 17]

[Figure 18]

τ! τ" . . . τ$

###### Policy

τ%

##### SimpleVLA-RL

Group Advantage Computation

###### Trajectories

Rewards

###### Rollout

- 0
- 1

"!

τ!

[Figure 19]

[Figure 20]

Policy

""

τ"

state action !! "!

…

…

…

|[Figure 21]|[Figure 22]<br><br>Environment|
|---|---|
|!!"#| |

0

"#

τ#

- Figure 2 | Overview of SimpleVLA-RL.

###### 3. SimpleVLA-RL

DeepSeek-R1 [Guo et al., 2025a] has achieved remarkable performance gains through online RL with the simple, scalable rule-based reward, highlighting a promising training paradigm. In this section, we introduce SimpleVLA-RL, which extends this rule-based online RL framework to VLA models for embodied manipulation tasks as shown in Figure 2. Specifically, our training framework proceeds as follows: we begin by generating multiple trajectories for each input via random sampling (§3.1). Each trajectory is then assigned a simple outcome reward (1 for success, 0 for failure) based on environment feedback (§ 3.2). Leveraging these rewards together with the corresponding action token probabilities, we compute the GRPO loss to update the policy model (§ 3.4).

###### 3.1. Interactive VLA Rollout

RL on VLA models differs fundamentally from LLMs in trajectory generation. To enable online RL, policy models need to generate diverse trajectories from an input for effective exploration. While LLMs naturally achieve this diversity through random sampling on text token distributions [Renze, 2024; De Rosa and Papa, 2021], VLA models face a unique challenge due to their action decoding strategies. Current VLA models often employ three strategies: (1) generating action token distributions similar to LLMs [Kim et al., 2024; Black et al., 2024], (2) diffusion-based denoising on latent states [Liu et al.,

- 2024; Cheang et al., 2025], and (3) deterministic regression via MLPs [Kim et al., 2025]. Among these, the token-based approach is most compatible with PPO-like RL algorithms, as it naturally provides action distributions necessary for both random sampling and policy gradient computation. Therefore, we adopt this approach, where the VLA model outputs action token probability distributions and employs random sampling to generate diverse trajectories.

Furthermore, for a given input query, LLM rollout proceeds by autoregressively generating tokens until reaching a stop token or max output length. In contrast, VLA rollout requires continuous interaction with the environment to update the visual observation and robot state dynamically (as detailed in Section 2). This closed-loop interaction is necessary because each robotic action alters the environment, and subsequent actions must be conditioned on real-time sensory feedback. We present the comparison of the rollout algorithm pseudo-code of LLMs and VLA in Listing 1.

def rollout(policy , dataset , number_sample=8, max_steps=None): rollout_dataset = [] for batch in dataset:

batch = batch.repeat(number_sample)

- - # LLM generates diverse outputs using random sampling

- - outputs = policy.generate(batch , temperature =1.0)

- - rollout_dataset.append((batch , outputs))

+ # Parallel env initialization and interaction

+ envs = env_process_pool.submit(batch.initialize)

+ states = env_process_pool.submit(envs.setup)

+ for t in range(max_steps):

+ # VLA generates diverse trajectories using temperature sampling on action tokens

+ actions = policy.generate(states , temperature =1.0)

+ rollout_dataset.append({f"{e.name}_step_{t}": (s,a) for e ,s,a in zip(envs ,states ,actions)})

+ states , dones = env_process_pool.submit(envs.step , actions)

+ # Remove completed tasks

+ active = [(e,s) for e,s,d in zip(envs ,states ,dones) if not d]

+ if not active:

+ break

+ envs , states = zip(*active) return rollout_dataset

Listing 1 | Pseudo-code for the adopted veRL rollout function: from LLM-based generation to interactive VLA sampling with synchronous environment parallelism.

###### 3.2. Outcome Reward Modeling

SimpleVLA-RL employs a straightforward binary reward function for RL training. Unlike traditional RL approaches that require carefully crafted reward functions [Hadfield-Menell et al., 2017; Knox et al., 2023; Booth et al., 2023], we follow DeepSeek-R1’s approach by assigning trajectory-level rewards of either 0 or 1 based solely on task completion. When the VLA model successfully completes a task, the entire trajectory is assigned a reward of 1; otherwise, it receives a reward of 0. For gradient computation, these trajectory-level rewards are uniformly propagated to the individual action tokens. Consequently, all tokens within successful trajectories are assigned a reward of 1, whereas those in unsuccessful trajectories are assigned a reward of 0. Our reward function is:

𝑅(𝑎𝑖,𝑡 | 𝑠𝑖,𝑡) =

1, is_successful[traj𝑖(𝑎𝑖, 𝑠𝑖)], 0, otherwise.

(9)

This simple outcome-level reward is simple yet effective: scalable, broadly applicable across environments, and free from complex process-based design [Wu et al., 2021]. By focusing solely on task completion, it avoids the non-transferability issues typical of task-specific rewards.

###### 3.3. Exploration Enhancements

Previous works [Yu et al., 2025; Liu et al., 2025b,c; An et al., 2025] have demonstrated that encouraging exploration during RL is critical. We observe that this factor becomes even more crucial

in VLA RL. Manipulation tasks typically allow for a wide range of valid solutions. However, VLA models tend to converge on a narrow set of solution patterns, largely due to the homogeneity of their training trajectories, which limits the efficiency of RL. Promoting exploration encourages models to discover novel strategies and broaden the solution space, a property that is particularly advantageous in scenarios with low success rates. Building on this insight, we implement three key modifications to enhance the exploration of RL training: 1) employing dynamic sampling during trajectory rollout, 2) adjusting the clip range in the GRPO training objective, 3) and increasing the sampling temperature during rollout.

Dynamic Sampling Critic-free RL algorithms suffer from vanishing gradients when trajectories are assigned the same rewards. For example, GRPO computes advantages using group-relative normalization, comparing each response’s reward to the mean and standard deviation of rewards within its group of sampled outputs. When all trajectories share identical rewards, their advantage estimation becomes zero, resulting in null gradients and causing unstable training dynamics.

We address this challenge through Dynamic Sampling [Yu et al., 2025; Cui et al., 2025a], a method that has been proven effective in LLM RL [Team et al., 2025b; Yu et al., 2025; Cui et al., 2025a; Shi et al., 2025]. During rollout, we exclude groups in which all trajectories either succeed or fail. Sampling proceeds until the batch consists solely of groups with mixed outcomes, which can be formally expressed as:

0 < {traj𝑖(𝑎𝑖, 𝑠𝑖) | is_successful[traj𝑖(𝑎𝑖, 𝑠𝑖)]} < 𝐺. (10) This ensures non-zero advantage estimates and stable gradient flow throughout training.

Clipping Higher PPO and GRPO employ clipping over the importance sampling ratio to restrict the trust region [Schulman et al., 2015] and enhance RL stability [Shao et al., 2024; Schulman et al., 2017]. However, the upper clipping threshold restricts the probability increase of low-probability tokens, thereby potentially constraining exploration. Following DAPO [Yu et al., 2025], we modify the clipping range in the GRPO training objective from [0.8, 1.2] to [0.8, 1.28].

Higher Rollout Temperature Recent works on LLM RL adjusting the rollout temperature to promote exploration have been widely shown to be effective, with sampling at higher temperatures yielding particularly notable improvements [Liu et al., 2025c; An et al., 2025; Liao et al., 2025]. To encourage the VLA model to generate more diverse trajectories during the rollout phase, we increase the sampling temperature from 1.0 to 1.6. As shown in Figure 3, these modifications led to notable improvements.

###### 10%

- 90

90

90

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### 15%

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

###### 15%

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

LIBERO-LongSR(%)

LIBERO-LongSR(%)

LIBERO-LongSR(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

80

80

80

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

70

70

70

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

60

60

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

50

50

50

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

40

40

40

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

30

30

30

w/ Dynamic-Sampling

w/ Clip-Higher

w/ Temperature-Higher

| |
|---|

| |
|---|

| |
|---|

w/o Dynamic-Sampling

w/o Clip-Higher

w/o Temperature-Higher

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

20

20

20

| |
|---|

| |
|---|

| |
|---|

0 50 100 150 200 250 300

0 50 100 150 200 250 300

0 50 100 150 200 250 300

RL Training Steps

RL Training Steps

RL Training Steps

(a) Dynamic Sampling

(b) Clip Higher

(c) Higher Rollout Temperature

- Figure 3 | The effectiveness of three key enhancements: dynamic sampling, higher rollout temperature, and clip higher.

- Table 1 | RoboTwin 2.0 task classification based on planning horizon and required steps.

Task Name Steps Horizon Horizon Group

Short Horizon Tasks (112-130 steps) lift_pot 112 Short

beat_block_hammer 113 Short pick_dual_bottles 127 Short place_phone_stand 130 Short

Average: 121 steps Count: 4 tasks

Medium Horizon Tasks (151-223 steps) move_can_pot 151 Medium

place_a2b_left 155 Medium place_empty_cup 174 Medium handover_mic 223 Medium

Average: 176 steps Count: 4 tasks

###### Long Horizon Tasks (283-313 steps)

handover_block 283 Long Average: 298 steps stack_bowls_two 313 Long Count: 2 tasks

Extra Long Horizon Tasks (466-637 steps) blocks_rank_rgb 466 Extra-Long Average: 552 steps put_bottles_dustbin 637 Extra-Long Count: 2 tasks Overall Statistics Total: 12 tasks, Average: 256 steps

###### 3.4. Training Objective

We use the adopted GRPO algorithm [Shao et al., 2024] for online RL training on VLA models, with modifications as introduced in Section 3.3. Moreover, we remove the KL divergence regularization following DAPO [Yu et al., 2025]. This eliminates the need for a reference model during training, reducing memory consumption and accelerating the training. Additionally, the KL penalty constrains policy divergence from a fixed reference, potentially limiting exploration of new behaviors. Therefore, the policy is optimized via the following objective:

∑︁|𝑎𝑖|

###### ∑︁𝐺

1

1 |𝑎𝑖|

min 𝑟𝑖,𝑡(𝜃)𝐴ˆ𝑖, clip 𝑟𝑖,𝑡(𝜃), 1 − 𝜀𝑙𝑜𝑤, 1 + 𝜀ℎ𝑖𝑔ℎ 𝐴 ˆ𝑖

###### J(𝜃) = 𝔼𝑠0∼D,{𝑎

𝑡}𝑖𝐺=1∼𝜋𝜃old (·|𝑠𝑡)

(11)

𝐺

𝑡=1

𝑖=1

s.t. 0 < {traj𝑖(𝑎𝑖, 𝑠𝑖) | is_successful[traj𝑖(𝑎𝑖, 𝑠𝑖)]} < 𝐺,

where

𝑅𝑖 − mean({𝑅𝑖}𝐺𝑖=1) std({𝑅𝑖}𝐺𝑖=1)

𝜋𝜃(𝑎𝑖,𝑡 | 𝑠𝑖,𝑡) 𝜋𝜃old(𝑎𝑖,𝑡 | 𝑠𝑖,𝑡)

, 𝐴ˆ𝑖 =

. (12)

𝑟𝑖,𝑡(𝜃) =

###### 4. Experiments

- 4.1. Experimental Setup Benchmarks We evaluate SimpleVLA-RL on the widely used simulation benchmark, LIBERO [Liu

- et al., 2023], RoboTwin1.0 [Mu et al., 2025], and RoboTwin2.0 [Chen et al., 2025a], and conduct real-world experiments on RoboTwin2.0 tasks. LIBERO is a lifelong learning benchmark focused on language-guided manipulation tasks across diverse object types, task specifications, and environments. It consists of five task suites: LIBERO-Goal, LIBERO-Spatial, LIBERO-Object, LIBERO-Long (10 tasks, each with 50 expert demonstrations), and LIBERO-90 (90 tasks for large-scale multitask evaluation).

- Table 2 | Main results of different VLA models on LIBERO.

Model

LIBERO Spatial Object Goal Long Avg Octo 78.9 85.7 84.6 51.1 75.1 OpenVLA 84.7 88.4 79.2 53.7 76.5 Nora 92.2 95.4 89.4 74.6 87.9 𝜋0 + FAST 96.4 96.8 88.6 60.2 85.5 𝜋0 96.8 98.8 95.8 85.2 94.2 UniVLA 96.5 96.8 95.6 92.0 95.2 OpenVLA-OFT 91.6 95.3 90.6 86.5 91.0

w/ ours 99.4 99.1 99.2 98.5 99.1 Δ +7.8 +3.8 +8.6 +12.0 +8.1

- Table 3 | Main results of different VLA models on RoboTwin1.0.

RoboTwin1.0

Model

Avg

Hammer Beat Block Handover Blocks Stack Shoe Place DP 0.0 12.0 7.1 4.3 5.9 DP3 64.7 84.3 24.0 59.3 58.1 OpenVLA-OFT 67.2 61.6 7.1 23.4 39.8

w/ ours 92.6 89.6 40.2 59.3 70.4 Δ +25.4 +28.0 +33.1 +35.9 +30.6

We evaluate performance using the average Success Rate (SR) across 50 held-out test scenarios for each task. Moreover, RoboTwin1.0 [Mu et al., 2025] and RoboTwin2.0 [Chen et al., 2025a] are simulation benchmarks for dual-arm manipulation. RoboTwin1.0 provides 17 bimanual tasks with limited scene and object diversity. RoboTwin2.0 extends to 50 tasks across multiple robot embodiments with 731 object instances and comprehensive domain randomization (clutter, lighting, background, tabletop height, and language instructions), which enhances task diversity and simto-real transfer. For RoboTwin2.0 training and evaluation, we employ the Agilex Piper robotic arm and domain-randomized task settings, with each task evaluated on 100 held-out test scenarios. For RoboTwin2.0, we select and categorize 12 tasks into 4 horizon levels based on the average number of steps per task for comprehensive evaluation. Table 1 summarizes the specific step counts for each task and the step ranges for different horizon levels of the various tasks in RoboTwin2.0.

Backbones We apply SimpleVLA-RL to OpenVLA-OFT [Kim et al., 2025], a state-of-the-art autoregressive VLA model that achieves high performance and inference efficiency. Based on OpenVLA [Kim

- et al., 2024], it employs vision encoders and LLaMA2-7B [Touvron et al., 2023] as the backbone with action chunk and parallel decoding designs, making it particularly suitable for online RL where model inference is frequent. Our implementation of the OpenVLA-OFT model differs from the official version. To achieve improved training and inference efficiency, we utilize only single-view images, language instructions, and robot proprioceptive states as model inputs, whereas the official model additionally incorporates wrist camera images. Additionally, in the LIBERO, we don’t use robot proprioceptive states in model inputs. Regarding the model architecture, we employ only parallel decoding and action chunking designs. We use the LLaMA2 output head to generate action tokens and the cross-entropy loss, whereas the official model uses an MLP to generate continuous actions and L1 regression. Due to the differences in model inputs and architecture, we cannot use the official checkpoints. We modify the official codebase and performed SFT from scratch using the same datasets and hyperparameters as the official implementation.

- Table 4 | Main results of different VLA models on RoboTwin2.0, organized by task horizon. Short Horizon Tasks (100-130 Steps)

Model Lift Pot Beat Hammer Block Pick Dual Bottles Place Phone Stand Avg 𝜋0 51.0 59.0 50.0 22.0 45.5 RDT 45.0 22.0 18.0 13.0 24.5 OpenVLA-OFT 10.1 28.1 29.7 17.1 21.3

w/ ours 64.1 87.5 68.3 39.6 64.9 Δ +54.0 +59.4 +38.6 +22.5 +43.6

Medium Horizon Tasks (150-230 Steps)

Model Move Can Pot Place A2B Left Place Empty Cup Handover Mic Avg 𝜋0 41.0 38.0 60.0 96.0 58.8 RDT 33.0 21.0 42.0 95.0 47.8 OpenVLA-OFT 28.1 37.5 77.3 45.3 47.1

w/ ours 61.2 45.3 94.2 89.2 72.5 Δ +33.1 +7.8 +16.9 +43.9 +25.4

Long (280-320 Steps) & Extra Long Horizon Tasks (450-650 Steps)

Model Handover Block Stack Bowls Two Blocks Rank Rgb Put Bottles Dustbin Avg 𝜋0 39.0 53.0 45.0 36.0 43.3 RDT 26.0 42.0 17.0 26.0 27.8 OpenVLA-OFT 33.1 40.6 70.2 42.2 46.5

w/ ours 57.8 75.8 81.3 60.9 69.0 Δ +24.7 +35.2 +11.1 +18.7 +22.4

Overall Avg RDT: 33.3 𝜋0: 49.2 OpenVLA-OFT: 38.3 w/ ours: 68.8 +30.5

Baselines We compare with recent advanced VLA models: UniVLA [Bu et al., 2025b], RDT-1B [Liu

- et al., 2024], 𝜋0 [Black et al., 2024], 𝜋fast [Pertsch et al., 2025], Nora [Hung et al., 2025], OpenVLA [Kim et al., 2024], Octo [Team et al., 2024], DP [Chi et al., 2024] and DP3 [Ze et al., 2024].

Implementation Details For training infrastructure, we employ 8 x NVIDIA A800 80GB for fullparameter training. The training hyperparameters are configured as follows: learning rate 𝑙𝑟 =

- 5 × 10−6, training batch size of 64, sampling count of 8, mini-batch size of 128, clip ratio 𝜀𝑙𝑜𝑤 = 0.2, 𝜀ℎ𝑖𝑔ℎ = 0.28, and temperature 𝑇 = 1.6. The number of action chunks is 8 in the LIBERO and 25 in the RoboTwin1.0&2.0. The model is configured with a total of 256 action tokens. The maximum environment interaction step is set to 512 in the LIBERO and 200, 400, or 800 in the RoboTwin1.0&2.0, depending on different tasks. During the rollout phase of RL training, we employ random sampling. For evaluation, we utilize greedy sampling, with each benchmark tested three times for reproducibility.

###### 4.2. Main Results

We evaluate SimpleVLA-RL on the three benchmarks: LIBERO, RoboTwin1.0, and RoboTwin2.0. For each benchmark, we employ a two-stage training paradigm in which SFT is followed by SimpleVLARL, both applied to OpenVLA-OFT. In contrast, the baseline models are trained solely with SFT. For the four task suites of LIBERO, we perform SFT using all 500 demonstrations per task suite, then perform RL on 500 corresponding simulation scenarios. For RoboTwin1.0, we use 50 demonstrations per task for single-task SFT, then perform RL on 100 simulation scenarios per task. For RoboTwin2.0, we use 1,000 demonstrations per task for SFT, then perform RL on 1,000 scenarios per task.

Tables 2, 3, and 4 present the results on the three benchmarks: LIBERO, RoboTwin1.0, and RoboTwin2.0, respectively. On four LIBERO task suites, SimpleVLA-RL improves the average success rate of the SFT-tuned OpenVLA-OFT model from 91% to 99%, achieving SoTA performance and surpassing all current VLA models such as 𝜋0 and UniVLA. For long-horizon tasks in LIBERO-Long, SimpleVLA-RL achieves 98.5% success rate, with a 12% improvement over the baseline (86.5%) and 13.3% gains over 𝜋0 (85.2%). The results demonstrate that SimpleVLA-RL can substantially improve model performance even when SFT has already achieved strong results. On RoboTwin1.0’s four dual-arm tasks, SimpleVLA-RL achieves a 30.6% gain over the fine-tuned OpenVLA-OFT baseline (from 39.8% to 70.4%). Across 12 dual-arm tasks of RoboTwin2.0, SimpleVLA-RL achieves an 80% relative improvement, lifting performance from 38.3% to 68.8% and outperforming SoTA methods including 𝜋0 (49.2%) and RDT (33.3%). Notably, even on two Extra-Long-Horizon tasks requiring multi-round dual-arm interactions such as “Blocks Rank Rgb” and “Put Bottles Dustbin”, SimpleVLA-RL achieves 11.1% and 18.7% points gain, respectively. SimpleVLA-RL shows consistent improvements in all horizon levels, validating the effectiveness of outcome-level reward even for complex long-horizon tasks. The results demonstrate that SimpleVLA-RL consistently improves model performance across diverse benchmarks without requiring additional demonstration data.

###### 5. Analysis

In this section, we analyze the role of SimpleVLA-RL in addressing three key challenges that hinder the further advancement and scaling of the VLA model: data, generalization, and real-world tasks. Below are several key takeaways:

Takeaways

- 1. Data: SimpleVLA-RL can significantly reduce reliance on demonstration data, effectively alleviating the data scarcity bottleneck that constrains VLA scaling (§ 5.1).
- 2. Generalization: Compared to SFT, SimpleVLA-RL demonstrates strong generalization in spatial configurations, object types, and task settings (§ 5.2).
- 3. Real-world Task: SimpleVLA-RL exhibits strong sim-to-real transfer, with large-scale simulation training remarkably improving real-world performance, indicating a promising path for scaling up real-world policy (§ 5.3).

###### 5.1. Overcoming Data Scarcity

Developing foundation VLA models for manipulation tasks requires large-scale demonstration data for training [Black et al., 2024; Liu et al., 2024; Intelligence et al., 2025]. This data scaling paradigm has been proven in the NLP area [Touvron et al., 2023; Hoffmann et al., 2022; Achiam et al., 2023]. However, acquiring high-quality trajectory data for embodied manipulation tasks remains expensive and difficult, creating a fundamental bottleneck for VLA model development [Zhong et al., 2025; Bi

- et al., 2025]. Therefore, we investigate whether SimpleVLA-RL can enhance VLA models even with extremely limited demonstration trajectories to overcome this limitation.

Settings To simulate scenarios with scarce demonstration data, we finetune OpenVLA-OFT using only one demonstration data per task, denoted as One-Trajectory SFT. Given that each of the four LIBERO task suites contains 10 distinct tasks, we utilize merely 10 demonstration data per task suite. For comparison, we also conduct an experiment using all available demonstration data for each task, 500 per task suite, denoted as Full-Trajectory SFT. Following both One-Trajectory SFT and Full-Trajectory SFT, we apply SimpleVLA-RL on the SFT model.

Results As shown in Table 5, the limitations of SFT become notable when demonstration data is limited. Compared to results under Full-Trajectory SFT setting, the SFT success rates on LIBEROSpatial, LIBERO-Object, and LIBEROGoal fall below 63.6%, while the performance on long-horizon tasks in LIBEROLong drops to only 17.3%, highlighting the challenge of relying solely on SFT for long-horizon tasks. Remarkably, after applying SimpleVLA-RL on One-Trajectory SFT, the average success rate across the four LIBERO task suites increases from 48.9% to 96.9%, even surpassing the

Table 5 | Comparisons between One-Trajectory and FullTrajectory SFT on LIBERO.

LIBERO Spatial Object Goal Long Avg

Model

One-Trajectory SFT

OpenVLA-OFT 63.6 54.9 59.6 17.3 48.9 w/ ours 98.2 98.7 98.8 91.7 96.9 Δ +34.6 +43.8 +39.2 +74.4 +48.0

Full-Trajectory SFT

OpenVLA-OFT 91.6 95.3 90.6 86.5 91.0 w/ ours 99.4 99.1 99.2 98.5 99.1 Δ +7.8 +3.8 +8.6 +12.0 +8.1

- 91% of Full-Trajectory SFT. Notably, LIBERO-Long improves from 17.3% to 91.7%, while the other three task suites all exceed 98% success rate, each exceeding their Full-Trajectory SFT performance. The performance gap between One-Trajectory SFT + RL (96.9%) and Full-Trajectory SFT + RL (99.1%) is only 2.2%. These results demonstrate that SimpleVLA-RL can substantially improve VLA model performance in data-scarce scenarios. This suggests that through trial-and-error exploration with outcome feedback, online RL methods such as SimpleVLA-RL can enable further scaling of VLA model training, even when only minimal demonstration data is available.

- 5.2. Generalization Analysis The generalization ability of VLA models remains a key challenge [Zhong et al., 2025; Liu et al.,

- 2025a; Intelligence et al., 2025]. This subsection evaluates how SFT and online RL methods like SimpleVLA-RL affect VLA generalization across three dimensions: spatial (LIBERO-Spatial), objects (LIBERO-Object), and tasks (LIBERO-Goal).

Settings We conduct this experiment using three task suites, LIBERO-Spatial, LIBERO-Object, and LIBERO-Goal, each of which comprises ten distinct tasks. For each LIBERO task suite, we randomly select nine tasks as seen tasks for RL or SFT training, while reserving the remaining task as the unseen task for out-of-distribution evaluation. For both methods, we first fine-tune OpenVLA-OFT under the One-Trajectory SFT setting to obtain a base model with non-zero success rates, since the original model achieves only 0% on LIBERO and is incapable of performing online RL. For SFT, we further fine-tune the One-Trajectory SFT base model (§5.1) using all 450 demonstration trajectories from 9 seen tasks on each task suite. Moreover, we perform SimpleVLA-RL on this One-Trajectory SFT base model. We plot how the models’ performance on unseen tasks changes as their training tasks’ success rates increase during training.

Results Figure 4 presents the results. During training, both SFT and RL achieve over 90% success rates on training tasks. However, their performance on unseen tasks diverges significantly. As the seen tasks’ success rates increase during training, SimpleVLA-RL shows consistent improvement on unseen tasks across different settings. In contrast, SFT suffers from severe overfitting. While achieving high performance on training tasks, SFT exhibits performance degradation on most unseen tasks, often shows catastrophic forgetting with success rates dropping to 0%. This indicates that RL training enables VLA models to retain previously acquired capabilities while learning generalizable skills from diverse tasks. On LIBERO-Goal’s three unseen tasks, the SFT success rates immediately drop to 0% at the beginning of training. This may be attributed to the fact that LIBERO-Goal tasks involve

###### Goal Unseen Task 1

put the wine bottle on top of the cabinet

UnseenTaskSR(%)

| |
|---|

75

| |
|---|

| |
|---|

| |
|---|

50

RL

25

SFT

0

70 80 90 100

Seen Tasks SR (%)

###### Object Unseen Task 1

pick up the cream cheese and place it in the basket

UnseenTaskSR(%)

75

| |
|---|

| |
|---|

| |
|---|

| |
|---|

RL

50

SFT

25

0

50 60 70 80 90 100

Seen Tasks SR (%)

###### Spatial Unseen Task 1

pick up the black bowl between the plate and ramekin

UnseenTaskSR(%)

80

| |
|---|

| |
|---|

60

RL

| |
|---|

SFT

40

20

70 75 80 85 90 95 100

Seen Tasks SR (%)

###### Goal Unseen Task 2

put the cream cheese in the bowl

UnseenTaskSR(%)

40

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

20

RL

SFT

0

70 80 90 100

Seen Tasks SR (%)

###### Object Unseen Task 2

pick up the salad dressing and place it in the basket

UnseenTaskSR(%)

80

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

| |
|---|

RL

40

SFT

20

0

60 70 80 90 100

Seen Tasks SR (%)

###### Spatial Unseen Task 2

pick up the black bowl next to the ramekin

UnseenTaskSR(%)

15

| |
|---|

| |
|---|

| |
|---|

10

| |
|---|

RL

SFT

5

0

80 85 90 95 100

Seen Tasks SR (%)

###### Goal Unseen Task 3

put the wine bottle on the rack

UnseenTaskSR(%)

| |
|---|

60

| |
|---|

RL

40

SFT

20

0

60 70 80 90 100

Seen Tasks SR (%)

###### Object Unseen Task 3

pick up the milk and place it in the basket

UnseenTaskSR(%)

80

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

RL

SFT

40

60 70 80 90 100

Seen Tasks SR (%)

###### Spatial Unseen Task 3

pick up the black bowl on the ramekin

UnseenTaskSR(%)

75

| |
|---|

| |
|---|

| |
|---|

| |
|---|

RL

50

SFT

25

0

70 80 90 100

Seen Tasks SR (%)

- Figure 4 | Generalization Analysis on LIBERO: Goal Unseen (Top), Object Unseen (Middle), Spatial Unseen (Bottom).

- Table 6 | Real-world experiment (sim2real) results.

Stack Bowls Place Empty Cup Pick Bottle Click Bell Avg RDT 60.0 4.0 10.0 20.0 23.5 OpenVLA-OFT 38.0 2.0 0.0 30.0 17.5 w/ ours 70.0 10.0 14.0 60.0 38.5

Δ +32.0 +8.0 +14.0 +30.0 +21.0

diverse objects and manipulation strategies with few transferable components across tasks, whereas SFT relies on similar task distributions to enable generalization. In contrast, our SimpleVLA-RL avoids performance degradation and even achieves 5%-15% improvements on the three unseen tasks. On LIBERO-Object, SFT improves from 57.8% to 74.6% on Unseen Task 3 but fails on the other two tasks. In contrast, our method improves across all three unseen tasks, with gains of 36.5% on Unseen Task 2 and 16.4% on Unseen Task 3. On LIBERO-Spatial, we observe a consistent trend: SFT degrades by 10% on Unseen Task 1 and completely fails on the remaining tasks. In contrast, our method improves performance on Unseen Task 1 from 43.3% to 71.8% and achieves gains of 7.1% and 13.3% on the other two tasks. SimpleVLA-RL consistently outperforms SFT in improving the generalization of VLA models, even in the most challenging cross-task scenarios.

###### 5.3. Real-World Experiments

To evaluate the real-world effectiveness of SimpleVLA-RL, we conduct sim-to-real experiments on four RoboTwin2.0 tasks2: Stack Bowls, Handover Block, Pick Bottle, and Click Bell. We employ OpenVLA-OFT as the policy model, RDT as the baseline model, and execute on two AgileX Piper robotic

2The detailed descriptions of the four tasks can be found at: https://robotwin-platform.github.io/doc/ tasks/index.html

arms. For each task, we first use 1000 simulation trajectories for SFT. Then we apply SimpleVLA-RL on the SFT model using 1000 simulation scenarios to obtain an RL model. The entire training process uses only simulation data without any real-world demonstrations. We evaluate both the SFT and RL models on clean tabletops with unseen backgrounds in the real world. Each task is tested with 50 trials. The RDT baseline model only undergoes the SFT stage.

The sim2real results in Table 6 demonstrate that SimpleVLA-RL significantly improves the real-world success rates of VLA models, with an average improvement from 17.5% to 38.5%, surpassing RDT’s 23.5%. For instance, in the Stack Bowls task, SimpleVLA-RL achieves a 96% relative improvement, lifting performance from 32% to 70% and outperforming RDT (60%). On the Pick Bottle task, which demands higher action precision, as the bottle will fall if the robotic arm is not perfectly aligned on the first attempt, the SFT model fails completely while SimpleVLA-RL achieves a 15% success rate, demonstrating its effectiveness in improving action precision. Using SimpleVLA-RL for low-cost, large-scale, and highly parallel RL training in simulation, we significantly improve the real-world performance of simulation-trained VLA models. This demonstrates a promising path for scaling real-world policies: using rich simulation assets and high-fidelity simulators for cost-effective RL training to achieve superior performance in real-world deployment.

###### 6. Discussions

- 6.1. “Pushcut”: Emergence of New Patterns through RL

Task: Place A2B Right

Task: Move Can Pot

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

#### Grasp

### Grasp

SFT

###### SFT

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

RL Push

Push

RL

(a) “move can pot” task: Model learned to push the can to the pot (bottom) instead of grasp-move-place in the demonstration data (top).

(b) "place a2b right" task: Model learned to push A to B’s right (bottom) instead of demonstrated grasp-moveplace (top).

- Figure 5 | Illustration of “pushcut”. Emergent pushing behaviors through RL in RoboTwin2.0 tasks.

In this subsection, we present an emergent behavior “pushcut” shown during RL training. We observe that the VLA model learns novel behaviors that are absent from the demonstration data during the training process of SimpleVLA-RL. Specifically, in the move can pot task of RoboTwin2.0, where the objective is to transport cans to positions adjacent to designated pots, all demonstration trajectories consistently adhere to a “grasp–move–place” strategy, as illustrated at the top of Figure 5a.

Remarkably, after RL training, the VLA model autonomously discovers a more efficient solution: instead of grasping, it pushes the can directly into the target location (Figure 5a, bottom). We refer to this emergent phenomenon as “pushcut” (a push-driven shortcut), highlighting the model’s ability to bypass the conventional grasp–move–place routine. Similar emergent behaviors are observed in

the place a2b left/right task. While the demonstration data complete the task by grasping Object A and placing it next to Object B, the RL-trained model instead learns to accomplish the task by simply pushing Object A into position, as shown in Figure 5b. The phenomenon of “pushcut” closely parallels the “Aha Moment” observed in DeepSeek-R1 [Guo et al., 2025a], as both emerge through RL, where the agent uncovers novel patterns.

This phenomenon highlights the fundamental distinction between SFT and RL. Whereas SFT merely replicates fixed patterns present in the demonstrations, RL facilitates exploration driven by rewards, thereby uncovering novel strategies. During RL training, effective behaviors are reinforced through positive rewards, while less efficient ones are gradually eliminated. The outcome-level reward further promotes such emergent strategies: since both grasping and pushing yield equivalent rewards upon successful completion, the sparse reward design avoids the procedural constraints of processlevel supervision, affording the agent a broader exploration space and enabling the discovery of unanticipated yet effective solutions.

###### 6.2. Failure Modes of SimpleVLA-RL

- Table 7 | Impact of initial model capability on SimpleVLA-RL performance.

RoboTwin2.0

Move Can Pot Place A2B Lift Place A2B Right Place Phone Stand Pick Dual Bottles Avg 0 trajs SFT 0 0 0 0 0 0

+RL 0 0 0 0 0 0 100 trajs SFT 9.4 7.8 7.8 10.1 1.2 7.3

+RL 51.6 25.0 27.2 18.8 4.3 25.4 Δ +42.2 +17.2 +19.4 +8.7 +3.1 +18.1

1000 trajs SFT 28.1 37.5 28.7 17.1 29.7 28.2

+RL 61.2 45.3 37.5 39.6 68.3 50.4 Δ +33.1 +7.8 +8.8 +22.5 +38.6 +22.2

This subsection investigates the failure conditions of SimpleVLA-RL and key influencing factors. Through experiments on five RoboTwin2.0 tasks, we find that the model priors are the critical factor determining RL effectiveness.

Settings Each task is trained under domain randomization with a single-task setting. We compare three model variants: (1) the OpenVLA-OFT base model without trajectory fine-tuning (0 trajectories SFT); (2) the model fine-tuned with 100 demonstration trajectories per task (100 trajectories SFT); and (3) the model fine-tuned with 1000 demonstration trajectories per task(1000 trajectories SFT). All models undergo SimpleVLA-RL training on 1000 training scenarios and are evaluated on 100 held-out test scenarios.

RL fails completely when the base model has no initial task ability. Table 7 reports the results. The base model (0-trajectory SFT) achieves a 0% success rate across all tasks, exhibiting no task-relevant behaviors. Despite extensive pretraining, OpenVLA shows extremely limited zero-shot generalization, consistent with findings in Kim et al. [2025]. Because no successful trajectories are generated during sampling and only outcome rewards (without process rewards) are employed, every trajectory receives zero reward. As a result, RL is unable to improve performance, which remains at 0%.

The model prior has a significant impact on the effectiveness of RL. Initial capability is strongly correlated with post-RL performance. The 100-trajectory SFT model improves from 7.3% to 25.4% (an 18.1% gain), while the 1000-trajectory SFT model improves from 28.2% to 50.4% (a 22.2% gain) in average success rate. This trend is consistent across tasks. For instance, in the move can pot task,

the 100-trajectory SFT model improves from 9.4% to 51.6%, whereas the 1000-trajectory SFT model improves from 28.1% to 61.2%. These results highlight that stronger initial capabilities provide more effective starting points for exploration, thereby facilitating greater performance improvements.

RL effectiveness has a threshold: when initial ability is too low, improvements remain negligible. Our findings further reveal that the effectiveness of RL is subject to a performance threshold. When initial success rates are very low, online RL with outcome rewards yields only marginal improvements. For example, in the pick dual bottles task, the 100-trajectory SFT model improves from 1.2% to 4.3%, while the 1000-trajectory SFT model improves from 29.7% to 68.3%. Similarly, in the place phone task, the 100-trajectory SFT model gains 8.7%, compared to a 22.5% gain for the 1000-trajectory SFT model. The results indicate that a minimal level of task competence is essential for effective RL. Below this threshold, exploration is ineffective and RL fails to produce meaningful gains.

###### 7. Related Works

###### 7.1. Reinforcement Learning for Large Language Models

Reinforcement Learning (RL) for Large Language Models (LLMs) has achieved remarkable success, demonstrating its ability to induce complex reasoning behaviors such as self-verification and iterative optimization, thereby significantly enhancing model performance on reasoning tasks [Guo et al., 2025a; Zeng et al., 2025; Cui et al., 2025a; Jaech et al., 2024; Liu et al., 2025d; Zuo et al., 2025]. Recent advancements in Large Reasoning Models (LRMs), such as DeepSeek-R1 [Guo et al., 2025a], highlight the effectiveness of RL in boosting reasoning capabilities even with simple rule-based rewards, as exemplified by GRPO [Shao et al., 2024]. This approach differs substantially from Reinforcement Learning from Human Feedback (RLHF) [Ouyang et al., 2022], which aligns base models with human preferences using algorithms like Proximal Policy Optimization (PPO) [Schulman et al., 2017] and heavily relies on preference modeling.

Recent studies have increasingly focused on enhancing exploration in reinforcement learning to enable longer training horizons and improved performance. DAPO [Yu et al., 2025] introduces ClipHigher, a decoupled variant of PPO clipping, which sets a higher upper bound relative to the lower one (e.g., 𝜖low = 0.2, 𝜖high = 0.28). This adjustment allows low-likelihood but potentially valuable tokens to increase in probability, thereby encouraging exploration. Building on this, POLARIS [An

- et al., 2025] employs a staged curriculum of temperature increases (e.g., 0.7 → 1.0 → 1.1 for a 7B model) to gradually expand trajectory diversity and facilitate more robust policy discovery. In parallel, Entropy Mechanism [Cui et al., 2025b] addresses entropy collapse, a persistent issue in extended training, through methods such as Clip-Cov and KL-Cov, which selectively clip probabilities or penalize high-covariance tokens to sustain effective exploration. Similarly, ProRL [Liu et al., 2025b] combines KL control with reference policy resetting to preserve stability and extend training without degrading performance. A complementary line of work regulates entropy via temperature tuning. Acereason-nemotron 1.1 [Liu et al., 2025c] advocates adjusting temperatures to stabilize post-scaling entropy around a target (e.g., 0.3), balancing exploration and exploitation. Liao et al. [2025] further proposes a dynamic scheduler that adapts temperature over time to maintain stable entropy, thereby supporting sustained performance gains.

- 7.2. Vision Language Action Models In the field of robotic manipulation tasks, VLA models [Kim et al., 2024; Black et al., 2024; Liu et al.,

- 2024; Kim et al., 2025; Bu et al., 2025b; Pertsch et al., 2025; Hung et al., 2025; Intelligence et al., 2025] have shown better performance and task generalization compared to traditional policy-based

approaches [Ma et al., 2022; Yuan et al., 2024]. These models integrate the VLM or LLM backbone with action modules through unified end-to-end training [Zhong et al., 2025]. This approach enables comprehensive multimodal understanding and fine-grained motor control [Firoozi et al., 2025]. Currently, many studies are focused on enhancing the effectiveness of VLA models. For example, E-COT [Zawalski et al., 2024; Chen et al., 2025b] introduced Embedded Chain of Thought (ECoT) to improve the spatial reasoning ability of VLA models. RDT-1B and VPP [Liu et al., 2024; Hu et al., 2024] proposed diffusion-based frameworks for VLA models. Agibot world and Roboverse [Bu et al.,

- 2025a; Geng et al., 2025] aim to build larger-scale simulation environments and trajectory datasets to improve the sim-to-real transfer and generalization capabilities of VLA models. Additionally, Dexmimicgen [Jiang et al., 2024] explores automated methods to generate high-quality trajectory data to address the issue of data scarcity in robotics. Despite the rapid advancements in the VLA domain, imitation learning remains the dominant training paradigm for VLA models [Kim et al., 2024; Sapkota et al., 2025; Black et al., 2024; Liu et al., 2024; Kim et al., 2025; Bu et al., 2025b; Pertsch et al., 2025; Hung et al., 2025; Intelligence et al., 2025]. Current VLA models typically follow a two-stage paradigm: pretraining on multimodal data (e.g., Open X-Embodiment [O’Neill et al., 2024]) followed by SFT on collected robot trajectories. However, imitation learning is limited by its dependence on expensive, high-quality trajectory data and poor generalization to unseen scenarios.

VLA RL Methods Recently, some efforts have attempted to apply RL to VLA training. GRAPE [Zhang et al., 2024] utilized Direct Preference Optimization (DPO) [Rafailov et al., 2023] to train VLA models by integrating human preferences. ConRFT [Chen et al., 2025c] introduced Reinforced FineTuning [Trung et al., 2024] to train VLA models in real-world environments, iteratively training VLAs through alternating RL and SFT rounds. ReinboT [Zhang et al., 2025] focused on dense reward design and optimized VLA models through reward maximization. Guo et al. [2025b] proposed an iterative training framework that combines Supervised Fine-Tuning (SFT) and RL stages to address training instability and computational overhead. More recent works have further advanced VLA RL methods. Our work is one of the earliest systematic explorations of VLA online RL, and we publicly released the code in May 20253. Concurrently, RIPT-VLA [Tan et al., 2025] investigates a closely related problem, employing RLOO [Ahmadian et al., 2024] for VLA RL training. Moreover, Liu et al. [2025a] investigates RL’s impact on VLA generalization capabilities, demonstrating significant improvements over SFT in unseen environments, objects, and textures. RLinf [Team, 2025] designed a flexible, scalable framework for VLA RL that unifies rendering, inference, and training, improving both VLA training efficiency and performance. VLA-RL [Lu et al., 2025] applies the PPO algorithm to the VLA model. TGRPO [Chen et al., 2025d] uses Claude3.7 to evaluate trajectories and optimizes VLA with GRPO. RFTF [Shu et al., 2025] uses value models to generate dense rewards in embodied scenarios for VLA online RL. Compared to the above works, our paper further explores the effectiveness of VLA RL on real-world robotic tasks. We also conduct comprehensive analyses on how VLA RL addresses data scarcity challenges and improves policy generalization.

###### 8. Conclusion

In this work, we present SimpleVLA-RL, an RL framework tailored for VLA models. By extending veRL with VLA-specific trajectory sampling and parallelized training–inference–rendering capabilities, SimpleVLA-RL enables scalable and sample-efficient online RL. SimpleVLA-RL demonstrate significant improvements in data efficiency, generalization, and sim-to-real transfer. The consistent performance gains across LIBERO and RoboTwin benchmarks highlight the potential of RL not only to alleviate the data scarcity challenge of SFT but also to substantially enhance the generalization capacity of VLA models. We hope these findings pave the way for more autonomous and adaptable robotic models.

3https://github.com/PRIME-RL/SimpleVLA-RL

###### References

Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, 44(5):701–739, 2025.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, et al. A survey on vision-language-action models: An action tokenization perspective. arXiv preprint arXiv:2507.01925, 2025.

Ranjan Sapkota, Yang Cao, Konstantinos I Roumeliotis, and Manoj Karkee. Vision-language-action models: Concepts, progress, applications and challenges. arXiv preprint arXiv:2505.04769, 2025.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. arXiv preprint arXiv:2108.03298, 2021.

Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xu Huang, Shu Jiang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025a.

Jensen Gao, Annie Xie, Ted Xiao, Chelsea Finn, and Dorsa Sadigh. Efficient data collection for robotic manipulation via compositional generalization. arXiv preprint arXiv:2403.05110, 2024.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025a.

Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, and Yu Wang. What can rl bring to vla generalization? an empirical study. arXiv preprint arXiv:2505.19789, 2025a.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025b.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Julian Ibarz, Jie Tan, Chelsea Finn, Mrinal Kalakrishnan, Peter Pastor, and Sergey Levine. How to train your robot with deep reinforcement learning: lessons we have learned. The International Journal of Robotics Research, 40(4-5):698–721, 2021.

Oliver Kroemer, Scott Niekum, and George Konidaris. A review of robot learning for manipulation: Challenges, representations, and algorithms. Journal of machine learning research, 22(30):1–82, 2021.

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models. arXiv preprint arXiv:2310.12931, 2023.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Matthew Renze. The effect of sampling temperature on problem solving in large language models. In Findings of the association for computational linguistics: EMNLP 2024, pages 7346–7356, 2024.

Gustavo H De Rosa and Joao P Papa. A survey on text generation using generative adversarial networks. Pattern Recognition, 119:108098, 2021.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi_0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024.

Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, et al. Gr-3 technical report. arXiv preprint arXiv:2507.15493, 2025.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan. Inverse reward design. Advances in neural information processing systems, 30, 2017.

W Bradley Knox, Alessandro Allievi, Holger Banzhaf, Felix Schmitt, and Peter Stone. Reward (mis) design for autonomous driving. Artificial Intelligence, 316:103829, 2023.

Serena Booth, W Bradley Knox, Julie Shah, Scott Niekum, Peter Stone, and Alessandro Allievi. The perils of trial-and-error reward design: misdesign through overfitting and invalid task specifications. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 5920–5929, 2023.

Zheng Wu, Wenzhao Lian, Vaibhav Unhelkar, Masayoshi Tomizuka, and Stefan Schaal. Learning dense rewards for contact-rich manipulation tasks. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 6214–6221. IEEE, 2021.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025b.

Zihan Liu, Zhuolin Yang, Yang Chen, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron 1.1: Advancing math and code reasoning through sft and rl synergy. arXiv preprint arXiv:2506.13284, 2025c.

Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https://hkunlp.github. io/blog/2025/Polaris.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025a.

Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, and Jieyu Zhao. Efficient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520, 2025.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Mengqi Liao, Xiangyu Xi, Ruinian Chen, Jia Leng, Yangen Hu, Ke Zeng, Shuai Liu, and Huaiyu Wan. Enhancing efficiency and exploration in reinforcement learning for llms. arXiv preprint arXiv:2505.18573, 2025.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

Yao Mu, Tianxing Chen, Shijia Peng, Zanxin Chen, Zeyu Gao, Yude Zou, Lunkai Lin, Zhiqiang Xie, and Ping Luo. Robotwin: Dual-arm robot benchmark with generative digital twins (early version). In European Conference on Computer Vision, pages 264–273. Springer, 2025.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025b.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U Tan, Navonil Majumder, Soujanya Poria, et al. Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854, 2025.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 2024.

Yanjie Ze, Gu Zhang, Kangning Zhang, Chenyuan Hu, Muhan Wang, and Huazhe Xu. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. arXiv preprint arXiv:2403.03954, 2024.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. pi_ {0.5}: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Hongzhe Bi, Lingxuan Wu, Tianwei Lin, Hengkai Tan, Zhizhong Su, Hang Su, and Jun Zhu. H-rdt: Human manipulation enhanced bimanual robotic manipulation. arXiv preprint arXiv:2507.23523, 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025d.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025b.

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030, 2022.

Zhecheng Yuan, Tianming Wei, Shuiqi Cheng, Gu Zhang, Yuanpei Chen, and Huazhe Xu. Learning to manipulate anywhere: A visual generalizable framework for reinforcement learning. arXiv preprint arXiv:2407.15815, 2024.

Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, and Sergey Levine. Training strategies for efficient embodied reasoning. arXiv preprint arXiv:2505.08243, 2025b.

Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.

Haoran Geng, Feishi Wang, Songlin Wei, Yuyang Li, Bangjun Wang, Boshi An, Charlie Tianyue Cheng, Haozhe Lou, Peihao Li, Yen-Jen Wang, et al. Roboverse: Towards a unified platform, dataset and benchmark for scalable and generalizable robot learning. arXiv preprint arXiv:2504.18904, 2025.

Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Fan, and Yuke Zhu. Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning. arXiv preprint arXiv:2410.24185, 2024.

Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, Siwei Han, Chaoqi Wang, Mingyu Ding, Dieter Fox, and Huaxiu Yao. Grape: Generalizing robot policy via preference alignment. arXiv preprint arXiv:2411.19309, 2024.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025c.

Luong Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7601–7614, 2024.

Hongyin Zhang, Zifeng Zhuang, Han Zhao, Pengxiang Ding, Hongchao Lu, and Donglin Wang. Reinbot: Amplifying robot visual-language manipulation with reinforcement learning. arXiv preprint arXiv:2505.07395, 2025.

Yanjiang Guo, Jianke Zhang, Xiaoyu Chen, Xiang Ji, Yen-Jen Wang, Yucheng Hu, and Jianyu Chen. Improving vision-language-action model with online reinforcement learning. arXiv preprint arXiv:2501.16664, 2025b.

Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Krähenbühl. Interactive post-training for visionlanguage-action models. arXiv preprint arXiv:2505.17016, 2025.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

RLinf Team. Rlinf: Reinforcement learning infrastructure for agentic ai. https://github.com/ RLinf/RLinf, 2025. GitHub repository.

Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025.

Zengjue Chen, Runliang Niu, He Kong, and Qi Wang. Tgrpo: Fine-tuning vision-language-action model via trajectory-wise group relative policy optimization. arXiv preprint arXiv:2506.08440, 2025d.

Junyang Shu, Zhiwei Lin, and Yongtao Wang. Rftf: Reinforcement fine-tuning for embodied agents with temporal feedback. arXiv preprint arXiv:2505.19767, 2025.

