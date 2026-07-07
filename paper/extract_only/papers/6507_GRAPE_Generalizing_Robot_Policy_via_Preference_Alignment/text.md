##### GRAPE: Generalizing Robot Policy via Preference Alignment

Zijian Zhang*1 Kaiyuan Zheng*2 Zhaorun Chen*3 Joel Jang2 Yi Li2 Siwei Han1 Chaoqi Wang3 Mingyu Ding1 Dieter Fox2 Huaxiu Yao1

# arXiv:2411.19309v2[cs.RO]4Feb2025

###### Abstract

Despite the recent advancements of visionlanguage-action (VLA) models on a variety of robotics tasks, they suffer from critical issues such as poor generalizability to unseen tasks, due to their reliance on behavior cloning exclusively from successful rollouts. Furthermore, they are typically fine-tuned to replicate demonstrations collected by experts under different settings, thus introducing distribution bias and limiting their adaptability to diverse manipulation objectives, such as efficiency, safety, and task completion. To bridge this gap, we introduce GRAPE: Generalizing Robot Policy via Preference Alignment. Specifically, GRAPE aligns VLAs on a trajectory level and implicitly models reward from both successful and failure trials to boost generalizability to diverse tasks. Moreover, GRAPE breaks down complex manipulation tasks to independent stages and automatically guides preference modeling through customized spatiotemporal constraints with keypoints proposed by a large vision-language model. Notably, these constraints are flexible and can be customized to align the model with varying objectives, such as safety, efficiency, or task success. We evaluate GRAPE across a diverse array of tasks in both real-world and simulated environments. Experimental results demonstrate that GRAPE enhances the performance of state-of-the-art VLA models, increasing success rates on in-domain and unseen manipulation tasks by 51.79% and 58.20%, respectively. Additionally, GRAPE can be aligned with various objectives, such as safety and efficiency, reducing collision rates by 37.44% and rollout step-length by 11.15%, respectively.

*Equal contribution 1UNC-Chapel Hill, Chapel Hill, NC, USA 2University of Washington, Seattle, WA, USA 3University of Chicago, Chicago, IL, USA. Correspondence to: Huaxiu Yao <huaxiu@cs.unc.edu>.

Preprint

Real-In-domain

Sim-Semantic

Real-Visual

48.33

34.67

39.77

Sim-Physical

Real-Subject

24.17

17.33

19.88

31.17

38.33

15.58

19.17

15.22 30.45

15.0

30.0 18.33

29.33

17.23

Sim-Subject

Real-Action

38.67

36.67

34.47

Sim-In-Domain

Real-Semantic Real-Language

Octo-SFT OpenVLA-SFT OpenVLA-DPO GRAPE

Figure 1: Comparison of GRAPE with SOTA VLA models fine-tuned on the same data across a large variety of generalization and in-domain tasks in both real-world and simulated environments.

###### 1. Introduction

The recent rapid proliferation of vision-language-action (VLA) models has streamlined general robotic manipulation tasks, demonstrating impressive capability across a range of tasks under controlled environmental variations (Black et al., 2024; Kim et al., 2024; Team et al., 2024; Brohan

- et al., 2023). However, these models face several critical challenges such as poor generalizability across new environments, objects, tasks, and semantic contexts (Kim et al., 2024). A significant factor contributing to this limitation is their reliance on supervised fine-tuning (SFT), where VLAs simply imitate actions from successful rollouts via behavior cloning while not developing a holistic understanding of the task goal or potential failure patterns (Kumar et al., 2021). While reinforcement learning (RL) algorithms such as PPO (Schulman et al., 2017) have proved promising in enhancing their generalizability (Zhai et al., 2024), the high cost of gathering sufficient online trajectories and explicitly defining reward make them impractical for training VLA (Team et al., 2024).

Furthermore, training VLAs to solely replicate expert behaviors often results in behavior collapse (Kumar et al., 2024) where the planned trajectories are often suboptimal (Kim

- et al., 2024). This is because the SFT datasets are usually

- (a) Temporal Stage Decomposition

- (b) Spatial-Temporal Keypoint Proposal

Cost Function

Customized Cost Generation

Customized Alignment Goals

|[Figure 1]<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>…<br><br>Stage-𝟏 Stage-𝒏|
|---|

Stage-𝟏: Pick up grape

[Figure 5]

[Figure 6]

[Figure 7]

Task Completion Safety

[Figure 8]

Pick up the grape and put it into the pot

Efficiency Resilience …

[Figure 9]

Task Instruction

|…<br><br>[Figure 10]<br><br>1 2<br><br>1 3 2<br><br>[Figure 11]<br><br>4 3| |
|---|---|
| | |

Vision Language Model

[Figure 12]

…

[Figure 13]

Stage-𝟐: Move over to pot

[Figure 14]

…

GPT-4o

Initial State

Stage-𝟑: Drop into the box

…

Iterative Preference Optimization

[Figure 15]

[Figure 16]

[Figure 17]

Objective-aligned Multi-stage Cost

[Figure 18]

[Figure 19]

Pick up the grape and put it into the pot

[Figure 20]

[Figure 21]

High

- Traj 1

- Traj 2

Traj-wise Preference

[Figure 22]

[Figure 23]

[Figure 24]

Trajectory Sampled Set

Model Self-Reward

User Instruction

[Figure 25]

Online Sampling

Preference Aligned VLA

SFT Base VLA Model

Optimization

…

[Figure 26]

[Figure 27]

[Figure 28]

Traj M

Task Success Binary Indicator

Low

Preference Ranking

Training Tasks

Iterative Online Sampling

- Figure 2: Overview of GRAPE. GRAPE first uses a VLM to decompose a manipulation task (top) into temporal stages and identify key spatial points for each subtask. Given user-specified alignment goals, it prompts a VLM to generate cost functions for each stage. During iterative preference optimization (bottom), offline trajectories are sampled from the base VLA model, scored using multi-stage cost, self-evaluation and task success indicators, and ranked to form preferences. GRAPE then optimizes the VLA models iteratively until convergence.

uncurated and consist of offline demonstrations collected from experts that embed implicitly different values (e.g. task completion, safety, and cost-efficiency) that are not clearly defined within the data (O’Neill et al., 2023; Walke

customized to align the model with varying manipulation objectives, such as task completion, robot-interaction safety, and cost-efficiency. We evaluate GRAPE across a wide range of real-world tasks and two simulated environments. Experimental results show that GRAPE outperforms stateof-the-art VLA models, improving success rates on both in-domain and unseen manipulation tasks by 51.79% and 58.20%, respectively. Moreover, GRAPE can be aligned to diverse objectives such as safety and efficiency, to further reduce collision rate by 37.44% and rollout step-length by 11.15%, respectively.

- et al., 2023). Simply imitating these behaviors via SFT can potentially confuse the model and result in suboptimal trajectories that deviate from the actual objective of the demonstrations. Some approaches attempt to address this challenge by explicitly defining a set of objectives and solving them hierarchically (Huang et al., 2024). However, this approach incurs additional inference overhead and lacks scalability (Li et al., 2024b).

###### 2. Generalizing Robot Policy via Preference Alignment

To address these issues, we propose GRAPE: Generalizing Robot Policy via Preference Alignment to alleviate the high cost of training VLAs with RL objective, while offering flexibility for aligning towards customized manipulation objectives. As shown in Figure 2, GRAPE introduces trajectory-wise preference optimization (TPO) to align VLA policies on a trajectory level by implicitly modeling reward from both successful and failure trials, boosting generalizability to diverse tasks. To further alleviate the difficulty in ranking trajectories and providing preferences towards arbitrary alignment objectives, GRAPE proposes to decompose the complex manipulation tasks into multiple independent stages and adopt a large vision model to propose keypoints for each stage, each associated with a spatial-temporal constraint. Notably, these constraints are flexible and can be

###### 2.1. Preliminaries

During inference, a VLA typically initializes with an task instruction q, and at each timestep t, it takes an environment observation ot (usually an image) and outputs an action at, where we can denote πθ(ai|(oi,q)) as the action policy of a VLA parameterized by θ. To complete the task, VLA iteratively interacts with the environment and obtains a trajectory ζ = {o1,a1,··· ,oT,aT|q} of length T. Typically, VLAs are fine-tuned to imitate expert behaviors via SFT:

T

log p(at|ot, q; πθ), (1)

###### LSFT = −

t=1

(ζ,q)∈D

where D = {(ζ1,q1),...,(ζN,qN)} denotes the training set containing N expert trajectories. Specifically, LSFT enforces VLA to memorize the action associated with each observation sampled from a distribution PD, resulting in poor generalizability to new task settings. It is worth to note that while we follow O’Neill et al. (2023); Brohan et al. (2023) and consider the step-wise policy based on the Markov decision process (MDP) assumption (Sutton, 2018), our approach can be easily adapted to both non-MDP case which takes past interaction histories (usually a video or a series of images) as state (Cheang et al., 2024) and diffusion policy (Chi et al., 2023) which generates multiple future steps all at once (Team et al., 2024).

###### 2.2. TPO: Trajectory-wise Preference Optimization

To improve generalization, we follow Schulman et al. (2017); Bai et al. (2022) and further fine-tune VLA policies via RL objective. Let rϕ denote a reward function parameterized by ϕ, we have

Eζ∼πθ [rϕ(ζ)] − βDKL [πθ(ζ) ∥ πref(ζ)] , (2)

max

πθ

where β controls the deviation from the base reference policy πref trained via SFT in Eq. (1) and π(ζ,q) is the likelihood of policy π generating the entire trajectory ζ under instruction q. Then we follow Rafailov et al. (2024) and derive the analytical reparameterization of the trajectory reward r(ζ) as:

πθ(ζ | q) πref(ζ | q)

+ β log Z(ζ). (3)

r(ζ, q) = β log

Similar to Rafailov et al. (2024), we adopt the Bradley-Terry (BT) (Bradley & Terry, 1952) model and model rϕ from a set of trajectories ranked with preferences. Specifically, let ζw and ζl denotes the chosen and rejected trajectory starting from the same initial state, we can formulate the trajectory-wise reward modeling objective as:

exp (r(ζw), q) exp (r(ζw), q) + exp (r(ζl), q)

. (4)

P (ζw ≻ ζl) =

Then, we follow Rafailov et al. (2024) and substitute Eq. (3) into Eq. (4) and obtain the following trajectory-wise preference optimization (TPO) loss LTPO equivalent to Eq. (2):

πθ(ζl) πref(ζl)

πθ(ζw) πref(ζw) − log

LTPO = −E(ζw,ζl)∼D log σ β log

(5)

where we can further draw from MDP and decompose the likelihood of a trajectory ζ into individual state-action pairs, i.e., π(ζ, q) = Ti=1 π(ai | (oi, q)) and further obtain

T

πθ(ai | (oi, q)) πref(ai | (oi, q))

πθ(ζ, q) πref(ζ, q)

. (6)

log

=

log

t=1

Then we can substitute Eq. (6) into Eq. (5) to obtain the TPO loss LTPO in terms of step-wise state-action pairs. Our TPO loss Eq. (6) is beneficial as it: (1) aligns policy πθ globally towards human preferences on a trajectory level while simply using step-wise rollouts collected by VLAs; (2) it stabilizes the policy and steers it towards the final goal by backpropagating the gradients throughout all the stateaction pairs along the trajectory; (3) it significantly boosts generalizability by learning from both successful and failed trajectories via a RL objective. Although Finn et al. (2016) indicates that expanding the size of the sampled trajectory can reduce the bias in reward modeling, it also increases the training costs. Thus while our method can be easily scaled up, we keep our discussion to the binary case where only one chosen/rejected trajectory is present.

###### 2.3. Guided-Cost Preference Generation

While given the TPO objective Eq. (5) we can align the policy towards arbitrary objectives defined through trajectories ranked by the corresponding preference, it incurs high costs as it requires human expertise and lengthy manual annotation. Thus to better scale up the preference synthesis towards arbitrary alignment objectives (e.g. task completion, safety, efficiency), we propose Guided-Cost Preference Generation (GCPG) to automatically curate such preferences that integrate different alignment objectives.

2.3.1. MULTI-STAGE TEMPORAL KEYPOINT CONSTRAINTS

Building on insights from Huang et al. (2024), we address the complexity of specifying precise trajectory preferences for complex manipulation tasks by decomposing trajectories into temporal stages and assigning costs to quantify performance at each stage. Then, we aggregate these stagespecific costs to obtain a holistic evaluation for each trajectory. Specifically, we adopt a VLM-based stage decomposer MD (detailed in Appendix A), to partition a trajectory ζ into a sequence of S consecutive stages, formulated as

{ζ1, . . . , ζS} = MD(ζ, q), ζi = {(oit, ait)}Tt=1i , (7)

where ζi represents the ith stage of trajectory ζ.

After obtaining the stage decomposition, we further employ a vision-language model (e.g. DINOv2 (Oquab et al., 2023)) to identify keypoints that serve as reference metrics across each stage. Then we prompt a powerful LLM (Achiam et al., 2023) to propose cost functions (see examples in Appendix E.2.) for each stage that corresponds with the alignment objective, where lower cost indicates better objective compliance. Specifically, the cost CS

,

i}) at stage Si is calculated using its corresponding keypoints {κS

({κS

i

i}.

Then to aggregate the costs for the entire trajectory, instead of summing each stage linearly, we apply an exponential

decay to capture the casual dependencies of each temporal stage (e.g. if a trajectory incurs high costs in preceding stages it is not expected to perform well subsequently), defined as the external reward:

###### S

Rext(ζ) =

i=1

Si({κSi}) (8)

e−C

where Eq. (8) aggregates the individual costs and subobjectives from each stage to tackle the curse of dimensionality and effectively adhere to the customized alignment.

- 2.3.2. GUIDED-COST PREFERENCE GENERATION

To further improve the stability and optimality of the preference synthesis, we draw inspirations from selfrewarding (Zhou et al., 2024b) and determine that a more optimal trajectory should be confirmed by both the external judge (as in Eq. (8)) and the model itself. Thus we incorporate two additional rewards and obtain the GCPG reward:

RGCPG(ζ) = λ1Rself(ζ) + λ2Rext(ζ) + λ3Isuccess(ζ) (9)

where Rself(ζ) is the self-evaluated score provided by π, which equals the log-likelihood of generating trajectory ζ:

Rself(ζ) = log(π(ζ, q)) = log(

T

i=1

π(ai | (oi, q))) (10)

and Isuccess(ζ) is a binary indicator function that indicates whether the trajectory ζ successfully completes the task:

Isuccess(ζ) =

1, if ζ is successful, 0, otherwise.

(11)

where λ are the weight parameters that adjust the importance of each reward. Intuitively, Eq. (10) can be seen as a dense approximation of the sparse signal provided by Eq. (11), which are further calibrated by Eq. (8) to obtain a holistic evaluation of the trajectory that accounts for both its optimality and degree of alignment to a customized objective specified through the external reward in Eq. (8).

- 2.4. Iterative Preference Optimization

After generating the preference, we then discuss our iterative preference optimization strategy. Inspired by the practices of on-policy RL (Schulman et al., 2017) which often yield more optimal policy than off-policy training, we iteratively fine-tune the SFT VLA model via TPO with trajectories collected online. For example, during the kth iteration, we (1) first sample numerous trajectories for a variety of tasks and obtain Dk; (2) then we calculate the costs for each trajectory using Eq. (9) and rank these trajectories accordingly per task; (3) we pair the top-m and bottom-m trajectories

Algorithm 1 GRAPE Iterative Preference Optimization

Require: Base VLA policy πθ, a collection of task instructions Q = {qi}, stage decomposer MD, max iterations K, reward weights {λ1,λ2,λ3}, stage-wise keypoints {κS

- i
- j } and thresholds {τS

i} cost functions {CS

- i
- j }

Ensure: policy π∗ aligned towards customized objective

- 1: for k = 1,...,K do
- 2: Sample trajectories Dk = {ζi}Mi=1 using πθ with Q
- 3: for trajectory ζ ∈ Dk do
- 4: Decompose ζ into multiple stages S {Eq. (7)}
- 5: Compute the cost for each stage CS

i

- 6: Calculate external reward Rext(ζ) {Eq. (8)}
- 7: Compute policy self-reward Rself(ζ) {Eq. (10)}
- 8: Examine task success Isuccess(ζ) {Eq. (11)}
- 9: Aggregate GCPG reward RGCPG(ζ) {Eq. (9)}
- 10: end for
- 11: Rank Dk by their RGCPG(ζ) rewards
- 12: Pair {ζw,ζl} from top-m and bottom-m trajectories
- 13: Update πθ using TPO loss {Eq. (5)}
- 14: end for

with each other for each task, and obtain m2 chosen-rejected trajectory pairs; (4) then we fine-tune the same sampling policy with TPO via Eq. (5) and obtain an updated policy. We iterate this process for K times and obtain the final model aligned with the target objective. We detail the GRAPE iterative preference optimization procedure in Algorithm 1.

###### 3. Experiment

In this section, we evaluate GRAPE’s performance in both real and simulated environments, addressing four key questions: (1) Does GRAPE improve the VLA model’s performance relative to SFT-based baseline models? (2) How effective are guided-cost preference selection and iterative preference optimization in enhancing the model’s performance? (3) What is the individual contribution of each reward component to overall model performance? (4) Can GRAPE support flexible alignment with different alignment objectives?

###### 3.1. Experimental Setups

Implementation Details. We employ OpenVLA (Kim et al., 2024) as the backbone model, using LoRA fine-tuning with the AdamW optimizer for both supervised and preference fine-tuning. In the supervised fine-tuning stage, we use a learning rate of 4 × 10−5 with a batch size of 16. For preference fine-tuning, we apply a learning rate of 2×10−5 with the same batch size. Further details on the training process and datasets are available in Appendices A and B.

Baseline Models. We first compare GRAPE with two leading robot learning models known for their strong perfor-

47.0

43.1

43.0

40.7 41.8

SUCCESSRATE(%)

40.0

34.7 33.8 34.5

32.7

33.7

32.0

29.5 28.0 29.5

28.0

24.0 25.0

18.6

20.0

| | |
|---|---|
|14.0<br><br>11.5| |

0.0

In-domain Subject Generalization Physical Generalization Semantics Generalization Average

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

put carrot on the plate

move the eggplant into the basket

stack green cube on the yellow cube

put the coke can on the towel

- Figure 3: Comparison of GRAPE with OpenVLA and Octo fine-tuned on the same data on the Simpler-Env environment. We report the in-domain performance, which includes four tasks and three generalization evaluations (subject, physical, and semantic), where each incorporates multiple tasks. mance in robot control tasks. The first model, Octo (Team

- et al., 2024), is a large transformer-based policy model. The second, OpenVLA (Kim et al., 2024), is a 7B VLA model. Both models were supervised fine-tuned using the same dataset sampled from corresponding environments. We denote the supervised fine-tuned models as Octo-SFT and OpenVLA-SFT, respectively. In addition, we compare GRAPE, which utilizes trajectory-wise preference optimization, with the original step-wise direct preference optimization, denoted as OpenVLA-DPO, which is directly trained to optimize preferences defined at each step.

###### 3.2. Evaluation in Simulation Environment

Evaluation Setup. Follow Kim et al. (2024), we evaluate GRAPE’s performance in two robot simulation environments: Simpler-Env (Li et al., 2024a) and LIBERO (Liu et al., 2023). In Simpler-Env, we evaluate the model’s indomain performance as well as its generalization across three aspects: subject (generalize to unseen objects), physical (generalize to unseen object sizes/shapes), and semantic (generalize to unseen instructions) generalization. In LIBERO, we test our model on four tasks: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. All tasks are in-domain tasks. Additional details about the experimental setup are provided in Appendix C.2.

Results. We use the success rate across all tasks in SimpleEnv and LIBERO as our primary evaluation metric, while we also record the grasping rate in Simpler-Env. The results of Simple-Env and LIBERO are reported in Figure 3 and Figure 4, respectively. According to the results, GRAPE outperforms Octo-SFT and OpenVLA-SFT in Simpler-Env by an average of 131.72% and 46.10%, respectively, and in LIBERO by an average of 8.53% and 7.36%, respectively. Additional results are provided in Appendix D. This outcome aligns with our expectations, as learning from preference comparisons enhances alignment with trajectory completion, thereby improving performance. Moreover, while GRAPE significantly boosts in-domain performance,

92.1

88.6

88.5

87.3

80.2

84.9 82.9

84.2

83.1

82.1 73.9

74.7

76.2

85.0

79.5

SUCCESSRATE(%)

77.4

77.6

65.0

57.2

52.6

51.8

50.3

45.0

LIBERO-Spatial LIBERO-Object LIBERO-Goal LIBERO-Long Average

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

pick up the black bowl on the stove and place it on the plate

pick up the alphabet soup and place it in the basket

open the middle drawer of the cabinet

put both moka pots on the stove

Figure 4: Comparison of GRAPE with OpenVLA and Octo fine-tuned on the same data on the LIBERO environment. We report the performance on four types of LIBERO tasks.

it also enhances the generalizability of VLA policies on OOD tasks by aligning task completion at the trajectory level. Furthermore, GRAPE outperforms OpenVLA-DPO in both environments, achieving an average improvement of 33.14%, demonstrating the effectiveness of trajectory-wise preference optimization due to learning from both success and failure from a global trajectory level without low-level step-wise noises.

###### 3.3. Evaluation in Real-World Robot Environment

Evaluation Setup. We conducted 300 real-world experiments across 30 tasks to evaluate the generalization capabilities of GRAPE. The evaluation focus on in-distribution evaluation and five out-of-distribution generalization types: visual, subject, action, semantic, and language grounding generalizations. Here, visual generalization assesses the ability to adapt to new visual environments; subject generalization evaluates the recognition and handling of unfamiliar objects; action generalization measures performance across diverse actions; semantic generalization evaluates responses to prompts with similar meanings; and language grounding generalization gauges comprehension of spatial directions. Detailed experimental setup are provided in Appendix C.1 and illustrated in Figure 5.

Results. In the real-world experiment, GRAPE significantly outperforms other models across a variety of tasks. Notably, in in-domain tasks, GRAPE achieves a success rate of 67.5%, which is a 17.5% improvement over OpenVLADPO’s 50%, OpenVLA-SFT’s 45% and substantially higher than Octo-SFT’s 20%. Additionally, in visual generalization tasks, GRAPE demonstrates higher adaptability with a success rate of 56%. In the more challenging action generalization tasks, although OpenVLA-SFT shows modest performance, GRAPE still outperforms OpenVLA-SFT, indicating its potential in understanding various actions and executing commands based on language. Considering tasks across all categories, GRAPE’s total average success rate is 50.3%, marking a 11% improvement over OpenVLADPO’s 39.3%, OpenVLA-SFT’s 32.3% and significantly

80.0

67.5

SUCCESSRATE(%)

55.0 52.5

60.0

50.3

50.0

50.0 46.7

45.0

45.0

45.0

42.5

39.3

37.5

35.7

35.0

33.3

40.0

| | |
|---|---|
|30.0 32.3| |

25.0 24.3

20.0

20.0

20.0

5.7

5.0 5.0 4.3

0.0 0.0

0.0

In-domain Visual Gen. Subject Gen. Action Gen. Semantics Gen. Language Grounding Average

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

fold the green towel from right to left

pick up the banana and put it in the black bowl

pick up the grape and put it in the black bowl

pick up the chips and put it in the red bowl

take green pepper and place it in the black bowl

push down the middle button

- Figure 5: Comparison of GRAPE with OpenVLA and Octo fine-tuned on the same data on the real-world environment. We report the in-domain performance, which includes four tasks and five generalization evaluations (visual, subject, action, semantic, and language grounding), where each incorporates multiple tasks. We also report the average performance across all tasks.

| | | | | | |
|---|---|---|---|---|---|
| |71.<br><br>GRAPE GRAPE<br><br>|00<br><br>Grasp Success|74.|00 74.|50|
| |SFT Gra SFT Suc|sp cess| | | |
|51.00| | | | | |
| | | | | | |
| |43.|00<br><br>45.| | |50|
| | | | |00 45.| |
|28.00| | | | | |
| | | | | | |
| | | | | | |

SFT iter 1 iter 2 iter 3

In-domain

30

40

50

60

70

80

SuccessRate(%)

SFT iter 1 iter 2 iter 3

Subject Gen.

35

40

45

50

55

60

65

SuccessRate(%)

60.00

62.66

64.33 64.66

32.67

40.66 40.34

41.67

GRAPE Grasp

GRAPE Success

SFT Grasp

SFT Success

| | | | | | |
|---|---|---|---|---|---|
| |63.<br><br>GRAPE GRAPE<br><br>|50<br><br>Grasp Success|65.|75 66.|00|
| |SFT Gra|sp| | | |
| |SFT Suc|cess| | | |
| | | | | | |
|46.75| | | | | |
| | |44.| |25 44.|50|
| |41.|75| | | |
| | | | | | |
|29.50| | | | | |
| | | | | | |
| | | | | | |

SFT iter 1 iter 2 iter 3

Physical Gen.

30

35

40

45

50

55

60

65

70

SuccessRate(%)

| | | | | |
|---|---|---|---|---|
| |72.|00<br><br>76.| |00 76.00|
| | | | | |
| | | | | |
|8.|00 47.|00<br><br>49.| |50 49.00|
| | | |GRA|PE Grasp|
|8.|00| |GRA<br><br>SFT SFT<br><br>|PE Success Grasp Success|
| | | | | |
| | | | | |

SFT iter 1 iter 2 iter 3

Semantics Gen.

30

40

50

60

70

80

SuccessRate(%)

- Figure 6: Performance of GRAPE during iterative preference optimization via TPO. We demonstrate the average success rate for each iteration across in-domain tasks and three types of generation tasks (subject, physical, semantics). ahead of Octo-SFT’s 5.7%. This performance highlights

on success alone; (2) all reward components contribute to model performance. These findings align with our expectations. Specifically, Rself(ζ) enhances the robustness of the GRAPE by encouraging it to select trajectories with higher generation probabilities. In parallel, Rext(ζ) guides the model toward learning specific behaviors, such as safety and efficiency. Finally, Isuccess(ζ) serves as a critical indicator, steering the model to prioritize successful trajectories.

- (1) GRAPE’s effectiveness and adaptability in handling complex and variable task environments and (2) validates the effectiveness of trajectory-wise preference optimization in learning from global success and failure patterns when compared to OpenVLA-DPO.

###### 3.4. Ablation Study of Reward Model

In this section, we conduct an ablation study to analyze the contribution of each reward component in Eq. (9) to the final performance: the external objective-aligned reward Rext(ζ), the self-evaluated reward Rself(ζ), and the success indicator Isuccess(ζ). Additionally, we perform a separate ablation study to emphasize the importance of utilizing the entire reward score for preference selection. This approach is compared against a method that randomly selects one successful trajectory as the preferred trajectory and one failed trajectory as the rejected trajectory. The results in the Simpler-Env environment are reported in Table 1.

The results indicate that: (1) incorporating the full reward score Eq. (9) for preference ranking significantly enhances performance compared to random selection based

###### 3.5. Analysis of Iterative Preference Optimization

In this section, we analyze the iterative preference optimization performance. We conduct the experiments on the Simpler-Env environment and report the results with respect to the training iterations in Figure 6. Here, SFT means the supervised fine-tuned OpenVLA model before preference optimization. In our experiments, GRAPE achieves 17.5%, 9.0%, 15.0%, 21.0% improvements in in-domain performance, subject generalization, physical generalization and semantic generation, respectively. The findings suggest that GRAPE progressively enhances model performance across iterations, showcasing its ability to enhance the quality of generated preference data and achieve better

- Table 1: Ablation study of reward score. Here, Random w/ Isuccess refers to randomly selecting one successful trajectory as the chosen trajectory and one failed trajectory as the rejected trajectory, Rself(ζ) is the self-evaluated score provided by the log-likelihood of generating trajectory ζ, Rext(ζ) represents objective-aligned multi-stage reward defined in Eq. (8), Isuccess(ζ) is a binary indicator function that indicates whether the trajectory ζ successfully completes the task.

In-domain Subject Gen. Physical Gen. Semantics Gen. Average

Grasp Success Grasp Success Grasp Success Grasp Success Grasp Success Random w/ Isuccess 62.00% 35.50% 60.33% 33.00% 44.00% 33.50% 54.50% 36.50% 55.21% 34.63% w/o Rself(ζ) 66.50% 38.00% 62.33% 37.00% 51.25% 36.75% 68.00% 42.50% 62.02% 38.56% w/o Rext(ζ) 63.50% 37.50% 61.00% 34.33% 48.50% 35.50% 62.50% 40.00% 58.88% 36.83% w/o Isuccess 58.50% 32.00% 59.67% 34.67% 42.25% 31.75% 58.50% 39.00% 54.73% 34.36% GRAPE 71.00% 43.00% 62.67% 40.67% 63.50% 41.75% 72.00% 47.00% 67.29% 43.11%

- Table 2: Results with respect to different objectives. GRAPE-Safety, GRAPE-Efficiency, GRAPE-TC are models trained with safety, efficiency, task completion objectives, respectively. Here, we use collision rate (CR), step length (SL), success rate (SR) to evaluate the safety, efficiency and task completion capabilities.

The results are reported in Table 2, where we use collision rates, step lengths, and success rates to evaluate safety, efficiency and generalization capabilities, respectively. According to Table 2, the GRAPE-Safety and GRAPE-Efficiency have better performance on collision rate and step length respectively, meanwhile maintain a comparable success rate, compared with OpenVLA-SFT. The results indicate that GRAPE can be easily adapted to account for flexible alignment objectives such as safety, efficiency by adjusting the multi-stage cost functions accordingly, while incurring minimal drop in task success rate.

###### Real-World Simulation CR ↓ SL ↓ SR ↑ CR ↓ SL ↓ SR ↑

Method

OpenVLA-SFT 53.33 142.32 34.61 66.50 72.68 27.50 GRAPE-Safety 29.84 146.11 54.31 46.00 74.49 37.00 GRAPE-Efficiency 58.45 125.79 51.67 57.50 64.92 38.50 GRAPE-TC 38.60 131.66 58.46 59.50 70.24 42.50

- 3.6.2. CASE STUDY

We further demonstrate a case study in Figure 7 to analyze GRAPE’s adaptability towards different alignment objectives. Specifically, we consider a safety-critical pickup task where an obstacle is placed between the object and the target. Specifically, OpenVLA-SFT fails to complete the task without preference alignment. However, we can see that while GRAPE aligned towards task completion (on the secondrow of Figure 7) can effectively pick up and place the object, it also collides with the obstacle, due to the policy is aligned to aggressively boost task success without explicitly addressing safety concerns. On the contrary, GRAPE-safety learns to avoid colliding with the obstacle while efficiently completing the task. Both Table 2 and Figure 7 indicates that by simply tweaking the cost function, GRAPE can effectively adapt to different objectives. More cases and detailed safety evaluation tasks could be found in Appendix E.1.

- 4. Related Works

generalization. Notably, the magnitude of improvement diminishes over time, aligning with our expectations as the model approaches convergence.

3.6. Analysis of Different Alignment Objectives 3.6.1. QUANTITATIVE ANALYSIS

After demonstrating the effectiveness of GRAPE in improving the generalization of the VLA model (measured by success rate), we further investigate its potential to align the model with flexible objectives, such as efficiency and safety. Revisiting Eq. (8), we observe that adjusting the threshold parameters can guide the model to prioritize specific objectives by influencing trajectory preference selection. In this study, we focus on two new alignment objectives: safety and efficiency. Safety aims to minimize collisions between the robot and objects, while efficiency seeks to reduce the average number of steps required for the robot to complete a task. To achieve these objectives, we set a lower threshold for collision costs to emphasize safety and a lower threshold for path costs to prioritize efficiency. These modified settings are then applied to the original real-world and simulation evaluations. We train models to align with the safety and efficiency objectives, referring to these models as GRAPESafety and GRAPE-Efficiency, respectively (see detailed experimental setup in Appendix C.2).

Vision-Language-Action Models. Previous robot learning works (Huang et al., 2024; Li et al., 2024b; Chen et al., 2024c; Mu et al., 2024a; Huang et al., 2023; Liang et al., 2023a; Mu et al., 2024b) typically take a hierarchical planning strategy. For example, Code as Policies (Liang et al., 2023a) and EmbodiedGPT (Mu et al., 2024b) use LLMs and VLMs to generate high-level action plans, then rely on a low-level controller for local trajectories. However, such models suffer from limited low-level skills and are hard to

[Figure 46]

[Figure 47]

[Figure 48]

OpenVLA-SFT

[Figure 49]

- Figure 7: Comparison of GRAPE aligned via safety objective (GRAPE-Safety) with GRAPE aligned via task-completion (GRAPE-TC) objective and OpenVLA-SFT. Specifically, we assess their performance on a safety-critical task with the instruction: pick up the white box and place into the black pot.

generalize to everyday tasks. VLAs tend to scale up lowlevel tasks by incorporating VLM as backbones and directly generating actions within the model. They generally achieve action planning via two mainstream approaches: (1) Discretizing the action space (Kim et al., 2024; Brohan et al., 2023; 2022), as in OpenVLA (Kim et al., 2024), preserves the autoregressive language decoding objective by truncating actions into a small set of action tokens. However, this introduces errors, leading some methods (Black et al., 2024) to adopt newer structures (Zhou et al., 2024a) that integrate diffusion heads for action prediction, avoiding discretization.

- (2) Diffusion models (Chi et al., 2023; Xian et al., 2023; Janner et al., 2022; Liang et al., 2023b; Ajay et al., 2023), such as Diffusion Policy (Chi et al., 2023), serve as the action head, generating a sequence of future actions through iterative denoising instead of stepwise action generation.

While these models vary in structure, they are consistently supervised-trained on successful rollouts via behavior cloning, which can hardly be generalized to unseen manipulation tasks. However, our GRAPE first aligns VLA policies on a trajectory level via trial and error, effectively boosting generalizability and customizability.

Reinforcement Learning and Preference Optimization. Reinforcement learning (RL) (Christiano et al., 2017; Ziegler et al., 2019; Schulman et al., 2017) plays a pivotal role in the post-training of foundation models (Dubey et al., 2024; Achiam et al., 2023; Chen et al., 2024d;a; Fan et al., 2024; Yang et al., 2024; Chen et al., 2024b; Wang

- et al., 2025), which has been extensively leveraged to align the pre-trained FMs to comply with human values embedded through preference data. In the meantime, RL has also shown tremendous success in training policies for robotics tasks (Chen et al., 2024c; Wang et al., 2024b; Chen et al., 2021; 2022; Zhu et al., 2019; Wu et al., 2024). While it is

intuitively beneficial to post-align VLA via RL, few prior works have reported such success, mainly due to that (1) manipulation objectives are usually diverse and complex, making the reward hard to define analytically (Finn et al., 2016); (2) while such reward can be modeled from human preferences, annotating such preferences in robotics manipulation tasks are usually lengthy (Walke et al., 2023); (3) the imperfect numerical differentiation of rewards usually leads RL algorithms such as PPO (Schulman et al., 2017) to collapse (Bu¸soniu et al., 2018). However, various recent works (Rafailov et al., 2024; Wang et al., 2024a) have successfully aligned the policy via RL without explicit reward modeling. Inspired, GRAPE aligns the policy by contrasting trajectories with each other, avoiding issues in rewarding modeling. Besides, we introduce an automatic preference synthesis pipeline that easily scales with diverse manipulation tasks and adapts to different alignment objectives.

###### 5. Conclusion

In this work, we addressed the critical challenges faced by vision-language-action (VLA) models, including limited generalizability and adaptability to diverse manipulation objectives. We proposed GRAPE, which aligns VLA policies on a trajectory level. GRAPE enhances generalizability by learning from both successful and failed trials, offering flexibility in aligning with objectives such as safety, efficiency, and task success through customized spatiotemporal constraints. Experimental results demonstrated significant improvements, with GRAPE enhancing success rates on both in-domain and unseen tasks while enabling flexible alignment on different objectives. Moreover, we have demonstrated the potential of GRAPE to align VLA with customized objectives, effectively resulting in an improvement of lower collision rate and average step lengths.

###### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

###### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Ajay, A., Du, Y., Gupta, A., Tenenbaum, J. B., Jaakkola, T. S., and Agrawal, P. Is conditional generative modeling all you need for decision making? In The Eleventh International Conference on Learning Representations, 2023.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter,

- B., et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Bu¸soniu, L., De Bruin, T., Toli´c, D., Kober, J., and Palunko, I. Reinforcement learning for control: Performance, stability, and deep approximators. Annual Reviews in Control, 46:8–28, 2018.

Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., and Mordatch, I. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34:15084–15097, 2021.

- Chen, Y., Wu, T., Wang, S., Feng, X., Jiang, J., Lu, Z., McAleer, S., Dong, H., Zhu, S.-C., and Yang, Y. Towards human-level bimanual dexterous manipulation with reinforcement learning. Advances in Neural Information Processing Systems, 35:5150–5163, 2022.
- Chen, Z., Du, Y., Wen, Z., Zhou, Y., Cui, C., Weng, Z., Tu, H., Wang, C., Tong, Z., Huang, Q., et al. Mjbench: Is your multimodal reward model really a good judge for text-to-image generation? arXiv preprint arXiv:2407.04842, 2024a.

Chen, Z., Pinto, F., Pan, M., and Li, B. Safewatch: An efficient safety-policy following video guardrail model with transparent explanations. arXiv preprint arXiv:2412.06878, 2024b.

Chen, Z., Zhao, Z., He, T., Chen, B., Zhao, X., Gong, L., and Liu, C. Safe reinforcement learning via hierarchical adaptive chance-constraint safeguards. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024c.

Chen, Z., Zhao, Z., Zhu, Z., Zhang, R., Li, X., Raj, B., and Yao, H. Autoprm: Automating procedural supervision for multi-step reasoning via controllable question decomposition. arXiv preprint arXiv:2402.11452, 2024d.

Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, pp. 02783649241273668, 2023.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Cheang, C.-L., Chen, G., Jing, Y., Kong, T., Li, H., Li, Y., Liu, Y., Wu, H., Xu, J., Yang, Y., et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024.

Finn, C., Levine, S., and Abbeel, P. Guided cost learning: Deep inverse optimal control via policy optimization. In International conference on machine learning, pp. 49–58. PMLR, 2016.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https:// openreview.net/forum?id=nZeVKeeFYf9.

Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., and Fei-Fei, L. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.

Huang, W., Wang, C., Li, Y., Zhang, R., and Fei-Fei, L. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024.

Janner, M., Du, Y., Tenenbaum, J., and Levine, S. Planning with diffusion for flexible behavior synthesis. In International Conference on Machine Learning, pp. 9902–9915. PMLR, 2022.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Kumar, A., Hong, J., Singh, A., and Levine, S. Should i run offline reinforcement learning or behavioral cloning? In International Conference on Learning Representations, 2021.

Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J. D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

- Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., Levine, S., Wu, J., Finn, C., Su, H., Vuong, Q., and Xiao, T. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024a.
- Li, Y., Deng, Y., Zhang, J., Jang, J., Memmel, M., Garrett, C. R., Ramos, F., Fox, D., Li, A., Gupta, A., and Goyal, A. HAMSTER: Hierarchical action models for open-world robot manipulation. In 1st Workshop on X-Embodiment Robot Learning, 2024b. URL https: //openreview.net/forum?id=yF3UekSJus.

Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P., and Zeng, A. Code as policies: Language model programs for embodied control. In 2023 IEEE

International Conference on Robotics and Automation (ICRA), pp. 9493–9500. IEEE, 2023a.

Liang, Z., Mu, Y., Ding, M., Ni, F., Tomizuka, M., and Luo, P. Adaptdiffuser: Diffusion models as adaptive self-evolving planners. In International Conference on Machine Learning, pp. 20725–20745. PMLR, 2023b.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

Mu, Y., Chen, J., Zhang, Q.-L., Chen, S., Yu, Q., Chongjian, G., Chen, R., Liang, Z., Hu, M., Tao, C., et al. Robocodex: Multimodal code generation for robotic behavior synthesis. In Forty-first International Conference on Machine Learning, 2024a.

Mu, Y., Zhang, Q., Hu, M., Wang, W., Ding, M., Jin, J., Wang, B., Dai, J., Qiao, Y., and Luo, P. Embodiedgpt: Vision-language pre-training via embodied chain of thought. Advances in Neural Information Processing Systems, 36, 2024b.

O’Neill, A., Rehman, A., Gupta, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar, A., et al. Open x-embodiment: Robotic learning datasets and rt-x models. arXiv preprint arXiv:2310.08864, 2023.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., ElNouby, A., et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. Dinov2: Learning robust visual features without supervision, 2024. URL https://arxiv.org/abs/2304.07193.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., R¨adle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Doll´ar, P., and Feichtenhofer, C. Sam 2: Segment anything in images and videos. arXiv

preprint arXiv:2408.00714, 2024. URL https:// arxiv.org/abs/2408.00714.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Sutton, R. S. Reinforcement learning: An introduction. A Bradford Book, 2018.

Team, O. M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

Walke, H. R., Black, K., Zhao, T. Z., Vuong, Q., Zheng, C., Hansen-Estruch, P., He, A. W., Myers, V., Kim, M. J., Du, M., et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pp. 1723–1736. PMLR, 2023.

Wang, C., Zhao, Z., Zhu, C., Sankararaman, K. A., Valko, M., Cao, X., Chen, Z., Khabsa, M., Chen, Y., Ma, H., et al. Preference optimization with multi-sample comparisons. arXiv preprint arXiv:2410.12138, 2024a.

Wang, C., Zhao, Z., Jiang, Y., Chen, Z., Zhu, C., Chen, Y., Liu, J., Zhang, L., Fan, X., Ma, H., et al. Beyond reward hacking: Causal rewards for large language model alignment. arXiv preprint arXiv:2501.09620, 2025.

Zhai, Y., Bai, H., Lin, Z., Pan, J., Tong, S., Zhou, Y., Suhr, A., Xie, S., LeCun, Y., Ma, Y., et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. arXiv preprint arXiv:2405.10292, 2024.

Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., and Levy, O. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024a.

Zhou, Y., Fan, Z., Cheng, D., Yang, S., Chen, Z., Cui, C., Wang, X., Li, Y., Zhang, L., and Yao, H. Calibrated self-rewarding vision language models. arXiv preprint arXiv:2405.14622, 2024b.

Zhu, H., Gupta, A., Rajeswaran, A., Levine, S., and Kumar, V. Dexterous manipulation with deep reinforcement learning: Efficient, general, and low-cost. In 2019 International Conference on Robotics and Automation (ICRA), pp. 3651–3657. IEEE, 2019.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

Wang, S., Chen, Z., Zhao, Z., Mao, C., Zhou, Y., He, J., and Hu, A. S. EscIRL: Evolving self-contrastive IRL for trajectory prediction in autonomous driving. In 8th Annual Conference on Robot Learning, 2024b. URL https:

//openreview.net/forum?id=1IzW0aniyg.

Wu, T., Gan, Y., Wu, M., Cheng, J., Yang, Y., Zhu, Y., and Dong, H. Unidexfpm: Universal dexterous functional pregrasp manipulation via diffusion policy. arXiv preprint arXiv:2403.12421, 2024.

Xian, Z., Gkanatsios, N., Gervet, T., Ke, T.-W., and Fragkiadaki, K. Chaineddiffuser: Unifying trajectory diffusion and keypose prediction for robotic manipulation. In 7th Annual Conference on Robot Learning, 2023.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., Mandlekar, A., Tan, R., Chao, Y.-W., Lin, B. Y., et al. Latent action pretraining from videos. arXiv preprint arXiv:2410.11758, 2024.

###### A. Additional Description of GRAPE and Hyperparameter Settings

Customized Cost Generation. In our real-world experiments, we first input image-text pairs containing prompts and initial states into the Vision-Language Model (VLM) Hamster (Li et al., 2024b). Using the stage information and stage points generated by Hamster, we segmented the collected trajectories. This helps analyze complex task sequences more precisely, giving detailed attention to each stage. And we utilized Grounded-SAM (Ravi et al., 2024) or methods combining SAM (Ravi et al., 2024) and DinoV2 (Oquab et al., 2024) to extract key point information from the images. These key points, combined with our self-collected trajectory data, enable us to refine the execution steps and path planning of tasks based on the stage information generated by the Hamster model. For example, for a simple pick-and-place task, we can decompose it into multiple explicit stages: Grasp the grape, Move the grape onto the plate, Place the grape on the plate.

To generate detailed operational information and cost functions for each stage, we utilized GPT-4o (Achiam et al., 2023) with customized prompts. This approach makes stage planning more precise and efficient, allowing us to meet specific task requirements and constraints. Furthermore, we enhanced our method by incorporating various task-specific constraints, including: Collision constraints: Ensuring the robot avoids collisions with obstacles. Path constraints: Optimizing the efficiency and safety of the robot’s movement path. By adopting this strategy, we achieve greater flexibility and specificity in task planning, and better adapting to different task scenarios.

Iterative Preference Optimization. For Iterative Preference Optimization, we first utilize the fine-tuned VLA model for online data sampling. For each task, we sample Nt trajectories to facilitate further selection. To simplify the experimental setup, we set Nt = 5 for each task, which has been found to perform effectively in practice.

After sampling, each trajectory is automatically labeled using the GCPG reward, as defined in Eq. (9). Based on the distribution of Rself, Rext, and Isuccess observed in preliminary experiments, we set λ1 = 0.01, λ2 = 0.01, and λ3 = 2. These values ensure that Rself, Rext, and Isuccess contribute comparably to the final reward value. Subsequent experiments validate the reasonableness of these parameter choices. Using the GCPG reward assigned to each trajectory, we identify the trajectory with the highest reward as yw and the trajectory with the lowest reward as yl for each task. This selection process enables the construction of the TPO Dataset, Dtraj, for TPO training.

For the TPO training process, we employ LoRA (Hu et al., 2022) and the AdamW optimizer, setting the learning rate to 2 × 10−5 and the batch size to 16. The model is trained for a single epoch before being utilized for iterative online sampling. During iterative online sampling, the experimental settings remain consistent with the aforementioned descriptions.

###### B. Detail Experiment Datasets

In this section, we describe the datasets collected for supervised fine-tuning (referred to as the SFT dataset) and preference alignment (referred to as the TPO dataset).

###### B.1. Real-World Dataset

SFT Dataset. In our real-world robot experiments, we use a robotic platform composed of a Franka robotic arm and a Robotiq gripper for data collection. To ensure consistency in data collection and evaluation, all operations are performed in the same experimental environment.

During data collection, we gathered a dataset of 220 instances of pick and place tasks involving common objects such as bananas, corn, milk, and salt. Additionally, we collected data on 50 instances of tasks involving pressing buttons of different colors. Since the number of objects used for the button-pressing tasks is limited, we introduced background noise and interfering objects during the testing phase to create unseen scenarios.

To further enhance the capabilities of OpenVLA in handling different actions, we also collected data on 50 instances of knock down tasks. These diverse task datasets help improve the model’s generalization ability in processing different types of actions.

TPO Dataset. In the real-world experiments, we utilized a model fine-tuned on the real-world SFT dataset via OpenVLA for trajectory sampling. Each task was conducted five times. In the TPO dataset, we experimented with 15 different tasks, including 10 pick and place tasks, 3 push button tasks, and 2 knock down tasks, accumulating a total of 75 data entries. After a selection process, we derived a preference dataset consisting of 30 trajectories.

###### B.2. Simulation Datasets

SFT Dataset: For Simpler-Env, the SFT dataset comprises 100 trajectories, amounting to approximately 2,900 transitions. These rollouts are generated from Simpler-Env using Octo, following the methodology described in Ye et al. (2024). For LIBERO, it is worth noting that we neither collect new data nor fine-tune the OpenVLA model. Instead, we directly utilize the OpenVLA-SFT model provided by the OpenVLA team, which significantly streamlines the pipeline.

TPO Dataset. In the case of Simpler-Env, trajectories are sampled for each task using the OpenVLA-SFT model, with five trials conducted per task. This process yields a TPO dataset consisting of 80 trajectories. For LIBERO, OpenVLA-SFT models (one model per task) are employed to sample data across four tasks in LIBERO. For each task, five trajectories are sampled for each sub-task, resulting in a TPO dataset comprising a total of 20 trajectories.

###### C. Detailed Experiment Settings and Additional Result

###### C.1. Real-World

- C.1.1. REAL-WORLD EXPERIMENT SETUP

In real-world experiment, we used the Franka robot arm, which is known for its precision and flexibility. However, we encountered a problem with the original Franka gripper, which was not long enough, limiting our ability to handle some of the tasks, resulting in inefficient completion and a high failure rate. To solve this problem, we decided to replace the original Franka grippers with Robotiq grippers, which are not only longer, but also provide more grip and flexibility, which greatly improves the efficiency and success rate of the tasks.

The purpose of this experiment was to assess the cross-task generalization capabilities of OpenVLA under the GRAPE framework and to compare its performance with several baseline models. Considering the generally poor zero-shot generalization performance of most VLA models, we performed supervised fine-tuning using the comprehensive rollout dataset Dr collected from real scenes to construct a fine-tuned model. The selection of baseline models included those adjusted with domain-specific data, as well as the Octo model, RVT-2 model, and OpenVLA-SFT model.

- C.1.2. REAL-WORLD TASKS

As shown in Figure 5, we performed a comprehensive evaluation on a real machine for several tasks. These tasks cover five different generalization scenarios: Visual Generalization, Subject Generalization, Action Generalization, Semantics Generalization, and Language Grounding. Specifically, for each generalization scenario, we set the following tasks:

- • Visual Generalization includes 8 tasks, e.g., pick up the GRAPE and put it in the black bowl, with noise objects and noisy backgrounds. Some tasks have only noisy backgrounds.
- • Subject Generalization includes 4 tasks, e.g., pick up the K and put it in the black bowl.
- • Action Generalization includes 7 tasks, e.g., fold the green towel from right to left .
- • Semantics Generalization includes 4 tasks, e.g., stack carrot and put it on the blue plates.
- • Language Grounding includes 3 tasks, e.g., pick up left object to left plate.

We conducted experiments on 30 total different tasks, attempting each task ten times, totaling 300 executions. To ensure fairness in the evaluation, we maintained the same starting position in each model test. Additionally, we matched the image resolution when training all models and used exactly the same initial object positions in all evaluations. We set specific success criteria for each task. For example, in the pick-and-place task, a successful grasp is defined as successfully grasping the target object. In the push-button and knock-down tasks, a successful grasp is defined as correctly approaching and manipulating the target object. Overall task success is defined as the object being accurately placed at the target location, successfully knocked down, or the target button being successfully pressed. Due to the strictness of these criteria, some models found it difficult to achieve success in specific tasks.

- Table 3: Comparison of GRAPE models in diffierent iteration rounds. We assess their performance in in-domain tasks and three kinds of generalization evaluations. Each task’s performance is evaluated on the overall grasp rate and success rate.

In-domain Subject Gen. Physical Gen. Semantics Gen. Average

Grasp Success Grasp Success Grasp Success Grasp Success Grasp Success

- Iter-1 71.00% 43.00% 62.67% 40.67% 63.50% 41.75% 72.00% 47.00% 67.29% 43.11%

- Iter-2 74.00% 45.00% 64.33% 40.33% 65.75% 44.25% 76.00% 49.50% 70.02% 44.77%

- Iter-3 74.50% 45.50% 64.67% 40.67% 66.00% 44.50% 76.00% 49.00% 70.29% 44.92%

###### C.2. Simulation Experiments

- C.2.1. SIMPLER-ENV

We utilize Simpler-Env (Li et al., 2024a) as the experimental environment in our study. SIMPLER (Li et al., 2024a) (Simulated Manipulation Policy Evaluation for Real Robot Setups) is a collection of simulated environments created to assess robot manipulation policies in a way that closely reflects real-world scenarios. By leveraging simulated environments, SIMPLER effectively serves as a practical alternative to real-world testing, which is often costly, time-consuming, and challenging to replicate.

Simpler-Env Tasks. In our paper, we use four in-domain tasks from WidowX robot in Simpler-Env. We also design three kinds of generalization tasks in Simpler-Env. These tasks are described below:

In-Domain Tasks Shown in Fig. 3:

- 1. Put Carrot on Plate: The robot is positioned in front of a platform with a plate and a carrot. The robot’s goal is to grasp the carrot and put it onto the plate.
- 2. Put Eggplant in basket: The robot is positioned in front of a sink with a basket and a Eggplant. The robot’s goal is to grasp the Eggplant and put it in the basket.
- 3. Stack Green Cube on Yellow Cube: The robot is positioned in front of a platform with a green cube and a yellow cube. The robot’s goal is to grasp the green cube and stack it on the yellow cube.
- 4. Put Spoon on towel: The robot is positioned in front of a platform with a spoon and a towel. The robot’s goal is to grasp the spoon and put it on the towel.

Three Kinds of Generalization Tasks Shown in Fig. 3:

- 1. Subject Generalization: The robot is positioned in front of a platform, similar to the environment in in-domain tasks. But the robot’s goal is to grasp some new objects(i.e. pepsi can, coke can, sprite can) and put it onto the plate.
- 2. Physical Generalization: The robot is positioned in front of a platform, similar to the environment in in-domain tasks. But the robot’s goal is to grasp some original objects with different sizes and collision boxes, then put it onto the plate.
- 3. Semantics Generalization: The robot is positioned in front of a platform, similar to the environment in in-domain tasks. And the instruction is similar to in-domain tasks, too. But the instruction has been modified by GPT-4o (Achiam et al.,

2023) while maintaining its original meaning.

- C.2.2. LIBERO

We further utilize LIBERO (Liu et al., 2023) as the experimental environment in our study. LIBERO (LIfelong learning BEnchmark on RObot manipulation tasks) includes a set of 130 language-conditioned robot manipulation tasks inspired by human activities, organized into four distinct suites. Each suite is crafted to examine distribution shifts in object types, spatial arrangements of objects, task goals, or a combination of these factors. LIBERO is built to be scalable, extendable, and specifically tailored for advancing research in lifelong learning for robotic manipulation.

LIBERO tasks In our paper, we use four in-domain tasks from LIBERO, which are shown in Fig. 4. These tasks is described below:

- • LIBERO-Spatial includes the same set of objects arranged in various layouts, testing the model’s ability to understand spatial relationships.
- • LIBERO-Object features consistent scene layouts with varying objects, evaluating the model’s ability to understand different object types.
- • LIBERO-Goal includes of the same objects and layouts but different task goals, testing the model’s knowledge of different task-oriented behaviors.
- • LIBERO-10 consists of long-horizon tasks with diverse objects, layouts, and tasks.

Eash task mentioned above has 10 sub-tasks, with similar task instructions and scenes. Here are some cases from various LIBERO tasks:

- • Open the top drawer of the cabinet and put the bowl in it.
- • Pick up the book and place it to the right of the caddy.
- • Turn on the stove and put the frying pan on it.
- • Stack the right bowl on the left bowl and place them in the tray.

###### D. Additional Real-World and Simulation Results

We provide additional results in Table 4 , Table 5, and Figure 12 with detailed task description. Each table has in-domain tasks and several kinds of generalization evaluations. These experiments are conducted across Octo-SFT, OpenVLA-SFT and GRAPE.

###### E. Case Study

###### E.1. Case Study of Real-World Generation Tasks

We provide an illustration for each specific task included in the suite evaluation for in-domain tasks in Figure 8 and for each type of generation task, including subject generalization in Figure 9, language grounding in Figure 13, visual generalization in Figure 10, action generalization in Figure 11, and semantic generalization in Figure 12. Specifically, we demonstrate the initial and final states of GRAPE in handling each of these challenging tasks, as detailed in the corresponding captions. In addition, we include a safety task to demonstrate how GRAPE adheres to safety requirements once aligned with safety constrains.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

## Indomain

pick up the banana and put it in the black bowl pick up the corn and put it in the black bowl

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

pick up the milk and put it in the white bowl pick up the salt bottle and put it in the white bowl

- Figure 8: Illustrations of real-world tasks that we evaluated for in-domain capabilities, where we report the detailed results in Table 4. Specifically, we demonstrate the initial and final state of GRAPE in handling each of the four challenging tasks detailed in the captions.

SubjectGeneralization

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

pick up the K and put it in the black bowl pick up the chips and put it in the red bowl

pick up the box juice and put it in the yellow plate pick up the Fanta can and put it in the white bowl

[Figure 64]

[Figure 65]

- Figure 9: Illustrations of real-world tasks that we evaluated for subject generation, where we report the detailed results in Table 4. Specifically, we demonstrate the initial and final state of GRAPE in handling each of the four challenging tasks detailed in the captions.

###### (w/o noise background and object)

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

pick up the salt bottle and put it in the white bowl

pick up the grape and put it in the black bowl

[Figure 70]

[Figure 71]

pick up the milk and put it in the white bowl

#### VisualGeneralization

(w/o noise background)

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

pick up the corn and put it in the black bowl pick up the grape and put it in the black bowl

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

pick up the banana and put it in the black bowl pick up the milk and put it in the white bowl

[Figure 80]

[Figure 81]

pick up the salt bottle and put it in the white bowl

- Figure 10: Illustrations of real-world tasks that we evaluated for visual generation, where we report the detailed results in Table 4. Specifically, we demonstrate the initial and final state of GRAPE in handling each of the eight challenging tasks detailed in the captions.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

push down the blue button push down the yellow button

[Figure 86]

[Figure 87]

ActionGeneralization

push down the green button

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

knock down the green button knock down the yellow box

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

fold the white towel from left to right fold the green towel from right to left

- Figure 11: Illustrations of real-world tasks that we evaluated for action generation, where we report the detailed results in Table 4. Specifically, we demonstrate the initial and final state of GRAPE in handling each of the seven challenging tasks detailed in the captions.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

### SemanticGeneration

Lift grape and place it in the black bowl take green pepper and place it in the black bowl

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

stack carrot and put it on the blue plates move icecream and put it in the red bowl

- Figure 12: Illustrations of real-world tasks that we evaluated for semantic generation, where we report the detailed results in Table 4. Specifically, we demonstrate the initial and final state of GRAPE in handling each of the four challenging tasks detailed in the captions.

LanguageGrounding

[Figure 104]

[Figure 105]

pick up right object to right plate

[Figure 106]

[Figure 107]

pick up left object to left plate

[Figure 108]

[Figure 109]

push down middle button

- Figure 13: Illustrations of real-world tasks that we evaluated for language generation, where we report the detailed results in Table 4. Specifically, we demonstrate the initial and final state of GRAPE in handling each of the five challenging tasks detailed in the captions.

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

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

###### OpenVLA-SFT

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

GRAPE-Safety

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

###### OpenVLA-SFT

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

GRAPE-Safety

- Figure 14: Illustrations of real-world tasks used for safety evaluation, extending the tasks presented in Figure 7. The figure shows key frames from GRAPE’s trajectory in two challenging scenarios. Due to the lack of safety reward alignment, the OpenVLA-SFT approach fails, while GRAPE-Safety successfully navigates obstacles and completes the task once the safety rewards are properly aligned.

- Table 4: We present the performance of various action policy on real-world robotic manipulation tasks categorized by different types of generalization. The tasks include in-domain, visual generalization with and without noise, subject generalization, action generalization, semantics generalization, and language grounding. Each task’s performance is evaluated based on the number of successful grasps and the overall success rate, comparing results from Octo-SFT, OpenVLA-SFT, OpenVLA-DPO, and GRAPE. Average success rates are calculated for each generalization category to demonstrate the effectiveness of the tested models under different conditions.

Generalization Task Octo-SFT OpenVLA-SFT OpenVLA-DPO GRAPE

Grasp Success Grasp Success Grasp Success Grasp Success

pick up the corn and put it in the black bowl 3 3 2 2 5 3 8 7 pick up the banana and put it in the black bowl 2 0 6 6 8 6 9 7 pick up the milk and put it in the white bowl 4 2 10 8 8 8 9 9 pick up the salt bottle and put it in the white bowl 4 3 4 2 5 3 6 4 Average 32.5% 20% 55% 45% 65% 50% 80% 67.5%

In-domain

pick up the corn and put it in the black bowl 2 1 6 3 6 4 6 6 pick up the banana and put it in the black bowl 0 0 3 2 4 1 4 1 pick up the milk and put it in the white bowl 4 0 4 4 6 6 9 7 pick up the salt bottle and put it in the white bowl 2 2 6 5 6 6 8 8 pick up the GRAPE and put it in the black bowl 0 0 6 5 8 5 8 6 Average 16% 6% 50% 38% 60% 44% 70% 56%

Visual Generalization (w/o noise background)

pick up the GRAPE and put it in the black bowl 1 0 4 2 5 3 6 4 pick up the milk and put it in the white bowl 2 1 7 5 6 4 5 4 pick up the salt bottle and put it in the white bowl 0 0 2 2 6 5 8 8 Average 10% 3.3% 43.3% 30% 56.7% 40% 63.3% 53.3%

Visual Generalization (w/o noise background and object)

pick up the chips and put it in the red bowl 4 0 2 2 4 3 6 5 pick up the K and put it in the black bowl 2 0 4 4 6 5 7 6 pick up the box juice and put it in the yellow plate 2 0 8 3 8 5 8 6 pick up the Fanta can and put it in the white bowl 2 2 4 1 5 2 6 4 Average 25% 5% 45% 25% 57.5% 37.5% 67.5% 52.5%

Subject Generalization)

push down the blue button 1 0 4 4 6 4 6 6 push down the green button 1 0 6 4 7 5 4 4 push yellow the button 2 2 6 3 7 4 8 5 knock down the green bottle 3 1 2 2 3 2 4 2 knock down the popcorn 0 0 4 2 4 3 4 3 fold the green towel from right to left 1 0 2 1 3 1 4 2 fold the white towel from left to right 1 0 3 1 4 2 4 3 Average 12.9% 4.3% 38.6% 24.3% 48.6% 30% 48.6% 35.7%

Action Generalization

take green pepper and place it in the black bowl 0 0 10 6 9 7 10 8 move icecream and put it in the red bowl 0 0 6 4 5 4 4 4 stack carrot and put it on the blue plates 0 0 8 8 6 5 6 6 Lift GRAPE and place it in the black bowl 0 0 2 0 3 2 2 2 Average 0% 0% 65% 45% 57.5% 45% 55% 50%

Semantics Generalization

pick up left object to left plate 0 0 4 0 5 1 5 2 push down right button 0 0 6 2 6 5 8 7 pick up right object to right plate 0 0 4 4 5 4 6 5 Average 0% 0% 46.7% 20% 53.3% 33.3% 63.3% 46.7%

Language Grounding

Total Average 14.3% 5.7% 48.3% 32.3% 56.3% 39.3% 62.6% 50.3%

- Table 5: We compared the performance of Octo-SFT, OpenVLA-SFT, and GRAPE across various robotic tasks within in-domain, subject, physical, and semantics generalization categories. It shows grasp percentages and success rates for each task, illustrating how each VLA performs under different generalizations.

Generalization Task Octo-SFT OpenVLA-SFT OpenVLA-DPO GRAPE

Grasp Success Grasp Success Grasp Success Grasp Success

put the carrot on the plate 32.00% 16.00% 36.00% 30.00% 46.00% 36.00% 68.00% 48.00% put the eggplant in the basket 70.00% 44.00% 58.00% 32.00% 70.00% 36.00% 84.00% 48.00% stack the green cube on the yellow cube 52.00% 0.00% 56.00% 20.00% 52.00% 26.00% 76.00% 40.00% put the spoon on the towel 54.00% 36.00% 52.00% 28.00% 52.00% 30.00% 56.00% 34.00% Average 52.00% 24.00% 50.50% 28.00% 55.00% 32.00% 71.00% 43.00%

In-domain

put the coke can on the towel 24.00% 14.00% 60.00% 38.00% 66.00% 36.00% 78.00% 32.00% put the pepsi can on the towel 28.00% 16.00% 58.00% 38.00% 60.00% 42.00% 64.00% 50.00% put the sprite can on the towel 24.00% 12.00% 62.00% 22.00% 58.00% 26.00% 46.00% 40.00% Average 25.33% 14.00% 60.00% 32.67% 61.33% 34.66% 62.67% 40.67%

Subject Generalization (unseen objects)

- put the carrot on the plate(size:0.5) 38.00% 22.00% 56.00% 38.00% 60.00% 46.00% 78.00% 64.00%

- put the carrot on the plate(size:1.1) 26.00% 12.00% 32.00% 24.00% 42.00% 30.00% 64.00% 42.00%

put the carrot on the plate(wider collision box) 28.00% 16.00% 34.00% 26.00% 46.00% 32.00% 62.00% 42.00% put the carrot on the plate(longer collision box) 32.00% 14.00% 38.00% 30.00% 50.00% 36.00% 66.00% 48.00%

Physical Generalization (unseen object sizes/shapes)

- put the spoon on the towel(size:0.5) 62.00% 38.00% 66.00% 40.00% 66.00% 38.00% 72.00% 38.00%

- put the spoon on the towel(size:1.1) 52.00% 32.00% 50.00% 28.00% 58.00% 32.00% 56.00% 30.00%

put the spoon on the towel(wider collision box) 48.00% 30.00% 44.00% 24.00% 46.00% 28.00% 50.00% 32.00% put the spoon on the towel(longer collision box) 56.00% 36.00% 54.00% 26.00% 54.00% 28.00% 60.00% 38.00% Average 42.75% 25.00% 46.75% 29.50% 52.75% 33.75% 63.50% 41.75%

put the vegetable on the plate 16.00% 6.00% 32.00% 28.00% 40.00% 32.00% 66.00% 48.00% move the eggplant into the basket 18.00% 8.00% 50.00% 30.00% 56.00% 34.00% 78.00% 44.00% put the green cube onto the yellow cube 32.00% 6.00% 62.00% 26.00% 74.00% 42.00% 88.00% 60.00% place the spoon onto the towel 42.00% 26.00% 48.00% 28.00% 48.00% 30.00% 56.00% 36.00% Average 27.00% 11.50% 48.00% 28.00% 54.50% 34.50% 72.00% 47.00%

Semantics Generalization (unseen instructions)

Total average 36.77% 18.63% 51.44% 29.54% 55.90% 33.73% 67.29% 43.11%

###### E.2. Case Study of Multi-stage Cost Functions

We demonstrate some case studies of the multi-stage cost functions generated using our proposed pipeline given different alignment objectives.

###### E.2.1. TASK COMPLETION Cost Functions for Task Completion Alignment

# The task involves picking up the grape and placing it in the black bowl. # The stages involved are:

- # 1. Grasp grape
- # 2. Move grape to black bowl
- # 3. Drop grape in black bowl num_stages = 3

- ### stage 1: Grasp grape

- def stage1_target_constraint1(end_effector, keypoints): """Align the end-effector with the grape’s center."""

grape_center = keypoints[0] target_cost = np.linalg.norm(end_effector - grape_center) return target_cost

### stage 2: Move grape to black bowl

- def stage2_target_constraint1(end_effector, keypoints): """Calculate the relative distance between grape and black bowl."""

black_bowl_center = keypoints[1]# Assuming keypoint 1 is the black bowl target_cost = np.linalg.norm(end_effector - black_bowl_center) return target_cost

### stage 3: Drop grape in black bowl

- def stage3_target_constraint1(end_effector, keypoints): """Ensure the grape rests in the black bowl."""

black_bowl_center = keypoints[1] target_cost = np.linalg.norm(end_effector - black_bowl_center) return target_cost

###### E.2.2. SAFETY Cost Functions for Cost-Efficiency Alignment

# The task involves picking up the grape and placing it in the black bowl. # The stages involved are:

- # 1. Grasp grape
- # 2. Move grape to black bowl
- # 3. Drop grape in black bowl

num_stages = 3

- ### stage 1: Grasp grape

- def stage1_collision_constraint1(end_effector, keypoints): """Approach the grape from above to avoid collision."""

grape_center = keypoints[0] collision_cost = 0 if end_effector[1] > grape_center[1] else 1 return collision_cost

### stage 2: Move grape to black bowl

- def stage2_collision_constraint1(end_effector, keypoints): """Ensure the grape is aligned above the black bowl."""

obstacles = keypoints[2:]#Assuming keypoints[2:] are obstacles threshold = 0.1 # Minimum allowable clearance collision_cost = sum(

max(0, threshold - np.linalg.norm(end_effector - obstacle)) for obstacle in obstacles

) return collision_cost

### stage 3: Drop grape in black bowl

- def stage3_collision_constraint1(end_effector, keypoints): """Approach the grape from above to avoid collision."""

black_bowl_center = keypoints[1] collision_cost = 0 if end_effector[1] > black_bowl_center[1] else 1 return collision_cost

###### E.2.3. COST-EFFICIENCY Cost Functions for Safety Alignment

# The task involves picking up the grape and placing it in the black bowl. # The stages involved are:

- # 1. Grasp grape
- # 2. Move grape to black bowl
- # 3. Drop grape in black bowl num_stages = 3

- ### stage 1: Grasp grape

- def stage1_path_constraint1(end_effector, keypoints): """Align the end-effector with the grape’s center.""" grape_center = keypoints[0]

distance = np.linalg.norm(end_effector - grape_center) step_size = 0.01 # Assuming a small step size path_cost = int(distance / step_size) return path_cost

- ### stage 2: Move grape to black bowl

- def stage2_path_constraint1(end_effector, keypoints): """Calculate the relative distance between grape and black bowl."""

black_bowl_center = keypoints[1]# Assuming keypoint 1 is the black bowl distance = np.linalg.norm(end_effector - black_bowl_center) step_size = 0.01 # Assuming a small step size path_cost = int(distance / step_size) return path_cost

### stage 3: Drop grape in black bowl

- def stage3_path_constraint1(end_effector, keypoints): """Ensure the grape rests in the black bowl."""

black_bowl_center = keypoints[1] distance = np.linalg.norm(end_effector - black_bowl_center) step_size = 0.01 # Assuming a small step size path_cost = int(distance / step_size) return path_cost

###### Prompt Template for Multi-stage Cost Proposal USER: Instructions

The image shows a robot stage point in a workspace, each point in the diagram represents the point of the stage split:

- • Stage point 0 : Represents the initial position of the carrot.

- • Stage point 1 : Represents the intermediate position above the carrot for grasping.

Determine how many stages are involved in the task. Grasping must be an independent stage. Some examples:

1. TASK: PUT THE CARROT ON THE PLATE Stages:

- • Grasp carrot
- • Move carrot to plate
- • Drop carrot on plate

- Stage 1: Grasp carrot

- • Path constraints:

– Align the end-effector with the carrot’s center.

- • Collision constraints:

– The end-effector must approach the carrot from above to avoid collision.

- Stage 2: Move carrot to plate

- • Path constraints:

– Calculate the relative distance between carrot and plate.

- • Collision constraints:

– The carrot is aligned above the plate.

- Stage 3: Drop carrot on plate

- • Path constraints:

- – The carrot must rest on the plate.
- – The carrot should not bounce out of the basket.

- • Collision constraints:

– The end-effector must approach the carrot from above to avoid collision.

###### Note:

- • Sum all Path constraints cost the path_cost variable.
- • Sum all Grasp constraints cost the grasp_cost variable.
- • Sum all Collision constraints cost the collision_cost variable.
- • Each constraint function takes an end-effector point and a set of keypoints as input, returning a numerical cost. The constraint is satisfied if this cost is zero or less.
- • Define any number of path constraints per stage, but avoid using ”if” statements in the functions.
- • Avoid using path constraints when manipulating deformable objects (e.g., towels).
- • Input format:

- – end_effector: np.array of shape (3,) representing the end-effector position.
- – keypoints: np.array of shape (K, 3) representing the keypoints positions.

- • Use Python and NumPy functions freely in constraint functions.
- • Use pairs of keypoints to create vectors if needed.
- • Keypoints are indexed starting from 0, matching their order in the keypoints array.

Structure your output in a single Python code block as follows: # ... num_stages = ? ### stage 1 path constraints (if any) def stage1_path_constraint1(end_effector, keypoints):

"""Put your explanation here."""

... return path_cost

# Add more constraints if needed

... ### stage 1 collision constraints (if any) def stage1_collision_constraint1(end_effector, keypoints):

"""Put your explanation here."""

... return collision_cost

# Add more constraints if needed

... # Repeat for more stages

...

Query Query Task: ”{instruction}” Query Image:

