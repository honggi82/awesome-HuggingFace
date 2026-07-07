arXiv:2505.23762v1[cs.AI]29May2025

# ZeroGUI: Automating Online GUI Learning at Zero Human Cost

Chenyu Yang2,1∗, Shiqian Su2,1*, Shi Liu1*, Xuan Dong2,1*, Yue Yu2,1*, Weijie Su1*,† , Xuehui Wang3,1, Zhaoyang Liu4, Jinguo Zhu1, Hao Li1, Wenhai Wang5,1, Yu Qiao1, Xizhou Zhu2,1 , Jifeng Dai2,1

1Shanghai Artificial Intelligence Laboratory 2Tsinghua University 3Shanghai Jiao Tong University 4Hong Kong University of Science and Technology 5The Chinese University of Hong Kong

{yangcy23,ssq24,x-dong21,yuyue21}@mails.tsinghua.edu.cn, {liushi,suweijie,zhujinguo,qiaoyu}@pjlab.org.cn,

{zhuxizhou,daijifeng}@tsinghua.edu.cn, wangxuehui@sjtu.edu.cn, {zyliumy,lihaothu}@gmail.com,whwang@ie.cuhk.edu.hk,

## Abstract

The rapid advancement of large Vision-Language Models (VLMs) has propelled the development of pure-vision-based GUI Agents, capable of perceiving and operating Graphical User Interfaces (GUI) to autonomously fulfill user instructions. However, existing approaches usually adopt an offline learning framework, which faces two core limitations: (1) heavy reliance on high-quality manual annotations for element grounding and action supervision, and (2) limited adaptability to dynamic and interactive environments. To address these limitations, we propose ZeroGUI, a scalable, online learning framework for automating GUI Agent training at Zero human cost. Specifically, ZeroGUI integrates (i) VLM-based automatic task generation to produce diverse training goals from the current environment state, (ii) VLM-based automatic reward estimation to assess task success without hand-crafted evaluation functions, and (iii) two-stage online reinforcement learning to continuously interact with and learn from GUI environments. Experiments on two advanced GUI Agents (UI-TARS and Aguvis) demonstrate that ZeroGUI significantly boosts performance across OSWorld and AndroidLab environments. The code is available at https://github.com/OpenGVLab/ZeroGUI.

## 1 Introduction

GUI agents are designed to perceive and interact with Graphical User Interfaces (GUIs). Early methods achieved this by building pipelines or relying on structured inputs such as HTML or DOM trees. Recently, the emergence of large Vision-Language Models (VLMs) [64, 63, 4] has enabled the development of end-to-end, pure-vison-based agents[20, 11, 71, 48], capable of perceiving GUI screenshots and performing actions such as clicking, scrolling, or typing to complete user-provided task instructions. These agents have demonstrated strong potential across various applications, including digital task automation, intelligent copilots, human-computer interaction, etc.

Despite these progress, as shown in Fig. 1, existing approaches usually adopt an offline learning framework, which presents two fundamental limitations: (1) they heavily rely on high-quality human annotations for both GUI grounding [11, 66, 26] (i.e., identifying screen elements) and action

∗Equal contribution. † Project lead. This work is done when Chenyu Yang, Shiqian Su, Shi Liu, Xuan Dong, Yue Yu, Xuehui Wang and Hao Li are interns at Shanghai Artificial Intelligence Laboratory. Corresponding to Weijie Su <suweijie@pjlab.org.cn> and Xizhou Zhu <zhuxizhou@tsinghua.edu.cn>.

###### Previous GUI Agents Our ZeroGUI

[Figure 1]

[Figure 2]

[Figure 3]

###### SFT RL

[Figure 4]

[Figure 5]

[Figure 6]

Reward

Label

VLM

(Eval Code)

Trajectory

GUI Agent

Reward Estimation

GUI Agent

Task

High-Quality Trajectories

###### Act

Obs

[Figure 7]

Initial State

(Designed Task)

[Figure 8]

[Figure 9]

[Figure 10]

Obs

VLM

Task Generation

Act

Environments

Environments

[Figure 11]

[Figure 12]

Offline Data High Labeling Cost Not Scalable Online Learning

Scalable

Zero Human Cost

- Figure 1: Left: Existing Offline Training Framework for GUI Agents incurs high human costs, relying on manually collected and annotated interaction trajectories, typically under a supervised fine-tuning (SFT) paradigm. Right: Our ZeroGUI is a scalable online learning framework with automated task generation and reward estimation at zero human cost. A VLM proposes diverse tasks, which are executed by the agent; the agent then receives VLM-based rewards and updates its policy via reinforcement learning (RL).

trajectories [10, 39, 71, 48] (i.e., a sequence of actions to complete a task). These human-annotated labels are expensive, time-consuming, and difficult to scale across diverse platforms and tasks. (2) they fall short in adapting to dynamic and interactive environments. GUIs in the real world are non-stationary and uncertain: elements may shift, disappear, or behave differently depending on the system’s state. Existing agents often overfit to static or narrowly defined tasks and struggle to generalize in open-ended scenarios.

To overcome these limitations, online learning, where GUI agents are continuously updated through interaction with GUI environments, is a desirable approach but remains challenging. Most existing environments, such as OSWorld [68] and AndroidLab [70], only provide a test set consisting of manually crafted tasks and verification functions. Constructing a training set with diverse tasks and associated success verifiers by hand is expensive and not scalable. Furthermore, in real-world environments, novel scenarios or tasks often lack ground-truth annotations, making it difficult to provide direct supervisory signals for agent learning.

To develop a scalable online learning framework, we focus on automating the construction of tasks and their corresponding success verifiers in GUI environments. This is feasible with the help of advanced VLMs, because they have been trained with large-scale GUI-related data and excel at understanding GUI elements, actions, and state transitions. They can assess task completion and propose relevant tasks based on observed information. In addition, when training GUI agents, it is sufficient to evaluate the encountered states rather than exhaustively covering all possible scenarios, which significantly reduces the complexity of automated task verification.

Specifically, we propose ZeroGUI, a fully automated online training framework in which GUI agents continuously interact with GUI environments to complete automatically generated tasks and update their policies using annotation-free rewards. As illustrated in Fig. 1, ZeroGUI consists of the following components: (1) VLM-Based Automatic Task Generation, which proposes a large and diverse set of training tasks based on random initial states. (2) VLM-Based Automatic Reward Estimation, which predicts binary rewards that indicate task success, serving as supervision signals. The estimator leverages the trajectories of GUI agents as input, eliminating the need for hand-crafted task verifiers. (3) Two-stage Online Reinforcement Learning, involves a training stage on generated tasks followed by a test-time adaptation stage. We adapt the RL framework to support multi-step interactions between the GUI agent and the environment.

We apply our ZeroGUI to two advanced VLM-based GUI agents (UI-TARS [48] and Aguvis [71]) and leverage both desktop (OSWorld [68]) and mobile (AndroidLab [70]) environments for evaluation. Experiments show that our ZeroGUI yields significant improvements in task success rates. Training on generated tasks extends the agent’s capability coverage, while test-time training helps the agent adapt to test tasks. In particular, on OSWorld, ZeroGUI-UI-TARS-7B achieves 14% relative improvements and ZeroGUI-Aguvis-7B achieves 63% relative improvements.

In summary, our contributions are as follows:

- • We propose ZeroGUI, a fully automated online learning framework that enables GUI agents to improve through interaction with GUI environments, eliminating the need for collecting and labeling offline training data.
- • We design automatic VLM-based task generation and reward estimation, which generate training tasks and provide supervisory rewards in GUI environments without human annotations.
- • We introduce a two-stage reinforcement learning strategy. In the first stage, training on generated tasks builds the agent’s general capabilities. In the second stage, test-time training enables the agent to adapt to target test tasks.
- • The proposed ZeroGUI significantly improves task success rates across multiple GUI environments and generalizes well to different base models.

## 2 Related Work

GUI Agents are AI systems aimed to perceive, understand, and act upon graphical user interfaces. Early systems [75, 65] heavily relied on structured representations such as HTML, DOM trees, but recent progress in Vison Language Models (VLMs) has enabled a shift toward purely vision-based approaches. However, due to the small size and visual variability of UI elements, general-purpose VLMs still struggle with accurate grounding. To mitigate this, several works [80, 17, 61, 77] incorporate specialized UI parsers to assist proprietary VLMs [1, 23, 2, 16]. Other efforts [20, 11, 66, 29, 71, 72, 26, 48] focus on building large-scale grounding datasets via manual labeling or automated pipelines for supervised fine-tuning.

Beyond grounding, the long-horizon and high-variability nature of GUI tasks makes planning another key challenge. To this end, works such as [10, 39] collect expert trajectories across platforms for imitation learning. Aguvis[71] enhances these datasets with VLM generated chain-of-thought annotations, while UI-TARS[48] introduces both positive and negative samples to facilitate selfreflection and error correction via direct preference optimization. Given the high cost of collecting high-quality demonstrations, recent methods [41, 67, 35] explore reinforcement fine-tuning with rule-based rewards and limited expert data. Nevertheless, several studies [3, 47] report that models trained solely on static trajectories often struggle to generalize in dynamic, real-world environments. To improve adaptability, works like [47, 32] train outcome reward models (ORMs) in dynamic and interactive environments [82, 33] to support more robust RL-based adaptation. However, many interactive GUI environments [68, 51] lack curated training task sets, making ORM training difficult. To address this, we propose VLM-based Automatic Task Generation and VLM-based Automatic Reward Estimation methods, enabling scalable RL training in dynamic and interactive environments and laying the foundation for a more generalizable GUI agent training framework.

Interactive Environments for GUI Agents. Environments for GUI agents can be broadly categorized into non-interactive and interactive settings. Non-interactive environments (e.g., [13, 80, 31] for the web domain; [28, 58, 21, 50, 78, 27, 39, 7] for Android; and cross-platform settings like [24, 10]) are typically static and predefined. Interactive environments (e.g., [15, 6, 79] for Linux or Windows; [57, 30, 74, 82, 25, 19, 40, 14, 45] for web; [62, 51, 70] for Android; and cross-platform frameworks such as [68, 69, 33]) facilitate autonomous perception and action, enabling agents to operate in more realistic and dynamic settings.

Interactive environments for GUI agents typically consist of an operating system platform (e.g., desktop OS or Android emulator), an action execution interface, and an observation module. Agents interact with the environment through a predefined action space, or using PyAutoGUI on desktops and Android Debug Bridge (ADB) on mobile platforms. The environment provides feedback in the form of screenshots and UI structure data, enabling agents to perceive both visual and structured information for decision-making. Evaluation metrics in this dynamic environments are primarily based on manually predefined rules or hardcoded scripts, which are complex and lack scalability.

RL for Post-Training. Reinforcement learning methods such as PPO[54] and DPO[49] have been widely used in the post-training of Large Language Models (LLMs), particularly within the RLHF framework[44]. Recently, RL methods such as GRPO[55], DAPO[76], and Dr.GRPO[36] have demonstrated strong effectiveness in improving reasoning abilities of LLMs, as exemplified by works like [18, 60]. Building on these developments, a series of studies [73, 81, 22, 12, 46, 37, 56, 8] extend

RL approaches to the post-training of VLMs, mainly focusing on mathematical reasoning and other general vision tasks where annotated labels are easily accessible.

In parallel, emerging paradigms such as test-time reinforcement learning[83] attract increasing attention, aiming to enable effective RL training on unlabeled data. Our work further advances this line of research by applying test-time RL to VLM post-training in dynamic, interactive GUI environments. Specifically, we adapt the GRPO algorithm to multi-step settings and enhance training stability through tuning the KL regularization term.

## 3 ZeroGUI

Existing offline methods rely on carefully collected trajectories and designed tasks, limiting their scalability and adaptability. To enable online adaptation with vision-language models, we introduce ZeroGUI, an automatic online training framework at zero human cost.

Formulation. The completion of a GUI task can be formulated as a Markov Decision Process (MDP) [5], denoted by (S,A,R,T ). Given a task instruction I, the GUI agent interacts with the environment. At each step t, the agent predicts an action at according to its policy:

at ∼ πθ (at|I,st), st = (ot,ht) (1)

where the agent’s state st combines the current observation ot and history information ht containing previous observations and actions. The process terminates when encountering a terminate action or reaching maximum number of steps, resulting in a trajectory:

τ = {I,(o1,a1),(o2,a2),...,(oT,aT)} (2)

Framework Overview. As illustrated in Fig. 2, ZeroGUI consists of three key components: automatic task generation, automatic reward estimation, and a two-stage online reinforcement learning process. In the first stage, a VLM automatically generates a set of training tasks, and the agent is trained using rewards estimated by a VLM-based evaluator. In the second stage, the agent performs testtime training on the test set without ground-truth labels, receiving rewards solely from the same VLM-based evaluator. The following sections describe each component of our ZeroGUI in detail.

###### 3.1 Automatic Task Generation

Current offline training methods heavily rely on high-quality human annotations for GUI grounding and trajectory data with manually designed task instructions. To enable online training without human supervision, it is crucial to develop a scalable and automated task generation pipeline.

A key challenge in task generation is to ensure generalization, particularly given the limited number of evaluation samples in existing GUI agent benchmarks. Our generated tasks must not only align with the operational constraints of the target environments but also exhibit sufficient diversity to cover a broad behavioral space. To address this, we propose the following prompting designs:

- (1) Example-Guided Prompting. Prompt with a combination of instruction exemplars and randomly sampled initial state screenshots, which guide the model toward environment-specific and realistic task proposals.
- (2) Multi-Candidate Generation. In each generation step, we request multiple task candidates simultaneously, encouraging the model to produce a diverse set rather than overfitting to a narrow task style.

- Fig. 2 illustrates this generation process and showcases representative generated tasks with test tasks from OSWorld as reference.

To further train the agents to recognize unachievable goals and provide appropriate feedback, we prompt the VLM to generate a subset of infeasible tasks. Such tasks are intentionally unsolvable within the environment and require the agent to explicitly output “FAIL” response.

###### 3.2 Automatic Reward Estimation

Existing interactive environments typically use script-based verifiers running within them to determine task success (e.g.checking file contents or system states). These verifiers often involve complex

Two-stage Online Reinforcement Learning

- Stage 1
- Stage 2

Automatic

Task

Obs

Environments

Task Generation

Act

Initial State

GUI Agent RL

Trajectory

Automatic

Task Reward

Test-set Tasks

Reward Estimation

Automatic Task Generation

###### Automatic Reward Estimation

Test-set Tasks

Generated tasks

[Figure 13]

###### Instruction:

I’d like to set up the “Prettier - Code Formatter” extension on VS

Please help me

“Browse the natural products database.”

install the autoDocstring extension in VS Code.

- 0

- 1

Generate Multiple at a Time

###### Task Instruction VLM

Code. Could you

guide me?

|Reward = 0|
|---|

Voting

Screenshots

Please help me set the current user’s line length to 50 characters in VS

Could you adjust the font size of the terminal in VS Code

VLM

Random Initial State

[Figure 14]

0

to 14?

Code.

VLM

Please help me install the autoDocstring extension in VS Code.

Could you help me set up autoindentation for all

Please help me increase the indent of line 2 to line 10 by one tab.

…

All Screenshots

programming

Task Exemplars

languages in VS Code?

- Figure 2: Top: Overview of ZeroGUI. It adopts a Two-stage Online Reinforcement Learning paradigm. In the first stage, tasks are automatically generated by a VLM, while in the second stage, tasks are drawn from the test set. These tasks are executed by the GUI agent. After each interaction, a reward is assigned automatically by the VLM based on the agent’s trajectory, and the policy network is updated via reinforcement learning. Bottom left: Automatic Task Generation. The VLM receives a random initial screenshot and a set of task exemplars to generate diverse novel tasks. Bottom right: Automatic Reward Estimation. The final reward is obtained via majority voting of multiple VLM evaluations based on all screenshots of the trajectory.

commands and logic to cover all possible cases, which heavily relies on manual implementation and debugging. Thus, setting up such verifiers for large-scale generated training tasks is both costly and unnecessary.

To support general task verification or reward estimation in online training, we employ a visionlanguage model (VLM) to assign binary rewards to trajectories. However, VLM-based assessment is often imperfect, e.g., it may overlook details or suffer from hallucinations, leading to incorrect labeling. Among the two error types, i.e., false positives and false negatives, our experiments show that false positives have a greater impact (Sec. 4.3). This is likely because positive rewards offer more informative signals for policy improvement, and too many false positives can distort the improvement. In contrast, negative rewards often provide less targeted guidance, making the effect of false negatives relatively minor.

Therefore, as illustrated in Fig. 2, the reward estimator focuses on reducing false positives and improving precision with the following designs:

- (1) All screenshots in the trajectory are included. The success of some tasks can only be determined by changes in the environment before and after an action, so all screenshots are needed.
- (2) The agent’s responses are excluded. They may contain hallucinations of success, even when the task actually fails. Such content can mislead the VLM to give false positive rewards.
- (3) A voting mechanism is adopted. The VLM is queried multiple times, and the reward is assigned based on either majority agreement or a stricter unanimous agreement (i.e., R = 1 only if all outputs indicate success). This voting strategy further reduces the risk of false positives.

###### 3.3 Two-stage Online Reinforcement Learning

With both automatic task generation and reward estimation mechanisms in place, the GUI agent can perform online learning by continuously interacting with the GUI environment and updating

its policy guided by the rewards. Furthermore, since our reward estimator does not rely on internal environment states or ground-truth labels, it can also provide rewards for test tasks, enabling test-time adaptation. To this end, we introduce a two-stage training strategy:

- (1) Training on generated tasks. The agent learns fundamental capabilities from generated tasks.
- (2) Test-time training. The agent adapts to target test tasks with rewards from the reward estimator.

Reinforcement learning (RL) is adopted for this two-stage online training. We start from the Group Relative Policy Optimization (GRPO) [55], which eliminates the need for an additional value function and is effective for the post-training of LLMs and VLMs in other scenarios [18, 73, 8]. To adapt the original GRPO algorithm to the online RL of GUI agents, we propose the following modifications:

- (1) Extend the optimization objective to multi-step trajectories. For a given task I, we follow GRPO

to sample a group of trajectories {τ(i)}Gi=1 and obtain their rewards {R(i)}Gi=1. The advantage Aˆ(i)of the trajectory τ(i) is computed by normalizing the rewards within the group (Eq. 4). The difference comes that in original GRPO, each sample is a single generated sequence, while in our setting, each trajectory consists of multiple action prediction sequences from model-environment interaction

(Eq. 2). For a trajectory τ(i), we assign each prediction sequence a(ti) with advantage Aˆ(i) and compute its objective as follow:

Jt(i)(θ) =

1 |a(ti)|

|a(ti)|

k=1

min rt,k(i)(θ)Aˆ(i),clip rt,k(i)(θ),1 − ϵ,1 + ϵ A ˆ(i) − βDKL(πθ||πref)

(3) where

rt,k(i)(θ) =

πθ a(t,ki) I,s(ti),a(t,<ki) πθ

old

a(t,ki) I,s(ti),a(t,<ki)

, Aˆ(i) =

R(i) − mean {R(i)}Gi=1 std {R(i)}Gi=1

(4)

where t denotes the t-th step in the trajectory. The action a(ti) is sampled according to Eq. 1, and k indicates the k-th output token. The final objective is the average over all sequences:

J (θ) =

1

G i=1 T(i)

G

i=1

T(i)

t=1

Jt(i)(θ) (5)

- (2) Modify the KL loss term for better training stability. GRPO uses the k3-estimator [53] for the KL

loss, i.e., DKLGRPO = πref/πθ − log πref/πθ − 1. However, we find that it can cause large gradients and is prone to overflow or underflow. We replace it with the k2-estimator [53], i.e., per-token MSE loss,

which provides more stable gradients and avoids numerical overflow:

2

- 1

- 2

log πθ a(t,ki) I,s(ti),a(t,<ki) − log πref at,k(i) I,s(ti),a(t,<ki)

(6)

DKL(πθ||πref) =

Experiments in Sec. 4.3 confirm that this modification improves training stability. Besides, removing the KL constraint may cause policy distribution drift, so simply dropping this loss term is not a desirable option. Further derivations and analyses are provided in the appendix.

## 4 Experiment

- 4.1 Experiment Settings

- 4.1.1 Evaluation Environments and Metrics

OSWorld. OSWorld [68] is a benchmark built upon computer environment designed for evaluating multi-modal agents on complex real-world tasks. It comprises 369 tasks that span web applications, desktop software, and OS-level operations. Among them, 30 tasks (8.1% of the test set) are infeasible by design to assess the ability to detect deprecated or hallucinated features. We report evaluation results on both the full test set and the subset of feasible tasks (i.e., excluding the infeasible ones).

Our evaluation is conducted on the Ubuntu platform from OSWorld with screenshot-only mode. The screen size is 1920×1080 and the maximum number of steps is limited to 15. To reduce the influence of network instability and environmental variability, we report the mean and standard deviation

of scores over 4 runs. Additionally, we incorporate the following metrics for further analysis: (1) pass@k: the expected proportion of tasks the model can solve within k trials, reflecting its potential capacity coverage. (2) all-pass@k: the expected proportion of tasks the model completes in all k trials, indicating the consistency in performance. The unbiased estimators [9] are given by:

n−ci k n k

pass@k := Ex

i∼D 1 −

, all-pass@k := Ex

i∼D

ci k n k

(7)

where each task xi is tested n times and ci is the number of correct samples. We set n = 8, k = 4, and the sampling temperature as 0.5 to estimate these two metrics.

AndroidLab. AndroidLab [70] is an interactive Android environment that includes the Android system and 9 offline-deployable apps (e.g., Clock, Calendar). It comprises 138 test tasks, which are categorized into two types: operational tasks and query-detecting tasks. Operational tasks involve completing goals through operations and are evaluated by predefined rules. Query-detecting tasks require the model to extract information and return a text answer, scored by GPT. We observe that the GPT-based evaluation for certain tasks is not fully reliable, so we report evaluation results on both the full test set and the subset of operation tasks. While existing methods have been evaluated in XML or SoM modes, we implement a screenshot-only setting to support our model and test the corresponding baseline. Success rate (SR) and sub-goal success rate (Sub-SR) are caculated as the metrics.

###### 4.1.2 Implementation Details

For task generation, we use GPT-4o [23] to generate 10 tasks for OSWorld and 5 tasks for AndroidLab at the same time. In total, more than 4,000 Ubuntu-based tasks and 225 Android-based tasks are generated. For training, 725 Ubuntu tasks and 175 Android tasks are randomly sampled from the generated pool, which is approximately twice the size of their respective test sets.

For reward estimation, Qwen2.5-VL-32B [4] is deployed locally for efficiency. We query the VLM 4 times with a temperature of 1.0 and use unanimous agreement voting to determine the reward.

For training, we choose UI-TARS-7B-DPO [48] and Aguvis-7B [71] as our base models. We use the AdamW [38] optimizer with a constant learning rate of 2e-6. For GRPO [55], we set the group size G = 64 and the KL coefficient β = 0.1. We adopt DAPO [76] dynamic sampling, which filters out tasks with accuracy equal to 1 or 0. For each rollout step, sampling continues until 16k sequences are collected, followed by a single gradient update. We train 1 epoch for both generated tasks and test-time tasks. Ablation studies are conducted on the Daily domain of OSWorld (including three apps: Chrome, Thunderbird, and VLC Player) to reduce experimental burden. More details are provided in the appendix.

- 4.2 Main Results

- 4.2.1 OSWorld

We evaluate our proposed ZeroGUI on the OSWorld benchmark, and the results are shown in Tab. 1. Other existing approaches are also listed as reference.

- (1) Compared to the base models, our proposed ZeroGUI leads to significant improvements in task success rate, especially on the feasible subset. Specifically, for UI-TARS-7B-DPO, we achieve a +2.5, 14% improvement on all tasks and +4.5, 40% on the feasible subset. For Aguvis-7B, although the base model performs poorly, our method still yields gains of +1.9, 63% and +2.1, 88%, respectively, with even greater relative improvements. This demonstrates the effectiveness and generalization of our self-improving online training framework.
- (2) Both of our training stages: generated task training and test-time training, contribute to performance gains. The pass@4 and all-pass@4 metrics further reveal their complementary roles. generated task training improves pass@4 significantly, indicating that large-scale and diverse generated tasks help to extent the model’s capability coverage. Test-time training mainly boosts all-pass@4, suggesting that the behavior consistency of the model is enhanced after being adapted to the target tasks. Notably, using only test-time training underperforms the two-stage setup, highlighting that generated training provides a beneficial ability foundation that allows RL in the next stage to unlock more tasks and gain more informative rewards.

- Table 1: Test results on OSWorld benchmark. Test settings and metrics are described in Sec. 4.1. The success rates are reported in “mean±std”. Absolute and relative improvements with respect to the base model are highlighted in green. * reported results taken from previous papers.

Model

Test set Feasible subset

SR pass@4 all-p@4 SR pass@4 all-p@4

GPT-4o [23] 5.0* - - - - Gemini-Pro-1.5 [16] 5.4* - - - - Claude Computer-Use [2] 14.9* - - - - OpenAI Operator [43] 19.7* - - - - -

CogAgent-9B-20241220 [20] 8.1* - - - - Aguvis-72B [71] 10.3* - - - - UI-TARS-72B-DPO [48] 22.7* - - - - -

Aguvis-7B [71] 3.0±0.4 7.2 1.4 2.4±0.5 6.5 0.6 + ZeroGUI (Gen. task only) 4.1±0.3 (+1.1, 37%) 8.0 1.3 3.6±0.4 (+1.2, 50%) 7.6 0.7 + ZeroGUI 4.9±0.4 (+1.9, 63%) 8.3 1.8 4.5±0.4 (+2.1, 88%) 7.9 1.1

UI-TARS-7B-DPO [48] 17.7±1.1 (18.7*) 25.5 9.4 11.3±0.6 18.5 4.9 + ZeroGUI (Test-time only) 18.2±0.9 (+0.5, 3%) 26.4 8.6 14.4±0.8 (+3.1, 27%) 21.7 6.9 + ZeroGUI (Gen. task only) 18.2±1.3 (+0.5, 3%) 27.8 7.5 14.7±1.0 (+3.4, 30%) 22.1 5.6 + ZeroGUI 20.2±1.0 (+2.5, 14%) 28.0 9.6 15.8±0.5 (+4.5, 40%) 22.2 7.3

- Table 2: Test results on AndroidLab benchmark. Test settings and metrics are described in Sec. 4.1. The success rates are reported in “mean±std”. Absolute improvements with respect to the base model are highlighted in green. * reported results taken from previous papers. Note: we fix some code errors in the original task verifiers, and some baseline methods get higher scores after the correction.

Test set Operation subset SR Sub-SR SR Sub-SR

Mode Model

Gemini-1.5-Pro [16] 16.7* 18.4* - Claude-3.5-Sonnet [2] 34.8 (29.0*) 38.5 (32.6*) 34.5 38.8 GPT-4o [23] 38.0 (31.2*) 44.3 (35.0*) 38.4 47.3 AutoGLM [32] 36.2* - - -

SoM

UI-TARS-7B-DPO [48] 45.7±1.52 50.5 54.6±1.72 61.5

Screenshot

+ ZeroGUI (Gen. task only) 46.4±2.05 (+0.7) 52.0 (+1.5) 55.6±2.06 (+1.0) 63.5 (+2.0) + ZeroGUI 47.5±2.12 (+1.8) 52.6 (+2.1) 57.4±2.30 (+2.8) 64.4 (+2.9)

- (3) The improvement on the full test set is smaller than on the feasible subset (e.g., +2.5 vs. +4.5 in the average success rate for UI-TARS-7B-DPO), indicating a decrease in the detection of infeasibility. This may be due to two reasons: (a) the VLM lacks detailed knowledge of specific software, making it hard to judge infeasibility; (b) noisy rewards with false positives may cause the model to become overconfident. To mitigate this, we included a portion of generated infeasible tasks in the training set, which has greatly alleviated this problem (see Sec. 4.3).

- 4.2.2 AndroidLab We also evaluate ZeroGUI on the AndroidLab benchmark, with the results shown in Tab. 2.

- (1) From the SR perspective, ZeroGUI achieves a +2.8 improvement on the operation subset and a

+1.8 improvement on the full test set. This demonstrates that the proposed ZeroGUI generalizes well across different interactive GUI environments.

- (2) From the Sub-SR perspective, ZeroGUI achieves a +2.9 improvement on the operation subset. Despite leveraging only the overall task rewards, it still yields performance gains in sub-task metrics.

- 4.3 Ablation Study

Task Generation. Tab. 3a ablates our designs of task generation. (1) Removing examples during task generation or generating only one task at a time leads to a drop in test performance. We attribute this to two factors: providing task examples helps align the distribution of generated tasks with the target domain, while generating multiple tasks increases diversity, which are crucial for training data.

- Table 3: Ablations of key components in task generation and reward estimation. The models are trained and tested on the Daily domain of OSworld.“SR”, “Feas.”, and “Infeas.” denotes the success rates on all test tasks, feasible tasks, and infeasible tasks, respectively. “w/o multi. cand.”: without multiple candidates. “All Screen.”: all screenshots. “With Res.”: with agent’s response.

(a) Task generation Method SR Feas. Infeas.

w/o examples 22.3 20.8 33.8 w/o multi. cand. 24.1 22.0 40.7 w/o infeasible 24.0 24.2 22.1 Ours 27.2 25.4 41.3

(b) Reward estimation All Screen. With Res. Voting Precision Recall SR

- - - 47.5 40.4 23.9 ✓ - - 53.7 61.7 25.6 ✓ ✓ - 44.3 74.5 24.0 ✓ - ✓ 61.5 51.1 27.2

(2) Excluding infeasible tasks results in a sharp decline on the infeasible subset, indicating that such tasks help the model identify unachievable goals and reduces overconfidence.

Reward Estimation. We ablate the key designs in the reward estimator. First, a set of trajectories (UI-TARS-7B-DPO on generated tasks) are randomly selected and manually labeled the ground-truth rewards. We then apply different reward estimation methods to this set and evaluate their precision and recall. In addition, we train separate models using rewards estimated by each method and compare their success rates on the test tasks.

The results are shown in Tab. 3b. (1) Using only the final screenshot instead of all screenshots results in low precision, recall, and test success rate. (2) Including the agent’s response during reward estimation yields the highest recall but significantly lowers both precision and test success rate, indicating that the VLM is misled by the response and produces many false positives. (3) Excluding the agent’s response and applying a voting mechanism increase precision while decreasing recall, and also lead to a notable improvement in test success rate. This suggests that false positive errors have a more detrimental effect on model training.

RL Training. To evaluate the effectiveness of our online RL training, we compare it against two baselines: offline rejection sampling fine-tuning (RFT) and online RFT. Results are reported in Tab. 4. Offline RFT first collects trajectories for all tasks using the base model and fine-tunes only on positive samples. Its performance is limited due to a distribution mismatch between the collected trajectories and the updated policy, and it fails to leverage rewards from new tasks discovered after policy updates. Online RFT performs better but still lags behind online RL. This is mainly because RFT discards all negative samples, while RL enables the model to learn from them and avoid repeating past mistakes.

We evaluate the effect of replacing the k3-KL loss in original GRPO with a k2-KL loss. As shown in Fig. 3, k2-KL yields higher and more stable training accuracy. Test success rates in Tab. 4 further validate the superiority of k2-KL in our setting.

- Table 4: Ablations of online RL training and the modified KL loss. The models are trained and tested on the Daily domain of OSworld.

Training Method SR

Offline RFT 22.4 Online RFT 24.5 Online RL (Ours) 27.2

k3-KL (GRPO) 26.1 k2-KL (ours) 27.2

0 5 10 15 20 25 30 Training Steps

0.1

0.2

0.3

0.4

0.5

0.6

0.7

TrainingAccuracy

GRPO (k3-KL)

Ours (k2-KL)

Figure 3: Comparison of training accuracies with k3-KL (GRPO) and k2-KL (ours).

- 5 Conclusion

In this work, we present ZeroGUI, a fully automated online learning framework for GUI agents that eliminates the need for manually collected and labeled offline training data. By leveraging vision-language models (VLMs), ZeroGUI enables automatic generation of training tasks and reward signals within GUI environments, removing human cost from both task design and evaluation.

We further introduce a two-stage reinforcement learning strategy: training on generated tasks to acquire general capabilities, followed by test-time adaptation using VLM-based rewards. We conduct extensive experiments across different base models (Aguvis-7B and UI-TARS-7B-DPO) and different GUI environments. Notably, ZeroGUI-Aguvis-7B achieves a 63% relative improvement, while ZeroGUI-UI-TARS-7B shows a 14% relative improvement.

Acknowledgements. The work is supported by the National Key R&D Program of China (NO. 2022ZD0161300), by the National Natural Science Foundation of China (U24A20325, 62321005, 62376134).

## A Implementation Details

###### A.1 Task Generation

We use GPT-4o [23] for task generation. Each prompt consists of task exemplars and the screenshot of a randomly sampled initial state. To encourage diversity, each prompt generates multiple task candidates: 10 per prompt in OSWorld [68] and 5 per prompt in AndroidLab [70]. In total, we generate over 4,000 Ubuntu-based tasks and 225 Android-based tasks, from which we randomly sample 725 and 175 tasks, respectively, as the final training set. The prompt template used to generate OSWorld tasks is shown below:

Task Generation Prompt Template

You will be given a screenshot of the current state of the computer. Based on the screenshot, generate {Number of Instructions} human-like instructions for an Ubuntu-based task related to using {App Name} app.

- 1. The task should be something that **can be achieved** using common interactions on a standard Ubuntu system from the current state.
- 2. The proposed task should be similar to the example tasks but not exactly the same.
- 3. The {Number of Instructions} instructions should cover different aspects of using {App Name}. During the generation, think about any other kind of tasks differ from the examples you have generated before.
- 4. The tasks should be clear and unambiguous, with specific and detailed instructions, and can be evaluated objectively by observing the action trajectory.
- 5. If the task you propose requires access to any internet resources, you must ensure that the URL or resource exists.
- 6. The task you propose must require at least 3 steps to complete, and the difficulty and the complexity should be evenly distributed.

The following are examples of good task instructions (their initial screenshots/states are different):

{List of Exemplars}

Please follow the **content, type, and distribution** of the examples provided to you, and generate task instructions based on the initial screenshot.

**Do not repeat the existing examples.** Now, you will be given a initial computer state screenshot. Please generate

{Number of Instructions} task instructions based on the screenshot. Return only the task instruction text. Each instruction should occupy a single line. {Number of Instructions} lines of instructions should be returned.

###### A.2 Reward Estimation

For efficient reward estimation, we use a locally deployed QwenVL-2.5-32B-Instruct [4]. The VLM assigns binary rewards (success/failure) to the agent’s trajectories. The VLM is prompted with all screenshots of the trajectory and the task instruction. To improve reliability, we use a voting mechanism: the VLM is queried multiple times with temperature 1.0, and the final reward is based on unanimous agreement. The reward estimation prompt for OSWorld tasks is shown below:

Reward Estimation Prompt Template

You will be given a task instruction and a series of screenshots of the task execution. Please analyze the screenshots and provide a detailed analysis of the task completion by following the steps below:

- 1. First, analyze and understand the task instruction. Describe what should the screenshots look like if the task is completed successfully.
- 2. Describe what you observe in each screenshot, analysis what actions were taken and what changes were made to the UI to achieve the task (or mistakes made).
- 3. When you analyze the screenshots, please pay attention to the very detailed elements and changes in the UI. Every small detail may affect the final result.
- 4. After all screenshots are analyzed, provide a overall reasoning about how the task was completed or failed at **the final state**. Make sure you have considered all demands of the task instruction.
- 5. Determine if the task was completed at **the final state** (the last screenshot) successfully (score 1 for success, 0 for failure). If the task is completed during the process but not at the final state, it should be considered as failure (0 score).

Provide your response strictly in the following format: TASK REQUIREMENT: [Your understanding of the task instruction]

SCREENSHOT ANALYSIS:

- Screenshot 1: [Analysis of first screenshot]
- Screenshot 2: [Analysis of second screenshot]

... REASONING: [Your reasoning] FINAL ANSWER: [Your final answer] SCORE: [0/1] Here is an example: (Task Instruction: Please help me backup my emails in "Bills" folder in Thunderbird and store the .eml files with only subject names to my Google Drive folder called "emails".) TASK REQUIREMENT:

- - Backup the emails in "Bills" folder in Thunderbird.
- - Store the backup .eml files with only subject names, and the emails should be saved in the Google Drive folder called "emails".
- - Once succeed, the emails should be visible in the Google Drive folder "emails". Or at least there should be a saving action performed. SCREENSHOT ANALYSIS:

- Screenshot 1:

- - Thunderbird email client is open.
- - The "Bills" folder is visible under "Local Folders."
- - There is no observable action performed yet in this screenshot.

- Screenshot 2:

- - The "Bills" folder has been selected, and the folder content is displayed.
- - Two emails are visible: "Amazon Web Services Invoice Available" and "Your receipt from X (formerly Twitter)."

- - No further actions are taken on the emails.

- Screenshot 3:

- - Both emails in the "Bills" folder are selected.
- - Content previews of both emails are displayed on the right-hand side.
- - No observable attempt to export or save the emails is visible.

- Screenshot 4:

- - The right-click context menu is accessed for the selected emails.
- - The "Save As..." option is hovered over, indicating intent to save the selected emails.

- Screenshot 5:

- - The file navigation window opens, allowing the user to choose a save destination.
- - No specific Google Drive folder (e.g., "emails") is accessed or visible in this screenshot.

- Screenshot 6:

- - The "Desktop" option in the file picker is hovered over.
- - Still no indication of Google Drive folder ("emails") selection.

- Screenshot 7:

- - The "Show other locations" option is hovered over in the file picker.
- - No confirmation that the user is navigating to Google Drive or saving the files with subject names only.

- Screenshot 8:

- - The "Software Updates Available" notification appears. The file picker is still open without any observable confirmation of file saving or destination selection.
- - It remains unclear where or if the emails have been saved.

REASONING: Based on the screenshots provided:

- 1. While there was some intent to save the emails (as shown by the selection and access of the "Save As..." function), there is no confirmation that the .eml files were saved with subject names only and placed in the required Google Drive folder ("emails").
- 2. The screenshots lack evidence of the completion of the task as per the instructions.

FINAL ANSWER: The task was not completed successfully due to the lack of observable saving action.

SCORE: 0 Now, please **strictly follow the format** and analyze the following screenshots (The last line should only be SCORE: [0/1], no other text): Task Instruction: {instruction} Screenshots (by order): {screenshots}

- A.3 RL Training Hyper-parameters of our RL training are listed in Tab. 5. B Analysis of KL Loss
- B.1 Theoretical Derivation and Analysis Here, we derive the gradient from the KL loss and show why our modification benefits stable training.

Table 5: Training hyper-parameters of ZeroGUI. GRPO group size 64 KL loss coefficient 0.1 rollout temperature 0.5 rollout top-p 0.9 rollout frequency penalty 1.0 train batch size 16384 optimizer AdamW learning rate 2e-6 lr schedule constant weight decay 0.0 optimizer momentum β1,β2 = 0.9,0.95

In our implementation, the model has only a single update after each exploration stage, ensuring that πθ = πθ

. Therefore, Eq. 3 can be simplified by removing the min and clip operation:

old

|a|

1 |a|

J (θ) =

k=1

πθ(ak) πθ

Aˆ − βDKL(πθ||πref) (8)

(ak)

old

where a denotes a predicted action sequence with the corresponding advantage Aˆ, πθ(ak) stands for πθ (ak|I,s,a<k), I is the task instruction and s is the agent’s state.

GRPO. GRPO [55] uses a per-token k3-estimator [53] for the KL loss:

πref(ak) πθ(ak) − log

πref(ak) πθ(ak) − 1 (9)

DKLGRPO(πθ||πref) =

The gradient with respect to the parameter θ is:

|a|

1 |a|

∇θJ GRPO(θ) =

k=1

A ˆ + β

πref(ak) πθ(ak) − 1 ∇θ log πθ(ak) (10)

Referring to GRPO, the Gradient Coefficient of the KL loss can be written as:

πref(ak) πθ(ak) − 1 (11)

GCKLGRPO(ak) =

Ours. We replace the original KL loss in GRPO with the k2-estimator [53], i.e., per-token MSE loss:

- 1

- 2

(log πθ(ak) − log πref(ak))2 (12) The gradient becomes:

DKL(πθ||πref) =

|a|

1 |a|

∇θJ (θ) =

k=1

A ˆ + β (log πref(ak) − log πθ(ak)) ∇θ log πθ(ak) (13)

GCKL(ak) = log πref(ak) − log πθ(ak) (14)

Analysis. We plot the curve of the gradient coefficients GCKL and GCKLGRPO in Fig. 4, with log πθ − log πref as the x-axis. We also record the KL loss and the token-wise maximum and minimum of log πθ − log πref during training, as shown in Fig. 5. As the training model θ gradually deviates from

20

GRPO (k3-KL)

Ours (k2-KL)

15

10

GCKL

5

0

3 2 1 0 1 2 3 log log ref

Figure 4: Gradient coefficient of KL loss.

0.25

0

GRPO (k3-KL)

GRPO (k3-KL)

GRPO (k3-KL)

30

Ours (k2-KL)

Ours (k2-KL)

Ours (k2-KL)

5

0.20

25

max(loglog)ref

min(loglog)ref

20

10

0.15

KLloss

15

15

0.10

10

20

0.05

5

25

0.00

0

0 5 10 15 20 25 30 Training Steps

0 5 10 15 20 25 30 Training Steps

0 5 10 15 20 25 30 Training Steps

(a) KL loss.

(b) max(log πθ − log πref)

(c) min(log πθ − log πref)

Figure 5: KL loss curve and token-wise maximum and minimum of log πθ − log πref during training. the reference model, there exist some tokens such that log πθ ≪ log πref (Fig. 5c). For GRPO, this leads to a huge gradient coefficient, which may cause training instability. In contrast, ours has a stable gradient coefficient and avoids large gradients. Some concurrent works [34] have also reached derivations and conclusions similar to ours.

Furthermore, in practical implementations, language models typically output log probabilities, i.e., log π, computed from logits. When computing the KL loss of GRPO (Eq. 9), exponentiation is required, which is prone to overflow or underflow. In contrast, our KL loss (Eq. 12) avoids this issue.

###### B.2 Supplementary Ablation Study

|26.0<br><br>26.7<br><br>27.2<br><br>25.5| | | | |
|---|---|---|---|---|
| | | | | |

TestSR

0 0.01 0.1 1.0 Value of

Figure 6: Test success rates with different KL loss coefficients β.

We compare different KL loss coefficients β, and the results shown in Fig. 6. Although some existing work [76, 36] suggests removing the KL penalty for general reasoning tasks, our findings differ in the context of training GUI agents. We observe that setting β yields the best test performance. Removing the KL loss entirely (β = 0) or using a small β (e.g., 0.01) leads to performance degradation, likely

due to the policy distribution drift that the model overfits to current tasks. In contrast, a large β (e.g.,

1) imposes excessive constraints on optimization, also resulting in worse results.

## C Analysis of Task Generation

###### C.1 Task Example

Table 6: Examples of generated tasks and test tasks on OSWorld

Domain Generated Task Instructions Test Task Instructions

Look up “MIT’s Deep Learning State of the Art lecture” mentioned in the article and play the video if it’s available.

Find a men’s T-Shirt that is in large size with a stripe pattern, short sleeve and under the Sales&Discount.

Chrome

Could you add a new layer to the image, call it “Overlay”, and set its mode to “Multiply”?

Could you make the background of this image transparent for me?

GIMP

Could you create a new column called “Average Sales” for each product in Zones 1, 2, and 3 by calculating the average of Q1, Q2, Q3, and Q4? Could you add the hyperlink “https://its.example.com” to the text “Information Technology Services (ITS)” in the main content?

I would like to pad all the numbers in the ‘Old ID’ column with zeros in front, to fill them up to seven digits in the ‘New 7 Digit ID’ column.

Office

Export the current document into PDF, keep the file name

Could you create a folder called “WorkProjects” in your home directory and put a copy of “installed_packages.txt” inside it?

Use terminal command to count all the lines of all php files in current directory recursively, show the result on the terminal

OS

Could you adjust the playback speed in VLC to 1.5x for this video? I want to finish it faster, even if the audio isn’t perfect.

Help me modify the folder used to store my recordings to Desktop

VLC

I’d like to stop the Welcome Page from showing up every time I open VS Code. Can you help me with that?

Please help me use VS Code to open the “project” in the “user” folder under “home”.

VS Code

As shown in Tab. 6 and Tab. 7, we list some of the generated tasks and the test tasks on OSWorld and AndroidLab for comparison.

The generated tasks leverage the world knowledge embedded in VLMs to extend common operations, thereby enhancing data diversity. For example, the model generates Android tasks such as enabling auto-rotate and setting screen timeout, and Ubuntu tasks like speeding up video playback—features not present in the original test tasks.

Additionally, the generated tasks are both feasible and well-grounded. For instance, tasks like copying a package list file to a new project folder and averaging sales data for Ubuntu, or setting reasonable alarms and editing calendars as humans would for Android, ensure practical applicability.

###### C.2 Visualization

We visualize the task instructions in the original test set and our generated set. Following OSGenesis [59], we use Sentence-BERT [52] to encode instructions into a high-dimensional semantic space, which is then reduced to two dimensions using UMAP [42]. As shown in Fig. 7, orange points represent instructions from the original OSWorld test set, while blue points correspond to instructions generated by our method.

Each cluster in the visualization corresponds to a distinct application domain (e.g., specific applications or task types), reflecting strong semantic coherence within domains. Notably, the generated instructions, conditioned on exemplars, not only align well with their target domains, but also extend

Table 7: Examples of generated tasks and test tasks on AndroidLab

Apps Generated Task Instructions Test Task Instructions

I need to set a 10:30PM clock every weekend, and label it as “Watch Football Games” to remind me. Does my alarm at 4PM turn on vibrate?

Set an alarm for 5:30AM every Tuesday and Thursday, label it as “Workout”, and disable vibrate.

Clock

Arrange an event titled “project deadline” on March 15th, and attach a note saying “Submit by noon.”

I want to add an event at 5:00PM today, whose Title is “work”.

Calendar

Change the type of the transaction on May 2, 2024, from “income” to“expense”, adjust the amount to 520 CNY, and change the note to “Wrong Operation”.

Modify the transaction on May 3, 2024, from “expense” to “income”, set the amount to 780 CNY, and update the note to “Refund”.

Bluecoins

Add birthday to AAA as 1996/10/24 Set contacts ABC’s website to be abc.github.com

Set Xu’s birthday to be 1989/06/15 and add his website as xuportfolio.com.

Contacts

Set screen timeout duration to 5 minutes. Enable the auto-rotate screen feature.

Turn on airplane mode of my phone Turn my phone to Dark theme

Setting

[Figure 15]

Figure 7: Visualization of the test and generated task instructions in OSWorld.

into previously unexplored regions of the instruction space. This illustrates the capacity of our method to produce semantically valid and diverse tasks beyond the original dataset.

## D Qualitative Comparisons

We conduct case studies to further demonstrate the effectiveness of our proposed ZeroGUI. We observe that the base model UI-TARS-7B-DPO [48] shows limited task comprehension and inadequate attention to detail, frequently falling into repetitive action loops during task execution. In contrast, after our ZeroGUI training, the model exhibits significantly more stable behavioral strategies and stronger task execution capabilities.

For example, as shown in Fig. 8 and 9, a task in the VS Code domain of OSWorld involves the instruction: “I want to make the tabs wrapped over multiple lines when exceeding available space, please help modify the setting of VS Code.” During execution, the base model attempts to modify the “Tab Size” parameter but fails to delete the default value before entering a new one. Instead, it

prepends the new number to the existing value, resulting in an incorrect setting. This faulty operation is then repeated multiple times, indicating that the model lacks the ability to detect invalid actions. In contrast, our model adopts a more robust action strategy: it first uses a keyboard shortcut to select all existing content, then inputs the correct value, and successfully completes the task.

Another example comes from a task in the LibreOffice Impress domain, where the instruction is:

“Add an image ‘none.png’ on the Desktop to slide 2 with 1cm*1cm size.” As shown in Fig. 10 and 11, after clicking the “Insert” menu, the base model attempts to select the “Image” option but misclicks a blank area due to inaccurate grounding, causing the menu to close prematurely. However, the model fails to detect this change and continues to attempt to click the “Image” option under the now-closed “Insert” menu, resulting in ineffective repetition. In contrast, our model completes the full insertion process more reliably. It successfully opens the image insertion interface, selects the correct image file, adjusts the width and height step by step, and ultimately finishes the long-horizon task accurately. This comparison further demonstrates the improved stability and general performance of our method in handling complex tasks.

When executing tasks in AndroidLab, the model also shows significant performance improvements. Take a Calendar task as an example, with the instruction: “You should use calendar to complete the following task: Arrange an event titled ‘homework’ for me at May 21st, and set the notification time to be 10 minutes before.”, as illustrated in Fig. 12 and 13. The base model shows insufficient understanding of the instruction, overlooking details such as the event title and notification time. It neither adds the event title nor sets the notification, and closes the interface ultimately. In contrast, our trained model accurately captures and executes these detailed requirements, successfully adding the event on the specified date and setting the notification, which demonstrates the effectiveness of our training framework in mitigating issues arising from overlooked details and improving task accuracy.

It is worth noting that the base model fails to complete any of the above tasks, indicating the lack of supervision from successful trajectories. Through training on generated tasks, our model learns more generalizable interaction strategies. This capability not only enhances its performance on specific tasks, but also demonstrates the significant potential and practical effectiveness of the proposed training framework in enabling the model to adapt to complex GUI environments.

#### Base Model

[Figure 16]

[Figure 17]

[Figure 18]

1 2 3

[Figure 19]

[Figure 20]

[Figure 21]

Thought: To modify the tab wrapping settings in VS Code, I need to open the settings panel first. In VS Code, this can be quickly accessed using the shortcut 'Ctrl+,'. By pressing 'Ctrl+,', I can open the settings interface and locate the options related to tab wrapping for further adjustments. Action: hotkey(key='ctrl ,')

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space. Action: click(start_box='(520,408)')

Thought: I have already opened the settings panel in VS Code. Now, I need to type "tab" into the search box to find the settings related to tab wrapping. By entering "tab" in the search field, I can quickly locate the relevant configuration options. Action: type(content='tab')

[Figure 22]

[Figure 23]

[Figure 24]

4 5 6

[Figure 25]

[Figure 26]

[Figure 27]

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space. Action: click(start_box='(520,464)')

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space. Action: click(start_box='(520,408)')

change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space.

Action: type(content='8')

[Figure 28]

[Figure 29]

[Figure 30]

7 8 9

[Figure 31]

[Figure 32]

[Figure 33]

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and

change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space.

change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space.

change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space.

Action: type(content='8')

Action: type(content='8')

Action: type(content='8')

- Figure 8: The trajectory of the base model UI-TARS-7B-DPO when executing a VS Code task. The instruction is “I want the tabs to wrap onto multiple lines when they exceed the available space. Please help modify the VS Code settings accordingly.” The original thoughts are in Chinese and have been translated into English for presentation purposes. Subsequent repetitive steps have been omitted.

Our Model

[Figure 34]

[Figure 35]

[Figure 36]

1 2 3

[Figure 37]

[Figure 38]

[Figure 39]

Thought: To modify the tab wrapping settings in VS Code, I need to open the settings panel first. In VS Code, this can be quickly accessed using the shortcut 'Ctrl+,'. By pressing 'Ctrl+,', I can open the settings interface and locate the options related to tab wrapping for further adjustments. Action: hotkey(key='ctrl ,')

Thought: I have already opened the settings panel in VS Code. Now, I need to type "tab" into the search box to find the settings related to tab wrapping. By entering "tab" in the search field, I can quickly locate the relevant configuration options. Action: type(content='tab')

Thought: I have already opened the settings panel in VS Code and searched for "tab", and now I can see the settings related to tabs. I need to locate the "Editor: Tab Size" option to configure tab wrapping. I should click on "Editor: Tab Size" and change its value to a larger number, which will enable automatic tab wrapping when the tabs exceed the available space. Action: click(start_box='(520,406)')

[Figure 40]

[Figure 41]

[Figure 42]

4 5 6

[Figure 43]

[Figure 44]

[Figure 45]

Thought: I have already opened the settings panel in VS Code and located the "Editor: Tab Size" option. Now I need to modify this value to enable the tab wrapping feature. I should change the current value from 4 to a larger number, such as 8, so that when the tab content exceeds 8 characters, it will automatically wrap. I need to double-click the current value 4 to activate the input box, then enter the new value 8. Action: left_double(start_box='(570,467)')

Thought: I have successfully opened the settings panel in VS Code and located the "Editor: Tab Size" option. Now, I need to change the current value from 4 to 8 to enable tab wrapping. I should first delete the current value 4, then enter the new value 8. To do this, I should press Ctrl+A to select all the text, and then type the new value 8. Action: hotkey(key='ctrl a')

Thought: I have selected the current value 4 of "Editor: Tab Size" and now need to enter the new value 8 to enable tab wrapping. I should simply type 8 to replace the selected value 4.。 Action: type(content='8')

[Figure 46]

[Figure 47]

7 8

[Figure 48]

[Figure 49]

Thought: I have successfully changed the tab wrapping setting in VS Code to 8 and saved the changes. The setting is now effective, and tabs will automatically wrap when their content exceeds 8 characters. The task is complete—I have successfully modified and saved the tab wrapping setting in VS Code, and no further actions are needed. Action: finished()

Thought: I have successfully changed the value of "Editor: Tab Size" from 4 to 8, and now I need to save this setting. In VS Code, I can use the shortcut Ctrl+S to save the settings. I should use the shortcut Ctrl+S to save the changes made to the tab wrapping settings in VS Code. Action: hotkey(key='ctrl s')

###### Figure 9: The trajectory of our trained model when executing a VS Code task. The instruction is

“I want the tabs to wrap onto multiple lines when they exceed the available space. Please help modify the VS Code settings accordingly.” The original thoughts are in Chinese and have been translated into English for presentation purposes.

Base Model

[Figure 50]

[Figure 51]

[Figure 52]

1 2 3

[Figure 53]

[Figure 54]

[Figure 55]

Thought: I need to add a 1cm by 1cm image to slide 2. From the current interface, I can see that I first need to click the slide thumbnail on the left to select slide 2. I should first click the thumbnail of the second slide in the left slide panel to begin editing. Action: click(start_box='(86,358)')

Thought: I have opened the Insert menu and now need to click the "Image..." option to insert an image. This option is located at the top of the menu. I should click the "Image..." option under the Insert menu to open the file selection dialog and choose the image to insert. Action: click(start_box='(133,70)')

Thought: I have successfully selected slide 2 and now need to insert an image. From the toolbar, I can see the "Insert" option. I should click it to insert the image. I need to click the "Insert" option in the top menu bar to access the image insertion options. Action: click(start_box='(122,69)')

[Figure 56]

[Figure 57]

[Figure 58]

4 5 6

[Figure 59]

[Figure 60]

[Figure 61]

Thought: I have opened the Insert menu and now need to click the "Image..." option to insert an image. This option is located at the top of the menu. I should click the "Image..." option under the Insert menu to open the file selection dialog and choose the image to insert. Action: click(start_box='(132,71)')

Thought: I have opened the Insert menu and now need to click the "Image..." option to insert an image. This option is located at the top of the menu. I should click the "Image..." option under the Insert menu to open the file selection dialog and choose the image to insert. Action: click(start_box='(132,71)')

Thought: I have opened the Insert menu and now need to click the "Image..." option to insert an image. This option is located at the top of the menu. I should click the "Image..." option under the Insert menu to open the file selection dialog and choose the image to insert. Action: click(start_box='(133,70)')

- Figure 10: The trajectory of the base model UI-TARS-7B-DPO when executing a LibreOffice Impress task. The instruction is “Add an image ‘none.png’ on the Desktop to slide 2 with 1cm*1cm size.” The original thoughts are in Chinese and have been translated into English for presentation purposes. Subsequent repetitive steps have been omitted.

#### Our Model

[Figure 62]

[Figure 63]

[Figure 64]

1 2 3

[Figure 65]

[Figure 66]

[Figure 67]

Thought: I see that a presentation is currently open in LibreOffice Impress, and I need to add a 1cm by 1cm image named "none.png" to slide 2. First, I need to click the second slide in the thumbnail pane on the left. I should click the second slide thumbnail in the left panel to start editing. From the thumbnail, I can see that slide 2 contains some content about marketing strategies. Action: click(start_box='(86,365)')

Thought: I have successfully selected slide 2. Now, I need to insert an image. In LibreOffice Impress, I can insert an image through the "Insert" option in the top menu bar. I should click the "Insert" menu at the top, then choose the option to insert an image in order to add our "none.png" picture. Action: click(start_box='(120,69)')

Thought: I have opened the Insert menu and now need to select the "Image..." option to insert an image. This option is located at the top of the menu. I should click the "Image..." option under the Insert menu to open the file selection dialog and choose our "none.png" image.。 Action: click(start_box='(133,84)')

[Figure 68]

[Figure 69]

[Figure 70]

4 5 6

[Figure 71]

[Figure 72]

[Figure 73]

Thought: I have successfully navigated to the Desktop directory and can now see the "none.png" file. I need to select and open this file to insert it into the slide. I should double-click the "none.png" file to select it, then click the "Open" button to insert it into the slide. Action: left_double(start_box='(321,251)')

Thought: I have successfully inserted the image into the slide and now need to resize it to 1cm by 1cm. In the right-side properties panel, I can see the "Position and Size" option. I need to click it to set the image size. I should click the "Position and Size" option in the right properties panel to open the image size settings dialog. Action: click(start_box='(830,628)')

Thought: I have opened the image insertion

dialog and now need to navigate to the Desktop to select the "none.png" file. I see the Desktop option in the left navigation pane. I should click the

Desktop option in the left sidebar to access the files on the desktop. Action: click(start_box='(239,264)')

[Figure 74]

[Figure 75]

[Figure 76]

7 8

9

[Figure 77]

[Figure 78]

[Figure 79]

Thought: I have opened the image size settings dialog and now need to resize the image to 1cm by 1cm. I see that the current width and height are both 13.55cm. I need to change both values to 1cm. I should first click the width input box and enter the value 1cm. Then, I also need to set the height to 1cm.。 Action: click(start_box='(911,720)')

Thought: I have selected all the text in the width input box and now need to enter the new width value of 1cm. I should simply type in the new width value, 1cm, to adjust the image width to 1cm. Action: type(content='1cm')

Thought: I have selected the width input box and now need to enter the new width value of 1cm. I should first delete the current value by pressing Ctrl+A to select all the text, then type in the new width value of 1cm. Action: hotkey(key='ctrl a’)

[Figure 80]

[Figure 81]

[Figure 82]

10 11 12

[Figure 83]

[Figure 84]

[Figure 85]

Thought: I have selected all the text in the height

Thought: I have successfully set the width to 1cm and now need to set the height to 1cm to complete the 1cm by 1cm image size adjustment. I should click the height input box and then enter the value 1cm. I need to click the height input box and type 1cm to finalize the image size setting. Action: click(start_box='(911,760)')

Thought: I have selected the height input box and now need to change the height value from 13.55cm to 1cm. I should first press Ctrl+A to select all the text, then enter the new height value. I should press Ctrl+A to select the current height value, then type the new height value, 1cm, to complete the image size adjustment. Action: hotkey(key='ctrl a’)

input box and now need to enter the new height value of 1cm to complete the 1cm by 1cm image size adjustment. I should simply type in the new

height value, 1cm, to finalize the image size setting. Action: type(content='1cm')

[Figure 86]

[Figure 87]

13 14

[Figure 88]

[Figure 89]

Thought: I have successfully set the image size to 1cm by 1cm and now need to save these changes to complete the task. I should use the shortcut

Thought: I have successfully completed all the tasks: inserting the "none.png" image from the desktop into slide 2 and resizing the image to 1cm by 1cm. From the screen, I can see that the image is correctly displayed on the slide and the size adjustment is complete. All task objectives, including inserting the image and setting the correct size, have been achieved. The task can now be concluded. Action: finished()

Ctrl+S to save the file. Using Ctrl+S will save the current modifications and ensure the image size settings are preserved.

Action: hotkey(key='ctrl s')

- Figure 11: The trajectory of our trained model when executing a LibreOffice Impress task. The instruction is “Add an image ‘none.png’ on the Desktop to slide 2 with 1cm*1cm size.” The original thoughts are in Chinese and have been translated into English for presentation purposes.

### Base Model

1 2 3

###### Thought:

###### Thought:

[Figure 90]

Thought: To proceed with arranging the event, I need to open the Calendar app. This requires identifying and tapping on the Calendar app icon, which is

[Figure 91]

[Figure 92]

To proceed with arranging the event, I need to initiate the process of creating a new event. The "+" button is the standard way to add a new event in most calendar applications, so tapping it will allow me to start entering the event details. Tap on the blue "+" button in the bottom right corner to begin creating a new event. Action: click(start_box='(357,254)')

To proceed with arranging the event, I need to initiate the process of creating a new event. The "+" button is the standard way to add a new event in most calendar applications, so tapping it will allow me to start entering the event details. Tap on the blue "+" button in the bottom right corner to begin creating a new event. Action: click(start_box='(895,831)')

| |
|---|

| |
|---|

clearly visible on the screen. Tap on the "Calendar" app

icon to open the Calendar application. Action: click(start_box='(606,381)')

| |
|---|

4 5 6

Thought: To proceed with arranging the event, I need to confirm or modify the notification time. Since the task specifies setting the notification time to 10 minutes before, I will tap on the "9:00 AM" field to access the time selection interface and

###### Thought:

[Figure 93]

[Figure 94]

Thought: To proceed with arranging the event, I need to input the title "homework" into the "Event..." field. This is the first step in defining the event's details, and entering the title is essential for identifying the event. Type "homework" into the "Event..." field to set the event's title. Action: type(content='homework')

[Figure 95]

To proceed with arranging the event, I need to select the "New event" option from the pop-up menu. This option is specifically designed for creating new events, which is the next logical step in completing the task. Tap on the "New event" option in the pop-up menu to begin creating a new event. Action: click(start_box='(678,690)')

| |
|---|

make the necessary adjustment. Tap on the "9:00 AM" field to

open the time selection interface for modifying the notification time. Action: click(start_box='(688,239)')

| |
|---|

7 8 9

[Figure 96]

[Figure 97]

[Figure 98]

###### Thought:

Thought: To set the notification time to 10 minutes before the event, I need to adjust the hour to 8:50 AM. The red circle indicates the target position on the clock face, which corresponds to the desired time. Tapping on this position will update the hour and minute accordingly.

Thought: To finalize the notification time of 8:50 AM, I need to confirm the selection by tapping the "OK" button. This action will save the chosen time and return to the event creation

To set the notification time to 10 minutes before the event, I need to adjust the minute to

50. The red circle indicates the target position on the clock face, which corresponds to the

desired time. Tapping on this

interface, allowing me to

| |
|---|

position will update the minute accordingly. Tap on the "50" position on the clock face to set the minute to 50. Action: click(start_box='(288,503)')

continue setting up the event. Tap on the "OK" button to confirm the selected time of 8:50 AM. Action: click(start_box='(804,760)')

| |
|---|

Tap on the "8" position on the clock face to set the hour to 8. Action:

| |
|---|

click(start_box='(288,616)')

10 11

[Figure 99]

[Figure 100]

Thought: To complete the task of arranging the event, I need to confirm the details and save the event. Tapping the "X" button will likely close the event creation interface, indicating that the event has been successfully created and saved. Tap on the "X" button in the top-left corner to confirm and save the event. Action: click(start_box='(67,87)')

Thought: The task of arranging the event titled "homework" for May

| |
|---|

21st with a notification time of 10 minutes before has been

successfully completed. The event is visible on the calendar, and no further actions are required to fulfill the task. I will now submit the task as completed.

Submit the task, confirming that it has been successfully completed.

Action: finished()

- Figure 12: The trajectory of the base model UI-TARS-7B-DPO when executing a Calendar task. The instruction is “You should use calendar to complete the following task: Arrange an event titled

‘homework’ for me at May 21st, and set the notification time to be 10 minutes before.”

##### Our Model

1 2 3

Thought: To proceed with arranging the event, I need to select May 21st on the calendar. This will allow me to specify the date for the event and move forward with setting the title and notification time. Tap on the date "May 21st" in the calendar to select it for the

Thought: To proceed with arranging the event, I need to initiate the process of creating a new event. Clicking on the "+" button will allow me to access the event creation interface, where I can input the event title, set the date, and configure the notification time.

Thought: To proceed with arranging the event, I need to open the Calendar app. This will allow me to create a new event and set the necessary details, such as the title, date, and notification time. Clicking on the Calendar app is the logical next step to achieve this.

[Figure 101]

[Figure 102]

[Figure 103]

| |
|---|

| |
|---|

event. Action:

Tap on the blue "+" button in the bottom right corner to start

Tap on the "Calendar" app icon to open the Calendar application.

click(start_box='(357,254)')

creating a new event. Action: click(start_box='(880,831)')

Action: click(start_box='(606,381)')

| |
|---|

4 5 6

[Figure 104]

Thought: To proceed with arranging the event, I need to type "homework" into the active text field. This step is essential to label the event appropriately and ensure it aligns with the task requirements. Type "homework" into the active "Event..." text field. Action: type(content='homework')

Thought: To proceed with arranging the event, I need to select the "New event" option from the pop-up menu. This will open the event creation interface, where I can input the event title, set the date, and configure the notification time as required. Tap on the "New event" option

[Figure 105]

[Figure 106]

Thought: To proceed with arranging the event, I need to tap on the "Event..." text field to activate it and input the title "homework." This is the logical next step to ensure the

| |
|---|

event is properly labeled and

aligns with the task requirements. Tap on the "Event..." text field to begin entering the event title. Action: click(start_box='(263,145)')

in the pop-up menu to begin

| |
|---|

creating a new event. Action: click(start_box='(892,690)')

7 8 9

[Figure 107]

[Figure 108]

[Figure 109]

Thought:

To proceed with arranging the event, I need to tap on the

- "9:00 AM" time field to open the time selection interface. This step is necessary to adjust the notification time to 10 minutes before the event, as specified in the task. Tap on the "9:00 AM" time field to open the time selection interface. Action: click(start_box='(688,238)')

Thought: To proceed with arranging the event, I need to adjust the hour to "10" on the clock interface. This step is necessary to set the correct hour for the event, as specified in the task, and move closer to completing the time configuration. Tap on the "10" position on the clock interface to set the hour to 10. Action: click(start_box='(288,506)')

Thought: To proceed with arranging the event, I need to adjust the minute to "55" on the clock interface. This step is necessary to set the correct time for the event, ensuring it aligns with the task requirement of setting the notification time to 10 minutes before the event. Tap on the "55" position on the clock interface to set the minute to 55. Action: click(start_box='(384,466)')

Thought: To proceed with arranging the event, I need to confirm the

selected time of "10:55 AM" by tapping the "OK" button. This

step is essential to finalize the time configuration and move forward in completing the event setup. Tap on the "OK" button to confirm the selected time of

- "10:55 AM.“ Action: click(start_box='(804,762)')

| |
|---|

| |
|---|

| |
|---|

10 11 Thought: To proceed with arranging the event, I need to select the "10 minutes before" option in the notification settings. This step is essential to ensure the notification is set to the correct time, as specified in the task, and aligns with the requirement to set the notification time to 10 minutes before the event. Tap on the "10 minutes before" option to set the notification time to 10 minutes before the event. Action: click(start_box='(162,485)')

12

[Figure 110]

Thought: To proceed with arranging the event, I need to tap on the "At time of event" option to open the notification settings. This step is necessary to adjust the notification time to 10 minutes before the event, as specified in the task, and ensure the event is properly configured. Tap on the "At time of event" option to open the

[Figure 111]

[Figure 112]

| |
|---|

| |
|---|

| |
|---|

notification settings.

Action: click(start_box='(428,307)')

13 14

Thought: To complete the task of arranging the event, I need to tap the "SAVE" button to confirm and save all the configured details, including the title, date, time, and notification settings. This step is essential to ensure the event is successfully created and

Thought: The task of arranging the event titled "homework" for May 21st with a notification time of 10 minutes before has been successfully completed. The event is now visible in the calendar, and all the required details have been configured correctly. I can now submit the task as completed.

[Figure 113]

[Figure 114]

aligned with the task requirements.

Submit the task, confirming that it has been successfully completed.

Tap on the "SAVE" button to finalize and save the event details. Action: click(start_box='(812,926)')

Action: finished()

| |
|---|

- Figure 13: The trajectory of our trained model when executing a Calendar task. The instruction is “You should use calendar to complete the following task: Arrange an event titled ‘homework’ for me at May 21st, and set the notification time to be 10 minutes before.”

## References

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Anthropic. Introducing claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet,

2024. Accessed: 2025-05-11.

- [3] H. Bai, Y. Zhou, J. Pan, M. Cemri, A. Suhr, S. Levine, and A. Kumar. Digirl: Training in-the-wild devicecontrol agents with autonomous reinforcement learning. Advances in Neural Information Processing Systems, 37:12461–12495, 2024.
- [4] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [5] R. Bellman. A markovian decision process. Journal of mathematics and mechanics, pages 679–684, 1957.
- [6] R. Bonatti, D. Zhao, F. Bonacci, D. Dupont, S. Abdali, Y. Li, Y. Lu, J. Wagle, K. Koishida, A. Bucker, et al. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264, 2024.
- [7] Y. Chai, S. Huang, Y. Niu, H. Xiao, L. Liu, D. Zhang, P. Gao, S. Ren, and H. Li. Amex: Android multi-annotation expo dataset for mobile gui agents. arXiv preprint arXiv:2407.17490, 2024.
- [8] L. Chen, L. Li, H. Zhao, Y. Song, and Vinci. R1-v: Reinforcing super generalization ability in visionlanguage models with less than $3. https://github.com/Deep-Agent/R1-V, 2025. Accessed: 202502-02.
- [9] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. D. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [10] W. Chen, J. Cui, J. Hu, Y. Qin, J. Fang, Y. Zhao, C. Wang, J. Liu, G. Chen, Y. Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317, 2024.
- [11] K. Cheng, Q. Sun, Y. Chu, F. Xu, Y. Li, J. Zhang, and Z. Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024.
- [12] H. Deng, D. Zou, R. Ma, H. Luo, Y. Cao, and Y. Kang. Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning. arXiv preprint arXiv:2503.07065, 2025.
- [13] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.
- [14] A. Drouin, M. Gasse, M. Caccia, I. H. Laradji, M. Del Verme, T. Marty, L. Boisvert, M. Thakkar, Q. Cappart, D. Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024.
- [15] D. Gao, L. Ji, Z. Bai, M. Ouyang, P. Li, D. Mao, Q. Wu, W. Zhang, P. Wang, X. Guo, et al. Assistgui: Task-oriented pc graphical user interface automation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13289–13298, 2024.
- [16] Google. Gemini 1.5 pro | generative ai on vertex ai. https://cloud.google.com/vertex-ai/ generative-ai/docs/models/gemini/1-5-pro, 2024. Accessed: 2025-05-11.
- [17] B. Gou, R. Wang, B. Zheng, Y. Xie, C. Chang, Y. Shu, H. Sun, and Y. Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024.
- [18] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [19] H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.
- [20] W. Hong, W. Wang, Q. Lv, J. Xu, W. Yu, J. Ji, Y. Wang, Z. Wang, Y. Dong, M. Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024.

- [21] Y.-C. Hsiao, F. Zubach, G. Baechler, V. Carbune, J. Lin, M. Wang, S. Sunkara, Y. Zhu, and J. Chen. Screenqa: Large-scale question-answer pairs over mobile app screenshots. arXiv preprint arXiv:2209.08199, 2022.
- [22] W. Huang, B. Jia, Z. Zhai, S. Cao, Z. Ye, F. Zhao, Z. Xu, Y. Hu, and S. Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.
- [23] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [24] R. Kapoor, Y. P. Butala, M. Russak, J. Y. Koh, K. Kamble, W. AlShikh, and R. Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In European Conference on Computer Vision, pages 161–178. Springer, 2024.
- [25] J. Y. Koh, R. Lo, L. Jang, V. Duvvur, M. C. Lim, P.-Y. Huang, G. Neubig, S. Zhou, R. Salakhutdinov, and D. Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649, 2024.
- [26] H. Li, J. Chen, J. Su, Y. Chen, Q. Li, and Z. Zhang. Autogui: Scaling gui grounding with automatic functionality annotations from llms. arXiv preprint arXiv:2502.01977, 2025.
- [27] W. Li, W. E. Bishop, A. Li, C. Rawles, F. Campbell-Ajala, D. Tyamagundlu, and O. Riva. On the effects of data scale on ui control agents. Advances in Neural Information Processing Systems, 37:92130–92154, 2024.
- [28] Y. Li, J. He, X. Zhou, Y. Zhang, and J. Baldridge. Mapping natural language instructions to mobile ui action sequences. arXiv preprint arXiv:2005.03776, 2020.
- [29] K. Q. Lin, L. Li, D. Gao, Z. Yang, S. Wu, Z. Bai, W. Lei, L. Wang, and M. Z. Shou. Showui: One vision-language-action model for gui visual agent. arXiv preprint arXiv:2411.17465, 2024.
- [30] E. Z. Liu, K. Guu, P. Pasupat, T. Shi, and P. Liang. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802, 2018.
- [31] J. Liu, Y. Song, B. Y. Lin, W. Lam, G. Neubig, Y. Li, and X. Yue. Visualwebbench: How far have multimodal llms evolved in web page understanding and grounding? arXiv preprint arXiv:2404.05955, 2024.
- [32] X. Liu, B. Qin, D. Liang, G. Dong, H. Lai, H. Zhang, H. Zhao, I. L. Iong, J. Sun, J. Wang, et al. Autoglm: Autonomous foundation agents for guis. arXiv preprint arXiv:2411.00820, 2024.
- [33] X. Liu, T. Zhang, Y. Gu, I. L. Iong, Y. Xu, X. Song, S. Zhang, H. Lai, X. Liu, H. Zhao, et al. Visualagentbench: Towards large multimodal models as visual foundation agents. arXiv preprint arXiv:2408.06327, 2024.
- [34] Y. Liu. Rethinking kl divergence in rlhf: From single sample to mini-batch to expectation. https://www.notion.so/Rethinking-KL-Divergence-in-RLHF-From-Single-Sample-to-Mini-Batch-toExpectation-1c18637cdeb3800ab47cd01d3fa33ea5, 2025. Notion Blog.
- [35] Y. Liu, P. Li, C. Xie, X. Hu, X. Han, S. Zhang, H. Yang, and F. Wu. Infigui-r1: Advancing multimodal gui agents from reactive actors to deliberative reasoners. arXiv preprint arXiv:2504.14239, 2025.
- [36] Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.
- [37] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.
- [38] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [39] Q. Lu, W. Shao, Z. Liu, F. Meng, B. Li, B. Chen, S. Huang, K. Zhang, Y. Qiao, and P. Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. arXiv preprint arXiv:2406.08451, 2024.
- [40] X. H. Lù, Z. Kasner, and S. Reddy. Weblinx: Real-world website navigation with multi-turn dialogue. arXiv preprint arXiv:2402.05930, 2024.
- [41] Z. Lu, Y. Chai, Y. Guo, X. Yin, L. Liu, H. Wang, G. Xiong, and H. Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025.

- [42] L. McInnes, J. Healy, and J. Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.
- [43] OpenAI. Introducing operator. https://openai.com/index/introducing-operator/, 2025. Accessed: 2025-05-16.
- [44] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
- [45] Y. Pan, D. Kong, S. Zhou, C. Cui, Y. Leng, B. Jiang, H. Liu, Y. Shang, S. Zhou, T. Wu, et al. Webcanvas: Benchmarking web agents in online environments. arXiv preprint arXiv:2406.12373, 2024.
- [46] Y. Peng, G. Zhang, M. Zhang, Z. You, J. Liu, Q. Zhu, K. Yang, X. Xu, X. Geng, and X. Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.
- [47] Z. Qi, X. Liu, I. L. Iong, H. Lai, X. Sun, W. Zhao, Y. Yang, X. Yang, J. Sun, S. Yao, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. arXiv preprint arXiv:2411.02337, 2024.
- [48] Y. Qin, Y. Ye, J. Fang, H. Wang, S. Liang, S. Tian, J. Zhang, J. Li, Y. Li, S. Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.
- [49] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36: 53728–53741, 2023.
- [50] C. Rawles, A. Li, D. Rodriguez, O. Riva, and T. Lillicrap. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems, 36:59708–59728, 2023.
- [51] C. Rawles, S. Clinckemaillie, Y. Chang, J. Waltz, G. Lau, M. Fair, A. Li, W. Bishop, W. Li, F. CampbellAjala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.
- [52] N. Reimers and I. Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.
- [53] J. Schulman. Approximating kl divergence. http://joschu.net/blog/kl-approx.html, 2020. Accessed: 2025-05-16.
- [54] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [55] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [56] H. Shen, P. Liu, J. Li, C. Fang, Y. Ma, J. Liao, Q. Shen, Z. Zhang, K. Zhao, Q. Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.
- [57] T. Shi, A. Karpathy, L. Fan, J. Hernandez, and P. Liang. World of bits: An open-domain platform for web-based agents. In International Conference on Machine Learning, pages 3135–3144. PMLR, 2017.
- [58] L. Sun, X. Chen, L. Chen, T. Dai, Z. Zhu, and K. Yu. Meta-gui: Towards multi-modal conversational agents on mobile gui. arXiv preprint arXiv:2205.11029, 2022.
- [59] Q. Sun, K. Cheng, Z. Ding, C. Jin, Y. Wang, F. Xu, Z. Wu, C. Jia, L. Chen, Z. Liu, et al. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. arXiv preprint arXiv:2412.19723, 2024.
- [60] K. Team, A. Du, B. Gao, B. Xing, C. Jiang, C. Chen, C. Li, C. Xiao, C. Du, C. Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [61] J. Wan, S. Song, W. Yu, Y. Liu, W. Cheng, F. Huang, X. Bai, C. Yao, and Z. Yang. Omniparser: A unified framework for text spotting key information extraction and table recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15641–15653, 2024.
- [62] J. Wang, H. Xu, J. Ye, M. Yan, W. Shen, J. Zhang, F. Huang, and J. Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024.

- [63] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu, C. Zhou, J. Zhou, and J. Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [64] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, S. XiXuan, et al. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37: 121475–121499, 2024.
- [65] H. Wen, Y. Li, G. Liu, S. Zhao, T. Yu, T. J.-J. Li, S. Jiang, Y. Liu, Y. Zhang, and Y. Liu. Autodroid: Llm-powered task automation in android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking, pages 543–557, 2024.
- [66] Z. Wu, Z. Wu, F. Xu, Y. Wang, Q. Sun, C. Jia, K. Cheng, Z. Ding, L. Chen, P. P. Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024.
- [67] X. Xia and R. Luo. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.
- [68] T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.
- [69] T. Xu, L. Chen, D.-J. Wu, Y. Chen, Z. Zhang, X. Yao, Z. Xie, Y. Chen, S. Liu, B. Qian, et al. Crab: Crossenvironment agent benchmark for multimodal language model agents. arXiv preprint arXiv:2407.01511, 2024.
- [70] Y. Xu, X. Liu, X. Sun, S. Cheng, H. Yu, H. Lai, S. Zhang, D. Zhang, J. Tang, and Y. Dong. Androidlab: Training and systematic benchmarking of android autonomous agents. arXiv preprint arXiv:2410.24024, 2024.
- [71] Y. Xu, Z. Wang, J. Wang, D. Lu, T. Xie, A. Saha, D. Sahoo, T. Yu, and C. Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.
- [72] Y. Yang, Y. Wang, D. Li, Z. Luo, B. Chen, C. Huang, and J. Li. Aria-ui: Visual grounding for gui instructions. arXiv preprint arXiv:2412.16256, 2024.
- [73] Y. Yang, X. He, H. Pan, X. Jiang, Y. Deng, X. Yang, H. Lu, D. Yin, F. Rao, M. Zhu, et al. R1onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.
- [74] S. Yao, H. Chen, J. Yang, and K. Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022.
- [75] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.
- [76] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, T. Fan, G. Liu, L. Liu, X. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [77] W. Yu, Z. Yang, J. Wan, S. Song, J. Tang, W. Cheng, Y. Liu, and X. Bai. Omniparser v2: Structured-pointsof-thought for unified visual text parsing and its generality to multimodal large language models. arXiv preprint arXiv:2502.16161, 2025.
- [78] J. Zhang, J. Wu, Y. Teng, M. Liao, N. Xu, X. Xiao, Z. Wei, and D. Tang. Android in the zoo: Chain-ofaction-thought for gui agents. arXiv preprint arXiv:2403.02713, 2024.
- [79] H. H. Zhao, D. Gao, and M. Z. Shou. Worldgui: Dynamic testing for comprehensive desktop gui automation. arXiv preprint arXiv:2502.08047, 2025.
- [80] B. Zheng, B. Gou, J. Kil, H. Sun, and Y. Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.
- [81] H. Zhou, X. Li, R. Wang, M. Cheng, T. Zhou, and C.-J. Hsieh. R1-zero’s" aha moment" in visual reasoning on a 2b non-sft model. arXiv preprint arXiv:2503.05132, 2025.
- [82] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.
- [83] Y. Zuo, K. Zhang, S. Qu, L. Sheng, X. Zhu, B. Qi, Y. Sun, G. Cui, N. Ding, and B. Zhou. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

