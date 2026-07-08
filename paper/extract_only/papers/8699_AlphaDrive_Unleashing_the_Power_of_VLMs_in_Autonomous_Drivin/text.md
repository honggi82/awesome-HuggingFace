## AlphaDrive: Unleashing the Power of VLMs in Autonomous Driving via Reinforcement Learning and Reasoning

Bo Jiang1,⋄ Shaoyu Chen1,2 Qian Zhang2 Wenyu Liu1 Xinggang Wang1, 1 Huazhong University of Science and Technology 2 Horizon Robotics https://github.com/hustvl/AlphaDrive

# arXiv:2503.07608v1[cs.CV]10Mar2025

### Abstract

||75|Supervised Fine-tuning|77.12<br><br>Achieving the best performance as training data increases.|
|---|---|---|
| |SFT + RL<br><br>Planning-oriented RL|72.41 70.83|
<br><br>65.40<br><br>59.33<br><br>53.02<br><br>55.64<br><br>45.46 41.12<br><br>35.31% improvement compared to SFT with limited data<br><br>25<br><br>35<br><br>45<br><br>55<br><br>65|
|---|

OpenAI o1 and DeepSeek R1 achieve or even surpass human expert-level performance in complex domains like mathematics and science, with reinforcement learning (RL) and reasoning playing a crucial role. In autonomous driving, recent end-to-end models have greatly improved planning performance but still struggle with long-tailed problems due to limited common sense and reasoning abilities. Some studies integrate vision-language models (VLMs) into autonomous driving, but they typically rely on pre-trained models with simple supervised fine-tuning (SFT) on driving data, without further exploration of training strategies or optimizations specifically tailored for planning. In this paper, we propose AlphaDrive, a RL and reasoning framework for VLMs in autonomous driving. AlphaDrive introduces four GRPO-based RL rewards tailored for planning and employs a two-stage planning reasoning training strategy that combines SFT with RL. As a result, AlphaDrive significantly improves both planning performance and training efficiency compared to using only SFT or without reasoning. Moreover, we are also excited to discover that, following RL training, AlphaDrive exhibits some emergent multimodal planning capabilities, which is critical for improving driving safety and efficiency. To the best of our knowledge, AlphaDrive is the first to integrate GRPO-based RL with planning reasoning into autonomous driving. Code will be released to facilitate future research.

PlanningAccuracy

20K 50K 110K

Amount of training data

Figure 1. Our planning-oriented RL and two-stage training strategy significantly boost planning accuracy. With just 20k samples, it outperforms SFT by 35.31%, showing strong performance even with limited data. As data increase, AlphaDrive consistently leads in planning performance.

expanding training data and increasing model parameters.

However, due to their black-box nature and lack of common sense, end-to-end models still face significant challenges when handling complex and long-tail driving scenarios. For instance, consider a situation where the vehicle ahead is carrying traffic cones while driving. An end-to-end model may fail to comprehend the relationship between the leading vehicle and the traffic cones, mistakenly assuming that the road ahead is under construction and thus impassable, leading to an incorrect decision to brake. Therefore, relying solely on end-to-end models to achieve high-level autonomous driving remains challenging.

### 1. Introduction

Autonomous driving has witnessed rapid advances in recent years, with end-to-end autonomous driving emerging as one of the most representative models [8, 16, 17, 22, 29]. They take sensor data as input and leverage learnable neural networks to plan the vehicle’s future trajectory. Benefiting from large-scale driving demonstrations, end-to-end models continuously improving their planning capabilities by

With the success of GPT [6], large language models (LLMs) show remarkable comprehension and reasoning abilities [38, 48]. Furthermore, their capabilities have evolved from unimodal text understanding to multimodal vision-language processing. [3, 10, 24]. The commonsense and reasoning abilities of VLMs hold great potential to mit-

⋄ Intern of Horizon Robotics. Corresponding author.

igate the limitations of end-to-end models.

Recently, OpenAI o1 [25], which incorporates reasoning techniques, achieves performance comparable to or even surpassing that of human experts in fields such as programming. Additionally, DeepSeek R1 [14], which leverages reinforcement learning, not only demonstrates “emergent abilities”and achieves top-tier performance but also requires significantly lower training costs compared to other models. These advances underscore the immense potential of reasoning techniques and RL in the development of large models.

Existing research on applying VLMs to autonomous driving can be broadly categorized into two directions. The first focuses on leveraging VLMs for the understanding of driving scenes [34, 49]. The second explores the use of VLMs for planning, where some studies treat VLMs as end-to-end systems that process driving images and other inputs to directly predict trajectories [7, 47]. However, unlike end-to-end models which are specifically designed for trajectory planning, VLMs operate in a language space and are not inherently suited for precise numerical predictions [12, 15]. Consequently, directly employing VLMs for trajectory planning may result in suboptimal performance and even pose safety risks.

Some studies leverage VLMs for high-level planning by formulating the ego vehicle’s future actions in natural language, such as “slow down and turn right” [18]. Although this approach circumvents the aforementioned drawbacks, existing works still lack further exploration of training methodologies. Most of them primarily rely on SFT, overlooking the impact of different training strategies on planning performance and the associated training costs.

In this paper, we explore the following question: How can RL and reasoning — which achieves remarkable success in general large models — be applied to autonomous driving, particularly in planning, to enhance the performance of VLMs in autonomous driving while reducing training costs?

Through preliminary experiments, we find that directly applying existing RL and reasoning techniques to planning results in suboptimal performance. We attribute this to three main factors. First, the reward design in RL for general tasks is not well-suited for planning. For example, in visual object counting, the reward can be simply determined based on whether the model predicts the correct answer. However, in autonomous driving, while high-level planning can be formulated as a multi-class classification problem, the varying significance of different driving behaviors makes it inappropriate to assign equal weights to all actions.

Second, unlike mathematical or counting, the solution of planning are usually not unique. For instance, on an open, straight road, one may choose to maintain a constant speed or accelerate, both of which are valid decisions. Therefore,

rigidly assessing whether the model’s planning output exactly matches the ground truth in the training data may not be the optimal approach.

Finally, while domains such as mathematics have abundant reasoning data, including textbooks and solution manuals that can be easily utilized, autonomous driving lacks readily available datasets that capture the reasoning process. Collecting such data is highly costly and requires extensive manual annotation. As a result, directly applying existing reasoning techniques to planning remains challenging.

To address the aforementioned challenges, this paper introduces AlphaDrive, a VLM-based reinforcement learning and reasoning framework specifically designed for autonomous driving planning. In particular, AlphaDrive employs a RL strategy based on Group Relative Policy Optimization (GRPO) [33]. Compared to Proximal Policy Optimization (PPO) [32] and Direct Preference Optimization (DPO) [31], GRPO exhibits better training stability and performance. Furthermore, the group relative optimization strategy in GRPO is well-suited for planning, as planning often involves multiple valid solutions, making relative optimization across multiple solutions a natural fit. Our experiments show that AlphaDrive exhibits some emergent multimodal planning capabilities, which we think can be attributed to the use of GRPO.

AlphaDrive introduces four GRPO rewards tailored for planning. The first is the planning accuracy reward, which evaluates the consistency between the model’s planning actions and the ground truth actions. The second is the actionweighted reward, which assigns different weights to various actions based on their importance to safety. For instance, actions such as braking and steering are critical for safety, so weighting them accordingly helps the model achieve better performance in planning key actions. The third is the planning diversity reward, which encourages the model to generate multiple diverse solutions. This prevents mode collapse and enhances overall planning performance. The last one is the planning format reward, where we define a specific output format and encourage the model to follow it. This ensures more structured outputs and contributes to more stable training.

In addition to RL, we propose a planning reasoning technique. Our approach employs a two-stage training strategy based on knowledge distillation, integrating SFT and RL. In the first stage, we leverage a large model, such as GPT-4o, to generate a small yet high-quality dataset containing planning reasoning processes derived from real driving actions. This dataset is then used to fine-tune our model via SFT, effectively distilling knowledge from the large model. In the second stage, we further refine the model using RL. Introducing the SFT stage as a warm-up step effectively mitigates hallucinations and instability commonly observed in the early stages of reinforcement learning, while also en-

hancing planning performance. Our contributions are summarized as follows:

- • We propose AlphaDrive, a VLM tailored for high-level planning in autonomous driving. To the best of our knowledge, AlphaDrive is the first to integrate GRPObased RL with planning reasoning to autonomous driving, significantly boosting both performance and training efficiency.
- • AlphaDrive introduces four GRPO rewards for planning: planning accuracy reward, action-weighted reward, planning diversity reward, and planning format reward. These optimized rewards make GRPO more suitable for autonomous driving.
- • We propose a two-stage reasoning training strategy based on knowledge distillation, integrating SFT and RL. Our approach achieves better planning performance compared to training with RL alone or without reasoning.
- • Experiments on a large-scale driving dataset validate the superiority of AlphaDrive. Compared to the SFT-trained model, AlphaDrive significantly improves the planning accuracy by 25.52% and, with only 20% of the training data, outperforms the SFT-trained model by 35.31%. We are also excited to discover that, following RL training, AlphaDrive exhibits some emergent multimodal planning capabilities, which is promising for improving driving safety and efficiency.

### 2. Related Work

Vision Language Models. Since the release of GPT [1, 6], the capabilities of large models have gradually expanded from single modality to multi-modalities. Large vision language models [1, 3, 24, 38] now demonstrate superior abilities in visual understanding and reasoning. Early works attempt to integrate visual models with large language models (LLMs), Flamingo [2] uses a visual encoder to process visual signals and adds attention layers in the LLM decoder to interact with the visual features. BLIP [20, 21] introduces the Q-Former architecture and cross-modal contrastive learning tasks to bridge the vision encoder with LLMs. LLaVA [23, 24] propose using vanilla MLP as the connector between the visual encoder and LLMs, which achieves impressive visual understanding capabilities with relatively limited data. The QwenVL [3, 41] series continuously improve the visual module, offering better support for high-resolution and dynamic resolution images, while also demonstrate excellent performance in multilingual tasks and spatial perception.

Reinforcement Learning and Reasoning. Autoregressive learning [39] is currently the mainstream pre-training strategy for LLMs. Besides, RL and reasoning techniques further enhance the capabilities of large models [26, 31–

33, 43]. For instance, GPT [1] employs RL with Human Feedback (RLHF) [26], which incorporates human feedback into the training process. By integrating human intentions and behavioral preferences, RLHF enables LLMs to generate outputs that align more closely with human habits and preferences. Direct Preference Optimization (DPO) [31] enhances the model’s performance by directly optimizing preference feedback. Building on this, Group Relative Policy Optimization (GRPO) [33] introduces a strategy of group relative optimization, which considers the relative superiority or inferiority between multiple output groups, further improving the stability and effectiveness of the training process.

The recent DeepSeek R1 [14] experiences an “Aha Moment”during training based on GRPO, where, without any explicit guidance, the model autonomously allocates more thinking to the problem and re-evaluates its initial approach. This highlights the potential of RL in enabling large models to evolve from mere imitation to emergent intelligence. In our experients, we are also excited to discover that, after GRPO-based RL training, AlphaDrive demonstrates some emergent multimodal planning capabilities, enabling it to generate multiple reasonable driving plans. We believe it has great potential to improve driving safety and efficiency.

In terms of reasoning, Chain-of-thought [43] has demonstrated great performance in solving complex problems by breaking them down and reasoning step by step. OpenAI o1 [25], which is based on Chain-of-thought, introduces inference-time scaling. By increasing the computational cost during inference and combining search strategies such as Monte Carlo Tree Search (MCTS) [35] and Beam Search [46], significant improvements have been achieved in areas such as science and programming that require complex reasoning. This also shows that, beyond scaling model parameters and training data, scaling the inference-time computation is also a promising direction for exploration.

Autonomous Driving Planning. Planning is the ultimate task of autonomous driving. The earliest planning algorithms are rule-based [27, 36], which have significant limitations in terms of generalizability and efficiency. Recently, end-to-end models [8, 13, 16, 17, 22, 29] has gained popularity, where a unified neural network is used to directly output planning trajectories or control signals from sensor data. By leveraging large-scale driving demonstrations, end-toend models are trained in a data-driven manner, achieving impressive planning performance. However, since end-toend models are black-box models that lack common-sense and reasoning capabilities, they still struggle to address the long-tailed problems in autonomous driving.

VLMs and Autonomous Driving. The common-sense and reasoning abilities of large models can effectively compensate for the limitations of end-to-end models in autonomous

[Figure 1]

[Figure 2]

###### Two-stage Training Paradigm

Ground Truth Action: accelerate, left_turn

###### User

[Figure 3]

Your current speed is 2m/s, the navigation command is ‘go straight’. What is your driving plan for the next three seconds? Output the planning reasoning process in <think> </think> and final planning answer in <answer> </answer> tags, respectively.

[Figure 4]

- Answer1: <think> I decide to move slowly because the light is green and the people are about to cross the road </think> <answer> keep, straight </answer>.
- Answer2: <think> The light is green, so I will start through the intersection and keep to the left to avoid pedestrians. </think> <answer> accelerate, left_turn </answer>

Reasoning Distillation

AlphaDrive

- Stage 1 Supervised Fine-tuning Warm-Up

[Figure 5]

Large

Models

Reason

I chose to slow down because there is a

construction area on

both sides, and two workers are on the roadside to my right.

[Figure 6]

- Stage 2 Reinforcement Learning Exploration

num_answers = 2

[Figure 7]

I see there is a

###### Planning-oriented Reinforcement Learning based on GRPO

###### pedestrian waiting

to cross the road,

- Reward for Answer1: 0
- Reward for Answer2: 1

- Reward for Answer1: 0.8
- Reward for Answer2: 1

- Reward for Answer1: 1
- Reward for Answer2: 1

- Reward for Answer1: 1
- Reward for Answer2: 1

and the light is red, so I should stop and wait.

[Figure 8]

###### GT: STOP GRPO

Planning Accuracy Action-weighted Planning Diversity Planning Format

- Figure 2. Overall training framework of AlphaDrive. AlphaDrive is trained using GRPO-based RL, and we design four planning rewards to help the model understand and learn planning. Besides, we propose a two-stage training paradigm, the first stage uses SFT to distill the planning reasoning process from a large model and serves as a warm-up, while the second stage employs RL to explore planning.

driving. In the field of robotics, Vision-Language-Action (VLA) models [5, 19, 44] have made significant progress in understanding language instructions and executing complex actions. A common approach is to use VLMs as the planning module to generate planning instructions, which are then translated into control signals through an action model. There have also been some works based on large models in the field of autonomous driving. DriveGPT4 [47] utilizes a VLM that takes front-view videos as input, and the model directly predicts control signals. ELM [49] leverages largescale, cross-domain video training for VLMs, showing that using data from various domains can effectively enhance the performance of VLMs in driving-related tasks. OmniDrive [42] proposes the use of sparse 3D tokens to represent driving scenes, which are then input into VLMs for scene understanding and planning.

In addition to the above works that directly apply VLMs to driving, DriveVLM [37] combines VLMs with end-toend models for the first time, where VLMs predict lowfrequency trajectories and an end-to-end model generates high-frequency trajectories. Senna [18] proposes a framework where VLMs handle high-level planning, while endto-end models are responsible for low-level trajectory prediction. Additionally, several datasets and benchmarks have been proposed [30, 34, 34, 45], which promote the application of VLMs in autonomous driving. However, most of the current works on VLMs in the field of autonomous driving involves directly using pre-trained models and then utilizing SFT on driving data, which lacks in-depth exploration on training strategies specifically designed for planning. Further effort is needed to adapt the impressive RL and reasoning techniques from general tasks to autonomous driving.

### 3. AlphaDrive

#### 3.1. Overview

AlphaDrive is a VLM designed for autonomous driving planning. Unlike previous approaches that rely solely on SFT, we explore the incorporation of RL and reasoning techniques to better align with the unique characteristics of driving planning: (1) the varying importance of different driving behaviors; (2) the existence of multiple feasible solutions; and (3) the scarcity of readily available reasoning data for planning decisions.

We propose four GRPO-based RL rewards tailored for planning, along with a two-stage planning-reasoning training strategy that integrates SFT with RL. Our experiments demonstrate that, compared to using SFT alone or training without reasoning, AlphaDrive achieves significant improvements in both planning performance and training efficiency. In the following sections, we will detail the design of each component.

#### 3.2. Planning-oriented Reinforcement Learning 3.2.1. Reinforcement Learning Algorithm

Current commonly used RL algorithms include PPO [32], DPO [31], and GRPO [33]. Given a query q, GRPO samples a group of outputs {o1,o2,··· ,oG} from the old policy πθ

and optimizes the new policy πθ by maximizing:

old

G

1 G

Li − βDKL(πθ||πref) ,

JGRPO(θ) = Eq,{o

i}∼πθold

i=1

(1) Li = min(wiAi,clip(wi,1 − ϵ,1 + ϵ)Ai), (2)

θ(oi|q)

where wi = π

πθold(oi|q), ϵ and β are hyper-parameters, and

the advantage Ai is computed using the normalized reward within the group.

We ultimately choose GRPO as the RL algorithm for AlphaDrive for two key reasons: (1) DeepSeek R1 [14] has demonstrated the effectiveness of GRPO in general domains. Compared to other algorithms, GRPO provides higher training stability and efficiency; (2) Moreover, the group relative optimization strategy introduced by GRPO is particularly well-suited for planning, as planning often involves multiple valid solutions, making relative optimization across multiple solutions is a natural fit. Experimental results further confirm that models trained with GRPO exhibit strong planning capabilities.

##### 3.2.2. Planning Reward Modeling

Planning Accuracy Reward. In fields such as mathematics or programming, the reward in GRPO can be intuitively determined based on whether the final answer is correct. However, planning is more complex, as it involves both lateral (direction) and longitudinal (speed) components. Furthermore, the set of possible actions is constrained. As a result, we use the F1-Score to evaluate the accuracy of both lateral and longitudinal decisions separately, and assign rewards accordingly.

Initially, we evaluate accuracy by checking whether the model’s prediction exactly matches the ground truth. However, due to imperfect format in the model’s early training phase, such as discrepancies in case sensitivity or the presence of extraneous outputs, this approach results in poor stability during the early stages of training. We then attempt to extract all the words from the prediction and check whether the ground truth is included among the words. This introduces a new issue where the model sometimes learns shortcut solutions, such as outputting all possible actions, which causes mode collapse. Ultimately, we adopt the F1-score for evaluation, as it not only prevents the model from learning shortcut solutions (where outputting all decisions could result in high recall but low accuracy) but also improves the stability during the early training phase.

Action-Weighted Reward. As mentioned above, the importance of different behaviors in planning varies. For instance, decelerating and stopping are more critical for safety than maintaining speed. Therefore, we assign different importance weights to various actions, incorporating them as weighted components in the final reward.

Planning Diversity Reward. Since planning is inherently multimodal, during GRPO-based RL training, the model generates multiple solutions for group relative optimization. In the later stages of training, we observe that the model’s outputs tend to converge to the same solution. Our goal is to encourage the model to generate a variety of feasible solutions, rather than merely aligning with the ground truth

Algorithm 1: Planning Reward Modeling.

Input: Planning answers A, Ground Truth action e Output: Planning Reward R

- 1 Initialization: Planning Reward R ← ∅, Action Weights W
- 2 Speed Action Set S, Path Action Set P, Answer Format F
- 3 # Pytorch-like Code
- 4 ans counter = Counter()

- 5 for ans in A do

- 6 action ans = re.search(r“F”, ans).group(1).strip()

- 7 ans counter.update(action ans)

- 8 speed ans = extract ans(action ans, S)

- 9 path ans = extract ans(action ans, P)

- 10 # Calculate Planning Accuracy Reward
- 11 speed acc R = cal f1 score(speed ans, e)

- 12 path acc R = cal f1 score(path ans, e)

- 13 # Calculate Action-Weighted Reward
- 14 speed weighted R = W[speed ans]

- 15 path weighted R = W[path ans]

- 16 # Calculate Planning Diversity Reward
- 17 plan div R = 0

- 18 if sum(ans counter.values()) != 0 then

- 19 plan div R = ans counter[action ans] / sum(ans counter.values())

- 20 end
- 21 # Up to 20% reduction in diversity reward
- 22 plan div R = 1 - min(0.2, plan div R)

- 23 # Calculate Planning Format Reward
- 24 format R = check format(ans, F)

- 25 # Final Planning Quality Reward
- 26 speed R = speed acc R * speed weighted R * plan div R

- 27 path R = path acc R * path weighted R * plan div R

- 28 R.append([speed R, path R, format R])

- 29 end
- 30 Return: R

extrat ans will extract substrings that match the specified pattern from the given string. cal f1 score will calculate F1 score given the predictions and ground truth. check format will check whether the given string matches the provided pattern based on regular expression matching.

actions in the training data. To achieve this, we propose the Planning Diversity Reward. When the model’s outputs differ, we assign a higher reward; otherwise, we reduce the reward.

Planning Format Reward. The last reward is used to regularize the output, making it easier to extract both the reasoning process and the final answer. This approach is inspired by R1. The reasoning process is encapsulated within the <think></think> tags, while the planning result is enclosed within the <answer></answer> tags. If the final output does not conform to this format, the format reward will be set to 0.

The Planning Accuracy Reward, the Action-Weighted Reward, and the Planning Diversity Reward are multiplied to compute the Planning Quality Reward. We calculate the Planning Quality Reward separately for speed planning and direction planning. Finally, the Planning Quality Reward and the Planning Format Reward are used to calculate the GRPO loss and update the model parameters. For details about Planning Reward Modeling, please refer to Alg. 1.

Path (F1) ↑ Speed (F1) ↑ BLEU-4 CIDEr METEOR

Method Size Acc. (%)

straight left right keep acc. dec. stop

InternVL2 [9] 2B 7.23 33.34 9.49 4.57 48.75 2.87 8.12 13.87 8.04 5.40 23.63 Qwen2VL [41] 2B 13.69 34.05 21.46 13.46 52.34 11.14 13.73 17.03 16.41 10.85 27.66 Llama3.2-V [11] 11B 11.61 32.56 27.78 13.67 42.71 12.77 20.81 18.04 23.23 15.87 26.30 Qwen2VL [41] 7B 19.28 45.92 33.09 19.20 54.13 12.86 27.01 23.48 30.30 16.16 33.36

InternVL2† [9] 2B 51.07 76.13 85.16 64.60 74.77 21.88 47.66 15.81 27.89 19.73 28.26 Qwen2VL† [41] 2B 55.84 82.68 80.31 70.04 75.97 34.92 55.55 72.64 24.46 23.14 34.26 Llama3.2-V† [11] 11B 58.21 85.58 84.64 79.12 74.79 35.56 58.99 76.20 32.05 21.25 37.70 Qwen2VL† [41] 7B 61.44 86.45 85.84 87.75 84.53 43.81 56.30 73.80 41.09 30.65 47.47 AlphaDrive 2B 77.12 96.62 89.83 93.25 86.80 56.33 71.40 86.63 43.54 38.97 55.23

Table 1. High-level planning and reasoning evaluation results on the MetaAD dataset. Except for AlphaDrive, which utilizes our proposed training strategy, all other models are trained based on SFT. † denotes fine-tuned on the MetaAD dataset.

Base Plan. Action Plan. Plan.

Path (F1) ↑ Speed (F1) ↑

ID

Acc. (%)

Acc. Acc. Weighted Diversity Format straight left right keep acc. dec. stop

- 1 ✓ 42.36 69.40 64.42 59.02 62.18 23.72 47.48 62.70

- 2 ✓ ✓ 55.71 83.19 77.34 71.65 67.37 34.07 59.87 76.56

- 3 ✓ ✓ 67.91 91.95 82.65 88.01 77.74 49.79 61.38 85.75

- 4 ✓ ✓ ✓ 72.20 95.93 85.39 88.80 82.54 52.64 67.60 86.76

- 5 ✓ ✓ ✓ 69.38 92.10 80.48 85.59 84.53 49.40 64.07 83.34

- 6 ✓ ✓ ✓ ✓ 77.12 96.62 89.83 93.25 86.80 56.33 71.40 86.63

Table 2. Ablations on the effectiveness of our proposed planning GRPO rewards.

#### 3.3. Reasoning: Distillation from Large Models

Unlike fields such as mathematics or science, which have abundant high-quality reasoning data available for training, the planning process in autonomous driving is difficult to record, and the cost of manual annotation is high. As a result, there is currently no large-scale, readily available planning reasoning dataset. We initially attempt to incorporate reasoning steps directly into the RL training process, but the final results are suboptimal, mainly due to the following shortcomings: (1) insufficient perception of key elements, such as traffic lights; (2) disorganized reasoning process with weak causal relationships; (3) reasoning outputs that are overly lengthy and ineffective.

Therefore, we adopt a more capable cloud-based large model, such as GPT-4o, to generate high-quality planning reasoning data from a small set of driving clips. Specifically, we provide the model with prompts that include the real driving actions in a given scenario, along with the vehicle’s current state and navigation information, prompting the model to generate a concise decision-making process. We find that the quality of the generated reasoning process is pretty good. After conducting a manual quality check and filtering out samples with obvious errors, we obtain a batch of high-quality planning reasoning data. Subsequently, our model can improve its planning reasoning ability through knowledge distillation based on this data.

#### 3.4. Training: SFT Warm-Up, RL Exploration

RL relies on sparse reward signals, whereas SFT is based on dense supervision, making it more suitable for knowledge distillation. Additionally, we find that relying solely on RL can lead to instability in the early stages of training. Therefore, we use a small amount of data for a warmup phase based on SFT, followed by RL training with the full dataset. We discover that this approach improves stability in the early stages of training and enhances the model’s planning reasoning performance, ultimately leading to better overall planning capabilities.

### 4. Experiments 4.1. Experimental Settings

Dataset. We adopt MetaAD, a large-scale real-world driving dataset, as our training and evaluation benchmark. This dataset consists of a total of 120k driving clips, each lasting three seconds. MetaAD is a high-quality dataset specifically designed for planning, supporting multi-sensor data and perception annotations. Furthermore, it maintains a well-balanced distribution across various driving environments and planning actions. The dataset is divided into 110k clips for training and 10k clips for validation. As for reasoning, we sample 30k data from the training dataset to

Path (F1) ↑ Speed (F1) ↑ BLEU-4 CIDEr METEOR Reason. Strategy straight left right keep acc. dec. stop

With Train.

Acc. (%)

- ✗ SFT 56.97 77.76 63.69 65.07 76.22 37.11 51.99 75.72 - - -

- ✗ RL 62.16 82.32 72.39 71.24 75.03 41.13 61.08 79.15 - - -

- ✗ SFT+RL 70.73 88.04 75.75 78.79 78.60 45.00 65.92 83.52 - - -

✓ SFT 65.40 92.52 71.28 68.65 81.91 36.48 59.31 71.55 37.21 34.30 47.54 ✓ RL 72.41 93.16 84.24 89.32 87.58 51.19 64.70 84.07 25.14 24.58 38.10 ✓ SFT+RL 77.12 96.62 89.83 93.25 86.80 56.33 71.40 86.63 43.54 38.97 55.23

Table 3. Ablations on different reasoning training strategies.

Train. Train.

Path (F1) ↑ Speed (F1) ↑ BLEU-4 CIDEr METEOR Data Strategy straight left right keep acc. dec. stop

Acc. (%)

20k SFT 41.12 56.15 36.72 35.59 40.63 17.14 16.74 19.19 27.18 15.42 31.17 20k RL 45.46 69.28 59.42 51.91 56.93 30.82 37.71 30.94 20.33 11.01 23.09 20k SFT+RL 55.64 68.25 64.06 56.87 58.61 45.19 53.68 44.09 32.84 17.02 35.93

50k SFT 53.02 73.74 62.45 65.43 70.07 33.83 38.94 53.96 34.48 26.83 42.85 50k RL 59.33 77.69 68.55 73.82 77.05 40.72 45.20 57.06 22.37 16.81 25.81 50k SFT+RL 70.83 82.30 78.05 82.17 84.80 47.27 58.29 64.67 32.30 30.38 46.38

110k SFT 65.40 82.52 71.28 68.65 81.91 36.48 59.31 71.55 37.21 34.30 49.54 110k RL 72.41 93.16 84.24 89.32 82.58 51.19 64.70 82.02 25.14 24.58 38.10 110k SFT+RL 77.12 96.62 89.83 93.25 86.80 56.33 71.40 86.63 43.54 38.97 55.23

Table 4. Ablations on the amount of training data.

generate the planning reasoning process. All reported results are obtained by training on the training set and evaluating on the validation set.

Training. We use Qwen2VL-2B [41] as the base model. Qwen2VL is currently one of the best-performing opensource models, and it offers a smaller 2B version that better meets the latency requirements for autonomous driving. Additionally, Qwen2VL provides better support for RL. The model’s inputs include a front-view image and a planning prompt, which contains the vehicle’s current speed and navigation information. The navigation data, consistent with real-world driving, is obtained from sparse navigation points via AMap (similar to Google Maps) and is converted into text form for inclusion in the prompt, such as “Go straight for 100m, then turn right”. Training is conducted using 16 NVIDIA A800 GPUs.

Evaluation. The evaluation metrics consist of two aspects. First, the accuracy of meta-action planning is measured by calculating the F1-Score for all categories of lateral and longitudinal meta-actions, followed by the overall planning accuracy. Additionally, for planning reasoning, we compute the similarity between the generated planning reasoning process and the annotated reasoning process in the dataset using BLEU-4 [28], CIDEr [40], and METEOR [4] scores.

#### 4.2. Main Results

Tab. 1 presents the performance of AlphaDrive in high-level planning. The first four rows show the results obtained by directly evaluating the corresponding pretrained models. It can be observed that, while these models demonstrate stronger general capabilities, their performance in planning is suboptimal, highlighting the need for further training with driving data. The subsequent five rows display the results of models fine-tuned on the MetaAD dataset. As shown, AlphaDrive significantly outperforms the other models. Compared to Qwen2VL-7B, the second-best performing model after AlphaDrive, the planning accuracy significantly improves by 25.5%. There is a noticeable enhancement in key decisions such as steering and acceleration/deceleration. Additionally, the quality of planning reasoning is the best among all models, demonstrating the effectiveness of our proposed two-stage RL training and reasoning strategies.

#### 4.3. Ablation Study

Planning Rewards. In Tab. 2, we validate the effectiveness of the four proposed GRPO planning rewards. The Base Accuracy reward directly determines the reward based on whether the response exactly matches the ground truth, a common approach in general domains. As shown, the model using the Base Accuracy reward lags significantly behind across all metrics (ID 1). The combination with the

[Figure 9]

[Figure 10]

SFT-Trained Model w/o Reasoning

###### SFT-Trained Model w/o Reasoning

<answer> stop, straight </answer>

<answer> stop, straight </answer>

Answer1:

- Answer1: <think> I should stop for pedestrians to pass. </think> <answer> stop, straight </answer>
- Answer2: <think> I plan to turn left slowly and watch out for pedestrians. </think> <answer> keep, left_turn </answer>

<think> I need to stop behind the car.

[Figure 11]

[Figure 12]

</think> <answer> stop, straight </answer>

Answer2: <think> I can slow down and change lanes

AlphaDrive

AlphaDrive

left to overtake a slow vehicle. </think>

<answer> decelerate, left_turn </answer>

- Figure 3. Qualitative results of AlphaDirve. After RL training, AlphaDrive exhibits some emergent multimodal planning capabilities, which has great potential for improving driving safety and efficiency.

Planning Format Reward yields a slight improvement. (ID 2). A significant improvement is seen with the adoption of our proposed Planning Accuracy Reward (ID 3). Further enhancement in acceleration/deceleration decisions is achieved by incorporating the Action-Weighted Reward (ID

- 4). Finally, by combining the Planning Diversity Reward, the best planning performance is achieved (ID 5-6).

with RL reaches a planning accuracy of 46.08%, which is significantly higher than that of the SFT-trained model. When using nearly half of the data, with 50k samples, AlphaDrive already achieves a planning accuracy of 70.83%, demonstrating the efficiency of our training strategy.

#### 4.4. Emergence of Multimodal Planning Capability

Fig. 3 illustrates the multimodal planning capability of AlphaDrive after RL training. In complex scenarios, it can effectively generate multiple feasible solutions, whereas the SFT-trained model can only produce a single planning decision. AlphaDrive can be integrated with a downstream action model to dynamically select the optimal solution from multiple options.

Reasoning Training Strategies. The ablation study of the reasoning training strategies is shown in Tab. 3. As observed, introducing planning reasoning under different training strategies effectively enhances model performance. Notably, the improvement is especially significant for complex actions such as acceleration and deceleration, demonstrating that reasoning can greatly enhance decision-making in complex scenarios. Furthermore, the model trained exclusively with RL performs worse in reasoning compared to the model trained with SFT. We attribute this to the limited parameter size of smaller models, which results in insufficient perception and reasoning capabilities. Therefore, incorporating SFT as a warm-up phase and using knowledge distillation to learn the reasoning process from a larger model can effectively address this issue. By combining SFT and RL, the model achieves the best planning reasoning capabilities.

### 5. Conclusions and Limitations

In this work, we propose AlphaDrive, a VLM for high-level planning in autonomous driving. Compared to previous models that solely employed the SFT, we explore the integration of advanced RL and reasoning in planning. Specifically, AlphaDrive introduces a planning-oriented RL strategy based on GRPO and further designs a two-stage planning reasoning training paradigm. To the best of our knowledge, AlphaDrive is the first to introduce the RL and reasoning to autonomous driving planning, significantly boosting both performance and training efficiency.

Amount of Training Data. Tab. 4 shows the impact of training data size on different training strategies. As observed, when the training data size decreases, SFT is more affected. With only 20k training samples, the model trained

Currently, due to a lack of rich data annotation, AlphaDrive is still unable to output more complex driving behaviors such as lane changes or nudges. Additionally, the

current planning reasoning data come from pseudo-labels generated by large models based on ground-truth driving actions, which still suffer from inaccurate perception and a failure to capture key factors. Therefore, further systematic validation is required to improve data quality and verify the performance upper bound of AlphaDrive.

### Acknowledgments

We sincerely thank Hao Gao, Tianheng Cheng, Bencheng Liao, Haoyi Jiang, and Dongli Hu for their valuable feedback on the draft.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 3

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 2022. 3
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 1, 3
- [4] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, 2005. 7
- [5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 4
- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 1, 3
- [7] Long Chen, Oleg Sinavski, Jan H¨unermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, and Jamie Shotton. Driving with llms: Fusing object-level vector modality for explainable autonomous driving. arXiv preprint arXiv:2310.01957, 2023. 2
- [8] Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243,

2024. 1, 3

- [9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of

- open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 6
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR,

2024. 1

- [11] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 6

- [12] Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Petersen, and Julius Berner. Mathematical capabilities of chatgpt. In NeurIPS, 2024. 2
- [13] Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, et al. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. arXiv preprint arXiv:2502.13144, 2025. 3
- [14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 2, 3, 5
- [15] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 2
- [16] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In CVPR, 2023. 1, 3
- [17] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In ICCV, 2023. 1, 3
- [18] Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024. 2, 4
- [19] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 4
- [20] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

- 2022. 3

[21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

- 2023. 3

- [22] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang,

Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. arXiv preprint arXiv:2411.15139, 2024. 1, 3

- [23] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR,

2024. 3

- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2024. 1, 3
- [25] OpenAI. Learning to reason with llms, 2024. 2, 3
- [26] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 2022. 3
- [27] Brian Paden, Michal C´ˇap, Sze Zheng Yong, Dmitry Yershov, and Emilio Frazzoli. A survey of motion planning and control techniques for self-driving urban vehicles. IEEE Transactions on intelligent vehicles, 2016. 3
- [28] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In ACL, 2002. 7
- [29] Aditya Prakash, Kashyap Chitta, and Andreas Geiger. Multimodal fusion transformer for end-to-end autonomous driving. In CVPR, 2021. 1, 3
- [30] Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, and Yu-Gang Jiang. Nuscenes-qa: A multi-modal visual question answering benchmark for autonomous driving scenario. In AAAI, 2024. 4
- [31] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023. 2, 3, 4
- [32] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 2, 4
- [33] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 2, 3, 4
- [34] Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Ping Luo, Andreas Geiger, and Hongyang Li. Drivelm: Driving with graph visual question answering. arXiv preprint arXiv:2312.14150, 2023. 2, 4
- [35] Maciej Swiechowski,´ Konrad Godlewski, Bartosz Sawicki, and Jacek Ma´ndziuk. Monte carlo tree search: A review of recent modifications and applications. Artificial Intelligence Review, 56(3):2497–2562, 2023. 3
- [36] Sebastian Thrun, Mike Montemerlo, Hendrik Dahlkamp, David Stavens, Andrei Aron, James Diebel, Philip Fong, John Gale, Morgan Halpenny, Gabriel Hoffmann, et al. Stanley: The robot that won the darpa grand challenge. Journal of field Robotics, 2006. 3
- [37] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Chenxu Hu, Yang Wang, Kun Zhan, Peng Jia, Xianpeng Lang, and

- Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024. 4
- [38] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 3
- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 3
- [40] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In CVPR, 2015. 7
- [41] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 3, 6, 7
- [42] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. arXiv preprint arXiv:2405.01533, 2024. 4
- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 3
- [44] Junjie Wen, Minjie Zhu, Yichen Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Chengmeng Li, Xiaoyu Liu, Yaxin Peng, Chaomin Shen, et al. Diffusion-vla: Scaling robot foundation models via unified diffusion and autoregression. arXiv preprint arXiv:2412.03293, 2024. 4
- [45] Dongming Wu, Wencheng Han, Tiancai Wang, Yingfei Liu, Xiangyu Zhang, and Jianbing Shen. Language prompt for autonomous driving. arXiv preprint arXiv:2309.04379,

2023. 4

- [46] Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan, Junxian He, and Michael Xie. Self-evaluation guided beam search for reasoning. Advances in Neural Information Processing Systems, 36:41618–41650, 2023. 3
- [47] Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kenneth KY Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. arXiv preprint arXiv:2310.01412,

2023. 2, 4

- [48] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024. 1
- [49] Yunsong Zhou, Linyan Huang, Qingwen Bu, Jia Zeng, Tianyu Li, Hang Qiu, Hongzi Zhu, Minyi Guo, Yu Qiao, and Hongyang Li. Embodied understanding of driving scenarios. arXiv preprint arXiv:2403.04593, 2024. 2, 4

