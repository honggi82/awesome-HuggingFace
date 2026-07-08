# arXiv:2508.10751v1[cs.LG]14Aug2025

[Figure 1]

## Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models

### Zhipeng Chen1,2,∗, Xiaobo Qin2, Youbin Wu2, Yue Ling2, Qinghao Ye2, Wayne Xin Zhao1,2,†, Guang Shi2,†

1Renmin University of China, 2ByteDance Seed ∗Work done at ByteDance Seed, †Corresponding authors

### Abstract

Reinforcement learning with verifiable rewards (RLVR), which typically adopts Pass@1 as the reward, has faced the issues in balancing exploration and exploitation, causing policies to prefer conservative actions, converging to a local optimum. Identifying an appropriate reward metric is therefore crucial. Regarding the prior work, although Pass@k has been used in evaluation, its connection to LLM exploration ability in RLVR remains largely overlooked. To investigate this, we first use Pass@k as the reward to train the policy model (i.e., Pass@k Training), and observe the improvement on its exploration ability. Next, we derive an analytical solution for the advantage of Pass@k Training, leading to an efficient and effective process. Building on this, our analysis reveals that exploration and exploitation are not inherently conflicting objectives, while they can mutually enhance each other. Moreover, Pass@k Training with analytical derivation essentially involves directly designing the advantage function. Inspired by this, we preliminarily explore the advantage design for RLVR, showing promising results and highlighting a potential future direction.

Date: August 15, 2025 Correspondence: Zhipeng Chen at zhipeng_chen@ruc.edu.cn Project Page: https://github.com/RUCAIBox/Passk_Training

[Figure 2]

Pass@1

Pass@K

Pass@1

Training

Training

Training

Claude 3.7 (Pass@1 Acc.)

GPT 4o (Pass@1 Acc.)

Vanilla RLVR (Pass@K Acc.)

Vanilla RLVR (Pass@1 Acc.)

Figure 1 Enigmata scores (Validation Set) of Pass@k Training on Qwen2.5-7B-Ins, which boosts its exploration ability, leading to continuous improvements in following training, surpassing native RLVR and powerful LLMs.

### 1 Introduction

Recently, reinforcement learning with verifiable rewards (RLVR) has emerged to solve complex reasoning tasks and dramatically boost the reasoning capabilities of large language models (LLMs) [26, 44, 58]. During RLVR training, LLMs generate various responses based on the given prompt and receive rewards for responses [17, 39, 43]. LLMs can possess the ability to generate a more comprehensive reasoning process by learning from outcome-level supervision [9, 13], thereby achieving higher performance on downstream tasks. The success of these large reasoning models (LRMs), such as OpenAI o1 [23] and DeepSeek R1 [13], suggests that RLVR training pushes the limits of the capacities of LLMs.

The current RLVR training that typically optimizes the Pass@1 objective, also known as Pass@1 Training, trains LLMs to learn from their exploration and generate the most confident response for the given prompt [21, 50, 54], leading to a major challenge of the balance of exploration and exploitation [10, 20]. Typically, exploration refers to performing novel and various actions [46], while exploitation requires LLMs to invoke reliable actions that the verifier prefers among the known behaviours [40]. During the Pass@1 Training process, LLMs tend to imitate the behaviours that can bring an increase of reward scores in previous attempts, and prevent the behaviours that receive low rewards [8, 33]. However, in outcome-supervision, which is the popular Pass@1 Training setting [13, 50, 54], the erroneous solution with correct answer will receive positive rewards, while the correct solution with wrong answer will be assigned negative rewards [8, 41]. In this case, the unsuccessful explorations, which contains the correct idea, will be associated with the high cost, as it often yields no reward, resulting in an imbalance between exploitation and exploration [12], might leading policy to abandon exploration and converge on a local optimum [4, 27]. Limited by the suboptimal nature of the reward under reinforcement learning methods (e.g., PPO and GRPO) [15, 34, 56], LLMs face the challenge of further learning, restricting the effectiveness and advancement potential of scaling RLVR process.

Based on the above discussion, to mitigate the issue of impaired LLM exploration ability in Pass@1 Training, we advocate for an optimization-centric approach with a higher tolerance for incorrect responses, since they might contain useful ideas or reasoning actions, preventing the model from becoming trapped in a local optimum, thereby extending the upper limit of its capabilities and enabling it to approach a global optimum gradually. Fortunately, with the development of LLM technologies, Pass@k has emerged to assess whether policy can generate correct responses within 𝑘 attempts, which is a common metric for evaluating the boundaries of LLM capabilities [5]. Compared with the Pass@1 metric, the Pass@k metric allows the policy to generate several incorrect responses. Thus, we consider whether the Pass@k metric can be utilized in the RLVR process to push the boundaries of LLM abilities. Unlike Pass@1 metric, in Pass@k evaluation, to maximize the probability that at least one of the 𝑘 samples is successful, a “smart” policy will generate 𝑘 candidate solutions that differ from each other and cover different regions of the solution space, rather than 𝑘 highly similar samples. The stronger exploration ability enables the model to acquire more comprehensive knowledge and stronger robustness.

Building on this idea, we leverage the Pass@k metric as the reward to continually train a model that has already undergone Pass@1 Training (named as Pass@k Training). We find that the model trained on this approach can achieve higher Pass@k scores on the test set, and maintain its Pass@1 scores. Since the naive implementation of Pass@k Training faces several critical issues, we further employ the bootstrap sampling [16, 60] and analytical derivation to optimize the training procedure, achieving effective and efficient Pass@k Training (Section 2). To further understand the feature and inner mechanism of Pass@k Training, we proposed five research questions to investigate how Pass@k Training balances the exploration and exploitation abilities of LLMs during RLVR training, observing the natural prevention of the decrease in the entropy of policy distribution, which is also a critical metric to indicate the exploration ability of policy [10, 12, 20] (Section 3). Furthermore, we consider whether Pass@k Training can provide guidance and inspiration for the future development of RLVR training. From the perspective of implicit reward design, we analyze the key factors contributing to the effectiveness of Pass@k Training and explore several possible avenues for its optimization (Section 4).

Overall, the vital takeaways of our work can be summarized as follows:

• Compared to Pass@1 Training, Pass@k Training significantly enhances the exploration ability of LLMs, improving Pass@k performance while not harming Pass@1 scores. Among its three progressive variants,

- 𝑦ො1: To find the daily … \boxed{2000}

- 𝑦ො2 : If 2% of the daily … \boxed{10}

- 𝑦ො3 : Since 40 calories … \boxed{2000}

- 𝑦ො4 : Let’s analyze this … \boxed{2000}

- 𝑦ො5 : Let the daily … \boxed{99}

- 𝑦ො6 : Let’s break down … \boxed{20}

- 𝑦ො7 : To determine the … \boxed{590}

- 𝑦ො8 : The snack's 40 … \boxed{900}

###### Problem

###### Rollout

If a snack-size tin of peaches has 40 calories and is 2% of a person's daily caloric requirement, how many calories fulfill a

| |
|---|
|𝜋𝜃 ∙ 𝑥)<br><br>|

person's daily caloric requirement?

###### Optimize

|Full Sampling<br><br>𝑦ො1 𝑦ො2 𝑦ො3 𝑦ො4<br><br>𝑦ො5 𝑦ො6 𝑦ො7 𝑦ො8<br><br>Advantage<br><br>Estimation<br><br>A(𝑦ො1) = 1.2<br><br>A(𝑦ො2) = -0.7<br><br>A(𝑦ො3) = 1.2<br><br>A(𝑦ො6) = -0.7<br><br>A(𝑦ො7) = -0.7<br><br><br>A(𝑦ො4) = 1.2 A(𝑦ො8) = -0.7<br><br><br>A(𝑦ො5) = -0.7<br><br>Analytical Derivation<br><br>𝑦ො1 𝑦ො2 𝑦ො3 𝑦ො4<br><br>𝑦ො5 𝑦ො6 𝑦ො7 𝑦ො8<br><br>𝓨෡𝟏<br><br>𝓨෡𝟐<br><br>Advantage<br><br>Estimation<br><br>Full Sampling<br><br>A(𝓨෡𝟏) = 0.5<br><br>A(𝓨෡𝟐) = -0.5<br><br>A(𝑦ො1) = 0.5<br><br>A(𝑦ො4) = 0.5<br><br>… …<br><br>A(𝑦ො6) = -0.5<br><br>A(𝑦ො8) = -0.5<br><br>… …<br><br>𝑦ො1 𝑦ො7 𝑦ො3 𝑦ො8<br><br>𝑦ො8 𝑦ො3 𝑦ො2 𝑦ො4<br><br>𝓨෡𝟏<br><br>𝓨෡𝟐<br><br>𝓨෡𝟖 𝑦ො5 𝑦ො4 𝑦ො8 𝑦ො2<br><br>… …<br><br>Advantage<br><br>Estimation<br><br>Bootstrap Sampling<br><br>A(𝓨෡𝟏) = 0.9<br><br>A(𝓨෡𝟖) = -0.9<br><br>… …<br><br>A(𝑦ො1) = 0.9 + … + 0<br><br>A(𝑦ො2) = 0 + ... - 0.9 … …<br><br><br>A(𝑦ො7) = 0.9 + … + 0<br><br>A(𝑦ො8) = 0.9 + … - 0.9<br><br><br>Pass@1 Training Pass@k Training<br><br>A(𝑦ො2) = 0.5<br><br>A(𝑦ො5) = -0.5<br><br>[Figure 3]<br><br>𝑦ො1 𝑦ො2 𝑦ො3 𝑦ො4<br><br>𝑦ො5 𝑦ො6 𝑦ො7 𝑦ො8<br><br>Advantage<br><br>Estimation<br><br>A(𝑦ො1) = 1.2<br><br>A(𝑦ො2) = -0.7<br><br>A(𝑦ො3) = 1.2<br><br>A(𝑦ො6) = -0.7<br><br>A(𝑦ො7) = -0.7<br><br><br>A(𝑦ො4) = 1.2 A(𝑦ො8) = -0.7<br><br><br>A(𝑦ො5) = -0.7<br><br>[Figure 4]|
|---|

imation

imation

vantage

antage

ation

tage

Figure 2 The overview and comparison between Pass@1 Training and Pass@k Training. The major difference between these training paradigms is in the reward calculation and advantage estimation process. Besides, full sampling, bootstrap sampling, and analytical derivation are three progressive enhancements for the Pass@k Training. To better demonstrate the Pass@k Training pipeline, we present the pseudo code in Appendix C.

bootstrap sampling offers higher training efficiency than full sampling, and analytical derivation serves as its theoretical asymptotic form that mitigates the variance introduced by sampling. (Section 2)

- • Compared to Pass@1 Training and its variants, Pass@k Training is both robust to different values of 𝑘 and generalizable across domains and tasks. Moreover, the enhancement of LLM exploration ability is helpful to improve their exploitation through continual training, leading 7B LLM to surpass the powerful LLMs (e.g., GPT-4o and Claude-3.7), highlighting the practical value of Pass@k Training. (Section 3)
- • Pass@k Training with analytical derivation, which directly designs the advantage function, can be viewed as a form of implicit reward design. Following this idea, empirical experiments suggest that implicit reward design allows finer-grained control over optimization, such as focusing on harder problems or improving training efficiency, without complex theoretical derivations, making it a promising direction for future RLVR development. (Section 4)

### 2 Pass@k as Reward in RLVR Training

In this section, we first formulate the reasoning tasks and provide the review of traditional Pass@1 Training (Section 2.1). Next, we introduce how to implement Pass@k as reward in RLVR training process (Section 2.2), and then propose two progressive enhancements to improve the training efficiency and effectivenss (Section 2 and Section 2.4). To better illustrate Pass@k Training, we present an overview in Figure 2 and the pseudo code in Appendix C, demonstrating the implementation details of Pass@k Training.

#### 2.1 Formulation of Reasoning Tasks and Pass@1 Training

The complex reasoning tasks can assess the reasoning and logical abilities of LLMs. Typically, a problem from the whole dataset 𝐷 contains a description 𝑥 and a ground truth answer 𝑦, and the policy 𝜋𝜃 (i.e., LLM with the parameters 𝜃) needs to generate a response 𝑦ˆ = {𝑡1, 𝑡2, . . . , 𝑡𝑙} based on the 𝑥, where 𝑡𝑖 and 𝑙 refer to the 𝑖-th token and the length of the response 𝑦ˆ. After obtaining the generated response 𝑦ˆ, verifiers are used to verify the correctness of the LLM response and provide a reward 𝑅(𝑦, 𝑦ˆ) ∈ {𝑅neg, 𝑅pos} (𝑅neg < 𝑅pos), where 𝑅neg is for negative responses and 𝑅pos is for positive responses. To simplify the notation, we use 𝑅 to represent 𝑅(𝑦, 𝑦ˆ). In our experiment, we adopt 𝑅neg = 0 and 𝑅pos = 1.

Based on the above formulation of reasoning tasks, in the Pass@1 Training process (e.g., GRPO [39]), the advantage is estimated through the average value and standard deviation of the response rewards within the same group, which can be shown as follows,

𝑁rollout∑︁

1 𝑁rollout

𝑅¯ =

𝑅𝑖, (1)

𝑖=1

𝑁rollout∑︁

1 𝑁rollout

(𝑅𝑖 − 𝑅¯)2, (2)

𝜎 =

𝑖=1

𝑅𝑖 − 𝑅¯ 𝜎

𝐴ˆ𝑖,1 = 𝐴ˆ𝑖,2 = · · · = 𝐴ˆ𝑖,|𝑦ˆ𝑖| =

, (3)

where 𝑁rollout denotes the number of the rolled-out responses for the corresponding question, and 𝑅𝑖 and 𝑦ˆ𝑖 refer to the rewards and the generated response of the 𝑖-th response, respectively. After obtaining the advantage values, GRPO utilizes the following equation to calculate the objective function J(𝜃) that is leveraged to perform gradient descent and optimize the parameters of the model,

∑︁𝐺

∑︁|𝑦ˆ𝑖|

1 𝐺

1 |𝑦ˆ𝑖|

min 𝑟𝑖,𝑡 𝐴ˆ𝑖,𝑡, clip 𝑟𝑖,𝑡, 1 − 𝜀, 1 + 𝜀 𝐴 ˆ𝑖,𝑡 − 𝛽𝐷kL . (4)

J(𝜃) = E(𝑞,𝑎)∼𝐷,{𝑜

𝑖}𝑖𝐺=1∼𝜋𝜃 (·|𝑞)

𝑡=1

𝑖=1

Since each token shares the same advantage value in GRPO, we will no longer distinguish at the token level in the following discussion, and use 𝐴ˆ𝑖 to represent the advantage value of the 𝑖-th response, instead.

To enhance the effectiveness and efficiency of the RLVR training process, we employ a variant of GRPO (i.e., DAPO [52]) in our following experiments, only retaining the clip-higher and token-level policy gradient loss.

#### 2.2 Pass@k Training

As discussed in previous work [42, 45], the behaviour of LLMs can be adjusted by the corresponding rewards. Following this idea, we consider whether the Pass@k metric can be adopted as a reward to push the boundary of LLM abilities, since the Pass@k can reflect LLM exploration ability. Thus, in this part, we first introduce the definition of the Pass@k metric and then incorporate the Pass@k metric into reward function of RLVR.

Definition of Pass@k Metric. Given the question 𝑥, the policy model is utilized to rollout the 𝑘 responses through a specific decoding strategy or searching algorithm (e.g., sampling-based decoding strategy or Monte Carlo Tree Search). The 𝑖-th sampled response 𝑦ˆ𝑖 will receive a reward 𝑅𝑖, which is provided by the verifier. Based on this, the value of the Pass@k metric is defined as the expected maximum reward obtained from the 𝑘 sampled responses. Formally, the Pass@k metric can be computed using the following equation,

𝑖}𝑖𝑘=1∼𝜋𝜃(·|𝑥) [max (𝑅1, . . . , 𝑅𝑘))] . (5)

Pass@k = E(𝑥,𝑦)∼𝐷,{𝑦ˆ

Pass@k Implementation: Full Sampling. To integrate the Pass@k metric into the RLVR process, we propose a basic implementation through the full sampling mechanism. First, we leverage the policy 𝜋𝜃 to rollout the 𝑁rollout responses Yˆ = {𝑦ˆ1, . . . , 𝑦ˆ𝑁

rollout} for the given question. In this situation, these responses are separated

Figure 3 Training progress of Pass@1 Training and Pass@k Training with Full Sampling on baseline setting.

into 𝑁group = ⌊ 𝑁

𝑘 ⌋ groups, and the redundant responses are discarded, where the 𝑗-th group contains the 𝑘 responses Yˆ 𝑗 = {𝑦ˆ𝑘×(𝑗−1)+1, . . . , 𝑦ˆ𝑘×(𝑗−1)+𝑘}. Next, we assign the reward scores to each group based on its Pass@k value. Concretely, the verifier will provide a reward for each response, and the group reward is computed by the maximum among the rewards of the responses belonging to it. Following the advantage estimation approach in the DAPO algorithm, the advantage value of 𝑗-th group 𝐴ˆ𝑗 can be calculated. After that, we divide the group advantage to the responses it contains, i.e., 𝐴ˆ𝑘×(𝑗−1)+1 = · · · = 𝐴ˆ𝑘×(𝑗−1)+𝑘 = 𝐴ˆ𝑗. Finally, we can utilize the sampled responses and their advantage value to optimize the model parameters.

rollout

Empirical Insight: Improving Exploration. To evaluate the effectiveness of employing Pass@k as a reward, we compare the performance between Pass@k Training with full sampling and the vanilla Pass@1 Training, as shown in Figure 3. We observe that during the Pass@1 Training, Pass@k performance on downstream tasks remains stable with slight improvement. As a result, while the Pass@1 metric improves in the early stages of training, it stagnates in the later stages, indicating that the model has fallen into a local optimum. In contrast, employing Pass@k as the reward during the RLVR process, the Pass@k performance of LLM on downstream tasks achieves continual improvement, and more training steps or a larger number of rollouts consistently bring further performance improvements of LLMs, demonstrating that Pass@k Training is scalable. Moreover, Pass@k Training does not compromise the model’s Pass@1 performance and even results in Pass@1 performance gains, suggesting that Pass@k Training and Pass@1 Training share a similar optimization objective and direction, and they can be improved together.

Takeaway from Section 2.2

Compared with the traditional RLVR training method that uses Pass@1 as the reward function, using Pass@k as the reward function for RLVR training can effectively improve the model’s Pass@k performance on downstream tasks without compromising its Pass@1 performance.

#### 2.3 Efficient Pass@k Training via Bootstrap Sampling

Although Pass@k Training can push the limit of LLM abilities, the rollout times increases significantly with the increase of 𝑁group, costing a higher computational resources. Thus, in this part, we consider utilizing the bootstrap sampling mechanism to reduce the rollout times while maintaining a constant number of groups.

To achieve the efficient Pass@k Training, during the rollout process, we first use the policy model 𝜋𝜃 to generate 𝑁rollout responses Yˆ = {𝑦ˆ1, . . . , 𝑦ˆ𝑁

rollout} for the given question 𝑥. Next, to construct the 𝑁group groups for the following optimization process, we randomly sample the 𝑘 responses from the previously generated response set Yˆ, and these sampled responses collectively constitute a group. Specifically, to construct the 𝑗-th group, we select 𝑘 distinct values from the range 1 to 𝑁rollout, obtaining a set P = {𝑝𝑗,1, . . . , 𝑝𝑗,𝑘}. Then the

- Figure 4 Training progress of Pass@1 Training and Pass@k Training with Bootstrap Sampling under various 𝑁rollout.

responses whose indices are in the set P constitute the current group Yˆ 𝑗 = {𝑦ˆ𝑝𝑗,

, . . . , 𝑦ˆ𝑝𝑗,𝑘}. This procedure will be repeated to 𝑁group times, collecting 𝑁group groups of responses. Once the groups are obtained, we can estimate the advantage value for each group and assign it to the responses. Since we use a bootstrap sampling strategy to construct groups, some responses may appear in multiple groups. Therefore, for each response, we compute its final advantage by summing the advantages of all groups to which it belongs, i.e.,

1

𝑁∑︁group

𝐴ˆ𝑗 · I[𝑦ˆ𝑖 ∈ Yˆ 𝑗], (6)

𝐴ˆ𝑖 =

𝑗=1

where I[𝑦ˆ𝑖 ∈ Yˆ 𝑗] is an indicator function, which returns 1 if and only if the 𝑖-th response 𝑦ˆ𝑖 belongs to the 𝑗-th group Yˆ 𝑗, while returns 0 for others. In practice, we set 𝑁group = 𝑁rollout for an efficient RLVR process.

Empirical Insight: Reduction in Training Budget. To assess the effectiveness of bootstrap sampling for Pass@k Training, we conduct Pass@1 Training and Pass@k Training with full sampling (described in Section 2.2) with different rollout times as the baseline methods, and present the evaluation in Figure 4. Given the same number of rollouts 𝑁rollout (i.e., “𝑁rollout = 32 w/ Full Sampling” v.s. “𝑁rollout = 32 w/ Bootstrap Sampling”), bootstrap sampling outperforms full sampling. This improvement arises from the fact that bootstrap sampling generates a larger number of groups, which in turn reduces the variance of the sampled reward distribution relative to the true distribution, leading to more stable and effective training. With the same number of groups 𝑁group, bootstrap sampling does not lead to significant performance degradation on the Pass@k metric compared to full sampling (i.e., “𝑁rollout = 128 w/ Full Sampling”), and it requires only one-fourth of the theoretical computational cost, resulting in higher training efficiency. Additionally, it achieves comparable performance to full sampling on the Pass@1 metric. In conclusion, Pass@k Training with bootstrap sampling outperforms Pass@1 Training and enhances the efficiency of the training process with full sampling.

Takeaway from Section 2.3

Compared with the full sampling-based Pass@k Training method, the bootstrap sampling-based training method can achieve better training results with the same number of rollouts. With the same number of groups, it can reduce computational overhead while achieving comparable performance.

#### 2.4 Analytical Derivation of Efficient and Effective Pass@k Training

Following the idea of the bootstrap sampling mechanism described in Section 2.3, we derive the analytical solution of the response advantage (i.e., 𝐴ˆpos and 𝐴ˆneg) to remove the variance brought by the sampling operation for constructing the groups. The details of the derivation are presented in Appendix B.

To deduce the analytical formula of the advantage, we start by analyzing the advantage reward and standard deviation of the groups, i.e., 𝑅¯group and 𝜎group. The group that contains at least one positive response (named as positive group) will be assigned the positive reward 𝑅pos, while the other groups (named as negative group) will be endowed with the negative reward 𝑅neg. Following the advantage estimation method of DAPO, it is critical to calculate the average and standard variance of the reward scores of the groups. First, the average reward of the group can be formulated as the following equation,

1

𝑁totalgroup × 𝑁posgroup × 𝑅pos + 𝑁neggroup × 𝑅neg , (7)

𝑅¯group =

where 𝑁totalgroup refers to the total number of groups, and 𝑁posgroup and 𝑁neggroup denote the number of positive and negative groups, respectively. To calculate the number of positive and negative groups, we first define the

number of positive responses as 𝑁pos and the number of negative responses as 𝑁neg. Typically, 𝑁pos + 𝑁neg = 𝑁rollout. Based on the above definition, as each group is constructed by selecting 𝑘 responses, we can obtain the total number of the group 𝑁totalgroup as follows,

𝑁totalgroup =

𝑁rollout 𝑘

= 𝑁posgroup + 𝑁neggroup. (8)

Since negative groups do not contain the positive responses, when and only when randomly sampling 𝑘 negative responses from the whole responses O, these sampled responses can construct a negative group. Thus, the number of negative groups can be calculated as follows,

𝑁neggroup =

𝑁neg 𝑘

. (9)

According to Eq. 8 and Eq. 9, we can obtain the number of the positive groups,

𝑁posgroup = 𝑁totalgroup − 𝑁neggroup =

𝑁rollout 𝑘 −

𝑁neg 𝑘

. (10)

Substitute Eq. 8, Eq. 9, and Eq. 10 into Eq. 7, we can obtain the average rewards of the group 𝑅¯group,

𝑁neg 𝑘 𝑁rollout 𝑘

𝑅¯group = 1 −

. (11)

Based on the average rewards of the group 𝑅¯group, the standard variance can be calculated as follows,

𝜎group = √︃𝑅¯group × 1 − 𝑅¯group . (12)

Based on the average (Eq. 11) and the standard variance (Eq. 12) of reward scores, we can finally deduce the advantage of the positive group 𝐴ˆposgroup and the negative group 𝐴ˆneggroup as follows,

𝑅pos − 𝑅¯group 𝜎group

𝐴ˆposgroup =

𝑅neg − 𝑅¯group 𝜎group

1 − 𝑅¯group 𝜎group

, 𝐴ˆneggroup =

=

= −

𝑅¯group 𝜎group

. (13)

To transfer the group-relative advantage 𝐴ˆposgroupand 𝐴ˆneggroup obtained in the previous section to the responserelative advantage 𝐴ˆ𝑝𝑜𝑠 and 𝐴ˆ𝑛𝑒𝑔, we need to consider, for each response, the correctness of the group it belongs to and compute the advantage value proportionally. Typically, a response will belong to 𝑁

rollout−1

𝑘−1

groups, because a group can be formed with the current response if and only if 𝑘 − 1 responses are selected from the remaining 𝑁rollout − 1 responses. Further, for a positive response, the groups that it belongs to can always pass the Pass@k verification (i.e., positive group). Thus, the advantage of a positive response 𝐴ˆpos can be calculated as follows,

1 − 𝑅¯group 𝜎group

𝐴ˆpos =

. (14)

- Figure 5 Training progress of Pass@1 Training and Pass@k Training with Analytical Derivation and Bootstrap Sampling on baseline setting.

Then, considering a negative response, the group that it belongs to is the negative group if and only if the other 𝑘 − 1responses are all negative responses. In this case, the number of required groups is 𝑁

neg−1

𝑘−1 , i.e., the current response can form a negative group with any 𝑘 − 1 responses selected from the remaining 𝑁neg − 1 negative responses. Based on the number of negative groups, we can compute the number of positive groups by subtracting the number of negative groups from the total number of groups, i.e., 𝑁

neg−1 𝑘−1 .

𝑘−1 − 𝑁

rollout−1

##### Thus, the advantage of a negative response 𝐴ˆneg can be calculated as follows,

𝑁neg−1 𝑘−1 𝑁rollout−1 𝑘−1

𝐴ˆneg = 1 − 𝑅¯group −

× (𝜎group)−1 . (15)

After obtaining the analytical solutions of response-relative advantage 𝐴ˆpos and 𝐴ˆneg, we directly employ them in the advantage estimation process and then optimize the model parameters. By examining the analytical solutions of the advantage value, we observe that it depends only on the total number of sampled responses 𝑁rollout, the number of positive responses 𝑁pos, the number of negative responses 𝑁neg, and the value of 𝑘. Therefore, after the rollout procedure, we can directly compute the advantage value of each response for RLVR training without going through the previously described cumbersome reward calculation process.

Empirical Insight: Further Improvement on Pass@k. For the evaluation and comparison, we unified the number of rollouts 𝑁rollout to 32 and compared the training effects of Pass@1 Training and Pass@k Training with bootstrap sampling and analytical derivation. The experimental results are presented in Figure 5. To present the comprehensive evaluation, we also conduct the external experiments about different LLMs on various tasks and present the results in Appendix E. In the experiment, we can observe that both types of Pass@k Training achieve better results than Pass@1 Training, which further confirms the effectiveness of Pass@k Training. When the number of training steps increases, the Pass@k Training based on bootstrap sampling experiences a relatively sharp performance fluctuation at 400 steps, with the Pass@k performance declining, which indicates that this method has certain instability. In contrast, for the method based on bootstrap sampling, Pass@k Training with analytical derivation eliminates the sampling process required for constructing groups. It directly reduces the variance caused by the sampling process through the calculation of analytical solutions, thereby providing a more stable training process. Thus, the method of Pass@k Training with analytical derivation can reduce fluctuations during the training process and bring about continuous performance improvements as the number of training steps increases.

Takeaway from Section 2.4

Pass@k Training with analytical derivation not only avoids the computational overhead caused by a large number of rollouts in full sampling, but also eliminates the variance introduced by sampling in bootstrap sampling. This makes the RLVR training process more efficient and effective, and can guide the model’s exploration ability to continuously improve as the number of training steps scales.

### 3 Balancing Exploration and Exploitation with Pass@k Training

In this section, we further investigate the features and effectiveness of Pass@k Training. First, we compare Pass@k Training with commonly used methods for enhancing model exploration ability [20, 38] to further verify its effectiveness (Section 3.1). Next, to further understand how Pass@k Training influences its exploration capability, we examine the diversity of model’s responses and the entropy of policy distribution (Section 3.2). After that, we wonder whether the improvement brought by Pass@k Training can be transferred to other domains or tasks, and then assess the generalization performance of it (Section 3.3). Moreover, as RLVR stability and robustness are widely concerned [6, 18, 19], we analyze how the value of 𝑘 affects the Pass@k Training process (Section 3.4). Finally, since Pass@1 is a more important metric in practical applications, we explore how to transfer the benefits of Pass@k Training to the model’s Pass@1 performance, and the experiment results demonstrate the high practical value of Pass@k Training (Section 3.5).

#### 3.1 How does Pass@k Training Compare to Noise Rewards or Entropy Regularization?

[Figure 11]

[Figure 12]

(a) Pass@k Performance of Noise Rewards. (b) Pass@k Performance of Entropy Regularization. Figure 6 Training progress of Noise Rewards and Entropy Regularization on baseline setting.

Inspired by the Pass@k Training procedure (Section 2.2) and the previous work [20], we conduct a comparison between Pass@k Training and two baseline approaches, i.e., Noise Rewards and Entropy Regularization.

Noise Rewards. Reviewing the RLVR procedure that leverages the Pass@k metric as the reward (as described in Section 2.2), we note that some negative responses may receive the positive reward 𝑅pos if they belong to a positive group. This raises the question of whether the improvement in Pass@k scores is partially driven by learning from these negative responses with counterfactual positive rewards. To investigate this, we conduct an experiment in which a certain proportion (i.e., 10%, 30%, and 50%) of the rewards assigned to negative responses are flipped. The results are presented in Figure 6a. Empirical results indicate that encouraging LLMs to learn from negative responses does not contribute to improving their reasoning abilities. On the contrary, introducing a higher proportion of noise into the reward significantly degrades model performance. As the proportion of flipped rewards increases, the model’s performance deteriorates progressively on both Pass@1 and Pass@k metrics. Furthermore, performance continues to decline with additional training steps. These findings suggest that naively incorporating noise into the reward does not enhance the reasoning capabilities of LLMs. Instead, the proportion of noise must be carefully controlled, such as through the structured design

of the Pass@k metric, which can guide LLMs beyond the limitations of their existing reasoning abilities.

Entropy Regularization. A surge of studies [10, 12, 20, 47] have pointed out that entropy can indicate the exploration ability of LLMs and can be incorporated into the objective function to preserve their exploration ability. Thus, following the previous work [20], we employ entropy regularization with the coefficient of {0.001, 0.003, 0.005} in the RLVR training process and present the results in the right part of Figure 6b. According to the results, we can find that a high coefficient of entropy regularization might cause the collapse of the model, like setting the coefficient to 0.005. Although the small coefficient of entropy regularization does not make LLM crush, it still cannot outperform Pass@k Training, and even lead to the decrease of the performance of LLMs with the increase of the training steps. The above phenomenon indicates that entropy regularization might affect training effectiveness and stability.

Discussion about Other Entropy-guided Approaches. We compare the effectiveness between Pass@k Training and the naive implementation of Entropy-guided Approach (i.e., Entropy Regularization). Moreover, there are several other methods, such as integrating the entropy into the advantage function [10] or focusing on tokens with high covariance [12]. Similarly, these methods might introduce a new trade-off: overly strict constraints may lead to underfitting and insufficient model training, while overly loose constraints can result in instability during training, potentially affecting the training effectiveness and model performance [6, 18, 19], since entropy conflicts with the Pass@1 metric. Therefore, the hyper-parameters should be carefully selected during the above methods to bring the performance improvement of LLMs. Actually, these methods are orthogonal to Pass@k Training, meaning that we can also combine these methods with Pass@k Training to achieve better training results. To verify this, we conduct the experiments in Section 4.2.3 to assess the effectiveness of Pass@k Training based on the guidance of policy entropy, demonstrating significant improvements.

Takeaway from Section 3.1

Pass@k Training outperforms Noise Rewards and Entropy Regularization: randomly flipping the reward of negative responses might degrade the performance of LLMs, and incorporating Entropy Regularization brings new trade-off issues, which hardly achieve continuous improvement.

#### 3.2 Does Pass@k Training Really Improve the Exploration Ability of LLMs?

[Figure 13]

[Figure 14]

(a) Diversity of Negative Responses on Maze Tasks. (b) Entropy of Policy Distribution on Puzzle Tasks. Figure 7 Training progress of Pass@1 Training and Pass@k Training on baseline setting.

To analyze the changes in exploration of LLMs during the RLVR training process, from the perspective of answer diversity and entropy of policy distribution, we conduct the related empirical study and show the corresponding results in Figure 7.

Answer Diversity of Negative Responses. We counter the accuracy and the ratio of different answers among the negative responses of Pass@k and Pass@1 Training, which is presented in Figure 7a, aiming to assess the

exploration ability of LLMs on the uncertain answer. According to the results, we observe that the answer diversity of the negative response stays at the same level during the RLVR training process, indicating that the LLMs try to select the “safe” actions and tends to generate similar answers in the exploration procedure, limiting the scope of exploration and constrain the effectivenss of RLVR. Differently, in Pass@k Training, the model is encouraged to achieve a higher Pass@k score and naturally learn the strategy to generate diverse answers when the model is not confident about this question. In this case, the exploration ability of LLM is enhanced and thereby improves its exploitation ability (i.e., Pass@1 score).

Entropy of Policy Distribution. In Figure 7b, the results show a similar conclusion to our previous discussion on answer diversity. Pass@k Training keeps the entropy of policy distribution at a relatively high level, while Pass@1 Training induces entropy to converge to a low value. This phenomenon suggests that LLMs can keep their exploration ability during Pass@k Training, but will lose their exploration ability during Pass@1 Training. On the other hand, we can also observe that Pass@k Training leads to an increase in entropy, starting from step 200 of the RLVR procedure. This phenomenon validates our hypothesis that using Pass@k as the training objective can encourage the model to conduct more exploration, thereby naturally increasing entropy.

In conclusion, exploration and exploitation do not conflict with each other, and they can improve mutually. Pass@k Training can achieve this goal.

Takeaway from Section 3.2

Pass@k Training can encourage the model to conduct more exploration, generating diverse answers that naturally leads to an increase in entropy, when the model does not have sufficient confidence to generate the correct answer.

#### 3.3 What is the Generalization Ability of LLMs After Pass@k Training?

Table 1 Pass@1/Pass@k Performance of Qwen2.5-7B-Instruct trained on different RLVR approaches.

In-Domain Tasks Out-of-Domain Tasks

Pass@1/Pass@k

ARC-AGI 1 Enigmate KORBench AIME 2025

Qwen2.5-7B-Instruct 2.4/4.8 4.8/10.1 36.5/45.9 4.2/15.8 + Pass@1 Training 3.3/3.8 12.9/21.3 37.7/45.6 5.4/19.1 + Pass@k Training 4.0/5.3 17.9/29.8 47.7/63.5 7.1/22.4

To analyze the generalization ability of Pass@k Training, we conduct the corresponding experiments and present the results in Table 1. We can observe that Pass@1 and Pass@k Training can enhance the model’s capacities on in-domain and out-of-domain tasks, suggesting the strong generalization ability of the RLVR training process. Further, comparing the performance between these two training approaches, the model trained through Pass@k outperforms the model trained on Pass@1. The reason behind it is that Pass@k Training encourages models to explore better solutions, which can be easily generalized to other tasks. In contrast, Pass@1 Training makes LLMs behave conservatively, thereby affecting LLMs’ OOD performance.

Takeaway from Section 3.3

Pass@k Training exhibits stronger generalization ability than Pass@1 Training, achieving greater improvements over the base model in both in-domain and out-of-domain testing.

#### 3.4 How does the Value of k Affect Pass@k Training?

To analyze the robustness of Pass@k Training, we adjust the value of 𝑘 in {4, 8, 16}to perform RLVR training on Maze tasks and present the training reward and Pass@k performance of the testset in Figure 8a and Figure 8b, respectively. Whatever the value of 𝑘, the training rewards can be improved to a relatively high level as the training progresses, indicating the value of 𝑘 is not a vital factor that can help LLMs escape

(a) Training Reward. (b) Pass@k Performance on Test set.

[Figure 17]

[Figure 18]

(c) Training Reward. (d) Pass@k Performance on Test set. Figure 8 Training progress of Pass@k Training under various 𝑘 and learning rate (LR).

from the local optimum of Pass@1 Training. However, with the increase of 𝑘, the improvement slows down, affecting training efficiency. Through analyzing the analytical solutions of advantage values (i.e., Eq. 14 and Eq. 15), we can realize that a larger value of 𝑘 will bring a smaller value of advantage, resulting to a shorter optimization step, causing lower training efficiency.

Based on this phenomenon, we investigate whether scaling the learning rate (LR) to enlarge the optimization step can improve training efficiency. Following this idea, we employ the learning rate in {1×10−6, 2×10−6, 4× 10−6} on the setting of 𝑁 = 32 and 𝑘 = 8, and present the results in Figure 8c and Figure 8d. With the increase of learning rate, the inflection point appears earlier, indicating higher training efficiency. The training efficiency of Pass@8 training even outperforms Pass@4 training when we employ 4 × 10−6 as learning rate. These results have shown that the issues of training efficiency can be easily mitigated.

Takeaway from Section 3.4

Pass@k Training exhibits strong robustness to the choice of the value of 𝑘, leading to a stable and effective training process. Although there is a decline in the model’s optimization efficiency as 𝑘 increases, this issue can be easily addressed by enlarging the learning rate.

#### 3.5 Can the Benefits from Pass@k Training Be Transferred to Pass@1 Performance?

To transfer the benefits brought by Pass@k Training to LLM Pass@1 performance, a native implementation is continually performing Pass@1 Training on the model, which is trained through Pass@k Training. We employ this approach in the RLVR training process and present the results of Qwen models on Puzzle tasks and Seed1.5-VL-Small (Internal Version) on multi-modal reasoning tasks in Table 2 and Table 3, respectively.

- Table 2 Enigmata Pass@1/Pass@k Performance of Qwen2.5 models trained on different RLVR approaches. “P@1 T.” and “P@k T.” denote the Pass@1 Training and Pass@k Training with analytical derivation, respectively.

Crypto Arithmetic Logic Grid Graph Search Sequential Overall Closed-source LLMs (Pass@1)

Grok-2-1212 10.1 9.4 50.0 12.8 17.6 3.9 6.4 13.6 GPT-4o-1120 26.2 1.9 34.5 17.8 19.3 6.0 3.9 14.2 Claude-3.7-Sonnet 38.1 16.7 60.0 22.9 22.4 7.8 15.0 22.7

RLVR on Qwen2.5-7B-Instruct (Pass@1/Pass@k)

- Baseline 0.1/0.7 1.0/3.3 28.1/48.4 3.7/9.2 3.0/11.6 0.3/1.0 2.7/5.4 4.7/10.1

+ P@1 T. 1.2/5.7 6.6/28.0 41.1/68.4 14.8/22.8 14.5/20.4 3.3/6.8 9.6/12.7 12.9/21.3 + P@k T. 14.0/39.7 27.4/63.0 46.0/74.0 18.8/27.8 15.2/21.5 5.3/12.3 14.1/18.3 17.9/29.8 + P@k T. + P@1 T. 96.9/98.3 36.2/67.7 49.3/71.8 30.9/37.5 20.3/30.7 25.8/37.5 10.6/12.9 30.8/40.6

RLVR on Qwen2.5-32B-Instruct (Pass@1/Pass@k)

- Baseline 1.5/5.3 4.6/16.0 45.7/71.1 11.6/20.6 8.0/26.7 2.3/6.4 7.7/16.3 10.9/21.6

+ P@1 T. 95.8/99.7 53.0/85.0 76.6/92.2 38.4/47.4 44.4/57.8 47.0/58.8 21.8/25.8 45.2/56.0 + P@k T. 93.8/99.3 51.1/86.3 74.8/92.4 39.0/49.6 42.7/61.3 45.9/59.9 21.5/26.6 44.5/57.4 + P@k T. + P@1 T. 95.9/99.3 49.6/84.3 82.0/94.9 40.0/51.0 48.2/60.2 48.8/60.8 22.2/26.2 46.8/57.9

- Table 3 Pass@1/Pass@k Performance of Seed1.5-VL-Small (Internal Version) trained on different RLVR approaches. Seed1.5-VL-Small (Internal Version) is an MoE model that contains fewer parameters than Seed1.5-VL [17].

Pass@1/Pass@k MathVision MMMU Avg.

Seed1.5-VL-Small (Internal Version) 54.6/72.5 71.2/80.2 62.9/76.4 + Pass@1 Training 55.3/74.0 72.0/83.7 63.7/78.9 + Pass@k Training 53.9/75.6 72.0/84.3 63.0/80.0 + Pass@k Training + Pass@1 Training 56.4/76.8 72.3/84.5 64.4/80.7

Moreover, to present the comprehensive evaluation, we also conduct the external experiments about different LLMs on Engimata and mathematical tasks (e.g., AIME 2024 [2] and AIME 2025 [3]) in Appendix E.

First, the Pass@1 Training following the Pass@k Training can significantly improve the reasoning ability of LLMs, achieving remarkable Pass@1 performance. According to the results, we can observe that even the 7B model can surpass the powerful closed-source LLMs, including Grok-2, GPT-4o, and Claude-3.7-Sonnet. This might be because Pass@k Training enhances the exploration ability of the LLM, guiding it to escape from the local optimum and unleashing the potential of the LLM in the subsequent RLVR training.

Second, either the small-scale or large-scale LLMs (e.g., Qwen2.5 with 7B or 32B parameters) can benefit from Pass@k Training. Besides, the model architecture and the model family do not influence the improvement of continual Pass@1 Training. The Qwen model is the dense model, while Seed1.5-VL-Small (Internal Version) is the MoE model. Their Pass@1 performance can be further improved after Pass@k Training.

Third, the domain and form of downstream tasks also do not affect the transfer from LLM Pass@k performance to their Pass@1 performance. Our evaluation includes synthetic puzzle tasks that are expressed in natural language, and multi-modal reasoning tasks that contain pictures in the problem description. These tasks require LLMs to possess different categories of abilities, and our Pass@k Training can specifically enhance the corresponding capabilities, showing strong effectiveness.

Takeaway from Section 3.5

The benefits brought by Pass@k Training can be transferred to Pass@1 performance of LLMs, which is not affected by the scale of model parameters (e.g., 7B or 32B), model architecture (e.g, dense model or MoE model), model family (i.e., Qwen model or Seed model), or downstream tasks (natural language tasks or multi-modal tasks).

(a) Pass@1 Training. (b) Pass@k Training.

Figure 9 The curves of advantage function of Pass@1 Training and Pass@k Training on the setting of 𝑁rollout = 32.

### 4 Generalizing Pass@k Training via Implicit Reward Design

As introduced in Section 2, we achieve effective and efficient Pass@k Training by deriving the analytical form of the advantage function. In this section, we further investigate the key factors contributing to the success of Pass@k Training from an advantage perspective (Section 4.1). Moreover, the advantage function design can be viewed as a form of implicit reward design. Motivated by this, we explore how to design advantage functions directly based on optimization objectives in scenarios where it is difficult to derive analytical solutions from the reward function (Section 4.2).

#### 4.1 Difference Between Pass@1 and Pass@k Training

##### 4.1.1 Analysis Based on Advantage Value Curves

To analyze why Pass@k Training can help LLMs escape the local optimum, we first visualize the advantage curves of Pass@1 Training and Pass@k Training across responses with different correctness levels, as in GRPO and its variants, the advantage value depends solely on the correctness of the model’s response. Furthermore, we observe that during the optimization process, the advantage value is directly multiplied by the gradient and can be interpreted as a scaling factor for the gradient. In this context, a larger absolute value of the advantage indicates a greater scaling of the gradient, and thus a larger update step for the corresponding sample. This implies that the model places greater optimization effort on samples with higher advantage magnitudes. Therefore, we argue that the absolute value of the advantage is also an important aspect worthy of investigation. Based on this insight, and to simplify the analysis, we compute the sum of the absolute advantage values across all responses, as defined below:

𝜂 = 𝑁pos × |𝐴ˆpos| + 𝑁neg × |𝐴ˆneg|, (16)

The curves of 𝜂 (named as Sum of Absolute Advantage) are added to our visualization and presented in Figure 9. Comparing the curves of 𝜂of Pass@1 Training and Pass@k Training, we can observe that there are three major differences.

Maximum of Sum of Absolute Advantage 𝜂. The maximum of 𝜂 of the Pass@1 Training approach is much higher than ones of Pass@k Training approach. As we discussed in Section 3.4, the maximum advantage values might affect training efficiency, and adding the coefficient on the loss function to adjust the advantage values can mitigate this issue. Thus, the maximum is not the critical factor that helps Pass@k Training outperform Pass@1 Training.

Argmax of Sum of Absolute Advantage 𝜂. According to the curves in Figure 9, the argmax of 𝜂 are significantly different between Pass@1 and Pass@8 Training. For Pass@1 Training, the maximum of 𝜂 appears at the

Figure 10 Training progress of various of Pass@1 Training and Pass@k Training on baseline setting.

position of 50% accuracy (i.e., 𝑁pos = 0.5 × 𝑁rollout), while the position of maximum of 𝜂 is 25% accuracy (i.e., 𝑁pos = 0.25 × 𝑁rollout). This phenomenon suggests that Pass@k Training focuses on optimizing harder problems, while Pass@1 Training focuses on problems with medium difficulty. This further demonstrates that Pass@k Training tends to guide the model toward solving previously unsolved or difficult problems, rather than overfitting to those it has already mastered.

Trend of Sum of Absolute Advantage 𝜂. Another key difference between the function curves of Pass@1 and Pass@k Training lies in the trend of the function itself. In the 𝜂 curve of Pass@k Training, the value increases until it reaches a peak, and then gradually decreases to zero. Under this setting, when the problem is relatively easy (i.e., the correctness is higher than 60%), the optimization strength applied by the model (as indicated by the value of 𝜂) becomes smaller than that for harder problems. This further demonstrates that Pass@k Training focuses more on optimizing problems the model has not yet mastered. In contrast, during Pass@1 Training, the 𝜂 curve is symmetric around the point of maximum value, indicating that the training process allocates equal attention to both easy and hard problems.

##### 4.1.2 Analysis Based on Model Performance

As we discussed in the previous section, the effectiveness of the argmax and the trend of the sum of absolute advantage 𝜂 still remain unclear. Thus, in this section, we design the corresponding experiments to analyze their effectiveness, based on model performance. Additionally, we designed two training methods that serve as intermediates between Pass@1 and Pass@k Training, i.e., removing the advantage values of the easy problems and combining the advantage estimation approaches of Pass@1 and Pass@k based on the accuracy of the current prompt. The curves of 𝐴ˆpos, 𝐴ˆneg, and 𝜂 of these four training approaches are presented in Figure 18a and Figure 18b.

First, when the correctness of a response is high, we design the advantage function to decrease gradually toward zero. This setting allows the training reward to increase steadily during the optimization process, indicating that the model avoids getting stuck in a local optimum (i.e., the blue line and purple line). When this optimization is removed, the reward on the training set fails to continue increasing, suggesting that the model has already converged to a local optimum and is no longer learning new knowledge during the RLVR process (i.e., the red line and green line). This phenomenon suggests that excessive learning from easy examples is a key factor causing the model to fall into local optima. Therefore, reducing the degree of learning from easy questions can help prevent the model from getting trapped in such suboptimal solutions.

Second, simply setting the reward for easy questions to zero is not sufficient to effectively prevent the model from over-optimizing on them; it merely delays the point at which the model falls into a local optimum. As shown in Figure 10, removing the optimization for easy questions (represented by the red line) leads to higher training rewards and better test performance compared to the baseline (represented by the green

Figure 11 Training progress of Pass@k Training and Exceeding Pass@k Training on baseline setting.

line). However, both curves exhibit similar trends: after an initial phase of improvement, model performance gradually plateaus, making further progress difficult.

Third, regarding the choice of the argmax position of the 𝜂 function, a comparison of the curves in Figure 10 reveals that shifting the argmax forward leads to higher optimization efficiency. Specifically, the model is able to escape from local optima more quickly, and a turning point in training reward appears earlier. This phenomenon suggests that hard problems contribute more significantly to model improvement and yield better optimization effects. Assigning greater optimization strength to harder problems can thus effectively enhance training efficiency, allowing the model to achieve better performance with fewer training steps.

Based on the above results and discussions, several preliminary conclusions can be made, i.e., the argmax of 𝜂influences the training efficiency and the trend of 𝜂 prevents model from falling into local optimum. Besides, it is important to note that this is only our initial conclusion. More comprehensive experiments tailored to specific tasks and scenarios are required for further validation.

Takeaway from Section 4.1

In the RLVR training process, simple problems might easily lead to overfitting. Appropriately reducing the optimization strength for these problems helps prevent the model from getting stuck in local optima, thereby achieving better overall performance.

#### 4.2 RLVR Training Through Implicit Reward Design

Building upon the analysis of the curve properties of advantage values in the previous section, we now explore preliminary modifications to the advantage function in this section, i.e., implicit reward design. Our goal is to explore the potential of implicit reward design and to propose several promising directions for future research.

##### 4.2.1 Exceeding Pass@k Training

In previous discussion, we have found that the position of maximum value of 𝜂will influence the training objective (focus on Pass@1 or Pass@k). Based on these observations and conclusions, we hypothesize that an earlier peak in the 𝜂 function leads to better optimization performance in Pass@k Training. To test this hypothesis, we design a transformation function as follows:

4 10log(𝑁pos + 0.5)

, 𝐴ˆ′ = 𝑓 (𝑁pos) × 𝐴.ˆ (17)

𝑓 (𝑁pos) =

The advantage value curve after applying the transformation function is shown in Figure 18c. We observe that, in the transformed curve, the peak of the 𝜂 function is shifted forward to the position where the correctness

Figure 12 Training progress of Pass@k Training and Combination Training on baseline setting.

is 321 . According to our hypothesis, such a modification of the advantage function is expected to result in better optimization performance for Pass@k Training. We integrate this transformed function into the RLVR

training process (named as Exceeding Pass@k Training), and the corresponding training results are presented in Figure 11.

From the experimental results, we observe that Exceeding Pass@k Training can effectively improve the model’s Pass@k performance during the early training stage. However, since this method places excessive emphasis on difficult problems, the improvement in Pass@1 performance on downstream tasks progresses more slowly. Based on these observations and analyses, we hypothesize that the computation of advantage values could be adaptively adjusted according to the model’s current state. We leave this as a direction for future work.

##### 4.2.2 Combination of Pass@1 and Pass@k Training

From the previous analysis, we observe that Pass@k Training focuses more on optimizing harder problems, and prevents the model from overfitting to the easy problems. Motivated by this observation, we consider whether combining Pass@1 and Pass@k Training could be beneficial. Thus, we design the following formula to estimate the final advantage value:

𝑁pos 𝑁 × 𝐴ˆPass@k + (1 −

𝑁pos 𝑁 ) × 𝐴ˆPass@1, (18)

𝐴ˆ =

Where 𝐴ˆPass@k and 𝐴ˆPass@k denote the advantage values estimated by Pass@k and Pass@1 Training approach, respectively. In the above formula (named as Combination Training), when the sampled response has a low correctness score, the advantage value from Pass@1 Training receives a higher weight and dominates the training process, leading to high training efficiency. Conversely, when the correctness is high, the advantage value from Pass@k Training is assigned a greater weight, thereby avoiding LLMs from overfitting to the problems that it has already mastered.

In Figure 12, we present the training results of the Qwen series models on the Enigmata benchmark. We observe that, for both Pass@1 and Pass@8 metrics, models trained with Combination Training consistently outperform those trained with standard Pass@k Training. During the Combination Training process, model performance improves rapidly and maintains a high growth rate. In contrast, Pass@k Training leads to slower performance gains. This is because difficult problems require extensive exploration for the model to learn effectively, making rapid improvement challenging. At the same time, easy problems receive low but sufficient optimization strength during training. These two factors together contribute to the lower optimization efficiency of Pass@k Training compared to Combination Training. The above analysis further supports the idea that adapting the advantage function based on the model’s current state can effectively enhance model performance.

Figure 13 Training progress of Pass@k Training and Adaptive Training on baseline setting.

##### 4.2.3 Adaptive Training based on Policy Entropy

Building on the insights from the previous section, we explore whether the training objective can be adaptively adjusted throughout the RLVR process. Besides, as discussed in previous work [12, 47], the entropy of policy distribution can indicate its exploration ability. Thus, we conduct the Pass@k Training based on the guidance of policy entropy (named as Adaptive Training). Concretely, we first compute the average entropy 𝐸¯ of the sampled responses of each problem, and then rank each problem based on its 𝐸¯. We designate the top 50% as high-exploration problems and the rest as low-exploration problems. For high-exploration problems, we use the Pass@1 advantage function to help the model exploit prior exploration. For low-exploration problems, we apply the Pass@K advantage function to encourage further exploration. This approach uses policy entropy to guide advantage computation, allowing us to combine the strengths of different training strategies.

We present the experimental results in Figure 13. Experimental results show that under Adaptive Training, the model achieves effective improvements in both Pass@1 and Pass@K performance, outperforming both Pass@1 Training and Pass@K Training. This indicates that Pass@1 and Pass@K training are complementary. By designing a proper adaptation mechanism, it is possible to better leverage the strengths of both training methods, enabling the model to achieve improved performance on downstream tasks. This also confirms that the entropy of the policy distribution can serve as an indicator of the model’s exploration ability, and that it integrates well with Pass@K training. Using entropy as a monitoring signal to adjust RLVR training yields better results than directly using it as a training objective.

Takeaway from Section 4.2

Implicit reward design allows for better control over the optimization process, avoiding the complex theoretical derivation. Concretely, increasing the optimization strength for more difficult problems can effectively enhance the model’s ability to solve them (i.e., Pass@k performance), and combining or dynamically adjusting different forms of advantage estimation make it possible to improve both exploration and exploitation capabilities simultaneously.

### 5 Related Work

#### 5.1 Reinforcement Learning with Verifiable Rewards

To unleash the potential of LLM reasoning ability, DeepSeek directly employs reinforcement learning with verifiable rewards (RLVR) on DeepSeek-V3, obtaining the large reasoning model DeepSeek-R1-Zero [13], which can perform the reasoning process with complex reasoning actions (e.g., reflection and verification). Given the success of the DeepSeek-R1, a surge of studies [9, 21, 54] have explored the effectiveness of RLVR on the popular open-source LLMs, like Qwen [51], Mistral [24], and LLaMA [14]. Moreover, the RLVR training

paradigm can help LLMs to control their reasoning time [1], switch the reasoning pattern [11, 49], enhance the specific performance metric [42], and enhance their abilities without supervision [22, 61]. However, recent work points out that the popular RLVR algorithms (e.g., PPO [36] and GRPO [39]) still face serious challenges, like training instability, model collapse, and reward noise [28, 30, 45, 52, 59]. To mitigate these issues, existing researches propose the optimization on the rollout strategy [52], objective function design [28, 30, 45], and data selection [59]. Specifically, previous work [45] utilizes Pass@k as the reward on the policy gradient algorithm [48] to encourage models to solve hard problems. However, the intrinsic connection between Pass@k RLVR training and LLM exploration ability has not been fully recognized. Thus, we further adopt the Pass@k metric in GRPO and its variants through three approaches (Figure 5), and derive the analytical solution of advantage values of Pass@k reward in RLVR training. Moreover, according to empirical experiments and theoretical analysis, we discuss the benefits of Pass@k Training in balancing the exploration and exploitation abilities of LLMs during the RLVR training procedure, showing the huge potential of Pass@k RLVR training and pointing out the promising future research directions.

#### 5.2 Effective Exploration in Test-time Scaling

Recently, test-time scaling has been proposed to improve the performance of LLMs by consuming more computational resources at inference time [57]. Since the LLMs continuously leverages exploration-derived experience to optimize its performance, effective exploration is important and necessary during the testtime scaling process [20, 32]. However, existing work reveals that the exploration ability is limited by the corresponding base model, hindering the continuous scaling of model performance [53]. To mitigate this issue, previous work proposed several approaches, including achieved by adjusting the sampling hyperparameters [9, 20, 21], performing self-verification and self-reflection [25, 29, 35], or leveraging external models to verify the reasoning process [31, 55]. Beyond these approaches from an external perspective of the model, it is equally important to explore the model’s exploration capability through its internal mechanisms. Current studies start from the perspective of the entropy of policy distribution, pointing out that entropy can indicate the exploration ability of LLMs [10, 12] and high-entropy tokens are vital for model optimization [47]. Based on these findings, training the critical tokens [47] and adding regularization [20, 28] are employed in the RLVR training process to avoid the degradation of the exploration capability of LLMs. Further, several studies focus on enhancing the exploration abilities of LLMs by selecting useful sampled experience [37, 59], integrating entropy into advantage estimation [12].

### 6 Conclusion

In this work, we proposed the Pass@k Training method within the RLVR framework, aiming to enable mutual improvement between the exploration and exploitation capabilities of the LLM, thereby pushing the limits of its overall performance. We first demonstrated that using Pass@k as the reward can effectively enhance the model’s ability to explore diverse outputs, which in turn improves its exploitation capability. Next, to improve training efficiency and effectiveness, we introduce the bootstrap sampling mechanism and analytical derivation of the advantage function to optimize the Pass@k Training procedure. After that, to better understand the inner mechanism of Pass@k Training, we proposed five research questions from different aspects to answer why the Pass@k Training works and what benefits can be brought from the Pass@k Training.

Moreover, inspired by the effectiveness of Pass@k Training, we further analyzed it from the perspective of implicit reward design. By examining the curves of advantage value, we preliminarily identified two key factors contributing to the success of Pass@k Training, i.e., the argmax and the trend of the sum of absolute advantage 𝜂. Building on these insights, we conducted an initial exploration into designing customized advantage functions to further improve model performance. This exploration is relatively preliminary, but it has shown remarkable effectiveness. We consider it a promising direction for future research.

### Acknowledgement

We sincerely thank Enigmata Team [7] to provide the training and validation sets of Enigmata and share experiences of RLVR training on logic puzzles. We appreciate Songhua Cai and other contributors of the Seed Infrastructure team for infrastructure support.

### References

- [1] Pranjal Aggarwal and Sean Welleck. L1: controlling how long A reasoning model thinks with reinforcement learning. CoRR, abs/2503.04697, 2025.

- [2] AIME2024. Aime2024, 2024. URL https://huggingface.co/datasets/HuggingFaceH4/aime_2024.
- [3] AIME2025. Aime2025, 2025. URL https://huggingface.co/datasets/opencompass/AIME2025.
- [4] Susan Amin, Maziar Gomrokchi, Harsh Satija, Herke van Hoof, and Doina Precup. A survey of exploration methods in reinforcement learning. CoRR, abs/2109.00157, 2021.

- [5] Bradley C. A. Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. CoRR, abs/2407.21787, 2024.

- [6] Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, Tony Tong Wang, Samuel Marks, Charbel-Raphaël Ségerie, Micah Carroll, Andi Peng, Phillip J. K. Christoffersen, Mehul Damani, Stewart Slocum, Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J. Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco, Peter Hase, Erdem Biyik, Anca D. Dragan, David Krueger, Dorsa Sadigh, and Dylan Hadfield-Menell. Open problems and fundamental limitations of reinforcement learning from human feedback. Trans. Mach. Learn. Res., 2023, 2023.

- [7] Jiangjie Chen, Qianyu He, Siyu Yuan, Aili Chen, Zhicheng Cai, Weinan Dai, Hongli Yu, Qiying Yu, Xuefeng Li, Jiaze Chen, Hao Zhou, and Mingxuan Wang. Enigmata: Scaling logical reasoning in large language models with synthetic verifiable puzzles. CoRR, abs/2505.19914, 2025.

- [8] Zhipeng Chen, Kun Zhou, Xin Zhao, Junchen Wan, Fuzheng Zhang, Di Zhang, and Ji-Rong Wen. Improving large language models via fine-grained reinforcement learning with minimum editing constraint. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 5694–5711. Association for Computational Linguistics, 2024.

- [9] Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, Lei Fang, Zhongyuan Wang, and Ji-Rong Wen. An empirical study on eliciting and improving r1-like reasoning models. CoRR, abs/2503.04548, 2025.

- [10] Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. CoRR, abs/2506.14758, 2025.

- [11] Stephen Chung, Wenyu Du, and Jie Fu. Thinker: Learning to think fast and slow. CoRR, abs/2505.21097, 2025.

- [12] Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models. CoRR, abs/2505.22617, 2025.

- [13] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and S. S. Li. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025.

- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux,

- Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024.
- [15] Saeed Ghadimi and Guanghui Lan. Stochastic first- and zeroth-order methods for nonconvex stochastic programming. SIAM J. Optim., 23(4):2341–2368, 2013.

- [16] Gary L Grunkemeier and YingXing Wu. Bootstrap resampling methods: something for nothing? The Annals of thoracic surgery, 77(4):1142–1144, 2004.

- [17] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, and Ke Shen. Seed1.5-vl technical report. CoRR, abs/2505.07062, 2025.

- [18] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner 1 technical report. CoRR, abs/2505.22312, 2025.

- [19] Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning. arXiv preprint arXiv:2507.01006, 2025.

- [20] Zhenyu Hou, Xin Lv, Rui Lu, Jiajie Zhang, Yujiang Li, Zijun Yao, Juanzi Li, Jie Tang, and Yuxiao Dong. Advancing language model reasoning through reinforcement learning and inference scaling. CoRR, abs/2501.11651, 2025.

- [21] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. CoRR, abs/2503.24290, 2025.

- [22] Jinwu Hu, Zhitian Zhang, Guohao Chen, Xutao Wen, Chao Shuai, Wei Luo, Bin Xiao, Yuanqing Li, and Mingkui Tan. Test-time learning for large language models. CoRR, abs/2505.20633, 2025.

- [23] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman,

- Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024.
- [24] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b. CoRR, abs/2310.06825, 2023.

- [25] Yuhua Jiang, Yuwen Xiong, Yufeng Yuan, Chao Xin, Wenyuan Xu, Yu Yue, Qianchuan Zhao, and Lin Yan. PAG: multi-turn reinforced LLM self-correction with policy as generative verifier. CoRR, abs/2506.10406, 2025.

- [26] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tülu 3: Pushing frontiers in open language model post-training. CoRR, abs/2411.15124, 2024.

- [27] Shilei Li, Meng Li, Jiongming Su, Shaofei Chen, Zhimin Yuan, and Qing Ye. PP-PG: combining parameter perturbation with policy gradient methods for effective and efficient explorations in deep reinforcement learning. ACM Trans. Intell. Syst. Technol., 12(3):35:1–35:21, 2021.

- [28] Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. CoRR, abs/2505.24864, 2025.

- [29] Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, and Dong Yu. Trust, but verify: A self-verification approach to reinforcement learning with verifiable rewards. CoRR, abs/2505.13445, 2025.

- [30] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. CoRR, abs/2503.20783, 2025.

- [31] Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling. CoRR, abs/2504.02495, 2025.

- [32] Tongxu Luo, Wenyu Du, Jiaxi Bi, Stephen Chung, Zhengyang Tang, Hao Yang, Min Zhang, and Benyou Wang. Learning from peers in reasoning models. CoRR, abs/2505.07787, 2025.

- [33] Chengqi Lyu, Songyang Gao, Yuzhe Gu, Wenwei Zhang, Jianfei Gao, Kuikun Liu, Ziyi Wang, Shuaibin Li, Qian Zhao, Haian Huang, Weihan Cao, Jiangning Liu, Hongwei Liu, Junnan Liu, Songyang Zhang, Dahua Lin, and Kai Chen. Exploring the limit of outcome reward for learning mathematical reasoning. CoRR, abs/2502.06781, 2025.

- [34] Yurii E. Nesterov and Vladimir G. Spokoiny. Random gradient-free minimization of convex functions. Found. Comput. Math., 17(2):527–566, 2017.

- [35] Kusha Sareen, Morgane M. Moss, Alessandro Sordoni, Rishabh Agarwal, and Arian Hosseini. Putting the value back in RL: better test-time scaling by unifying LLM reasoners with verifiers. CoRR, abs/2505.04842, 2025.

- [36] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017.

- [37] Amrith Setlur, Matthew Y. R. Yang, Charlie Snell, Jeremy Greer, Ian Wu, Virginia Smith, Max Simchowitz, and Aviral Kumar. e3: Learning to explore enables extrapolation of test-time compute for llms. CoRR, abs/2506.09026, 2025.

- [38] Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. Spurious rewards: Rethinking training signals in RLVR. CoRR, abs/2506.10947, 2025.

- [39] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024.

- [40] Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J. Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron T. Parisi, Abhishek Kumar, Alexander A. Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Fathy Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson,

- Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura Culp, Lechao Xiao, Maxwell L. Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yundi Qian, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. Beyond human data: Scaling self-training for problem-solving with language models. Trans. Mach. Learn. Res., 2024, 2024.
- [41] Haoxiang Sun, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang, Lei Fang, and Ji-Rong Wen. Challenging the boundaries of reasoning: An olympiad-level math benchmark for large language models. CoRR, abs/2503.21380, 2025.

- [42] Yunhao Tang, Kunhao Zheng, Gabriel Synnaeve, and Rémi Munos. Optimizing language models for inference time objectives using reinforcement learning. CoRR, abs/2503.19595, 2025.

- [43] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025.

- [44] Luong Quoc Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 7601–7614. Association for Computational Linguistics, 2024.

- [45] Christian Walder and Deep Karkhanis. Pass@k policy optimization: Solving harder reinforcement learning problems. CoRR, abs/2505.15201, 2025.

- [46] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 9426–9439. Association for Computational Linguistics, 2024.

- [47] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. CoRR, abs/2506.01939, 2025.

- [48] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn., 8:229–256, 1992.

- [49] Siye Wu, Jian Xie, Yikai Zhang, Aili Chen, Kai Zhang, Yu Su, and Yanghua Xiao. ARM: adaptive reasoning model. CoRR, abs/2505.20258, 2025.

- [50] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing LLM reasoning with rule-based reinforcement learning. CoRR, abs/2502.14768, 2025.

- [51] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024.

- [52] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song,

- Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an open-source LLM reinforcement learning system at scale. CoRR, abs/2503.14476, 2025.
- [53] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? CoRR, abs/2504.13837, 2025.

- [54] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. CoRR, abs/2503.18892, 2025.

- [55] Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, and Dina Katabi. RL tango: Reinforcing generator and verifier together for language reasoning. CoRR, abs/2505.15034, 2025.

- [56] Qining Zhang and Lei Ying. Zeroth-order policy gradient for reinforcement learning from human feedback without reward inference. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

- [57] Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Zhihan Guo, Yufei Wang, Irwin King, Xue Liu, and Chen Ma. What, how, where, and how well? A survey on test-time scaling in large language models. CoRR, abs/2503.24235, 2025.

- [58] Yuxiang Zhang, Yuqi Yang, Jiangming Shu, Yuhang Wang, Jinlin Xiao, and Jitao Sang. Openrft: Adapting reasoning foundation model for domain-specific tasks with reinforcement fine-tuning. CoRR, abs/2412.16849, 2024.

- [59] Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in LLM reasoning. CoRR, abs/2506.01347, 2025.

- [60] Eric R. Ziegel. Resampling methods. Technometrics, 48(4):576, 2006.

- [61] Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, and Bowen Zhou. TTRL: test-time reinforcement learning. CoRR, abs/2504.16084, 2025.

## Appendix

### A Experiment Setup

- A.1 Details of Downstream Tasks In this section, we present detailed information of each downstream evaluation task.

Maze. We follow the framework proposed by previous work to synthesize the different sizes of mazes. Each maze is represented by text, containing 𝑛 rows and 𝑛 columns, a total of 𝑛 ∗ 𝑛 characters. Concretely, each of them is one of the following four characters “S”, “E”, “*”, and “.”, denoting the start point, destination, available place, and unavailable place, respectively. Given the maze, LLMs can first generate the thought or reasoning process and then generate the final answer, which includes one of the four actions “U”, “D”, “L”, and “R”, refering to moving up, down, left, and right, respectively. For training data, we construct the mazes with sizes of 9 × 9, 11 × 11, 13 × 13, and 15 × 15, to increase the diversity of training data. For test data, to evaluate the generalization of the RLVR process, we not only conduct the same sizes of the mazes with the training dataset, but also collect the mazes with sizes of 7 × 7, 17 × 17, 19 × 19, and 21 × 21. To ensure the validity of the experiment, we performed strict deduplication operations after generating the training and test data. The statistical information of the datasets is presented in Table 4. For better understanding, we present a test instance in Figure 14. To present the empirical insights more clearly, we only showed the results of the 9 × 9 maze in the above text, and the remaining results are presented in Appendix E.3.

Table 4 The statistical information of the Maze task.

7 × 7 9 × 9 11 × 11 13 × 13 15 × 15 17 × 17 19 × 19 21 × 21

Training Set - 10,000 10,000 10,000 10,000 - - Test Set 75 100 100 100 100 100 100 100

Enigmata. To assess the reasoning and logical abilities of LLMs, Enigmata proposed a comprehensive benchmark that includes the 36 categories of synthetic verifiable puzzles of 7 primary categories, including Crypto Puzzle, Arithmetic Puzzle, Logic Puzzle, Grid Puzzle, Graph Puzzle, Search Puzzle, and Sequential Puzzle. Each category can assess different abilities of LLMs. For better understanding, we present a test instance in Figure 15.

MathVision. MathVision selects 3,040 high-quality problems from human math competitions, each accompanied by relevant images. Solving these problems requires both careful interpretation of the visual information and rigorous mathematical reasoning. MathVision provides a benchmark for assessing a model’s multimodal understanding as well as its ability to perform rigorous mathematical reasoning. For better understanding, we present a test instance in Figure 16.

MMMU. MMMU includes college-level reasoning and comprehension tasks across six academic subjects, including Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering. Moreover, MMMU includes a wide range of image types, enabling a comprehensive assessment of a model’s capability to process and reason over different forms of visual information. For better understanding, we present a test instance in Figure 17.

- A.2 Implementation Details

Training. In our experiment, we adapt Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct as the backbone model and train it through DAPO. To enhance the efficiency of the training process, we only retain the clip-higher (i.e., 𝜀low = 0.2 and 𝜀high = 0.28) and token-level policy gradient loss, and remove other optimizations. For the training hyper-parameters, we set the learning rate for the policy model as 1 × 10−6 with 10 warmup steps, and employ 128, 32, and 32 as prompt batch size 𝐵𝑆prompt, mini-batch size 𝐵𝑆mini, and rollout times

Question

You need to solve the following maze.

'*' denotes the wall that you cannot walk through, '.' denotes available area that you can walk through. 'S' denotes the starting point, 'E' denotes the destination.

You need to start from the starting point and cross through the available area to reach the destination. There are four movement actions,

including Left, Right, Up, Down.

You need to use L to denote Left movement, R to denote Right movement, U to denote Up movement, and D to denote Down movement.

You can analysis the maze to find the correct path, and you should write the final path in the <answer> </answer>, e.g., <answer> LLRRDUL </answer>.

## Maze

*********

- *.......*
- *.*****.*
- *E......*
- *.*.*.*.*
- *.*.*...*
- *.*.*.*.*
- *S......*
- ********* Now try to analyze the maze and put the final path in the <answer> </answer>.

Ground Truth

<answer> UUUU <answer>

Figure 14 An example of Maze Task.

𝑛rollout, respectively. For the reward, the responses that pass the verification (named as positive responses) will be assigned the positive reward 𝑅pos = 1, while the other responses (named as negative responses) will be endowed with the negative reward 𝑅neg = 0. Additionally, we do not employ any regularization methods, such as KL or Entropy regularization.

Evaluation. To evaluate the performance of LLMs, we adopt 1.0 and 0.95 as the Temperature and Top_P. For each question, we sample 32 responses from LLMs for the Maze task and sample 8 responses from LLMs for other tasks, and then utilize the sampled response to compute the Pass@1 and Pass@k scores.

### B Details of Analytical Derivation

We present the details of the analytical derivation procedure mentioned in Section 2.4, including the derivation of the average of the group reward, standard deviation of the group reward, and response-relative advantage.

#### B.1 Derivation of the Average of Group Reward

1

𝑁totalgroup × 𝑁posgroup × 𝑅pos + 𝑁neggroup × 𝑅neg (19)

𝑅¯group =

𝑁rollout 𝐾 −

𝑁neg 𝐾 × 1 +

𝑁neg 𝐾 × 0 (20)

1

=

×

𝑁rollout 𝐾

𝑁neg 𝐾

. (21)

= 1 −

𝑁rollout 𝐾

Question

Apply a function to the final input list to generate the output list. Use any preceding inputs and outputs as examples to find what is the

function used. All example outputs have been generated using the same function.

### Response Format:

- Please output your answer within a code block (```), formatted as a list of numbers, for example: ``` [0, 2, 3] ``` # Examples

- Example 1:

- [6, 4] -> [6, 8, 4]

Example 2:

- [8, 3, 2, 0, 9, 7] -> [8, 5, 3, 2, 0, 9, 7]

- Example 3: [1, 2, 6, 0, 9, 3] -> [1, 5, 2, 6, 0, 9, 3]
- Example 4: [9, 7, 8] -> [9, 8, 7, 8] Example 5:

[1, 9, 6, 5, 0, 3, 8, 4, 7, 2] -> [1, 5, 9, 6, 5, 0, 3, 8, 4, 7, 2]

Example 6:

- [9, 8] -> [9, 8, 8]

# Test Problem:

- [7, 4, 6, 8, 0, 1, 3] ->

Ground Truth

```[7, 5, 4, 6, 8, 0, 1, 3]```

Figure 15 An example of Enigmata Task.

#### B.2 Derivation of the Standard Deviation of Group Reward

𝜎group = √︂ 1 𝑁rollout × 𝑁posgroup × 𝑅 ¯group − 𝑅pos 2 + 𝑁neggroup × 𝑅 ¯group − 𝑅neg 2 (22)

2

2

(𝑁

(𝑁

𝐾 ) (𝑁

𝐾 ) (𝑁

neg

neg

𝐾 − 𝑁

+ 𝑁

𝑁rollout

neg

neg

− 0

𝐾 × 1 −

− 1

𝐾 × 1 −

𝐾 )

𝐾 )

rollout

bootstrap

(23)

=

𝑁rollout 𝐾

2

2

𝑁neg 𝐾

𝑁neg 𝐾

𝑁neg 𝐾

𝑁neg 𝐾

(24)

= 1 −

+

× 1 −

×

𝑁bootstrap 𝐾

𝑁rollout 𝐾

𝑁rollout 𝐾

𝑁rollout 𝐾

𝑁neg 𝐾

𝑁neg 𝐾

𝑁neg 𝐾

𝑁neg 𝐾

(25)

= 1 −

×

× 1 −

+

𝑁rollout 𝐾

𝑁rollout 𝐾

𝑁rollout 𝐾

𝑁rollout 𝐾

𝑁neg 𝐾

𝑁neg 𝐾

(26)

= 1 −

×

𝑁rollout 𝐾

𝑁rollout 𝐾

= √︃𝑅¯group × 1 − 𝑅¯group . (27)

Question

Which number should be written in place of the question mark?

[Figure 29]

Ground Truth

60

Figure 16 An example of MathVision Task.

Question

Each of the following situations relates to a different company.

[Figure 30]

For company B, find the missing amounts.

- (A) $63,020
- (B) $58,410
- (C) $71,320
- (D) $77,490

Ground Truth

D

Figure 17 An example of MMMU Tasks.

#### B.3 Derivation of the Response-Relative Advantage

𝑁rollout − 1 𝐾 − 1 × 𝐴ˆgroup𝑝𝑜𝑠 + 0 × 𝐴ˆgroup𝑛𝑒𝑔 (28)

1

𝐴ˆpos =

×

𝑁rollout−1 𝐾−1

1 − 𝑅¯group 𝜎group

. (29)

=

𝑁neg − 1 𝐾 − 1 × 𝐴ˆgroup𝑛𝑒𝑔 (30)

𝑁rollout − 1 𝐾 − 1 −

𝑁neg − 1 𝐾 − 1 × 𝐴ˆgroup𝑝𝑜𝑠 +

1

𝐴ˆneg =

×

𝑁rollout−1 𝐾−1

𝑁neg−1 𝐾−1

𝑁neg−1 𝐾−1

× 𝐴ˆposgroup +

× 𝐴ˆneggroup (31)

= 1 −

𝑁rollout−1 𝐾−1

𝑁rollout−1 𝐾−1

𝑁neg−1 𝐾−1

𝑁neg−1 𝐾−1

1 − 𝑅¯group

𝑅¯group 𝜎group

(32)

= 1 −

𝜎group +

×

× −

𝑁rollout−1 𝐾−1

𝑁rollout−1 𝐾−1

𝑁neg−1 𝐾−1

= 1 − 𝑅¯group −

× (𝜎group)−1 . (33)

𝑁rollout−1 𝐾−1

### C Pseudo Code for Pass@k Training

We present the pseudo code for Pass@k Training with full sampling (Algorithm 1), bootstrap sampling (Algorithm 2), and analytical derivation (Algorithm 3).

Algorithm 1: The Pseudo Code for Pass@k Training with Full Sampling. Input : A tensor of reward R ∈ R𝑁rollout of the responses for the problem, the number of the rollouted responses

𝑁rollout, and the 𝑘 for Pass@k metric.

Output: A tensor of estimated advantages of the responses for this problem Aˆ ∈ R𝑁rollout.

- 1 # Construct the groups and discard the redundant instances.
- 2 Separate R ∈ R𝑁rollout into ⌊ 𝑁rollout𝐾 ⌋ group and each group contains 𝑘 instances.

- 3 Compute the reward of the groups Rgroup ∈ R⌊

𝑁rollout

𝐾 ⌋ using Eq. 5.

- 4 # Follow GRPO advantage estimation method to compute group-relative advantage.
- 5 Compute the average reward of the groups 𝑅¯group using Eq. 1.
- 6 Compute the standard deviation of the groups 𝜎group using Eq. 2.
- 7 Based on 𝑅¯group and 𝜎group, compute the group-relative advantage Aˆgroup using Eq. 3.
- 8 # Compute response-relative advantage.
- 9 Assign the Aˆgroup to the responses that the group contains, obtaining the response-relative advantage Aˆ.

Algorithm 2: The Pseudo Code for Pass@k Training with Bootstrap Sampling. Input : A tensor of reward R ∈ R𝑁rollout of the responses for the problem, the number of the rollouted responses

𝑁rollout, and the 𝑘 for Pass@k metric.

Output: A tensor of estimated advantages of the responses for this problem Aˆ ∈ R𝑁rollout.

- 1 # Construct the groups through bootstrap sampling.
- 2 for 𝑖 from 1 to 𝑁group do

- 3 Randomly sample 𝑘 instances from R to construct the 𝑖-th group.
- 4 Compute the reward of 𝑖-th group using Eq. 5.

- 5 Obtain the reward of the groups Rgroup ∈ R𝑁group.
- 6 # Follow GRPO advantage estimation method to compute group-relative advantage.
- 7 Compute the average reward of the groups 𝑅¯group using Eq. 1.
- 8 Compute the standard deviation of the groups 𝜎group using Eq. 2.
- 9 Based on 𝑅¯group and 𝜎group, compute the group-relative advantage Aˆgroup using Eq. 3.
- 10 # Calculate response-relative advantage.
- 11 Based on Aˆgroup, compute response-relative advantage Aˆ using Eq. 6.

### D Curves of Advantage Function

We present the curves of the advantage function of different training approaches in Figure 18, including Pass@k Training w/o easy problems, Pass@k Training w/ combination, exceeding Pass@k Training, and combination training.

### E Experiments on Various LLMs and Tasks

In this section, to further verify the effectiveness of Pass@k Training, we provide the performance of various LLMs trained through Pass@k Training on Mathematical Tasks (i.e., AIME 2024 [2], AIME 2025 [3], and OlymMATH [41]) and Synthetic Puzzle Task (i.e., Enigmata [7]).

Algorithm 3: The Pseudo Code for Pass@k Training with Analytical Derivation. Input : A tensor of reward R ∈ R𝑁rollout of the responses for the problem, the number of the rollouted responses

𝑁rollout, and the 𝑘 for Pass@k metric.

Output: A tensor of estimated advantages of the responses for this problem Aˆ ∈ R𝑁rollout.

- 1 # Calculate the average and standard deviation of the group reward scores.
- 2 Compute the average reward of the groups 𝑅¯group using Eq. 11.
- 3 Compute the standard deviation of the groups 𝜎group using Eq. 12.
- 4 # Calculate response-relative advantage.
- 5 Compute the advantage of the positive responses 𝐴ˆpos using Eq. 14.
- 6 Compute the advantage of the negative responses 𝐴ˆneg using Eq. 15.
- 7 Based on 𝐴ˆpos, 𝐴ˆneg, and R, assign the advantage to each instance, obtaining response-relative advantage Aˆ.

#### E.1 Pass@k Training on Mathematical Tasks

We follow the experiment settings described in Appendix A.2 to perform Pass@k Training on LLaMA models [14] (i.e., LLaMA3.2-3B-Instruct and LLaMA3.1-8B-Instruct) and DeepSeek-R1-Distill-Qwen [13] (i.e., 1.5B and 7B version). For LLaMA models, we set the maximum prompt length and response length as 2048 and 6144, respectively. For DeepSeek-R1-Distill-Qwen, we extend the response length to 10240. Specifically, to adapt the LLMs to the mathematical tasks, we adopt the training data used in previous work [9] during the RLVR training process. Besides, we follow the settings in Appendix A.2 to perform the evaluation, and present the results in Table 5. Since the single turn of Pass@k Training followed by Pass@1 Training can significantly improve the Pass@1 performance of LLMs, we conduct the experiment about multiple turns of the above training process in Table 5, named as “(P@k T. + P@1 T.) × 2”.

- Table 5 Pass@1/Pass@k Performance on mathematical tasks of LLaMA and DeepSeek-R1-Distill-Qwen models trained through different RLVR approaches. “P@1 T.” and “P@k T.” denote the Pass@1 Training and Pass@k Training with analytical derivation, respectively. “(P@k T. + P@1 T.) × 2” refers to that the process of Pass@k Training followed by Pass@1 Training is repeated twice.

AIME 2024 AIME 2025 OlymMATH-Easy OlymMATH-Hard Avg. RLVR on LLaMA3.2-3B-Instruct (Pass@1/Pass@k) Baseline 1.5/17.3 0.1/2.1 1.7/14.4 1.1/9.2 1.1/10.8 + P@1 T. 13.6/26.7 1.1/6.6 3.8/4.0 2.0/6.3 5.1/10.9 + P@k T. 12.7/32.0 1.7/12.9 3.7/8.8 1.7/7.7 5.0/15.4 + P@k T. + P@1 T. 14.6/32.1 1.3/8.6 4.1/7.7 2.0/7.5 5.5/14.0 RLVR on LLaMA3.1-8B-Instruct (Pass@1/Pass@k)

Baseline 3.4/17.9 0.2/4.3 0.8/7.5 0.5/7.6 1.0/9.3

+ P@1 T. 4.4/32.1 0.9/7.7 1.4/4.1 1.1/6.2 2.0/12.5 + P@k T. 7.1/40.0 1.8/10.6 1.5/8.9 1.4/8.2 3.0/17.0 + P@k T. + P@1 T. 8.7/29.7 0.9/8.7 1.8/7.9 1.6/6.8 3.3/13.3

RLVR on DeepSeek-R1-Distill-Qwen-1.5B (Pass@1/Pass@k) Baseline 22.7/61.4 20.5/37.2 6.6/36.7 0.6/5.2 12.6/35.1 + P@1 T. 36.7/76.0 28.8/49.4 16.7/51.7 2.5/17.5 21.2/48.7 + P@k T. 36.5/79.3 27.0/55.5 17.6/59.3 2.4/17.4 20.9/52.9 + P@k T. + P@1 T. 42.3/71.7 30.4/57.8 20.7/60.5 3.4/18.9 24.2/52.2 + (P@k T. + P@1 T.) × 2 44.2/77.2 31.5/57.7 22.6/62.7 4.4/21.2 25.7/54.7

RLVR on DeepSeek-R1-Distill-Qwen-7B (Pass@1/Pass@k) Baseline 43.2/80.9 31.5/59.1 22.2/66.0 1.4/13.7 24.6/54.9 + P@1 T. 48.5/79.5 35.5/59.4 27.9/69.1 3.1/22.5 28.8/57.6 + P@k T. 48.2/80.9 36.5/66.7 28.1/72.7 3.3/23.3 29.0/60.9 + P@k T. + P@1 T. 50.3/81.0 39.3/61.9 32.3/68.9 3.5/22.7 31.4/58.6

#### E.2 Pass@k Training on Enigmata Task

We follow the experiment settings described in Appendix A.2 to perform Pass@k Training on various LLMs (i.e., LLaMA3.2-3B-Instruct [14] and LLaMA3.1-8B-Instruct [14]), and set the maximum of the prompt length and response length as 4096 and 4096, respectively. The results are presented in Table 6. For evaluation, we follow the settings described in Appendix A.2.

- Table 6 Enigmata Pass@1/Pass@k Performance of LLaMA models trained on different RLVR approaches. “P@1 T.” and “P@k T.” denote the Pass@1 Training and Pass@k Training with analytical derivation, respectively.

Crypto Arithmetic Logic Grid Graph Search Sequential Overall RLVR on LLaMA3.2-3B-Instruct (Pass@1/Pass@k)

Baseline 0.0/0.0 0.2/1.6 19.3/44.3 2.7/4.7 1.7/8.8 5.4/11.1 0.4/1.8 3.1/7.3

+ P@1 T. 0.0/0.0 0.2/1.3 19.7/27.0 17.4/18.0 5.3/12.8 12.9/14.5 9.8/10.7 11.1/13.0 + P@k T. 0.0/0.0 0.2/0.7 22.0/31.0 17.3/18.1 6.1/14.2 12.2/16.6 10.7/12.0 11.5/14.1 + P@k T. + P@1 T. 0.0/0.0 0.4/2.2 22.8/27.7 16.7/17.4 6.5/13.5 14.6/16.0 12.0/13.0 12.3/14.0

RLVR on LLaMA3.1-8B-Instruct (Pass@1/Pass@k)

Baseline 0.0/0.0 0.1/1.1 21.6/41.7 3.7/4.6 1.5/7.8 6.0/17.0 1.2/5.0 3.8/9.0

+ P@1 T. 0.0/0.0 0.1/0.9 29.2/38.0 12.1/13.2 3.8/8.5 12.1/14.7 5.3/7.4 8.7/11.1 + P@k T. 0.0/0.0 0.1/0.9 30.5/39.3 12.9/14.8 5.5/12.2 12.2/14.7 7.5/10.9 9.9/13.0 + P@k T. + P@1 T. 0.0/0.0 0.2/1.1 34.4/44.7 12.5/14.2 7.5/17.8 13.2/15.7 8.7/10.3 10.8/13.7

E.3 Pass@k Training on Maze Task

In this part, we present the full results of Pass@k Training on the Maze task in Table 7. Without any RLVR training, it is really difficult for the model to solve the Maze task. Thus, we do not report the performance of the backbone model.

- Table 7 The Pass@1/Pass@k performance of Qwen2.5-7b-Instruct trained on different approaches on various Maze sizes. “P@1 T.” and “P@k T.” denote the Pass@1 Training and Pass@k Training with analytical derivation, respectively. “FS”, “BS”, and “AD” denote the full sampling, bootstrap sampling, and analytical derivation, respectively.

7 × 7 9 × 9 11 × 11 13 × 13 15 × 15 17 × 17 19 × 19 21 × 21 Avg.

+ P@1 T. 36.0/38.2 32.4/33.0 10.6/11.0 14.0/14.0 8.1/9.0 5.0/5.0 2.0/2.0 3.0/3.0 13.9/14.4 + P@k T. w/ FS 34.6/67.4 26.4/47.6 13.7/26.0 11.0/18.5 8.6/17.6 3.0/7.6 2.2/7.9 1.9/5.6 12.7/24.7 + P@k T. w/ BS 45.3/70.6 37.8/51.0 15.4/27.0 12.8/20.7 12.3/19.9 3.3/8.8 4.8/9.0 2.3/6.3 16.8/26.7 + P@k T. w/ AD 86.8/98.2 94.6/100.0 75.2/98.3 55.2/84.6 39.2/72.0 10.5/29.2 16.7/28.3 3.5/9.7 47.7/65.0

[Figure 31]

[Figure 32]

(a) Pass@k Training w/o Easy Problems. (b) Pass@k Training w/ Combination.

[Figure 33]

[Figure 34]

(c) Exceeding Pass@k Training. (d) Combination Training.

Figure 18 The curves of the advantage function on the setting of 𝑁rollout = 32.

