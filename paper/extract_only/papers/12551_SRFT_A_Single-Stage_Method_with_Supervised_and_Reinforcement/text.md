arXiv:2506.19767v1[cs.CL]24Jun2025

SRFT: A SINGLE-STAGE METHOD WITH SUPERVISED AND REINFORCEMENT FINE-TUNING FOR REASONING

Yuqian Fu♠♣∗, Tinghong Chen♠♣∗, Jiajun Chai♢, Xihuai Wang♡, Songjun Tu♠♣, Guojun Yin♢, Wei Lin♢, Qichao Zhang♠♣, Yuanheng Zhu♠♣†, Dongbin Zhao♠♣† ♠ Institute of Automation, Chinese Academy of Sciences

♣ School of Artificial Intelligence, University of Chinese Academy of Sciences

♢ Meituan ♡ Shanghai Jiao Tong University {fuyuqian2022,yuanheng.zhu}@ia.ac.cn

ABSTRACT

Large language models (LLMs) have achieved remarkable progress in reasoning tasks, yet the optimal integration of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) remains a fundamental challenge. Through comprehensive analysis of token distributions, learning dynamics, and integration mechanisms from entropy-based perspectives, we reveal key differences between these paradigms: SFT induces coarse-grained global changes to LLM policy distributions, while RL performs fine-grained selective optimizations, with entropy serving as a critical indicator of training effectiveness. Building on these observations, we propose Supervised Reinforcement Fine-Tuning (SRFT), a single-stage method that unifies both fine-tuning paradigms through entropy-aware weighting mechanisms. Our approach simultaneously applies SFT and RL to directly optimize the LLM using demonstrations and self-exploration rollouts rather than through twostage sequential methods. Extensive experiments show that SRFT achieves 59.1% average accuracy, outperforming zero-RL methods by 9.0% on five mathematical reasoning benchmarks and 10.9% on three out-of-distribution benchmarks.

Project Website: https://anonymous.4open.science/w/SRFT2025 Model Website: https://huggingface.co/Yuqian-Fu/SRFT

- 1 INTRODUCTION

Recent advances in Large Language Models (LLMs) for reasoning (OpenAI, 2025; Guo et al., 2025; Anthropic, 2025) have demonstrated remarkable capabilities in complex problem-solving tasks. Despite these remarkable achievements, fine-tuning strategies for enhancing reasoning capabilities remain an active area of research, presenting both opportunities and challenges.

Initial approaches often treat Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) as distinct, sequential phases. For instance, SFT might be used for instruction-following, followed by RL for alignment. However, this separation presents challenges: SFT can lead to models that memorize patterns without developing true reasoning abilities, potentially overfitting the training dataset (Chu et al., 2025; Chen et al., 2025a). Conversely, RL methods, while promising for exploration and reward optimization, can be sample inefficient, struggle with effective exploration in vast solution spaces (Gao et al., 2025; Dou et al., 2025; Schmied et al., 2025), or suffer from issues like mode collapse, where the model repeatedly generates similar, suboptimal outputs (Cai et al., 2025).

Instead of simple sequential approaches, recent work (Yan et al., 2025; Wu et al., 2025; Liu et al., 2025a; Chen et al., 2025b; Liu et al., 2025b) has shown a movement towards integrated frameworks that unify SFT and RL paradigms, or dynamically switch between the two fine-tuning methods during the LLM training process. As illustrated in Figure 1(a), SFT guides LLM policies toward demonstration distributions, while RL enables policies to explore improved solutions in the neighborhood of the base policy. Our illustration demonstrates a special case: when the base policy is positioned near a suboptimal policy, the RL rollouts alone cannot effectively navigate to the optimal policy. Beyond applying SFT and RL individually, the unified integration of SFT and RL within a single-stage method

∗Equal contribution. †Corresponding authors. ‡Work in progress.

Fit Demonstrations Fit & Explore Explore Neighborhood

Rollout Buffer

Demonstration

Reward

Reward

Reward

Policy

Policy

Policy

###### RFT SFT&RFT

q ModelPolicy RewardModel Rollout(s)

Response

Response Response

Update Policy Model

SFT SRFT RL

with Entropy Indicator

SFT Policy SRFT Policy RL Policy Base Policy Demonstration/ RL Rollout Reward

* For clarity, we use Gaussian distributions to represent the complex LLM distributions.

(a) Toy illustration of SFT, RL, and SRFT for LLM reasoning training on a single prompt.

(b) Framework of SRFT. Our method effectively leverages demonstrations to improve reasoning capabilities.

Figure 1: Overview of SRFT’s motivation and framework.

(e.g., our proposed SRFT) enables policies to directly optimize toward better solutions across an expanded space. However, a challenge remains in determining the balance between SFT’s knowledge distillation and RL’s policy optimization: insufficient integration risks error propagation and limits RL’s improvements, while excessive reliance on demonstrations leads to overfitting that constrains exploration beyond the base policy distribution. This trade-off creates confusion for practitioners in choosing between SFT for leveraging demonstrations and RL for policy exploration.

To address these issues, in this work, we study how to build single-stage LLM fine-tuning algorithms that are not only effective for LLM reasoning from SFT datasets but also well-suited to continuous improvement with RL rollouts. We conduct a comprehensive analysis of the roles that SFT and RL play in fine-tuning LLM reasoning. Through our analysis in Sec. 3, we obtain the following key findings that guide our subsequent algorithm design.

Key Findings

- ◦ Policy distribution effects (Sec. 3.1.1 and Sec. 3.1.2): During fine-tuning, SFT induces coarse-grained global changes to the LLM’s policy distribution, while RL performs finegrained selective modifications.
- ◦ Single-stage optimization (Sec. 3.1.2 and Sec. 3.2.2): Single-stage integration of SFT and RL enables direct optimization for reasoning capabilities and achieves superior training efficiency compared to sequential SFT→RL approaches.
- ◦ Entropy as an indicator (Sec. 3.2.1): Entropy dynamics reveal the underlying mechanisms of training processes, enabling balanced weighting between the two paradigms.

Based on these insights, we propose Supervised Reinforcement Fine-Tuning (SRFT), a single-stage method for LLM reasoning. As shown in Figure 1(b), we integrate SFT into RL and use entropy as an indicator to control the balance between these two paradigms. Specifically, for samples generated by LLM policy rollouts, we employ different RL training losses based on whether sample rewards are positive or negative. For samples from demonstration datasets, we simultaneously apply both SFT and RL objectives. This unified approach enables stable learning from demonstrations at multiple granularities while effectively bridging the complementary strengths of SFT and RL.

We evaluate our method on five competition-level mathematical reasoning benchmarks and three out-of-distribution (OOD) benchmarks. Our proposed SRFT achieves an accuracy of 59.1% based on Qwen-2.5-Math-7B (Yang et al., 2024), outperforming previous SFT and RL baselines by significant margins. Moreover, SRFT demonstrates superior generalization capability, achieving an average improvement of over 4.7% compared to other methods utilizing demonstrations.

## Overall, our key contributions are:

- • We conduct a comprehensive analysis of SFT and RL in LLM reasoning, examining their differential effects on policy distributions and learning dynamics. Besides, we analyze the integration of SFT and RL through an entropy-based lens.
- • We propose SRFT, a single-stage fine-tuning approach that combines supervised fine-tuning and reinforcement learning with entropy-aware weighting mechanisms, enabling effective utilization of demonstrations while maintaining stable exploration dynamics.
- • We demonstrate SRFT’s superior performance across eight challenging benchmarks, achieving substantial improvements of 9.0% and 10.9% over zero-RL baselines on mathematical reasoning and out-of-distribution tasks, respectively.

- 2 PRELIMINARIES

- 2.1 SFT AND RL FOR LLM REASONING

Supervised Fine-Tuning (SFT) is a standard approach for adapting pre-trained language models to specific downstream tasks or imparting particular stylistic characteristics. Given a dataset D = {(xi,yi)}Ni=1, where xi is an input prompt and yi is the corresponding target response generated by the behavior policy πβ, the objective is to train the language model policy πθ (with parameters θ) to maximize the conditional probability of generating the target response yi given xi. This is typically achieved by minimizing the negative log-likelihood over the dataset:

LSFT(θ) = E(x,y)∈D[−log πθ(y|x)], (1) where yj is the j-th token in the response y, and y<j denotes the sequence of tokens in y before yj. Reinforcement Learning (RL) is typically applied after SFT to further align LLMs with complex human preferences or desired behaviors (e.g., reasoning abilities, harmlessness) that are challenging to specify exhaustively through static datasets. In RL training, the LLM’s token generation process is modeled as a Markov Decision Process (MDP) (Puterman, 2014). We define a state st at step t as the concatenation of the input prompt x and all tokens generated so far y<t. This state serves as input to the policy model πθ(·|st). Specifically, the policy processes st = (x,y<t) = (x1,x2,...,xl,y1,y2,...,yt−1), where xi denotes the i-th token of the input x and yj represents the token generated by πθ at step j. An action at corresponds to the selection of the next output token yt. The LLM, acting as a policy πθ(at|st), generates a trajectory y (a sequence of tokens) in response to the prompt x. A reward function R(x,y) = Tt=1 r(x,yt) provides a scalar score for the entire trajectory y given prompt x, typically derived from human evaluations or automated metrics. In the context of RL, the behavior policy πβ(y|x) refers to the model that generated the responses in the replay buffer. This policy is crucial for RL, particularly for off-policy learning, as it enables proper importance sampling corrections to account for the distribution shift between the data-generating model and the current training model. The MDP formulation in LLMs presents several notable characteristics:

- • Sequential state representation: At each step t, the state st ∈ S consists of the concatenation of the input prompt x and all actions (tokens) generated so far y<t. This state serves as input to the policy model πθ(·|st).
- • Sparse and delayed rewards: Rewards R(x,y) are typically sparse, provided only upon completion of a sequence y. This dependency on the final output’s overall quality complicates credit assignment across the generation process.

- 2.2 POLICY OPTIMIZATION IN REINFORCEMENT LEARNING

To optimize the LLM policy, Group Relative Policy Optimization (GRPO) (Shao et al., 2024) offers a different RL algorithm that is presented as a memory-efficient variant of Proximal Policy Optimization (PPO) (Schulman et al., 2017). A key characteristic is that GRPO typically operates without a learned value function. Instead, for a given prompt x, it often generates a group of G responses {y1,...,yG} using the current policy. The rewards {R(x,y1),...,R(x,yG)} for these responses are then used to compute a relative advantage for each response:

R(x,yk) − mean({R(x,yk)|k = 1,2,...,G}) std({R(x,yk)|k = 1,2,...,G})

Aˆk =

. (2)

Then, GRPO maximizes a clipped surrogate objective function to ensure stable updates. Let πθ

be the policy before the update. For each token yk,t in a trajectory yk (from state st), the importance sampling ratio is rk,t(θ) = ππθ(yk,t|st)

old

θold(yk,t|st). The objective function for GRPO can then be expressed as:

|yk|

G

1 G

1 |yk|

min rk,t(θ) · Aˆk,clip{rk,t(θ),1 − ϵ,1 + ϵ} · Aˆk , (3)

JGRPO(θ) =

t=1

k=1

where ϵ is a small hyperparameter that defines the clipping range. Through this mechanism, the LLM policy is updated while maintaining stable gradient constraints.

###### Token Probability Visualization

Generate Model: Qwen2.5-Math-7B-curr-part2 Reference Model: Qwen2.5-Math-7B (base)

Lower Probability After LUFFY Similar Probability Higher Probability After LUFFY

<think> Okay, let's see. So Aya has a 9-kilometer walk every morning, and then she stops at a coffee shop for t minutes. The problem gives two different scenarios where her walking speed changes, and the total time (including the coffee shop break) is different. Then we need to find out how long it takes her when she walks at s + 1/2 kilometers per hour, including the same t minutes in the coffee shop. First, let me parse the information given. When she walks at speed s km/h, the total time is 4 hours, which includes t minutes in the coffee shop. Omit part of the

output... Therefore, her walking speed is 2.5 km/h. Then, the question is, when she walks at s + 1/2 km/h, which is 2.5 + 0.5 = 3 km/h, how long does the walk take, including t minutes? So first, we need to find t. Once we have t, we can compute the total time as walking time + t. But we can find t from the first equation: 9/s + t/60 = 4 We know s = 2.5, so: 9 / 2.5 + t / 60 = 4 Compute 9 / 2.5: 9 divided by 2.5 is 3.6 So 3.6 + t/60 = 4 Subtract 3.6: t/60 = 0.4 Multiply both sides by 60: t = 24 minutes So t is 24 minutes. Therefore, when walking at 3 km/h, the walking time is 9 / 3 = 3 hours, which is 180 minutes. Then adding t minutes (24 minutes) gives total time of 180 + 24 = 204 minutes. But the question asks for the number of minutes the walk takes her, including t minutes spent in the coffee shop. So the answer is 204 minutes. Wait, but let me check with the second scenario to make sure. In the second scenario, walking speed s +

- 2 = 2.5 + 2 = 4.5 km/h. So walking time is 9 / 4.5 = 2 hours. Then adding t = 24 minutes gives 2 hours and 24 minutes, which is 2.4 hours, which matches the given total time. So that checks out. So when she walks at 3 km/h, walking time is 9 / 3 = 3 hours, which is 180 minutes. Adding 24 minutes gives 204 minutes total. So the answer is 204 minutes. But let me just verify the quadratic equation step again. When we set up 9/s - 9/(s + 2) = 1.6, which is the difference in walking time. Then solving gives s = 2.5. Then t is found from first equation: 9/s + t/60 =

- 4. 9/2.5 is 3.6, so t = (4 - 3.6)*60 = 0.4*60 = 24. Correct. Then 9/(s + 0.5) = 9/3 = 3 hours = 180 minutes, plus t = 24 minutes, total 204 minutes. So 204 minutes is the answer. But wait, the problem says "the number of minutes the walk takes her, including the t minutes spent in the coffee shop". So the answer is 204 minutes. So boxed answer would be \boxed{204}

**Final Answer** \boxed{204} </think> Given that Aya walks 9 kilometers every morning and stops at a coffee shop for \( t \) minutes, we need to find the total time taken when she walks at \( s

+ \frac{1}{2} \) kilometers per hour, including the same \( t \) minutes in the coffee shop. First, we set up the equations for the two scenarios given: 1. When walking at \( s \) km/h, the total time (including \( t \) minutes) is 4 hours: \[ \frac{9}{s} + \frac{t}{60} = 4 \] 2. When walking at \( s + 2 \) km/h, the total time (including \( t \) minutes) is 2 hours and 24 minutes (which is 2.4 hours): \[ \frac{9}{s + 2} + \frac{t}{60} = 2.4 \] Subtracting the second equation from the first to eliminate \( t \): \[ \frac{9}{s} - \frac{9}{s + 2} = 1.6 \] Simplifying this equation: \[ 9 \left( \frac{1}{s} - \frac{1}{s + 2} \right) = 1.6 \] \[ 9 \left( \frac{2}{s(s + 2)} \right) = 1.6 \] \[ \frac{18}{s(s + 2)} = 1.6 \] \[ s(s + 2) = \frac{18}{1.6} = 11.25 \] Solving the quadratic equation \( s^2 + 2s - 11.25 = 0 \) using the quadratic formula: \[ s = \frac{-2 \pm \sqrt{4 + 45}}{2} = \frac{-2 \pm 7}{2} \] Taking the positive root, we get \( s

= 2.5 \) km/h. Next, we find \( t \) using the first equation: \[ \frac{9}{2.5} + \frac{t}{60} = 4 \] \[ 3.6 + \frac{t}{60} = 4 \] \[ \frac{t}{60} = 0.4 \] \[ t = 24 \text{ minutes} \] When walking at \( s + \frac{1}{2} = 3 \) km/h, the walking time is: \[ \frac{9}{3} = 3 \text{ hours} = 180 \text{ minutes} \] Adding the coffee shop time \( t = 24 \) minutes, the total time is: \[ 180 + 24 = 204 \text{ minutes} \] Thus, the number of minutes the walk takes her, including the \( t \) minutes spent in the coffee shop, is \(\boxed{204}\).

Token Probability Visualization

Generate Model: Qwen2.5-Math-7B-curr-part2 Reference Model: Qwen2.5-Math-7B (base)

Lower Probability After LUFFY Similar Probability Higher Probability After LUFFY

To solve this problem, we need to determine the values of \(s\) and \(t\) first. We know the following: 1. When Aya walks at a speed of \(s\) km/h, the total time including the coffee shop is 4 hours. So, the time spent walking is \(4 - \frac{t}{60}\) hours. The distance walked is 9 km, so we can write the equation: \[ s \left(4 - \frac{t}{60}\right) = 9 \] 2. When Aya walks at a speed of \(s + 2\) km/h, the total time including the coffee shop is 2 hours and 24 minutes, which is 2.4 hours. So, the time spent walking is \(2.4 - \frac{t}{60}\) hours. The distance walked is still 9 km, so we can write the equation: \[ (s + 2) \left(2.4 - \frac{t}{60}\right) = 9 \] We now have a system of two equations: \[ s \left(4 - \frac{t}{60}\right) = 9 \quad \text{(1)} \] \[ (s + 2) \left(2.4 - \frac{t}{60}\right) = 9 \quad \text{(2)} \] Let's solve these equations step by step. First, we can express \(t\) from equation (1): \[ t = 60 \left(4s 9\right) / s \] Next, we substitute \(t\) into equation (2): \[ (s + 2) \left(2.4 - \frac{60 (4s - 9)}{60s}\right) = 9 \] \[ (s + 2) \left(2.4 - \frac{4s - 9}{s}\right) = 9 \] \[ (s + 2) \left(2.4 - 4 + \frac{9}{s}\right) = 9 \] \[ (s + 2) \left(-1.6 + \frac{9}{s}\right) = 9 \] \[ -1.6s - 3.2 + 9 + \frac{18}{s} = 9 \] \[ -1.6s + \frac{18}{s} = 3.2 \] Multiplying through by \(s\) to clear the fraction: \[ -1.6s^2 + 18 = 3.2s \] \[ 1.6s^2 + 3.2s - 18 = 0 \] Dividing the entire equation by 1.6: \[ s^2 + 2s - 11.25 = 0 \] We can solve this quadratic equation using the quadratic formula \(s = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\), where \(a = 1\), \(b = 2\), and \(c = -11.25\): \[ s = \frac{-2 \pm \sqrt{4 + 45}}{2} = \frac{-2 \pm \sqrt{49}}{2} = \frac{-2 \pm 7}{2} \] This gives us two solutions: \[ s = \frac{5}{2} = 2.5 \quad \text{or} \quad s = \frac{-9}{2} = -4.5 \] Since speed cannot be negative, we have \(s = 2.5\) km/h. Now we can find \(t\): \[ t = 60 \left(4 \cdot 2.5 - 9\right) / 2.5 = 60 \left(10 - 9\right) / 2.5 = 60 / 2.5 = 24 \text{ minutes} \] Now, we need to find the time it takes for Aya to walk at a speed of \(s + \frac{1}{2} = 2.5 + 0.5 = 3\) km/h. The time spent walking is: \[ \text{Time} = \frac{9}{3} = 3 \text{ hours} \] Including the 24 minutes spent in the coffee shop, the total time is: \[ 3 \text{ hours} + 24 \text{ minutes} = 3 \times 60 + 24 = 180 + 24 = 204 \text{ minutes} \] So, the number of minutes the walk takes her, including the \(t\) minutes spent in the coffee shop, is \(\boxed{204}\).

SFT

RL

Question: Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.

Lower Probability After Fine-tuning Similar Probability Higher Probability After Fine-tuning

(a) (b)

| | | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

Figure 2: Visualization of LLM distribution changes during fine-tuning. (a) Heatmap visualization comparing responses generated by fine-tuned and base models, where darker background colors indicate larger probability changes. (b) Distribution of token probability changes across five mathematical reasoning benchmarks.

- 3 ANALYSIS OF SFT AND RL IN LLM REASONING

In this section, we provide a comprehensive analysis of the roles of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) in LLM reasoning. We first examine their differential effects on token distributions (Sec. 3.1.1) and learning dynamics (Sec. 3.1.2), then investigate their integration mechanisms through an entropy-based perspective (Sec. 3.2). All experiments are conducted across five mathematical reasoning benchmarks (AIME24, AMC, MATH500, Minerva, and Olympiad) with results averaged. We tune hyperparameters for all baseline methods to ensure fair and optimal performance comparisons.

- 3.1 SFT AND RL EFFECTS ON LLMS: SLEDGEHAMMER VS. SCALPEL

- 3.1.1 EFFECTS ON TOKEN DISTRIBUTIONS

To understand the differential impact of SFT and RL for reasoning, we visualize probability changes across response tokens to identical prompts before and after fine-tuning the same base model (Qwen-2.5-Math-7B). As illustrated in Figure 2(a), the results reveal a fundamental asymmetry: SFT substantially alters the probability distribution across the entire response sequence, while RL selectively modifies probabilities for only a small subset of tokens, while leaving numerical content and mathematical proof statements largely unchanged. We further quantify these distribution shifts across five benchmarks, as shown in Figure 2(b). The results demonstrate that SFT produces more pronounced changes to policy distributions compared to RL, with token probability changes in RL clustering near zero while SFT exhibits substantially larger magnitude shifts. From a theoretical perspective, this behavior can be understood through the gradient of the SFT objective function:

 

 , (4)

|y|

∇θLSFT = E(x,y)∼D

(πθ(v|x,y<t) − 1v=y

)∇θ log πθ(v|x,y<t)

t

t=1 v∈V

where V is the LLM vocabulary, and 1v=y

is an indicator function that equals 1 when token v

t

matches the target token yt and 0 otherwise. The detailed derivation is provided in Appendix D. This formulation reveals that SFT systematically sharpens the model distribution by increasing probabilities for target tokens while decreasing probabilities for all other tokens in the vocabulary, leading to more deterministic outputs.

- Table 1: Performance comparison of SFT and RL integration strategies across multiple benchmarks. Bold and underlined indicate the best and second-best performance, respectively.

Model AIME24 AMC MATH500 Minerva Olympiad Avg.

Qwen2.5-Math-7B 14.1 44.8 64.8 16.5 29.6 34.0 SFT 21.2 53.2 83.0 37.1 42.2 47.3 RL 21.2 59.3 83.6 36.4 46.6 49.4 RL→SFT 10.5 40.4 73.6 32.0 30.7 37.4 RL→SFTKL 13.1 45.2 70.2 26.5 36.3 38.3 SFT→RL 24.5 59.3 86.4 39.3 53.1 52.5

- 3.1.2 VISUALIZATION OF LEARNING DYNAMICS

Beyond the token probability analysis, we analyze the training paradigms from the perspective of learning dynamics. Since directly measuring the LLM feature space is computationally intractable, we propose a novel visualization approach that maps each model to a point in the vocabulary probability space, treating models as functions that transform prompts to output probability distributions over the vocabulary. We establish three reference models—the base model (Qwen-2.5-Math-7B), DeepSeek-R1, and QwQ-32B (Team, 2025)—as coordinate frames, enabling indirect measurement of model evolution at different fine-tuning steps through the distance in probability space (two models are considered close if they assign similar output probabilities to all tokens across all prompts). Detailed methodology for this visualization is provided in Appendix E.

[Figure 1]

Figure 3: Learning dynamics during different fine-tuning paradigms in three-dimensional probability space. The number denotes the final performance of each training process.

The visualization is shown in Figure 3, which demonstrates that all fine-tuning paradigms exhibit performance improvements while simultaneously moving away from the base model space (Qwen-2.5-Math7B). Specifically, SFT exhibits greater distribution changes from the base model compared to RL and achieves higher performance, which further validates our observations in Sec. 3.1.1 that SFT induces larger changes to model distributions while RL fine-tuning within a neighborhood of the initialization point. We further analyze two integration approaches: the two-stage SFT→RL method and our proposed single-stage SRFT approach detailed in the following section. The results reveal that the learning dynamics of the two-stage SFT→RL method traverse from the post-SFT model toward a higher-performance region that paradoxically lies closer to the base model, suggesting that the initial SFT phase may induce excessive distributional deviation from the base model, thereby compromising the effectiveness of subsequent RL. In contrast, our single-stage method demonstrates more constrained yet targeted changes in the probability space, enabling more precise optimization compared to sequential integration approaches.

- 3.2 INTEGRATION OF SFT AND RL: FROM TWO-STAGE TO SINGLE-STAGE

- 3.2.1 SEQUENTIAL INTEGRATION ANALYSIS

In this section, we examine the integration of SFT and RL through the lens of entropy dynamics to understand their complementary roles in LLM fine-tuning. We begin by systematically analyzing two sequential integration approaches: SFT→RL and RL→SFT, as shown in Figure 4.

As demonstrated in Table 1 and Figure 4(a), applying SFT after RL consistently yields suboptimal performance across all benchmarks. To mitigate the detrimental policy shifts induced by RL→SFT, we introduced a KL divergence constraint (SFTKL, detailed in Appendix B) to regularize the distribution changes. However, even with this constraint, the performance improvements remained limited,

| |
|---|

| |
|---|

0.6

| |
|---|

| |
|---|

| |
|---|

| |
|---|

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

45

0.5

40

0.4

Accuracy

Entropy

35

0.3

30

| |
|---|

| |
|---|

| |
|---|

| |
|---|

25

0.2

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

| |
|---|

0.1

RL SFT SFT RL

| |
|---|

| |
|---|

| |
|---|

15

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

0.0

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Step

Step

(a) Reasoning capability comparison between SFT→RL and RL→SFT.

(b) Entropy dynamics of two different finetuning paradigms.

Figure 4: Comparison between SFT→RL and RL→SFT.

suggesting fundamental incompatibility in this ordering. In contrast, existing methods successfully achieve substantial performance gains through RL when applied after the base model SFT, as evidenced in Table 1. This asymmetric behavior reveals that the sequence of fine-tuning paradigms critically affects the final model performance, motivating our entropy-based analysis to understand the underlying mechanisms.

To understand this asymmetric behavior, we analyze the training dynamics of SFT and RL from an entropy perspective. As illustrated in Figure 4(b), policies after RL exhibit significantly lower entropy, approaching deterministic outputs. However, the distribution shift introduced by subsequent SFT causes a rapid increase in entropy (corresponding to the sharp performance drop in Figure 4(a)), followed by a gradual decline. Moreover, models after RL demonstrate a limited capacity for further learning through SFT, as evidenced by the entropy plateau occurring after approximately 90 training steps (Figure 4(b)). In contrast, base models undergoing SFT exhibit a brief initial entropy increase followed by a sustained decrease, ultimately yielding performance improvements. This distinct entropy trajectory suggests that while RL effectively enhances LLM performance, it simultaneously reduces the model’s plasticity—its capacity to adapt through subsequent training. These findings establish entropy as a crucial indicator for effective SFT and RL integration.

- 3.2.2 SINGLE-STAGE INTEGRATION ANALYSIS

Building upon the analysis above, we establish that the SFT→RL paradigm demonstrates superior suitability for LLM reasoning compared to RL→SFT. However, beyond these sequential integration approaches, we investigate a single-stage approach that directly unifies both paradigms (SFT+RL), with the combined objective LSFT+RL = LSFT + LRL. We conduct a preliminary experiment comparing pure RL, sequential SFT→RL with varying SFT steps, and single-stage SFT+RL, as illustrated in Figure 5. Our empirical findings reveal that the singlestage SFT+RL method achieves superior training efficiency compared to the sequential SFT→RL approach. Notably, we observe an intriguing phenomenon in models with extensive SFT pre-training (350 steps SFT followed by 150 steps RL): a transient performance degradation during the initial phases of RL. We attribute this behavior to two principal factors: First, SFT datasets derived from other models’ responses may not consistently represent optimal solutions, even when sourced from high-quality demonstrations, potentially leading to suboptimal policy learning during the SFT phase. Second, pure RL exhibits limited data efficiency due to its inability to effectively leverage demonstrations. In the sequential SFT→RL training paradigm, the RL phase may induce catastrophic forgetting of knowledge acquired during SFT (Cai et al., 2025), resulting in transient performance deterioration.

| |from SFT 150 steps<br><br>from SFT 350 steps| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Figure 5: Preliminary comparison across pure RL, sequential SFT→RL, and single-stage SFT+RL integration approaches.

In contrast, the single-stage SFT+RL method effectively leverages demonstrations through unified optimization. This approach enables direct policy optimization toward the target objective while preserving the knowledge distillation benefits of supervised learning from datasets. Importantly, both datasets utilization methods significantly outperform pure RL across all performance metrics.

- 4 METHOD

In this section, we present the Supervised and Reinforcement Fine-tuning (SRFT) algorithm, which integrates the advantages of Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) in a single-stage approach. Building upon the RL framework described in Sec. 2.2, SRFT incorporates flexible guidance from demonstrations, enabling the algorithm to harness the complementary strengths of both fine-tuning paradigms. The core innovation of SRFT lies in its single-stage learning mechanism: coarse-grained behavior policy approximation through SFT and fine-grained policy refinement through RL, both applied to demonstration data and self-generated trial-and-error data.

- 4.1 LEARNING FROM DEMONSTRATIONS

Given a dataset containing demonstrations Ddemo. (e.g., reasoning responses generated by DeepSeekR1), SRFT employs a dual-pronged strategy to effectively harness this valuable resource. First, we leverage SFT to perform a coarse-grained approximation of the behavior policy underlying the expert’s responses. The behavior policy πβ(y|x) captures the underlying generation patterns that produced these high-quality responses, which we seek to approximate through supervised learning:

Ldemo.SFT = E(x,y)∼D

[−log πθ(y|x)]. (5) Second, we adopt an off-policy RL approach similar to LUFFY (Yan et al., 2025) to perform finegrained learning of the behavior policy through RL. Specifically, we directly augment the LLM’s on-policy rollout group with demonstrations, creating a heterogeneous training batch:

demo.

### Gaug. = {(xi,yi)}|iG=1roll.| ∪ {(xj,yj)}j|G=1demo.|, (6)

where Groll. denotes the on-policy rollout group, Gdemo. denotes the demonstration group. The advantage estimation for the entire group is given by:

r(x,yk) − mean({r(x,yk)|k = 1,2,...,|Gaug.|}) std({r(x,yk)|k = 1,2,...,|Gaug.|})

Aˆk =

. (7)

Since responses generated by expert LLMs typically exhibit higher rewards, their inclusion increases the advantage estimation for the entire group as shown in Eq. (2), promoting optimistic exploration in the LLM policy through this mechanism.

To address the distribution mismatch between behavior policies πβ of demonstrations and the current training policy πθ identified in our analysis, we implement two key mitigation strategies:

- • For SFT on demonstrations, our entropy analysis in Sec. 3.2 demonstrates that entropy serves as a crucial indicator for effective SFT and RL integration. Motivated by this insight, we introduce an adaptive weighting mechanism that dynamically adjusts based on the current

policy entropy, employing wSFT = 0.5 ∗ stop_grad(exp(−H(πθ))) as the SFT weight, where stop_grad(·) denotes the gradient stopping operation. This entropy-aware mechanism ensures that when the policy exhibits high entropy (indicating uncertainty), the SFT training loss exerts diminished influence on the model updates, thereby mitigating performance degradation caused by distribution mismatch between the behavior policy of demonstrations and the current policy while still enabling effective behavior policy approximation:

Ldemo.SFT (θ) = wSFT · E(x,y)∼D

demo.

[−log πθ(y|x)]. (8)

- • For off-policy RL training, we introduce an importance sampling term similar to GRPO (Shao et al., 2024) and PPO (Schulman et al., 2017) to account for the distribution shift between the behavior policy and the current policy:

### Ldemo.RL (θ) = −E(x,y)∼D

min rk,t(θ) · Aˆk,clip{rk,t(θ),1 − ϵ,1 + ϵ} · Aˆk , (9)

demo.

πθ(yk,t|xt) πβ(yk,t|xt)

. (10)

rk,t(θ) =

Following practices established in recent work (Yan et al., 2025; Ma et al., 2025), we set the behavior policy πβ = 1 to avoid tokenization complexities that arise when aligning the current training policy with the behavior policy, thereby facilitating easy integration of off-the-shelf datasets without requiring recomputation of behavior policy probabilities. Additionally, we omit the clipping operation, as the standard clipping mechanism becomes imbalanced and potentially unstable when πβ = 1.

- 4.2 LEARNING FROM SELF-EXPLORATION

In addition to leveraging the demonstration data, SRFT enables the LLM policy to learn simultaneously from its own exploration rollouts. While traditional RL methods learn from both positive and negative samples generated during rollouts, we observe that under on-policy RL with binary rewards {1,−1}, the basic RL objective function can be naturally decomposed into two distinct components:

Lself-rolloutRL = −Ex∼D,y∼π

θ(·|x)[R(x,y)log πθ(y|x)]

= Ex∼D,y+∼πθ(·|x)[−log πθ(y+|x)]

Positive Sample 1

+Ex∼D,y−∼πθ(·|x)[log πθ(y−|x)]

Negative Sample 2

, (11)

where D denotes the RL training dataset, and y+ and y− represent the correct and incorrect responses, respectively. A critical insight emerges from this decomposition: the positive sample objective 1 exhibits structural similarity to supervised fine-tuning, as it maximizes the likelihood of correct responses. However, these positive samples are generated on-policy by the current policy πθ rather than sourced from SFT datasets, distinguishing our approach from conventional supervised learning paradigms. The negative reward component 2 implements likelihood minimization, systematically reducing the probability mass assigned to incorrect responses. This structural correspondence suggests that learning from positive samples constitutes a coarse-grained optimization strategy that necessitates careful balance. Moreover, in contrast to learning from demonstrations, self-exploration induces rapid entropy reduction as the model converges toward increasingly deterministic outputs, potentially compromising exploration capabilities. To mitigate this phenomenon and preserve training stability, inspired by our analysis in Sec. 3.1.1, we introduce an entropy-adaptive weighting mechanism wRL = 0.1 ∗ stop_grad(exp(H(πθ))) specifically for the positive sample objective. This mechanism is similar to our formulation in Eq. (8) but serves the complementary purpose of maintaining exploration diversity. The complete self-exploration objective is formulated as:

Lself-rolloutRL (θ) = wRL · Ex∼D,y+∼πθ(·|x)[−log πθ(y+|x)] + Ex∼D,y−∼πθ(·|x)[log πθ(y−|x)]. (12)

- 4.3 INTEGRATING DEMONSTRATIONS WITH SELF-EXPLORATION ROLLOUTS IN A SINGLE-STAGE APPROACH

By leveraging both demonstrations and self-generated rollouts, SRFT effectively balances the coarsegrained adjustments of SFT with the fine-grained refinements of RL throughout the single-stage fine-tuning process. The total loss function combines all four components:

LSRFT(θ) = Ldemo.SFT (θ) + Ldemo.RL (θ) + Lself-rolloutRL (θ). (13)

This objective enables SRFT to simultaneously benefit from demonstrations and self-exploration rollouts while maintaining stable training dynamics through two entropy-aware weighting mechanisms.

- 5 EXPERIMENTS.

- 5.1 EXPERIMENTAL SETUPS

Training Datasets. We employ OpenR1-Math-46k-81921 (Yan et al., 2025) as the training dataset for SRFT, which constitutes a subset of OpenR1-Math-220k (Face, 2025) comprising 46,000 mathematical problems sourced from NuminaMath 1.5 (LI et al., 2024), accompanied by high-quality reasoning responses generated by DeepSeek-R1 (Guo et al., 2025). The dataset undergoes filtering through Math-Verify2 to exclude instances with unverifiable answers or responses exceeding 8,192

- 1https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192
- 2https://github.com/huggingface/Math-Verify

tokens in length. This dataset serves multiple purposes in our framework: providing prompts for policy rollouts, ground-truth answers for reward computation, and high-quality demonstrations for SRFT. The dataset details are provided in Appendix C.

Evaluation. We conduct a comprehensive evaluation on several widely-adopted mathematical reasoning benchmarks, including AIME24 (Li et al., 2024), AMC (Li et al., 2024), Minerva (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), and MATH500 (Hendrycks et al., 2021). For datasets with limited sample sizes (AIME24 and AMC), we report the avg@32 metric; for the remaining three datasets, we adopt pass@1 as the evaluation criterion. Given that our method primarily focuses on mathematical reasoning capabilities, we further assess the model’s generalization ability on three out-of-distribution benchmarks: ARC-C (Clark et al., 2018) (open-domain reasoning), GPQA-Diamond (Rein et al., 2024) (graduate-level scientific knowledge, denoted as GPQA-D), and MMLU-Pro (Wang et al., 2024) (reasoning problems from academic examinations and textbooks). To mitigate potential information leakage, we randomly shuffle the option orders for all multiplechoice questions. During inference, we set the generation temperature to 0.6 with a maximum response length of 8,192 tokens. We employ Math-verify as the verifier for training validation and the OAT-Grader (Liu et al., 2024) for final evaluation.

Baseline Methods. We benchmark SRFT against the following baselines using Qwen2.5-Math-7B as the base model. SFT methods: (1) SFT on OpenR1-Math-46k-8192 dataset; (2) SFT training with KL divergence constraints incorporated into the loss function (SFTKL). RL methods: (3) RLGRPO (Shao et al., 2024), a simplified PPO variant trained on the same 46k dataset; (4) SimpleRL-Zero (Zeng et al., 2025), applying GRPO to approximately 24k mathematical samples from GSM8K and MATH; (5) OpenReasoner-Zero (Hu et al., 2025), a PPO-based approach trained on 129k multi-source samples including AIME; (6) PRIME-Zero (Cui et al., 2025), conducting policy rollouts on 150k NuminaMath queries with implicit process rewards and final labels. SFT and RL methods: (7) SFT→RL, sequential training with SFT the same 46k dataset followed by GRPO; (8) ReLIFT (Ma et al., 2025), an approach that interleaves RL with online Fine-Tuning on the hardest questions; (9) LUFFY (Yan et al., 2025), a mixed-policy GRPO approach using the same 46k dataset; (10) TAPO (Wu et al., 2025), dynamically integrating structured external knowledge within the GRPO framework, trained on 5.5k samples from the MATH dataset.

Implementation Details. Following recent work (Yan et al., 2025; Wu et al., 2025; Cui et al., 2025), we use the Qwen2.5-Math-7B (Yang et al., 2024) model as the base model. In SRFT, we generate 8 rollout trajectories per prompt with a maximum sequence length of 8,192 tokens. All experiments are conducted over 500 training steps. Comprehensive experimental details are provided in Appendix B.

5.2 EXPERIMENTAL RESULTS

Reasoning Benchmark Performance. Our main results are shown in Table 2, where we compare SRFT with several zero-RL baselines, as well as direct SFT, and SFT+RL methods. Across five challenging competition-level reasoning benchmarks, SRFT achieves an average score of 59.1, significantly outperforming existing RL methods by a margin of +9.0 points over the best baseline, clearly demonstrating the benefit of integrating demonstrations with self-exploration in LLM reasoning. We also observe that SRFT achieves a +4.8 points improvement over the SFT methods, indicating that the self-exploration component can effectively refine the policy distribution learned from demonstrations. Compared to the SFT+RL methods, SRFT achieves a +3.4 points improvement, demonstrating that the single-stage design and entropy-aware weighting mechanism can effectively balance the benefits of demonstrations and self-exploration.

Out-of-Distribution Generalization. Regarding out-of-distribution performance, the results in

- Table 2 show that SRFT also achieves an average score of 62.5 and outperforms the best baseline by +4.7 points. These results highlight SRFT’s effectiveness in combining demonstrations with self-exploration to improve generalization ability.

Training Dynamics. Figure 6 shows the training dynamics of SRFT, including training rewards, response lengths, and training entropy. As shown in Figure 6(a), SRFT achieves faster performance improvement compared to RL, with both SRFT and RL exhibiting an increasing trend in training

Table 2: Overall performance on five competition-level mathematical reasoning benchmarks and three out-of-distribution benchmarks based on Qwen2.5-Math-7B. Bold and underlined indicate the best and second-best performance, respectively.

In-Distribution Performance Out-of-Distribution Performance Model

AIME24 AMC MATH500 Minerva Olympiad Avg. ARC-C GPQA-D MMLU-Pro Avg. Qwen2.5-Math 11.4 32.6 48.8 8.7 15.8 23.5 18.2 11.1 16.9 15.4 Qwen2.5-Math-Instruct 12.9 48.3 81.2 33.1 39.8 43.1 70.3 24.7 34.1 43.0 Supervised Fine-Tuning

SFT 31.1 62.8 85.2 39.1 53.3 54.3 76.2 25.8 45.7 49.2 SFTKL 13.0 45.2 70.2 26.5 36.3 38.2 33.3 22.2 30.4 28.6 Reinforcement Learning

RLGRPO (Shao et al., 2024) 24.7 61.6 79.2 33.7 47.1 49.3 75.6 31.3 42.1 49.7 SimpleRL-Zero∗ (Zeng et al., 2025) 27.0 54.9 76.0 25.0 34.7 43.5 30.2 23.2 34.5 29.3 OpenReasoner-Zero∗ (Hu et al., 2025) 16.5 52.1 82.4 33.1 47.1 46.2 66.2 29.8 58.7 51.6 PRIME-Zero∗ (Cui et al., 2025) 17.0 54.0 81.4 39.0 40.3 46.3 73.3 18.2 32.7 41.4 Oat-Zero∗ (Zeng et al., 2025) 33.4 61.2 78.0 34.6 43.4 50.1 70.1 23.7 41.7 45.2 SFT and RL

SFT → RL 32.5 67.1 84.2 34.1 54.6 54.5 76.4 37.9 49.6 54.6 LUFFY (Yan et al., 2025) 29.4 65.6 87.6 37.5 57.2 55.5 80.5 39.9 53.0 57.8 TAPO∗ (Wu et al., 2025) 33.3 77.5 83.4 38.2 46.2 55.7 81.6 37.9 49.6 56.4 ReLIFT (Ma et al., 2025) 28.2 64.8 85.0 37.1 54.9 54.0 74.9 40.9 51.9 55.9 SRFT (ours) 35.3 74.3 89.8 39.7 58.3 59.5 85.3 46.4 55.9 62.5

∗This method’s performance is taken from the corresponding paper.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.7

OutcomeRewards

0.6

0.5

0.4

0.3

0 100 200 300 400 500

Step

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

3500

ResponseLength

3000

2500

2000

1500

1000

0 100 200 300 400 500

Step

| |SRFT<br><br>RL| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.5

0.4

Entropy

0.3

0.2

0.1

0.0

0 100 200 300 400 500

Step

(a) Training Rewards

(b) Response Lengths

(c) Training Entropy

Figure 6: Training dynamics during RL and SRFT training, including training rewards, response lengths, and training entropy.

rewards. In terms of response length, as shown in Figure 6(b), when faced with challenging training data, RL exhibits a tendency toward generating more concise responses, whereas SRFT shows a progressive lengthening of responses, indicating the development of more thorough and detailed reasoning processes. From an entropy perspective in Figure 6(c), compared to the rapid entropy decline exhibited by RL, our method SRFT maintains more stable entropy, indicating that the policy can continue exploring during training, which also demonstrates the effectiveness of the entropy-aware weighting mechanism.

Ablation Study. We conduct an ablation study to assess the effectiveness of each component. As shown in Table 3, we evaluate the impact of the two key entropy-aware weighting mechanisms: wSFT for demonstrations learning and wRL for positive self-exploration samples. We ablate these mechanisms by setting their values to a fixed constant of 1.0 to evaluate the contribution of these two weighting components. Removing the SFT weighting mechanism (w/o wSFT) results in a performance drop of -4.0 points, while removing the RL weighting (w/o wRL) leads to a -2.9 point decrease, demonstrating that both components contribute significantly to overall performance. The ablation results validate our theoretical analysis, confirming that entropy-aware weighting mechanisms enable SRFT to dynamically balance supervised learning and reinforcement learning components, leading to more stable training and superior performance compared to fixed weighting schemes.

Table 3: Ablation results on SRFT, including the impact of wSFT and wRL.

Model AIME24 AMC MATH-500 Minerva Olympiad Avg. Qwen2.5-Math 11.4 32.6 48.8 8.7 15.8 23.5 SRFT w/o wSFT 30.1 65.8 87.0 36.8 55.8 55.1 SRFT w/o wRL 32.6 67.2 87.5 37.4 56.5 56.2 SRFT 35.3 72.2 89.8 39.7 58.3 59.1

- 6 RELATED WORK

Reinforcement Learning for LLM Reasoning. The pursuit of complex reasoning capabilities in LLMs has witnessed remarkable progress, with RL emerging as a pivotal methodology for enhancing reasoning abilities beyond the limitations of SFT alone. Recent approaches such as GRPO (Shao et al., 2024; Guo et al., 2025), DAPO (Yu et al., 2025), DR.GRPO (Liu et al., 2025c), and VAPO (Yue et al., 2025a) have demonstrated substantial improvements in mathematical reasoning and complex problemsolving tasks. However, the precise mechanisms through which RL enhances reasoning capabilities remain incompletely understood. Several empirical investigations suggest that reinforcement learning primarily serves to elicit, refine, or improve the sampling of pre-existing reasoning abilities rather than instilling entirely novel fundamental reasoning skills from scratch. For instance, Yue et al. (2025b) question whether current reinforcement learning with verifiable rewards (RLVR) genuinely expands the reasoning boundary (pass@k) or primarily improves the sampling efficiency of already known solutions (pass@1). Similarly, Wang et al. (2025a) highlight that base models already possess substantial reasoning capabilities that reinforcement learning can effectively unlock or redirect. Nevertheless, ProRL (Liu et al., 2025d) demonstrates that RL-trained models can achieve improved success rates on tasks where base models completely fail, suggesting that sustained and stable reinforcement learning training can indeed expand the reasoning capability boundaries of LLMs. In this work, we design a single-stage method that combines SFT and RL, maintaining stable entropy during training and achieving continuous performance improvement.

Integrating Supervised Fine-Tuning and Reinforcement Learning. The synergistic interaction between SFT and RL represents a critical area of investigation in modern LLM development. SFT on high-quality reasoning chains can establish a robust initial policy foundation, which RL can subsequently optimize. Cai et al. (2025) explore the necessary extent of exploration following SFT, finding that RL continues to provide substantial benefits by enabling models to deviate from potentially suboptimal SFT trajectories. Recent research suggests that SFT may equip models with structured reasoning templates that RL subsequently validates and improves (Chen et al., 2025a). Nevertheless, determining the optimal strategy for combining these complementary paradigms remains an active area of debate. To enhance sample efficiency and provide structured guidance for RL exploration, researchers have investigated various approaches for integrating external supervision into the reinforcement learning framework. UFT (Liu et al., 2025a) proposes a novel paradigm that merges SFT and RL into a single process, using informative supervision signals like hints from partial solutions to guide exploration and accelerate convergence. Addressing the limitations of on-policy learning, LUFFY (Yan et al., 2025) augments RLVR by incorporating off-policy reasoning traces from stronger models, dynamically balancing imitation with on-policy exploration to improve capabilities. ReLIFT (Ma et al., 2025) addresses the limitations of pure RL by interleaving reinforcement learning with supervised fine-tuning on high-quality demonstrations collected during training, enabling models to acquire new knowledge beyond their original capabilities. TAPO (Wu et al., 2025) enhances RL by incorporating external high-level guidance in the form of "thought patterns" abstracted from prior samples, adaptively integrating these to balance model-internal exploration with external strategy exploitation. SASR (Chen et al., 2025b) offers a hybrid framework that theoretically unifies SFT and RL, using SFT for initial warm-up and then adaptively blending it with an online RL method based on training dynamics to maintain core reasoning while exploring diverse paths, using high-quality SFT demonstrations as a key external data source. Furthermore, the single-stage integration of SFT and RL helps mitigate the catastrophic forgetting problem that previous methods encountered when transitioning from SFT to RL (Chen et al., 2025b; Liu et al., 2025b). These approaches collectively underscore an emerging trend toward more sophisticated integrations of supervised signals within reinforcement learning frameworks to improve reasoning alignment and overall performance.

- 7 CONCLUSION

In this work, we investigate the integration of SFT and RL for LLM reasoning. Through comprehensive analysis, we reveal that SFT performs coarse-grained global adjustments while RL conducts fine-grained selective optimizations, with entropy serving as a crucial training indicator. Building on these observations, we propose SRFT, a single-stage approach that unifies both paradigms through entropy-aware weighting mechanisms. Extensive experiments demonstrate SRFT’s effectiveness, achieving 59.1% average accuracy and outperforming zero-RL baselines by 9.0% on reasoning tasks and 10.9% on out-of-distribution benchmarks.

Limitations. While our work demonstrates the effectiveness of entropy-aware SFT-RL singlestage integration, our current utilization of entropy dynamics remains relatively simple with basic exponential weighting functions. The rich temporal patterns of entropy during training suggest opportunities for more sophisticated entropy-based control mechanisms. Future work could explore adaptive entropy scheduling or multi-timescale entropy analysis to better capture the interplay between SFT and RL signals, potentially leading to more principled hybrid training algorithms. Additionally, our approach assumes access to high-quality demonstrations, and future research could investigate the potential for training with imperfect demonstrations to enhance the method’s applicability.

REFERENCES

OpenAI. Introducing OpenAI o3 and o4-mini, 2025. URL https://openai.com/index/introdu cing-o3-and-o4-mini/. 1

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1, 8, 11, 22

Anthropic. Introducing Claude 4, 2025. URL https://www.anthropic.com/news/claude-4. 1 Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V

Le, Sergey Levine, and Yi Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025. 1

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. SFT or RL? an early investigation into training R1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025a. 1, 11

Jingtong Gao, Ling Pan, Yejing Wang, Rui Zhong, Chi Lu, Qingpeng Cai, Peng Jiang, and Xiangyu Zhao. Navigate the unknown: Enhancing LLM reasoning with intrinsic motivation guided exploration. arXiv preprint arXiv:2505.17621, 2025. 1

Shihan Dou, Muling Wu, Jingwen Xu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. Improving RL exploration for LLM reasoning through retrospective replay. arXiv preprint arXiv:2504.14363, 2025. 1

Thomas Schmied, Jörg Bornschein, Jordi Grau-Moya, Markus Wulfmeier, and Razvan Pascanu. LLMs are greedy agents: Effects of RL fine-tuning on decision-making abilities. arXiv preprint

- arXiv:2504.16078, 2025. 1

Hongyi James Cai, Junlin Wang, Xiaoyin Chen, and Bhuwan Dhingra. How much backtracking is enough? exploring the interplay of SFT and RL in enhancing LLM reasoning. arXiv preprint

- arXiv:2505.24273, 2025. 1, 6, 11

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025. 1, 7, 8, 9, 10, 11, 18

Jinyang Wu, Chonghua Liao, Mingkuan Feng, Shuai Zhang, Zhengqi Wen, Pengpeng Shao, Huazhe Xu, and Jianhua Tao. Thought-augmented policy optimization: Bridging external guidance and internal capabilities. arXiv preprint arXiv:2505.15692, 2025. 1, 9, 10, 11, 18

Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. UFT: Unifying supervised and reinforcement fine-tuning. arXiv preprint arXiv:2505.16984, 2025a. 1, 11

Jack Chen, Fazhong Liu, Naruto Liu, Yuhan Luo, Erqu Qin, Harry Zheng, Tian Dong, Haojin Zhu, Yan Meng, and Xiao Wang. Step-wise adaptive integration of supervised fine-tuning and reinforcement learning for task-specific LLMs. arXiv preprint arXiv:2505.13026, 2025b. 1, 11

Yihao Liu, Shuocheng Li, Lang Cao, Yuhang Xie, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, and Dongmei Zhang. SuperRL: Reinforcement learning with supervision to boost LLM reasoning. arXiv preprint arXiv:2506.01096, 2025b. 1, 11

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024. 2, 9, 22

Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014. 3

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3, 7, 9, 10, 11, 18

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3, 7

Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/. 5, 22

Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Bin Cui, et al. Learning what reinforcement learning can’t: Interleaved online fine-tuning for hardest questions. arXiv preprint arXiv:2506.07527, 2025. 8, 9, 10, 11, 18

Hugging Face. Open R1: A fully open reproduction of DeepSeek-R1, January 2025. 8, 18 Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang,

Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath, 2024. 8

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13:9, 2024. 9

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022. 9

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024. 9

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 9

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. 9

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. 9

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. MMLU-Pro: A more robust and challenging multitask language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. 9

Zichen Liu, Changyu Chen, Xinyi Wan, Chao Du, Wee Sun Lee, and Min Lin. OAT: A researchfriendly framework for LLM online alignment. https://github.com/sail-sg/oat, 2024. 9, 18

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. SimpleRLZoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025. 9, 10

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-Reasoner-Zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025. 9, 10

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025. 9, 10

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 11

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025c. 11

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, et al. VAPO: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025a. 11

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? arXiv preprint arXiv:2504.13837, 2025b. 11

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025a. 11

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. ProRL: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025d. 11

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. arXiv preprint arXiv:2506.01939, 2025b. 17

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024. 18

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 18

# Appendix

- A More Experimental Results 17

- A.1 Token Probability Visualization of SRFT . . . . . . . . . . . . . . . . . . . . . . . 17
- A.2 Entropy-aware Gradient Clipping . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- B Experimental Details 18
- C Dataset Details 19

- C.1 Training Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.2 Evaluation Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D SFT Gradient Derivation 21

- D.1 Problem Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.2 Gradient Derivation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- E Details of Visualization of Learning Dynamics 22

- E.1 Theoretical Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.2 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- A MORE EXPERIMENTAL RESULTS

- A.1 TOKEN PROBABILITY VISUALIZATION OF SRFT

We visualize the token probability distribution of SRFT after training, as shown in Figure A1. We observe that the token probability changes are moderate, achieving a balanced point between SFT and RL that enhances the model’s reasoning capabilities while preserving its base abilities.

SRFT

Question: Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.

Lower Probability After Fine-tuning Similar Probability Higher Probability After Fine-tuning

|<think> To solve this problem , we need to determine the time it takes for A ya to walk 9 kilometers at a speed of \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including the time spent in the coffee shop , which is \( t \ ) minutes . We are given two scenarios where we know the total time taken for the walk plus the time spent in the coffee shop . First , let 's define the variables : - \( s \ ): speed in kilometers per hour when the walk takes 4 hours total ( including \( t \ ) minutes in the coffee shop ). - \( s + 2 \ ): speed in kilometers per hour when the walk takes 2 hours and 2 4 minutes total ( including \( t \ ) minutes in the coffee shop ). - \( t \ ): time spent in the coffee shop in minutes . We need to find the time taken when walking at \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including \( t \ ) minutes . We know the total distance of the walk is 9 kilometers . The total time taken in each scenario is the sum of the walking time and the time spent in the coffee shop . Let 's denote : - \( T _ 1 \ ): total time taken in the first scenario ( 4 hours ). - \( T _ 2 \ ): total time taken in the second scenario ( 2 hours and 2 4 minutes ). Omit part of the output... Now , when walking at \( s + \ frac { 1 }{ 2 } = 2 . 5 + 0 . 5 = 3 \ ) km /h , the walking time is \( \ frac { 9 }{ 3 } = 3 \ ) hours , which is 1 8 0 minutes . Adding the time spent in the coffee shop \( t = 2 4 \ ) minutes gives \( 1 8 0 + 2 4 = 2 0 4 \ ) minutes . Therefore , the answer should be \ boxed { 2 0 4 }. </ think > Every morning A ya goes for a 9 -k il ometer -long walk and stops at a coffee shop afterwards . When she walks at a constant speed of \( s \ ) kilometers per hour , the walk takes her 4 hours , including \( t \ ) minutes spent in the coffee shop . When she walks \( s + 2 \ ) kilometers per hour , the walk takes her 2 hours and 2 4 minutes , including \( t \ ) minutes spent in the coffee shop . We need to find the number of minutes the walk takes her when she walks at \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including the \( t \ ) minutes spent in the coffee shop . First , we denote the time spent walking in each scenario : 1 . For the first scenario ( speed \( s \ )): - Total time : 4 hours - Walking time : \( 4 - \ frac { t }{ 6 0 } \ ) hours - Distance equation : \( s \ left ( 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) 2 . For the second scenario ( speed \( s + 2 \ )): - Total time : 2 hours and 2 4 minutes = 2 . 4 hours - Walking time : \( 2 . 4 - \ frac { t }{ 6 0 } \ ) hours - Distance equation : \( ( s + 2 ) \ left ( 2 . 4 \ frac { t }{ 6 0 }\ right ) = 9 \ ) We solve these two equations : 1 . \( s \ left ( 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) 2 . \( ( s + 2 ) \ left ( 2 . 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) Let \( \ tau = \ frac { t }{ 6 0 } \ ). The equations become : 1 . \( s ( 4 - \ tau ) = 9 \ ) 2 . \( ( s + 2 )( 2 . 4 - \ tau ) = 9 \ ) From the first equation , \( s = \ frac { 9 }{ 4 - \ tau } \ ). Sub stit uting into the second equation : \[ \ left ( \ frac { 9 }{ 4 - \ tau } + 2 \ right )( 2 . 4 - \ tau ) = 9 \ ] S impl ifying : \[ \ left ( \ frac { 1 7 - 2 \ tau }{ 4 - \ tau } \ right )( 2 . 4 - \ tau ) = 9 \ ] Multip lying both sides by \( 4 - \ tau \ ): \[ ( 1 7 - 2 \ tau )( 2 . 4 - \ tau ) = 9 ( 4 - \ tau ) \ ] Exp anding both sides : Left side : \( 4 0 . 8 - 2 1 . 8 \ tau + 2 \ tau ^ 2 \ ) Right side : \( 3 6<br><br>- 9 \ tau \ ) Setting equal and simpl ifying : \[ 2 \ tau ^ 2 - 1 2 . 8 \ tau + 4 . 8 = 0 \ ] Div iding by 2 : \[ \ tau ^ 2 - 6 . 4 \ tau + 2 . 4 = 0 \ ] S olving the quadratic equation : \[ \ tau = \ frac { 6 . 4 \ pm 5 . 6 }{ 2 } \ ] This gives \( \ tau = 6 \ ) or \( \ tau = 0 . 4 \ ). Since \( \ tau = 6 \ ) leads to a negative speed , we have \( \ tau = 0 . 4 \ ). Thus , \( t = 2 4 \ ) minutes and \( s = 2 . 5 \ ) km /h . When walking at \( s + \ frac { 1 }{ 2 } = 3 \ ) km /h , the walking time is \( \ frac { 9 }{ 3 } = 3 \ ) hours , which is 1 8 0 minutes . Adding the time spent in the coffee shop \( t = 2 4 \ ) minutes gives a total of \( 1 8 0 + 2 4 = 2 0 4 \ ) minutes . The answer is \ boxed { 2 0 4 }.<br><br>|
|---|

Token Probability Visualization

Generate Model: LUFFY-Qwen-Math-7B-Zero Reference Model: Qwen2.5-Math-7B (base)

Lower Probability After LUFFY Similar Probability Higher Probability After LUFFY

Figure A1: Token probability distribution visualization for SRFT.

- A.2 ENTROPY-AWARE GRADIENT CLIPPING

Our investigation into the entropy characters of tokens modified by fine-tuning reveals that reinforcement learning predominantly targets tokens with high entropy distributions, a finding that aligns with recent work on selective optimization in language models (Wang et al., 2025b). To empirically validate this observation, we design controlled experiments implementing gradient truncation for highprobability tokens during RL training. As demonstrated in Figure A2, the model’s performance remains comparable to the original RL algorithm even when gradients are truncated for low-entropy tokens, providing strong empirical support for our hypothesis. This evidence confirms that RL operates with remarkable selectivity, precisely adjusting tokens with uncertain distributions while leaving confident predictions largely unchanged. In contrast, SFT applies broad modifications across the entire token space, fundamentally altering the model’s distributional characters in a less discriminative manner.

| |Prob_clip 0.5<br><br>Prob_clip 0.8<br><br>Prob_clip 0.9<br><br><br>Base Policy| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

48

46

Accuracy

44

42

40

0 100 200 300 400 500

Step

Figure A2: Performance of RL with gradient clipping for low-entropy tokens.

- B EXPERIMENTAL DETAILS

Training. We follow the SFT configuration of OpenR1-Qwen-7B (Face, 2025), performing full fine-tuning on DeepSeek-R1 generated reasoning traces and prompts. The training hyperparameters include a batch size of 128, learning rate of 5×10−6, linear learning rate schedule with 10% warmup, and training for 3 epochs. For SFT with KL regularization, we use identical settings while adding a KL divergence regularization between the current policy and the base model (Qwen2.5-Math-7B) with weight λ = 0.2. The SFTKL loss is:

### (θ) = E(x,y)∼D[−log πθ(y|x)] + λLKL(θ,θbase), (A1)

LSFTKL

where LKL(θ,θbase) is the KL divergence between the current policy and the base model. For RL, we train for 500 steps with 8 rollouts per prompt. The learning rate is fixed at 1 × 10−6. For our method SRFT, we use the same training settings as RL. Since the maximum sequence length for Qwen2.5-Math-7B is 4096, which is insufficient for our tasks, we increase the RoPE theta from 10,000 to 40,000 and expand the window size to 16,384. For all experiments, we use verl3 (Sheng et al., 2024) as the implementation framework. All experiments are conducted on 64 A100 GPUs.

Supervised and Reinforcement Fine-Tuning

- A VISUALIZATION

SRFT

Question: Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.

Lower Probability After Fine-tuning Similar Probability Higher Probability After Fine-tuning

Token Probability Visualization

Generate Model: LUFFY-Qwen-Math-7B-Zero Reference Model: Qwen2.5-Math-7B (base)

Lower Probability After LUFFY Similar Probability Higher Probability After LUFFY

Figure 7: Token probability distribution visualization for SRFT.

- B EXPERIMENTAL DETAILS

Evaluation. All evaluations are conducted using VLLM (Kwon et al., 2023) with temperature set to 0.6 and maximum generation length of 8192 tokens. For datasets with limited sample sizes (AIME24 and AMC), we report the avg@32 metric; for the remaining three datasets, we adopt pass@1 as the evaluation criterion. We verify the correctness of generated solutions using Math-Verify and OAT-Grader (Liu et al., 2024). For baseline comparisons, we independently validate the results of base model, SFT-related baselines, GRPO (Shao et al., 2024), LUFFY (Yan et al., 2025), and ReLIFT (Ma et al.,

|2025), while results for TAPO (Wu et al., 2025) and other taken from the TAPO (because we cannot find the<br><br>evaluate the impact of our method, we adopt a simple employ the same reward function.<br><br>R(x,y) =<br><br>1, if y is correct 0, otherwise<br><br>.<br><br><think> To solve this problem , we need to determine the time it takes for A ya to walk 9 kilometers at a speed of \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including the time spent in the coffee shop , which is \( t \ ) minutes . We are given two scenarios where we know the total time taken for the walk plus the time spent in the coffee shop . First , let 's define the variables : - \( s \ ): speed in kilometers per hour when the walk takes 4 hours total ( including \( t \ ) minutes in the coffee shop ). - \( s + 2 \ ): speed in kilometers per hour when the walk takes 2 hours and 2 4 minutes total ( including \( t \ ) minutes in the coffee shop ). - \( t \ ): time spent in the coffee shop in minutes . We need to find the time taken when walking at \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including \( t \ ) minutes . We know the total distance of the walk is 9 kilometers . The total time taken in each scenario is the sum of the walking time and the time spent in the coffee shop . Let 's denote : - \( T _ 1 \ ): total time taken in the first scenario ( 4 hours ). - \( T _ 2 \ ): total time taken in the second scenario ( 2 hours and 2 4 minutes ). Omit part of the output... Now , when walking at \( s + \ frac { 1 }{ 2 } = 2 . 5 + 0 . 5 = 3 \ ) km /h , the walking time is \( \ frac { 9 }{ 3 } = 3 \ ) hours , which is 1 8 0 minutes . Adding the time spent in the coffee shop \( t = 2 4 \ ) minutes gives \( 1 8 0 + 2 4 = 2 0 4 \ ) minutes . Therefore , the answer should be \ boxed { 2 0 4 }. </ think > Every morning A ya goes for a 9 -k il ometer -long walk and stops at a coffee shop afterwards . When she walks at a constant speed of \( s \ ) kilometers per hour , the walk takes her 4 hours , including \( t \ ) minutes spent in the coffee shop . When she walks \( s + 2 \ ) kilometers per hour , the walk takes her 2 hours and 2 4 minutes , including \( t \ ) minutes spent in the coffee shop . We need to find the number of minutes the walk takes her when she walks at \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including the \( t \ ) minutes spent in the coffee shop . First , we denote the time spent walking in each scenario : 1 . For the first scenario ( speed \( s \ )): - Total time : 4 hours - Walking time : \( 4 - \ frac { t }{ 6 0 } \ ) hours - Distance equation : \( s \ left ( 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) 2 . For the second scenario ( speed \( s + 2 \ )): - Total time : 2 hours and 2 4 minutes = 2 . 4 hours - Walking time : \( 2 . 4 - \ frac { t }{ 6 0 } \ ) hours - Distance equation : \( ( s + 2 ) \ left ( 2 . 4 \ frac { t }{ 6 0 }\ right ) = 9 \ ) We solve these two equations : 1 . \( s \ left ( 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) 2 . \( ( s + 2 ) \ left ( 2 . 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) Let \( \ tau = \ frac { t }{ 6 0 } \ ). The equations become : 1 . \( s ( 4 - \ tau ) = 9 \ ) 2 . \( ( s + 2 )( 2 . 4 - \ tau ) = 9 \ ) From the first equation , \( s = \ frac { 9 }{ 4 - \ tau } \ ). Sub stit uting into the second equation : \[ \ left ( \ frac { 9 }{ 4 - \ tau } + 2 \ right )( 2 . 4 - \ tau ) = 9 \ ] S impl ifying : \[ \ left ( \ frac { 1 7 - 2 \ tau }{ 4 - \ tau } \ right )( 2 . 4 - \ tau ) = 9 \ ] Multip lying both sides by \( 4 - \ tau \ ): \[ ( 1 7 - 2 \ tau )( 2 . 4 - \ tau ) = 9 ( 4 - \ tau ) \ ] Exp anding both sides : Left side : \( 4 0 . 8 - 2 1 . 8 \ tau + 2 \ tau ^ 2 \ ) Right side : \( 3 6<br><br>- 9 \ tau \ ) Setting equal and simpl ifying : \[ 2 \ tau ^ 2 - 1 2 . 8 \ tau + 4 . 8 = 0 \ ] Div iding by 2 : \[ \ tau ^ 2 - 6 . 4 \ tau + 2 . 4 = 0 \ ] S olving the quadratic equation : \[ \ tau = \ frac { 6 . 4 \ pm 5 . 6 }{ 2 } \ ] This gives \( \ tau = 6 \ ) or \( \ tau = 0 . 4 \ ). Since \( \ tau = 6 \ ) leads to a negative speed , we have \( \ tau = 0 . 4 \ ). Thus , \( t = 2 4 \ ) minutes and \( s = 2 . 5 \ ) km /h . When walking at \( s + \ frac { 1 }{ 2 } = 3 \ ) km /h , the walking time is \( \ frac { 9 }{ 3 } = 3 \ ) hours , which is 1 8 0 minutes . Adding the time spent in the coffee shop \( t = 2 4 \ ) minutes gives a total of \( 1 8 0 + 2 4 = 2 0 4 \ ) minutes . The answer is \ boxed { 2 0 4 }.<br><br>|
|---|

2 zero-shot reinforcement learning models are open-source code or model) and LUFFY papers.

Reward Design. To reward function as below. All training experiments

(A2)

Chat Template. Following Yan et al. (2025); Ma et al. (2025), for all training paradigms (SFT, RL, SRFT), we employ a unified system prompt that encourages systematic reasoning, as shown in Figure A3. We also experimented with alternative templates, as shown in Figure A4.

Chat Template (Ours)

Your task is to follow a systematic, thorough reasoning process before providing the ﬁnal solution. This involves analyzing, summarizing, exploring, reassessing, and reﬁning your thought process through multiple iterations. Structure your response into two sections: Thought and Solution.

In the Thought section, present your reasoning using the format:“<think>\n {thoughts} </think>\n”. Each thought should include detailed analysis, brainstorming, veriﬁcation, and reﬁnement of ideas.

After “</think>\n” in the Solution section, provide the ﬁnal, logical, and accurate answer, clearly derived from the exploration in the Thought section.

If applicable, include the Answer in \boxed{} for closed-form results like multiple choices or mathematical solutions.

Question: {question} Answer: {answer}

Chat Template (Qwen) Please reason step by step, and put your ﬁnal answer within \\boxed{}. Question: {question} Answer: {answer}

Figure A3: Chat template for all training paradigms (SFT, RL, SRFT).

Template Ablation. To minimize template influence, we evaluated the base Qwen-7B-Math model with different templates. Results are shown in Table A1, which indicates that our template design effectively guides the model’s reasoning process while maintaining consistency across different mathematical domains.

3https://github.com/volcengine/verl

18

13

Supervised and Reinforcement Fine-Tuning

- A VISUALIZATION

SRFT

Question: Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.

Lower Probability After Fine-tuning Similar Probability Higher Probability After Fine-tuning

Token Probability Visualization

Generate Model: LUFFY-Qwen-Math-7B-Zero Reference Model: Qwen2.5-Math-7B (base)

Lower Probability After LUFFY Similar Probability Higher Probability After LUFFY

|<think> To solve this problem , we need to determine the time it takes for A ya to walk 9 kilometers at a speed of \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including the time spent in the coffee shop , which is \( t \ ) minutes . We are given two scenarios where we know the total time taken for the walk plus the time spent in the coffee shop . First , let 's define the variables : - \( s \ ): speed in kilometers per hour when the walk takes 4 hours total ( including \( t \ ) minutes in the coffee shop ). - \( s + 2 \ ): speed in kilometers per hour when the walk takes 2 hours and 2 4 minutes total ( including \( t \ ) minutes in the coffee shop ). - \( t \ ): time spent in the coffee shop in minutes . We need to find the time taken when walking at \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including \( t \ ) minutes . We know the total distance of the walk is 9 kilometers . The total time taken in each scenario is the sum of the walking time and the time spent in the coffee shop . Let 's denote : - \( T _ 1 \ ): total time taken in the first scenario ( 4 hours ). - \( T _ 2 \ ): total time taken in the second scenario ( 2 hours and 2 4 minutes ). Omit part of the output... Now , when walking at \( s + \ frac { 1 }{ 2 } = 2 . 5 + 0 . 5 = 3 \ ) km /h , the walking time is \( \ frac { 9 }{ 3 } = 3 \ ) hours , which is 1 8 0 minutes . Adding the time spent in the coffee shop \( t = 2 4 \ ) minutes gives \( 1 8 0 + 2 4 = 2 0 4 \ ) minutes . Therefore , the answer should be \ boxed { 2 0 4 }. </ think > Every morning A ya goes for a 9 -k il ometer -long walk and stops at a coffee shop afterwards . When she walks at a constant speed of \( s \ ) kilometers per hour , the walk takes her 4 hours , including \( t \ ) minutes spent in the coffee shop . When she walks \( s + 2 \ ) kilometers per hour , the walk takes her 2 hours and 2 4 minutes , including \( t \ ) minutes spent in the coffee shop . We need to find the number of minutes the walk takes her when she walks at \( s + \ frac { 1 }{ 2 } \ ) kilometers per hour , including the \( t \ ) minutes spent in the coffee shop . First , we denote the time spent walking in each scenario : 1 . For the first scenario ( speed \( s \ )): - Total time : 4 hours - Walking time : \( 4 - \ frac { t }{ 6 0 } \ ) hours - Distance equation : \( s \ left ( 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) 2 . For the second scenario ( speed \( s + 2 \ )): - Total time : 2 hours and 2 4 minutes = 2 . 4 hours - Walking time : \( 2 . 4 - \ frac { t }{ 6 0 } \ ) hours - Distance equation : \( ( s + 2 ) \ left ( 2 . 4 \ frac { t }{ 6 0 }\ right ) = 9 \ ) We solve these two equations : 1 . \( s \ left ( 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) 2 . \( ( s + 2 ) \ left ( 2 . 4 - \ frac { t }{ 6 0 }\ right ) = 9 \ ) Let \( \ tau = \ frac { t }{ 6 0 } \ ). The equations become : 1 . \( s ( 4 - \ tau ) = 9 \ ) 2 . \( ( s + 2 )( 2 . 4 - \ tau ) = 9 \ ) From the first equation , \( s = \ frac { 9 }{ 4 - \ tau } \ ). Sub stit uting into the second equation : \[ \ left ( \ frac { 9 }{ 4 - \ tau } + 2 \ right )( 2 . 4 - \ tau ) = 9 \ ] S impl ifying : \[ \ left ( \ frac { 1 7 - 2 \ tau }{ 4 - \ tau } \ right )( 2 . 4 - \ tau ) = 9 \ ] Multip lying both sides by \( 4 - \ tau \ ): \[ ( 1 7 - 2 \ tau )( 2 . 4 - \ tau ) = 9 ( 4 - \ tau ) \ ] Exp anding both sides : Left side : \( 4 0 . 8 - 2 1 . 8 \ tau + 2 \ tau ^ 2 \ ) Right side : \( 3 6<br><br>- 9 \ tau \ ) Setting equal and simpl ifying : \[ 2 \ tau ^ 2 - 1 2 . 8 \ tau + 4 . 8 = 0 \ ] Div iding by 2 : \[ \ tau ^ 2 - 6 . 4 \ tau + 2 . 4 = 0 \ ] S olving the quadratic equation : \[ \ tau = \ frac { 6 . 4 \ pm 5 . 6 }{ 2 } \ ] This gives \( \ tau = 6 \ ) or \( \ tau = 0 . 4 \ ). Since \( \ tau = 6 \ ) leads to a negative speed , we have \( \ tau = 0 . 4 \ ). Thus , \( t = 2 4 \ ) minutes and \( s = 2 . 5 \ ) km /h . When walking at \( s + \ frac { 1 }{ 2 } = 3 \ ) km /h , the walking time is \( \ frac { 9 }{ 3 } = 3 \ ) hours , which is 1 8 0 minutes . Adding the time spent in the coffee shop \( t = 2 4 \ ) minutes gives a total of \( 1 8 0 + 2 4 = 2 0 4 \ ) minutes . The answer is \ boxed { 2 0 4 }.<br><br>|
|---|

Figure 7: Token probability distribution visualization for SRFT.

- B EXPERIMENTAL DETAILS

Chat Template (Ours)

Your task is to follow a systematic, thorough reasoning process before providing the ﬁnal solution. This involves analyzing, summarizing, exploring, reassessing, and reﬁning your thought process through multiple iterations. Structure your response into two sections: Thought and Solution.

In the Thought section, present your reasoning using the format:“<think>\n {thoughts} </think>\n”. Each thought should include detailed analysis, brainstorming, veriﬁcation, and reﬁnement of ideas.

After “</think>\n” in the Solution section, provide the ﬁnal, logical, and accurate answer, clearly derived from the exploration in the Thought section.

If applicable, include the Answer in \boxed{} for closed-form results like multiple choices or mathematical solutions.

Supervised and Reinforcement Fine-Tuning

Question: {question} Answer: {answer}

Chat Template (Qwen) Please reason step by step, and put your ﬁnal answer within \\boxed{}. Question: {question} Answer: {answer}

Figure A4: Chat template for Qwen-2.5-Math.

Table A1: Template ablation results on mathematical reasoning benchmarks Template AIME24 AMC MATH500 Minerva Olympiad Average

13

No Template 0.302 0.132 0.596 0.424 0.134 0.318 Qwen2.5-Math 0.144 0.088 0.446 0.303 0.111 0.218 SRFT 0.296 0.165 0.648 0.448 0.141 0.340

- C DATASET DETAILS

- C.1 TRAINING DATASET

OpenR1-Math-220k is a comprehensive dataset designed for mathematical reasoning, comprising 220,000 math problems. Each problem is associated with two to four reasoning traces generated by the DeepSeek-R1 model. The traces have been verified using tools like Math Verify and Llama3.3-70B-Instruct, ensuring at least one correct reasoning path per problem. This dataset challenges models to understand and replicate complex reasoning processes across various mathematical domains including algebra, geometry, number theory, and calculus.

- C.2 EVALUATION BENCHMARKS

To evaluate the models above, we use eight benchmarks categorized into mathematical reasoning benchmarks and out-of-distribution (OOD) benchmarks as described below.

- C.2.1 MATHEMATICAL REASONING BENCHMARKS

AIME24 is a benchmark dataset based on problems from the 2024 American Invitational Mathematics Examination, a prestigious high school mathematics competition in the United States. The AIME24 benchmark tests a model’s ability to solve challenging mathematics problems by generating step-bystep solutions and providing the correct answers. This dataset contains problems from the American Invitational Mathematics Examination (AIME) 2024, organized in JSONL format where each line represents a complete problem. Concepts typically covered include topics in elementary algebra, geometry, trigonometry, as well as number theory, probability, and combinatorics. The examination consists of 15 problems with integer answers between 0 and 999, requiring advanced mathematical reasoning and problem-solving skills.

AMC is a validation dataset containing problems from the American Mathematics Competitions, specifically AMC12 from 2022 and 2023. All 83 problems come from AMC12 2022, AMC12 2023, and have been extracted from the AOPS wiki page. The AMC 10 is a 25-question, 75-minute multiplechoice competition designed for students in grades 10 and below. The content covers mathematical topics such as elementary algebra, basic geometry, area and volume formulas, elementary number theory, and elementary probability. This dataset serves as an internal validation set and focuses on competition-level mathematical problems comparable in difficulty to AMC12 and AIME exams.

MATH500 is a carefully curated subset of mathematical problems designed for robust evaluation. MATH500 is a subset of 500 randomly sampled questions from Hendrycks’ 2021 MATH dataset, created by OpenAI in late 2024 as a consequence of their appropriation of 90% of the original 5000 MATH questions for training data for reinforcement learning on o1-series models. The dataset

Table A2: Benchmarks used in this study. “–” indicates the split is not officially provided.

Dataset #Train #Test Task Type Domain License Source

Training Dataset OPENR1-MATH-220K 220,000 – Math reasoning Mathematics Apache 2.0 [Link] Mathematical Reasoning Benchmarks

AIME24 – 30 Math competition Mathematics MIT [Link]

AMC – 83 Math competition Mathematics Apache 2.0 [Link] MATH500 – 500 Mathematical reasoning Mathematics - [Link]

MINERVA – 272 Mathematical reasoning Mathematics Apache 2.0 [Link] OLYMPIAD – 674 Math competition Mathematics Apache 2.0 [Link]

##### Out-of-Distribution (OOD) Benchmarks

ARC-C – 1,172 Science reasoning General science CC-BY-SA-4.0 [Link] GPQA-D – 198 Scientific reasoning Bio, Phys, Chem CC-BY-4.0 [Link]

MMLU-PRO – 12,032 Multi-task understanding Multidisciplinary MIT [Link]

maintains the diversity and complexity of the original MATH benchmark while providing a clean evaluation set that avoids potential data contamination issues.

Minerva is a mathematical reasoning benchmark that encompasses a wide range of mathematical domains and difficulty levels. The dataset is designed to evaluate models’ capabilities in advanced mathematical problem-solving, including topics from high school to undergraduate level mathematics. It includes problems requiring multi-step reasoning, symbolic manipulation, and deep mathematical understanding across various mathematical fields.

Olympiad refers to mathematical olympiad-level problems that represent some of the most challenging mathematical reasoning tasks. Unlike existing Olympiad-related benchmarks, datasets in this category focus exclusively on mathematics and comprise vast collections of competition-level problems. These problems are meticulously categorized into 33+ sub-domains and span across 10+ distinct difficulty levels. These problems require exceptional mathematical insight, creativity, and advanced problem-solving techniques typically seen in international mathematical competitions. We specifically utilize the OE_TO_MATHS_EN_COMP subset for our evaluation.

- C.2.2 OUT-OF-DISTRIBUTION (OOD) BENCHMARKS

ARC-C (AI2 Reasoning Challenge-Challenge) is a dataset of grade-school level science questions that require commonsense reasoning and knowledge application. The dataset consists of multiplechoice questions that are designed to be easy for humans but challenging for AI systems, testing the model’s ability to apply scientific knowledge and reasoning in everyday contexts beyond pure mathematical domains.

GPQA-D (Graduate-Level Google-Proof Q&A-Diamond) is a challenging benchmark designed to evaluate advanced reasoning capabilities in scientific domains. GPQA consists of 448 multiple-choice questions designed to evaluate the capabilities of LLMs and scalable oversight mechanisms. This dataset provides "Google-proof" questions in biology, physics, and chemistry, designed to test deep domain expertise and reasoning under challenging conditions. The diamond subset contains 198 hard problems. The questions require graduate-level knowledge and are specifically designed to be difficult to answer even with internet search.

MMLU-Pro (Massive Multitask Language Understanding-Professional) is an enhanced version of the original MMLU benchmark designed to be more challenging and robust. The MMLU-Pro dataset is an enhanced version of the Massive Multitask Language Understanding benchmark. It’s designed to be more robust and challenging, aiming to rigorously evaluate language understanding capabilities. MMLU is a comprehensive benchmark that covers 57 subjects across fields like mathematics, history, law, and medicine. It assesses not only factual knowledge but also the model’s capacity to apply this knowledge in context-specific scenarios. MMLU-Pro increases the difficulty and reduces potential shortcuts while maintaining broad coverage across academic disciplines.

- D SFT GRADIENT DERIVATION

In this section, we provide the detailed mathematical derivation for the gradient of the Supervised Fine-Tuning (SFT) loss function, which was referenced in Sec. 3.1.1.

- D.1 PROBLEM SETUP

Given a dataset D = {(xi,yi)}Ni=1 where xi is an input prompt and yi = (yi,1,yi,2,...,yi,T

i

) is the corresponding target sequence, the SFT objective function is (the same as Eq. (1)):

LSFT(θ) = E(x,y)∼D[−log πθ(y|x)]. (A3) Since the sequence probability is factorized as:

πθ(y|x) =

T

t=1

πθ(yt|x,y<t). (A4) The SFT loss becomes:

LSFT(θ) = E(x,y)∼D −

T

t=1

log πθ(yt|x,y<t) . (A5)

- D.2 GRADIENT DERIVATION To derive the gradient ∇θLSFT(θ), we need to compute:

T

∇θ log πθ(yt|x,y<t) . (A6)

∇θLSFT(θ) = E(x,y)∼D

data −

t=1

Now, let’s consider the key insight: at each time step t, the model produces a probability distribution over the entire vocabulary V. The gradient of the log probability for the target token yt can be expressed in terms of the model’s prediction probabilities for all vocabulary tokens.

For any token v ∈ V at position t, we have:

1

πθ(v|x,y<t)∇θπθ(v|x,y<t) (A7) Since v∈V πθ(v|x,y<t) = 1, we have the constraint:

∇θ log πθ(v|x,y<t) =

∇θπθ(v|x,y<t) = 0. (A8)

v∈V

Using the chain rule and the fact that the softmax normalization affects all vocabulary tokens, the gradient can be written as:

t − πθ(v|x,y<t))∇θ log πθ(v|x,y<t), (A9)

∇θ log πθ(yt|x,y<t) =

(1v=y

v∈V

where 1v=y

is the indicator function that equals 1 when v = yt and 0 otherwise. Substituting this back into the SFT gradient, we obtain:

t

 

 . (A10)

|y|

∇θLSFT = E(x,y)∼D

(πθ(v|x,y<t) − 1v=y

)∇θ log πθ(v|x,y<t)

t

t=1 v∈V

This formulation reveals the fundamental mechanism of SFT: at each time step, the gradient encourages the model to increase the probability of the target token (1v=y

= 1) while decreasing the probabilities of all other tokens in the vocabulary (1v=y

t

= 0). The magnitude of the decrease for each non-target token v is proportional to its current probability πθ(v|x,y<t).

t

This analysis confirms our empirical observations that SFT produces broad, coarse-grained changes to the model’s probability distributions across the entire vocabulary, systematically sharpening the distribution toward the target tokens in the training data.

- E DETAILS OF VISUALIZATION OF LEARNING DYNAMICS

To characterize model dynamics during training, in Sec. 3.1.2, we formulate a language model as a mapping from input prompts to output probability distributions over the vocabulary. Two models are considered similar or close if they assign identical probabilities to every output token across all input prompts. Under this equivalence, each model can be theoretically represented as a point in an infinite-dimensional probability space, where each dimension corresponds to the model’s assigned probability for a specific token at a specific position within all possible outputs.

- E.1 THEORETICAL DEFINITION

Model Space Formalization. We define the model space M as the set of all possible conditional probability distributions over token sequences. Each model M ∈ M can thus be represented as a vector in an infinite-dimensional space, with each dimension corresponding to the conditional probability pM(vi | x,y<t) for vocabulary token vi at position t.

Measure Model Dynamics via Reference Models. To make this space tractable for analysis and visualization, we measure the model dynamics by the distance between each model and a set of reference models R = {R1,R2,...,Rk} (reference frames):

dR(M) = (dR

1

(M),dR

2

(M),...,dR

k

(M))⊺ , (A11) where dR

i

(M) quantifies the distance between model M and reference model Ri in terms of assigned sequence probabilities.

- E.2 EXPERIMENTAL SETUP

To instantiate the above definition, we construct a dataset comprising 1,024 response sequences generated by each of the reference models under identical prompts. These prompts are drawn from a diverse mixture of mathematical reasoning benchmarks (including AIME24, Minerva, Olympiad, AMC, and MATH-500) to ensure broad coverage across problem domains and difficulty levels.

For each response, we compute the distance between the probabilities assigned by model M to the reference tokens, and then aggregate overall responses to obtain the final distance dR

(M) for each Ri. These distances collectively define the model’s position in the projected subspace. We select three reference models to construct the projection basis: (1) DeepSeek-R1 (Guo et al., 2025), representing state-of-the-art reasoning performance; (2) QwQ-32B (Team, 2025), serving as a high-performing but structurally distinct baseline; and (3) Qwen-2.5-Math-7B (Yang et al., 2024), which acts as the base model prior to fine-tuning. Together, they span a spectrum from foundational to advanced capabilities, forming a semantically meaningful coordinate system.

i

For every model checkpoint throughout training, we evaluate its distance based on its probability assignments over the reference responses. This yields a trajectory in the model space that traces how the model evolves over time. By comparing these trajectories across training paradigms (e.g., SFT and RL), we uncover distinct optimization dynamics and convergence behaviors. This three-dimensional distance framework provides both theoretical grounding and practical interpretability for analyzing training dynamics. It enables direct comparison of different model variants and reveals how specific training strategies influence progression through the reasoning capability landscape.

