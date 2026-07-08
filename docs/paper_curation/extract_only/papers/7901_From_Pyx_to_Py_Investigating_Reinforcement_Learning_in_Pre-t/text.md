# arXiv:2604.14142v1[cs.LG]15Apr2026

## From P(y|x) to P(y): Investigating Reinforcement Learning in Pre-train Space

Yuqiao Tan1,2,∗Minzheng Wang1,2,∗Bo Liu3, Zichen Liu3, Tian Liang4, Shizhu He1,2†, Jun Zhao1,2, Kang Liu1,2

- 1 Institute of Automation, Chinese Academy of Sciences
- 2 University of Chinese Academy of Sciences
- 3 National University of Singapore 4 Tencent AI Lab {tanyuqiao2025,wangminzheng2023}@ia.ac.cn, {shizhu.he}@nlpr.ia.ac.cn

##### Abstract

While reinforcement learning with verifiable rewards (RLVR) significantly enhances LLM reasoning by optimizing the conditional distribution P(y|x), its potential is fundamentally bounded by the base model’s existing output distribution. Optimizing the marginal distribution P(y) in the Pre-train Space addresses this bottleneck by encoding reasoning ability and preserving broad exploration capacity. Yet, conventional pre-training relies on static corpora for passive learning, leading to a distribution shift that hinders targeted reasoning enhancement. In this paper, we introduce PreRL (Pre-train Space RL), which applies reward-driven online updates directly to P(y). We theoretically and empirically validate the strong gradient alignment between log P(y) and log P(y|x), establishing PreRL as a viable surrogate for standard RL. Furthermore, we uncover a critical mechanism: Negative Sample Reinforcement (NSR) within PreRL serves as an exceptionally effective driver for reasoning. NSR-PreRL rapidly prunes incorrect reasoning spaces while stimulating endogenous reflective behaviors, increasing transition and reflection thoughts by 14.89× and 6.54×, respectively. Leveraging these insights, we propose Dual Space RL (DSRL), a Policy Reincarnation strategy that initializes models with NSR-PreRL to expand the reasoning horizon before transitioning to standard RL for fine-grained optimization. Extensive experiments demonstrate that DSRL consistently outperforms strong baselines, proving that pre-train space pruning effectively steers the policy toward a refined correct reasoning subspace. Code is available here.

##### 1 Introduction

Reinforcement learning (RL) has emerged as a key driver in advancing the complex reasoning capabilities of large language models (LLMs) (Sutton et al., 1998; Achiam et al., 2023; DeepMind, 2024; Yang et al., 2025). Notably, the success of DeepSeek-R1 has solidified reinforcement learning with verifiable rewards (RLVR) as a potent post-training paradigm for enhancing LLMs across diverse domains (Guo et al., 2025a; Team et al., 2026).

Existing work has demonstrated that incentivized reasoning paths already exist within the base model’s output distribution, suggesting that the reasoning capacity of RLVR-trained models may be fundamentally bounded by the capabilities of the base model (Yue et al., 2025; Zuo et al., 2026; Peng et al., 2025; Chen et al., 2025b). Pre-train Space optimization focuses on strengthening this foundation by directly targeting P(y) (Gupta et al., 2023; Huang et al., 2026). The core value of this paradigm lies in its deep and broad knowledge internalization: by encoding reasoning capability directly into model parameters, it establishes an enhanced

∗Equal contribution †Corresponding author

###### (a) Post-train Space RL Pre-train Space RL

[Figure 1] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile1.png>)

[Figure 2] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile2.png>)

###### Dual Space RL (Pre-train Space→Post-train Space)

[Figure 3] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile3.png>)

###### (b) Post-train Space RL Dual Space RL (c)

[Figure 4] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile4.png>)

[Figure 5] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile5.png>)

1.6x

2.5x

- Figure 1: (a) Gradient objectives of Post-train Space RL, Pre-train Space RL and their combination in Dual Space RL. (b) Conceptual comparison between standard RL (trapped in local optima) and DSRL (broad exploration via NSR pruning toward a refined correct reasoning subspace) (c) Training dynamics of DSRL vs. GRPO, where DSRL achieves higher average accuracy with improved sample efficiency and sustained response length growth driven by NSR-PreRL’s exploration.

foundation that preserves broad exploratory capacity, and provides stronger initialization for subsequent optimization of P(y|x) within the Post-train Space.

However, this paradigm has a fundamental limitation: pre-training relies on static corpora for passive learning, leading to a pronounced distribution shift between pre-trained knowledge and the task distribution encountered during post-training, which hinders targeted reasoning enhancement (Zhou et al., 2025; Paster et al., 2023; Zhang et al., 2025; Wang et al., 2025b). In contrast, RL approaches have achieved remarkable capability gains by shifting the output distribution toward higher-reward trajectories, thereby improving sampling efficiency during online optimization.

This observation motivates a natural idea: introducing RL into the pre-train space to synergize knowledge internalization with reward-driven optimization. Built upon this stronger initialization, post-train RL then focuses on fine-grained optimization of P(y|x), concentrating on conditional trajectory refinement and leading to better exploration.

In this paper, we explore the feasibility of Pre-train Space RL (PreRL). As shown in Figure 1 (a), we optimize the marginal distribution P(y), which removes the question condition during updates. We provide theoretical justification and empirical verification that the gradients of log P(y) and log P(y|x) are strongly aligned, which ensures that optimizing the marginal distribution effectively improves the conditional policy, offering a viable surrogate that naturally preserves the exploration capacity and encourages high rewards trajectories.

Dissecting the roles of positive and negative samples in PreRL (Zhu et al., 2025) reveals a critical asymmetry. Specifically, Positive Sample Reinforcement (PSR) in the pre-train space degrades performance by accumulating probability mass on self-generated correct samples, whereas Negative Sample Reinforcement (NSR) proves to be an exceptionally instructive mechanism. We find that NSR-PreRL rapidly prunes incorrect trajectories in the pre-train space, promoting enhanced exploration in the post-train space. Simultaneously, it stimulates the model’s endogenous reasoning capabilities, with transition and reflection thoughts increasing by 14.89× and 6.54× respectively, effectively broadening the search space through richer probabilistic exploration (Chen et al., 2025a; Zeng et al., 2025).

Leveraging these insights, we adopt the Policy Reincarnation strategy (Liang et al., 2025; Agarwal et al., 2022) to synergize NSR-PreRL with standard RL which learns through pretrain space to post-train space, called Dual Space RL (DSRL). In this framework, the model is initialized via NSR-PreRL to expand its reasoning horizon and elicit intrinsic reasoning

capabilities, and subsequently switched to standard RL for direct policy optimization. Extensive experiments demonstrate that DSRL consistently outperforms strong baselines, achieving superior accuracy with better sample efficiency and robust Pass@K improvements.

- • We introduce Pre-train Space RL (PreRL), a novel paradigm that optimizes the marginal distribution P(y) within RL. We theoretically and empirically validate the alignment between P(y) and P(y|x), establishing PreRL as a viable surrogate for standard RL that naturally preserves the model’s intrinsic exploration capacity.
- • We identify Negative Sample Reinforcement (NSR) in the pre-train space as an exceptionally inspirational mechanism. NSR-PreRL effectively prunes incorrect reasoning paths while stimulating endogenous reasoning behaviors, and achieving comparable accuracy with fewer training steps than standard RL.
- • We adopt the Policy Reincarnation strategy to synergize NSR-PreRL with standard RL into Dual Space RL (DSRL). DSRL consistently outperforms standard RL and achieves higher Pass@K, indicating that pre-train space learning steers the policy toward a refined correct reasoning subspace.

#### 2 From Post-train Space P(y|x) to Pre-train Space P(y)

We first formally define the optimization spaces for LLMs. In the context of RL for LLMs, the policy πθ is equivalent to the output distribution P of LLMs parameterized by θ. Notably, the fundamental distinction between these spaces lies in how the policy is optimized.

###### Definition 1 (Post-train Space Optimization). Given an input distribution X, the paradigm of

Post-train Space optimization is defined as the optimizing the conditional policy πθ(y|x), where the learning objective strictly conditions the update on the specific input query x ∼ X.

###### Definition 2 (Pre-train Space Optimization). The Pre-train Space optimization is defined as

optimizing the marginal policy πθ(y), which directly optimizes the intrinsic distribution of reasoning trajectories. We provide a detailed comparison with traditional pre-training in Appendix A.1.

We visualize the token probability distributions over the vocabulary to analyze their relationship. The conditional and marginal distributions often exhibit similar token rankings, representing aligned cases (Figure 10), though significant misalignments also occur (Figure 11). As detailed in Figure 12, log-probabilities align closely for high-probability, deterministic tokens, but diverge notably for early-sequence or highly uncertain tokens.

###### 2.1 Post-train Space Reinforcement Learning: P(y|x)

Language model generation can be formulated as a token-level Markov Decision Process (MDP). At each step t, the state st = [x; y<t] consists of the input question and generated tokens so far. The policy πθ(·|st) samples the next token yt from vocabulary, transitioning to st+1 = [st; yt]. Generation terminates upon producing [eos] or reaching the budget. To optimize the policy in the post-train space, standard RL maximizes the expected return conditioned on inputs:

[R(y)] − βDKL[πθ(·|x))||πref(·|x)] , (1)

JRL(πθ) = E

###### E

x∼X

y∼πθ(·|x)

where R(y) = ∑|ty=|1 r(yt) is the cumulative return, |y| is the length of y and πref is the reference policy. We adopt sparse rewards where rt = 0 for t < |y| and r|y| ∈ {0, 1} indicates task success. Assuming β = 0 for simplicity following Hu et al. (2025a), the gradient of the objective function is given by the Policy Gradient theorem (Sutton et al., 1998):

|y|

### ∑

∇θ log πθ(yt|x, y<t) · R(y)] . (2)

∇θJRL(πθ) = Ex∼X Ey∼πθ(·|x)[

t=1

Policy Optimization. We employ GRPO (Shao et al., 2024) as our primary optimization algorithm with the optimized token-level policy gradient loss from Yu et al. (2025). For

a) Gradient Dot Product

b) Gradient Cosine Similarity

c) log P(y|x) - log P(y)

50

Max: 46.1835 Min: 0.9386 Mean: 9.2263

Max: 0.7083 Min: 0.0649 Mean: 0.4433

Max: 0.6021 Min: 0.0258 Mean: 0.1569

25

20

40

20

15

30

Frequency

Frequency

Frequency

15

10

20

10

5

10

5

0

0

0

0 10 20 30 40

0.2 0.3 0.4 0.5 0.6

0.1 0.2 0.3 0.4 0.5 0.6

Dot Product

Cosine Similarity

Log Prob Difference

- Figure 2: Synergistic effect analysis of log P(y|x) and log P(y) using Qwen3-4B on AMC23.

(a) Gradient dot product between ∇θ log πθ(y) and ∇θ log πθ(y|x). (b) Gradient cosine similarity. (c) Per-token log probability difference between log P(y|x) and log P(y).

each input query x, GRPO samples a group of G responses {y1, . . . , yG} from the old policy πθold and computes their corresponding returns R = {R1, . . . , RG}. The advantage for each token in response yi is then estimated by normalizing the returns within the group:

Aˆi,t = Ri−stdmean(R)(R). The optimization objective is formulated as:

|yi|

G

1 ∑iG=1 |yi|

min ρi,tAˆi,t,clip(ρi,t,1 − ϵ,1 + ϵ)Aˆi,t ,

### ∑

### ∑

JGRPO(πθ) = Ex∼X,{y

i}iG=1∼πθold

t=1

i=1

(3) where ρi,t = ππθ(yi,t|x,yi,<t)

θold(yi,t|x,yi,<t) denotes the importance sampling ratio, and ϵ is the clipping parameter to constrain policy updates. This group-relative advantage estimation effectively reduces variance and stabilizes training without a value network (Schulman et al., 2015).

###### 2.2 Pre-train Space Reinforcement Learning: P(y)

Pre-training paradigms operate by optimizing the entire sequences to encode knowledge while preserving an effective exploration space. Inspired by this, we propose PreRL (Pretrain Space Reinforcement Learning), which directly optimizes the marginal policy πθ(y) instead of the conditional policy πθ(y|x). The policy gradient of PreRL is defined as:

|y|

### ∑

∇θ log πθ(yt| x, y<t) · R(y) , (4)

∇θJPreRL(πθ) = Ex∼X Ey∼πθ(·|x)

t=1

where x signifies that PreRL removes the input dependency during the gradient update.

However, a critical question remains: how does optimizing the marginal log-likelihood log πθ(yt|y<t) affect the conditional log-likelihood log πθ(yt|x, y<t) required for the task? Following Hu et al. (2025b), our theoretical justification is grounded in a fundamental premise regarding Shared Parameter Influence: the model parameters θ simultaneously govern both the marginal distribution πθ(y) and the conditional distribution πθ(y|x).

Building on this premise, we show that optimizing log πθ(y) improves log πθ(y|x) when their gradients align. Let θ′ = θ + η∇θ log πθ(y) · R(y) denote the parameters after one PreRL step updating the marginal log-probability. The first-order Taylor expansion yields:

###### +O(η2). (5)

log πθ′(y|x) ≈ log πθ(y|x) + η · R(y) · [∇θ log πθ(y)]⊤ ∇θ log πθ(y|x)

cross-gradient term

A core premise underlying PreRL’s effectiveness is that for reasoning trajectories y which semantically aligned with x, the gradients of the marginal and conditional objectives exhibit a non-negative inner product: ⟨∇θ log πθ(y), ∇θ log πθ(y|x)⟩ ≥ 0. This ensures the crossgradient term is non-negative, meaning that updating the marginal log πθ(y) concurrently

[Figure 6] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile6.png>)

- (a)
- (b)

[Figure 7] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile7.png>)

[Figure 8] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile8.png>)

- Figure 3: (a) Training dynamics of PreRL v.s. RL. Effects on reward, response length and top-1 token log-probability. (b) Comparison of distinct reasoning thoughts. NSR-PreRL trains for 20 steps and GRPO for 25 steps. The backbone model is Qwen3-4B.

influences the conditional log πθ(y|x) in the same direction. Consequently, this alignment validates PreRL as an effective surrogate for standard RL. To empirically validate this, we conduct a detailed analysis on Qwen3-4B using 400 rollouts from the AMC23 (MAA, 2023).

###### Observation 1: The gradient inner product is consistently non-negative. We evaluate the

synergy between the marginal and conditional objectives via ⟨∇θ log πθ(y), ∇θ log πθ(y|x)⟩. As Figure 2(a) visualizes, the empirical distribution is entirely non-negative, satisfying the condition for 100% of samples with an average inner product of +9.2. Figure 2(b) reinforces this with a strongly positive cosine similarity distribution. Ultimately, this high gradient alignment confirms that optimizing the πθ(y) inherently benefits the πθ(y|x).

###### Observation 2: The distributions of log πθ(y|x) and log πθ(y) are closely aligned. As

- Figure 2(c) shows, the strongly overlapped distributions validate log πθ(y) as a faithful

surrogate for log πθ(y|x). By operating in the pre-train space, this optimization internalizes question-agnostic reasoning, facilitating broad task transfer and enhanced general capacity.

2.3 In-depth Analysis of Pre-train Space Reinforcement Learning To understand PreRL’s dynamics, we implement it using GRPO as the base algorithm. As

- Figure 3(a) shows, PreRL achieves comparable performance to standard RL for the first 150 steps before suffering a significant collapse. To investigate this failure, we decompose the optimization into positive and negative sample reinforcement (PSR and NSR), following

Zhu et al. (2025). While PSR maximizes log πθ(y|x) to reinforce correct responses, NSR minimizes it to suppress incorrect ones. In PreRL, this dichotomy is governed by the

advantage: samples with Ai > 0 act as positive signals and those with Ai < 0 as negative.

PSR-PreRL suffers from on-policy learning collapse. Unlike PSR-RL, PSR-PreRL targets log πθ(y). While their gradient directions align, their learning dynamics diverge. As Figure 3(a) shows, PSR-PreRL successfully increases the conditional probability πθ(y|x), corroborating their synergistic effect. However, it fails to effectively learn from self-generated on-policy trajectories, ultimately degrading performance. In contrast, QFFT (Liu et al., 2025a) successfully optimizes the same objective maxθ πθ(y) using out-of-distribution longCoT trajectories from a teacher model. This suggests that maximizing πθ(y) in the pre-train space strictly requires high-quality, out-of-distribution expert demonstrations, indicating a potential need for off-policy data integration during RL (Yan et al., 2025).

- Table 1: Avg@32 results on MATH500, AMC23, AIME24, AIME25, Minerva and OlympiadBench. Bold and underlined denote the best and second best.

Methods AMC MATH500 AIME24 AIME25 Minerva OlympiadBench Avg. Qwen3-4B

Vanilla 68.28 80.17 23.13 20.00 23.62 32.33 41.26 PPO 87.89 89.64 47.50 41.15 30.19 40.52 56.15 Reinforce++ 79.37 86.88 35.83 30.00 27.30 38.15 49.59 RLOO 85.00 89.38 45.10 40.62 30.27 40.46 55.14 Dr.GRPO 87.58 89.22 51.04 42.60 31.05 40.75 57.04 DAPO 88.20 89.05 47.08 40.00 30.30 40.28 55.82 GRPO 87.81 89.17 46.46 40.94 30.06 40.29 55.79

- DSRL (Ours) 89.22 89.68 51.15 43.44 30.48 41.26 57.54

Qwen3-8B

Vanilla 69.53 80.24 25.73 19.27 22.33 32.61 41.62 PPO 89.69 89.40 51.98 41.25 30.30 41.45 57.34 Reinforce++ 84.14 88.61 47.40 35.10 29.19 39.52 53.99 RLOO 87.27 90.09 53.85 38.33 29.52 40.65 56.61 Dr.GRPO 87.66 89.71 52.60 37.29 29.22 40.45 56.16 DAPO 88.52 89.60 52.08 39.37 29.08 40.71 56.56 GRPO 88.05 89.91 54.06 39.37 29.65 40.97 57.00

- DSRL (Ours) 90.00 90.31 56.15 42.19 30.32 41.82 58.47

NSR-PreRL alone demonstrates surprising effectiveness. Negative samples contain undesired patterns and receive negative rewards to diminish their probability mass. As Figure 3(a) shows, NSR-PreRL successfully decreases πθ(y|x), supporting our prior analysis. Notably, NSR-PreRL strongly elicits reasoning capabilities, significantly increasing response length. We follow Chen et al. (2025a) to categorize reasoning steps into transition, reflection, and execution thoughts (Table 7). As shown in Figure 3(b), after just 20 NSR-PreRL steps, the model generates 14.89× more transitions and 6.54× more reflections than the vanilla model, substantially outperforming GRPO. This confirms that NSR-PreRL activates internal knowledge for deeper reasoning. Consequently, this enhanced reasoning rapidly boosts performance, achieving an accuracy of 86% with 3× fewer training steps than standard RL.

However, this phenomenon is a double-edged sword. On one hand, it yields better performance and preserves exploration ability by redistributing probability mass away from incorrect trajectories in the pre-train space, effectively pruning wrong reasoning paths (Zhu et al., 2025). On the other hand, it eventually leads to excessively long outputs that hinder continuous training (Chen et al., 2024; Tan et al., 2025a; Wang et al., 2026c).

##### 3 Dual Space RL: Combining PreRL and RL via Policy Reincarnation

Our analysis reveals that NSR-PreRL excels by pruning incorrect reasoning trajectories in the pre-train space while eliciting deeper reasoning. To capitalize on these benefits, we adopt the Policy Reincarnation strategy (Agarwal et al., 2022; Liang et al., 2025; Tan et al., 2025b): midway through training, we replace the base model with the NSR-PreRL checkpoint and restart on-policy RL. We term this sequential framework Dual Space RL (DSRL), as it unifies the pre-train and post-train spaces. Its formulation is given by:

|y|

∇θ log πθ(yt|xI[s>S], y<t) · R(y) · I[s > S ∨ R(y) < 0] ,

### ∑

∇θJDSRL(πθ) = Ex∼X,y∼πθ(·|x)

t=1

(6) where s denotes the current step, and S specifies the transition threshold from pre-train space to post-train space optimization. The term xI[s>S] controls the conditioning: in PreRL phase when s ≤ S, the input x is masked. Crucially, the indicator I[s > S ∨ R(y) < 0] enforces the NSR-PreRL constraint during the initial phase, ensuring that updates are only

AIME25

###### AIME24

###### AMC23

###### MATH500

OlympiadBench

###### Minerva

###### AVG

- 89

- 90

- 91

- 92

- 93

- 94

- 95

- 96

80

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

52

48

85

75

75

80

96

50

45

70

75

70

43

48

Qwen3-4B

65

94

70

40

60

46

65

65

92

38

55

60

44

35

50

DSRL GRPO

55

90

60

42

32

45

50

88

40

30

40

45

55

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

78

50

80

85

52

75

- 90

- 91

- 92

- 93

- 94

- 95

48

75

96

80

73

45

70

50

75

70

###### Qwen3-8B

43

65

94

48

68

70

40

60

92

46

65

38

55

65

62

50

35

44

60

90

60

45

32

42

55

58

40

30

88

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Figure 4: Pass@K performance comparison between DSRL and GRPO across LLMs.

GRPO DSRL

Subgoal Setting

Enumeration

8

8

| | | | | | |
|---|---|---|---|---|---|
| || |
|---|
<br><br>| || |
|---|
|| |
|---|
|| |
|---|
|
|| |
|---|
| || |
|---|
<br><br>| | | |
| | | | | | |

6

6

| |
|---|

AvgCount

| |
|---|

| |
|---|

- 4

4

| |
|---|

| |
|---|

2

2

0

0

0 50 100 150 200 250

0 50 100 150 200 250

Verification

Backtracking

6

18

15

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- 4

12

AvgCount

| |
|---|

| |
|---|

| |
|---|

| |
|---|

9

| |
|---|

| |
|---|

2

6

3

0

0

0 50 100 150 200 250

0 50 100 150 200 250

Training Steps

Training Steps

Figure 5: Reasoning behaviors over RL training with different methods.

[Figure 9] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile9.png>)

Figure 6: Evolution of problem-solving status (Solved vs. Unsolved) on training dataset.

applied to negative samples to prune incorrect reasoning paths, while standard RL utilizes all samples in s > S. In Eq. 6, we employ GRPO as the representative RL algorithm.

##### 4 Experiments

###### 4.1 Experimental Setup

Training Setup. We employ Qwen3-4B and Qwen3-8B (Yang et al., 2025) as base models and train on the MATH dataset (Lewkowycz et al., 2022), which contains 7,500 problems. Detailed training hyperparameters and setups are provided in Appendix B.1.

Evaluation Setup. We evaluate DSRL against several RL baselines, including GRPO (Shao

- et al., 2024), PPO (Schulman et al., 2017), Reinforce++ (Hu, 2025) and RLOO (Ahmadian et al., 2024). Also with the optimized version of GRPO: Dr.GRPO (Liu et al., 2025b) and DAPO (Yu
- et al., 2025) with only clip higher mechanism. The benchmarks cover: MATH500 (Lightman

- et al., 2023), AMC23 (MAA, 2023), AIME24 (MAA, 2024), AIME25 (MAA, 2025), Minerva (Lewkowycz et al., 2022) and OlympiadBench (He et al., 2024). Due to high output variance, we report Avg@32 and Pass@K (K ∈ [1,256]) estimated from n = 300 samples per problem (Chen et al., 2021). We report more evaluation details in Appendix B.1.

###### 4.2 Main Results

As shown in Table 1, DSRL consistently improves Avg@K over strong RL baselines across all benchmarks and models. For Qwen3-4B, DSRL significantly outperforms GRPO by 4.69 points on AIME24 and 2.50 points on AIME25. It also surpasses advanced methods like Dr.GRPO and DAPO by 0.50 and 1.72 points on average, respectively. Similarly, Qwen3-8B demonstrates robust gains, achieving a best average score of 58.47. Overall, these results indicate that initializing with NSR-PreRL internalizes crucial knowledge to build a stronger foundation, thereby enabling subsequent RL to guide the policy toward higher accuracy.

- Table 2: OOD generalization results. Bold denotes the best.

57.74

- 55

- 56

- 57

- 58

57.54

57.27

57.11

56.32

AvgScore

Methods GPQA-Diamond MMLU-Pro BBH HumanEval Qwen3-4B

55.69

GRPO 55.79

Vanilla 42.12 58.61 75.55 78.66 GRPO 39.39 61.12 80.37 81.10 DSRL (Ours) 43.18 66.49 82.41 81.10 ∆ vs. GRPO +3.79 +5.37 +2.04 +0.00

5 10 15 20 25 30

Warmup Steps

Qwen3-8B

Vanilla 45.86 59.72 79.57 87.20 GRPO 50.96 59.47 82.35 87.80 DSRL (Ours) 53.48 63.79 82.31 90.24 ∆ vs. GRPO +2.52 +4.32 -0.04 +2.44

Figure 7: Ablation on warmup steps. Bar color intensity reflects score magnitude.

To comprehensively evaluate DSRL, we compare its Pass@K performance against GRPO (Figure 4). DSRL consistently outperforms the baseline across the entire range of K. On Qwen3-8B, DSRL achieves superior results for all K, maintaining a clear lead on challenging benchmarks like AIME24 and AIME25 as K increases. This robust sampling scalability indicates that DSRL not only boosts Pass@1 but also diversifies the high-quality solution space. Similarly, Qwen3-4B results show DSRL outperforming the baseline across most K budgets, efficiently concentrating probability mass on correct reasoning paths. Ultimately, these findings demonstrate that DSRL’s pre-train space optimization fosters a more exploration-friendly policy landscape, fundamentally enhancing reasoning capacity.

To evaluate generalization beyond mathematical reasoning, we test on four out-ofdistribution (OOD) benchmarks: GPQA-Diamond (Rein et al., 2024), MMLU-Pro (Wang

- et al., 2024a), BBH (Suzgun et al., 2023) and HumanEval (Chen et al., 2021). As Table 2 shows, DSRL consistently matches and outperforms GRPO across both scales. Specifically, DSRL achieves substantial gains on knowledge-intensive tasks, improving GPQA-Diamond by

+3.79 and MMLU-Pro by +5.37 for Qwen3-4B, with corresponding gains of +2.52 and +4.32 for Qwen3-8B. DSRL also generalizes well to code generation, achieving a +2.44 improvement on HumanEval for Qwen3-8B. These results indicate that pre-train space optimization not only enhances in-domain reasoning but also cultivates a highly generalizable policy with superior OOD transferability.

###### 4.3 Analysis

Training Efficiency in Dual-Space RL. Figure 1(c) illustrates DSRL’s superior training dynamics over GRPO across three dimensions. First, DSRL consistently leads in performance, achieving a 61.6 average accuracy vs. GRPO’s 57.7. Second, it demonstrates exceptional efficiency by reaching 45.0% and 58.0% accuracy with 2.5× and 1.6× fewer steps, respectively.

Evolution of Reasoning Behaviors. Following Zeng et al. (2025), we track four key reasoning behaviors on AIME25 (Subgoal Setting, Enumeration, Verification, Backtracking). In Figure 5, RL struggles with slow and limited behavior emergence due to its constrained initial warmup. Conversely, DSRL drives rapid, sustained growth and achieves significantly higher frequency ceilings across all patterns. This confirms that pre-train space optimization effectively removes conditional constraints, unlocking intrinsic capacities for both rigorous self-correction and complex, structured reasoning exploration (Chen et al., 2025a).

Foundation Reasoning Enhanced via NSR-PreRL. To assess foundational problem-solving abilities, we track the count of Fully Solved (all rollouts correct) and Fully Unsolved (all rollouts incorrect) questions during training. These metrics indicate whether the model has genuinely internalized core error patterns rather than succeeding by chance. In Figure 6, DSRL achieves rapid mastery during the NSR-PreRL phase, marked by a sharp surge in Fully Solved counts and a decline in Fully Unsolved ones. This proves that NSR-PreRL systematically eliminates failure modes by internalizing fundamental error logic. This efficiency strongly validates our policy reincarnation strategy. By pruning universal incorrect patterns during NSR-PreRL, DSRL establishes a robust foundation. The subsequent RL

phase can then focus on refining problem-specific nuances. Consequently, DSRL ultimately achieves more Fully Solved questions and a lower error rate, confirming that pre-train space optimization elevates the policy’s reasoning ceiling beyond RL limits.

- 4.4 Ablation Study

Impact of Warmup Steps. We investigate the optimal duration for the NSR-PreRL warmup phase in Figure 7. The results exhibit an inverted-U trend, with 10 to 25 steps emerging as the optimal range. Conversely, fewer steps provide insufficient stimulation, while excessive warmup acts as a double-edged sword, causing over-exploratory generation that hinder subsequent fine-grained optimization in the post-train space. Thus, a carefully chosen transition point S ∈ [10,25] is crucial for the success of policy reincarnation.

Table 3: Ablation on warmup strategy.

Methods Avg. Qwen3-4B

Vanilla 41.26 GRPO 55.79 NSR-RL Warmup 54.38 NSR-PreRL Warmup (DSRL) 57.54

Pre-train Space vs. Post-train Space Warmup. To investigate whether negative samples reinforcement in the post-train space can achieve similar benefits, we compare DSRL against an NSR-RL warmup baseline for both 20 steps. As shown in Table 3, NSR-RL warmup paradoxically underperforms GRPO with an average score of 54.38 compared to 55.79, whereas DSRL achieves the highest score of 57.54. As shown in Figure 13, by step 20, NSR-PreRL outperforms NSR-RL by an average of 6.6 points across the AMC23, AIME24, and AIME25, providing a vastly superior initialization for the subsequent RL phase.

- 5 Related Works

- 5.1 Reinforcement learning with verifiable rewards

Reinforcement learning with verifiable rewards (RLVR) (Guo et al., 2025a) has emerged as an effective post-training paradigm for enhancing LLMs reasoning (Shao et al., 2024; Cai & Sugiyama, 2026; Zhao et al., 2025; Liu et al., 2026; Wang et al., 2026a; Fang et al., 2026b;a). Since RLVR has been argued to merely over-sharpen the sample distribution of the base model, potentially constraining the exploration space (Yue et al., 2025), a complementary line of work investigates the balance between exploration and exploitation in the post-train space RL (Li et al., 2025a; Wang et al., 2026b; Yu et al., 2026; Guo et al., 2025b). While these studies offer insights into how reward signals shape policy behavior, their scope remains confined to the conditional distribution P(y|x). Whether and how such reward-driven mechanisms can operate on the marginal distribution P(y) remains unexplored.

###### 5.2 Pre-train Space Optimization

Directly training LLMs on downstream tasks may suffer from distribution shift induced by novel data (Cossu et al., 2024). Continual pre-training addresses this by encoding broad knowledge into parameters, optimizing the marginal distribution P(y) through offline training on static corpora, and thereby providing a stronger foundation for downstream tasks (Sun et al., 2020; Cossu et al., 2024; Ou et al., 2025; Su et al., 2025). Such pre-train space optimization establishes general knowledge enhancement, preserving broad exploration capacity that is conducive to subsequent RL (Huang et al., 2026; Wang et al., 2024b; 2025b; Zhou et al., 2025). Our work departs from them by bringing online RL directly into the pre-train space, transforming knowledge internalization from a passive, data-driven process into an active, reward-guided one. We also provide a detailed comparison between PreRL and the emerging Reinforcement Learning Pre-Training (RLPT) paradigm in Appendix A.2.

##### 6 Conclusion

In this paper, we propose Pre-train Space RL (PreRL), integrating reward-driven reinforcement learning into the pre-train space. We theoretically and empirically demonstrate a strong gradient alignment between the conditional objective log P(y|x) and the marginal objective log P(y). Dissecting the PreRL, we reveal that Negative Sample Reinforcement (NSR) acts as an exceptionally effective driver by selectively pruning incorrect reasoning paths and eliciting endogenous reasoning capabilities. Building on this, we introduce Dual Space RL (DSRL), which uses a Policy Reincarnation strategy to synergize NSR-PreRL with standard RL. By initializing with NSR-PreRL to eliminate fundamental errors and expand the exploration horizon, DSRL consistently outperforms strong baselines across extensive benchmarks. Ultimately, our work proves that pre-train space optimization inherently establishes a more robust reasoning foundation for superior final performance.

##### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. URL https://arxiv.org/ab s/2303.08774.

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C Courville, and Marc Bellemare. Reincarnating reinforcement learning: Reusing prior computation to accelerate progress. Advances in neural information processing systems, 35:28955–28971, 2022. URL https://proceedings.neurips.cc/paper files/paper/2022/hash/ba1c5356d9164bb64 c446a4b690226b0-Abstract-Conference.html.

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨ un,¨ and Sara Hooker. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings of ACL, pp. 12248–12267, 2024. URL https://aclanthology.org/2024.acl-long.662/.

Marcin Andrychowicz, Anton Raichuk, Piotr Stanczyk,´ Manu Orsini, Sertan Girgin, Raphael Marinier, L´eonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al. What matters in on-policy reinforcement learning? a large-scale empirical study. arXiv preprint arXiv:2006.05990, 2020. URL https://arxiv.org/abs/2006.05990.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. URL https://proceedings.neurips.cc/paper files/paper/2020/ hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html?utm source=transaction&ut

- m medium=email&utm campaign=linkedin newsletter.

Xin-Qiang Cai and Masashi Sugiyama. Vi-curl: Stabilizing verifier-independent rl reasoning via confidence-guided variance reduction. arXiv preprint arXiv:2602.12579, 2026. URL https://arxiv.org/abs/2602.12579.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, and Zhangyang Wang. Seal: Steerable reasoning calibration of large language models for free. arXiv preprint arXiv:2504.07986, 2025a. URL https://arxiv.org/abs/2504.07986.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187, 2024. URL https://arxiv.org/abs/2412.21187.

Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@ k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025b. URL https://arxiv.org/abs/ 2508.10751.

Andrea Cossu, Antonio Carta, Lucia Passaro, Vincenzo Lomonaco, Tinne Tuytelaars, and Davide Bacciu. Continual pre-training mitigates forgetting in language and vision. Neural Networks, 179:106492, 2024. URL https://www.sciencedirect.com/science/article/pi i/S0893608024004167.

Google DeepMind. Gemini 2.0 flash thinking, 2024. URL https://deepmind.google/tech nologies/gemini/flash-thinking/.

Qingxiu Dong, Li Dong, Yao Tang, Tianzhu Ye, Yutao Sun, Zhifang Sui, and Furu Wei. Reinforcement pre-training. arXiv preprint arXiv:2506.08007, 2025. URL https://arxiv. org/abs/2506.08007.

Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chaowen Hu, Lu Pan, Ke Zeng, and Xunliang Cai. How to allocate, how to learn? dynamic rollout allocation and advantage modulation for policy optimization. arXiv preprint arXiv:2602.19208, 2026a. URL https://arxiv.org/abs/2602.19208.

Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chang Liu, and Peilin Zhao. Proximity-based multi-turn optimization: Practical credit assignment for llm agent training. arXiv preprint arXiv:2602.19225, 2026b. URL https://arxiv.org/abs/2602.19225.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025a. URL https: //www.nature.com/articles/s41586-025-09422-z.

Ruohao Guo, Afshin Oroojlooy, Roshan Sridhar, Miguel Ballesteros, Alan Ritter, and Dan Roth. Tree-based dialogue reinforced policy optimization for red-teaming attacks. arXiv preprint arXiv:2510.02286, 2025b. URL https://arxiv.org/abs/2510.02286.

Benjamin Gupta, Kshitij ou2025llmsand Th´erien, Adam Ibrahim, Mats L Richter, Quentin Anthony, Eugene Belilovsky, Irina Rish, and Timoth´ee Lesort. Continual pre-training of large language models: How to (re) warm your model? arXiv preprint arXiv:2308.04014, 2023. URL https://arxiv.org/abs/2308.04014.

Ali Hatamizadeh, Syeda Nahida Akter, Shrimai Prabhumoye, Jan Kautz, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, and Yejin Choi. Rlp: Reinforcement as a pretraining objective. arXiv preprint arXiv:2510.01265, 2025. URL https://arxiv.org/abs/2510.0 1265.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024. URL https://aclanthology.org/2024.acl-long.21 1/.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025. URL https://arxiv.org/abs/2501.03262.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025a. URL https: //arxiv.org/abs/2503.24290.

Jinwu Hu, Zhitian Zhang, Guohao Chen, Xutao Wen, Chao Shuai, Wei Luo, Bin Xiao, Yuanqing Li, and Mingkui Tan. Test-time learning for large language models. arXiv preprint arXiv:2505.20633, 2025b. URL https://arxiv.org/abs/2505.20633.

Junjie Huang, Jiarui Qin, Di Yin, Weiwen Liu, Yong Yu, Xing Sun, and Weinan Zhang. Remit: Rl-guided mid-training for iterative llm evolution. arXiv preprint arXiv:2602.03075, 2026. URL https://arxiv.org/abs/2602.03075.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. URL https://arxiv.org/abs/2410.21276.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of SOSP, pp. 611–626, 2023. URL https://dl.acm.org/doi/abs/10.1145/3600006.3613165.

Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. In Advances in Neural Information Processing Systems, 2022. URL https:

//openreview.net/forum?id=IFXTZERXdM7.

Long Li, Jiaran Hao, Jason Klein Liu, Zhijian Zhou, Yanting Miao, Wei Pang, Xiaoyu Tan, Wei Chu, Zhe Wang, Shirui Pan, et al. The choice of divergence: A neglected key to mitigating diversity collapse in reinforcement learning with verifiable reward. arXiv preprint arXiv:2509.07430, 2025a. URL https://arxiv.org/abs/2509.07430.

Siheng Li, Kejiao Li, Zenan Xu, Guanhua Huang, Evander Yang, Kun Li, Haoyuan Wu, Jiajia Wu, Zihao Zheng, Chenchen Zhang, et al. Reinforcement learning on pre-training data. arXiv preprint arXiv:2509.19249, 2025b. URL https://arxiv.org/abs/2509.19249.

Jing Liang, Hongyao Tang, Yi Ma, Jinyi Liu, Yan Zheng, Shuyue Hu, Lei Bai, and Jianye Hao. Squeeze the soaked sponge: Efficient off-policy reinforcement finetuning for large language model. arXiv preprint arXiv:2507.06892, 2025. URL https://arxiv.org/abs/25 07.06892.

Huanxuan Liao, Zhongtao Jiang, Yupu Hao, Yuqiao Tan, Shizhu He, Jun Zhao, Kun Xu, and Kang Liu. Resadapt: Adaptive resolution for efficient multimodal reasoning. arXiv preprint arXiv:2603.28610, 2026. URL https://arxiv.org/abs/2603.28610.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In Proceedings of ICLR, 2023. URL https://openreview.net/forum?id=v8L0pN6EOi.

Wanlong Liu, Junxiao Xu, Fei Yu, Yukang Lin, Ke Ji, Wenyu Chen, Lifeng Shang, Yasheng Wang, Yan Xu, and Benyou Wang. Qfft, question-free fine-tuning for adaptive reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https://openreview.net/forum?id=CrBWOjZoKc.

Weiting Liu, Han Wu, Yufei Kuang, Xiongwei Han, Tao Zhong, Jianfeng Feng, and Wenlian Lu. Automated optimization modeling via a localizable error-driven perspective. arXiv preprint arXiv:2602.11164, 2026. URL https://arxiv.org/abs/2602.11164.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In Proceedings of COLM, 2025b. URL https://openreview.net/forum?id=5PAF7PAY2Y.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858, 2025. URL https://arxiv.org/abs/2504.09858.

MAA. American mathematics contest 12 (amc 12), November 2023. URL https://artofpro blemsolving.com/wiki/index.php/AMC 12 Problems and Solutions.

- MAA. American invitational mathematics examination (aime), February 2024. URL https: //artofproblemsolving.com/wiki/index.php/AIME Problems and Solutions.

- MAA. American invitational mathematics examination (aime), February 2025. URL https: //artofproblemsolving.com/wiki/index.php/AIME Problems and Solutions.

Yixin Ou, Yunzhi Yao, Ningyu Zhang, Hui Jin, Jiacheng Sun, Shumin Deng, Zhenguo Li, and Huajun Chen. How do llms acquire new knowledge? a knowledge circuits perspective on continual pre-training. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 19889–19913, 2025. URL https://aclanthology.org/2025.findings-acl.1021/.

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. In The Twelfth International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=jKHmjlpViu.

Ruotian Peng, Yi Ren, Zhouliang Yu, Weiyang Liu, and Yandong Wen. Simko: Simple pass@ k policy optimization. arXiv preprint arXiv:2510.14807, 2025. URL https://arxiv.org/ab s/2510.14807.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. URL https://storage.prod.researchhub.com/uploads/papers/2020/06/01/language-model s.pdf.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. URL http://www.jmlr.org/papers/v21/20-074.html.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level googleproof q&a benchmark. In First conference on language modeling, 2024. URL https://openre view.net/forum?id=Ti67584b98&utm campaign=The%20Batch&utm source=hs email&utm

medium=email& hsenc=p2ANqtz-90r63gYMgFYuSQZ9JoytPrtpZaoBY-JZFgXvnw6DWrSpcX8 f71dkkwBrUJo8TTxNEsIv7i.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015. URL https://arxiv.org/abs/1506.02438.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. URL https://arxi v.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. URL https://arxiv.org/abs/2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025. URL https://dl.acm.org/doi/abs/10.1145/3689031.3696075.

Liangcai Su, Zhen Zhang, Guangyu Li, Zhuo Chen, Chenxi Wang, Maojia Song, Xinyu Wang, Kuan Li, Jialong Wu, Xuanzhong Chen, et al. Scaling agents via continual pre-training. arXiv preprint arXiv:2509.13310, 2025. URL https://arxiv.org/abs/2509.13310.

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, and Haifeng Wang. Ernie 2.0: A continual pre-training framework for language understanding. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 8968–8975, 2020. URL https://ojs.aaai.org/index.php/aaai/article/view/6428.

Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. Challenging bigbench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13003–13051, 2023. URL https://aclantholo gy.org/2023.findings-acl.824/.

Yuqiao Tan, Shizhu He, Kang Liu, and Jun Zhao. The zero-step thinking: An empirical study of mode selection as harder early exit in reasoning models. In NeurIPS 2025 Workshop on Efficient Reasoning, 2025a. URL https://openreview.net/forum?id=CPXmurtK0H.

Yuqiao Tan, Minzheng Wang, Shizhu He, Huanxuan Liao, Chengfeng Zhao, Qiunan Lu, Tian Liang, Jun Zhao, and Kang Liu. Bottom-up policy optimization: Your language model policy secretly contains internal policies. arXiv preprint arXiv:2512.19673, 2025b. URL https://arxiv.org/abs/2512.19673.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026. URL https://arxiv.org/abs/2602.02276.

Minzheng Wang, Yongbin Li, Haobo Wang, Xinghua Zhang, Nan Xu, Bingli Wu, Fei Huang, Haiyang Yu, and Wenji Mao. Adaptive social learning via mode policy optimization for language agents. In The Fourteenth International Conference on Learning Representations, 2026a. URL https://openreview.net/forum?id=GG7YQnsdhp.

Tianyi Wang, Long Li, Hongcan Guo, Yibiao Chen, Yixia Li, Yong Wang, Yun Chen, and Guanhua Chen. Anchored policy optimization: Mitigating exploration collapse via support-constrained rectification. arXiv preprint arXiv:2602.05717, 2026b. URL https: //arxiv.org/abs/2602.05717.

Yanbo Wang, Yongcan Yu, Jian Liang, and Ran He. A comprehensive survey on trustworthiness in reasoning with large language models, 2025a. URL https://arxiv.org/abs/2509

.03871.

Yanbo Wang, Minzheng Wang, Jian Liang, Lu Wang, Yongcan Yu, and Ran He. Mitigating the safety-utility trade-off in llm alignment via adaptive safe context learning, 2026c. URL https://arxiv.org/abs/2602.13562.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024a. URL https://proceedings.neurip s.cc/paper files/paper/2024/hash/ad236edc564f3e3156e1b2feafb99a24-Abstract-D atasets and Benchmarks Track.html.

Zengzhi Wang, Xuefeng Li, Rui Xia, and Pengfei Liu. Mathpile: A billion-token-scale pretraining corpus for math. Advances in Neural Information Processing Systems, 37:25426– 25468, 2024b. URL https://proceedings.neurips.cc/paper files/paper/2024/hash/2 d0be3cd5173c10b6ec075d1c393a13d-Abstract-Datasets and Benchmarks Track.html.

Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025b. URL https://arxiv.org/abs/2506.20512.

Xingrun Xing, Zhiyuan Fan, Jie Lou, Guoqi Li, Jiajun Zhang, and Debing Zhang. Pretrainzero: Reinforcement active pretraining. arXiv preprint arXiv:2512.03442, 2025. URL https://arxiv.org/abs/2512.03442.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945,

2025. URL https://arxiv.org/abs/2504.14945.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https://arxiv.org/abs/2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. URL https://arxiv.org/ abs/2503.14476.

Zhiqi Yu, Zhangquan Chen, Mengting Liu, Heye Zhang, and Liangqiong Qu. Unveiling implicit advantage symmetry: Why grpo struggles with exploration and difficulty adaptation. arXiv preprint arXiv:2602.05548, 2026. URL https://arxiv.org/abs/2602.05548.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=4OsgYD7em5.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025. URL https://arxiv.org/abs/2503.1 8892.

Charlie Zhang, Graham Neubig, and Xiang Yue. On the interplay of pre-training, midtraining, and rl on reasoning language models. arXiv preprint arXiv:2512.07783, 2025. URL https://arxiv.org/abs/2512.07783.

Fei Zhao, Chonggang Lu, Zheyong Xie, Ziyan Liu, Haofu Qian, Jianzhao Huang, Fangcheng Shi, Zijie Meng, Hongcheng Guo, Mingqian He, et al. Redone: Revealing domain-specific llm post-training in social networking services. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 2648–2674, 2025. URL https://aclanthology.org/2025.emnlp-industry.180/.

Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, and Eric P Xing. Megamath: Pushing the limits of open math corpora. In Second Conference on Language Modeling, 2025. URL https://openreview.net/for um?id=SHB0sLrZrh.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in LLM reasoning. In Proceedings of NeurIPS, 2025. URL https://openreview.net/forum?id=ftVlLG9cks.

Yuxin Zuo, Bingxiang He, Zeyuan Liu, Shangziqi Zhao, Zixuan Fu, Junlin Yang, Kaiyan Zhang, Yuchen Fan, Ganqu Cui, Cheng Qian, et al. How far can unsupervised rlvr scale llm training? In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=VesLZukY5E.

##### A Detailed Comparison with Related Works

###### A.1 Comparison of PreRL with Pre-training and Continual Pre-training

While Pre-training, Continual Pre-training, and PreRL all operate in the pre-train space by optimizing the marginal distribution P(y), they differ fundamentally in their data sources, learning signals, and knowledge acquisition mechanisms. We summarize the key distinctions in Table 4.

- Table 4: Comparison of Pre-training, Continual Pre-training, and PreRL. All three paradigms optimize the marginal distribution P(y) in the pre-train space, but differ in data sources, learning signals, and training paradigms.

Dimension Pre-training Continual Pre-training PreRL (Ours)

Optimization Target P(y) P(y) P(y) Training Data General web text Task-specific corpora Reasoning tasks Data Format Raw documents Documents / QA pairs Response-only trajectories Learning Signal Next-token prediction Next-token prediction Verifiable rewards Learning Paradigm Offline, passive Offline, passive Online, active Task Alignment Low Moderate High

Pre-training establishes general language capabilities by learning from large-scale web corpora (e.g., Common Crawl, Wikipedia) through Next-Token Prediction (NTP) (Brown et al., 2020; Radford et al., 2019; Raffel et al., 2020). While it builds broad linguistic knowledge, the resulting distribution P(y) is not aligned with downstream reasoning tasks, leading to a significant distribution gap when transitioning to post-training.

Continual Pre-training narrows this gap by further training on task-specific corpora, which may include domain-relevant documents, general-purpose responses, or question-answer pairs (Gupta et al., 2023; Wang et al., 2024b; Zhou et al., 2025). Although this approach injects targeted knowledge into P(y), it still relies on static, pre-collected data and passive NTP, limiting its ability to adapt to the model’s evolving capabilities during training.

PreRL fundamentally departs from both paradigms by replacing passive data consumption with active, reward-guided learning. Rather than training on external corpora, PreRL generates reasoning trajectories online through self-rollouts and uses verifiable rewards to selectively reinforce or suppress these trajectories. Crucially, PreRL updates only on the response y with the input question x removed, maintaining the pre-train space objective of optimizing P(y). This design transforms knowledge internalization from a static, datadriven process into a dynamic, feedback-driven one, ensuring that the optimized P(y) remains tightly aligned with the task distribution encountered during subsequent RL.

###### A.2 Comparison with Reinforcement Pre-Training Paradigm

To overcome the substantial verification data-wall, Reinforcement Learning Pre-Training (RLPT) has recently emerged as a promising direction. Several methods have attempted to integrate RL objectives directly into the pre-training phase (Hatamizadeh et al., 2025; Dong et al., 2025; Li et al., 2025b; Xing et al., 2025). While these RLPT methods share the conceptual goal of applying RL beyond the standard post-training phase to enhance foundational reasoning capabilities, our proposed PreRL fundamentally differs from the general RLPT paradigm in data sources, learning signals, and optimization spaces. We summarize the key distinctions in Table 5.

To overcome the substantial verification data-wall, Reinforcement Learning Pre-Training (RLPT) has recently emerged as a promising direction. Several methods have attempted to conduct RL directly on the pre-training corpus (Hatamizadeh et al., 2025; Dong et al., 2025; Li et al., 2025b; Xing et al., 2025). While these RLPT methods can be regarded as a effective warmup to subsequent RL, our proposed PreRL fundamentally differs from the general RLPT paradigm in training data, learning signals, and optimization spaces and so on. We summarize the key distinctions in Table 5.

- Table 5: Comparison of the Reinforcement Pre-Training (RLPT) paradigm and PreRL. PreRL distinguishes itself by operating on real reasoning tasks with truly verifiable rewards, strictly optimizing the marginal distribution P(y) through active online exploration.

Dimension RLPT Paradigm PreRL (Ours)

Optimization Target P(y|x) P(y) Training Data Pre-training corpora Reasoning tasks Data Format Full Reasoning trajectories Response-only trajectories Learning Signal Next-token prediction Verifiable rewards Learning Paradigm Online, active Online, active Task Alignment Moderate High

Reinforcement Pre-Training (RLPT) operates primarily on static pre-training corpora. Since raw corpora lack objective verification signals, RLPT methods typically convert the Next-Token Prediction (NTP) task into a pseudo-reward. Specifically, the model generates self-rollouts as intermediate thinking, and the ultimate reward is determined by whether these rollouts successfully predict the subsequent ground-truth tokens. While this approach attempts to elicit reasoning, it is fundamentally constrained by the proxy nature of the NTP reward. Unlike mathematical reasoning tasks that offer strictly verifiable and definitive answers, next-token continuations are inherently open-ended with multiple valid possibilities. Consequently, relying on a single static continuation renders the optimization objective simplistic and lacking in rigorous logical verification.

PreRL fundamentally departs from the RLPT paradigm by operating directly within the standard RLVR framework on reasoning tasks. Instead of relying on static dataset continuations and NTP pseudo-rewards, PreRL utilizes objective task success as the verifiable reward. Crucially, the fundamental distinction lies in the optimization space: we restrict the policy update strictly to the pre-train space by masking the question during gradient updates. This allows PreRL to actively explore reasoning trajectories online and effectively learn through verifiable rewards. By doing so, PreRL answers the foundational question of how to enhance reasoning capabilities autonomously, without the confounding factors of supervised interventions or static data grounding.

##### B Detailed Experiment Settings

###### B.1 Training and Evaluation Setup

Training Setup. We employ Qwen3-4B and Qwen3-8B (Yang et al., 2025) as the base models for all experiments, since they exhibit strong capability in advanced reasoning and use non-thinking mode (Ma et al., 2025). For the training set, we use MATH (Lewkowycz et al., 2022), which contains 7,500 problems. We train the models using the verl framework (Sheng et al., 2025). The prompt batch size is 128, with 8 rollouts generated per prompt. The sampling temperature during training is set to 1.0, and the maximum response length is set to 8192 tokens. We update the model with a mini-batch size of 128 and a learning rate of 1e-6 to conduct on-policy RL for 6 epochs (Andrychowicz et al., 2020). We sample a group of responses per question and estimate advantages using the group-based normalization.

Evaluation Setup. For evaluation, We use vLLM (Kwon et al., 2023) with temperature=1.0 and top p=1.0. Due to high output variance in reasoning tasks, we report Avg@32 (Pass@1 averaged over 32 rollouts). For Pass@K, which is defined as Pass@K :=

Ex∼D 1 − (nK−c)/(Kn) , where c denotes the number of correct completions out of n generated responses (Chen et al., 2021). To reduce evaluation variance on those datasets, we set

n = 300.

###### B.2 The Template for Experiments

We adopt the following template for all experiments involving Qwen models, building the non-thinking template for Qwen3 (Yang et al., 2025).

Table 6: RL Hyperparameters

###### Hyperparameter Value

Optimizer AdamW Policy learning rate 1e−6 Training batch size 128 Samples per prompt 8 Mini-batch size 128 Training epochs 6 Max prompt length 1024 tokens Max response length 8192 tokens Rollout temperature 1.0

Qwen3 non-thinking template <|im start|>system Please reason step by step, and put your final answer within \boxed{}. <|im end|> <|im start|>user {problem} <|im end|> <|im start|>assistant <think> </think>

Figure 8: The Qwen3-NoThinking prompt template used for inference.

###### B.3 Implementation of Reflection, Transition, and Execution Thoughts

Following the reflection, transition, execution taxonomy defined in Chen et al. (2025a), we classify each reasoning step into one of three categories using two types of heuristic rules. As shown in Table 7, the first is a prefix rule: steps beginning with keywords such as ’wait’ or ’alternatively’ are classified as transition or reflection. The second is a phrase rule: steps containing predefined mid-sentence phrases indicative of self-correction or redirection are likewise classified as transition or reflection. Steps satisfying neither criterion are classified as execution. Classification is case-insensitive. The proportion of each category is then computed over all steps in the generated responses.

Table 7: Extraction of reflection, transition, and execution thoughts. Category Type Keywords / Phrases Transition

Prefix Alternatively Phrase

think differently, another way, another approach, another method, another solution, another strategy, another technique

Prefix Wait Phrase

Reflection

verify, make sure, hold on, think again, ’s correct, ’s incorrect, Let me check, seems right

###### B.4 Implementation of Reasoning Behaviors

To analyze the evolution of reasoning behaviors during training, we follow Zeng et al. (2025) and use GPT-4o (Hurst et al., 2024) to identify four key cognitive behaviors in model

responses, with the annotation prompt provided in Figure 9. We report the proportion of responses exhibiting each behavior throughout training.

- • Backtracking: The model actively identifies errors during response generation and explicitly revises previously used methods.
- • Verification: The model systematically checks intermediate results to ensure correctness.
- • Subgoal Setting: The model decomposes complex problems into smaller, manageable steps.
- • Enumeration: The model exhaustively considers multiple cases or possibilities to solve problems.

###### Reasoning Evaluation Prompt

Below is a chain-of-reasoning generated by a Language Model when attempting to solve a math problem. Evaluate this chain-of-reasoning to determine whether it demonstrates beneficial problem-solving behaviors that deviate from typical linear, monotonic reasoning patterns.

<start of reasoning> {segment text} <end of reasoning>

Specifically, identify and emphasize beneficial behaviors such as:

- • Backtracking: Explicitly revising approaches upon identifying errors or dead ends.
- • Verification: Systematically checking intermediate results or reasoning steps.
- • Subgoal Setting: Breaking down complex problems into smaller, manageable steps.
- • Enumeration: Exhaustively considering multiple cases or possibilities.

Additionally, remain attentive to other beneficial behaviors such as creative analogies, abstraction to simpler cases, or insightful generalizations. If no beneficial behaviors are observed, explicitly return ‘none’.

Output format: {

"behaviour": "", "example": ""

}

- Figure 9: System prompt used to evaluate beneficial reasoning behaviors in model-generated chain-of-thought, targeting non-linear problem-solving patterns including backtracking, verification, subgoal setting, and enumeration.

 : In a table tennis tournament every participant played every other participant exactly once...What is the total number of games played?  :... (since there are twice as many right-handed players)\nSo

, (Predicted token)

Post-train Space Conditional Distribution  ( | )

Pre-train Space Marginal Distribution  ( )

[Figure 10] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile10.png>)

[Figure 11] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile11.png>)

。1

Vocabulary Vocabulary

- Figure 10: Aligned case: Token probability distribution with and without input conditioning under the same context. x is a math problem and y is a partially generated answer. The two distributions P(y|x) and P(y) show similar token rankings, suggesting the post-train space conditional distribution largely inherits the structure of the pre-train space marginal distribution.

[Figure 12] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile12.png>)

[Figure 13] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile13.png>)

 : In a table tennis tournament every participant played every other participant exactly once...What is the total number of games played?  :... ### ✅ Final Answer:\n$$\n\boxed{36}\n$$\nSo the total (Predicted token)

Post-train Space Conditional Distribution  ( | )

Pre-train Space Marginal Distribution  ( )

。1

Vocabulary Vocabulary

- Figure 11: Misaligned case: Token probability distribution with and without input conditioning under the same context. x is a math problem and y is a partially generated answer. The top-1 token ”total” under P(y|x) receives near-zero probability under P(y), indicating a significant discrepancy between the two distributions.

[Figure 14] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile14.png>)

- (a) Conditional token log-probabilities: log P(y|x).

[Figure 15] to P(y) Investigating Reinforcement Learning in Pre-train Space_images/imageFile15.png>)

- (b) Marginal token log-probabilities: log P(y).

- Figure 12: Comparison of token-level generation log-probabilities with and without input conditioning.

Figure 13: Training dynamics of NSR-PreRL warmup vs. NSR-RL warmup.

