# arXiv:2509.06923v1[cs.LG]8Sep2025

STAYING IN THE SWEET SPOT: RESPONSIVE REASONING EVOLUTION VIA CAPABILITY-ADAPTIVE HINT SCAFFOLDING

Ziheng Li1∗, Zexu Sun2∗, Jinman Zhao3, Erxue Min2, Yongcheng Zeng4, Hui Wu5, Hengyi Cai2, Shuaiqiang Wang2, Dawei Yin2, Xu Chen6†, Zhi-Hong Deng1†

- 1 School of Intelligence Science and Technology, Peking University
- 2 Baidu Inc. 3 Department of Computer Science, University of Toronto

- 4 Institute of Automation, Chinese Academy of Sciences
- 5 Aerospace Information Research Institute, Chinese Academy of Sciences
- 6 Gaoling School of Artificial Intelligence, Renmin University of China {liziheng, zhdeng}@pku.edu.cn, sunzexu0826@gmail.com, xu.chen@ruc.edu.cn

ABSTRACT

Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in enhancing the reasoning capabilities of large language models (LLMs). However, existing RLVR methods often suffer from exploration inefficiency due to mismatches between the training data’s difficulty and the model’s capability. LLMs fail to discover viable reasoning paths when problems are overly difficult, while learning little new capability when problems are too simple. In this work, we formalize the impact of problem difficulty by quantifying the relationship between loss descent speed and rollout accuracy. Building on this analysis, we propose SEELE, a novel supervision-aided RLVR framework that dynamically adjusts problem difficulty to stay within the high-efficiency region. SEELE augments each training sample by appending a hint (part of a full solution) after the original problem. Unlike previous hint-based approaches, SEELE deliberately and adaptively adjusts the hint length for each problem to achieve an optimal difficulty. To determine the optimal hint length, SEELE employs a multi-round rollout sampling strategy. In each round, it fits an item response theory model to the accuracy-hint pairs collected in preceding rounds to predict the required hint length for the next round. This instance-level, real-time difficulty adjustment aligns problem difficulty with the evolving model capability, thereby improving exploration efficiency. Experimental results show that SEELE outperforms Group Relative Policy Optimization (GRPO) and Supervised Fine-tuning (SFT) by +11.8 and +10.5 points, respectively, and surpasses the best previous supervision-aided approach by +3.6 points on average across six math reasoning benchmarks.

GitHub: https://github.com/ChillingDream/seele

1 INTRODUCTION

Recent large language models (LLMs) such as OpenAI-o1 (OpenAI et al., 2024), DeepSeekR1 (DeepSeek-AI et al., 2025a), and Kimi-K2 (Team et al., 2025) have made remarkable breakthroughs in reasoning ability, benefiting from long chain-of-thought that incorporates self-reflection and revision. This capability can be realized through pure reinforcement learning with verifiable rewards (RLVR), which forgoes memorizing annotated reasoning processes and instead exploits the model’s inherent capabilities by strengthening correct exploratory behaviors (Chu et al., 2025). At present, RLVR has become the common practice for building high-performance reasoning models.

∗The first two authors contribute equally. †Corresponding authors.

##### Problem

Recall that a set is uncountable if … any non-empty set that is power set-like has an uncountable subset not in bijection with the original set ... Therefore, any uncountable set implies such a subset. <answer> True </answer>

Determine whether the following statement is true or false: every uncountable set has an uncountable subset that is not in bijection with the original set?

##### Policy LLM Hint

[Figure 1]

|Understanding the Statement An uncountable set is one whose cardinality ...|
|---|

|… The existence of 𝜔𝜔1 demonstrates that not all uncountable sets have an uncountable subset of strictly smaller cardinality. <answer> False </answer><br><br>|
|---|

[Figure 2]

[Figure 3]

|Analyzing 𝝎𝝎𝟏𝟏 as a Counterexample Any uncountable subset 𝑆𝑆 ⊆ 𝜔𝜔1 is cofinal …|
|---|

…

- Figure 1: Comparison between the direct rollout (blue) and hinted rollout (purple). The hint consists of the first few steps from an annotated solution. Adding a hint can simplify the problem and guide the LLM toward completing correct solutions.

However, on-policy exploration inherently constrains the learning efficiency, exhibiting strong data dependency (Gao et al., 2025; Dou et al., 2025; Schmied et al., 2025; Yu et al., 2025; Zhang et al., 2025b; Sun et al., 2025). RLVR is driven by the rewards from extensive online sampled rollouts, which collapse to zero when the training problems are too difficult for LLMs to produce a correct response. Conversely, overly simple problems yield nearly all correct rewards, producing minor advantage value. It remains unclear what problem difficulty maximizes learning efficiency and how to curate such data. Moreover, as recent studies (Gandhi et al., 2025; Yue et al., 2025; Zhao et al., 2025) have found, RLVR merely amplifies existing behaviors rather than fostering novel cognitive capabilities, thereby limiting the achievable performance to that of the base model. Supervised finetuning (SFT) (K¨opf et al., 2023) is a naive way to improve the ability of LLMs before RL with expert data. However, existing works (Zhang et al., 2025c; Chen et al., 2025) show that directly using SFT-then-RL is not an effective way, which even underperforms pure RL.

To overcome these limitations, recent works have attempted to integrate SFT into the RL framework, enabling synergistic learning when SFT-then-RL is ineffective. LUFFY (Yan et al., 2025) and SRFT (Fu et al., 2025) incorporate parallel off-policy guidance into the rollout set, allowing simultaneous exploration and imitation. UFT (Liu et al., 2025a), TAPO (Wu et al., 2025), StepHint (Zhang et al., 2025a), Hint-GRPO (Huang et al., 2025a), and Prefix-RFT (Huang et al., 2025b) append an off-policy hint prefix to each problem to reduce exploration difficulty. While these methods allow for a certain degree of problem difficulty modulation, they have a salient shortcoming: off-policy guidance is applied statically and indiscriminately across problems and hint levels, causing most training samples’ difficulty to mismatch the model’s evolving capability. Consequently, for those challenging problems, mild off-policy guidance does not effectively resolve the low efficiency of on-policy exploration, whereas for those easy problems, excessive off-policy intervention may impede the LLM from developing its own reasoning patterns. This observation raises a critical question:

What is the appropriate problem difficulty under off-policy guidance, and how can the difficulty be dynamically adjusted in accordance with the model’s evolving capability?

In this paper, we propose SEELE: reSponsive rEasoning Evolution via capabiLity-adaptivE hint scaffolding, a theoretically-grounded supervision-aided RVLR approach that keeps high learning efficiency throughout the whole training stage via dynamically adjusting the proportion of the offpolicy prefixes. SEELE explicitly formalizes that an appropriately difficult problem should yield a rollout accuracy of approximately 50% through a theoretical analysis of the loss descent magnitude. By appending an instance-specific, dynamically determined hint after the original problem (Figure 1), SEELE is able to control the problem difficulty within the “sweet spot”. Unlike previous hint-based methods (Liu et al., 2025a; Huang et al., 2025b), our approach operates at the instance level rather than the whole dataset and involves the model’s real-time feedback, thereby achieving a more precise alignment between problem difficulty and model capability. Concretely, we split a rollout sampling stage into several rounds, across which we establish a regression model to predict the accuracy given the hinting rate (proportion of the full solution) under the guidance of item response theory (Chen et al., 2021). At each round, we fit the accuracy prediction model using the feedback from the previous rounds and predict how long the current hint should be for a 50%

accuracy. We conduct experiments on six math reasoning benchmarks and three general domain reasoning benchmarks, on which SEELE significantly outperforms previous RLVR methods.

Our contributions can be summarized as:

- • We present a theoretical analysis showing that the learning efficiency of RLVR algorithms follows a quadratic negative relationship with rollout accuracy, and reaches its maximum when the accuracy is 50%.
- • Guided by our theoretical analysis, we propose a novel capability-adaptive RLVR framework that manipulates the problem difficulty via multi-round rollout sampling and accuracy prediction, maintaining high learning efficiency throughout the entire training process.
- • We demonstrate SEELE’s superiority over previous RLVR methods on 9 challenging benchmarks, achieving remarkable improvements of +11.8 points on average compared with GRPO on math reasoning.

- 2 RELATED WORK

Reinforcement Learning for LLM Reasoning. Recent work has demonstrated the effectiveness of reinforcement learning (RL) in improving the reasoning capabilities of large language models (LLMs), as exemplified by systems such as DeepSeek-R1 (DeepSeek-AI et al., 2025a), OpenAI-

- o1 (OpenAI et al., 2024), and Kimi-K2 (Team et al., 2025). These studies show that RL with purely verifiable rewards can drive LLMs to autonomously develop extended chains of thought, incorporating substantial self-reflection and iterative refinement. Among these approaches, GRPO-based methods have emerged as a pivotal paradigm for enhancing reasoning performance. Subsequent studies such as DAPO (Yu et al., 2025), Dr.GRPO (Liu et al., 2025b), GSPO (Zheng et al., 2025) focus on addressing GRPO’s optimization limitations by removing length bias, difficulty bias, etc. However, recent works (Yue et al., 2025; Wang et al., 2025) argue that RL approaches are bounded by the capabilities of the base model and do not acquire new reasoning skills, which suggests pure RL may not be the ultimate solution.

Supervision-Aided Reinforcement Learning. Recent works (OpenAI et al., 2024; Qwen et al., 2025) show that SFT on high-quality reasoning chains can effectively enhance the reasoning ability, and is typically employed as a precursor to the RL stage. Recent works investigate integrating offpolicy supervision into RL as a single process to overcome the efficiency and capacity limitations. LUFFY (Yan et al., 2025) augments the standard RL workflow by adding off-policy annotated traces to the rollout pool, enabling the model to learn external guidance. SRFT (Fu et al., 2025) computes both SFT and RL loss and combines them with an entropy-based weight mechanism. UFT (Liu et al., 2025a), StepHint (Zhang et al., 2025a), and Hint-GRPO (Huang et al., 2025a) propose to add a partial solution (a.k.a hint) generated from a stronger model behind the original problem to handle challenging problems. Prefix-RFT (Huang et al., 2025b) also uses hints and excludes the low-entropy hint tokens from the gradient update to avoid over-imitation. TAPO (Wu et al., 2025) uses high-level “thought patterns” rather than a true solution to encourage the learning of external strategies. However, existing methods mostly provide static supervision without considering the model’s requirements, which causes sub-optimal exploration and imitation efficiency when the problem is overly difficult or simple with respect to the current model’s capability. In contrast, SEELE grounds hint assignment in a principled RL optimization theory and introduces a capability-adaptive manipulation framework to realize it, achieving dynamic explicit difficulty adaptation.

- 3 PRELIMINARIES

Reinforcement Learning with Verifiable Rewards (RLVR) formulates the generation process of an LLM as a Markov Decision Process (MDP), where the state is defined as the concatenation of the prompt x and the tokens generated so far y1:t−1, and the action corresponds to selecting the next token yt from the policy, i.e., yt ∼ πθ(·|x,y1:t−1). The objective of RLVR is to train the policy model to generate outputs that achieve high scores under a rule-based reward function r(x,y). Formally, RLVR optimizes the expected reward over the on-policy rollouts generated from

the policy model πθ

at the previous step, and the objective can be written as L(θ) = −Ex∼D, y∼π

old

, (1)

### +β Ex∼D[DKL(πθ

(·|x)∥πθ(·|x))] LKL(θ)

θ(·|x) Aθ

### (x,y)

old

ref

Lpolicy(θ)

where Aθ

θold(·|x)[r(x,y)] denotes the advantage function and the hyperparameter β controls the strength of KL regularization with respect to the reference model πref.

(x,y) = r(x,y) − Ey∼π

old

GRPO offers an elegant advantage function implementation, which has been widely applied in RLVR studies (Shao et al., 2024). Given an input x, GRPO samples a group of outputs {y1,y2,··· ,yn} from πθ

(·|x) and uses the group mean to replace the expectation. Recently, Liu et al. (2025b) proposes an improved advantage function in their Dr.GRPO approach as

old

### A(x,yi) = r(x,yi) − mean({r(x,yj)|j = 1,2,··· ,n}), (2)

where the normalization term is removed to mitigate optimization bias. In the following, our analysis is conducted based on this unnormalized formulation.

- 4 METHODOLOGY

In this section, we present SEELE from three aspects: (1) a theoretical foundation that identifies the optimal problem difficulty in terms of rollout accuracy (§ 4.1); (2) a multi-round sampling framework that decomposes rollout generation into sequential rounds, thereby enabling capability-aware adaptation of problem difficulty (§ 4.2); (3) a rollout accuracy prediction model based on Item Response Theory, which supports accurate and responsive adjustment of problem difficulty (§ 4.3).

- 4.1 RELATIONSHIP BETWEEN LEARNING EFFICIENCY AND ROLLOUT ACCURACY

We begin by formulating a quantitative relationship between reinforcement learning (RL) training efficiency and problem difficulty. The model prediction accuracy is employed as the difficulty measure. Given a policy model πθ and a binary reward function r(x,y), for a problem x, we define the prediction accuracy for x with respect to the policy model πθ as

### [r(x,y)]. (3)

### aθ(x) = Ey∼π

θ

Prediction accuracy is the expectation of the rollout accuracy and reflects the difficulty of this training instance relative to the current model capability.

Next, we consider a one-step gradient descent from the last-step policy model πθ

with the update vector d. For analytical convenience, we assume θref = θold. The optimization objective (Eq.(1)) can be reformulated by letting θ = θold + d:

old

old+d(·|x))]}. (4)

{Lpolicy(θold + d) + βEx∼D [DKL (πθ

L(θold + d) = min

(·|x)||πθ

min

old

d

d

We conclude that the magnitude of loss descent is upper bounded by the quadratic envelope of aθ(x):

L(θold) − L(θold + d) ≤

- 1

- 2β

Ex∼D[aθ

old(x)(1 − aθ

old

### (x))]. (5)

Here, we briefly outline the derivation. Since L is defined as the sum over independent prompts x, we analyze the instance-level loss descent L(x;θold) − L(x;θold + d) and then aggregate across all instances. For a specific instance x, its minimizer d∗x can be approximated by applying first-order Taylor expansion on Lpolicy and second-order Taylor expansion on LKL at θold:

β 2

Lpolicy(x;θold) + ∇θLpolicy(x;θ) Tθ=θ

d∗x ≈ arg min

dTF(θold)d , (6)

d +

old

d

where F(θold) is the Fisher Information Matrix. Since F(θold) is always positive semi-definite, the right side of Eq. (6) is convex and has a unique global minimizer

1 β

d∗x = −

F−1(θold)∇θLpolicy(x;θold). (7)

Policy Gradient For 𝑖𝑖 = 1,2,⋯, 𝑚𝑚

𝐴𝐴1(1)

- 𝑜𝑜1(𝑖𝑖)

…

Reward Calculation

Accuracy Calculation

- 𝑜𝑜2(𝑖𝑖)

𝑥𝑥: Query 𝑦𝑦1:𝑙𝑙 𝑖𝑖 : Hint

𝑥𝑥: Query 𝑦𝑦: Solution

𝑜𝑜(𝑖𝑖)~𝜋𝜋𝜃𝜃 ⋅ 𝑥𝑥,𝑦𝑦1:𝑙𝑙 𝑖𝑖

𝐴𝐴(1)2

- 𝑟𝑟1(1)

- 𝑟𝑟2(1) …

…

𝑙𝑙 𝑖𝑖 = round 𝑝𝑝 𝑖𝑖 𝑦𝑦

𝐴𝐴(𝑚𝑚)𝑛𝑛−1

𝑎𝑎∗: Target Accuracy

𝑝𝑝(𝑖𝑖) = 𝑓𝑓𝜙𝜙−1(𝑎𝑎∗)

𝑝𝑝 𝑗𝑗 ,𝑎𝑎 𝑗𝑗

𝑜𝑜𝑛𝑛(𝑖𝑖)

𝑟𝑟𝑛𝑛(1)

𝐴𝐴(𝑚𝑚)𝑛𝑛

Optimize 𝜙𝜙

- Figure 2: Overview of SEELE. In each step, SEELE conducts m rollout rounds. In round i, an adaptive hint y1:l(i) is appended to the original problem, where y1:l(i) is a prefix of the full solution y with length l(i), determined by the accuracy-hint model fϕ to achieve the target rollout accuracy a∗. The output accuracies within each round are then used to update fϕ, enabling more accurate predictions in subsequent rounds. Finally, SEELE computes the advantage function over the outputs of all m rounds, which is used to update the policy model.

By substituting d∗x into the Taylor expansion of L(θold + d), we derive the loss descent value:

- 1

- 2β ∇θLpolicy(x;θ) Tθ=θ

L(x;θold) − L(x;θold + d∗x) =

F−1(θold)∇θLpolicy(x;θ) θ=θ

(8)

old

old

- 1

- 2β ∇θaθ(x) Tθ=θ

F−1(θold)∇θaθ(x) θ=θ

. (9)

=

old

old

Finally, note that r(x,y) is an unbiased estimator of aθ(x). By applying the vector parameter Cram´er–Rao bound to Eq. (9) and summing over x ∼ D, we get the upper bound shown in Eq (5)

(the equality becomes an inequality because d may not simultaneously satisfy all d∗x). The full derivation is provided in Appendix A. Eq. (5) indicates the convergence rate is correlated with the

problem difficulty. The policy model learns little from too easy (aθ(x) → 1) or too hard problems (aθ(x) → 0), and the maximal efficiency upper bound is achieved at 50% accuracy.

- 4.2 DIFFICULTY-AWARE HINT MANIPULATION VIA MULTI-ROUND SAMPLING

From Eq. (5), we have established how problem difficulty affects learning efficiency. A natural follow-up problem is whether we can deliberately adjust the problem difficulty to lie within the high-efficiency region (around 50%). Recent studies (Liu et al., 2025a; Huang et al., 2025b; Yan et al., 2025; Fu et al., 2025) have shown that incorporating off-policy hint guidance into on-policy exploration can enhance the success rate of exploration on challenging samples. However, their strategies for controlling hint length lack a principled target and fail to consider instance-level difficulty as well as the model’s real-time capability. As a result, the difficulty of hinted problems may deviate from the optimal region, thereby limiting optimization efficiency.

We draw inspiration from these hint-based methods and propose SEELE, a novel instance-level capability-adaptive hint manipulation approach built on GRPO. SEELE adds a dynamic length hint after the original problem to control difficulty to match the increasing model capability, maintaining the prediction accuracy around 50%. To determine the optimal hint length, it is necessary to estimate the difficulty of the problem relative to the model at the current timestep. For this purpose, we introduce an accuracy estimator fϕ that captures the relationship between prediction accuracy and hint length (the specific form of fϕ and its optimization is introduced in Section 4.3).

As shown in Figure 2, we design a multi-round adaptive sampling framework. Different from standard GRPO rollout generation, SEELE distributes the rollouts into m rounds to gradually fit the parameter ϕ. In round i, SEELE first predicts a hinting rate p(i) for reaching target accuracy a∗ by the inverse function of fϕ. Then, the corresponding part of the solution will be concatenated after x as the input of the policy model for generating n outputs o(1i),o(2i),··· ,o(ni). We calculate the accuracy within this round as a(i) and add the current hint-accuracy pair (p(i),a(i)) to the memory

1.0

- 0.5

- 1.0 k = 4.46 µ = −0.50 b = 0.00

- 0.5

- 1.0 k = 43.15 µ = −0.84 b = 0.02

- 0.5

- 1.0 k = 19.03 µ = −0.64 b = 0.45

k = 50.00 µ = −0.59 b = 0.00

Accuracy

0.5

0.0

0.0

0.0

0.0

0.0 0.5 1.0 Hinting rate

0.0 0.5 1.0 Hinting rate

0.0 0.5 1.0 Hinting rate

0.0 0.5 1.0 Hinting rate

Observed Estimated

- Figure 3: Cases of the accuracy-hint curves and the 3PL fitted curve and parameters. We select 4 typical curves to demonstrate the expressive power of 3PL model.

and update the parameter of fϕ. As the number of rounds increases, fϕ will model the accuracy more and more accurately and finally make the rollout accuracy stabilize at a∗. Specifically, the first

round needs a cold start where we use a default hinting rate |y||−y|1 in preparation for the worst model capability. The final hinting rate after m rounds will be stored in the dataset so that in the next epoch SEELE can begin exploration from the last predicted rate.

After completing all rollout rounds, the mn outputs will be used to calculate the advantages. In GRPO implementation, the advantage is computed at the response level and then distributed to all tokens, which will undermine the model’s output probability on input hints if the model fails to explore a correct completion. Hence, we only compute RLVR loss on the generated tokens and impose a supervised loss on the hint tokens to encourage imitation. The final loss is

(·|x)||πθ(·|x)], (10) where y1:l is the hint decided by multi-round sampling.

L(θ) = −Ex∼D,o∼π

### (x,o) + γπθ(y1:l|x)] + βEx∼D[DKL(πθ

θ(·|x,y1:l)[Aθ

old

old

- 4.3 ROLLOUT ACCURACY PREDICTION

The effectiveness of SEELE critically depends on the accuracy of the prediction model fϕ, making its design particularly important. fϕ should be sufficiently expressive to fit the accuracy-hint relationship while also capable of generalizing from only a few data points. To construct such an accurate yet tractable model, we adopt an established framework used in humans. In psychometrics, Item Response Theory (IRT) (Chen et al., 2021) studies the relationship between an individual’s performance on a test item and the test takers’ levels of performance on an overall measure of the ability. The three-parameter logistic (3PL) model is a widely used IRT model that gives the probability that a person with a given ability level will answer correctly:

1 − b

1 + e−k0(Θ−ν), (11) where a and Θ indicate the tester’s prediction accuracy and capability, and three model parameters ν,k0,b represent difficulty, discrimination, and guessing chance, respectively.

a(Θ) = b +

In our approach, the difficulty of problem is tunable depending on the hinting rate. We assume the measure of difficulty ν is linearly related to the hinting rate p as ν = −kk

p + ν0. Through some symbol substitution and algebraic transformations, we derive the relationship between the accuracy and hinting rate at a certain level of model capability:

0

1 − b

1 + e−k(p+µ) , (12) where ϕ represents the parameters {k,µ,b} and µ = k

fϕ(p) = b +

k (Θ − ν0) is the shifted capability measure.

0

The relationship described in (12) aligns with both our intuition and empirical observations. When

- p is small, the problem is too difficult for the model to explore a correct reasoning path, resulting in near-zero accuracy. As p increases and critical steps are gradually revealed, the LLM becomes capable of completing the solution, leading to a rapid increase in accuracy. Once all critical steps are revealed, the model’s accuracy approaches 1, and providing a longer hint yields no further gain.

Across different problems and training stages, the accuracy-hint curve varies in terms of its starting point, slope, and upper bound. We analyze these curves for 100 randomly sampled problems, with the model generating 100 outputs per problem at each hint level, and find that the 3PL model is sufficiently expressive to capture all observed patterns. Full illustrations are provided in Appendix C, while Figure 3 presents representative examples. The observed trends are consistent with IRT predictions, and the 3PL model accurately captures the relationship. Moreover, as the 3PL model involves only three parameters, it requires only a few rounds to obtain an accurate fit. For the model fitting, SEELE employs non-linear least squares:

ϕ = arg min

ϕ

- i
- j=1

(fϕ(p(j)) − aˆ(j))2, where aˆ(j)=mean({a(w)|p(w)=p(j),w=1,2··· ,i}). (13)

Here aˆ(j) denotes the averaged accuracy across all rounds with the same hinting rate, which helps reduce variance in the accuracy estimation.

- 5 EXPERIMENTS

- 5.1 SETUP

Training Datasets. We select DeepMath-103K as our training dataset. DeepMath-103K (He et al., 2025) is a large-scale, decontaminated SFT dataset featuring challenging and verifiable mathematical problems, with a strong focus on higher-difficulty problems. To construct a challenging training subset, we filter out 22k problems that are particularly difficult, i.e., those on which Qwen2.57B (Qwen et al., 2025) fails to produce a correct answer. For these problems, we annotate step-bystep reasoning traces using DeepSeek-V3 (DeepSeek-AI et al., 2025b). We instruct DeepSeek-V3 to generate concise and coherent CoTs and decompose the CoT into logically complete steps. Detailed data synthesis procedure and the annotation prompt are included in the Appendix E.

Implementation Details. We adopt GRPO as the RL algorithm, setting the KL coefficient β =

- 0.001 and the imitation coefficient γ = 0.001. Our rollout batch size is 256 and the update batch size is 64. For our approach and all other RL-based baselines, we generate 32 rollouts in total with a maximum length 2,048 tokens for each sample. For our multi-round sampling, we set the number of rounds m = 4 and each round will generate n = 8 rollouts. Temperature is set to 1.0 for the rollout generation. More details about the implementation of policy optimization are included in Appendix B. Our training is based on veRL (Sheng et al., 2025). We use MathRuler (hiyouga, 2025) to verify the correctness of the model’s outputs and use the TRF non-linear least squares algorithm

provided by the LMFIT library (Newville et al., 2025) to fit fϕ. Our experiments are conducted under the Zero-RL setting, where we use the DeepSeek-R1-Zero prompting template and train on base models Qwen2.5-1.5B and Qwen2.5-3B for 400 steps.

Evaluation. Following the common practice in the previous studies (Zeng et al., 2025; Huang et al., 2025b), we evaluate SEELE on 6 math reasoning benchmarks and 3 general domain reasoning benchmarks. Math reasoning benchmarks include GSM8K (Cobbe et al., 2021), MATH500 (Hendrycks et al., 2021), Minerva (Lewkowycz et al., 2022), OlympiadBench (He

- et al., 2024), AIME24 (LI et al., 2024), and AMC23 (He et al., 2024). General domain reasoning benchmarks include ARC-Challenge (Clark et al., 2018), GPQA-Diamond (Rein et al., 2023), and MMLU-Pro (Wang et al., 2024). We report avg@32 for AIME24 and AMC23 as the test set is small and set generation temperature as 0.6. For other benchmarks, we use greedy-decoding sampling and report pass@1 accuracy.

Baselines. We compare SEELE with pure GRPO and SFT baselines and recent supervisionaided RLVR methods: LUFFY (Yan et al., 2025), UFT (Liu et al., 2025a), and Prefix-RFT (Huang

- et al., 2025b). UFT generates all rollouts from hinted partial solutions and gradually decays the hint proportion from 1 to 0 as training progresses. LUFFY and Prefix-RFT mix the off-policy guidance with self-explored traces. For a fair comparison, we train all the methods on the same dataset, prompt template, and models. We set the hyperparameters following their paper.

Table 1: Performance on math and general domain reasoning using Qwen2.5 1.5B and 3B models.

Math Reasoning General Domain Reasoning GSM8K MATH500 Minerva Olympiad AIME24 AMC23 Avg. ARC-C GPQA-D MMLU-Pro Avg.

Model

Qwen2.5-3B 73.3 29.0 7.0 10.7 0.6 11.6 22.0 66.6 23.7 15.2 35.2 + SFT 73.1 52.6 18.4 18.5 2.3 25.0 31.7 75.7 30.3 40.1 48.7 + GRPO 78.5 45.8 17.3 15.0 1.7 23.9 30.4 75.6 36.9 40.5 51.0 + LUFFY 81.1 56.6 21.0 21.3 2.7 27.0 35.0 78.4 32.8 41.5 50.9 + UFT 84.8 62.6 22.1 25.9 5.8 41.6 40.5 79.4 35.4 42.9 52.6 + Prefix-RFT 77.4 57.0 21.3 21.9 6.0 31.5 35.9 82.2 35.9 37.9 52.0 + SEELE (Ours) 86.3 66.4 26.1 28.9 5.9 39.4 42.2 81.2 34.3 44.0 53.2

Qwen2.5-1.5B 61.9 22.8 9.6 6.7 0.7 9.1 18.5 45.1 15.7 12.2 24.3 + SFT 67.4 43.6 13.6 12.6 1.4 16.4 25.8 63.7 25.8 30.2 39.9 + GRPO 70.1 36.4 10.7 11.1 1.8 15.2 24.2 64.8 25.3 23.1 37.7 + LUFFY 67.2 45.4 11.0 12.9 1.6 16.5 25.8 64.3 26.8 24.7 38.6 + UFT 72.6 50.4 12.9 15.9 3.9 26.6 30.4 66.1 23.7 28.5 39.4 + Prefix-RFT 71.5 48.0 13.6 14.5 2.1 22.7 28.7 65.3 23.7 25.0 38.0 + SEELE (Ours) 76.5 58.0 16.2 19.9 4.1 30.4 34.2 68.3 27.8 31.7 42.6

- 5.2 MAIN RESULTS

The results are presented in Table 1. In the in-domain mathematical reasoning setting, SEELE achieves an average improvement of +11.8 points over GRPO and +10.5 points over SFT, surpassing the best baseline by 1.7% (Qwen2.5-3B) and 3.8% (Qwen2.5-1.5B). The improvement is consistent across the six benchmarks, underscoring the benefits of introducing off-policy demonstrations and leveraging dynamic data difficulty adaptation. We further observe that GRPO generally underperforms SFT on complex reasoning tasks (e.g., MATH500, AIME24), while showing comparable or slightly better performance on relatively easy tasks (e.g., GSM8K, ARC-C), demonstrating the limitations of purely on-policy exploration. Moreover, the relatively low performance of GRPO and SFT indicates that exclusive self-exploration or pure imitation alone is insufficient to cultivate strong complex reasoning capabilities. In the out-of-domain general reasoning setting, SEELE demonstrates strong generalization, achieving an average improvement of +2.7 points over SFT when using Qwen2.5-1.5B, while other baselines exhibit only comparable performance.

Notably, SEELE demonstrates similar effectiveness for both 3B and 1.5B models, whereas other supervision-aided RLVR baselines achieve substantially smaller gains over GRPO when using the

- 1.5B model. We speculate that the smaller model possesses weaker evolution ability and is therefore more sensitive to the problem difficulty, requiring a more elaborated training strategy. Additionally, among the three supervision-aided RLVR baselines, LUFFY and Prefix-RFT perform much worse than ours and UFT, which we attribute to their hybrid strategy that keeps pure exploration. In contrast, integrating off-policy guidance into each rollout enables all outputs to contribute to the model update, thereby achieving higher learning efficiency.

- 5.3 TRAINING DYNAMICS

- Figure 4 shows the training dynamics (reward, response length, and validation accuracy) of the SEELE, GRPO, and the two dynamic hinting baselines using Qwen2.5-3B.

Precise Difficulty Control After the initial warm-up, the reward of SEELE rapidly rises to around 0.5 and maintains throughout the remaining training process, indicating our multi-round sampling and regression framework is able to precisely control the rollout accuracy. Due to the adoption of a pessimistic cold-start strategy, the initial reward exceeds the target 0.5 and gradually decreases as training progresses. To further understand SEELE’s difficulty control mechanism, we visualize the intermediate results from the multi-round rollout sampling process in Figure 5. The results of Qwen2.5-1.5B and Qwen.5-3B are quite similar. Relatively larger fluctuations and deviations are observed in Rounds 1 and 2, primarily because the 3PL regression model receives too few samples to produce accurate fits. Particularly, there is a peak in the first 80 steps (corresponding to the first epoch), which is caused by the high cold-start hinting rate. From the second epoch onward, SEELE uses the hinting rate rectified in the preceding epoch for the initial round, resulting in more accurate

0.8

800

40

ValidationAccuracy(%)

ResponseLength

600

0.6

Reward

35

400

0.4

30

200

0.2

0

25

0 100 200 300 400 Step

0 100 200 300 400 Step

0 100 200 300 400 Step

GRPO UFT Preﬁx-RFT SEELE

- Figure 4: Training dynamics of RL compared with baselines. Left: training rewards; Middle: Response length; Right: averaged accuracy on math reasoning validation sets.

0 100 200 300 400 Step

20

40

60

80

AccuracywithinRound(%)

- Round 1

- Round 2

- Round 3

- Round 4

0 100 200 300 400 Step

20

40

60

80 Round 1

- Round 2

- Round 3

- Round 4

- Figure 5: Accuracy within each rollout round during training. The red dotted line denotes the targeted accuracy. Left: Qwen2.5-1.5B; Right: Qwen2.5-3B.

predictions. By Round 3 and Round 4, the accuracy is very close to the target 50%, suggesting that the minimal data requirement for the 3PL model is approximately three samples, while four samples are sufficient to fit a model with adequate precision in practice.

Accelerated Learning Compared with GRPO, UFT, and Prefix-RFT, SEELE consistently achieves higher accuracy throughout the entire training process and converges more rapidly. Within the first 100 steps, both SEELE and UFT quickly widen the accuracy gap over GRPO and Prefix-RFT, underscoring the importance of external guidance. GRPO reaches its performance ceiling around step 100, showing minimal improvement thereafter. Notably, its reward continues to increase throughout the remainder of training and ultimately surpasses that of UFT and Prefix-RFT. This behavior suggests that GRPO primarily reinforces previously acquired skills rather than facilitating the learning of new capabilities. From the 100 to 200 steps, the performance of UFT stagnates because its problem difficulty fails to adapt to the model’s evolving capabilities. UFT’s holistic pre-designed decaying schedule cannot adequately meet the changing learning requirements. In contrast, SEELE sustains a high growth rate until approximately step 300. From the perspective of response length, SEELE exhibits a more stable growth trend than UFT and Prefix-RFT, which aligns with the performance growth, highlighting the effectiveness of our hint adjustment schedule.

- 5.4 TARGET ACCURACY ANALYSIS

To verify the critical role of rollout accuracy in RLVR, we conduct an ablation study by varying the target accuracy α∗ and examining its impact on performance. Specifically, we set α∗ to values in the range {i/n | i = 2,3,...,n − 2}, while excluding the boundary cases i = 1 and i = n − 1 to avoid situations where rollouts become entirely incorrect or entirely correct. The results, summarized in Figure 6, lead to the following observations: (1) Setting the target accuracy to 0.5

80

a∗ = 0.25

a∗ = 0.375

a∗ = 0.5

60

a∗ = 0.625

Accuracy(%)

a∗ = 0.75

40

20

0

GSM8K MATH500 Minerva Olympiad AIME24 AMC23 ARC-C GPQA MMLU-Pro Math Avg. General Avg.

Dataset

Figure 6: Performance for different target accuracy a∗ using Qwen2.5-1.5B.

Table 2: Performance of different multi-round configurations using Qwen2.5-1.5B.

Multi-Round Math Reasoning General Domain Reasoning

Rollout Scheme GSM8K MATH500 Minerva Olympiad AIME24 AMC23 Avg. ARC-C GPQA-D MMLU-Pro Avg. mn = 16 m = 4 76.8 51.0 15.4 19.0 3.3 28.4 32.3 66.3 24.7 29.7 40.2

- m = 3 75.1 53.4 14.7 20.0 4.6 31.8 33.3 67.9 21.9 29.8 39.9

- m = 4 77.1 54.4 16.9 19.9 3.0 32.1 33.9 66.6 24.7 33.2 41.5 m = 6 77.6 53.8 15.8 18.5 2.9 30.5 33.2 68.9 23.8 33.0 41.9

mn = 24

m = 4 76.5 58.0 16.2 19.9 4.1 30.4 34.2 68.3 27.8 31.7 42.6 m = 8 76.9 54.2 16.5 19.3 3.3 31.4 33.6 69.8 23.2 32.4 41.8

mn = 32

yields the best performance, while performance degrades gradually as α∗ deviates from this value; (2) The degradation trend is approximately symmetric, whether the target accuracy is increased or decreased. These findings are consistent with both intuition and theoretical analysis: data that is either overly difficult or overly simple hinders effective training. For completeness, we also report detailed accuracy control results in Appendix D, which further demonstrate that our method can precisely regulate the rollout accuracy to any desired target value.

- 5.5 ROLLOUT SCHEME ANALYSIS

Given a fixed total number of rollouts, the number of rounds can be configured in multiple ways. Increasing the number of rounds provides more samples for regression, but it reduces the number of samples per round, which in turn increases the variance in estimating accuracy for a specific hinting rate. To examine the actual effect of the multi-round rollout configuration and identify the optimal setting, we train Qwen2.5-1.5B with varying numbers of rollout rounds under different total rollout budgets. The results are presented in Table 2. Our observations are as follows: (1) The performance of SEELE is relatively insensitive to the specific rollout configuration; (2) A three-round scheme yields the lowest performance given the same total number of rollouts, as only two sample points are available for constructing the three-parameter logistic (3PL) model, which is insufficient for accurate estimation; (3) A four-round scheme generally achieves the highest performance. Increasing the number of rounds beyond four reduces the number of samples per round, leading to higher estimation variance in single-round accuracy. Based on these findings, we conclude setting m = 4 as a practical choice for optimal performance.

- 6 CONCLUSION

In this paper, we present SEELE, a novel reinforcement learning with variable reward (RLVR) framework that leverages off-policy demonstrations to dynamically align problem difficulty with the evolving capability of the model, thereby optimizing training efficiency. This framework is mo-

tivated by our quantitative analysis, which shows that the learning efficiency of RL algorithms is maximized when the policy model’s rollout accuracy is approximately 50%. To maintain rollout accuracy within this “sweet spot”, we employ Item Response Theory and construct a multi-round regression model to predict the optimal hint length for fine-grained difficulty adjustment. A key distinguishing feature of our approach, compared with previous supervision-aided RL methods, is that the difficulty manipulation operates at the instance level and incorporates real-time feedback, rendering the training process more responsive and enhancing the utility of each individual training sample. Extensive experiments demonstrate that SEELE significantly outperforms GRPO, SFT, and other RLVR baselines. Our study provides preliminary insights into the types of data favored by RL algorithms and offers a novel direction for improving data efficiency in reinforcement learning.

REFERENCES

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468, 2025.

Yunxiao Chen, Xiaoou Li, Jingchen Liu, and Zhiliang Ying. Item Response Theory – A Statistical Framework for Educational and Psychological Measurement, August 2021.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training. In Forty-Second International Conference on Machine Learning, June 2025.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge, March 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems, November 2021.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu

Zhang, and Zhen Zhang. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, January 2025a.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. DeepSeek-V3 Technical Report, February 2025b.

Shihan Dou, Muling Wu, Jingwen Xu, Rui Zheng, Tao Gui, Qi Zhang, and Xuanjing Huang. Improving RL Exploration for LLM Reasoning through Retrospective Replay, July 2025.

Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning, June 2025.

Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D. Goodman. Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs, March 2025.

Jingtong Gao, Ling Pan, Yejing Wang, Rui Zhong, Chi Lu, Qingpeng Cai, Peng Jiang, and Xiangyu Zhao. Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration, July 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. DeepMath-103K: A Large-Scale, Challenging, Decontaminated, and Verifiable Mathematical Dataset for Advancing Reasoning, May 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. In

Joaquin Vanschoren and Sai-Kit Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, Virtual, 2021.

hiyouga. Mathruler. https://github.com/hiyouga/MathRuler, 2025. Qihan Huang, Weilong Dai, Jinlong Liu, Wanggui He, Hao Jiang, Mingli Song, Jingyuan Chen,

Chang Yao, and Jie Song. Boosting MLLM Reasoning with Text-Debiased Hint-GRPO, June 2025a.

Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M. Ponti, and Ivan Titov. Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling, July 2025b.

Andreas K¨opf, Yannic Kilcher, Dimitri Von R¨utte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Rich´ard Nagyfi, et al. Openassistant conversations-democratizing large language model alignment. Advances in neural information processing systems, 36:47669–47681, 2023.

Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving Quantitative Reasoning Problems with Language Models. In Advances in Neural Information Processing Systems, October 2022.

Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://github.com/project-numina/aimo-progress-prize](https: //github.com/project-numina/aimo-progress-prize/blob/main/ report/numina_dataset.pdf), 2024.

Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar. UFT: Unifying Supervised and Reinforcement Fine-Tuning, May 2025a.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-Like Training: A Critical Perspective, March 2025b.

Matthew Newville, Renee Otten, Andrew Nelson, Till Stensitzki, Antonino Ingargiola, Daniel Allan, Austin Fox, Faustin Carter, and Michal Rawlik. Lmfit: Non-linear least-squares minimization and curve-fitting for python, July 2025. URL https://doi.org/10.5281/zenodo. 16175987.

OpenAI, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Qui˜nonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan

Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. OpenAI o1 System Card, December 2024.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 Technical Report, January 2025.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A Graduate-Level Google-Proof Q&A Benchmark, November 2023.

Thomas Schmied, J¨org Bornschein, Jordi Grau-Moya, Markus Wulfmeier, and Razvan Pascanu. LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities, April 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, April 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A Flexible and Efficient RLHF Framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, March 2025. doi: 10.1145/3689031.3696075.

Zexu Sun, Yiju Guo, Yankai Lin, Xu Chen, Qi Qi, Xing Tang, Ji-Rong Wen, et al. Uncertainty and influence aware reward model refinement for reinforcement learning from human feedback. In The Thirteenth International Conference on Learning Representations, 2025.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu,

Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jing Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Junjie Yan, Yuzi Yan, Xiaofei Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yangkun Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian Zhao, Yikai Zhao, Huabin Zheng, Shaojie Zheng, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi K2: Open Agentic Intelligence, July 2025.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Reinforcement Learning for Reasoning in Large Language Models with One Training Example, May 2025.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, November 2024.

Jinyang Wu, Chonghua Liao, Mingkuan Feng, Shuai Zhang, Zhengqi Wen, Pengpeng Shao, Huazhe Xu, and Jianhua Tao. Thought-Augmented Policy Optimization: Bridging External Guidance and Internal Capabilities, May 2025.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to Reason under Off-Policy Guidance, May 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An Open-Source LLM Reinforcement Learning System at Scale, May 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?, May 2025.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. SimpleRLZoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild, August 2025.

Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, and Rui Yan. StepHint: Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason, July 2025a.

Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo, Yufei Wang, Niklas Muennighoff, et al. A survey on test-time scaling in large language models: What, how, where, and how well? arXiv preprint arXiv:2503.24235, 2025b.

Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. On-policy rl meets off-policy experts: Harmonizing supervised fine-tuning and reinforcement learning via dynamic weighting. arXiv preprint arXiv:2508.11408, 2025c.

Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining, August 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group Sequence Policy Optimization, July 2025.

- A DERIVATION OF THE RL OPTIMIZATION EFFICIENCY AND ROLLOUT ACCURACY

After one-step gradient update, the new loss value is

### L(θold + d) =Lpolicy(θold + d) + βEx∼D [DKL (πθ

old+d(·|x))]. (14) Note that L is defined as the sum of loss on each prompt x. For each x, we approximate Lpolicy(x;θold + d) and DKL (πθ

(·|x)||πθ

old

old+d(·|x)) using first-order and second-order Taylor expansion at θold, respectively. We first expand

(·|x)||πθ

old

### Lpolicy(x;θold + d) ≈ Lpolicy(x;θold) + ∇θLpolicy(x;θ) Tθ=θ

### d. (15) DKL (πθ

old

old+d(·|x)) =DKL (πθ

(·|x)||πθ

### (·|x)||πθ

(·|x))

old

old

old

### (·|x)||πθ(·|x)) Tθ=θ

+ ∇θDKL (πθ

d

(16)

old

old

- 1

- 2

dT∇2θDKL (πθ

### (·|x)||πθ(·|x)) θ=θ

+

d.

old

old

Expand the first and second order derivatives of KL-divergence.

(y|x) πθ(y|x)

πθ

(17)

### ∇θDKL (πθ

### (·|x)||πθ(y|x)) = ∇θEy∼π

old

θold(·|x) log

old

θold(·|x) log πθ(y|x) (18)

= −∇θEy∼π

∇θπθ(y|x) πθ(y|x)

(19)

= −Ey∼π

θold(·|x)

(y|x) πθ(y|x) ∇θπθ(y|x). (20)

### πθ

old

### = −

y

∇θDKL (πθ

old

### (·|x)||πθ(y|x)) θ=θ

### =

old

y

### (y|x) πθ

### πθ

old

### (y|x)∇θπθ(y|x) θ=θ

old

old

(21)

(22)

### = ∇θ

πθ(y|x)

y

θ=θold

= 0. (23)

### ∇2θDKL (πθ

old

(y|x) πθ(y|x)

πθ

### (·|x)||πθ(·|x)) = ∇2θEy∼π

(24)

old

θold(·|x) log

= −∇2θEy∼π

θold(·|x) log πθ(y|x) (25)

∇2θπθ(y|x) πθ(y|x) −

∇θπθ(y|x)∇θπθ(y|x)T πθ2(y|x)

(26)

= −Ey∼π

θold(·|x)

∇2θπθ(y|x) πθ(y|x)

∇θπθ(y|x)T πθ(y|x)

+ ∇θπθ(y|x) πθ(y|x)

(27)

θold(·|x) −

= Ey∼π

∇2θπθ(y|x) πθ(y|x)

+ ∇θ log πθ(y|x)∇θ log πθ(y|x)T

θold(·|x) −

= Ey∼π

(28)

∇2θπθ(y|x) πθ(y|x)

+ F(θ). (29) (30)

θold(·|x) −

= Ey∼π

We substitute θ = θold for the first term.

Ey∼π

θold(·|x)

∇2θπθ(y|x) πθ(y|x) θ=θ

### =

old

y

### (y|x) πθ

### πθ

### (y|x)∇2θπθ(y|x) θ=θ

old

old

old

(31)

### =∇2θ

(32)

πθ(y|x)

y

θ=θold

=0. (33) So, we get

β 2

### L(x;θold + d) ≈ Lpolicy(x;θold) + ∇θLpolicy(x;θ) Tθ=θ

dTF(θold)d. (34)

d +

old

Since the Fisher information matrix is positive semi-definite, the right hand of Eq. (34) convex has a unique global minimizer. To find the minimizer d∗x, let the derivative be zero.

+ βF(θold)d∗x = 0. (35)

∇θLpolicy(x;θ) θ=θ

old

1 β

d∗x = −

F(θold)−1∇θLpolicy(x;θ) θ=θ

. (36) Substitute back to Eq. (34)

old

- 1

- 2β ∇θLpolicy(x;θ) Tθ=θ

L(x;θold) − L(x;θold + d∗x) ≈

old

F−1(θold)∇θLpolicy(x;θ) θ=θ

old

. (37)

Now, we need to connect ∇θLpolicy(x;θ) θ=θ

(x). According to the definition ∇θLpolicy(x;θ) =∇θEy∼π

with aθ

old

old

(x,y) (38)

θ(·|x)Aθ

old

θold(·|x)r(x,y) (39)

=∇θEy∼π

θ(·|x) r(x,y) − Ey∼π

θ(·|x) [r(x,y)] (40)

=∇θEy∼π

(x). (41)

=∇θaθ

old

- 1

- 2β ∇θaθ(x) Tθ=θ

### F−1(θold)∇θaθ(x) θ=θ

L(x;θold) − L(x;θold + d∗x) ≈

. (42)

old

old

Note that r(x,y) is an unbiased estimator of aθ(x). By applying the vector parameter Cram´er–Rao bound to Eq. (42), we get

- 1

- 2β

L(x;θold) − L(x;θold + d∗x) ≈

θold(·|x)[r(x,y)] (43)

Vary∼π

- 1

- 2β

(x)). (44) So, for an arbitrary d

### (x)(1 − aθ

aθ

=

old

old

1 2β

(x)). (45) Now we have got the equality for a single x, taking the expectation we get

L(x;θold) − L(x;θold + d) ≤

### (x)(1 − aθ

aθ

old

old

L(θold) − L(θold + d) ≤

- 1

- 2β

Ex∼D[aθ

old

### (x)(1 − aθ

old

### (x))]. (46)

- B POLICY OPTIMIZATION IMPLEMENTATION DETAILS

In the practical implementation, we apply the clipping techniques used in GRPO (Shao et al., 2024) and use πref rather than πold in KL regularization for simplicity, as shown in Eq. (48) and Eq. (47), which slightly differs from the formulation in Eq. (10).

Algorithm 1 SEELE Policy Optimization Input: initial policy model πinit; training set D

policy model πθ ← πinit reference model πref ← πinit for step=1,2,··· ,T do

old ← πθ Sample a batch B from D C ← {x : ∅}x∈B (Here Cx refers to {(p(xj),a(xj))}ij−=11) for i = 1,2,··· ,m do

πθ

Bi ← ∅ for x,y ∈ B do

Cˆx ← Cx ∪ {(w,w)|w ∈ {0,1}\{p(xj)}ij−=11} Optimize ϕ on Cˆx (Follow Eq.(13).)

p(xi) = fϕ−1(a∗) xˆ = x ⊕ y1:l, l = round(p(xi)|y|) Bi ← Bi ∪ {(ˆx,y)}

end for Sample n outputs {o(ji)}nj=1 ∼ πθ

(·|xˆ) for each xˆ ∈ Bi Compute rewards {rj(i)}nj=1 for each sampled output Compute rollout accuracy a(xi) over {rj(i)}nj=1 for each problem x for x ∈ B do

old

Cx ← Cx ∪ {(p(xi),a(xi))}

end for end for Compute advantage {A(j,ti)}m,ni=j=1 for each sampled token for GRPO iteration=1,2··· ,u do

Update the policy model πθ by maximizing the objective in Eq.(47).

end for end for return πθ

θ(·|x,y1:l)[Aˆθ

L(θ) = −Ex∼D,o∼π

old

(x,o) + γπθ(y1:l|x)] + βEx∼D[DKL(πθ

ref

(·|x)||πθ(·|x)], (47)

o

πθ(ot|x,oˆ <t) πθ

πθ(ot|x,oˆ <t) πθ

Aˆθ

,1 − ϵ,1 + ϵ At (48)

(x,o) =

At,clip

min

old

(ot|x,oˆ <t)

(ot|x,oˆ <t)

old

old

t=1

In Algorithm 1, we present the detailed workflow of SEELE policy optimization. At each training step, SEELE first samples a batch from the training set and constructs a hint–accuracy collection Cx for each problem x. Then, SEELE performs m rounds of generation. In each round, Cx is augmented by adding the margin points (0,0) and (1,1) whenever px = 0 or px = 1 has not yet been evaluated. This augmentation substantially enhances fitting accuracy and stability during the early rounds. Subsequently, the predictor fϕ is optimized on Cx and used to estimate the optimal hint length l. With the estimated hint length, SEELE invokes the policy model to generate n rollouts for the hinted problem xˆ = x ⊕ y1:l and computes the corresponding rewards. These rewards are then aggregated to calculate the intra-round accuracy ax, which is used to update Cx. Finally, after

- m rounds of rollouts, SEELE computes the advantages over all mn outputs following the Dr.GRPO formulation (Liu et al., 2025b), and updates the policy model according to Eq. (47).

- C RELATIONSHIP BETWEEN PREDICTION ACCURACY AND HINTING RATE

We analyze the relationship between prediction accuracy and hinting rate using the Qwen2.5-3B checkpoint after 30 training steps, corresponding to the early rising stage of learning. For each

0.8

0.7

0.6

0.5

Reward

0.4

0.3

0.2

0.1

0 50 100 150 200 250 300 350 400 Step

a * = 0.25 a * = 0.375 a * = 0.5 a * = 0.625 a * = 0.75

Figure 7: Reward across the training steps for various target accuracy.

example, we enumerate the hint length and prompt the model to complete the hinted problem. Accuracy is computed over 100 randomly sampled traces with a temperature setting of 1.

Figure 8 illustrates the accuracy–hinting rate curves for these 100 examples. In most cases, the curves exhibit an S-shaped trend: accuracy remains close to zero until a critical proportion of the solution is revealed, after which it rises rapidly to nearly 1. This pattern is highly consistent with the predictions of the three-parameter logistic (3PL) model.

- D ROLLOUT ACCURACY MANIPULATION

To verify that our multi-round sampling framework can arbitrarily manipulate the rollout accuracy, we set the target accuracy to i/n,i = 2,3,··· ,n − 2. We exclude 1/n and (n − 1)/n because they are too close to the boundary, which easily leads to all wrong/correct rollout sampling. We set

- n = 8 following the main experiment setup. The results are shown in Table 7. After the cold start for an epoch, the reward converges to the target accuracy across all settings with very little error (less than 0.02 for a∗ > 0.25)), showing a stable trend until the end of the training. The fluctuation and deviation diminish as the training progresses. The results of a∗ = 0.25 are slightly higher than the target because they touch the difficulty lower bound, where the policy model can generate sufficiently correct outputs without the aid of hints.

- E TRAINING DATA SYNTHESIS

Our data synthesis procedure consists of two phases: filtering and annotation. We use DeepMath103K (He et al., 2025) as the initial dataset. First, to construct a more challenging subset, we employ Qwen2.5-7B (Qwen et al., 2025) to sample 8 reasoning traces per instance ((temperature = 0.6, maximum length = 2048)) and retain only those instances for which all traces are incorrect. Second, we use DeepSeek-V3 (DeepSeek-AI et al., 2025b) to generate step-by-step reasoning annotations. The annotation prompt is presented below. We instruct DeepSeek-V3 to produce a logically complete and concise solution based on the reference solution provided in the original dataset.

## Step-by-Step Annotation Prompt

Task: Generate a clear, step-by-step, and complete solution to the following problem with these requirements:

- 1. Five or fewer Numbered Steps: Ensure no logical jumps—every non-trivial inference must be justified.
- 2. Key Explanations Included:

- a. Briefly explain ”why” for non-obvious steps (e.g., ”We use X method because...”).
- b. Avoid redefining terms/concepts already introduced.

- 3. Full Calculations:

- a. Show at least one intermediate step for computations (e.g., a = b + c → a = 5 + 3 = 8).
- b. For symbolic math, state the rules/theorems used (e.g., “By the chain rule...”).

- 4. Final Answer: Mark clearly with \boxed{}.

Original Question: {question} Reference Solution (for guidance only): {reference solution} Begin your new solution:

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

0.0 0.5 1.0

1.0

1.0

1.0

1.0

1.0

Accuracy

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

0.5

0.5

0.5

0.5

0.5

0.0

0.0

0.0

0.0

0.0

0.0 0.5 1.0 Hinting rate

0.0 0.5 1.0 Hinting rate

0.0 0.5 1.0 Hinting rate

0.0 0.5 1.0 Hinting rate

0.0 0.5 1.0 Hinting rate

Figure 8: Accuracy with respect to the hinting rate for 100 training examples using Qwen2.5-3B.

