[Figure 1]

First version: June 2, 2025; last revised: November 13, 2025.

## Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

Shenzhi Wang1,2, Le Yu1, Chang Gao1, Chujie Zheng1, Shixuan Liu1, Rui Lu2, Kai Dang1, Xiong-Hui Chen1, Jianxin Yang1, Zhenru Zhang1, Yuqiong Liu1, An Yang1, Andrew Zhao2, Yang Yue2, Shiji Song2, Bowen Yu1, ,†, Gao Huang2, , Junyang Lin1 1 Qwen Team, Alibaba Inc. 2 LeapLab, Tsinghua University

Project Page: https://shenzhi-wang.github.io/high-entropy-minority-tokens-rlvr

#### Abstract

Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model’s entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen332B (+11.04 on AIME’25 and +7.71 on AIME’24) and Qwen3-14B (+4.79 on AIME’25 and +5.21 on AIME’24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning.

# arXiv:2506.01939v2[cs.CL]13Nov2025

[Figure 2]

###### (a) (b)

high-entropy minority tokens fork the path

low-entropy majority tokens follow the path

[Figure 3]

[Figure 4]

RL training on all tokens

RL training on forking tokens

reasoning path

Q: What is 1 + 1 in base 2? A: In decimal, 1 + 1 = 2. But how does that translate

to base 2? Well, in binary […]

reasoning path

Figure 1: (a) In CoTs, only a minority of tokens exhibit high entropy and act as "forks" in reasoning paths, while majority tokens are low-entropy. (b) RLVR using policy gradients of forking tokens delivers significant performance gains that scale with model size. With a 20k maximum response length, our 32B model sets new SoTA scores (63.5 on AIME’24 and 56.7 on AIME’25) for RLVR on base models under 600B. Extending the maximum response length to 29k further boosts the AIME’24 score to 68.1.

Accepted to NeurIPS 2025. : Corresponding authors †: Project lead Emails: wangshenzhi99@gmail.com yubowen.ph@gmail.com gaohuang@tsinghua.edu.cn

#### 1 Introduction

The reasoning capabilities of Large Language Models (LLMs) have advanced substantially in domains like mathematics and programming, propelled by test-time scaling methodologies employed in OpenAI o1 (OpenAI, 2024), Claude 3.7 (Anthropic, 2025), DeepSeek R1 (DeepSeek-AI et al., 2025), Kimi K1.5 (Team et al., 2025), and Qwen3 (Yang et al., 2025). A pivotal technique driving these improvements is Reinforcement Learning with Verifiable Rewards (RLVR) (Lambert et al., 2025; DeepSeek-AI et al., 2025; Yang et al., 2025), where models optimize outputs through RL objectives tied to automated correctness verification. While recent RLVR advancements have stemmed from algorithmic innovations (Yu et al., 2025; Yue et al., 2025b; Guan et al., 2025), cross-domain applications (Xue et al., 2025; Liu et al., 2025; Pan et al., 2025), and counterintuitive empirical insights (Wang et al., 2025; Yue et al., 2025a; Zhao et al., 2025), existing implementations directly train over all tokens with limited understanding of which tokens actually facilitate reasoning. These approaches neglect the heterogeneous functional roles tokens play in reasoning processes, potentially hindering further performance gains by failing to prioritize critical decision points in sequential reasoning trajectories.

In this paper, we analyze the underlying mechanisms of RLVR through an innovative lens of token entropy patterns, investigating how tokens with varying entropy impact reasoning performance. We first point out that in the Chain-of-Thought (CoT) processes of LLMs, the entropy distribution exhibits a distinct pattern where the majority of tokens are generated with low entropy, while a critical minority of tokens emerge with high entropy. Through comparing the textual meanings of these two parts of tokens, we observe that the tokens with lowest average entropy primarily complete the ongoing linguistic structures, while the tokens with highest average entropy function as pivotal decision points that determine the trajectory of reasoning among multiple potential pathways (referred to as forks), as depicted in Figure 1(a). In addition to qualitatively anslysis, we conduct controlled experiments by manually modulating the entropy of forking tokens during decoding. Quantitative results reveal that moderately increasing the entropy of these high-entropy forking tokens leads to measurable improvements in reasoning performance, while artificially reducing their entropy results in performance degradation, confirming the importance of maintaining high entropy and the role as "forks" for these high-entropy tokens. Furthermore, by analyzing the evolution of token entropy during RLVR training, we find that the reasoning model largely retains the entropy patterns of the base model, exhibiting only gradual and relatively minor changes as training progresses. Additionally, RLVR primarily changes the entropy of high-entropy tokens, while the entropy of low-entropy tokens varies only within a small range. The above observations highlight the critical role high-entropy minority tokens may play in CoTs and RLVR training.

Building upon the discovery of forking tokens, we further refine RLVR by retaining policy gradient updates for only 20% of tokens with the highest entropy and masking gradients of the remaining 80%. We observe that although solely utilizing 20% of tokens, our approach can still achieve competitive reasoning performance on Qwen3-8B base model compared to full-gradient updates. Moreover, its effectiveness increases with model size, yielding reasoning improvements of +11.04 on AIME’25 and +7.71 on AIME’24 for the Qwen3-32B base model, and +4.79 on AIME’25 and +5.21 on AIME’24 for the Qwen3-14B base model, as shown in Figure 1(b). Notably, the 32B model trained with only 20% high-entropy tokens attains scores of 63.5 on AIME’24 and 56.7 on AIME’25, setting a new state-of-the-art (SoTA) for reasoning models trained directly from base models with fewer than 600B parameters. Extending the maximum response length from 20k to 29k further elevates our 32B model’s AIME’24 score from 63.5 to 68.1. Conversely, training exclusively on the 80% lowest-entropy tokens results in severe performance degradation. These observations show that 20% of tokens achieve performance comparable to or exceeding 100%, even surpassing the 80/20 rule. The results demonstrate that the high-entropy minority tokens, functioning as critical decision points in reasoning trajectories, account for nearly all performance gains in RLVR.

Finally, we explore why retaining a small fraction of the highest-entropy tokens leads to strong performance in RLVR via a series of ablation studies. We adjust the chosen fraction of forking tokens, either by decreasing it from 20% to 10% or increasing it to 50% or 100%, and report the corresponding reasoning metrics and overall entropy during the training process. Experimental results demonstrate that retaining approximately 20% of the highest-entropy tokens optimally balances exploration and performance, while deviating from 20% reduces the overall entropy with diminished exploration and incurs worse performance. This suggests that only a critical subset of high-entropy tokens meaningfully contributes to the exploration during RL while others may be neutral or even detrimental. Reducing the proportion to 10% removes certain useful tokens, which weakens exploration. Increasing the proportion to 50% or 100% adds low-entropy tokens, which also reduces the effectiveness of exploration. Last but not least, retaining the top 20% of high-entropy tokens results in the largest performance gains for the 32B model, followed by the 14B model, and the smallest gains for the 8B model. This may be due to the insufficient capacity of the smaller model, which restricts its ability to benefit from increased exploration. These findings highlight the importance of preserving an appropriate proportion of high-entropy tokens in RLVR. As model size increases, the strategy of selecting high-entropy tokens appears to scale effectively.

In summary, our findings emphasize the pivotal role of high-entropy minority tokens in shaping the reasoning abilities of LLMs. We hope this inspires further analyses from the perspective of token entropy and informs more effective RLVR algorithms that strategically leverage these tokens to enhance reasoning performance. The key takeaways of our paper are as follows:

- • In CoTs, the majority of tokens are generated with low entropy, while only a small subset exhibits high entropy. These high-entropy minority tokens often act as "forks" in the reasoning process, guiding the model toward diverse reasoning paths. Maintaining high entropy at these critical forking tokens is beneficial for reasoning performance. (§3)
- • During RLVR training, the reasoning model largely preserves the base model’s entropy patterns, showing only gradual and minor changes. RLVR primarily adjusts the entropy of high-entropy tokens, while the entropy of low-entropy tokens fluctuates only within a narrow range. (§4)
- • High-entropy minority tokens drive nearly all reasoning performance gains during RLVR, whereas lowentropy majority tokens contribute little or may even hinder performance. One possible explanation is that, prior to performance convergence, a subset (∼ 20% in our experiments) of high-entropy tokens facilitates exploration, while low-entropy tokens offer minimal benefit or may even impede it. (§5)
- • Based on the insights above, we further discuss (i) high-entropy minority tokens as a potential reason why supervised fine-tuning (SFT) memorizes but RL generalizes, (ii) how prior knowledge and readability requirements shape the different entropy patterns seen in LLM CoTs compared to traditional RL trajectories, and (iii) the advantage of clip-higher over entropy bonus for RLVR. (§6)

#### 2 Preliminaries

- 2.1 Token entropy calculation The token-level generation entropy (referred to as token entropy for brevity) for token t is defined as

Ht := −

V

∑

j=1

pt,j log pt,j, where (pt,1, · · · , pt,V) = pt = πθ(· | q, o<t) = Softmax

zt T

. (1)

Here, πθ denotes an LLM parameterized by θ, q is the input query, and o<t = (o1, o2, · · · , ot−1) represents the previously generated tokens. V is the vocabulary size, zt ∈ RV denotes the pre-softmax logits at time step t, pt ∈ RV is the corresponding probability distribution over the vocabulary, and T ∈ R is the decoding temperature. In off-policy settings, sequences are generated by a rollout policy πϕ while the training policy is πθ, with ϕ ̸= θ. The entropy is still calculated using πθ, as defined in Equation (1), to measure the uncertainty of the training policy in the given sequence.

"Token entropy" corresponds to the token generation distribution, not a specific token. Throughout our paper, we clarify that the token entropy Ht refers to the entropy at index t, which is determined by the token generation distribution pt rather than by any specific token ot sampled from pt. For brevity, when discussing the token ot sampled from the distribution pt, we describe its associated entropy as Ht and refer to Ht as the token entropy of ot. However, if there exists another index t′ ̸= t such that ot′ = ot, the token entropy of ot′ is not necessarily equal to Ht.

- 2.2 RLVR Algorithms

Proximal Policy Optimization (PPO) PPO (Schulman et al., 2017) is a widely adopted policy gradient algorithm in RLVR. To stabilize training, PPO restricts policy updates to remain within a proximal region

of the old policy πθold using the following clipped surrogate to maximize the objective:

old(·|q) min rt(θ)Aˆt, clip(rt(θ),1 − ϵ,1 + ϵ)Aˆt , with rt(θ) =

JPPO(θ) =EB∼D,(q,a)∼B,o∼πθ

πθ(ot|q, o<t) πθold(ot|q, o<t)

. (2)

Here, D is a dataset of queries q and corresponding ground-truth answers a, B is a batch sampled from D, ϵ ∈ R is a hyperparameter typically set to 0.2, and Aˆt is the estimated advantage computed using a value network.

Group Relative Policy Optimization (GRPO) Building on the clipped objective in Equation (2), GRPO (Shao et al., 2024) discards the value network by estimating advantages using the average reward within a group of sampled responses. Specifically, for each query q and its ground-truth answer a, the

[Figure 5]

[Figure 6]

(b) Frequent tokens with the highest average entropy

[Figure 7]

(a) Distribution of token entropy (c) Frequent tokens with the lowest average entropy

Figure 2: Entropy patterns in the chain of thoughts of LLMs. (a) Token entropy distribution. The Y-axis frequency is on a log scale. A minority of tokens exhibit high entropy, while the majority have low entropy, often approaching zero. (b) & (c) Word clouds of the top 100 tokens with the highest and lowest average entropy, respectively, selected from the set of frequently occurring tokens. A larger font size indicates a higher average token entropy. Tokens with the highest average entropy typically function as "forks" to determine reasoning directions, whereas tokens with the lowest average entropy tend to execute reasoning steps along the established path.

rollout policy πθold generates a group of responses {oi}iG=1 with corresponding outcome rewards {Ri}iG=1, where G ∈ R is the group size. The estimated advantage Aˆit is then computed as:

ri − mean({Ri}iG=1) std({Ri}iG=1)

1.0 if is_equivalent(a, oi), 0.0 otherwise.

Aˆit =

, where Ri =

(3)

In addition to this modified advantage estimation, GRPO adds a KL penalty term to the clipped objective in Equation (2).

Dynamic sAmpling Policy Optimization (DAPO) Building on GRPO, DAPO (Yu et al., 2025) removes the KL penalty, introduces a clip-higher mechanism, incorporates dynamic sampling, applies a tokenlevel policy gradient loss, and adopts overlong reward shaping, leading to the following maximization

objective, where rti(θ) is defined as in Equation (2), and Aˆit is computed as in Equation (3):

|oi|

G

1 ∑iG=1 |oi|

min rti(θ)Aˆit,

### ∑

### ∑

JDAPO(θ) =EB∼D,(q,a)∼B,{oi}G

i=1∼πθold(·|q)

t=1

i=1

clip rti(θ),1 − ϵlow,1 + ϵhigh A ˆit , s.t. 0 < oi | is_equivalent(a, oi) < G. (4)

DAPO is one of the state-of-the-art RLVR algorithms without a value network. In this work, we use DAPO as the baseline for our RLVR experiments.

#### 3 Analyzing Token Entropy in Chain-of-Thought Reasoning

Although prior works (Yang et al., 2025; Yu et al., 2025; Yue et al., 2025b) have highlighted the importance of generation entropy in chain-of-thought reasoning, they typically analyze the entropy of all tokens collectively. In this section, we take a closer look at generation entropy in chain-of-thought reasoning by examining it at the token level. To this end, we use Qwen3-8B (Yang et al., 2025), one of the most recent and capable reasoning models within a comparable parameter scale, to generate responses for queries from AIME’24 and AIME’25, using a decoding temperature of T = 1.0. We enforce the use of the thinking mode for every question and collect over 106 response tokens. For each token, the entropy is computed according to the formulation in Equation (1). The statistical analysis of the entropy values of these 106 tokens is presented in Figure 2. Furthermore, a visualization of token entropy for an entire long CoT response is provided in Figures 12 to 17 in the Appendix. From these analyses, we identify the following entropy patterns:

- Entropy Pattern 1 in CoTs: Typically, only a minority of tokens are generated with high entropy, while a majority of tokens are outputted with low entropy. We can observe in Figure 2(a) that the entropy of a large amount of tokens are quite small, and only a small amount of tokens have high entropy. Specifically, the entropy of over half the tokens (approximately 50.64%) is below 10−2, while only 20% of tokens have entropy greater than 0.672.
- Entropy Pattern 2 in CoTs: Tokens with the highest entropy typically serve to bridge the logical connection between two consecutive parts of reasoning, while tokens with the lowest entropy tend to complete the current part of a sentence or finish constructing a word. Other tokens combine these two functions to varying degrees. In Figure 2(b) and (c), we select the 100 tokens generated with the highest average entropy and the lowest average entropy from a total of 106 tokens, respectively. To mitigate the impact of noise on the average entropy, we only consider tokens with frequencies above 100. High-entropy tokens often act as logical connectors within and across sentences, such as "wait," "however," and "unless" (indicating contrasts or shifts), "thus" and "also" (showing progression or addition), or "since" and "because" (expressing causality). Similarly, tokens like "suppose," "assume," "given," and "define" frequently appear in mathematical derivations to introduce assumptions, known conditions, or definitions. Conversely, low-entropy tokens are often word suffixes, source code fragments, or mathematical expression components, all of which exhibit high determinism. Additionally, Figures 12 to 17 provide detailed visualization of token entropy in a long CoT, showing that most tokens outside the highest-entropy or lowest-entropy groups blend bridging and continuation functions to varying degrees.

High-entropy tokens as "forks" in chain-of-thoughts Based on the two observed patterns above, we refer to high-entropy tokens as "forking tokens", as they often lead to different potential branches with high uncertainty in the reasoning process. To further confirm the role of forking tokens in a quantitative way, using Qwen3-8B (Yang et al., 2025) with a maximum response length of 28,672, we assign different decoding temperatures to the forking tokens and the other tokens in the evaluation on AIME 2024 and AIME 2025. Specifically, to analyze the effects of varying combinations of these temperatures on their

behavior, we adjust probability distribution p′t ∈ RV for each token t as follows:

zt Tt′

Thigh if Ht > hthreshold, Tlow otherwise.

##### p′

, where Tt′ =

t = Softmax

(5)

Here, zt ∈ RV denotes the pre-softmax logits for token t,

71.67

71.67 72.22 70.94

72

71.11

71.04

and Tt′ ∈ R represents the adjusted temperature for token t;

70.52

70.76

70.73

70

hthreshold = 0.672 is the entropy threshold used to distinguish forking tokens from the other tokens, and is estimated by calculating the 80th percentile among the sampled 106 tokens above; Thigh ∈ R and Tlow ∈ R correspond to the temperatures for forking tokens and the other tokens, respectively.

70.07

68.96

68

AIMEScore

68.33

66

64

62.71

62

The effects of varying Thigh and Tlow are presented in Figure 3. It can be seen that lowering Thigh significantly degrades performance compared to lowering Tlow. In contrast, increasing Thigh results in substantially better performance than increasing Tlow, which can even cause LLMs generating nonsensical outputs. These results suggest that forking tokens benefit from being assigned a relatively higher temperature compared to other tokens. Given that forking tokens naturally exhibit higher entropy than other tokens, this further supports the need for them to operate at an even higher entropy level. This observation aligns with their role as "forks," where high entropy enables them to branch into diverse reasoning directions.

25

20.14

0.14

0

0.0 0.05 0.3 0.6 1.0 2.03.0 5.0 Temperature (symlog scale, linthresh=0.2)

Figure 3: Average scores of AIME 2024 and AIME 2025. Red curve: varying Thigh with Tlow = 1. Blue curve: varying Tlow with Thigh = 1.

#### 4 RLVR Preserves and Reinforces Base Model Entropy Patterns

In this section, building on the observations of entropy patterns in CoTs discussed in Section 3, we further investigate how these patterns evolve throughout RLVR training.

RLVR primarily preserves the existing entropy patterns of the base models To analyze the evolution of entropy patterns during RLVR training, we apply DAPO (Yu et al., 2025) to the Qwen3-14B base model (details in Section 5). Using the reasoning model after RLVR, we generate 16 responses per question across the six benchmarks in Table 2. For each token in these responses, we compute logits using reasoning models from various RLVR stages and identify those in the top 20% entropy. We then calculate the overlap ratio (i.e., the fraction of shared top 20% high-entropy positions) between each intermediate model and both the base and final RLVR models. As shown in Table 1, although overlap with the base model gradually decreases and overlap with the final RLVR model increases, the base model’s overlap still remains above 86% at convergence (step 1360), suggesting that RLVR largely retains the base model’s entropy patterns regarding which tokens exhibit high or low uncertainty.

- Table 1: The progression of the overlap ratio in the positions of the top 20% high-entropy tokens, comparing the base model (i.e., step 0) with the model after RLVR training (i.e., step 1360).

|Compared w/<br><br>|Step 0 Step 16 Step 112 Step 160 Step 480 Step 800 Step 864 Step 840 Step 1280 Step 1360|
|---|---|
|Base Model RLVR Model|100% 98.92% 98.70% 93.04% 93.02% 93.03% 87.45% 87.22% 87.09% 86.67%<br><br>86.67% 86.71% 86.83% 90.64% 90.65% 90.64% 96.61% 97.07% 97.34% 100%|

RLVR predominantly alters the entropy of high-entropy tokens, whereas the entropy of low-entropy tokens remains comparatively stable with minimal variations. Using the same setup as Table 1, we compute the average entropy change after RLVR for each 5% entropy percentile range of the base model. It is observed that tokens with higher initial entropy in the base model tend to exhibit larger increases in entropy after RLVR. This observation could also further reinforce that RLVR primarily preserves the entropy patterns of the base model. Moreover, Figure 5 illustrates the evolution of entropy percentiles during RLVR training using the Qwen3-14B base model. The figure reveals that as we move from the 0th to the 100th percentile, the range of fluctuations during the whole RLVR training steadily diminishes. Thus, these observations suggest that throughout the whole training process, RLVR primarily adjusts the entropy of high-entropy tokens, while the entropy of low-entropy tokens exhibits minor variation and remains relatively stable.

AverageChange(logscale)

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

10 1

10 2

10 3

10 4

0% 5% 10% 15% 20% 25% 30% 35% 40% 45% 50% 55% 60% 65% 70% 75% 80% 85% 90% 95%100% Percentile Boundaries

- Figure 4: Average entropy change after RLVR within each 5% entropy percentile range of the base model. x% percentile means that x% of the tokens in the dataset have entropy values less than or equal to this value. It is worth noting that the Y-axis is presented on a log scale. Tokens with higher initial entropy tend to experience greater entropy increases after RLVR.

#### 5 High-Entropy Minority Tokens Drive Effective RLVR

Reinforcement learning with verifiable rewards (RLVR) has become one of the most widely used approaches for training reasoning models (Yang et al., 2025; DeepSeek-AI et al., 2025; Yu et al., 2025; Yue et al., 2025b). However, there is a lack of research on which types of tokens contribute the most to the learning of reasoning models. As highlighted in Section 3 and Section 4, high-entropy minority tokens are particularly important. In this section, we investigate the contribution of these high-entropy minority tokens, also referred to as forking tokens, on the development of reasoning capabilities during RLVR.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

- Figure 5: The evolution of entropy percentiles during RLVR training. x-th percentile means that x% of the tokens in the dataset have entropy values less than or equal to this value. In other words, it represents the threshold below which the entropy values of the lowest x% of tokens fall, allowing us to track how different segments of the entropy distribution change throughout training.

##### 5.1 Formulation of RLVR Using Only Policy Gradients of the Highest-Entropy Tokens

Building on DAPO’s objective in Equation (4), we discard the policy gradients of low-entropy tokens and train the model using only the policy gradients of high-entropy tokens. For each batch B sampled from the dataset D, we calculate the maximum objective as:

|oi|

G

1 ∑iG=1 ∑|o

I Hti ≥ τρB

∑

∑

JHighEntB (θ) = EB∼D,(q,a)∼B,{oi}G

i=1∼πθold(·|q)

i|

t=1 I Hti ≥ τρB

t=1

i=1

(6)

· min rti(θ)Aˆit, clip rti(θ),1 − ϵlow,1 + ϵhigh A ˆit , s.t. 0 < oi | is_equivalent(a, oi) < G.

Here, Hti denotes the entropy of token t in response i, I[·] is the indicator function that evaluates to 1 if the condition inside holds and 0 otherwise, ρ ∈ (0,1] is a predefined ratio specifying the top proportion of

high-entropy tokens to be selected within a batch, and τρB is the corresponding entropy threshold within the batch B such that only tokens with Hti ≥ τρB, comprising the top-ρ fraction of all tokens in the batch, are used to compute the gradient.

Comparing Equation (6) with Equation (4), there are only two differences, as highlighted in red in Equation (6): (i) The advantage term is multiplied by I Hti ≥ τρB , ensuring that within each (micro-)batch B, only tokens oti whose corresponding entropy Hti ≥ τρB are involved in the policy gradient loss calculation; (ii) The normalization term for the number of tokens is adjusted to ∑iG=1 ∑|o

i|

t=1 I Hti ≥ τρB , meaning that only tokens whose entropy is at least the threshold τρB are considered.

##### 5.2 Experimental Setup

Training details We adapt our training codebase from verl (Sheng et al., 2024) and follow the training recipe of DAPO (Yu et al., 2025), one of the state-of-the-art RL algorithms for LLMs. Both configurations, RLVR with full gradients (vanilla DAPO depicted in Equation (4)) and RLVR with only policy gradients on forking tokens (described in Equation (6)), employ techniques such as clip-higher, dynamic sampling,

- Table 2: Comparison between vanilla DAPO using all tokens and DAPO using only the top 20% high-entropy tokens (i.e. forking tokens) in policy gradient loss, evaluated on the Qwen3-32B, Qwen3-14B and Qwen38B base models. "Acc@16" and "Len@16" denotes the average accuracy and response length over 16 evaluations per benchmark, respectively.

DAPO w/ All Tokens DAPO w/ Forking Tokens Improvement

Benchmark

Acc@16 Len@16 Acc@16 Len@16 Acc@16 Len@16

RLVR from the Qwen3-32B Base Model

|AIME’24<br>AIME’25 AMC’23 MATH500 Minerva Olympiad<br>|55.83 9644.15 63.54 12197.54 45.63 9037.48 56.67 11842.25<br><br>91.88 5285.03 94.22 5896.47 94.36 2853.51 94.88 3366.01 45.70 2675.28 45.82 2759.88 66.16 5597.37 69.02 7300.01<br><br>|+7.71 +2553.39<br><br>+11.04 +2804.77 +2.34 +611.44<br><br>+0.52 +512.5<br><br>+0.12 +84.6<br><br>+2.86 +1702.64<br><br>|
|---|---|---|
|Average|66.59 5848.80 70.69 7227.03|+4.10 +1378.22|

RLVR from the Qwen3-14B Base Model

|AIME’24<br><br>AIME’25 AMC’23 MATH500 Minerva Olympiad<br><br><br>|45.21 7945.15 50.42 11814.36 38.13 7056.98 42.92 12060.48 89.53 4509.37 91.56 7095.13 92.23 2348.22 93.59 3970.10 42.16 2011.16 43.20 2959.32 61.14 4642.07 64.62 7871.25|+5.21 +3869.21<br><br>+4.79 +5003.5<br><br>+2.03 +2585.76<br><br>+1.37 +1621.88<br><br>+1.03 +948.16<br><br>+3.48 +3229.18<br>|
|---|---|---|
|Average<br><br>|61.40 4752.16 64.39 7628.44<br><br>|+2.99 +2876.28|

RLVR from the Qwen3-8B Base Model

|AIME’24<br><br>AIME’25 AMC’23 MATH500 Minerva Olympiad<br><br><br>|33.33 6884.89 34.58 9494.29 25.42 5915.91 26.25 8120.20 77.81 3967.91 77.19 5450.62 89.24 2059.00 89.70 2672.91 39.77 1450.68 40.26 2068.41 56.67 3853.55 57.43 5241.54|+1.25 +2609.40 +0.83 +2204.29 -0.625 +1482.71<br><br>+0.46 +613.91 +0.48 +617.73 +0.76 +1387.99<br><br>|
|---|---|---|
|Average<br><br>|53.71 4021.99 54.23 5508.00|+0.53 +1486.01|

token-level policy gradient loss, and overlong reward shaping (Yu et al., 2025). For fair comparisons, we apply the same hyperparameters as recommended by DAPO: for clip-higher, ϵhigh = 0.28, ϵlow = 0.2; for overlong reward shaping, the maximum response length is 20480 and the cache length is 4096. Furthermore, we use a training batch size of 512 and a mini-batch size of 32 in verl’s configuration, resulting in 16 gradient steps per training batch, with a learning rate of 10−6 and no learning rate warmup or scheduling. Importantly, the training process excludes both KL divergence loss and entropy loss. To evaluate the scaling ability of these methods, we perform RLVR experiments across the Qwen332B base and Qwen3-8B base models, using DAPO-Math-17K (Yu et al., 2025) as the training dataset. For main results, we set ρ = 20% in Equation (6), meaning that the policy is updated using only the gradients of the top 20% highest-entropy tokens within each batch. The chat template we use for Qwen3 models is "User:\n[question]\nPlease reason step by step, and put your final answer within \boxed{}.\n\nAssistant:\n" with "<|endoftext|>" serving as the EOS token, where "[question]" should be replaced by the specific question.

Evaluation We evaluate our models on 6 standard mathematical reasoning benchmarks commonly used for assessing reasoning capabilities: AIME’24, AIME’25, AMC’23, MATH500 (Hendrycks et al.,

- 2021), Minerva, and OlympiadBench (He et al., 2024). All evaluations are conducted in a zero-shot setting. For each question, we generate 16 independent responses under a decoding temperature T = 1.0, and report the average accuracy and the average number of tokens per response.

##### 5.3 Main Results

High-entropy tokens drive reinforcement learning for LLM reasoning. Figure 6 and Table 2 compare vanilla DAPO which uses all tokens, and our approach that retains only the top 20% high-entropy tokens in the policy gradient loss. Surprisingly, discarding the bottom 80% low-entropy tokens does

[Figure 14]

[Figure 15]

(a) AIME’24 scores trained from Qwen3-32B base.

(b) Response lengths trained from Qwen3-32B base.

[Figure 16]

[Figure 17]

(c) AIME’24 scores trained from Qwen3-14B base. (d) Response lengths trained from Qwen3-14B base.

[Figure 18]

[Figure 19]

(e) AIME’24 scores trained from Qwen3-8B base. (f) Response lengths trained from Qwen3-8B base.

- Figure 6: A comparison of vanilla DAPO with full tokens and DAPO with top 20% high-entropy (forking) tokens in policy gradient loss was conducted on Qwen3-32B, Qwen3-14B, and Qwen3-8B models. (a) & (b) Qwen3-32B: Dropping the bottom 80% low-entropy tokens stabilizes training and improves the AIME’24 score by 7.73. (c) & (d) Qwen3-14B: Similarly, removing 80% low-entropy tokens yields a 5.21 increase in the AIME’24 score. (e) & (f) Qwen3-8B: Retaining only the top 20% forking tokens maintains performance. Additionally, using only the top 20% high-entropy tokens increases response length across all model sizes.

[Figure 20]

[Figure 21]

(a) AIME’24 scores trained from Qwen3-8B base. (b) Overall entropy trained from Qwen3-8B base.

[Figure 22]

[Figure 23]

(c) AIME’24 scores trained from Qwen3-14B base. (d) Overall entropy trained from Qwen3-14B base.

[Figure 24]

[Figure 25]

(e) AIME’24 scores trained from Qwen3-32B base. (f) Overall entropy trained from Qwen3-32B base.

- Figure 7: Comparison among DAPO using different range of tokens in policy gradient loss. Top x% means only using the x% of the tokens with highest entropy (x = 10,20,50), bottom 80% means only using the 80% of the tokens with lowest entropy, and 100% means using all tokens (i.e., vanilla DAPO). Furthermore, "overall entropy" refers to the average entropy over all tokens.

not degrade reasoning performance and can even lead to improvements across six benchmarks. On the Qwen3-32B base model, this approach delivers gains of 7.71 points on AIME’24 and 11.04 points on AIME’25. Similarly, the Qwen3-14B base model shows improvements of 5.21 points on AIME’24 and 4.79 points on AIME’25. For the Qwen3-8B base model, performance remains unaffected. These findings suggest that the gains in reasoning ability during RLVR are driven primarily by high-entropy tokens, while low-entropy tokens may have little effect on or could even hinder reasoning performance, particularly on the Qwen3-32B and Qwen3-14B base models.

To conduct a deeper analysis, we vary the proportion, denoted as ρ in Equation (6), for experiments, as shown in Figure 7(a). The results show that the performance of the Qwen3-8B base model remains relatively consistent across different proportions, such as 10%, 20%, and 50%. For the Qwen3-14B and Qwen3-32B base model, Figure 7(c) and (e) reveal that reducing ρ from 20% to 10% leads to a slight drop in performance, while increasing it sharply to 100% results in a notable decline. These observations indicate that within a reasonable range, reasoning performance is largely insensitive to the exact value of ρ. More importantly, they suggest that focusing on high-entropy tokens, rather than using all tokens, generally preserves performance, and could even offer substantial gains in larger models.

Low-entropy tokens contribute minimally to reasoning performance. As illustrated in Figure 7(a) and (b), retaining only the bottom 80% of tokens with low entropy during RLVR leads to a substantial decline in performance, even though these tokens account for 80% of the total token count used in training. This finding indicates that low-entropy tokens contribute minimally to enhancing reasoning capabilities, highlighting the greater importance of high-entropy tokens for effective model training.

The effectiveness of high-entropy tokens may lie in their ability to enhance exploration. Our analysis reveals that focusing on a subset of high-entropy tokens, approximately 20% in our experiments, strikes an effective balance between exploration and training stability in RLVR. As illustrated in Figure 7(b), (d) and (f), adjusting the ratio ρ from 20% to either 10%, 50%, or 100% leads to persistently lower overall entropy starting from the early training phase and continuing up to the point where performance begins to converge. Moreover, training with the bottom 80% of low-entropy tokens results in significantly reduced overall entropy. These findings indicate that retaining a certain proportion of high-entropy tokens may facilitate effective exploration. Tokens outside this range could be less helpful or possibly even detrimental to exploration, particularly during the critical phase before performance convergence. This might explain why, on the Qwen3-32B base model, DAPO using only the top 20% high-entropy tokens outperforms vanilla DAPO, as shown in Figure 6(a). However, on the Qwen3-8B base model, probably due to the model’s lower capacity, the benefits of enhanced exploration appear limited.

Focusing on forking tokens in the policy gradient loss benefits larger reasoning models. We present the scaling trend when utilizing only forking tokens in Figure 8. On the AIME’24 and AIME’25 benchmarks, we observe that as the model size increases, the performance gain over vanilla DAPO becomes

[Figure 26]

[Figure 27]

(a) Scaling Trend on AIME 2024 (b) Scaling Trend on AIME 2025

- Figure 8: Scaling trend of DAPO using only forking tokens (i.e., top 20% of high-entropy tokens) in policy gradient loss. These results suggest that concentrating exclusively on forking tokens in the policy gradient loss may yield greater benefits in larger reasoning models.

increasingly significant. This suggests a promising conclusion: focusing solely on forking tokens in the policy gradient loss could offer greater advantages in larger reasoning models.

##### 5.4 Analysis

Generalization ability to other domains As outlined in Section 5.2, we used the DAPO-Math-17K dataset, which primarily consists of mathematical data, for our RLVR experiments. Here, we test whether DAPO, when trained on a math dataset and using only a small fraction of high-entropy tokens in the policy gradient loss, can still surpass vanilla DAPO on out-of-distribution test sets, such as LiveCodeBench (Jain et al., 2024). The results comparing DAPO with only top 10% or 20% tokens with highest entropy to vanilla DAPO (which uses 100% tokens) on the Qwen3-32B base are illustrated in Figure 9, using the same setup described in Section 5.2. From these results, we observe that even when retaining only top 10% or 20% tokens with highest entropy, DAPO still significantly outperforms vanilla DAPO on the out-of-distribution test dataset LiveCodeBench. This finding suggests that high-entropy tokens may be associated with the generalization capabilities of reasoning models. Retaining only a small subset of tokens with the highest entropy could potentially enhance the generalization ability of reasoning models.

- 24

- 25

- 26

- 27

- 28

Top 10% Top 20% 100%

Accuracy(avg@16)

0 200 400 600 800 1000 1200 1400 Gradient Steps

- Figure 9: Comparison among DAPO using different range of tokens in policy gradient loss trained from the Qwen3-32B base model on the out-of-distribution LiveCodeBench Benchmark (Jain et al., 2024) (v5, Aug. 2024 to Feb. 2025). Top x% means only using the x% of the tokens with highest entropy (x = 10, 20), and 100% means using all tokens (i.e. vanilla DAPO). Due to the high variance of the accuracy curves, we smooth the curves using window smoothing with a window size of 10.

Unlocking more potential of RLVR with only forking tokens In the experiments described in Section 5.3, we set a maximum response length of 20480. As shown in Figure 10, we increased the maximum response length for DAPO w/ only forking tokens (depicted in Figure 6(a) and (b))—trained from the Qwen3-32B base model—to 29696. This adjustment resulted in an improvement of the already SoTA performance on AIME’24, increasing from 63.54 to 68.12. These findings suggest that the full potential of our approach may not yet be realized, and with a longer context length or potentially more challenging training data, even greater performance gains could be achieved.

Results on models other than Qwen We compare DAPO using only forking tokens (i.e., the top 20% tokens with the highest entropy) against vanilla DAPO on models other than the Qwen series, specifically the Llama-3.1-8B model. When DAPO is applied to the Llama-3.1-8B base model, we observe that it achieves very low accuracy (approximately 1%) on the training dataset (i.e., DAPO-MATH-17K (Yu et al., 2025)) and often generates responses with repetitive words early in the RL training process. To address this, we use the Qwen3-32B model (Yang et al., 2025) as a teacher to generate responses for DAPO-MATH17K queries. From the generated queries, we randomly sample 10,000 with correct answers to serve as cold-start data and perform supervised fine-tuning (SFT) on the Llama-3.1-8B base model (Grattafiori

- et al., 2024). The remaining 7,398 queries are reserved for RL after the cold-start phase. The AIME’24 score and response length during RL training are plotted in Figure 11. The results indicate that DAPO with only forking tokens still surpasses vanilla DAPO, while also producing longer responses on average. However, given the relatively low performance of both configurations on AIME’24, we believe the results on Llama-3.1-8B are less convincing compared to those observed on the Qwen3 models.

[Figure 28]

[Figure 29]

(a) AIME’24 scores (b) Response length

- Figure 10: By extending the maximum response length from 20,480 to 29,696 and continuing training from the SoTA 32B model shown in Figure 6(a) and (b), the AIME’24 scores improve further from 63.54 to 68.12, alongside a notable increase in response length.

[Figure 30]

[Figure 31]

(a) AIME’24 scores trained from Llama-3.1-8B after cold-start.

(b) Response length trained from Llama-3.1-8B after cold-start.

- Figure 11: Comparison of DAPO using only forking tokens and vanilla DAPO, both trained from Llama3.1-8B after cold-start.

#### 6 Discussions

- Discussion 1: High-entropy minority tokens (i.e., forking tokens) could play a key role in explaining why RL generalizes while SFT memorizes. Chu et al. (2025) demonstrated empirically that RL, particularly with outcome-based rewards, exhibits strong generalization to unseen, rule-based tasks, whereas supervised fine-tuning (SFT) is prone to memorizing training data and struggles with generalization outside the training distribution. We hypothesize that one critical factor underlying the differing generalization capabilities of RL and SFT may be related to entropy in forking tokens. Our experiments (e.g., Figure 5 and Figure 7) suggest that RL tends to preserve or even increase the entropy of forking tokens, maintaining the flexibility of reasoning paths. In contrast, SFT pushes output logits towards one-hot distributions, leading to reduced entropy in forking tokens and, consequently, a loss of reasoning path flexibility. This flexibility may be a crucial determinant of a reasoning model’s ability to generalize effectively to unseen tasks.
- Discussion 2: Unlike traditional RL, LLM reasoning integrates prior knowledge and must produce readable output. Consequently, LLM CoTs contain a mix of low-entropy majority tokens and highentropy minority tokens, whereas traditional RL can assume uniform action entropy throughout a

trajectory. As shown in Figure 2(a), most LLM CoT tokens have low entropy, with only a small fraction exhibiting high entropy. In contrast, traditional RL typically formulates each action distribution as Gaussian with a predefined standard deviation (Schulman et al., 2017; Raffin et al., 2021; Weng et al.,

- 2022), resulting in uniform entropy across actions. We attribute this distinct entropy pattern in LLM CoTs to their pretraining on large-scale prior knowledge and the need for language fluency. This forces most tokens to align with memorized linguistic structures, yielding low entropy. Only a small set of tokens that are inherently uncertain in the pretraining corpus allows for exploration, and thus exhibits high entropy. This deduction is consistent with our results in Table 1.

- Discussion 3: In RLVR, entropy bonus may be suboptimal, as it increases the entropy of low-entropy majority tokens. In contrast, clip-higher effectively promotes entropy in high-entropy minority tokens. In RL, entropy bonus is commonly added to the training loss to encourage exploration by increasing the entropy of actions—a well-established practice in traditional tasks (Schulman et al., 2017; Williams, 1992; Mnih et al., 2016), and recently applied to LLM reasoning (Sheng et al., 2024; Hu et al., 2024). However, as discussed above, unlike typical RL trajectories, LLM CoTs display distinct entropy patterns. Increasing entropy across all tokens can degrade performance by disrupting the low-entropy majority, while selectively increasing the entropy of high-entropy minority tokens improves performance (Figure 3). Thus, uniformly applied entropy bonuses are suboptimal for CoT reasoning. Instead, clip-higher (Yu

- et al., 2025), which moderately raises ϵhigh in Equation (4), better targets high-entropy tokens. Empirically,

we observe that tokens with high importance ratios rt(θ) (as defined in Equation (2)) tend to have higher entropy. By including more of these tokens in training, clip-higher increases overall entropy without significantly affecting low-entropy tokens, as supported by Yu et al. (2025) and illustrated in Figure 5.

#### 7 Related Work

Reinforcement learning for LLM Before the advent of reasoning-capable models like OpenAI’s o1 (OpenAI, 2024), reinforcement learning (RL) was widely used in reinforcement learning from human feedback (RLHF) to improve large language models’ (LLMs) instruction-following and alignment with human preferences (Ouyang et al., 2022). RLHF methods are broadly categorized into online and offline preference optimization. Online methods, such as PPO (Schulman et al., 2017), GRPO (Shao et al., 2024), and REINFORCE (Williams, 1992), generate responses during training and receive real-time feedback. Offline methods like DPO (Rafailov et al., 2023), SimPO (Meng et al., 2024), and KTO (Ethayarajh et al., 2024) optimize policies using pre-collected preferences, typically from human annotators or LLMs. While offline methods are more training-efficient, they often underperform compared to online approaches (Tang et al., 2024). Recently, RL with verifiable rewards (RLVR)(Lambert et al., 2025) has emerged as a promising approach for enhancing reasoning in LLMs, particularly in domains like mathematics and programming(Shao et al., 2024; DeepSeek-AI et al., 2025; Yang et al., 2025; Lambert et al., 2025). OpenAI o1 (OpenAI, 2024) was the first to show that RL can effectively incentivize reasoning at scale. Building on o1, models such as DeepSeek R1 (DeepSeek-AI et al., 2025), QwQ (Team, 2025), Kimi k1.5 (Team et al., 2025), and Qwen3 (Yang et al., 2025) have aimed to match or exceed its performance. DeepSeek R1 stands out for showing that strong reasoning can emerge through outcome-based optimization using the online RL algorithm GRPO (Shao et al., 2024). It also introduced a “zero RL” paradigm, where reasoning abilities are elicited from the base model without conventional RL fine-tuning. Inspired by these results, subsequent methods such as DAPO (Yu et al., 2025), VAPO (Yue et al., 2025b), SimpleRLZoo (Zeng et al., 2025), and Open-Reasoner-Zero (Hu et al., 2025) have further explored RL-based reasoning. In this work, we use DAPO as our baseline to investigate key aspects of RL applied to LLMs.

Analysis on reinforcement learning with verifiable rewards Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a prevalent approach to enhance the reasoning capabilities of large language models (LLMs). Several studies have analyzed the characteristics of RLVR and its related concepts. Gandhi et al. (2025) find that the presence of reasoning behaviors, rather than the correctness of answers, is the key factor driving performance improvements in reinforcement learning. Similarly, Li et al. (2025) show that the structure of long chains of thought (CoT) is critical to the learning process, while the content of individual reasoning steps has minimal impact. Vassoyan et al. (2025) identify "critical tokens" in CoTs, which are decision points where models are prone to errors, and propose encouraging exploration around these tokens by modifying the KL penalty. Lin et al. (2024) also identify critical tokens that significantly influence incorrect outcomes and demonstrate that identifying and replacing these tokens can alter model behavior. Our finding that RLVR primarily focuses on forking tokens in reasoning paths may share some common ground with the observations of Gandhi et al. (2025) and Li et al. (2025), who suggest that RLVR primarily learns the format rather than the content. However, our analysis goes further by identifying the finding at the token level. Moreover, the concept of critical tokens in Vassoyan et al. (2025) and Lin et al. (2024) is closely related to the high-entropy minority tokens we introduce. In

contrast to prior work, which judges token importance based on correctness of the output, we propose token entropy as a criterion that may more accurately reflect the underlying mechanisms of LLMs.

#### 8 Conclusion

In this work, we analyze RLVR through a novel perspective of token entropy, providing fresh insights into the mechanisms of reasoning in LLMs. Our study of CoT reasoning shows that only a small subset of tokens exhibit high entropy and serve as forks in reasoning paths that influence reasoning directions. Additionally, our analysis of entropy dynamics during RLVR training reveals that the reasoning model largely retains the base model’s entropy patterns, with RLVR mainly modifying the entropy of already high-entropy tokens. Building on these findings, which underscore the significance of high-entropy minority tokens, we restrict policy gradient updates in RLVR to the top 20% highest-entropy tokens. This approach achieves performance comparable to, or even surpassing, full-token RLVR training, while exhibiting a strong scaling trend with model size. In contrast, directing optimization toward the lowentropy majority results in a significant decline in performance. These findings indicate that RLVR’s effectiveness stems primarily from optimizing this high-entropy subset, suggesting more focused and efficient strategies for improving LLM reasoning capabilities.

Limitations We believe there is still room for improvement in our work. First, our experiments could be extended to models beyond the Qwen family. Although we attempted to evaluate our approach on LLaMA models, they struggled to achieve meaningful performance on the AIME benchmarks. Additionally, the scope of our dataset could be expanded to encompass domains beyond mathematics, such as programming or more complex tasks like ARC-AGI (Chollet et al., 2025; Chollet, 2019). Furthermore, our findings are based on specific experimental settings, and it is possible that the observations and conclusions presented in this paper may not generalize to all RLVR scenarios. For instance, in a different RLVR setting, the effective proportion of 20% observed in our experiments may need to be adjusted to a different value to achieve optimal results.

Future Directions Future directions involve developing new RLVR algorithms to better leverage highentropy minority tokens and exploring how these insights can enhance not only RLVR but also other approaches, such as supervised fine-tuning (SFT), distillation, inference, and multi-modal training.

#### References

Anthropic. Claude 3.7 Sonnet. https://www.anthropic.com/claude/sonnet, 2025. [Accessed 01-05-2025]. Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, and Henry Pinkard. Arc-agi-2: A new challenge

for frontier ai reasoning systems. arXiv preprint arXiv: 2505.11831, 2025. François Chollet. On the measure of intelligence. arXiv preprint arXiv: 1911.01547, 2019. Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine,

and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas

Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rstar-math: Small llms can master math reasoning with self-evolved deep thinking. arXiv preprint arXiv:2501.04519, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. arXiv preprint arXiv:2405.11143, 2024.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2025.

Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G Patil, Matei Zaharia, et al. Llms can easily learn to reason from demonstrations structure, not content, is what matters! arXiv preprint arXiv:2502.07374, 2025.

Zicheng Lin, Tian Liang, Jiahao Xu, Xing Wang, Ruilin Luo, Chufan Shi, Siheng Li, Yujiu Yang, and Zhaopeng Tu. Critical tokens matter: Token-level contrastive estimation enhence llm’s reasoning capability. arXiv preprint arXiv:2411.19943, 2024.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 124198–124235. Curran Associates, Inc., 2024.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Proceedings of The 33rd International Conference on Machine Learning, 2016.

OpenAI. Learning to Reason with LLMs. https://openai.com/index/learning-to-reason-with-llms/, 2024. [Accessed 01-05-2025].

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning, 2025.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stablebaselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 2021.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Yunhao Tang, Daniel Zhaohan Guo, Zeyu Zheng, Daniele Calandriello, Yuan Cao, Eugene Tarassov, Rémi Munos, Bernardo Ávila Pires, Michal Valko, Yong Cheng, et al. Understanding the performance gap between online and offline alignment algorithms. arXiv preprint arXiv:2405.08448, 2024.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github. io/blog/qwq-32b/.

Jean Vassoyan, Nathanaël Beau, and Roman Plaud. Ignore the kl penalty! boosting exploration on critical tokens to enhance rl fine-tuning. arXiv preprint arXiv:2502.06533, 2025.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, and Yelong Shen. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025.

Jiayi Weng, Huayu Chen, Dong Yan, Kaichao You, Alexis Duburcq, Minghao Zhang, Yi Su, Hang Su, and Jun Zhu. Tianshou: A highly modularized deep reinforcement learning library. Journal of Machine Learning Research, 2022.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 1992.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025a.

Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, Xiangyu Yu, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yonghui Wu, and Lin Yan. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025b.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025.

Okay , let 's tackle this problem step by step . The problem is about finding a

list of positive integers with specific properties : the sum is 3 0 , the unique

mode is 9 , and the median is a positive integer not present in the list . Then ,

we need to find the sum of the squares of all the items in the list . \n\n First ,

let me recall what each term means . The mode is the number that appears most

frequently in the list . Since it 's specified as the unique mode , that means 9

appears more times than any other number in the list . The median is the middle

value when the list is ordered from smallest to largest . If the list has an odd

number of elements , the median is the middle one ; if even , it 's the average of

the two middle numbers . However , the problem says the median is a positive

integer that does not appear in the list . So the median must be an integer ,

which implies that the list must have an odd number of elements because if it

were even , the median would be the average of two numbers , which could be a non

- integer . Therefore , the list must have an odd number of elements , say $ n $ ,

where $ n $ is odd . \n\n Let me note down the key points again :\n\n 1 . Sum of

all elements = 3 0 .\n 2 . Unique mode is 9 : so 9 appears more times than any other

number .\n 3 . Median is a positive integer not in the list .\n\n Our goal is to

find the sum of squares of all elements .\n\n First , let 's think about the median

. Since the median is not in the list , and the median is an integer , let 's

denote the number of elements as $ n $ , which is odd . Let me denote $ n = 2 k

+ 1 $ , where $ k $ is an integer . Then the median is the $ ( k + 1 ) $ - th

term in the ordered list . Since the median is not in the list , that term must

be a number that does not appear in the list . \n\n Wait , but how can the median

not be in the list ? Let me think . For example , if the list is [ 1 , 2 , 3 ], the

median is 2 , which is in the list . If the list is [ 1 , 3 , 5 ], median is 3 , which

is in the list . So how can the median not be in the list ? Maybe the median is a

number that is between two numbers in the list ? Wait , but the median is a

|[Figure 32]| | | | |
|---|---|---|---|---|
| | | | | |

0.000 1.316 2.632 3.948 5.264 6.580 Token Entropy

Figure 12: Visualization of token entropy (part 1).

specific value . If the list is ordered , the median is the middle element . If

that element is not in the list , that seems impossible . Wait , but maybe the

list has duplicates ?\n\n Wait , hold on . For example , suppose the list is [ 1 , 2 ,

2 , 3 , 4 ]. The median is 2 , which is in the list . If the list is [ 1 , 2 , 3 , 4 , 5 ],

median is 3 , which is in the list . So how can the median not be in the list ? Is

there a case where the median is not present ?\n\n Wait , maybe if the list has

multiple instances of the median ? But then the median would be present . Wait ,

perhaps the median is not an element of the list ? Wait , that 's impossible .

Because the median is the middle element of the ordered list . So it must be an

element of the list . Therefore , how can the median not be in the list ? \n\n Wait

, maybe the problem is not saying that the median is not present as an element ,

but that the median is a positive integer that does not appear in the list .

Wait , that 's exactly what it says . So the median is a positive integer ( so it 's

an integer , so n must be odd , as discussed before ), and it does not appear in

the list . But if the median is the middle element of the list , then the median

must be an element of the list . Therefore , there 's a contradiction here . Wait ,

unless the list is empty ? But the problem says it 's a list of positive integers

. So that can 't be . \n\n Wait , maybe I made a mistake here . Let me check again .

The median is the middle value when the list is ordered . So for example , if the

list is [ 1 , 2 , 3 , 4 , 5 ], the median is 3 . If the list is [ 1 , 1 , 2 , 3 , 4 ], the

median is 2 . So the median is an element of the list . Therefore , how can the

median not be in the list ? \n\n Wait , unless the median is not an element ? But

that contrad icts the definition . Wait , unless there 's a misunderstanding . Maybe

the problem is referring to the median value , not the median element ? For

example , if the list has an even number of elements , then the median is the

average of the two middle numbers . But the problem says the median is a

positive integer . So if the list has an even number of elements , the average of

|[Figure 33]| | | | |
|---|---|---|---|---|
| | | | | |

0.000 1.316 2.632 3.948 5.264 6.580 Token Entropy

Figure 13: Visualization of token entropy (part 2).

two numbers must be an integer , so they must add up to an even number . But then

the median is an integer , but it 's not necessarily an element of the list . \n\n

Ah ! So perhaps the list has an even number of elements , and the median is the

average of the two middle numbers , which is an integer not present in the list .

That would make sense . Therefore , the list must have an even number of elements

. Therefore , my previous assumption that it must be odd was wrong . \n\n Let me re

-ex amine that . The problem says the median is a positive integer that does not

appear in the list . So if the median is not an element of the list , then the

list must have an even number of elements , so that the median is the average of

two numbers . Let me check with an example . Suppose the list is [ 1 , 2 , 3 , 4 ].

The median is ( 2 + 3 )/ 2 = 2 . 5 , which is not an integer . If the list is [ 1 , 2 , 2

, 3 ], the median is ( 2 + 2 )/ 2 = 2 , which is an integer and is in the list . If

the list is [ 1 , 2 , 3 , 4 , 5 , 6 ], median is ( 3 + 4 )/ 2 = 3 . 5 , not integer . If the

list is [ 1 , 2 , 3 , 5 , 5 , 7 ], the median is ( 3 + 5 )/ 2 = 4 , which is an integer .

However , 4 is not in the list . Therefore , in this case , the median is 4 , which

is not in the list . So yes , if the list has an even number of elements , the

median can be an integer not present in the list . Therefore , our list must have

an even number of elements . \n\n So , that was my mistake earlier . Therefore , the

number of elements is even . Let me denote n = 2 k , where k is a positive integer

. Then , the median is the average of the k -th and ( k + 1 )- th elements . For the

median to be an integer , the sum of the k -th and ( k + 1 )- th elements must be even

. Therefore , they are both even or both odd . \n\n So , to recap :\n\n - The list has

an even number of elements , say n = 2 k .\n\n - The sum of the elements is 3 0 .\n\n

- The unique mode is 9 , so 9 occurs more times than any other number .\n\n - The

median is an integer not in the list .\n\n Our goal is to find the sum of squares

of all items in the list .\n\n Therefore , we need to figure out possible n , the

number of elements , and the composition of the list .\n\n First , let 's consider

|[Figure 34]| | | | |
|---|---|---|---|---|
| | | | | |

0.000 1.316 2.632 3.948 5.264 6.580 Token Entropy

Figure 14: Visualization of token entropy (part 3).

)/ 2 = 8 , which is not in the list . Wait , this is the same as the list [ 5 , 7 , 9 , 9 ]

but with a = 1 and b = 7 . \n\n But sum is 1 + 7 + 9 + 9 = 2 6 3 0 . So need a + b + 9 + 9 = 3

- 0 => a + b = 1 2 . Therefore , in this case , a + b = 1 2 , with a <= b <= 9 . So possible

pairs ( a ,b ) as before . For example , ( 3 , 9 ) gives list [ 3 , 9 , 9 , 9 ], median is 9 . ( 4

, 8 ) gives [ 4 , 8 , 9 , 9 ], median 8 . 5 . ( 5 , 7 ) gives [ 5 , 7 , 9 , 9 ], median 8 . \n\n Wait , so [

- 1 , 1 1 ,... ] but b can 't be 1 1 . So only those pairs . \n\n So yes , the only valid

list is [ 5 , 7 , 9 , 9 ]. \n\n Therefore , sum of squares is 5 ² + 7 ² + 9 ² + 9 ² = 2 5 + 4 9

+ 8 1 + 8 1 = 2 3 6 . \n\n But just to make sure , let me check if there is another

list with different numbers but same properties . \n\n Sup pose the list has three

9 s and another number . Wait , but that leads to median being 9 . \n\n Or another

arrangement where the median is not in the list . For example , suppose the list

is [ a , b , c , d ] with two 9 s , and the median is ( b + c )/ 2 , which is not in the

list . We found that with a + b + c + d = 3 0 , and two 9 s , which gives only [ 5 , 7 , 9 , 9

]. \n\n Alternatively , suppose the two 9 s are not consecutive ? Wait , in a sorted

list , they must be consecutive or not ? If two 9 s are not consecutive , then

there are numbers between them . For example , [ 5 , 7 , 9 , 9 ] is consecutive . \n\n But

in that case , the median is ( 7 + 9 )/ 2 = 8 . Which is not in the list . \n\n

Alternatively , [ 5 , 9 , 7 , 9 ], sorted is [ 5 , 7 , 9 , 9 ]. Same as before . \n\n Therefore ,

no other possibilities . \n\n H ence , I think this is the only list . \n\n Therefore

, the answer should be 2 3 6 . \n\n But to be thorough , let me check if there are

other possibilities for n = 4 where the mode is 9 . \n\n Sup pose the list is [ x , y ,

z , w ], with two 9 s and two other numbers , but with the two other numbers being

same . Wait , but then the mode would be 9 and the other number , making two modes

. Which is invalid . \n\n For example , [ 5 , 5 , 9 , 9 ]. The sum is 5 + 5 + 9 + 9 = 2 8 . Not 3 0

. \n\n If [ 5 , 6 , 9 , 1 0 ], sum is 5 + 6 + 9 + 1 0 = 3 0 . The mode is no mode , or all numbers

appear once . So no mode . Therefore invalid . \n\n If [ 3 , 9 , 9 , 9 ], sum is 3 0 . As

before , median is 9 . \n\n If [ a , b , c , d ], with three 9 s and a different number ,

|[Figure 35]| | | | |
|---|---|---|---|---|
| | | | | |

0.000 1.316 2.632 3.948 5.264 6.580 Token Entropy

Figure 15: Visualization of token entropy (Part 4). For brevity, we omit the CoT following Figure 14 and preceding Figure 15.

but sum 3 0 . For example , two 9 s , but wait no . \n\n Alternatively , [ a , b , 9 , 9 ]

with a + b = 1 2 . If a and b are same , then [ a , a , 9 , 9 ], with a + a = 1 2 => a = 6 .

Then list is [ 6 , 6 , 9 , 9 ]. The median is ( 6 + 9 )/ 2 = 7 . 5 . Not integer . \n\n But the

mode is 6 and 9 , both appearing twice . Hence , not unique . \n\n Therefore ,

invalid . \n\n Therefore , only the list [ 5 , 7 , 9 , 9 ] is valid . \n\n Thus , the sum of

squares is 5 ² + 7 ² + 9 ² + 9 ² = 2 5 + 4 9 + 8 1 + 8 1 = 2 5 + 4 9 is 7 4 , plus 8 1 is 1 5 5 ,

plus 8 1 is 2 3 6 . \n\n Therefore , the answer is 2 3 6 . \n\n ** Final Answer **\n \ boxed {

2 3 6 }\n </think> \n\n To solve this problem , we need to determine the sum of the

squares of the items in a list of positive integers that satisfies the

following conditions :\n\n 1 . ** Sum of the list is 3 0 **\n 2 . ** The unique mode is

9 **\n 3 . ** The median is a positive integer that does not appear in the list

itself **\n\n ---\n\n ### Step 1 : Determine the Number of Elements in the List \n\n

Let s denote the number of elements in the list as $ n $. Since the median is

a positive integer ** not in the list **, this implies that the list must have an

** even number of elements ** . \n\n For an odd number of elements , the median is

the middle element of the sorted list , which is ** in the list ** . Therefore , it

cannot be an integer ** not in the list ** . Thus , the list must have an even

number of elements .\n\n Let s consider small even values of $ n $ and test for

the validity of the conditions .\n\n ---\n\n ### Step 2 : Consider $ n = 4 $\n\n

Let s assume the list has 4 elements . Let the elements be sorted as $ [ a , b , c ,

d ] $, so the median is :\n\n $$ \n \ text { Median } = \ frac { b + c }{ 2 }\n $$ \n\n We

are told that the ** median is a positive integer not in the list **, and that

the ** unique mode is 9 ** .\n\n For 9 to be the ** unique mode **, it must appear

more often than any other number . Since we only have 4 elements , the only

possibility is that 9 appears ** tw ice **, and all other numbers appear ** once ** .

Thus , the list must be of the form : $ [ x , y , 9 , 9 ] $, where $ x \ ne q y $, and

$ x , y \ ne q 9 $ .\n\n We also require :\n\n - $ x + y + 9 + 9 = 3 0 \ Rightarrow x +

|[Figure 36]| | | | |
|---|---|---|---|---|
| | | | | |

0.000 1.316 2.632 3.948 5.264 6.580 Token Entropy

Figure 16: Visualization of token entropy (part 5).

y = 1 2 $\n - $\ frac { y + 9 }{ 2 }$ is an integer ( so $ y + 9 $ must be even , hence

$ y $ is odd )\n - $\ frac { y + 9 }{ 2 }$ is not in the list \n\n Let s test all

integer pairs $ ( x , y ) $ such that $ x + y = 1 2 $, $ x < y $, $ x \ ne q y $,

and $ x \ ne q 9 $, $ y \ ne q 9 $ :\n\n - $ x = 4 , y = 8 $: List is $ [ 4 , 8 , 9 , 9 ]$

Median = $ \ frac { 8 + 9 }{ 2 } = 8 . 5 $ ( not integer )\n - $ x = 5 , y = 7 $: List is

$ [ 5 , 7 , 9 , 9 ]$ Median = $ \ frac { 7 + 9 }{ 2 } = 8 $ ( integer ) and ** 8 is not in

the list **\n\n This list satisfies all the conditions :\n\n - ** Sum **: $ 5 + 7 + 9

+ 9 = 3 0 $\n - ** Unique mode **: 9 ( appe ars twice , others once )\n - ** Median is 8

**, which is not in the list \n\n Thus , the list $ [ 5 , 7 , 9 , 9 ]$ is valid .\n\n

---\n\n ### Step 3 : Compute the Sum of Squ ares \n\n Now we calculate the sum of the

squares of the elements in this list :\n\n $$ \n 5 ^ 2 + 7 ^ 2 + 9 ^ 2 + 9 ^ 2 = 2 5 + 4 9

+ 8 1 + 8 1 = 2 3 6 \n $$ \n\n ---\n\n ### Final Answer \n\n $$ \n \ boxed { 2 3 6 }\n $$

|[Figure 37]| | | | |
|---|---|---|---|---|
| | | | | |

0.000 1.316 2.632 3.948 5.264 6.580 Token Entropy

Figure 17: Visualization of token entropy (part 6).

