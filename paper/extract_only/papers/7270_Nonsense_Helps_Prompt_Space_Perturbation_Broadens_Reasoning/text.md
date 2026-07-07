# arXiv:2605.05566v1[cs.AI]7May2026

## Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration

Langlin Huang, Chengsong Huang, Jinyuan Li, Donghong Cai, Yuyi Yang, Jiaxin Huang Washington University in St. Louis {h.langlin, jiaxinh}@wustl.edu

##### Abstract

Reinforcement learning with verifiable rewards, particularly Group Relative Policy Optimization (GRPO), has significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, in complex tasks, GRPO frequently suffers from the “zero-advantage problem”: when all sampled rollouts for a query fail, the relative advantage collapses to zero. Consequently, the model loses effective training signals for these questions, wasting the training data and computational budget. While simply increasing the sampling budget for these questions is a common remedy, the static sampling policy inherently constrains reasoning exploration, limiting the success rate. In this paper, we propose Lorem Perturbation for Exploration (LOPE), a simple yet effective training framework to break this exploration bottleneck. We posit that task-irrelevant prompt-space perturbations can shift the model’s output distribution enough to unlock orthogonal reasoning pathways for hard questions. Specifically, LOPE prepends sequences stochastically assembled from Lorem Ipsum vocabulary (a pseudo-Latin placeholder text) to the prompts before resampling. Experiments across 1.7B, 4B, and 7B models demonstrate that LOPE significantly outperforms resampling with the original prompts. Further analysis reveals that other Latin-based random sequences with low perplexity are also effective perturbations. Our results establish LOPE as a strong baseline for broadening exploration in LLM reinforcement learning.

##### 1 Introduction

Code: https://github.com/shrango/LoPE

In recent years, Reinforcement Learning with Verifiable Rewards (RLVR) has proven highly effective in enhancing the reasoning capabilities of large language models (LLMs). Notably, Group Relative Policy Optimization (GRPO) (Yang et al., 2024a; Guo et al., 2025) has been widely recognized as a promising method. By leveraging the relative advantages among multiple responses generated for the same query, GRPO eliminates the need for a separate value model. However, this approach is severely compromised by the “zero-advantage problem”: when all sampled responses to a question fail, their relative advantages collapse to zero. As a result, the vital training signal for that query is lost, wasting not only valuable training data but also a massive computational cost during LLM rollouts.

A simple solution to this problem is to generate more responses per question. To achieve this, many works have explored adaptive rollout budget allocation (Liao et al., 2025; Li et al., 2025; Xiong et al., 2025). By providing more sampling attempts to hard questions, LLM has a better chance of hitting a correct answer and recovering the lost training signal. However, this approach has a clear limitation. Because these questions are simply too difficult for the model’s current policy, merely increasing the sampling budget could still yield a low resample success rate.

Prior research has widely shown that modifications to the input context implicitly influence an LLM’s output distribution (Xie et al., 2022; Dai et al., 2023; Goldwaser et al., 2025). Building on this principle, we hypothesize that a deliberate, prompt-level perturbation could alter the output distribution just enough to rescue the model from the zero-advantage

- ① Standard Rollout ② Standard Training

- ② Resample with Lorem-

[Figure 1]

number of samples: G

To solve this.... \boxed{1}

[Figure 2]

On success

Naive Prompt: Please reason step by step, and put your ﬁnal answer within

To solve it.... \boxed{2}

...

\boxed{}. Question:

GRPO

So the problem.... \boxed{2}

###### When all fail

Perturbed Prompt

③ Regroup Responses

number of samples: G'

[Figure 3]

Let's think.... \boxed{0}

Lorem Ipsum: Lorem ipsum dolor sit amet, consectetur adipiscing elit

Naive Prompt

+

[Figure 4]

...

number of samples: G

To solve this.... \boxed{1}

[Figure 5]

First, let me.... \boxed{-1}

To solve it.... \boxed{2}

...

④ Build Pseudo Rollouts and Train

Naive Prompt

+

GRPO

Let's think.... \boxed{0}

Figure 1: Overview of LOPE. During the standard rollout phase, if all G responses fail, LOPE prepends a random Lorem Ipsum sequence to the prompt and resamples G′ responses. Successful reasoning responses are regrouped with original failed responses to form a mixed batch of size G for policy update.

trap. If the model is persistently failing on a hard question, perturbing the prompt during the rollout phase might unlock orthogonal reasoning pathways and discover successful trajectories that standard resampling cannot reach.

To test this hypothesis without introducing misleading facts or task-relevant hints, we require a task-irrelevant perturbation. Therefore, we draw inspiration from Lorem Ipsum, a pseudo-Latin placeholder text designed to mimic natural language without conveying actual semantic meaning. Specifically, we construct a perturbation by randomly sampling words from the Lorem Ipsum vocabulary. By prepending the random Lorem Ipsum to the standard prompt, we introduce a pure prompt-space perturbation. We refer to these modified inputs as Lorem-perturbed prompts. Based on this insight, we propose Lorem Perturbation for Exploration (LOPE), a simple yet highly effective rollout-and-resample framework designed to address the zero-advantage issue. We find that resampling with Lorem-perturbed prompts achieves a higher success rate on previously failed questions. This improvement is consistently observed throughout the entire RLVR training process and ensures effective training signals on a broader set of training questions than repeatedly sampling with the original unmodified prompt.

LOPE follows a similar training procedure to GRPO but differs in how it handles zeroadvantage cases. For questions where all initial responses fail, we resample using Loremperturbed prompts instead of the naive prompt. Experimental results show that our method consistently improves model performance across multiple mathematical reasoning benchmarks, achieving an average gain of +2.79 points on Qwen3-1.7B-Base, +4.62 points on Qwen3-4B-Base, and +6.20 points on Qwen2.5-Math-7B.

Furthermore, we conduct a comprehensive comparison of various prompt-space perturbation methods. While not all random perturbation strategies yield substantial improvements as LOPE does, the success of LOPE is not an isolated case. A few other perturbations, such as random sequences composed of high-frequency Latin words, achieve comparable results. We observe that the most effective perturbations share two decisive characteristics: (1) they use pseudo-Latin vocabularies to prevent interference with the English reasoning context, and (2) they maintain low perplexity to ensure high-quality rollouts. Overall, our results demonstrate that LOPE serves as a strong and generalizable baseline for broadening exploration in LLM reinforcement learning.

##### 2 Background: Group Relative Policy Optimization (GRPO)

Group Relative Policy Optimization (GRPO) (Shao et al., 2024) is a widely used reinforcement learning algorithm for improving LLM reasoning capabilities. Compared with PPObased approaches (Schulman et al., 2017), GRPO is free of an explicit reward model and leverages the relative correctness among multiple responses sampled for the same question.

Formally, given a query q and prompt p, it samples a group of G responses {oi}iG=1 from the old policy πθold, where each response is a sequence oi = (oi,1, . . . , oi,|oi|). The training objective is to maximize the following equation:

|oi|

G

1 G

1 |oi|

∑

∑

min ρi,tAi,clip ρi,t,1 − ϵ,1 + ϵ Ai − βDKL[πθ ∥πref] ,

JGRPO(θ) = Eq,{o

i}iG=1∼πθold(O|p,q)

t=1

i=1

(1) where ρi,t is the importance sampling ratio, defined as:

πθ(oi,t | p, q, oi,<t) πθold(oi,t | p, q, oi,<t)

. (2)

ρi,t =

Here, πθ is the current policy, πθold is the old policy used for sampling. πref is the reference policy that serves as a regularizer to prevent πθ from deviating excessively from the initial distribution. This is achieved by a Kullback-Leibler (KL) divergence term, where β controls the weight of KL. The clipping parameter ϵ prevents excessively large policy updates that could destabilize training.

Let ri denote the scalar reward of the ith response, where i ∈ {1,2,..., G}. The rollout-level advantage Ai is computed by normalizing the rewards within the same group:

ri − mean(r) std(r)

, r = [r1, . . . ,rG]. (3)

Ai =

Particularly, when all sampled responses to a question fail, resulting in a zero reward vector (r = 0), the advantage Ai collapses to 0 for all i. Consequently, the training batch yields a zero gradient, wasting the computational budget allocated for the rollouts.

##### 3 The Limitation of Logit-Space Exploration

When GRPO encounters the zero-advantage issue, a common and straightforward remedy is to resample additional responses for those questions. However, if an LLM fails to produce any correct answer within the first G rollouts (e.g., G = 8), it indicates that the question is intrinsically difficult under the current generation policy. In such cases, standard resampling is unlikely to significantly improve the resample success rate.

Distribution of successful questions (Hard subset, Pass@8, Total=352)

Distribution of successful questions (Pass@8, Total=500)

Naive Prompt (Base, T=0.6)

Naive Prompt (Base, T=0.6)

Naive Prompt (High-temp, T=1.2)

Naive Prompt (High-temp, T=1.2)

16

16

1

8

9

14

53

23 6

79 20

50

73

Lorem-perturbed Prompt (T=0.6)

Lorem-perturbed Prompt (T=0.6)

(a) 500-question subset

(b) Hard 352-question subset

Figure 2: Venn diagrams of successfully resolved questions (Pass@8) between naive prompting, Lorem perturbation, and high-temperature settings.

Traditionally, LLM generation encourages exploration by operating in the logit space, such as high-temperature sampling. We hypothesize that using high-temperature sampling

alone is insufficient to shift the model out of its local reasoning basin. Previous work extensively studied that In-Context Learning (ICL) is essentially changing the model’s output distribution (Xie et al., 2022). In this paper, we investigate whether prompt-space perturbation, which perturbs the input context, can more effectively force the model to explore orthogonal reasoning trajectories compared to logit-space perturbation.

To this end, we conduct a preliminary experiment comparing three settings: (1) Naive Prompt (Base): The original prompt with the system prompt and question only with a standard evaluation temperature of 0.6, serving as the base setting, (2) Naive Prompt (Hightemp): the original prompt with a higher temperature of 1.2 to encourage greater logit-space exploration, and (3) Lorem-perturbed Prompt: we prepend a randomly generated Lorem Ipsum sequence to the naive prompt while keeping the temperature at 0.6.

Lorem Ipsum is a standard placeholder text widely used in publishing and graphic design. It consists of meaningless pseudo-Latin text that mimics the typical structural and statistical properties of natural language (such as word lengths and sentence boundaries) without carrying any meaningful semantic content. We use the python-lorem implementation (Jarry Shaw, 2024), where each word is uniformly sampled from a pool of 63 Latin words.

We evaluate these prompt formulations on 500 randomly sampled questions from the Openr1-Math-46k-8192 dataset (Yan et al., 2025), using the Qwen3-1.7BBase model (Team, 2025). To visually quantify the exploration overlap among different prompting strategies, we plot Venn diagrams (Figure 2) to show the set of distinct and overlapping questions successfully resolved within Pass@8 by each prompting formulation. Results for the 500-question evaluation set are shown in Figure 2a. Furthermore, we construct a hard subset consisting of 352 questions that fail under the initial Pass@8 under the naive prompt setting. We then re-evaluate all three prompting formulations with a secondary 8-rollout sample budget on this subset, with results presented in Figure 2b.

Entropy

Perplexity

20

40

Naive Prompt (Base) (mean=0.187)

Naive Prompt (Base) (mean=1.121)

Naive Prompt (High-temp) (mean=0.973)

Naive Prompt (High-temp) (mean=3.043)

15

30

Lorem-perturbed Prompt (mean=0.239)

Lorem-perturbed Prompt (mean=1.157)

Density

Density

10

20

5

10

0

0

0.0 0.1 0.2 0.3 0.4 0.5 0.6

1.0 1.1 1.2 1.3 1.4

Figure 3: Probability distributions of response entropy and perplexity across different prompting and sampling formulations.

Our observations are twofold. First, the Lorem-perturbed generations can actually resolve a large number of questions compared to both standard logit-space approaches (base and higher temperature). Second, prompt-space perturbation unlocks orthogonal reasoning spaces that logit-space methods fail to explore. As shown in Figure 2b, when resampling the 352 hard questions, the Lorem-perturbed responses independently resolve 50 unique questions that neither of the other two methods could answer. This suggests that promptspace perturbation can effectively broaden the exploration of LLM without degrading its overall reasoning ability, particularly on more challenging questions.

To further understand this phenomenon, we analyze the generated responses at the token level. Figure 3 presents the probability density distributions of token-level entropy and perplexity across all responses. We observe that responses from the naive prompt (base) setting are heavily concentrated around low-entropy (near 0) and low-perplexity (near 1) regions, indicating highly confident but potentially over-constrained generation. In contrast, Lorem-perturbed responses eliminate the near-zero entropy spike and slightly right-shift the distribution, reflecting higher uncertainty and exploring behavior during generation. The high temperature setting, however, produces many responses with much higher entropy and perplexity, which can hurt the reasoning quality and accuracy.

##### 4 Lorem Perturbation for Exploration (LoPE)

Inspired by our findings that Lorem-perturbed prompts recover more failed questions and improve LLM exploration, we propose Lorem Perturbation for Exploration (LOPE), a simple yet effective resampling strategy to enhance exploration in reinforcement learning. We describe the details below.

Rollout with Perturbation. During the rollout stage, resampling is triggered only for questions where all G responses {oj}Gj=1 generated from πθold(o | p, q) under the naive prompt p fail. For such cases, LOPE prepends a random Lorem Ipsum sequence to the original prompt, serving as a text perturbation δ. This results in a perturbed prompt δ ⊕ p, which is then used to generate an additional set of G′ responses: {o′j}Gj=′1 ∼ πθold(o′|δ ⊕ p, q). An illustration of this process is presented in Figure 1.

Regroup LLM Responses. In the policy update stage, LOPE maintains the group size of G for advantage calculation. Specifically, we construct a hybrid batch of rollouts by combining failed responses from the original rollouts with successful responses from the

resampled set. Let c denote the number of correct responses in the resampled set {o′j}Gj=′1. We randomly select Ns = min(c, G − 1) correct responses from the resampled pool and use them to replace an equal number of failed responses in the original group. Importantly, we ensure that at least one incorrect response remains in the group, so that relative advantages are non-zero and meaningful for optimization.

Construct Pseudo Rollout with Resampled Responses. Directly grouping and comparing responses generated from different input contexts can cause a biased advantage estimation. To align the context of all responses for a given question, we construct pseudo rollouts by pairing each resampled response o′ with the naive prompt p and question q for training. Concretely, the full sequence used for training is (p, q, o) for original rollouts and (p, q, o′)

for resampled rollouts, despite o′ being generated under policy πθold(o′ | δ ⊕ p, q).

This substitution, however, results in an off-policy optimization scenario. To correct for the discrepancy between the sampling and training policies, we apply the importance sampling ratio defined in Eq. (4) for the resampled responses.

πθ(oi′,t | p, q, oi′,<t) πθold(oi′,t | δ ⊕ p, q, oi′,<t)

. (4)

ρi,t =

Removal of KL Regularization. In addition, LOPE removes the KL regularization term in Eq. (1). The introduction of random-word perturbations is intended to promote broader exploration, while a KL constraint restricts such distributional shifts and therefore counteracts this objective. Empirically, prior work (Wang et al., 2026) has shown that removing KL regularization is beneficial when training with multiple prompt patterns.

##### 5 Training Signal Shaping

Within the foundational LOPE framework, resampling via prompt space perturbation effectively enhances training data utilization. However, the off-policy training often diminishes the gradient magnitude of rare reasoning trajectories, as the policy probability πθ in Eq. (4) becomes small for these instances. Furthermore, while response regrouping concentrates computational resources on positive rollouts, calculating advantages solely based on the G selected responses underestimates the difficulty of questions and reduces the advantage signals for rare success samples.

To address these limitations, we introduce training signal shaping, which incorporates a policy shaping strategy and an advantage shaping strategy. These components are specifically

designed to mitigate the issues stemming from the importance sampling ratio ρi,t and the biased advantage estimation, respectively.

Policy Shaping. Training on pseudo-rollouts inherently constitutes an off-policy process due to the distributional discrepancy between the training policy πθ(oi′,t | p, q, oi′,<t) and the sampling policy πθold(oi′,t | δ ⊕ p, q, oi′,<t). Consequently, tokens with relatively low probabilities under πθ suffer from suppressed training weights (Wang et al., 2025). To address this issue, we adapt the policy shaping mechanism proposed by Yan et al. (2025):

ρi,t ρi,t + γ

, (5)

f(ρi,t) =

where γ is set to 0.1. This function constraints the gradient magnitude for high-probability tokens while amplifying it for low-probability ones. This adjustment is particularly crucial for resampled responses, as critical reasoning steps are often assigned low probabilities under the naive policy and would otherwise be inappropriately underweighted during

training. Notably, whereas Yan et al. (2025) assumes πθold ≡ 1, LOPE utilizes the exact values of πθold. A detailed analysis of how policy shaping impacts the training when relaxing the assumption of a fixed πθold is provided in Appendix C.1.

Advantage Shaping. In standard GRPO, the advantage for each response is computed by normalizing rewards within the sampled group of G responses, as defined in Eq. (3). In our resampling setting, however, the G responses selected for training comprise a mixture of original failed rollouts and resampled successful ones. Critically, the G′ discarded responses consist almost exclusively of failed attempts. Consequently, calculating the advantage solely on the G selected responses underestimates the difficulty of the question. This underestimation suppresses the absolute value of positive advantages, subsequently reducing the training weight assigned to rare successful samples.

To mitigate this bias, we propose an advantage shaping mechanism that computes the advantage over the complete set of G + G′ responses:

ri − mean(rall) std(rall)

Aˆi =

, rall = [r1, . . . ,rG,r1′ , . . . ,r′

G′], (6)

while restricting the gradient updates to the G selected responses. This formulation ensures that the normalization statistics faithfully reflect the true question difficulty, thereby restoring the authentic advantage values of successful samples and appropriately amplifying the reward signals for the rare successes. We quantitatively analyse the effect of advantage shaping in Appendix C.2, which amplifies the positive advantages by a factor of 2.1 to 5.0.

Full Training Objective. Combining the components above, the complete training objective of LOPE is formulated as:

JLOPE(θ) =

G

1 G

#### ∑

Eq,{o

i}iG=Ns+1∼πθold(O|p,q)

i=Ns+1

Ns

1 |oi|

#### ∑

###### +Eq,{o

i}iN=s1∼πθold(O|δ⊕p,q)

i=1

|oi|

1 |oi|

min ρi,tAˆi, clip ρi,t,1 − ϵ,1 + ϵ A ˆi

#### ∑

t=1

|oi|

f(ρi,t)Aˆi

#### ∑

t=1

(7)

where the first term corresponds to the standard GRPO updates on the original rollouts, and the second term incorporates policy shaping via f(ρi,t) for the resampled responses. The application of training signal shaping effectively resolves the limitations of the standard LOPE.

##### 6 Experiment

###### 6.1 Experiment Setup

We evaluate LOPE on three base models: Qwen3-1.7B-Base, Qwen3-4B-Base (Team, 2025), and Qwen2.5-MATH-7B (Yang et al., 2024a). For Qwen2.5-MATH-7B, whose original context length is 4096, we follow Yan et al. (2025) to extend the context window to 16384. During training, the maximum response length is set to 8192 tokens, and the maximum input length is 2048 tokens. Our implementation is based on EasyR1 (Zheng et al., 2025). Experiments on the 1.7B and 4B models are conducted on 4 × 80GB A100 GPUs, and those on the 7B model are conducted on 8 × 80GB A100 GPUs.

We use the OpenR1-Math-46k-8192 dataset (Yan et al., 2025) for training. For evaluation, we consider a diverse set of math reasoning benchmarks, including MATH-500 (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021), AMC (AMC 2022, 2023, and 2024), AIME 2024, and AIME 2025. We use EvalScope (Team, 2024) for evaluation, with a sampling temperature of 0.6 and top-p set to 0.95. For MATH-500, GSM8K, and AMC, we report Acc@1. For the more challenging benchmarks AIME 2024 and AIME 2025, we report Mean@32.

We compare LOPE against standard GRPO and the naive-prompt resampling baseline. The group size is set to G = 8, and the resampling size is G′ = 24. All rollouts are performed with a default temperature of 1.0. For fair comparison, all resampling-based methods remove the KL regularization term.

For perturbation generation, Lorem Ipsum text is sampled using the python-lorem package1. The sequence length is uniformly sampled between 100 and 300 tokens. Empirically, we append a short boundary instruction to the end of each perturbation sequence: “\nPlease reason step by step, and put your final answer within \boxed.” This simple design effectively reduces cases in which the perturbation negatively interferes with the model and causes it to generate corrupted outputs, like random symbols and characters.

###### 6.2 Main Results

Model & Method MATH-500 GSM8K AMC AIME24 AIME25 Avg. Qwen3-1.7B-Base 63.40 76.92 26.87 5.33 2.00 34.90

+ GRPO 64.20 82.71 27.61 6.15 4.47 37.03

+ Resample

w/ Naive Prompt 67.00 82.18 28.36 8.70 4.58 38.16 w/ LOPE (w/o Training Signal Shaping) 68.00 83.55 33.58 7.97 5.83 39.79 w/ LOPE (w/ Training Signal Shaping) 68.80 82.94 32.84 8.80 5.73 39.82

Qwen3-4B-Base 65.80 82.71 32.84 9.38 7.24 39.59 + GRPO 77.80 91.74 47.76 16.41 13.12 49.37

+ Resample

w/ Naive Prompt 79.80 92.87 45.52 14.90 11.67 48.95 w/ LOPE (w/o Training Signal Shaping) 85.40 92.95 52.99 19.01 13.85 52.84 w/ LOPE (w/ Training Signal Shaping) 82.60 92.95 58.21 19.90 16.27 53.99

Qwen2.5-Math-7B 52.80 65.50 35.40 12.90 7.90 34.90 + GRPO 78.00 85.06 47.76 17.66 9.90 47.68

+ Resample

w/ Naive Prompt 78.20 83.02 50.00 17.19 9.17 47.52 w/ LOPE (w/o Training Signal Shaping) 77.40 86.35 47.01 15.31 10.52 47.32 w/ LOPE (w/ Training Signal Shaping) 81.80 90.30 61.19 19.58 16.51 53.88

Table 1: Performance comparison across different model scales. LOPE with training signal shaping consistently outperforms GRPO and resampling with the naive prompt baselines.

- Table 1 presents the main evaluation results of LOPE compared to standard GRPO and the naive resampling baseline. LOPE improves the reasoning capabilities of the base models, yielding the highest average performance across the evaluated benchmarks. On the Qwen31.7B-Base model, LOPE achieves an average score of 39.79, outperforming standard GRPO by 2.76 points, and surpassing the Naive Prompt resampling baseline by 1.63 points. This

1https://pypi.org/project/python-lorem/

demonstrates that expanding exploration via prompt-space perturbation is more effective than simply allocating more compute to do naive resampling for logit-space perturbation. Similarly, on the Qwen3-4B-Base model, LOPE outperforms standard GRPO by 3.47 points. Another finding is that Naive Prompt resampling actually degrades performance compared to standard GRPO, probably due to policy drift without KL regularization. However, LOPE discovers orthogonal, high-quality reasoning trajectories in resampling, therefore injecting highly variant responses that act as implicit regularizers. On the Qwen2.5-Math-7B model, naive resampling and LOPE perform similarly to standard GRPO, while LOPE with training signal shaping significantly outperforms GRPO by 6.20 points. This suggests that although LOPE increases the resample success rate, the gain is weakened by optimization inefficiency under off-policy update and biased advantage estimation. Training signal shaping alleviates this issue by amplifying learning signals on rare successful responses and informative low-frequency tokens.

###### 6.3 Successful Training-Time Exploration

In Section 3, we show that LOPE improves resample accuracy. A natural question is whether this advantage persists throughout the RL training process. To investigate this, we track the resample accuracy of Qwen3-1.7B-Base during training, including both the question-level and response-level success rates, corresponding to pass@G′ and Mean@G′ respectively, where G′ = 24 is the resampling size.

- 0.0

- 0.1

- 0.2

- 0.3

- 0.4

- 0.5

- 0.6

SuccessRate

Resample Question-Level Success Rate (Pass@24)

Naive Prompt

Naive Prompt (High Temperature)

LoPE

LoPE+Training Signal Shaping

0 25 50 75 100 125 150 175 200

Step

0.00

0.02

0.04

0.06

Accuracy

Resample Response-Level Accuracy (Mean@24)

Naive Prompt

Naive Prompt (High Temperature)

LoPE

LoPE+Training Signal Shaping

Figure 4: Resample success rate and accuracy of Qwen3-1.7B-Base during training.

We compare LOPE against two baselines: Naive prompt resampling and Naive prompt resampling with a high temperature of 1.2. The results are shown in Figure 4. LOPE and LOPE with training signal shaping are similar in both subsets, indicating the training signal shaping does not alter rollout behavior. Interestingly, while LOPE and Naive prompt resampling achieve comparable performance in terms of response-level accuracy, LOPE consistently achieves a significantly higher question-level success rate. This indicates that the randomness introduced by LOPE enables the model to explore correct solutions across a broader set of questions, preventing the model from only optimizing on a narrow subset of training questions. Qwen3-4B-Base and Qwen2.5-Math-7B have similar observations, whose training-time accuracies are shown in Appendix B.

Takeaway: LOPE maintains a higher resample success rate throughout the GRPO training process and ensures effective training signals on a broader set of questions.

- 7 Analysis: What Makes a Good Prompt Space Perturbation?

0 25 50 75 100 125 150 175 200

Step

In this section, we systematically investigate why Lorem Ipsum succeeds as a prompt space perturbation. We first propose a broad range of methods to generate random noise perturbations. Next, we conduct a comprehensive analysis of the properties of these perturbations. Finally, we perform RL training with these methods and conclude the underlying principles that define an effective prompt space perturbation.

###### 7.1 Exploring Alternative Prompt Space Perturbation

We first introduce several methods for generating prompt space perturbations, primarily focusing on randomly generated noise:

- • Random Fake English. This setting uses the Faker package2 to generate fake English sentences, where each word is independently sampled from a commonly used English word pool3. In addition, it applies initial capitalization and inserts random punctuation to mimic natural sentence structures.
- • Random ASCII. This setting uniformly samples printable ASCII characters to form random sequences.
- • Random Tokens. This setting uniformly samples tokens from the model’s vocabulary to form random sequences. Special tokens are excluded to prevent functional errors.
- • English Unigram Model. In this setting, we follow the same random generation procedure as LOPE, but replace the candidate word pool with the top-50 most frequent words extracted from the English subset of the C4 multilingual corpus (Raffel et al., 2020)4, which is widely used for pretraining. Non-English text is filtered out using langid5 and fastText (Joulin et al., 2016). Words are sampled uniformly from the word pool.
- • Latin Unigram Model. This setting is similar to English Unigram Model, but uses the top-50 most frequent Latin words as the candidate word pool.
- • Latin 3-Gram Model. In this setting, we explore locally coherent random sequences generated by a 3-gram language model. Concretely, we train a 3-gram language model using markovify6 on the Latin subset of the C4 corpus, and use this model for random sequence generation during training.
- • Filtered Latin Natural Language. In this setting, we use natural Latin text as the prompt space perturbation. Specifically, we adopt the Latin subset of the C4 corpus. We filter out non-Latin sentences using langid7 and fastText (Joulin et al., 2016), and additionally remove sentences containing the canonical Lorem Ipsum incipit (“lorem ipsum dolor sit amet, consectetur adipiscing elit”). After deduplication and retaining only sequences within the 100–300 token range, we obtain a corpus of approximately 65K sequences.

The length of all prompt space perturbations is uniformly sampled between 100 and 300 tokens to match the main experiment.

###### 7.2 How LLMs Perceive Prompt-Space Noise

To understand how prompt space perturbations influence the model’s generation process, we conduct a progressive three-step analysis. Specifically, we systematically investigate (i) the intrinsic properties of the perturbation text itself and (ii) its impact on the model’s understanding of the question.

How Does the LLM Understand the Random Perturbation Sequences? We begin by analyzing the intrinsic properties of different perturbation sequences, focusing on how well they align with the language model’s distribution. For each method, we generate 500 perturbation sequences of 200 tokens each and compute their perplexity under Qwen3-1.7BBase. As a reference for in-distribution natural language, we also report the perplexity of the question text from the 500-question evaluation subset.

The resulting distributions are presented in Figure 5. We characterize each method’s sequences by their mean perplexity (indicating how far they deviate from natural language) and their standard deviation (indicating the consistency of the perturbation strength across samples).

- 2https://pypi.org/project/Faker/.
- 3https://github.com/joke2k/faker/blob/master/faker/providers/lorem/en US/ init .py.

- 4https://huggingface.co/datasets/allenai/c4/viewer/la
- 5https://pypi.org/project/langid/
- 6https://pypi.org/project/markovify/
- 7https://pypi.org/project/langid/

|Near-natural perturbation| |
|---|---|
|0 5 10 15 20 25 30<br><br>0.00<br><br>0.05<br><br>0.10<br><br>0.15<br><br>0.20<br><br>0.25<br><br>density<br><br>Question Text<br><br>Mean = 4.822 Std = 3.577<br><br>20 25 30 35<br><br>0.000<br><br>0.025<br><br>0.050<br><br>0.075<br><br>0.100<br><br>0.125<br><br>0.150<br><br>Lorem Ipsum<br><br>Mean = 25.12 Std = 2.843<br><br>0 50 100 150 200 250<br><br>0.000<br><br>0.005<br><br>0.010<br><br>0.015<br><br>0.020<br><br>Latin Natural Language<br><br>Mean = 46.09 Std = 42.63<br><br>30 40 50 60 70 80<br><br>0.00<br><br>0.01<br><br>0.02<br><br>0.03<br><br>0.04<br><br>0.05<br><br>0.06<br><br>density<br><br>Latin Unigram Model<br><br>Mean = 51.32 Std = 8.297<br><br>80 90 100<br><br>0.00<br><br>0.02<br><br>0.04<br><br>0.06<br><br>0.08<br><br>0.10<br><br>English Unigram Model<br><br>Mean = 85.30 Std = 5.238<br><br>25 50 75 100 125 150 175<br><br>0.000<br><br>0.005<br><br>0.010<br><br>0.015<br><br>Latin 3-Gram Model<br><br>Mean = 91.45 Std = 30.03| |

densitydensity

|Moderately out-of-distribution perturbation| |Severely out-of-distribution perturbation| |
|---|---|---|---|
|300 400 500 600 700 800<br><br>0.000<br><br>0.001<br><br>0.002<br><br>0.003<br><br>0.004<br><br>0.005<br><br>0.006<br><br>density<br><br>Random ASCII<br><br>Mean = 492.93 Std = 83.01<br><br>2000 2200 2400 2600 2800 3000<br><br>0.0000<br><br>0.0005<br><br>0.0010<br><br>0.0015<br><br>0.0020<br><br>0.0025<br><br>Random Fake English<br><br>Mean = 2429.9 Std = 171.95| |2e5 3e5 4e5 5e5 6e5 7e5 8e5<br><br>0<br><br>1<br><br>2<br><br>3<br><br>4<br><br><br>1e 6<br><br>Random Token<br><br>Mean = 4.6e+05 Std = 1.1e+05| |

nsity

- Figure 5: Perplexity distributions of randomly generated sequences from each perturbation type, with the mean and standard deviation reported in each subplot. “Question Text” from the 500-question evaluation set serves as an English natural language reference. Perturbations are grouped by their perplexity computed on Qwen3-1.7B-Base. Perturbations in the first two rows show a near-natural language property, yielding better final performance, while those in the last row are out of the model distribution, resulting in worse performance as shown in Table 2.

The mean perplexities span several orders of magnitude, naturally partitioning the methods into three regimes. (i) Near-natural perturbations (mean below 100): Lorem Ipsum (25.12), Filtered Latin Natural Language (46.09), Latin Unigram Model (51.32), English Unigram Model (85.30), and Latin 3-Gram Model (91.45) all sit within roughly an order of magnitude of the Question Text reference (4.82), suggesting they behave as structurally plausible noise from the model’s perspective. (ii) Moderately out-of-distribution perturbations (mean in the hundreds to low thousands): Random ASCII (492.93) drifts further from the language manifold, while Random Fake English reaches a mean of 2429.9. (iii) Severely out-of-distribution perturbations: Random Token explodes to a mean perplexity of 4.6 × 105, reflecting a complete collapse of linguistic structure.

Beyond the mean perplexity, the dispersion of these distributions also varies substantially. Lorem Ipsum exhibits both the lowest mean value among the synthetic perturbations and a tightly concentrated distribution (std 2.84), indicating that every sampled sequence imposes a consistent, controlled distributional shift. In contrast, other low-mean perturbations exhibit a much wider spread. Most notably, Filtered Latin Natural Language presents a long-tailed distribution (std 42.63) that extends beyond a perplexity of 200, despite having a comparable mean. Such uneven perturbation strength would inevitably lead to high within-batch variance during RL training. We therefore consider a filtered variant of Filtered Latin Natural Language with sequences having perplexity between 20 and 30.

Takeaway: Random sequences generated by sampling from a limited word pool have lower perplexity, showing a near-natural property.

Impact of Random Perturbations on Question Comprehension. The previous analysis demonstrates that random sequences introduce varying degrees of perturbation into the prompt space, implicitly altering the model’s policy. We next investigate the extent to

which these perturbations disrupt the model’s comprehension of the input, particularly the question text itself.

Question Text

Question Text

English Unigram Model

40

Lorem Ipsum

Latin Unigram Model

Lorem Ipsum

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

Random Fake English

Latin 3-Gram Model

Random Fake English

Random ASCII

Latin Natural Language

30

Random ASCII

Random Token

Random Token

0.8

20

English Unigram Model

Latin Unigram Model

density

| | | | | | | | |
|---|---|---|---|---|---|---|---|

10

Latin 3-Gram Model

0.6

| | |
|---|---|
| | |
| | |

Latin Natural Language

0

| | |
|---|---|
| | |
| | |

0.4

10

0.2

20

30

0.0

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |
| | | |

0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0

Mean Entropy of Question Text

40 20 0 20 40

(a) Entropy distribution of 500 question prompts. Most perturbations preserve a distribution close to the Question Text. The distribution of Random Token is clearly right-shifted, indicating a corrupted understanding of the question.

(b) Question representation visualization via tSNE under different perturbations. Each color corresponds to one question (8 samples per perturbation). Random Token perturbations drift far from the original meaning.

- Figure 6: The influence of various prompt space perturbations on question comprehension.

To achieve this, we analyze both the token-level entropy and the sentence-level representations of the questions. Figure 6a presents the entropy distribution across 500 question prompts, where each prompt is characterized by the average entropy of its constituent tokens. The entropy distributions of most perturbations largely overlap with the Question Text, indicating that they alter LLM’s question comprehension only slightly. In contrast, Random Tokens exhibit a significant right-shifted deviation.

Figure 6b illustrates the impact of various perturbations on the semantic representations of the question text. We select 10 questions that are successfully solved by all methods within 8 rollouts and compute their sentence representations, defined as the mean pooling of the final-layer hidden states across all tokens in the question. We then apply t-SNE for visualization. Points of the same color correspond to the identical underlying question, while different marker shapes indicate distinct prompt perturbation types.

We observe that the representations under near-natural or moderately out-of-distribution perturbations cluster tightly around those of the Question Text, implying that the model maintains a consistent semantic understanding. Conversely, Random Token perturbations consistently drift far from these clusters, suggesting that severe noise significantly distorts LLM’s interpretation of the input question.

Collectively, these findings demonstrate that excessively high-perplexity perturbations can corrupt the model’s understanding of the input question, thereby hindering its ability to discover correct solutions during the RL training process.

Takeaway: Near-natural perturbation (low perplexity) successfully shifts the model’s generation distribution without corrupting its semantic understanding of the core question. Conversely, excessively noisy perturbation destroys the input representation.

###### 7.3 The Recipe for an Effective Perturbation

We conduct experiments with these prompt space perturbation methods. For Filtered Latin Natural Language, we observe in Figure 5 that the corpus contains many sequences with relatively low perplexity, which aligns with the characteristics of Lorem Ipsum. Therefore, we further filter the corpus to retain sequences with a perplexity between 20 and 30, resulting in approximately 38K sequences.

As a baseline for comparison, we select resampling with the naive prompt. Additionally, we experiment with resampling with a higher temperature, which encourages broader exploration in rollouts.

Model & Method MATH-500 GSM8K AMC AIME24 AIME25 Avg. Qwen3-1.7B-Base 63.40 76.92 26.87 5.33 2.00 34.90

+ GRPO 64.20 82.71 27.61 6.15 4.47 37.03

+ Resample w/o perturbation

w/ Naive Prompt 67.00 82.18 28.36 8.70 4.58 38.16 w/ Naive Prompt (Temp=1.2) 64.40 82.87 31.34 8.65 4.48 38.35

+ Resample w/ perturbation + Training Signal Shaping

w/ LOPE 68.80 82.94 32.84 8.80 5.73 39.82 w/ Random Fake English 65.80 81.96 32.09 7.50 5.42 38.55 w/ Random ASCII 66.20 82.94 28.36 8.12 5.32 38.19 w/ Random Token 64.20 81.50 29.85 8.08 4.63 37.65 w/ Filtered Latin Natural Language 68.80 82.71 32.84 9.32 5.57 39.85 w/ Latin Unigram Model 69.40 83.32 32.09 7.19 6.35 39.67 w/ Latin 3-Gram Model 68.80 81.88 29.85 7.92 5.93 38.88 w/ English Unigram Model 67.00 83.32 28.36 8.49 5.42 38.52

- Table 2: Comparison of different prompt space perturbation methods. The three methods with the smallest perplexity value (LOPE, Filtered Latin Natural Language, and Latin Unigram Model) achieve the best performance. Random Token with the highest perplexity even harms the training. This suggests that mild perturbations are sufficient to drive improvement while avoiding detrimental effects of excessive noise.

The experiment results are presented in Table 2, showing that performance varies across different methods. While not all candidates match the superior results of LOPE, LOPE is not the only effective approach. Specifically, Filtered Latin Natural Language and Latin Unigram Model both achieve average scores exceeding 39.6, suggesting that the success of prompt space perturbation is not isolated to the Lorem Ipsum format. As illustrated in Figure 5, these high-performing methods share two defining characteristics: they consist of Latin words and exhibit the lowest perplexity among all evaluated perturbations.

We conjecture the reason for their success as follows. Such perturbations provide sufficient signals to drive the rollout policy away from the naive policy, encouraging model exploration. Meanwhile, their relatively low perplexity avoids introducing excessive noise, thereby preserving the quality of the rollout responses.

Furthermore, we observe that English Unigram Model underperforms Latin 3-Gram Model, despite the former possessing a slightly lower average perplexity. This indicates that English-based perturbations are more prone to interfering with the model’s original English reasoning context, which subsequently hinders performance.

Takeaway: The most effective prompt space perturbation share these characteristics: (i) pseudo-Latin vocabularies to prevent interference with the English reasoning context, and (ii) maintaining low perplexity to ensure high-quality rollouts.

##### 8 Related Work

Zero-Advantage Recovery in RLVR. GRPO-style reinforcement learning (Shao et al., 2024; Guo et al., 2025; Yu et al., 2025) improves LLM reasoning but suffers from zero-reward signals on hard prompts. Recent efforts focus on improving rollout efficiency and recovering useful training signals, such as adaptive budget allocation, targeted exploration, scaffolded hints, and off-policy guidance (Zhang et al., 2026b; Le et al., 2026; Bamba et al., 2025; Zhang et al., 2026a; Yan et al., 2025; Zhao et al., 2026). Another line of work explores modifying prompt patterns or prefixes during training, such as prompt selection, augmentation, and prefix-level guidance (Wang et al., 2026; Lu et al., 2026; Mundada et al., 2026).

Context-level Perturbation. While in-context learning with logit-level arithmetics explicitly alters output distribution (Huang et al., 2026a), a group of work shows that prompt context can also implicitly influence the model’s generation behavior, where small changes can induce substantial shifts (Xie et al., 2022; von Oswald et al., 2023). Empirical studies on prompting further demonstrate that reasoning performance is highly sensitive to prompt

format, demonstrations, and instruction structure (Wei et al., 2023; Wang et al., 2023; Huang et al., 2025; Zhou et al., 2023; Yang et al., 2024b; Huang et al., 2026b). More recent work highlights that adding synthetic or meaningless tokens can shift model activations and alter reasoning behavior (Shi et al., 2025; Gan & Isola, 2026).

##### 9 Conclusion

This paper proposes Lorem Perturbation for Exploration (LOPE), a simple yet effective prompt space perturbation method that prepends a randomly generated Lorem Ipsum sequence to the naive prompt. This introduces a controllable perturbation that enables LLM to explore alternative reasoning trajectories and succeed on previously failed questions. Moreover, resampling with LoPE during GRPO training significantly improves questionlevel success rates, enhances data utilization, and yields consistent performance gains.

Furthermore, a systematic comparison of various prompt space perturbation methods reveals that the success of LOPE is not an isolated case. Specifically, our analysis indicates that effective perturbations share two crucial characteristics: they are Latin-based and have relatively low perplexity. These findings highlight the importance of controlled perturbation in improving exploration for LLM reasoning.

##### Ethics Statement

This work explores prompt-space perturbations using randomly generated sequences. As these sequences are generated automatically, we cannot guarantee that they are entirely free of potentially toxic, biased, or inappropriate words or expressions.

In addition, we observe that when the perturbation is excessively strong, the model may produce incoherent or nonsensical outputs that, in some cases, include undesirable or harmful content. This highlights a limitation of our approach: increased exploration via perturbation may come at the cost of reduced controllability over model outputs.

To mitigate these risks, our method emphasizes controlled perturbation, and our experiments suggest that moderate, language-like perturbations are less likely to negatively affect model behavior. Nevertheless, ensuring the safety and robustness of such perturbations remains an open challenge.

In future work, we plan to systematically study how different types and strengths of perturbations influence model behavior, with the goal of minimizing the risk of generating toxic or harmful content while preserving the benefits of improved exploration.

##### References

Udbhav Bamba, Minghao Fang, Yifan Yu, Haizhong Zheng, and Fan Lai. Xrpo: Pushing the limits of grpo with targeted exploration and exploitation, 2025. URL https://arxiv.org/ abs/2510.06672.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers, 2023. URL https://arxiv.org/abs/2212.10559.

Yulu Gan and Phillip Isola. Neural thickets: Diverse task experts are dense around pretrained weights, 2026. URL https://arxiv.org/abs/2603.12228.

Adrian Goldwaser, Michael Munn, Javier Gonzalvo, and Benoit Dherin. Equivalence of context and parameter updates in modern transformer blocks, 2025. URL https: //arxiv.org/abs/2511.17864.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645 (8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. Efficient test-time scaling via self-calibration. arXiv preprint arXiv:2503.00031, 2025.

Chengsong Huang, Langlin Huang, and Jiaxin Huang. Divide, reweight, and conquer: A logit arithmetic approach for in-context learning. In Vera Demberg, Kentaro Inui, and Llu´ıs Marquez (eds.), Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 229–249, Rabat, Morocco, March 2026a. Association for Computational Linguistics. ISBN 979-8-89176-380-7. doi: 10.18653/ v1/2026.eacl-long.11. URL https://aclanthology.org/2026.eacl-long.11/.

Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. CaTS: Calibrated test-time scaling for efficient LLM reasoning. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id= jrSc4RJXy1.

Jarry Shaw. Lorem ipsum generator, 2024. URL https://github.com/JarryShaw/lorem. Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, H´erve J´egou, and

Tomas Mikolov. Fasttext.zip: Compressing text classification models. arXiv preprint arXiv:1612.03651, 2016.

Thanh-Long V. Le, Myeongho Jeon, Kim Vu, Viet Lai, and Eunho Yang. No prompt left behind: Exploiting zero-variance prompts in llm reinforcement learning via entropyguided advantage shaping, 2026. URL https://arxiv.org/abs/2509.21880.

Ziniu Li, Congliang Chen, Tianyun Yang, Tian Ding, Ruoyu Sun, Ge Zhang, Wenhao Huang, and Zhi-Quan Luo. Knapsack rl: Unlocking exploration of llms via optimizing budget allocation, 2025. URL https://arxiv.org/abs/2509.25849.

Mengqi Liao, Xiangyu Xi, Ruinian Chen, Jia Leng, Yangen Hu, Ke Zeng, Shuai Liu, and Huaiyu Wan. Enhancing efficiency and exploration in reinforcement learning for llms,

2025. URL https://arxiv.org/abs/2505.18573.

Wenquan Lu, Hai Huang, and Randall Balestriero. Prompt augmentation scales up grpo training on mathematical reasoning, 2026. URL https://arxiv.org/abs/2602.03190.

Gagan Mundada, Zihan Huang, Rohan Surana, Sheldon Yu, Jennifer Yuntong Zhang, Xintong Li, Tong Yu, Lina Yao, Jingbo Shang, Julian McAuley, and Junda Wu. Ws-grpo: Weakly-supervised group-relative policy optimization for rollout-efficient reasoning, 2026. URL https://arxiv.org/abs/2602.17025.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300.

Zeru Shi, Yingjia Wan, Zhenting Wang, Qifan Wang, Fan Yang, Elisa Kreiss, and Ruixiang Tang. Meaningless tokens, meaningful gains: How activation shifts enhance llm reasoning,

2025. URL https://arxiv.org/abs/2510.01032. ModelScope Team. EvalScope: Evaluation framework for large models, 2024. URL https:

//github.com/modelscope/evalscope. Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388. Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ao Sacramento, Alexander

Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent, 2023. URL https://arxiv.org/abs/2212.07677.

Hanbin Wang, Jingwei Song, Jinpeng Li, Fei Mi, and Lifeng Shang. Group pattern selection optimization: Let lrms pick the right pattern for reasoning, 2026. URL https://arxiv. org/abs/2601.07238.

Jiakang Wang, Runze Liu, Lei Lin, Wenping Hu, Xiu Li, Fuzheng Zhang, Guorui Zhou, and Kun Gai. Aspo: Asymmetric importance sampling policy optimization, 2025. URL https://arxiv.org/abs/2510.06062.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023. URL https://arxiv.org/abs/2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/2201.11903.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference, 2022. URL https://arxiv.org/abs/ 2111.02080.

Wei Xiong, Chenlu Ye, Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian, Nan Jiang, and Tong Zhang. Reinforce-ada: An adaptive sampling framework under non-linear rl objectives, 2025. URL https://arxiv.org/abs/2510.04996.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance, 2025. URL https://arxiv.org/ abs/2504.14945.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement, 2024a. URL https://arxiv.org/abs/ 2409.12122.

Yuyi Yang, Charles Alba, Chenyu Wang, Xi Wang, Jami Anderson, and Ruopeng An. Gpt models can perform thematic analysis in public health studies, akin to qualitative researchers. Journal of Social Computing, 5(4):293–312, 2024b.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale,

- 2025. URL https://arxiv.org/abs/2503.14476.

Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, and Jiaya Jia. Scaf-grpo: Scaffolded group relative policy optimization for enhancing llm reasoning,

- 2026a. URL https://arxiv.org/abs/2510.19807.

Zhi Zhang, Zhen Han, Costas Mavromatis, Qi Zhu, Yunyi Zhang, Sheng Guan, Dingmin Wang, Xiong Zhou, Shuai Wang, Soji Adeshina, Vassilis Ioannidis, and Huzefa Rangwala. Train less, learn more: Adaptive efficient rollout optimization for group-based reinforcement learning, 2026b. URL https://arxiv.org/abs/2602.14338.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models, 2026. URL https://arxiv.org/abs/2601.18734.

Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong, and Zhang Richong. Easyr1: An efficient, scalable, multi-modality rl training framework. https://github.com/hiyouga/EasyR1, 2025.

Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models, 2023. URL https://arxiv.org/ abs/2205.10625.

##### A Example of Lorem Ipsum Prompt

Lorem Ipsum Prompt System in laboris velit ex est ex est exercitation sint ex enim ut incididunt tempor cupidatat nisi occaecat deserunt laboris quis voluptate qui consequat proident minim pariatur velit est aliquip do labore ut dolor et ullamco proident in non culpa est amet ipsum officia velit exercitation aute consectetur aliquip ullamco laboris laborum deserunt quis enim consectetur ea veniam officia commodo cupidatat culpa fugiat consequat eu enim mollit commodo enim eiusmod deserunt laborum ullamco minim irure consectetur quis ut nisi cillum adipiscing proident culpa id cupidatat ad amet sunt qui adipiscing mollit proident dolor pariatur minim eu quis ad excepteur nulla reprehenderit ea enim Please reason step by step, and put your final answer within \boxed{}. User Given abc ̸= 0, and a + b + c = 0. Then the value of the algebraic expression

a2 bc

b2 ac

c2 ab

+

+

is:

- (A) 3
- (B) 2
- (C) 1
- (D) 0

You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.

Assistant

##### B Training-Time Resample Accuracy for Qwen3-4B-Base and Qwen2.5-Math-7B

We present the training-time resample success rate and accuracy of Qwen3-4B-Base and Qwen2.5-Math-7B in Figure 7 and Figure 8, where LOPE consistently achieves a significantly higher question-level success rate.

###### Resample Question-Level Success Rate (Pass@24)

###### Resample Response-Level Accuracy (Mean@24)

0.6

Naive Prompt

Naive Prompt

0.08

LoPE

LoPE

0.5

LoPE+Training Signal Shaping

LoPE+Training Signal Shaping

SuccessRate

0.4

0.06

Accuracy

0.3

0.04

0.2

0.02

0.1

0.0

0.00

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Step

Step

###### Figure 7: Resample success rate and accuracy during Qwen3-4B-Base training.

###### Resample Question-Level Success Rate (Pass@24)

###### Resample Response-Level Accuracy (Mean@24)

0.08

Naive Prompt

Naive Prompt

0.5

LoPE

LoPE

LoPE+Training Signal Shaping

LoPE+Training Signal Shaping

0.4

0.06

SuccessRate

Accuracy

0.3

0.04

0.2

0.02

0.1

0.0

0.00

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

Step

Step

Figure 8: Resample success rate and accuracy during Qwen2.5-MATH-7B training.

##### C The Effectiveness of Training Signal Shaping on Training

###### C.1 Policy Shaping

Gradient Analysis of the Resampling Objective. To better understand the role of the reward shaping function f(·) in our setting, we follow the derivation in Yan et al. (2025) to analyze the gradient of the resampling part of the training objective in Eq. (7). Notably,

unlike their formulation, we relax the assumption that πθold = 1. The objective is given by:

|oi|

Ns

1 |oi|

#### ∑

#### ∑

Jresample(θ) = Eq,{oi}∼πθ

old

t=1

i=1

f(ρi,t) Aˆi , (8)

where ρi,t = πθ/πθold is the importance sampling ratio. For brevity, we omit the conditioning variables in the following derivation and denote πθ := πθ(oi,t | q, oi,<t) and πθold := πθold(oi,t | δ ⊕ p, q, oi,<t).

Gradient with respect to θ. Applying the logarithmic derivative ∇θπθ = πθ∇θ log πθ, the derivative of f(ρi,t) w.r.t. θ is f′(ρi,t) ρi,t ∇θ log πθ. Thus, the gradient of the resampling objective is:

   . (9)

###   ∑

1 |oi|

ρi,t ∇θ log πθ · Aˆi

f′(ρi,t) shaped weight

∇θJresample(θ) = Eq,{oi}∼πθ

old

i,t

The term f′(ρi,t) acts as a weighting function of the gradient. When f(x) = x, we have f′(x) = 1, degenerating to the vanilla importance sampling gradient.

Per-token Logit Gradient and Upper Bound. To analyze how the resampling objective updates the model’s predictive distribution at the token level, we project the parameter gradient ∇θ log πθ(oi,t) onto an individual output logit Mθ(τ), where τ ranges over the action space at position t. Since the policy is parameterized as a softmax over logits,

exp Mθ(oi,t) ∑τ′ exp Mθ(τ′)

, (10)

πθ(oi,t) =

taking logarithms gives log πθ(oi,t) = Mθ(oi,t) − log ∑τ′ exp Mθ(τ′) , and differentiating with respect to Mθ(τ) yields the standard softmax Jacobian identity:

∂ log πθ(oi,t) ∂Mθ(τ)

= 1[τ = oi,t] − πθ(τ). (11)

(1 ) ( +

S( ) =

)2 , = 0.1

G( ) = (1 )

C( ) = (1 )

[

1 + ], = 0.2, A > 0

old

old

old

old

old

= 0.6

= 0.6

= 0.6

0.4

0.4

old

old

old

- = 0.8

old

- = 1.0

- = 0.8

old

- = 1.0

- = 0.8

old

- = 1.0

0.20

old

old

old

0.3

0.3

0.15

Weights

Weights

Weights

0.2

0.2

0.10

0.1

0.1

0.05

0.0

0.0

0.00

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

- Figure 9: Per-token gradient weight under three formulations, plotted as a function of πθ. Left: Vanilla gradient bound G(πθ). Middle: GRPO-clipped gradient C(πθ) under positive advantage (Aˆ > 0). Gradients are truncated to zero when ρi,t > 1 + ϵ. Right: Policy-shaped

gradient S(πθ). The peak is relocated to the low-probability regime, with bounded peak value 1/4.

By taking the derivative of Eq. (9) with respect to Mθ(τ) and applying Eq. (11), we obtain the per-token logit gradient and its upper bound:

⇒

∂Jresample(θ) ∂Mθ(τ)

###### = Eq,{oi}∼πθ

old

∂Jresample(θ) ∂Mθ(oi,t) ≤ Eq,{oi}∼πθ

old

f′(ρi,t) ρi,t (1[τ = oi,t] − πθ(τ)) · Aˆi

f′(ρi,t) ρi,t (1 − πθ) · A ˆi ,

(12)

where the upper bound corresponds to the identity case τ = oi,t, yielding |1[τ = oi,t] − πθ(τ)| = 1 − πθ. This case captures the dominant gradient signal that elevates the logit of the resampled token itself. From Eq. (12), under the vanilla setting f(x) = x, the gradient

scale is bounded by ρi,t(1 − πθ) = πθ(1 − πθ)/πθold. To visualize this bound and motivate our shaping choice, Figure 9 plots the per-token gradient weight as a function of πθ across several πθold values, under the situation of a positive advantage (Aˆ > 0). The left panel illustrates the unclipped vanilla gradient bound G(πθ) = πθ(1 − πθ)/πθold, while the middle panel displays its GRPO-clipped counterpart C(πθ). In C(πθ), the gradient is additionally truncated to zero when ρi,t > 1 + ϵ, which corresponds to the active branch in the GRPO objective for Aˆ > 0. As illustrated in Figure 9, both the vanilla and GRPO-clipped gradient bounds suffer from two fundamental limitations:

- (i) Vanishing gradient at low πθ. The bound decays to zero as πθ → 0, yielding weak learning signals for unfamiliar tokens where the current policy is highly uncertain. Crucially,

the distribution discrepancy between the resampled trajectories {oi} ∼ πθold and the training policy πθ systematically drives πθ to small values, suppressing gradients exactly where off-policy guidance is most needed. GRPO clipping fails to address this issue, as it only operates on the ρi,t > 1 + ϵ region.

- (ii) Inappropriate handling of low πθold tokens. The peak value of the vanilla bound is

1/(4πθold), which diverges as πθold → 0. This induces excessive gradient magnitudes on rare tokens of the resampling policy, causing training instability. While GRPO clipping

avoids this by hard-truncating the gradients to zero, it entirely sacrifices the learning signals associated with these tokens.

To overcome these limitations, we adopt the policy shaping function f(x) = x/(x + γ) proposed by Yan et al. (2025) to reshape the gradients.

Specialization to f(x) = x/(x + γ). The derivative of the shaping function is f′(x) = γ/(x + γ)2. Substituting this into the bound in Eq. (12) yields:

∂Jresample(θ) ∂Mθ(oi,t) ≤ E

γ πθold πθ(1 − πθ)

(πθ + γ πθold)2 · |Aˆi| . (13)

Treating the integrand of Eq. (13) as a function of πθ for a fixed πθold, it attains its maximum at:

1

γ πθold 1 + 2γ πθold

πθ⋆ =

, with a peak value of

. (14)

4 1 + γ πθold

Equation (14) formalizes two complementary properties of the shaping function, which are visually corroborated in Figure 9 (right).

- (i) Low-probability emphasis. The peak location shifts from πθ = 1/2 (the vanilla case) into

the low-probability regime at πθ⋆ ≈γ πθold. Consequently, the shaping function amplifies the learning signal for the unfamiliar yet highly rewarded tokens that off-policy resampling

aims to introduce.

- (ii) Bounded and stable peak value. The peak value 1/[4(1 + γ πθold)] is strictly bounded by 1/4 for all πθold ∈[0, 1] and remains stable across different πθ and πθold values. In contrast, the vanilla bound scales as 1/(4πθold) and can grow unboundedly as πθold →0.

By reshaping the learning signal toward under-confident tokens and stabilizing the gradient magnitude across the full spectrum of probabilities, policy shaping effectively reweights the parameter updates. It assigns greater importance to low-probability yet effective actions while gracefully attenuating updates for tokens the model has already mastered.

###### C.2 Advantage Shaping

[Figure 6]

- Figure 10: Comparison of advantages for positive responses before and after advantage shaping. Left: The absolute advantage values of the vanilla advantage (A+) and shaped advantage (Aˆ+), as a function of the number of correct responses c. Right: The amplification factor (Aˆ+/A+) achieved by applying advantage shaping.

Quantitative Effect of Advantage Shaping. Since the advantage term acts as a constant multiplier in the training objective (Eq. 7) and the gradient (Eq. 9), its effect is independent of other variables. Therefore, we quantitatively analyze the advantage term in isolation.

We follow the notations in Sections 4 and 5: the rollout size for the first-time sampling and the resampling is G and G′, where the number of correct responses is 0 and c, respectively. The gradient update is performed on a regrouped set of G responses, consisting of Ns = min(c, G − 1) correct resampled responses and G − Ns incorrect first-time sampling responses. The vanilla LOPE calculates the advantage, denoted as A, solely within the G selected rollouts. In contrast, advantage shaping computes the advantage, denoted as Aˆ, over the entire pool of G + G′ rollouts.

In these two scenarios, the means and standard deviations of the rewards are µ = Ns/G, σ =

(G − Ns)Ns/G and µˆ = c/(G + G′), σˆ = (G + G′ − c)c/(G + G′), respectively. Substituting these into Eq. 3 and Eq. 6, we obtain the advantages for the positive samples:

A+ =

G − Ns Ns

, Aˆ+ =

(G + G′) − c c

. (15)

Figure 10 visualizes these quantities and their ratio using our actual sampling budget (G = 8, G′ = 24) across the regime of c ∈ [1, G′]. Since resampling is triggered exclusively on hard questions where the initial G samples all fail, c is typically smaller than G in practice (c < G). With in this range, the amplification factor Aˆ+/A+ ranges from 2.10× at c = 1 to a peak of 5.00× at c = G − 1 = 7.

According to Eq. (9), the amplification of advantage directly applies to the gradient weight. Therefore, advantage shaping effectively assigns a larger training weight to the rare correct trajectories that drive learning on hard questions.

