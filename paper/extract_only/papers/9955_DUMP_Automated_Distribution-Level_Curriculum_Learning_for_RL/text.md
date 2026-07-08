# arXiv:2504.09710v3[cs.LG]11Oct2025

## DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training

#### Zhenting Wang1 Guofeng Cui1 Yu-Jhe Li 2 Kun Wan2∗ Wentian Zhao2∗

1Rutgers University 2Adobe Inc.

### Abstract

Recent advances in reinforcement learning (RL)-based post-training have led to notable improvements in large language models (LLMs), particularly in enhancing their reasoning capabilities to handle complex tasks. However, most existing methods treat the training data as a unified whole, overlooking the fact that modern LLM training often involves a mixture of data from diverse distributions—varying in both source and difficulty. This heterogeneity introduces a key challenge: how to adaptively schedule training across distributions to optimize learning efficiency. In this paper, we present a principled curriculum learning framework grounded in the notion of distribution-level learnability. Our core insight is that the magnitude of policy advantages reflects how much a model can still benefit from further training on a given distribution. Based on this, we propose a distribution-level curriculum learning framework for RL-based LLM post-training, which leverages the Upper Confidence Bound (UCB) principle to dynamically adjust sampling probabilities for different distrubutions. This approach prioritizes distributions with either high average advantage (exploitation) or low sample count (exploration), yielding an adaptive and theoretically grounded training schedule. We instantiate our curriculum learning framework with GRPO as the underlying RL algorithm and demonstrate its effectiveness on logic reasoning datasets with multiple difficulties and sources. Our experiments show that our framework significantly improves convergence speed and final performance, highlighting the value of distributionaware curriculum strategies in LLM post-training. Code: https://github.com/ ZhentingWang/DUMP.

### 1 Introduction

Reinforcement learning (RL)-based post-training has emerged as a powerful approach for enhancing the capabilities of large language models (LLMs), particularly in areas requiring structured reasoning, multi-step inference, and task-specific generalization [1–4]. By leveraging reward signals derived from task performance, human feedback, or domain-specific metrics, RL provides a flexible alternative to supervised fine-tuning. Unlike imitation-based methods that merely mimic reference outputs, RL-based approaches allow models to optimize directly toward behavioral objectives, making them especially effective for boosting model performance on complex reasoning and agentic tasks.

While RL-based post-training has become a key technique for enhancing LLM capabilities in reasoning, alignment, and coding, one foundational challenge remains underexplored: how to dynamically schedule training across heterogeneous data distributions. In practice, LLMs are posttrained on datasets drawn from a wide variety of sources—ranging from factual QA to math problems and coding tasks—each differing in knowledge/capability relevance, and learning difficulty [5– 7]. This heterogeneity is evident in large-scale post-training datasets such as Tülu 3 [7], where prompts span general dialogue, logic puzzles, STEM problems, and multilingual instructions, with

Preprint.

widely varying counts, formats, and alignment objectives. More recently, next-generation posttraining pipelines (e.g., Seed-Thinking v1.5 [8]) have shifted toward synthetic data generation with controllable parameters—e.g., configuring logical puzzle difficulty. This allows fine-grained control over the data distribution, making distribution-level curriculum learning both feasible and increasingly important. Despite this, most RL-based pipelines still treat all data distributions equally—uniformly sampling tasks throughout training or relying on static, hand-designed curricula. This static treatment ignores the model’s evolving learning needs and underutilizes the training budget. Moreover, it is difficult to handcraft effective curricula when the post-training data comes from multiple distributions lacking clear difficulty labels. As reinforcement learning becomes increasingly used in post-training and training costs continue to rise, a data-driven curriculum mechanism that dynamically prioritizes learnable distributions is not just desirable, but necessary.

This motivates the need for automated distribution-level curriculum learning: a dynamic strategy that adjusts sampling probabilities across data distributions throughout training. While prior work has explored instance-level curricula based on sample difficulty [9], and static/heuristic multi-stage schedules have been applied in LLM post-training [10, 11], little attention has been paid to automated, distribution-level scheduling—especially in the context of RL for capability-oriented post-training. The central challenge lies in identifying signals that reflect the current learnability of each distribution and in designing algorithms that can stably and efficiently leverage these signals to guide sampling.

In this paper, we present DUMP (Automated Distribution-level cUrriculuM learning for RL-based LLM Post-training), a simple but theoretically grounded approach to address this challenge. Our central insight is that the magnitude of policy advantages—the expected absolute difference between a model’s predicted return and its baseline value—serves as a natural proxy for distribution-level learnability. High advantages on specific data distribution indicate underfitting and high potential for improvement on it, while low advantages suggest diminishing returns. Moreover, the statistical reliability of these advantage estimates improves with the number of samples drawn from each distribution. DUMP operationalizes this insight by using bandit-style Upper Confidence Bound (UCB) scores to schedule distribution sampling. It maintains a sliding window of recent advantage magnitudes for each distribution and computes a score that balances exploitation (high advantage) and exploration (low visitation). These scores are normalized via a softmax to form sampling weights, which are then used to generate training batches. Unlike fixed or heuristic curricula, DUMP adapts throughout training based on empirical signals, and can be seamlessly integrated into standard LLM RL pipelines. We instantiate DUMP with GRPO [3], but the method is compatible with any advantage-based RL algorithm. We evaluate DUMP on logic reasoning corpora. Our experiments show that DUMP significantly accelerates convergence and yields stronger performance compared to uniform sampling. Furthermore, we provide theoretical analysis that supports the use of absolute advantages as a surrogate for distribution-level learnability, formalizing its connection to sample efficiency and regret minimization.

We summarize our contributions as follows. ① We highlight the underexplored challenge of curriculum learning at the distribution level for RL-based post-training aimed at capability enhancement. ② We propose DUMP, a theoretically grounded framework that leverages advantage-based UCB scores to adaptively guide training over data distributions. ③ We demonstrate DUMP’s effectiveness through empirical results and theoretical analysis, showing that it enables faster, more efficient improvement on LLM capabilities.

### 2 Background

RL-based LLM Post-training. Reinforcement learning (RL) plays a central role in post-training large language models (LLMs), especially for tasks involving reasoning, subjective preference, or long-horizon control. The RLHF framework [1, 12–15] laid the foundation by aligning models using reward signals derived from human preferences. Beyond preference alignment, recent RLbased post-training approaches have notably enhanced LLMs’ capabilities in complex reasoning tasks, particularly coding and mathematics. For instance, RL post-trained model OpenAI o1 [16], o3 [17, 18], DeepSeek-R1 [4] significantly outperform LLMs without RL post-training such as pre-trained versions of GPT-4o [19] and DeepSeek-V3 [20] on challenging mathematics and coding benchmarks (e.g., AIME [21] and Codeforces [22]). Proximal Policy Optimization (PPO) [23] is widely used in post-training due to its clipped objective, which stabilizes training by preventing large policy updates. PPO remains a strong baseline in many LLM alignment settings. Direct Preference Optimization (DPO) [2] simplifies the pipeline by replacing RL rollouts with a classification-style loss

derived from a KL-constrained reward maximization objective. While DPO works well on pairwise preference data, it does not naturally support group-wise or comparative feedback. Group Relative Policy Optimization (GRPO) [3] addresses this limitation by leveraging group-based feedback. For each input prompt x, GRPO samples a group of G candidate outputs {o1,...,oG} ∼ πref(·|x) from a frozen reference policy πref. Each output oi is assigned a reward ri, and the advantage of oi is computed by normalizing its reward relative to others in the group:

ri − mean({r1,...,rG}) std({r1,...,rG}) + ϵ

Aˆi =

, (1)

where ϵ > 0 is a small constant for numerical stability. These normalized advantages capture the relative quality of outputs within the group. The model policy πθ is then updated by maximizing the following clipped surrogate objective:

G

πθ(oi|x) πold(oi|x)

πθ(oi|x) πold(oi|x)

1 G

, 1 − ϵ, 1 + ϵ A ˆi − β DKL(πθ∥πref) ,

Aˆi, clip

JGRPO(θ) = Ex,{oi}

min

i=1

(2)

where πθ(oi|x) is the probability assigned by the current model to output oi, πold(oi|x) is the same under the model from previous step, and πref(oi|x) is that under the reference model. The first term inside the summation is a clipped policy ratio scaled by Aˆi, similar to PPO [23], which prevents overly large updates. The outer expectation is taken over prompts x and their sampled output groups {oi}. The second term is a KL divergence penalty that regularizes the updated policy πθ to stay close to πref, weighted by a hyperparameter β. This formulation eliminates the need for an explicit value baseline and stabilizes training by comparing outputs within local groups.

Curriculum Learning for RL. Curriculum learning [24, 25] organizes training by progressing from easy to hard examples. In RL, curricula often follow task complexity [26–28], or are learned via teacher-student frameworks modeled as partially observable Markov decision process [29, 30]. With the adoption of RL in LLM post-training, curriculum learning has shown potential for improving both training efficiency and model effectiveness. For example, Curri-DPO [9] constructs instance-level curricula by ranking preference pairs based on the score gap between preferred and dispreferred responses, introducing harder pairs gradually during DPO fine-tuning. Kimi k1.5 [10] and LogicRL [11], on the other hand, use manually defined heuristic curricula with fixed training stages, e.g., models are first trained on “easy” samples for a pre-specified number of steps, then switched to “hard” samples. These strategies rely on static schedules and heuristic difficulty labels, without adapting to the model’s learning progress. While these works demonstrate the benefit of curriculum learning in LLM post-training, most existing approaches focus on instance-level difficulty or use static, manually designed strategies. In contrast, automatic curriculum learning at the distribution level, especially in RL-based post-training, remains underexplored. In this paper, we propose DUMP to fill this gap by adaptively scheduling training over distributions using advantage-based learnability signals.

### 3 Method

In this section, we introduce DUMP, a distribution-level curriculum learning framework for RLbased LLM post-training. We first introduce expected absolute advantage as a proxy for learnability, and formalize the scheduling problem as a multi-armed bandit. We then describe a UCB-based strategy to guide distribution selection, followed by the full implementation of DUMP.

#### 3.1 Measuring Learnability via Absolute Advantage

We aim to dynamically assess the usefulness of different data distributions during LLM reinforcement learning post-training. Intuitively, a distribution is more useful (or “learnable”) if the model can gain more from training on its samples. To help understand and measure the learnability of the data samples from different distributions, we provide the following theorem:

Theorem 3.1 (Expected Advantage Magnitude Reflects Learnability). Given a policy πθ and a data distribution d, the expected absolute advantage Ex∼d Eo

i∼πθ(·|x) |Aˆi| serves as a proxy for how much that distribution d can help the model improve, where the distribution d consisting of prompts x ∼ d, each prompt has a group of sampled outputs {o1,...,on}, and Aˆi denotes the advantage of output oi.

The proof can be found in Appendix A. Intuitively, if training on a distribution results in a larger expected advantage magnitude, then that distribution is considered more learnable. The advantage function measures the deviation between an action’s predicted value and its actual return; a large advantage—either positive or negative—indicates that the model’s current policy is still far from optimal on those samples but has a large potential to improve. A small advantage magnitude does not necessarily imply mastery—it may also occur when a task is too difficult or noisy for the model to learn from effectively, resulting in weak or unstable learning signals. To capture this deviation in both directions, we take the absolute value of the advantage. Without this, positive and negative advantages within a batch may cancel out, masking the true extent of the model’s uncertainty or suboptimality. By averaging the absolute advantage over multiple sampled outputs and prompts, we obtain a robust estimate of how much learning signal remains in a given distribution. This expected absolute advantage thus acts as a practical proxy for distribution-level learnability: it reflects how much the model can benefit from training on that distribution. It also has the strength of being lightweight to compute in RL pipelines, as advantage estimates are already generated during rollout.

#### 3.2 Formalizing Distribution-Level Curriculum Learning as Multi-armed Bandit

We aim to design a curriculum learning strategy that dynamically allocates training focus across multiple data distributions to maximize overall model improvement. Let D = {d1,...,dN} be a set of data distributions. At each training step, we sample a batch of examples Bt by drawing prompts from these distributions according to a learnable sampling policy, and use the batch to update model parameters θ via reinforcement learning. The goal is to assign higher sampling probabilities to distributions that offer greater learning potential, thereby maximizing cumulative capability gain.

- As motivated in Theorem 3.1, we quantify the learning potential of a distribution d via its expected

θ(·|x) |Aˆ(o)| . Our objective is to dynamically adjust the sampling distribution over D such that, over the training horizon T, we approximately maximize the total expected learnability gain Tt=1 Ed∼P

absolute advantage, defined as L(d) = Ex∼d Eo∼π

[L(d)], where Pt is the sampling distribution at step t. This setup resembles a multi-armed bandit (MAB) problem, where each distribution acts as an arm and its reward corresponds to its learnability. In this setting, the central challenge is to estimate and balance each distribution’s potential: exploiting those with high observed advantage while still exploring under-sampled ones that may offer long-term benefit. To this end, we adopt the classic Upper Confidence Bound (UCB) principle [31], which provides theoretical guarantees for balancing exploration and exploitation in bandit problems. Specifically, UCB-based algorithms achieve sublinear regret compared to the optimal fixed-arm strategy, and we show in Appendix B that applying UCB on empirical advantage statistics yields a near-optimal schedule under mild assumptions. To allow smoother allocation of sampling probabilities without hard cutoffs and reducing variance in learning, we adopt a soft-selection mechanism: instead of choosing one distribution at each step, we compute a UCB score for every distribution and normalize the scores with a softmax function to obtain a sampling distribution. This soft-selection formulation preserves the spirit of UCB—higher scoring distributions are sampled more—but enables partial exploration of all arms, and it is easier to integrate into LLM training pipelines. The resulting sampling distribution provides a convex mixture over data sources, where each distribution dj is selected with probability. Each training batch is then composed by drawing examples from multiple distributions in proportion to their scores. To estimate learnability in practice, we maintain a sliding window Awdj of recent absolute advantages for each distribution dj, and define its empirical reward as the mean absolute advantage: Lˆ(dj) = |A1w

t

|a|. We also track the total number of samples drawn from each distribution nd

| a∈Awd

dj

j

, and the global sample count ntotal = j nd

. The UCB score for each distribution is:

j

j

UCB(dj) = Lˆ(dj) +

2log(ntotal + 1) nd

+ 1

j

(3)

The first term encourages exploitation of distributions with high observed advantages, while the second term ensures sufficient exploration of rarely sampled distributions. To obtain the final sampling weights, we apply a softmax over the UCB scores. Specifically, the probability of selecting distribution

dj is computed as: P(dj) = exp(UCB(dj)/τ) N j=1 exp(UCB(dj)/τ), where τ > 0 is a temperature hyperparameter that controls the sharpness of the sampling distribution. A lower τ results in more peaked selection around the top-scoring distributions, while a higher τ leads to a smoother, more exploratory curriculum. This

Algorithm 1 Automated Distribution-Level Curriculum Learning with UCB Sampling Input: Dataset D = {d1,...,dN}; pre-trained model parameters θ Output: Post-trained model parameters θ

- 1: function DUMP(D, θ)
- 2: ▷ Initialize distribution-level statistics
- 3: for each dj ∈ D do
- 4: Awdj ← [ ] ▷ Sliding window for absolute advantages
- 5: nd

j ← 0 ▷ Total samples seen from dj

- 6: P(dj) ← N1 ▷ Equal initial weights

- 7: for training step t = 1,2,...,T do
- 8: Sample batch Bt from D according to P(dj)
- 9: Compute advantages Aˆ(o) for all o ∈ Bt via model rollout
- 10: for each dj with samples in Bt do
- 11: nd

j ← nd

j

+ |Bt,dj| ▷ Update sample count; Bt,dj

: subset of batch from dj

- 12: Awdj ← Awdj ∪ |Aˆ(o)| | x ∈ Bt,dj

, o ∼ πθ(·|x) ▷ Append new advantages from dj

- 13: Awdj ← Awdj[−k :] ▷ k: Window Size; Keep last k elements
- 14: ▷ Compute UCB scores for each distribution
- 15: ntotal ← dj∈D nd

j

- 16: for each dj ∈ D do
- 17: Lˆ(dj) ← |A1w

dj

| a∈Awd

j

a ▷ Mean of absolute advantages

- 18: UCB(dj) ← Lˆ(dj) + 2 log(n

total+1)

ndj+1 ▷ Eq. 3

- 19: ▷ Update sampling distribution
- 20: P(dj) ← Nexp(UCB(dj)/τ) j=1 exp(UCB(dj)/τ) ∀dj ∈ D ▷ τ: temperature

- 21: Update θ using Bt with an RL algorithm (e.g., GRPO)
- 22: return θ

bandit-based formulation provides a lightweight, adaptive, and reward-sensitive curriculum learning mechanism. It balances the need to focus on learnable distributions while avoiding premature neglect of underexplored ones. In the next section, we present the complete algorithmic implementation of DUMP, including its integration with rollout procedures and online statistics tracking.

- 3.3 Algorithm The detailed curriculum learning procedure is illustrated in Algorithm 1. The algorithm takes as

input a dataset D = {d1,...,dN} composed of multiple distributions and returns the optimized model parameters θ through a reinforcement learning loop. In lines 3–6, we initialize per-distribution

statistics: each distribution dj ∈ D is associated with an empty sliding window Awdj to store recent absolute advantages, a counter nd

for tracking the number of samples drawn from dj, and an initial sampling probability P(dj) = N1 indicating uniform sampling. At each training step t (line 8), a batch Bt is sampled according to the current distribution weights P(dj). Advantages Aˆ(o) are then computed via model rollouts for each sampled output o ∈ Bt (line 9). For every distribution dj that contributes samples in the current batch, we update its sample count nd

j

(line 11), append the corresponding advantages to its sliding window Awdj (line 12), and truncate the window to retain only the most recent k entries (300 by default) in line 13. This ensures that our estimate of perdistribution learnability remains up-to-date and robust to noise. In lines 15–18, we compute the Upper Confidence Bound (UCB) score UCB(dj) for each distribution. The score consists of two terms: the empirical mean absolute advantage Lˆ(dj) over the sliding window Awdj, and an exploration bonus inversely proportional to the square root of the number of samples nd

j

. This balances prioritization of distributions that are either highly learnable or underexplored. In line 20, the sampling probabilities P(dj) are updated by applying a softmax over the UCB scores with a temperature parameter τ (0.1 by default). Lower values of τ result in sharper distributions that concentrate more heavily on top-ranked distributions, while higher τ values induce a smoother, more exploratory curriculum. Finally, in line

j

21, the model parameters θ are updated using the current batch Bt with a reinforcement learning algorithm such as GRPO. After T steps, the algorithm returns the post-trained model θ, which has been adaptively guided to learn from the most informative distributions.

### 4 Experiments and Results

In this section, we first introduce our experiments setup including used models datasets and more implementation details. We then demonstrate the results for the effectiveness of our method DUMP. More discussion about the comparison to static heuristic curriculum [11, 10] can be found in Appendix C.

#### 4.1 Experiments Setup

RL Algorithm and LLM Models. We use GRPO [3] as the underlying RL algorithm in our experiments, which is commonly used in capability-oriented LLM post-training [4]. We use Qwen2.57B-Instruct-1M [32] and Qwen2.5-3B-Instruct [32] in our experiments.

Datasets and Settings. Multiple datasets are used in our experiments, including Knights and Knaves (K&K) puzzle dataset [33], RuleTaker [34], ProofWriter [35], AR-LSAT [36], LogiQA [37], LogicNLI [38], LongICLBench [39], GSM-8K [40], and AIME 1983-2024 [21]. In our experiments, we consider three different settings. The prompt template used in shown in Figure 3 in the Appendix.

- Setting 1: Post-training on K&K puzzles with varying character numbers. The Knights and Knaves (K&K) dataset [33] contains procedurally generated logic puzzles where each character is either a knight (always truthful) or a knave (always lying), and the goal is to infer each character’s identity. The dataset supports fine-grained difficulty control by adjusting the number of characters. We generate puzzles with 3 to 14 characters, treating each character count as a separate distribution—yielding 12 distinct distributions. Each distribution includes 900 training and 100 test samples. We post-train Qwen2.5-7B-Instruct-1M on the combined dataset across all distributions.
- Setting 2: Post-training on diverse logic reasoning distributions. We perform post-training using a mixture of logic reasoning datasets, including RuleTaker [34], ProofWriter [35], AR-LSAT [36], LogiQA [37], LogicNLI [38], LongICLBench Geomotion [39], and Knights and Knaves (K&K) [33]. For RuleTaker, ProofWriter, and K&K, we further partition the data distributions by complexity levels: RuleTaker by 2, 3, and 5 required reasoning steps; ProofWriter by 3, 4, and 5 required reasoning steps; and K&K by the number of characters (3–7). In total, we construct 15 logic distributions, each containing 400 training samples. We use Qwen2.5-7B-Instruct-1M for this setting.
- Setting 3: Post-training on diverse math reasoning distributions. We also explore post-training on diverse math data. For AIME, we split the data into four distributions based on competition years—1983–1993, 1994–2004, 2005–2015, and 2016–2024—since problem styles evolve significantly over time. We also include GSM-8K as a complementary math dataset. This results in five math distributions in total, with 7473 (GSM-8K), 124, 194, 283, and 238 training samples, respectively. We use Qwen2.5-3B-Instruct for this setting.

Reward Implementation. We adopt the rule-based reward mechanism Shao et al. [3] to provide stable and hack-resistant training signals during RL-based post-training and follow the detailed reward implemetation in Logic-RL [11]. Specifically, each model response is expected to follow a structured format with the reasoning process enclosed in <think> tags and the final answer enclosed in <answer> tags. The reward system consists of two components:

- • Format Reward. A binary reward based on whether the output strictly adheres to the expected format. If the model includes exactly one well-formed <think> and one <answer> section in the correct order, it receives a reward of +1; otherwise, it receives a penalty of −1.
- • Answer Reward. We evaluate the correctness of the final answer. If the predicted identities fully match the ground truth, the model receives a reward of +2; if the answer is incorrect, −1.5; and if the answer is missing or unparsable, −2.

Other Implementation Details. All experiments are conducted on servers equipped with 8 Nvidia A100 GPUs. Our method is implemented with VeRL [41] LLM Reinforcement Learning framework. We use GRPO [3] as the training algorithm and follow standard practice for actor rollout and optimization. The actor learning rate is set to 1e−6, training batch size is set to 128, and the PPO

Data Distribution without DUMP with DUMP

- RuleTaker 2 Steps 0.79 0.79
- RuleTaker 3 Steps 0.76 1.02 RuleTaker 5 Steps 0.56 0.98

- ProofWriter 3 Steps 1.18 1.09
- ProofWriter 4 Steps 0.97 1.09
- ProofWriter 5 Steps 1.24 1.05 AR-LSAT -0.70 -0.52

LogiQA 1.94 1.70

LogicNLI -0.29 -0.23 LongICLBench Geomotion 0.54 0.25

- K & K 3 Characters 2.00 2.00
- K & K 4 Characters 1.54 1.76
- K & K 5 Characters 1.53 1.84
- K & K 6 Characters 0.83 1.42
- K & K 7 Characters 0.56 1.02 Average 0.90 1.17

Table 1: Test Answer Reward (see Section 4.1) on diverse logic reasoning distributions (Setting 2). The model used here is Qwen2.5-7B-Instruct-1M.

Data Distribution without DUMP with DUMP

GSM-8K 1.50 1.47 AIME 1983-1993 -0.76 -0.39 AIME 1994-2004 -1.50 -1.02 AIME 2005-2015 -0.94 -0.94 AIME 2016-2024 -1.27 -1.27

Average -0.59 -0.43

Table 2: Test Answer Reward (see Section 4.1) on diverse math reasoning distributions (Setting 3). The model used here is Qwen2.5-3B-Instruct.

mini-batch size is 32. KL divergence regularization is applied to encourage alignment with the reference policy, with a KL loss coefficient of 0.001. Each rollout batch contains 16 responses. If not specified, we allow for a maximum response length of 20480 and 4096 tokens during training for Qwen2.5-7B-Instruct-1M and Qwen2.5-3B-Instruct, respectively. The window size k and the temperature τ in our curriculum learning framework is set to 300 and 0.1, respectively.

#### 4.2 Effectiveness of DUMP

- Setting 1: Post-training on the combination of K&K puzzle datasets with different number of characters. To evaluate the effectiveness of DUMP in improving post-training efficiency and performance, we compare it against a uniform distribution sampling baseline across 12 distinct data distributions in the K&K puzzle dataset. Each distribution corresponds to a fixed number of characters in the puzzle, ranging from 3 to 14. Figure 1 plots the test answer reward over training steps for each distribution, with and without DUMP. Across all distributions, DUMP consistently outperforms the baseline, achieving faster convergence and higher test performance. The gains are particularly notable in mid- to high-difficulty distributions (e.g., 6 to 12 characters), where uniform sampling tends to struggle due to data underutilization. For example, in the 9-character distribution (Figure 1g), the model trained with DUMP achieves a reward of over 0.5, whereas the baseline remains below 0.0. These results validate the core intuition of DUMP: dynamically adjusting the sampling focus toward high-learnability distributions accelerates policy improvement while avoiding wasted effort on over-saturated or low-signal data. Notably, the improvement is achieved without any curriculum heuristics or manual data ordering—only by observing advantage signals and adapting online.
- Setting 2: Post-training on diverse logic reasoning distributions. We apply DUMP to 15 logic reasoning distributions including subsets of RuleTaker, ProofWriter, and K&K (with varying difficulty levels), as well as datasets such as AR-LSAT, LogiQA, LogicNLI, and LongICLBench. As shown in Table 1, DUMP improves the average test answer reward from 0.90 to 1.17. Notable improvements are observed on complex tasks such as AR-LSAT, where the reward increases from -0.70 to -0.52, and K&K 7 Characters, from 0.56 to 1.02. These results demonstrate that DUMP adaptively prioritizes undertrained but learnable distributions, leading to more efficient capability gains.
- Setting 3: Post-training on diverse math data distributions. We further evaluate DUMP on GSM-8K and different subsets of AIME grouped by competition years. As shown in Table 2, DUMP raises the average test answer reward from -0.59 to -0.43, with the most significant gain on AIME 1994–2004, where performance improves from -1.50 to -1.02. These results highlight DUMP’s robustness under distribution shifts and data imbalance.

#### 4.3 Ablation Study on the Sampling Strategy

In this section, we ablate the sampling strategy used in DUMP’s UCB-based scheduler. As described in Algorithm 1, our method applies soft sampling controlled by a temperature parameter. The greedy variant (temperature = 0) always selects the distribution with the highest UCB score, while our default uses a small temperature (0.1) to enable probabilistic sampling. We conduct experiments under Setting 1, with a maximum training response length of 10240 tokens. After 100 training steps, the

(d) 6 Characters (e) 7 Characters (f) 8 Characters

[Figure 7]

[Figure 8]

[Figure 9]

(g) 9 Characters (h) 10 Characters (i) 11 Characters

[Figure 10]

[Figure 11]

[Figure 12]

(j) 12 Characters (k) 13 Characters (l) 14 Characters

- Figure 1: Effectiveness of DUMP on the K&K puzzle dataset mixed with 12 distributions defined by the number of characters in each puzzle (Setting 1). DUMP consistently achieves higher answer reward on test dataset compared to baseline. The model used here is Qwen2.5-7B-Instruct-1M.

greedy strategy significantly underperforms due to its lack of exploration—it tends to lock onto a single distribution early and fails to adapt. For instance, on the 13- and 14-character K&K tasks, the greedy variant achieves test answer rewards of −0.91 and −1.38, while soft sampling reaches −0.66 and −1.16, respectively. These results highlight the importance of maintaining exploration via a non-zero temperature to prevent the scheduler from collapsing onto suboptimal distributions.

#### 4.4 Analyzing the Automated Curriculum by DUMP

To understand how DUMP dynamically allocates training effort across data distributions, we analyze the sampling patterns induced by its UCB-based curriculum mechanism. Figure 2 shows the cumulative number of samples drawn from each distribution (3 to 14 characters) over the course of training on K&K puzzles with varying character numbers (Setting 1). We observe a clear curriculum-like progression: distributions corresponding to simpler puzzles (e.g., 3–5 characters) are heavily sampled in the early stages of training, while more complex distributions (e.g., 10–14 characters) are gradually introduced and increasingly prioritized as training progresses. This pattern aligns with the model’s evolving capacity—early training favors distributions with high initial advantage magnitudes, and as the model saturates on those, DUMP shifts focus to underexplored but learnable distributions. Importantly, this adaptive sampling behavior emerges automatically from empirical advantage signals

(d) 6 Characters (e) 7 Characters (f) 8 Characters

[Figure 19]

[Figure 20]

[Figure 21]

(g) 9 Characters (h) 10 Characters (i) 11 Characters

[Figure 22]

[Figure 23]

[Figure 24]

(j) 12 Characters (k) 13 Characters (l) 14 Characters

- Figure 2: Curriculum (sample counts) induced by DUMP across 12 K&K puzzle distributions with increasing difficulty defined by the number of characters in each puzzle (Setting 1). Simpler distributions are automatically prioritized in early training, while more complex ones are progressively emphasized—both in an entirely automated manner—demonstrating automated distribution scheduling.

without requiring manual specification of curriculum order. These results highlight DUMP’s ability to construct an implicit, data-driven curriculum that mirrors traditional easy-to-hard strategies, while remaining responsive to online training dynamics.

### 5 Conclusion

In this work, we introduce a distribution-level curriculum learning framework for RL-based posttraining of large language models. DUMP leverages the expected absolute advantage as a learnability signal to adaptively allocate training focus across heterogeneous distributions. By formalizing scheduling as a multi-armed bandit and adopting a UCB-based sampling strategy, DUMP balances exploitation and exploration in a principled way. Experiments demonstrate that DUMP consistently improves convergence and final performance over baselines. These results highlight the value of distribution-aware curriculum learning in LLM RL post-training.

### References

- [1] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [2] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.
- [3] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [4] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [5] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. In International Conference on Machine Learning, pages 22631–22648. PMLR, 2023.
- [6] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, et al. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv e-prints, pages arXiv–2309, 2023.
- [7] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\" ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.
- [8] ByteDance Seed. Seed-thinking-v1.5: Advancing superb reasoning models with reinforcement learning. Technical report, ByteDance, 2025. URL https://github.com/ ByteDance-Seed/Seed-Thinking-v1.5.
- [9] Pulkit Pattnaik, Rishabh Maheshwary, Kelechi Ogueji, Vikas Yadav, and Sathwik Tejaswi Madhusudhan. Enhancing alignment using curriculum learning & ranked preferences. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12891–12907, 2024.
- [10] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [11] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.
- [12] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.
- [13] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
- [14] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

- [15] Amelia Glaese, Nat McAleese, Maja Tr˛ebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022.
- [16] OpenAI. Learning to reason with llms. Technical report, OpenAI, 2024. URL https: //openai.com/index/learning-to-reason-with-llms/.
- [17] OpenAI. Openai o3-mini. Technical report, OpenAI, 2025. URL https://openai.com/ index/openai-o3-mini/.
- [18] Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, et al. Competitive programming with large reasoning models. arXiv preprint arXiv:2502.06807, 2025.
- [19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [20] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [21] Aime_1983_2024 (revision 6283828), 2025. URL https://huggingface.co/datasets/ di-zhang-fdu/AIME_1983_2024.
- [22] Mikhail Mirzayanov. Codeforces. https://codeforces.com/. Accessed: 2025-04-13.
- [23] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [24] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.
- [25] Alex Graves, Marc G Bellemare, Jacob Menick, Remi Munos, and Koray Kavukcuoglu. Automated curriculum learning for neural networks. In international conference on machine learning, pages 1311–1320. Pmlr, 2017.
- [26] Niels Justesen, Ruben Rodriguez Torrado, Philip Bontrager, Ahmed Khalifa, Julian Togelius, and Sebastian Risi. Illuminating generalization in deep reinforcement learning through procedural level generation. arXiv preprint arXiv:1806.10729, 2018.
- [27] Rui Wang, Joel Lehman, Jeff Clune, and Kenneth O Stanley. Paired open-ended trailblazer (poet): Endlessly generating increasingly complex and diverse learning environments and their solutions. arXiv preprint arXiv:1901.01753, 2019.
- [28] Richard Li, Allan Jabri, Trevor Darrell, and Pulkit Agrawal. Towards practical multi-object manipulation using relational reinforcement learning. In 2020 ieee international conference on robotics and automation (icra), pages 4051–4058. IEEE, 2020.
- [29] Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. Teacher–student curriculum learning. IEEE transactions on neural networks and learning systems, 31(9):3732–3740, 2019.
- [30] Rémy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. In Conference on Robot Learning, pages 835–853. PMLR, 2020.
- [31] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47:235–256, 2002.
- [32] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

- [33] Chulin Xie, Yangsibo Huang, Chiyuan Zhang, Da Yu, Xinyun Chen, Bill Yuchen Lin, Bo Li, Badih Ghazi, and Ravi Kumar. On memorization of large language models in logical reasoning. arXiv preprint arXiv:2410.23123, 2024.
- [34] Peter Clark, Oyvind Tafjord, and Kyle Richardson. Transformers as soft reasoners over language. arXiv preprint arXiv:2002.05867, 2020.
- [35] Oyvind Tafjord, Bhavana Dalvi Mishra, and Peter Clark. Proofwriter: Generating implications, proofs, and abductive statements over natural language. arXiv preprint arXiv:2012.13048, 2020.
- [36] Wanjun Zhong, Siyuan Wang, Duyu Tang, Zenan Xu, Daya Guo, Jiahai Wang, Jian Yin, Ming Zhou, and Nan Duan. Ar-lsat: Investigating analytical reasoning of text. arXiv preprint arXiv:2104.06598, 2021.
- [37] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. arXiv preprint arXiv:2007.08124, 2020.
- [38] Jidong Tian, Yitian Li, Wenqing Chen, Liqiang Xiao, Hao He, and Yaohui Jin. Diagnosing the first-order logical reasoning ability through logicnli. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3738–3747, 2021.
- [39] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. Long-context llms struggle with long in-context learning. arXiv preprint arXiv:2404.02060, 2024.
- [40] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [41] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

### A Proof for Theorem 3.1

- Theorem A.1 (Expected Advantage Magnitude Reflects Learnability). Given a policy πθ and a data

i∼πθ(·|x) |Aˆi| serves as a proxy for how much that distribution d can help the model improve, where the distribution d consisting of prompts x ∼ d, each prompt has a group of sampled outputs {o1,...,on}, and Aˆi denotes the advantage of output oi.

distribution d, the expected absolute advantage Ex∼d Eo

Proof. Let πθ be the current model policy. Consider a data distribution d, where x ∼ d are prompts and {o1,...,on} ∼ πθ(·|x) are sampled outputs. For each output oi, the advantage is estimated as

Aˆi = ri − b(x),

where ri is the reward assigned to oi and b(x) is a baseline (e.g., the mean reward over the group). The policy gradient under common policy-gradient methods (e.g., PPO or GRPO) can be written as:

i∼πθ(·|x) A ˆi · ∇θ log πθ(oi | x) .

∇θJ (θ) = Ex∼d Eo

Now consider the magnitude of the gradient vector. The strength of the training signal from d depends on the expected norm of the gradient, which is bounded below by:

∥∇θJ (θ)∥ ≳ Ex∼d Eo

i∼πθ(·|x) |Aˆi| · ∥∇θ log πθ(oi | x)∥ .

Assuming that ∥∇θ log πθ(oi | x)∥ is bounded and varies slowly across d, the dominant term affecting the gradient norm is:

i∼πθ(·|x) |Aˆi| .

Ex∼d Eo

Thus, the expected absolute advantage serves as a proxy for the learning signal magnitude contributed by distribution d. The expected absolute advantage reflects how much training on distribution d can improve the model parameters, making it a suitable signal for curriculum scheduling.

| |
|---|

### B Theoretical Justification for UCB-Based Distribution Scheduling

We provide a theoretical justification for using Upper Confidence Bound (UCB) as a strategy for scheduling training over data distributions in RL-based post-training. Our objective is to maximize the cumulative learnability gain over T training steps, defined as:

max

{dt}Tt=1

T

θ(·|x) |Aˆ(o)| .

L(dt), where L(d) = Ex∼d Eo∼π

t=1

This setting can be viewed as a stochastic multi-armed bandit (MAB) problem, where each data distribution dj ∈ D corresponds to an arm with unknown reward L(dj), interpreted as the expected absolute advantage from training on samples from dj. At each training step t, the learner selects a distribution dt and obtains an empirical reward Lˆ(dt) by averaging the absolute advantages observed in the batch.

We define the regret as the gap between the cumulative learnability gain of the best fixed distribution d∗ = arg maxd L(d) and that of the learner’s actual selections:

Regret(T) =

T

T

L(d∗) −

L(dt).

t=1

t=1

To analyze this regret, we make the following assumptions:

- 1. For each distribution dj, the per-output absolute advantages |Aˆ(o)|, where o ∼ πθ(·|x), are i.i.d. and bounded in [0,C] for some constant C > 0.

- 2. The true expected advantage L(dj) remains approximately stationary over a local training window, enabling meaningful online adaptation.

Note: In practice, we can clip or normalize |Aˆ(o)| to satisfy the boundedness condition. The introduction of the constant C only scales the regret by a constant factor and does not affect the asymptotic rate of convergence.

Under these assumptions, the following regret bound holds:

- Theorem B.1. Let D = {d1,...,dN} be a set of data distributions with fixed expected rewards L(dj) ∈ [0,C]. Then, applying the UCB1 algorithm to the empirical reward observations yields the regret bound:

 , where ∆j = L(d∗) − L(dj).

 C ·

log T ∆j

Regret(T) ≤ O

j:∆j>0

Proof. This result is a direct application of the classical UCB1 regret bound [31], extended to the case where reward values lie in [0,C]. Let d∗ = arg maxd L(d) be the optimal distribution, and let ∆j = L(d∗) − L(dj) denote the suboptimality gap for each arm dj.

- At each time step t, UCB1 selects the distribution dj with the highest upper confidence bound:

UCB(dj) = Lˆ(dj) +

2C2 log t nj

,

where nj is the number of times distribution dj has been sampled so far, and Lˆ(dj) is the empirical mean of observed rewards (mean absolute advantages).

Under the assumptions that rewards are i.i.d. and bounded in [0,C], the Hoeffding inequality guarantees that with high probability the empirical mean concentrates around the true mean L(dj), and the UCB selection mechanism will only pick suboptimal arms a logarithmic number of times. Based on UCB1 regret bound [31], The cumulative regret is therefore bounded by:

Regret(T) ≤

j:∆j>0

8C2 log T ∆j

+ O(∆j) ,

which simplifies to the stated asymptotic bound:

 .

 C ·

log T ∆j

Regret(T) = O

j:∆j>0

| |
|---|

This result shows that our distribution-level scheduling strategy, when driven by UCB over empirical advantage rewards, is provably efficient. It dynamically concentrates training on distributions with high estimated learnability while ensuring sufficient exploration, with regret that scales logarithmically in T and linearly in 1/∆j.

### C Comparison to Heuristic Curriculum

Heuristic curricula, which manually specify a fixed training schedule over data distributions—e.g., training on Distribution A for N steps before switching to Distribution B—have been explored in prior work [11, 10], particularly in environments where task difficulty or domain progression is well understood. However, such approaches have several limitations that make them less suitable for our setting. First, effective heuristic scheduling requires strong prior knowledge about the relative difficulty and learnability of each distribution. In our setting, which involves diverse domains such as logic reasoning, mathematics, and programming, such prior knowledge is often unavailable or misleading. For example, a distribution may appear “easier” but provide low learning signal, or seem “harder” but actually yield high gradient utility. This makes it extremely difficult to construct

Example of Prompt

You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. [Problem]

Figure 3: Example of prompt used.

robust, generalizable heuristics across tasks. Second, heuristic curricula are static and cannot adapt to the evolving needs of the model during training. In contrast, DUMP dynamically adjusts sampling priorities based on actual model performance—measured via policy advantages—allowing it to focus on the most beneficial distributions at each stage of learning. Finally, the lack of standardized or widely accepted heuristic curricula for our task suite makes it hard to conduct fair and meaningful comparisons. Instead, we benchmark DUMP against uniform sampling and adaptive baselines, which are more reflective of current best practices in large-scale post-training pipelines.

### D Limitations

First, while the core idea of distribution-level curriculum learning is broadly applicable, we evaluate DUMP only in the context of large language models (LLMs) and do not extend the experiments to multimodal large language models (MLLMs) due to computational constraints. Second, our experiments are limited to 7B-scale models. Scaling our method to larger models remains an important direction for future work.

