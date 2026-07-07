# arXiv:2510.18927v1[cs.LG]21Oct2025

[Figure 1]

Fudan NLP Lab 2025-10-23

### BAPO: Stabilizing Off-Policy Reinforcement Learning for LLMs via Balanced Policy Optimization with Adaptive Clipping

Zhiheng Xi1∗†, Xin Guo1∗, Yang Nan1, Enyu Zhou1, Junrui Shen1, Wenxiang Chen1, Jiaqi Liu1, Jixuan Huang1, Zhihao Zhang1, Honglin Guo1, Xun Deng2, Zhikai Lei2, Miao Zheng2, Guoteng Wang2, Shuo Zhang2, Peng Sun2, Rui Zheng2, Hang Yan2, Tao Gui1,3†, Qi Zhang1†, Xuanjing Huang1 1Fudan University 2Shanghai Qiji Zhifeng Co., Ltd. 3Shanghai Innovation Institute

zhxi22@m.fudan.edu.cn, {tgui,qz}@fudan.edu.cn

Reinforcement learning (RL) has recently become the core paradigm for aligning and strengthening large language models (LLMs). Yet, applying RL in off-policy settings–where stale data from past policies are used for training–improves sample efficiency, but remains challenging: policy entropy declines sharply, optimization often becomes unstable and may even collapse. Through theoretical and empirical analysis, we identify two key insights: (i) an imbalance in optimization, where negative-advantage samples dominate the policy gradient, suppressing useful behaviors and risking gradient explosions; and (ii) the derived Entropy-Clip Rule, which reveals that the fixed clipping mechanism in PPO-like objectives systematically blocks entropy-increasing updates, thereby driving the policy toward over-exploitation at the expense of exploration. Building on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a simple yet effective method that dynamically adjusts clipping bounds to adaptively re-balance positive and negative contributions, preserve entropy, and stabilize RL optimization. Across diverse off-policy scenarios–including sample replay and partial rollout–BAPO achieves fast, stable, and data-efficient training. On AIME 2024 and AIME 2025 benchmarks, our 7B BAPO model surpasses open-source counterparts such as SkyWork-OR1-7B, while our 32B BAPO model not only achieves state-of-the-art results among models of the same scale but also outperforms leading proprietary systems like o3-mini and Gemini-2.5-Flash-Thinking.

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

Figure 1 | Performance of BAlanced Policy Optimization with Adaptive Clipping (BAPO).

*Equal contribution. †Corresponding authors. 1Our code are available at https://github.com/WooooDyy/BAPO.

#### 1. Introduction

Reinforcement learning (RL) has become a pivotal paradigm for optimizing large language models (LLMs) (Zhang et al., 2025), delivering significant improvements in complex tasks such as reasoning (Guo et al., 2025; Jaech et al., 2024), coding (Anthropic, 2025), and agentic decision-making (Bai et al., 2025). Among RL methods, off-policy RL–where the rollout policy (behavior policy) differs from the training policy (target policy)–emerges as particularly promising (Arnal et al., 2025; Roux et al., 2025). It offers high sample efficiency and tolerance to data staleness, making it well-suited for extremely long-horizon and challenging scenarios, while also aligning more naturally with features in modern AI infrastructures such as partial rollout (Fu et al., 2025; Team et al., 2025).

However, applying off-policy RL to LLMs introduces substantial challenges (Arnal et al., 2025; Yu et al., 2025). As shown in Figure 2, increasing data staleness leads to unstable optimization, exploding gradient and even collapse. Meanwhile, policy entropy declines sharply, reflecting reduced exploratory capacity and a bias toward over-exploitation. By contrast, on-policy training–where rollout and target policies coincide–remains stable across metrics, consistent with prior studies (Arnal et al., 2025; Roux et al., 2025; Tang et al., 2024).

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

× × ×

- Figure 2 | Preliminary results with different data staleness. As the staleness increases, the model suffers from unstable optimization, decreasing entropy, and even a sudden collapse in training.

To understand the instability of off-policy training, we conduct a comprehensive theoretical and empirical analysis to reveal two key insights. We first demonstrate an imbalance in optimization: policy updates are often dominated by negative-advantage samples, producing excessive penalization signals that suppress even neutral or correct actions and may cause gradient explosions (Gülçehre et al., 2023). We then derive and empirically validate the Entropy-Clip Rule in the widely-used PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024), showing that the clipping mechanism in PPOlike objectives blocks many low-probability positive tokens while over-penalizing low-probability negatives. This systematically excludes entropy-increasing updates, sharpens the output distribution, and drives policies toward over-exploitation at the cost of exploration.

Based on these insights, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO),

a new method for stable and effective off-policy RL. BAPO dynamically adjusts the clipping bounds to re-balance positive and negative contributions for each update step, incorporate low-probability positives while filtering excessive negatives, and preserve policy entropy–achieving a better balance between exploration and exploitation. An overview of our approach is illustrated on the right side of Figure 3.

Experiments across diverse off-policy scenarios–including sample replay, partial rollout, and varying degrees of staleness–on base models such as DeepSeek-R1-Distill-Qwen-7B (Guo et al., 2025) and OctoThinker-Llama3.2-3B-Long-Zero (Wang et al., 2025b) show that BAPO consistently yields significant improvements. Our 7B model achieves scores of 70.8 on AIME24 and 62.5 on AIME25,

###### GRPO: Sharpen distribution → Entropy ( ) ↓

###### BAPO: Smooth distribution → Entropy ( ) ↑

Probability

Probability

[Negative samples] [Positive samples]

[Negative samples] [Positive samples]

token (optimized)

token

###### token (clipped)

token (clipped) (optimized)

풉풊 풉

풍  

0 1 1 ∞ 0  −  − 1 1  +  + ∞

푙   ℎ  ℎ

###### Before Optimization After GRPO After BAPO

- Figure 3 | An illustration of our proposed BAPO. (Left) Baseline methods like GRPO use symmetric fixed clipping bounds, reinforcing high-probability positive tokens while penalizing excessive lowprobability negatives, leading to sharp distributions and entropy collapse. (Right) BAPO dynamically

adjusts the clipping bounds 𝑐low and 𝑐high based on the loss contributions from positive tokens. It excludes overly negative tokens to maintain a smoother distribution and incorporates previously clipped positive tokens to preserve entropy balance.

[Figure 2]

[Figure 3]

surpassing open-source counterparts such as SkyWork-OR1-7B (He et al., 2025). Moreover, our 32B model reaches 87.1 on AIME24 and 80.0 on AIME25, outperforming both comparably scaled open-source models like Qwen3-32B (Yang et al., 2025a) and leading proprietary systems including o3-mini-medium (OpenAI, 2025) and Gemini-2.5-Flash-Thinking (Comanici et al., 2025).

Our contributions are summarized as follows:

- • We identify and analyze two key insights behind instability in off-policy RL for LLMs: the imbalanced optimization and the Entropy-Clip Rule. (§3)
- • We propose BAPO, a new RL algorithm that dynamically adjusts clipping bounds to balance positive and negative signals, preserving entropy for exploration, and stabilizing training. (§4)
- • We validate BAPO across multiple backbones, model scales, and off-policy settings, showing that it achieves stable optimization and competitive results with proprietary systems. (§5)

#### 2. Preliminaries

###### 2.1. Policy Gradient

In the field of LLM RL (Jaech et al., 2024; Trung et al., 2024), policy gradient-based (PG) algorithms (Williams, 1992) are widely used. Specifically, given an input prompt 𝒙, an LLM 𝜋𝜃 sequentially generates a 𝑇-token response 𝒚 = (𝑦1, ..., 𝑦𝑇):

𝜋𝜃(𝒚|𝒙) = 𝑇𝑡=1 𝜋𝜃(𝑦𝑡|𝒙, 𝒚<𝑡). (1)

Given a training dataset D = {𝒙1, ..., 𝒙𝑁} and reward function 𝑅, the RL objective is to maximize the expected reward:

𝜃(·|𝒙) [𝑅(𝒙, 𝒚)] . (2) PG algorithms then leverage gradient ascent to optimize the policy with the following gradient:

𝐽(𝜃) = 𝔼𝒙∼D, 𝒚∼𝜋

∑︁ 𝑇

∇𝜃 log𝜋𝜃(𝑦𝑡|𝒙, 𝒚<𝑡) · 𝐴𝑡 , (3)

###### ∇𝜃𝐽(𝜃) = 𝔼𝒙∼D, 𝒚∼𝜋

𝜃(·|𝒙)

𝑡=1

where 𝐴𝑡 denotes the estimated advantage at time step 𝑡, i.e., how much better action 𝑦𝑡 is than the expected action under the current policy.

###### 2.2. Importance Sampling and PPO Objective

To improve sample efficiency and adapt to modern infrastructure, mainstream RL algorithms for LLMs typically adopt a PPO-like surrogate objective (Schulman et al., 2017):

∑︁𝑇

𝐽PPO(𝜃) = 𝔼𝒙∼D, 𝒚∼𝜋

[min(𝑟𝑡 · 𝐴𝑡, clip(𝑟𝑡, 1 − 𝜀, 1 + 𝜀) · 𝐴𝑡)] , (4)

𝜃rollout (·|𝒙)

𝑡=1

𝜃rollout(𝑦𝑡|𝒙,𝒚<𝑡) is the importance weight that corrects for the distribution mismatch, estimating the expected advantage of tokens generated by the behavior policy 𝜋𝜃rollout under the target policy 𝜋𝜃. The clipping mechanism in PPO serves to implicitly enforce a trust region between the behavior and target policies, preventing overly large policy updates that could destabilize training. The hyperparameter 𝜀 ∈ (0, 1) determines the width of this clipping interval.

where 𝑟𝑡 = 𝜋 𝜋𝜃(𝑦𝑡|𝒙,𝒚<𝑡)

We then analyze data with positive and negative advantages respectively. The policy gradient can then be expressed as:

∇𝐽PPO = ∑︁ 𝐴𝑡>0

+ ∑︁

𝜋𝜃(𝑦𝑡) · 𝕀{𝑟𝑡 < 1 + 𝜀} · 𝐴𝑡 · ∇ log𝜋𝜃(𝑦𝑡)

𝜋𝜃(𝑦𝑡) · 𝕀{𝑟𝑡 > 1 − 𝜀} · 𝐴𝑡 · ∇ log𝜋𝜃(𝑦𝑡)

, (5)

𝐴𝑡<0

positive tokens

negative tokens

where 𝕀 represents the indicator function.

#### 3. Motivation: Imbalanced Optimization and Entropy-Clip Rule

In this section, we first conduct preliminary experiments to show the influence of data staleness on the RL optimization process. Next, we perform in-depth empirical and theoretical analysis to reveal the underlying mechanisms and provide new insights.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | |
|---|---|
| | |
| | |

Figure 4 | Contribution of positive and negative tokens to the policy-gradient loss and their proportion of tokens during training.

Figure 5 | Relationship between token probability and importance sampling weight.

Training instability with data staleness. We perform experiments under different levels of data staleness using the popular GRPO algorithm. Results in Figure 2 show that, compared to on-policy training, off-policy RL typically suffers from instability, and entropy decreases rapidly, reflecting reduced exploratory capacity (He et al., 2025). As staleness increases, the entropy decline becomes more severe and a larger number of tokens are clipped; meanwhile, training becomes more unstable. In the following paragraphs, we attempt to explain this phenomenon from different perspectives and summarize the motivation behind our method.

###### Excessive negative samples lead to imbalanced optimization.

Within the PPO-like objective for policy updates, we analyze tokens with positive and negative advantages separately, as shown in Equation 5. Empirical results in Figure 4 reveal a pronounced imbalance: positive samples constitute a minority both in number and in their contribution to the policy-gradient loss. We attribute this skew to two main factors: (i) the model tends to generate longer trajectories on difficult queries, thereby producing more tokens in negative samples (Figure 6); and (ii) in early stages of training, the model has not yet acquired sufficient capability, resulting in a higher proportion of negative samples. This observation may help explain the effectiveness of certain curriculum-based approaches (Xi et al., 2024; Yuan et al., 2025).

Figure 6 | Average model response length during training.

In the RL training of LLMs, reinforcing positive samples is often more efficient for driving performance gains than attempting to “suppress” the vast number of negative samples (Gülçehre et al., 2023; Zhu et al., 2025). To this end, prior work has proposed amplifying positive signals through the clip-higher technique (Yu et al., 2025). However, merely enlarging the clipping upper bound does not mitigate the influence of negative data, thus failing to prevent them from dominating the optimization process. Moreover, as shown in Equation 5, the accumulation of low-probability negative tokens (i.e., 𝜋𝜃(𝑦𝑡) → 0, driving the log term toward −∞) may trigger gradient explosion, further destabilizing training (Yang et al., 2025c).

The Entropy-Clip Rule exposes insufficient entropy promotion in optimization, leading to entropy collapse. Theoretically, we derives Equation 6 (see Appendix B for detailed derivations) for PPO surrogate objective to reveal the factors that influence the policy entropy (Roux et al., 2025):

𝜃(·|𝒙) [log𝜋𝜃(𝑦𝑡|𝒙, 𝒚<𝒕), 𝐴𝑡 · X(𝑦𝑡) + 𝐶] , (6) where 𝐶 is a constant, and

ΔH(𝜋𝜃) ≈ −𝜂 · Cov𝒚∼𝜋

 

1, if 𝐴𝑡 > 0 & 𝑟𝑡 < 1 + 𝜖

(7)

or 𝐴𝑡 < 0 & 𝑟𝑡 > 1 − 𝜖 0, otherwise.

X(𝑦𝑡) =

 

We observe that changes in policy entropy are driven by the influence of unclipped tokens, which is determined by the covariance between their log probabilities and advantages. We term this as the Entropy-Clip Rule. The left side of Figure 3 illustrates how the optimization of different tokens influences the probability distribution, thereby affecting entropy. The Entropy-Clip Rule theoretically explains the following statement: Specifically, updating the policy with positive high-probability tokens (high advantage, high probability) and negative low-probability tokens (low advantage, low probability) sharpens the distribution and consequently reduces entropy. Conversely, updating the policy with negative high-probability tokens and positive low-probability tokens smooths the distribution, resulting in an increase in entropy (detailed proofs are available in Appendix B.4.2).

Empirically, our statistical analysis on token probabilities and their importance sampling (IS) weights further clarifies this dynamic. As shown in Figure 5, we find that tokens with either very high or very low IS weights tend to have low probabilities. However, in standard algorithms with symmetric clipping bounds (e.g., [0.8,1.2]), a majority of positive, low-probability tokens are prevented from contributing to the optimization. This systematic exclusion of entropy-increasing up-

dates causes a continuous decline in entropy, ultimately crippling the model’s exploratory capacity and resulting in a performance bottleneck.

Summary of motivation. Based on the above analysis, we summarize two main motivations: (1) to balance the contributions of positive and negative tokens while preventing gradient explosion, and (2) to preserve policy entropy for sustaining exploration and preventing collapse.

#### 4. Methodology

###### 4.1. Validation Experiment: Asymmetric Clipping

The main idea of our method is to stabilize the training and maintain exploration ability of the policy by asymmetrically adjusting the trust region for positive and negative tokens, i.e., adjusting 𝑐low and 𝑐high.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

We then conduct preliminary experiments to examine whether asymmetrically adjusting the clipping range could influence training dynamics. The results, shown in Figure 7, together with Figure 5, reveal that increasing the upper bound 𝑐high (which introduces more lowprobability positive tokens to policy updates) improves performance while counteracting the downward trend of entropy, albeit at a rapid pace. In contrast, relaxing the lower bound 𝑐low (which introduces more low-probability negative tokens to policy updates) not only degrades performance but also accelerates entropy collapse. These findings confirm the effectiveness of entropy control through asymmetric clipping. Nevertheless, this strategy remains relatively rigid and manually specified, providing limited flexibility and adaptation.

Figure 7 | Training dynamics of asymmetric clipping experiments.

###### 4.2. BAPO: BAlanced Policy Optimization with Adaptive Clipping

To this end, we propose BAlanced Policy Optimization with Adaptive Clipping (BAPO), a new method to achieve stable, fast RL optimization for LLMs. The core insight of BAPO lies in its adaptive clipping mechanism, which dynamically adjusts the clipping bounds 𝑐high and 𝑐low, to regulate the positive contribution to the policy loss and maintain a balance in entropy throughout RL training. Formally, for each update with a batch, our goal is to find a pair of 𝑐high and 𝑐low that satisfy:

| 𝐴𝑡>0 𝜋𝜃rollout(𝑦𝑡) · min(𝑟𝑡 · 𝐴𝑡, clip(𝑟𝑡, 0, 𝑐high) · 𝐴𝑡) | | 𝐴𝑡 𝜋𝜃rollout(𝑦𝑡) · min(𝑟𝑡 · 𝐴𝑡, clip(𝑟𝑡, 𝑐low, 𝑐high) · 𝐴𝑡) |

≥ 𝜌0 , (8)

where 𝜌0 is the target contribution of positive signals to the policy gradient loss. Specifically, BAPO gradually increases 𝑐high and 𝑐low with step sizes of 𝛿1 and 𝛿2, respectively, until the condition in Equation 8 is met. We present an overview of BAPO in Figure 3 and summarize it in Algorithm 1.

Overall, BAPO offers several significant benefits. First, by dynamically adjusting 𝑐high and 𝑐low for each step, we can increase the contribution of positive tokens to the policy-gradient loss while preventing negative tokens from excessively dominating the optimization objective. Second, based on our earlier analysis of the relationship between IS weights and token probabilities in Figure 5, BAPO incorporates more low-probability positive tokens and filters out more low-probability negative

###### Algorithm 1: BAPO

Input: Initialized LLM policy 𝜋𝜃, training dataset D, reward function 𝑅, staleness 𝐸, movable range of clipping bounds [𝑎−, 𝑏−] and [𝑎+, 𝑏+], step size of upper bound 𝛿1, step size of lower bound 𝛿2, positive token contribution threshold 𝜌0

- 1 for step 𝑠 = 1...𝑆 do

- 2 Procedure Sample and filter out responses

- 3 Update the old LLM policy 𝜋𝜃rollout ← 𝜋𝜃 ;
- 4 Sample the 𝑠-th batch D𝑠 from D ;
- 5 Sample 𝐺 responses {𝒚𝑖}𝐺𝑖=1 ∼ 𝜋𝜃rollout(·|𝒙), where 𝒙 ∈ D𝑠 ;
- 6 Compute reward and advantage for each 𝒚𝑖 based on reward function 𝑅 ;
- 7 for staleness = 0...𝐸 do

- 8 Procedure Dynamically adjusting the clipping bounds 𝑐high and 𝑐low

- 9 Initialize clipping bounds 𝑐low = 𝑎− and 𝑐high = 𝑎+ ;
- 10 while the positive token contribution 𝜌 < 𝜌0 and 𝑐low + 𝛿2 ≤ 𝑏−
- 11 do

- 12 if 𝑐high + 𝛿1 ≤ 𝑏+ then

- 13 𝑐high ← 𝑐high + 𝛿1
- 14 else

- 15 𝑐low ← 𝑐low + 𝛿2
- 16 end
- 17 end
- 18 Procedure Update the LLM policy 𝜋𝜃

- 19 Update the LLM policy 𝜋𝜃 by maximizing the following objective:
- 20 𝐽BAPO(𝜃) = 𝔼𝒚∼𝜋 𝜃rollout (·|𝒙)

𝑇 𝑡=1 min(𝑟𝑡 · 𝐴𝑡, clip(𝑟𝑡, 𝑐low, 𝑐high) · 𝐴𝑡)

- 21 end
- 22 end

tokens, both of which contribute to maintaining entropy. Third, by setting the target contribution from positive tokens, BAPO prevents uncontrolled entropy growth, avoids situations where positive tokens overwhelm the loss, and mitigates tail degradation–where the model overfits to easy problems but fails to handle more challenging ones (Ding et al., 2025).

###### 4.3. Analysis

Stable and fast training of BAPO. As shown in Figure 9, BAPO enables a more stable optimization process, characterized by rapidly increasing training rewards, greater contributions from positive tokens, steady gradient normalization, and stable policy entropy–resulting in an improved balance between exploration and exploitation.

We further visualize the adjustment process of the clipping bounds in BAPO. As shown in Figure 8, the averaged upper and lower bounds both fluctuate during training, confirming that BAPO dynamically adjusts the clipping for both types of data and adaptively balances their contributions to the loss. In contrast to approaches such as DAPO (Yu et al., 2025) or the asymmetric clipping in Section 4.1, which rely on empirical tuning, BAPO eliminates the need for complex manual hyperparameter tuning, making it simple yet effective.

Figure 8 | Clipping bounds.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Figure 9 | Training dynamics of BAPO.

Figure 10 | Relationship among token probabilities, importance sampling weights, and entropy.

Effectiveness of BAPO across different staleness. We conduct experiments using the R1-Distill model (Guo et al., 2025) on the SkyWork-OR1-RL dataset (He et al., 2025), with a maximum sequence length of 32𝑘. The results in Figure 11 show that under different data staleness, our method consistently outperforms both the baseline and the clip-higher approach, demonstrating its superiority.

| |
|---|

| |
|---|

| |
|---|

Figure 11 | Results with different data staleness.

The working mechanism of BAPO and its connection to prior work. To better understand the working mechanism of BAPO, we present the relationship among token probabilities, IS weights, and entropy during training in Figure 10. We find that as IS weights deviate further from 1, the corresponding token probabilities decrease, and such low-probability tokens often exhibit higher entropy. Based on this observation, we explain how BAPO relates to prior work. For example, Clip-Higher in Yu et al. (2025) sets the clipping upper bound to 1.28, thereby including more low-probability positive tokens in training, which stabilizes entropy while balancing the contributions of positive and negative tokens. Similarly, Wang et al. (2025a) retain only the top 20% highest-entropy tokens for training, ensuring stable entropy throughout optimization and preserving the model’s exploratory capability, and the target entropy technique in He et al. (2025) plays a similar role, which aligns with our motivation.

#### 5. Experiments and Discussion

###### 5.1. Experimental Setups

Datasets and Models. We use SkyWork-OR1-RL-Data (He et al., 2025) as our RL dataset, as it is widely adopted and of high quality. For evaluation, we employ both the AIME 2024 and the

Table 1 | Main evaluation results.

Model Model Size AIME 2024 AIME 2025 Average ≥ 100B Models and Proprietary Models

Qwen3-235B-A22B (Yang et al., 2025a) 235B 85.7 81.5 83.6 DeepSeek-R1 (Guo et al., 2025) 671B 79.8 70.0 74.9 DeepSeek-R1-0528 (Guo et al., 2025) 671B 91.4 87.5 89.5

- o1medium (Jaech et al., 2024) - 83.3 79.0 81.2
- o3-minimedium (OpenAI, 2025) - 79.6 76.7 78.2
- o3-minihigh (OpenAI, 2025) - 87.3 86.5 86.9 Gemini-2.0Flash-Thinking (Google, 2024) - 73.3 53.5 63.4 Gemini-2.5Flash-Thinking-0520 (Comanici et al., 2025) - 82.3 72.0 77.2

10B - 100B Scale Models Qwen3-30B-A3B (Yang et al., 2025a) 30B − 61.3 − R1-Distill-Qwen-32B (Guo et al., 2025) 32B 72.6 54.9 63.8 QwQ-32B (Qwen, 2025) 32B 79.5 65.3 72.4 Qwen3-32B (Yang et al., 2025a) 32B 81.4 72.9 77.2 SkyWork-OR1-32B (He et al., 2025) 32B 82.2 73.3 77.8 BP-Math-32BSFT 32B 84.4 78.1 81.3 BP-Math-32BGRPO 32B 84.6 78.8 81.7 BP-Math-32BBAPO 32B 87.1 80.0 83.5

≤ 10B Models

R1-Distill-Qwen-7B (Guo et al., 2025) 7B 54.2 38.4 46.3 Light-R1-7B-DS (Wen et al., 2025) 7B 59.1 44.2 51.7 AReaL-boba-RL-7B (Fu et al., 2025) 7B 61.9 48.3 55.1 AceReason-Nemotron-7B (Chen et al., 2025) 7B 69.0 53.6 61.3 SkyWork-OR1-7B (He et al., 2025) 7B 70.2 54.6 62.4 BP-Math-7BSFT 7B 66.9 59.0 62.9 BP-Math-7BGRPO 7B 69.2 59.2 64.2 BP-Math-7BBAPO 7B 70.8 62.5 66.7

newly released AIME 2025 (AIME, 2025) benchmarks. Our experiments cover a range of backbone models, including DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-32B (Guo et al., 2025), and OctoThinker-Llama3.2-3B-Long-Zero (Wang et al., 2025b). In addition, we incorporate two our own supervised fine-tuning (SFT) models, BP-Math-7B and BP-Math-32B, which are derived from Qwen2.5-Math (Yang et al., 2024) through fine-tuning.

Implementation details. We leverage GRPO as the basis for BAPO. Both our preliminary and validation experiments are conducted using DeepSeek-R1-Distill-Qwen-7B, with the maximum response length set to 8𝑘, learning rate to 2 × 10−6, and temperature to 0.6. For main results on BPMath models, we set the maximum response length to 64𝑘 to align with the SFT setting. To introduce staleness, we adopt multiple strategies, including experience reuse through ppo_epoch (Schulman et al., 2017) and the modern partial rollouts (Fu et al., 2025; Team et al., 2025). For BAPO, we set the target contribution 𝜌0 = 0.4, the movable range 𝑎− = 0.6, 𝑏− = 0.9, 𝑎+ = 1.2, 𝑏+ = 3.0, and the step size 𝛿1 = 0.05, 𝛿2 = 0.02. These hyperparameters are not finely tuned, as they already demonstrate strong empirical performance. For evaluation, we report results averaged over 16 rollouts.

Baselines. We include a variety of commercial and open-source models of different scales as baselines, as shown in Table 1, and report their performance as extracted from prior work. In addition, we compare different training approaches, including SFT and GRPO.

- 5.2. Main Results The main results are shown in Figure 1 and Table 1.

Significant performance improvements across models of varying sizes. For strong SFT models, GRPO provides only marginal benefits–for instance, it improves performance by just 0.2 and 0.7 points on AIME24 and AIME25 with the BP-Math-32B model. In contrast, BAPO delivers substantial gains across models of different scales. Specifically, with the BP-Math-32B model, BAPO outperforms SFT by 2.7 and 1.9 points on AIME24 and AIME25, respectively; with the BP-Math-7B model, it achieves even larger improvements of 3.9 and 3.5 points.

SOTA performance over open-source models of comparable sizes and competitive results against proprietary models. Compared to open-source models of similar sizes, our BAPO-trained models achieve state-of-the-art (SOTA) performance. For instance, among 32B models, BP-Math-32BBAPO outperforms Qwen3-32B by 5.7 and 7.1 points on AIME24 and AIME25, respectively, and surpasses SkyWork-OR1-32B by 4.9 and 6.7 points. Among 7B models, BP-Math-7BBAPO also delivers a notable 7.9-point improvement over SkyWork-OR1-7B on AIME25.

Moreover, BP-Math-32BBAPO even outperforms some larger-scale models–for example, it surpasses DeepSeek-R1 by 7.3 and 10.0 points on AIME24 and AIME25, respectively–while achieving performance comparable to o3-mini. Notably, even the smaller BP-Math-7BBAPO yields results on par with Gemini-2.0-Flash-Thinking, underscoring the competitiveness of our approach against commercial models.

- 5.3. Discussion

Partial rollout. To speed up rollouts in LLM reinforcement learning, modern AI infrastructures have introduced several techniques, with partial rollout being particularly noteworthy (Fu et al., 2025; Team et al., 2025). In this approach, long trajectories are split into segments: when a rollout exceeds a fixed token budget, the unfinished portion is stored in a replay buffer and resumed in later iterations instead of being regenerated from scratch. While this improves training efficiency, it also introduces off-policy learning, since different parts of the same trajectory may come from multiple outdated policies. We evaluate BAPO under this setting, as shown in Figure 12. Compared to the baseline GRPO, BAPO exhibits greater robustness to such off-policy infrastructures and achieves more stable optimization.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Figure 12 | Training dynamics with partial rollout.

Results on OctoThinker-Llama3.2-3B-Long-Zero. In addition to the DeepSeek-R1-Distill-Qwen, we also conducted experiments on Llama-based models (Wang et al., 2025b). As shown in Table 2 and Figure 13 in Appendix A, our method achieves more competitive results and exhibits greater stability in training dynamics.

Table 2 | Performance of Llama-based models.

Method AIME 2024 AIME 2025 MATH GRPO 2.5% 2.9% 58.4% BAPO 5.4% 5.8% 66.0%

#### 6. Related Work

Recent landmark models, like OpenAI o1 (Jaech et al., 2024), DeepSeek-R1 (Guo et al., 2025), Gemini 2.5 (Comanici et al., 2025), QwQ (Qwen, 2025), have demonstrated that reinforcement learning can effectively enable long chain-of-thought reasoning in LLMs (Shao et al., 2024; Zhang et al., 2025). Mainstream algorithms include PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024): PPO constrains updates via a clipping-based surrogate objective, while GRPO enhances longhorizon reasoning through group-based rewards.

Despite the remarkable success of RL for LLMs, ensuring stability and efficiency in optimization remains a major challenge (Cui et al., 2025; Yu et al., 2025). Recent studies have sought to better understand the underlying mechanisms of RL and proposed new methods to achieve a balance (Cui et al., 2025; Wang et al., 2025a; Yang et al., 2025b; Zheng et al., 2025). For example, DAPO (Yu et al., 2025) introduces techniques such as Clip-Higher and dynamic sampling to raise the performance ceiling; Wang et al. (2025a) explore optimizing only a small subset of high-entropy tokens for improved efficiency. He et al. (2025), Cui et al. (2025), and other works (Cheng et al., 2025; Liu et al., 2025; Zheng et al., 2025) systematically investigate how to maintain entropy stability during training, thereby preserving the model’s exploration ability. For off-policy RL, Roux et al. (2025) and Arnal et al. (2025) introduce asymmetric clipping mechanisms. The most similar to our work is DCPO (Yang et al., 2025b), which adjusts token-level clipping based on token prior probabilities. However, our approach takes a holistic optimization perspective: we observe the imbalance in loss contributions and derive the Entropy-Clip Rule for the PPO objective, enabling dynamic control over global clipping bounds. We further validate the effectiveness of our method through larger-scale experiments.

#### 7. Conclusion

In this paper, we begin by analyzing the impact of data staleness on model training through both empirical and theoretical studies. We reveal the imbalance between positive and negative samples in RL optimization, and derive as well as empirically validate the Entropy-Clip Rule for PPO-like objectives. Building on these insights, we propose BAPO, which dynamically adjusts the clipping bounds to balance positive and negative samples while preserving the model’s exploratory capability during training. We conduct extensive experiments across different models and settings to validate our method. We hope our work provides key insights for the LLM RL community.

#### References

AIME. Aime problems and solution, 2025. URL https://artofproblemsolving.com/wiki/ index.php/AIME_Problems_and_Solutions.

Anthropic. Claude code, 2025. URL https://docs.anthropic.com/en/docs/claude-code.

Charles Arnal, Gaëtan Narozniak, Vivien Cabannes, Yunhao Tang, Julia Kempe, and Rémi Munos. Asymmetric REINFORCE for off-policy reinforcement learning: Balancing positive and negative rewards. CoRR, abs/2506.20520, 2025. doi: 10.48550/ARXIV.2506.20520. URL https://doi. org/10.48550/arXiv.2506.20520.

Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Hongcheng

Gao, Peizhong Gao, Tong Gao, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Hao Hu, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Chao Hong, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Lijun Lu, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Xinjie Sun, Flood Sung, Heyi Tang, Jiawen Tao, Qifeng Teng, Chensi Wang, Dinglu Wang, Feng Wang, and Haiming Wang. Kimi K2: open agentic intelligence. CoRR, abs/2507.20534, 2025. doi: 10.48550/ARXIV.2507.20534. URL https://doi.org/10.48550/arXiv.2507.20534.

Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. CoRR, abs/2505.16400, 2025. doi: 10.48550/ARXIV.2505.16400. URL https://doi. org/10.48550/arXiv.2505.16400.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. CoRR, abs/2506.14758, 2025. doi: 10.48550/ARXIV.2506.14758. URL https://doi.org/10.48550/arXiv.2506.14758.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit S. Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, NanJiang Jiang, Krishna Haridasan, Ahmed Omran, Nikunj Saunshi, Dara Bahri, Gaurav Mishra, Eric Chu, Toby Boyd, Brad Hekman, Aaron Parisi, Chaoyi Zhang, Kornraphop Kawintiranon, Tania Bedrax-Weiss, Oliver Wang, Ya Xu, Ollie Purkiss, Uri Mendlovic, Ilaï Deutel, Nam Nguyen, Adam Langley, Flip Korn, Lucia Rossazza, Alexandre Ramé, Sagar Waghmare, Helen Miller, Nathan Byrd, Ashrith Sheshan, Raia Hadsell Sangnie Bhardwaj, Pawel Janus, Tero Rissa, Dan Horgan, Sharon Silver, Ayzaan Wahid, Sergey Brin, Yves Raimond, Klemen Kloboves, Cindy Wang, Nitesh Bharadwaj Gundavarapu, Ilia Shumailov, Bo Wang, Mantas Pajarskas, Joe Heyward, Martin Nikoltchev, Maciej Kula, Hao Zhou, Zachary Garrett, Sushant Kafle, Sercan Arik, Ankita Goel, Mingyao Yang, Jiho Park, Koji Kojima, Parsa Mahmoudieh, Koray Kavukcuoglu, Grace Chen, Doug Fritz, Anton Bulyenov, Sudeshna Roy, Dimitris Paparas, Hadar Shemtov, Bo-Juen Chen, Robin Strudel, David Reitter, Aurko Roy, Andrey Vlasov, Changwan Ryu, Chas Leichner, Haichuan Yang, Zelda Mariet, Denis Vnukov, Tim Sohn, Amy Stuart, Wei Liang, Minmin Chen, Praynaa Rawlani, Christy Koh, JD Co-Reyes, Guangda Lai, Praseem Banzal, Dimitrios Vytiniotis, Jieru Mei, and Mu Cai. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR, abs/2507.06261, 2025. doi: 10.48550/ARXIV.2507.06261. URL https://doi.org/10.48550/arXiv.2507.06261.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models. CoRR, abs/2505.22617, 2025. doi: 10.48550/ARXIV.2505.22617. URL https://doi. org/10.48550/arXiv.2505.22617.

Yiwen Ding, Zhiheng Xi, Wei He, Lizhuoyuan Lizhuoyuan, Yitao Zhai, Shi Xiaowei, Xunliang Cai, Tao Gui, Qi Zhang, and Xuanjing Huang. Mitigating tail narrowing in LLM self-improvement via socratic-guided sampling. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings

of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, pages 10627–10646. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.NAACL-LONG.533. URL https://doi.org/10. 18653/v1/2025.naacl-long.533.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, Tongkai Yang, Binhang Yuan, and Yi Wu. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. CoRR, abs/2505.24298, 2025. doi: 10. 48550/ARXIV.2505.24298. URL https://doi.org/10.48550/arXiv.2505.24298.

##### Google. Introducing gemini 2.0: our new ai model for the agentic era, December 2024. URL https://blog.google/technology/google-deepmind/ google-gemini-ai-update-december-2024/.

Çaglar Gülçehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (rest) for language modeling. CoRR, abs/2308.08998, 2023. doi: 10.48550/ARXIV.2308.08998. URL https://doi.org/10. 48550/arXiv.2308.08998.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025. doi: 10.1038/s41586-025-09422-z. URL https://doi.org/10.1038/s41586-025-09422-z.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang

Liu, and Yahui Zhou. Skywork open reasoner 1 technical report. CoRR, abs/2505.22312, 2025. doi: 10.48550/ARXIV.2505.22312. URL https://doi.org/10.48550/arXiv.2505.22312.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, and Ilge Akkaya. Openai o1 system card. CoRR, abs/2412.16720, 2024. doi: 10.48550/ARXIV.2412.16720. URL https://doi.org/10.48550/arXiv.2412.16720.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. CoRR, abs/2503.20783, 2025. doi: 10.48550/ARXIV.2503.20783. URL https://doi.org/10.48550/arXiv.2503.20783.

##### OpenAI. Openai o3-mini system card, 2025. URL https://cdn.openai.com/ o3-mini-system-card-feb10.pdf.

Qwen. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https:// qwenlm.github.io/blog/qwq-32b/.

Nicolas Le Roux, Marc G. Bellemare, Jonathan Lebensold, Arnaud Bergeron, Joshua Greaves, Alexandre Fréchette, Carolyne Pelletier, Eric Thibodeau-Laufer, Sándor Tóth, and Sam Work. Tapered offpolicy REINFORCE: stable and efficient reinforcement learning for llms. CoRR, abs/2503.14286, 2025. doi: 10.48550/ARXIV.2503.14286. URL https://doi.org/10.48550/arXiv.2503. 14286.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707. 06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi. org/10.48550/arXiv.2402.03300.

Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Yuan Cao, Eugene Tarassov, Rémi Munos, Bernardo Ávila Pires, Michal Valko, Yong Cheng, and Will Dabney. Understanding the performance gap between online and offline alignment algorithms. CoRR, abs/2405.08448, 2024. doi: 10.48550/ARXIV.2405.08448. URL https://doi.org/10.48550/arXiv.2405.08448.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025. doi: 10.48550/ARXIV.2501. 12599. URL https://doi.org/10.48550/arXiv.2501.12599.

Luong Quoc Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 7601–7614. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.410. URL https://doi.org/10.18653/v1/2024.acl-long.410.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. CoRR, abs/2506.01939, 2025a. doi: 10. 48550/ARXIV.2506.01939. URL https://doi.org/10.48550/arXiv.2506.01939.

Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling. CoRR, abs/2506.20512, 2025b. doi: 10.48550/ARXIV.2506.20512. URL https://doi.org/10.48550/arXiv.2506.20512.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng Zhang. Light-r1: Curriculum sft, DPO and RL for long COT from scratch and beyond. CoRR, abs/2503.10460, 2025. doi: 10.48550/ARXIV.2503.10460. URL https://doi.org/10.48550/arXiv.2503.10460.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn., 8:229–256, 1992. doi: 10.1007/BF00992696. URL https://doi.org/ 10.1007/BF00992696.

Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, and Xuanjing Huang. Training large language models for reasoning through reverse curriculum reinforcement learning. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=t82Y3fmRtk.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren,

and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via selfimprovement. CoRR, abs/2409.12122, 2024. doi: 10.48550/ARXIV.2409.12122. URL https: //doi.org/10.48550/arXiv.2409.12122.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, 2025a. doi: 10.48550/ARXIV.2505.09388. URL https://doi.org/10.48550/arXiv.2505.09388.

Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, and Rihui Xin. Dcpo: Dynamic clipping policy optimization. arXiv preprint arXiv:2509.02333, 2025b.

Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, and Yunjian Xu. Do not let low-probability tokens over-dominate in RL for llms. CoRR, abs/2505.12929, 2025c. doi: 10.48550/ARXIV.2505.12929. URL https://doi.org/10.48550/arXiv.2505.12929.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an open-source LLM reinforcement learning system at scale. CoRR, abs/2503.14476, 2025. doi: 10.48550/ARXIV. 2503.14476. URL https://doi.org/10.48550/arXiv.2503.14476.

Ruifeng Yuan, Chenghao Xiao, Sicong Leng, Jianyu Wang, Long Li, Weiwen Xu, Hou Pong Chan, Deli Zhao, Tingyang Xu, Zhongyu Wei, Hao Zhang, and Yu Rong. Vl-cogito: Progressive curriculum reinforcement learning for advanced multimodal reasoning. CoRR, abs/2507.22607, 2025. doi: 10.48550/ARXIV.2507.22607. URL https://doi.org/10.48550/arXiv.2507.22607.

Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, et al. A survey of reinforcement learning for large reasoning models. arXiv preprint arXiv:2509.08827, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. CoRR, abs/2507.18071, 2025. doi: 10.48550/ARXIV.2507.18071. URL https://doi.org/10. 48550/arXiv.2507.18071.

Xuekai Zhu, Daixuan Cheng, Dinghuai Zhang, Hengli Li, Kaiyan Zhang, Che Jiang, Youbang Sun, Ermo Hua, Yuxin Zuo, Xingtai Lv, et al. Flowrl: Matching reward distributions for llm reasoning. arXiv preprint arXiv:2509.15207, 2025.

## Appendix

#### A. Performance on OctoThinker-Llama

We illustrate the training dynamics on OctoThinker-Llama in Figure 13. Since Llama family models behave badly in RL training, we choose the model after mid-training (Wang et al., 2025b) to show the robustness of BAPO. We can find that BAPO provides consistent and significant improvement in training. For training details, we set the low bound as 0.8-0.9, high bound as 1.2-2.0, and target positive loss contribution as 0.45.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

Figure 13 | Training dynamics of OctoThinker-Llama-3B-Long-Zero.

#### B. Proofs of Equation 6

###### B.1. Explanations for all variables and expressions

All notation used in the following justification, including variables and expressions, is provided with detailed explanations in Table 3.

###### B.2. Preparation: Rewrite the PPO derivatives

To facilitate the justification of the propositions below, we rewrite the PPO loss function in the following form:

where

###### ∇𝐽PPO = ∑︁

𝜋𝜃(𝑦𝑡) · 𝕀{𝑟(𝑦𝑡) < 1 + 𝜀} · 𝐴(𝑦𝑡) · ∇ log𝜋𝜃(𝑦𝑡)

𝐴(𝑦𝑡)>0

positive tokens

###### + ∑︁

𝜋𝜃(𝑦𝑡) · 𝕀{𝑟(𝑦𝑡) > 1 − 𝜀} · 𝐴(𝑦𝑡) · ∇ log𝜋𝜃(𝑦𝑡)

𝐴(𝑦𝑡)<0

negative tokens

𝜋𝜃(𝑦𝑡 | 𝒙, 𝒚<𝒕) 𝜋𝜃rollout(𝑦𝑡 | 𝒙, 𝒚<𝒕)

𝜋𝜃(𝑦𝑡) = 𝜋𝜃(𝑦𝑡|𝒙, 𝒚<𝒕) , 𝑟(𝑦𝑡) =

, 𝐴(𝑦𝑡) = 𝐴(𝑦𝑡|𝒙, 𝒚<𝒕) .

Table 3 | Notation used in justification below.

Category Symbol Meaning

Variables

𝜋𝜃 The policy parameterized by 𝜃 𝜋𝜃𝑟𝑜𝑙𝑙𝑜𝑢𝑡 The standard sampling policy 𝒙 Given prompt 𝒚 A T-token response generated by 𝜋𝜃 when given 𝒙

𝑦𝑡 The t-th token of y 𝜂 Learning rate

𝜋𝜃(·|𝒙, 𝒚<𝑡) Probability of generating token · under policy 𝜋𝜃 given input 𝒙 and previous tokens 𝒚<𝑡

𝜋𝜃𝑟𝑜𝑙𝑙𝑜𝑢𝑡(𝑦𝑡|𝒙, 𝒚<𝑡) Probability of generating token · under standard sampling policy 𝜋𝜃𝑟𝑜𝑙𝑙𝑜𝑢𝑡 given input 𝒙 and previous tokens 𝒚<𝑡

𝐴(·|𝒙, 𝒚<𝒕) The measurement of how much better(or worse) selecting token · is compared to the expected value under the current policy, given 𝒙 and 𝒚<𝒕

H(·|𝒙, 𝒚<𝑡) The information entropy of policy · given 𝒙 and 𝒚<𝑡

𝐶𝑜𝑣𝑦𝑡∼𝜋𝜃(·|𝒙,𝒚<𝑡)(𝑎(𝑦𝑡), 𝑏(𝑦𝑡)) The expected covariance of 𝑎(𝑦𝑡) and 𝑏(𝑦𝑡) over 𝑦𝑡 sampled from the policy 𝜋𝜃, given 𝒙 and 𝒚<𝑡

𝕀(𝑎 = 𝑏) Indicator function that equals 1 if 𝑎 = 𝑏 and 0 otherwise

Expressions

𝑄(𝜋𝜃)(·, 𝒙) The expected cumulative reward obtained by taking token · given input 𝒙 and previous tokens under policy 𝜋𝜃 𝑉(𝜋𝜃)(𝒙) The expected return of the new taking token given input 𝒙 and previous tokens under policy 𝜋𝜃

𝑧𝒚,𝒙 A quantity representing the cumulative weight of sequence 𝒚 given input 𝒙 under policy 𝜋𝜃, reflecting its contribution to the policy taken at the current optimization step

∇𝜃𝑦𝑡,𝒙 𝐽(𝜃) The gradient of the policy taken with respect to the logit parameter 𝜃𝑦𝑡,𝒙, representing how the policy 𝜋𝜃 should be adjusted for token 𝑦𝑡 given input 𝒙

###### B.3. Proofs of the main Propositions

The following derivation is inspired by the proof framework in Cui et al. (2025). While the original work focuses mainly on the basic gradient formulation of naive REINFORCE to provide a heuristic explanation, our study advances this approach by deriving the gradient expression specific to the PPO objective. This refinement offers a specific, intuitive yet theoretical account of how policy entropy is intrinsically shaped by the interaction between token-level advantages and their sampling

probabilities.

- B.3.1. Preclaims Proofs of these three lemmas below are available in Cui et al. (2025).

- Lemma 1. Let the actor policy 𝜋𝜃 be a tabular softmax policy, the difference of information entropy given prompt 𝑥 between two consecutive steps 𝑘 and 𝑘 + 1 satisfies

H(𝜋𝜃𝑘+1|𝒙, 𝒚<𝒕) − H(𝜋𝜃𝑘|𝒙, 𝒚<𝒕) ≈ −Cov𝑦

𝑡∼𝜋𝜃𝑘(·|𝒙,𝒚<𝒕) log𝜋𝜃𝑘(𝑦𝑡), 𝑧𝑘𝒚+,𝒙1 − 𝑧𝑘𝒚,𝒙 .

- Lemma 2 (Derivative of softmax function). 𝜕log𝜋𝜃(𝑦𝑡)

𝜕𝜃𝑦′

𝑡 ,𝒙

= 𝕀{𝑦𝑡 = 𝑦𝑡′} − 𝜋𝜃(𝑦𝑡′)

- Lemma 3 (Expectation of Advantage function given prompt 𝑥).

𝔼𝑦

𝑡∼𝜋𝜃(·|𝑥,𝒚<𝒕) 𝐴𝜋𝜃(𝑦𝑡) = 𝔼𝑦

𝑡∼𝜋𝜃(·|𝑥,𝒚<𝒕) 𝑄𝜋𝜃(𝑦𝑡, 𝒙) − 𝑉𝜋𝜃(𝒙)

= 𝔼𝑦

𝑡∼𝜋𝜃(·|𝑥,𝒚<𝒕) 𝑄(𝑦𝑡, 𝒙) − 𝔼𝑦

𝑡∼𝜋𝜃(·|𝑥,𝒚<𝒕) 𝑉(𝒙)

= 𝑉(𝒙) − 𝑉(𝒙)

= 0

- B.3.2. Principle Propositions

- Proposition 1: Assume the actor policy 𝜋𝜃 follows a tabular softmax policy and is optimized via the PPO objective, the difference of 𝑧𝒚,𝒙 between two consecutive steps k and k+1 satisfies

𝑧𝑘𝒚+,𝒙1 − 𝑧𝑘𝒚,𝒙 = 𝜂 · 𝜋𝜃(𝑦𝑡) · [𝐴(𝑦𝑡) · X(𝑦𝑡) + 𝐶],

where

 

1, if 𝐴(𝑦𝑡) > 0 & 𝑟(𝑦𝑡) < 1 + 𝜖

or 𝐴(𝑦𝑡) < 0 & 𝑟(𝑦𝑡) > 1 − 𝜖 0, otherwise

X(𝑦𝑡) =

 

and 𝐶 includes all clauses irrelevant to 𝑦𝑡.

###### It is worth noting that X(𝒚𝒕) = 0 if and only if 𝒚𝒕 is clipped.

Proof. In tabular softmax policy, each trajectory-prompt pair (𝒚, 𝒙) is associated with an individual logit parameter 𝑧𝒚,𝒙 = 𝜃𝑦𝑡,𝒙. Through gradient backtracking, 𝑧𝒚,𝒙 is updated via 𝑧𝑘+1

𝒚,𝒙 = 𝑧𝑘𝒚,𝒙 + 𝜂 · ∇𝜃𝑦𝑡,𝒙 𝐽(𝜃). According to the loss function of PPO, we have

𝑧𝑘𝒚+,𝒙1 − 𝑧𝑘𝒚,𝒙 = 𝜂 · ∇𝜃𝑦𝑡,𝒙 𝐽𝑃𝑃𝑂(𝜃) = 𝜂 · 𝔼𝑦𝑡′∼𝜋𝜃(·|𝒙,𝒚<𝒕)

𝕀{𝑟(𝑦𝑡′) < 1 + 𝜀} · ∇𝜃𝑦𝑡,𝒙

log𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

𝐴(𝑦𝑡′)>0

𝕀{𝑟(𝑦𝑡′) > 1 − 𝜀} · ∇𝜃𝑦𝑡,𝒙

log𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

+ 𝜂 · 𝔼𝑦𝑡′∼𝜋𝜃(·|𝒙,𝒚<𝒕)

𝐴(𝑦𝑡′)<0

log𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

= 𝜂 · 𝔼𝑦𝑡′∼𝜋𝜃(·|𝒙,𝒚<𝒕) ∇𝜃𝑦𝑡,𝒙

1 − 𝜂 · 𝔼𝑦𝑡′∼𝜋𝜃(·|𝒙,𝒚<𝒕)

𝕀{𝑟(𝑦𝑡′) > 1 + 𝜀} · ∇𝜃𝑦𝑡,𝒙

log𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

𝐴(𝑦𝑡′)>0

2 − 𝜂 · 𝔼𝑦𝑡′∼𝜋𝜃(·|𝒙,𝒚<𝒕)

𝕀{𝑟(𝑦𝑡′) < 1 − 𝜀} · ∇𝜃𝑦𝑡.𝒙

log𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

𝐴(𝑦𝑡′)<0

3

= 1 − ( 2 + 3 ) (8)

We first perform the derivation on the term marked as 1 :

𝜕log𝜋𝜃(𝑦𝑡′) 𝜕𝜃𝑦𝑡,𝒙 · 𝐴(𝑦𝑡′)

1 = 𝜂 · 𝔼𝑦𝑡′∼𝜋𝜃(·|𝒙,𝒚<𝒕)

- Lemma= 2 𝜂 · ∑︁

𝑦𝑡′

𝜋𝜃(𝑦𝑡′) · (𝕀{𝑦𝑡′ = 𝑦𝑡} − 𝜋𝜃(𝑦𝑡)) · 𝐴(𝑦𝑡′)

= 𝜂 · 𝜋𝜃(𝑦𝑡) ·





(1 − 𝜋𝜃(𝑦𝑡)) · 𝐴(𝑦𝑡) − ∑︁ 𝑦𝑡′≠𝑦𝑡

𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

  

= 𝜂 · 𝜋𝜃(𝑦𝑡) ·

  

𝐴(𝑦𝑡) − ∑︁

𝑦𝑡′

𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

 

- Lemma= 3 𝜂 · 𝜋𝜃(𝑦𝑡) · [𝐴(𝑦𝑡) − 0] 

= 𝜂 · 𝜋𝜃(𝑦𝑡) · 𝐴(𝑦𝑡)

To keep the presentation concise, we provide only the resulting derivations of Term 2 and 3 , as the detailed steps follow similarly to those for Term 1 .

2 + 3 = 𝜂 · 𝜋𝜃(𝑦𝑡) · 𝐴(𝑦𝑡) · (1 − X(𝑦𝑡))

− 𝜂 · 𝜋𝜃(𝑦𝑡) · ∑︁

𝕀{𝑟(𝑦𝑡′) > 1 + 𝜀} · 𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

𝐴(𝑦𝑡′)>0

###### − 𝜂 · 𝜋𝜃(𝑦𝑡) · ∑︁

𝕀{𝑟(𝑦𝑡′) < 1 − 𝜀} · 𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

𝐴(𝑦𝑡′)<0

By substituting the results of the above derivation into Clause (8), we observe that:

(8) = 1 − ( 2 + 3 )

= 𝜂 · 𝜋𝜃(𝑦𝑡) · 𝐴(𝑦𝑡) · X(𝑦𝑡)

+ ∑︁

𝕀{𝑟(𝑦𝑡′) > 1 + 𝜀} · 𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

𝐴(𝑦𝑡′)>0

+ ∑︁

𝕀{𝑟(𝑦𝑡′) < 1 − 𝜀} · 𝜋𝜃(𝑦𝑡′) · 𝐴(𝑦𝑡′)

𝐴(𝑦𝑡′)<0

By grouping all elements unrelated to 𝑦𝑡 into 𝐶, we are able to successfully establish our proposition.

###### □

Building on Proposition 1, we establish the relationship between policy entropy and the covariance of specific tokens, which is stated as Proposition 2 below.

- Proposition 2 (Equation 6): Let the actor policy 𝜋𝜃 be tabular softmax policy, and 𝜋𝜃 is updated via PPO objective, the difference of information entropy given prompt 𝑥 and trajectory part 𝑦<𝑡 between two consecutive steps k and k+1 satisfies

H(𝜋𝜃𝑘+1|𝒙, 𝒚<𝒕) − H(𝜋𝜃𝑘|𝒙, 𝒚<𝒕) ≈ −𝜂 · Cov𝑦

𝑡∼𝜋𝜃𝑘(·|𝒙,𝒚<𝒕) log𝜋𝜃𝑘(𝑦𝑡), 𝐴(𝑦𝑡) · X(𝑦𝑡) + 𝐶 .

Proof. Leveraging the conclusions of Lemma 1 and Proposition 1, we find that, under policy optimization and iteration via the PPO algorithm, the following relationship is satisfied:

𝑧𝑘𝒚+,𝒙1 − 𝑧𝑘𝒚,𝒙 = 𝜂 · (𝐴(𝑦𝑡) · X(𝑦𝑡) + 𝐶). Applying this into Lemma 1, we have

H(𝜋𝜃𝑘+1|𝒙, 𝒚<𝒕) − H(𝜋𝜃𝑘|𝒙, 𝒚<𝒕) ≈ −𝜂 · Cov𝑦

𝑡∼𝜋𝜃𝑘(·|𝒙,𝒚<𝒕) log𝜋𝜃𝑘(𝑦𝑡), 𝐴(𝑦𝑡) · X(𝑦𝑡) + 𝐶 .

□

###### B.4. Analysis

- B.4.1. Direct Analysis: Why Varying 𝜀 Alters Entropy? We begin by examining the covariance of the clipped token, denoted as 𝛼.

Based on the observation stated above, the contribution of 𝛼 to the entropy can be expressed as:

−𝜂 · 𝜋𝜃𝑘(𝛼) · Cov log𝜋𝜃𝑘(𝛼), 𝐶 = 0,

which indicates that only the retained tokens contribute to the overall entropy.

In other words, we manipulate the number of tokens that can contribute to the entropy by altering the parameter 𝜀.

- B.4.2. Advanced Analysis : Which Type of Tokens Matter Most for Entropy? To understand how individual tokens contribute to the overall entropy, we first revisit the Proposition
- B.3.2 established above. In this section, we provide a more precise definition of tokens with low/high probabilities and advantages. It should be noted that in the analysis experiment (Figure 5), we adopt the naive REINFORCE algorithm without clipping. Consequently, tokens with high or low advantages are defined according to the sign of their advantage values, i.e., > 0 for high advantage and < 0 for low advantage.

H(𝜋𝜃𝑘+1|𝒙, 𝒚<𝒕) − H(𝜋𝜃𝑘|𝒙, 𝒚<𝒕) ≈ −𝜂 · Cov𝑦

𝑡∼𝜋𝜃𝑘(·|𝒙,𝒚<𝒕) log𝜋𝜃𝑘(𝑦𝑡), 𝐴(𝑦𝑡) · X(𝑦𝑡) + 𝐶

∑︁𝑇

𝑖∼𝜋𝜃𝑘(·|𝒙,𝒚<𝒕)[log𝜋𝜃𝑘(𝑦𝑖)] · 𝐴(𝑦𝑝) · X(𝑦𝑝) − 𝔼𝑦

𝜋𝜃𝑘(𝑦𝑝|𝒙, 𝒚<𝒕) · log𝜋𝜃𝑘(𝑦𝑝) − 𝔼𝑦

= −𝜂 ·

𝑝=1

𝑖∼𝜋𝜃𝑘(·|𝒙,𝒚<𝒕)[𝐴(𝑦𝑖) · X(𝑦𝑖)] .

where T is the size of the dictionary. For convenience, we denote 𝔼𝑦

𝑖∼𝜋𝜃𝑘(·|𝒙,𝒚<𝒕). As only retained tokens contribute to the entropy, we focus only on tokens that are not clipped. We begin by making the following simplification:

as 𝔼𝑦

𝑖

𝑖(𝐴(𝑦𝑖) · X(𝑦𝑖)) = 𝔼𝑦clipped(𝐴(𝑦𝑖) · 0) + 𝔼𝑦retained(𝐴(𝑦𝑖) · 1) = 𝔼𝑦retained(𝐴(𝑦𝑖)) . So for a selected token 𝑦𝑠, its contribution to the overall entropy can be expressed as:

𝔼𝑦

−𝜂 · 𝜋𝜃(𝑦𝑠) · (log𝜋𝜃(𝑦𝑠) − 𝔼𝑦

𝑖(log𝜋𝜃(𝑦𝑖))) · (𝐴(𝑦𝑠) − 𝔼𝑦retained𝐴(𝑦retained)).

Next, we analyze how different types of tokens contribute to the overall entropy. To avoid ambiguity, we first give strict definitions that distinguish between tokens with high/low probabilities and tokens with high/low advantages.

Definition 1. For a token 𝑦𝑠, we classify it as follows:

- • High advantage: if 𝐴(𝑦𝑠) > 𝔼𝑦retained𝐴(𝑦retained)

Otherwise, it is called low advantage.

- • High probability: if

𝑖(log𝜋𝜃(𝑦𝑖))) Otherwise, it is called low probability.

𝜋𝜃(𝑦𝑠) > exp 𝔼𝑦

Secondly, we present two propositions that directly follow from the above definitions.

- Proposition 3. For a token 𝑦𝑠, we have

 

> 0, if 𝑦𝑠 is a high-advantage token, < 0, if 𝑦𝑠 is a low-advantage token.

𝐴(𝑦𝑠) − 𝔼𝑦retained𝐴(𝑦retained)

 

- Proposition 4. For a token 𝑦𝑠, we have

 

> 0, if 𝑦𝑠 is a high-probability token, < 0, if 𝑦𝑠 is a low-probability token.

𝜋𝜃(𝑦𝑠) · (log𝜋𝜃(𝑦𝑠) − 𝔼𝑦

𝑖(log𝜋𝜃(𝑦𝑖)))

 

Proof. Let us denote

𝑖(log𝜋𝜃(𝑦𝑖))),

𝐶 = 𝔼𝑦

which is independent of 𝑦𝑠, and let 𝑥 = 𝜋𝜃(𝑦𝑠). As 𝜋𝜃(𝑦) < 1 for every 𝑦, 𝐶 < 0. Consider the function

𝑓 (𝑥) = 𝑥 · (log(𝑥) − 𝐶). Figure 14 illustrates the behavior of this function.

y

C

0 eC 1 eC 1 eC 1

x

Figure 14 | Graph of the function 𝑓 (𝑥) = 𝑥(log 𝑥 − 𝐶).

The proposition follows directly from the properties of 𝑓 (𝑥) as observed in the figure. □

Due to the propositions given above, we have the table below:

ΔH(𝑦𝑠) ≈ −𝜂 · 𝜋𝜃(𝑦𝑠) · (log𝜋𝜃(𝑦𝑠) − 𝔼𝑦

𝑖(log𝜋𝜃(𝑦𝑖)))

· (𝐴(𝑦𝑠) − 𝔼𝑦retained𝐴(𝑦retained)

4

5

Table 4 | Influence of token characteristics on ΔH(𝑦𝑠). The “prob” denotes the probability 𝜋𝜃(𝑦𝑠), and the “adv” represents the advantage 𝐴(𝑦𝑠).

###### Token properties 4 5 𝚫H(𝒚𝒔) (−𝜼 · 4 · 5 )

high prob, high adv > 0 > 0 < 0 high prob, low adv > 0 < 0 > 0 low prob, high adv < 0 > 0 > 0 low prob, low adv < 0 < 0 < 0

It should be noted that a token 𝑦𝑠 decreases the entropy if ΔH(𝑦𝑠) < 0, and increases it otherwise.

Therefore, we observe that tokens which are positive with high probabilities and high advantages, or negative with low probabilities and low advantages, contribute to a reduction in the overall entropy. Conversely, positive tokens with high probabilities but low advantages, and negative tokens with high probabilities but low advantages, contribute to an increase in the overall entropy. This observation justifies the statement made in the main part of the thesis.

