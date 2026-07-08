## arXiv:2407.14622v1[cs.LG]19Jul2024

[Figure 1]

# BOND: Aligning LLMs with Best-of-N Distillation

###### Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Nino Vieillard, Alexandre Ramé, Bobak Shariari, Sarah Perrin, Abe Friesen, Geoffrey Cideron, Sertan Girgin, Piotr Stanczyk, Andrea Michi, Danila Sinopalnikov, Sabela Ramos, Amélie Héliou, Aliaksei Severyn, Matt Hoffman, Nikola Momchev, Olivier Bachem

Google DeepMind

Reinforcement learning from human feedback (RLHF) is a key driver of quality and safety in state-ofthe-art large language models. Yet, a surprisingly simple and strong inference-time strategy is Best-of-N sampling that selects the best generation among 𝑁 candidates. In this paper, we propose Best-of-N Distillation (BOND), a novel RLHF algorithm that seeks to emulate Best-of-N but without its significant computational overhead at inference time. Specifically, BOND is a distribution matching algorithm that forces the distribution of generations from the policy to get closer to the Best-of-N distribution. We use the Jeffreys divergence (a linear combination of forward and backward KL) to balance between mode-covering and mode-seeking behavior, and derive an iterative formulation that utilizes a moving anchor for efficiency. We demonstrate the effectiveness of our approach and several design choices through experiments on abstractive summarization and Gemma models. Aligning Gemma policies with BOND outperforms other RLHF algorithms by improving results on several benchmarks.

Keywords: LLM, Alignment, RLHF, Best-of-N

#### 1. Introduction

State-of-the-art large language models (LLMs) such as Gemini (Gemini Team, 2023; Reid et al., 2024) and GPT-4 (OpenAI, 2023) are generally trained in three stages. First, LLMs are pre-trained on large corpora of knowledge using next-token prediction (Radford et al., 2018, 2019). Second, the pre-trained models are fine-tuned to follow instructions via supervised fine-tuning (SFT) (Raffel et al., 2020; Wei et al., 2022). Lastly, reinforcement learning from human feedback (RLHF) (Christiano et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020) is used to further increase the quality of generations. The RLHF step generally consists of learning a reward model (RM) (Ouyang et al., 2022) on human preferences and then optimizing the LLM to maximize predicted rewards using reinforcement learning algorithms.

RLHF algorithms and their challenges. Fine-tuning LLMs with reinforcement learning (RL) is challenging (Casper et al., 2023), notably since it can cause forgetting (French, 1992) of pre-trained knowledge, and since loopholes in the RM (Clark and Amodei, 2016; Pan et al., 2022) can cause reward hacking (Askell et al., 2021; Skalse et al., 2022). The standard strategy is to use policy-gradient methods (Williams, 1992) with KL regularization towards the SFT policy. Those RL algorithms seek Pareto-optimal policies with high reward at low KL, to preserve the general capabilities of the original model and tackle the misalignment (Ngo et al., 2022) concerns.

Best-of-N sampling. In practice, a surprisingly simple inference-time approach is often used to improve the quality of generations: Best-of-N sampling (Stiennon et al., 2020). It consists of drawing 𝑁 candidate generations from the reference (typically, supervised fine-tuned) model and selecting the one with the highest reward according to the RM. This strategy empirically achieves excellent reward-KL trade-offs (Nakano et al., 2021; Gao et al., 2023; Touvron et al., 2023) but increases the computational cost by a factor of 𝑁.

Corresponding author: piergs@google.com

[Figure 2]

- Figure 1 | Best-of-N is an inference-time strategy that selects the best generation among 𝑁 candidates from a reference LLM policy, improving quality at the cost of a large computational (need to sample and score 𝑁 times from the model). In contrast, the proposed BOND approach aims at obtaining a fine-tuned policy that can directly sample the Best-of-N generation. This would inherit the quality of Best-of-N sampling, while requiring a single sample at inference time. We achieve this by distilling the Best-of-N strategy into the policy via online distribution matching.

BOND. In this paper, we propose BOND (Best-of-N Distillation), a novel RLHF algorithm that learns a policy that achieves the strong performance of Best-of-N sampling but, crucially, requires only a single sample at inference time, as depicted in Figure 1. Our key idea is to cast the alignment of the policy as a distribution matching problem, where we fine-tune the policy to emulate the Best-of-N distribution. To achieve this, we first derive an analytical expression for the Best-of-N distribution. This allows us to consider and optimize different divergence metrics. We first show how to minimize the forward KL divergence using samples from the Best-of-N strategy, leading to a standard imitation learning setup with a mode covering behavior. We also show how to minimize the backward KL, leading to a new form of quantile-based advantage, which does not depend on the reward scale, and corresponds to a mode seeking behavior. Then, we propose to minimize a linear combination of forward and backward KL, also known as Jeffreys divergence, which retains the best of both approaches. Furthermore, to optimize performance while keeping a reduced sample-complexity, we propose an iterative BOND approach which consists of iteratively distilling the Best-of-N of a moving anchor policy. Finally, based on the aforementioned ideas, we propose J-BOND (J for Jeffreys), a novel, stable, efficient and practical RLHF algorithm to align LLMs.

Experiments. We first demonstrate the effectiveness of BOND and of our design choices on the abstractive summarization XSum (Narayan et al., 2018) task. Then, in Section 6, we apply J-BOND to align Gemma (Gemma Team, 2024) policies. J-BOND not only improves the KL-reward Pareto front compared to standard RL algorithms, but also enhances performance on academic benchmarks and side-by-side comparisons against open-source variants such as Mixtral (Jiang et al., 2024).

#### 2. Problem Setup

We consider a LLM based on the transformer (Vaswani et al., 2017) architecture, defining a policy 𝜋(𝑥, ·) by auto-regressively generating token sequences 𝑦 from the prompt 𝑥. Given a pre-trained and typically supervised fine-tuned reference policy 𝜋ref, we seek to further align it to human preferences. To achieve this, throughout the rest of the paper we assume access to a reward model (RM) which we denote as 𝑟(·), trained to reflect human preferences.

Standard RLHF. Most RL algorithms optimize a linear combination of the expected reward and a KL

divergence between the current and reference policy:

𝜋RL = argmax𝜋 𝔼𝜋[𝑟(𝑦)] − 𝛽RL · KL(𝜋 || 𝜋ref), (1) with regularization strength 𝛽RL ≥ 0. This KL regularization forces the policy to remain close to its initialization 𝜋ref (Geist et al., 2019; Lazaridou et al., 2020), reducing forgetting (French, 1992) and reward hacking (Skalse et al., 2022). Equation (1) is usually optimized with online algorithms, as they perform better than their offline counterparts (Tang et al., 2024). Moreover, simple methods have demonstrated the best results, e.g., REINFORCE (Williams, 1992) with a sampled baseline for variance reduction (Li et al., 2023; Ahmadian et al., 2024) outperform PPO (Schulman et al., 2017).

Best-of-N. A complementary alignment strategy is Best-of-N, which is an inference-time strategy that involves sampling multiple times from 𝜋ref and selecting the generation with highest reward according to the RM 𝑟. In contrast to RLHF strategies, Best-of-N does not fine-tune the weights of the LLM, but instead modifies the inference procedure. Best-of-N was empirically shown to be efficient (Touvron et al., 2023) when looking at reward/KL trade-offs, and comes with theoretical guarantees (Qiping Yang et al., 2024) in terms of Pareto-optimality. Unfortunately, Best-of-N comes at a significantly higher inference cost which increases linearly with 𝑁, since producing 𝑁 generations is (in general) 𝑁 times more costly than sampling a single one.

Motivated by the above considerations, we propose a novel alignment method which we name BOND for Best-of-N Distillation. The goal of BOND is to distill the Best-of-N strategy into the policy. This allows the policy to reach the strong performance of Best-of-N sampling, while requiring only a single sample at inference time. We outline our overall approach in the next section.

### 3. The BOND Approach

We formulate the BOND approach in two main steps. First, we derive an analytical expression for the Best-of-N distribution (Section 3.1). Second, using the derived expression, we phrase the problem as a distribution matching problem (Section 3.2), i.e., we want to steer the policy closer to the Best-of-N distribution. In Section 3.3, we draw insightful connections between BOND and standard RLHF.

##### 3.1. The Best-of-N distribution

In this section, we derive the exact analytical distribution of Best-of-N sampling and study its properties. For simplicity, we drop the context 𝑥 from all notation without loss of generality and assume that the reward 𝑟(𝑦) induces a strict ordering on all generations 𝑦1. We can affirm the following main theorem (proof in Appendix A.1).

Theorem 1. For any generation 𝑦, let

𝑝<(𝑦) = ℙ𝑦′∼𝜋ref[𝑟(𝑦′) < 𝑟(𝑦)] (2) denote the probability that a random generation 𝑦′ from 𝜋ref is strictly worse than 𝑦 and let

𝑝≤(𝑦) = ℙ𝑦′∼𝜋ref[𝑟(𝑦′) ≤ 𝑟(𝑦)], (3) the probability that 𝑦′ is not better than 𝑦 (thus including the equality case). Then, the probability that 𝑦 is the output of Best-of-N sampling is given by

∑︁𝑁

𝑖−1

𝑝<(𝑦) 𝑝≤(𝑦)

𝜋BoN(𝑦) = 𝜋ref(𝑦) × 𝑝≤(𝑦)𝑁−1

. (4)

×

𝑖=1

(A)

(B)

1To distinguish between generations with the same reward, ties can be broken with any arbitrary strict ordering.

Interpretation. Theorem 1 provides an intuitive explanation on the behavior of Best-of-N sampling: it essentially reweights the original sampling distribution 𝜋ref, by the multiplicative terms (A) and (B).

- The term (A) corresponds to a penalty exponential in 𝑁 based on the fraction of generations (for the same prompt) that are worse or equal to the considered generation 𝑦. Intuitively, this ensures that we sample exponentially less from bad generations when increasing 𝑁.
- The term (B) is an additional correction factor due to the potential of collisions among generations. Importantly, it is at most linear in 𝑁 as it is always bounded within [1, 𝑁]:

∑︁𝑁

∑︁𝑁

∑︁𝑁

𝑖−1

𝑖−1

𝑝<(𝑦) 𝑝≤(𝑦)

𝑝<(𝑦) 𝑝≤(𝑦)

1 ≤ 1 +

1 ≤ 𝑁 (5)

=

≤

𝑖=2

𝑖=1

𝑖=1

It achieves its minimum at 1 for the worst generation 𝑦− since we have 𝑝<(𝑦−) = 0 by definition. This is not surprising, as we need to sample 𝑦− exactly 𝑁 times in a row and which corresponds to 𝜋BoN(𝑦−) = 𝜋ref(𝑦−)𝑁 (note that 𝑝<(𝑦−) = 𝜋ref(𝑦−)). In contrast, if the likelihood of individual generations 𝑦 are low and such generations are good, then 𝑝<(𝑦) is almost 𝑝≤(𝑦) and term (b) is close to 𝑁. Intuitively, this corresponds to the case where sampling a generation 𝑦 multiple times is unlikely. In the extreme case when 𝜋ref is a continuous distribution, term (B) is constant and equal to 𝑁 (see Appendix A.2).

##### 3.2. The BOND objective

The analytical characterization of the Best-of-N distribution allows us to formulate BOND as a distribution matching problem. That is, we want to solve the objective:

𝜋BOND = argmin

𝐷(𝜋 ∥ 𝜋BoN), (6)

𝜋∈Π

where 𝐷(· ∥ ·) is a divergence metric steering the training policy 𝜋 towards 𝜋BoN. For this, a toolbox of possible divergences exist in the literature including, e.g., forward and backward KL (Kullback, 1959). Moreover, we can employ existing distribution matching techniques to estimate 𝐷 from online and offline samples. We defer the choice of suitable divergences and resulting BOND algorithms to Section 4.

###### 3.3. Connection with standard RLHF

In this section, we draw important connections between the two seemingly different objectives of standard RLHF (Equation (1)) and BOND (Equation (6)).

It is well known (see, e.g., Vieillard et al. (2020); Rafailov et al. (2023)) that the policy maximizing the RLHF objective from Equation (1) is:

1

𝑟(𝑦) . (7)

𝜋RL(𝑦) ∝ 𝜋ref(𝑦) exp

𝛽RL

From the derived expression of 𝜋BoN in Theorem 1, we see that the Best-of-N sampling distribution coincides with the optimal solution of standard RLHF when using the following specific BOND reward:

∑︁𝑁

𝑖−1

1 𝑁 − 1

𝑝<(𝑦) 𝑝≤(𝑦)

𝑟BOND(𝑦) = log 𝑝≤(𝑦)

log

, (8)

+

𝑖=1

(A)

(B)

and the specific regularization strength 𝛽BOND = 𝑁1−1. The term (B) corresponds to the correction factor in Theorem 1, which is bounded in 0, log𝑁−𝑁1 for all generations 𝑦. Instead term (A) lies in (−∞,0]. This provides two interesting insights for Best-of-N sampling:

- 1. Best-of-N sampling corresponds to the solution of a standard KL-regularized RLHF problem where the choice of 𝑁 determines the level of KL regularization.
- 2. Best-of-N sampling corresponds to optimizing the expected log reward quantile, i.e., the log likelihood that the generation has larger reward than a random sample from the reference

distribution. Interestingly, due to the concavity of the logarithm, 𝑟BOND(𝑦) strongly encourages the model to avoid bad generations rather than encouraging to generate good ones. Moreover, 𝑟BOND(𝑦) is invariant to monotone transformations of the reward 𝑟(·), since it depends only on the rank among the generations. We conjecture that both these features make the BOND reward 𝑟BOND(𝑦) more robust to reward hacking compared to standard RLHF.

The connection to RLHF also inspires the proposed approach in this manuscript: if we can compute the BOND reward or equivalently the Best-of-N distribution 𝜋BoN, then we can steer the policy towards Best-of-N via distribution matching. In the next section we explore different algorithms to tackle the main underlying challenges.

### 4. BOND Challenges and Algorithms

Implementing the BOND approach induces the three following challenges: (1) how to estimate the reward quantiles, (2) which is the appropriate divergence metric to use, and (3) how to choose the hyperparameter 𝑁 representing the number of sampled generations in Best-of-N. We discuss and address these challenges in the next three subsections.

- 4.1. Monte-Carlo quantile estimation One key difficulty in estimating the 𝜋BoN distribution is that we need to estimate the quantile

𝑝≤(𝑦) = ℙ𝑦′∼𝜋ref[𝑟(𝑦′) ≤ 𝑟(𝑦)], (9)

of a given generation 𝑦. The quantile 𝑝≤(𝑦) measures the quality of 𝑦 compared to generations from 𝜋ref when conditioned on the same prompt (recall that we have suppressed the conditioning on 𝑥 in our notation). A very simple but effective quantile estimation method is Monte-Carlo sampling, sampling 𝑘 generations from 𝜋ref and obtaining the following empirical estimate:

###### ∑︁𝑘

1

ˆ𝑝≤(𝑦) =

𝕀{𝑟(𝑦𝑖) ≤ 𝑟(𝑦)}. (10)

𝑘

𝑖=1

We found this to be a very effective in our experiments, even with a limited number of samples. In principle, though, one could also use alternative approaches, e.g., training a learned quantile model (as we explore in Appendix B.1).

###### 4.2. Jeffreys divergence as a robust objective

The choice of the divergence metric used in BOND is of crucial importance: different divergences can steer the policy to very different solutions. Here, we propose the Jeffreys divergence as a robust distribution matching objective.

The Jeffreys divergence (Jeffreys, 1946) between two distributions is defined as:

𝐽effreys𝛽 (𝑝 ∥ 𝑞) := (1 − 𝛽) · KL(𝑞 ∥ 𝑝) Forward KL

+ 𝛽 · KL(𝑝 ∥ 𝑞)

. (11)

Backward KL

The (generalized) Jeffreys divergence is a weighted average (with weight 𝛽 ∈ [0, 1]) between the forward and backward KL divergence. Notably, when fine-tuning policy 𝑝, the forward KL(𝑞 ∥ 𝑝) encourages that generations likely under 𝑞 are also likely under 𝑝, thus encouraging a mode-covering behavior. Instead, the reverse KL(𝑝 ∥ 𝑞) is well-known to have a mode-seeking effect, steering policy 𝑝 to produce generations that have a high likelihood according to 𝑞 (Agarwal et al., 2024). While the forward KL may produce over-spread distributions, the backward KL can lead to policy and entropy collapses. Instead, we empirically show that the Jeffreys divergence inherits the best of both divergences, producing better aligned policies.

In the context of BOND, this translates into minimizing the divergence 𝐽effreys𝛽 (𝜋 ∥ 𝜋BoN) which we can estimate using samples from the training policy 𝜋 and reference policy 𝜋ref as follows.

###### Estimation of the forward KL. The forward KL defined as

KL(𝜋BoN ∥ 𝜋) = 𝔼𝑦∼𝜋BoN[log𝜋BoN(𝑦) − log𝜋(𝑦)] (12)

can be estimated directly drawing samples from the 𝜋BoN (i.e., sampling 𝑁 times from 𝜋ref and selecting the best one) and can be seen as a supervised fine-tuning loss on the Best-of-N samples:

∇𝜋KL(𝜋BoN ∥ 𝜋) = −𝔼𝑦∼𝜋BoN∇ log𝜋(𝑦). (13) Estimation of the backward KL. The backward KL defined as

KL(𝜋 ∥ 𝜋BoN) = 𝔼𝑦∼𝜋[log𝜋(𝑦) − log𝜋BoN(𝑦)] (14) can be estimated from the policy samples (note the expectation w.r.t. 𝜋) and their estimated loglikelihood under 𝜋BoN. In particular, by the analogies drawn in Section 3.3, we show (in Appendix A.3) that its gradient coincides with a policy gradient (e.g., used by REINFORCE (Williams, 1992) in standard RLHF):

∇𝜋KL(𝜋 ∥ 𝜋BoN) = −(𝑁 − 1) 𝐸

∇𝜋 log𝜋(𝑦) 𝑟BOND(𝑦) − 𝛽BOND log𝜋(𝑦) − log𝜋ref(𝑦) , (15)

𝑦∼𝜋

with the equivalent reward 𝑟BOND and regularization 𝛽BOND defined in Section 3.3. Note that 𝑟BOND(𝑦) depends on the true unknown quantile 𝑝≤(𝑦) and on the correction factor (B) defined in Equation (8). In practice, we substitute the true quantile by its estimate, while we observed the correction factor does not play a significant role. Thus, we use 𝑟BOND(𝑦) = ˆ𝑝≤(𝑦). Moreover, to reduce the resulting variance, we use a policy gradient baseline (Sutton and Barto, 1998) which we compute as the average return for the other generations in the batch.

Thus, the overall 𝐽effreys𝛽 loss is a linear weighted combination between a supervised fine-tuning and a policy gradient loss.

Experiments. In Figure 2, we consider the abstractive summarization XSum task (Narayan et al., 2018) with 𝜋ref being a T5 supervised fine-tuned policy and 𝑟(·) being a T5 NLI reward model (Roit

- et al., 2023). We run BOND with 𝐽effreys𝛽 objective and 𝛽 ∈ {0,0.5,1}. We use 16 MC samples (per prompt) to estimate quantiles during training. During eval (each 500 training steps), we use 32 MC samples to estimate the backward and forward KL divergences between the training policy and the

𝜋BoN distribution. We set 𝑁 = 8 in Figure 2, and provide similar plots in Appendix B.2 for 𝑁 = 4 and 16. The results confirm our intuition: the Jeffreys divergence (𝛽 = 0.5) allows to minimize both divergences from 𝜋BoN (left and middle plot), compared to when solely the backward (𝛽 = 1) or the forward (𝛽 = 0) KL divergence is minimized. In addition, in the rightmost plot we report the reward log quantiles (averaged over the eval batches) of the training policy. Interestingly, BOND with 𝛽 = 0.5 maximizes the quantiles similarly to the mode-seeking 𝛽 = 1 choice, while the mode-covering forward KL (𝛽 = 0) lags behind.

[Figure 3]

[Figure 4]

[Figure 5]

###### Figure 2 | BOND with 𝑁 = 8 and different values of 𝛽 for the Jeffreys divergence objective (cf. Equation (11)). When using 𝛽 = 0.5 (Jeffreys divergence), BOND minimizes both backward (left plot)

and forward (middle plot) KL divergences from 𝜋BoN, achieving best of both objectives (𝛽 = 0 and 𝛽 = 1). Moreover, it optimizes the reward quantiles (right plot) significantly more than when using 𝛽 = 0 (forward KL minimization). Results for 𝑁 = 4 and 𝑁 = 16 are reported in Appendix B.2.

- 4.3. Iterative BOND

Finally, we discuss the choice of the parameter 𝑁. In practice, choosing 𝑁 may be difficult for three main reasons: (1) As in standard RLHF, 𝑁 plays the role of regularization (see Section 3.3): a large 𝑁 improves downstream performance, but if 𝑁 is too large it will eventually cause reward over optimization (Gao et al., 2023). (2) The larger the 𝑁 the more the estimate of 𝜋BoN is sensitive to errors in the estimated quantiles (since 𝜋BoN(𝑦) ∝ 𝑝≤(𝑦)𝑁−1). (3) Estimating the forward KL divergence requires sampling from 𝜋BoN which is prohibitive for large 𝑁.

To address the above challenges, we propose the iterative BOND approach. The approach is inspired by the fact that Best-of-N sampling from a Best-of-N distribution, coincides with Best-of-N2 sampling from the original distribution. More generally, by informally defining Bo𝑁(·) as an operator that performs Best-of-N sampling from a base distribution, we have:

𝜋ref))) ≡ BoN𝑀(𝜋ref). (16)

BoN(· · ·BoN(BoN(

𝑀 times

This suggests the key idea behind iterative BOND: if we know how to distill the Best-of-N distribution (i.e., via BOND), then we can apply BOND recursively (say 𝑀 times), equivalently to distilling a Bestof-N𝑀 of the initial distribution 𝜋ref. This allows fixing a small 𝑛 (i.e., 𝑛 = 2) and running BOND (with 𝑁 = 𝑛) in an iterative fashion, as an improvement operator. For this, we can introduce an auxiliary anchor policy 𝜋anchor initialized as 𝜋ref. We can then run BOND against 𝜋anchor (i.e., we can distill the Best-of-n version of 𝜋anchor) and, after a given number of distillation steps, update 𝜋anchor to be the current training policy 𝜋𝑡. The overall approach is depicted in Figure 3 and summarized in Algorithm 1.

[Figure 6]

###### Figure 3 | Iterative BOND approach. The policy 𝜋𝑡 is trained to iteratively distill a Best-of-𝑁 (in the figure, 𝑁 = 2) version of a moving anchor. This allows to continuously improve the policy performance without requiring to set a (large) 𝑁 upfront. Moreover, it leads to better training stability and a minimal computational complexity, since a small 𝑛 is used at each distillation step.

- Algorithm 1 Iterative BOND (meta algorithm) Inputs: 𝜋ref, 𝑛 ∈ ℕ.

Initialize 𝜋0 = 𝜋ref, 𝜋0anchor = 𝜋ref. for 𝑡 = 0, . . . , do */ iterations

𝜋𝑡+1 = argmin𝜋∈Π 𝐷(𝜋 ∥ Best-of-n(𝜋𝑡anchor)) */ distill the Best-of-n version of 𝜋𝑡anchor 𝜋𝑡anchor = 𝜋𝑡

[Figure 7]

[Figure 8]

[Figure 9]

- Figure 4 | Iterative BOND (with 𝑛 = 2 and 𝑛 = 4) compared to BOND with 𝑁 = 4,8,16. Iterative BOND continuously improves rewards (left plot) and log quantiles (middle plot), while they saturate for non-iterative BOND (the smaller the 𝑁, the sooner). It allows achieving the same reward/KL trade-off (right plot) as non-iterative BOND while keeping a small 𝑛 and smoothly moving away from 𝜋ref.

In a nutshell, iterative BOND allows exponential scaling to arbitrary large 𝑁 (in fact, it does not require setting 𝑁 in advance) while keeping a reduced sample complexity and a stable optimization. The claim is validated in the results below.

Experiments. In Figure 4, we consider the same experimental setup than in Figure 2 from Section 4.2, fix the BOND objective to 𝐽effreys0.5 and run iterative BOND with 𝑛 ∈ {2,4} where the moving anchor is updated every 1000 steps. We report the average reward (left plot) and average log quantile (middle plot) obtained during training and compare them to (non-iterative) BOND run with 𝑁 ∈ {4,8,16}. As expected, both reward signals saturate early for non-iterative BOND (the smaller the 𝑁 the earlier the reward saturates), while the iterative BOND approach continuously improves performance (the higher the 𝑛, the faster). Moreover, in the rightmost plot we plot the obtained log quantiles against the KL from the reference policy. The plot shows that iterative BOND essentially has the same reward/KL trade-off of the non-iterative BOND runs, but crucially allows keeping a small 𝑛 and to smoothly but continuously move away from 𝜋ref.

- 5. The J-BOND Algorithm

In this section we present J-BOND, a concrete and practical BOND algorithm motivated by the results discussed in the previous sections. We describe its main components below, and summarize it in the pseudo-code of Algorithm 2.

J-BOND follows the template of iterative BOND (Algorithm 1) with 𝑛 = 2, i.e., it fine-tunes policy 𝜋𝑡 to iteratively distill the Best-of-2 version of a moving anchor 𝜋𝑡anchor, initialized as 𝜋ref. The name J-BOND stands for Jeffreys divergence BOND because it uses the Jeffreys divergence as distribution matching objective, i.e., it minimizes 𝐽effreys𝛽 (𝜋 ∥ Best-of-2(𝜋𝑡anchor)) as defined in Section 4.2.

Minimal sample complexity. Compared to the BOND algorithms tested in the previous section, J-BOND has a minimal sample complexity: for each prompt in the batch it generates 1 sample from

- Algorithm 2 The J-BOND algorithm Inputs: Prompt dataset D, reference policy 𝜋ref, reward 𝑟(·), 𝛽, 𝜂 ∈ [0, 1], 𝛾 ≥ 0.

Initialize policy and anchor 𝜋0 = 𝜋0anchor = 𝜋ref. for 𝑡 = 0, . . . do

Sample batch of prompts D𝑡 ⊆ D For each 𝑥 ∈ D𝑡 generate: 1 policy sample 𝑦 ∼ 𝜋𝑡(𝑥) and 2 anchor samples 𝑦′

1, 𝑦′

2 ∼ 𝜋𝑡anchor(𝑥)

###### */ Forward KL */

Extract Best-of-2 sample: 𝑦′

2} 𝑟(𝑦′). Compute forward KL gradient: 𝐺FW(𝑥, 𝜋𝑡) = −∇𝜋𝑡

Bo2 = argmax𝑦′∈{𝑦′

1,𝑦′

log𝜋𝑡(𝑥, 𝑦′

Bo2)

###### */ Backward KL */

Compute 𝑟J-BOND(𝑥, 𝑦) according to Equation (17). Compute return: 𝑅(𝑥, 𝑦) = 𝑟J-BOND(𝑥, 𝑦) − (log𝜋𝑡(𝑥, 𝑦) − log𝜋𝑡anchor(𝑥, 𝑦)). [Optional] Compute baseline 𝐵, e.g. average return of the other generations in the batch. Compute backward KL gradient: 𝐺BW(𝑥, 𝜋𝑡) = −∇𝜋𝑡

log𝜋𝑡(𝑥, 𝑦) · (𝑅(𝑥, 𝑦) − 𝐵).

###### */ Additional KL regularization */

log𝜋𝑡(𝑥, 𝑦) · (log𝜋𝑡(𝑥, 𝑦) − log𝜋𝑡anchor(𝑥, 𝑦)).

KL regularization gradient: 𝐺Reg(𝑥, 𝜋𝑡) = −∇𝜋𝑡

###### */ Overall policy update: Jeffreys divergence + KL regularization */

Update policy weights 𝜃𝑡+1 with the overall stochastic gradient: 𝔼𝑥∼D

𝑡 (1 − 𝛽) · 𝐺FW(𝑥, 𝜋𝑡) + 𝛽 · 𝐺BW(𝑥, 𝜋𝑡) + 𝛾 · 𝐺Reg(𝑥, 𝜋𝑡)

###### */ Update moving anchor */

Update anchor weights with EMA: 𝜃𝑡+1

anchor ← (1 − 𝜂) · 𝜃anchor𝑡 + 𝜂 · 𝜃𝑡+1

the policy 𝜋𝑡 and 2 samples from the anchor 𝜋𝑡anchor. While more anchor samples are generally useful for a better divergence estimation (in Section 4 we used 16 MC samples), autoregressive sampling is

the main bottleneck of online RLHF and we have thus opted for a practical approach working with a small number of samples.

Crude divergence estimate based on 2 anchor samples. The policy and anchor samples are used to obtain a crude estimate of the forward and backward KL components of 𝐽effreys𝛽 (𝜋 ∥ Best-of-2(𝜋𝑡anchor)) as described next.

We can minimize the forward KL as described in Section 4.2, by doing supervised fine-tuning on the best of the 2 anchor samples. To minimize the backward KL, we utilize the policy gradient-style loss of Equation (15) replacing 𝑟BOND(𝑦) with a different reward which we denote as 𝑟J-BOND(𝑦). The reason for this is that when only 2 anchor samples are available, the reward 𝑟BOND(𝑦) = log ˆ𝑝≤(𝑦) would be quite uninformative due to ˆ𝑝≤(𝑦) being a very noisy MC estimate. Let 𝑦 be the policy sample and {𝑦′

2} be the corresponding anchor samples, we instead define 𝑟J-BOND(𝑦) as

1, 𝑦′

𝑟J-BOND(𝑦) = −log(16) if 𝑟(𝑦) < min{𝑟(𝑦′

1), 𝑟(𝑦′

2)} 0 otherwise

. (17)

That is, generation 𝑦 receives a negative reward of −log(16) if it has worse reward than both anchor samples, while receives 0 reward otherwise. The above definition is motivated by the following two main reasons:

(i) We negatively reward 𝑦 only if it is worse than both the anchor samples, to mimic the concavity of the ideal (and unknown) reward function 𝑟BOND = log 𝑝≤(·).

[Figure 10]

[Figure 11]

[Figure 12]

- Figure 5 | J-BOND with periodic anchor updates (every 50 steps) vs. EMA anchor (𝜂 = 0.02) on Gemma 7B. While attaining the same reward (left), using the EMA anchor displays a significantly lower KL than the reference policy (middle) and thus a better reward/KL trade-off (right).

(ii) We choose value −log(16) to ensure that: 𝔼𝑦′

2∼𝜋𝑡anchor 𝑟J-BOND(𝑦) = log 𝑝≤(𝑦) when 𝑝≤(𝑦) = 0.5. The interested reader is referred to Appendix A.4 for a derivation and illustration of this fact. In words, the value −log(16) calibrates the reward function 𝑟J-BOND(·) so that, in expectation under the 2 anchor samples, it matches with the ideal reward log 𝑝≤(·) for generations 𝑦 that have median reward (i.e., when 𝑝≤(𝑦) = 0.5).

1,𝑦′

Exponential Moving Average (EMA) anchor. An important component of J-BOND, which refines the vanilla iterative BOND of Section 4.3, is the use of an Exponential Moving Average (EMA) anchor.

That is, instead of using a periodically updated anchor, we update the anchor weights 𝜃anchor𝑡 at each fine-tuning step as a moving average of the policy weights 𝜃𝑡:

𝜃anchor𝑡+1 ← (1 − 𝜂) · 𝜃anchor𝑡 + 𝜂 · 𝜃𝑡+1. (18)

Consistently with WARP (Ramé et al., 2024), we observed that this weight averaging procedure has a positive effect on training stability by reducing variance, and can improve the overall reward/KL trade-off of J-BOND. We provide an ablation in Section 6.

Additional KL regularization. Finally, we further regularize the policy to stay closer to the moving anchor via an extra2 KL regularization term, modulated by a tunable hyperparameter 𝛾 ≥ 0. The scope is to further stabilize the policy updates, viewing the overall operator as a constrained optimization one:

𝜋𝑡+1 = argmin 𝜋∈Π

𝐽effreys𝛽 (𝜋 ∥ Best-of-2(𝜋𝑡anchor)) + 𝛾 · KL(𝜋𝑡 ∥ 𝜋𝑡anchor). (19)

#### 6. Experiments

We test J-BOND on relevant use cases with the following main goals. First, we ablate and showcase important aspects of J-BOND: the benefits of the EMA anchor, and the effects of the anchor speed and the additional KL regularization. Then, we compare J-BOND to classical RLHF baselines using REINFORCE, demonstrating its efficacy and better performance.

Setup. We consider Gemma (2B and 7B) models (Gemma Team, 2024) which we aim to fine-tune into better conversational agents. For this task, we consider a set of conversational prompts D, a reference policy 𝜋ref previously supervised fine-tuned on similar prompts, and a large previously trained reward model 𝑟(·). We use a batch size 128 and the Adam optimizer (Kingma and Ba, 2015) with learning rate 3𝑒−6 and 100 warm-up steps. For the Jeffreys divergence objective, we set 𝛽 = 0.5.

2Note that KL regularization is already present in the backward KL component of the Jeffreys divergence.

[Figure 13]

[Figure 14]

[Figure 15]

- Figure 6 | J-BOND: the role of the EMA mixing parameter 𝜂 and regularization 𝛾. The left plot shows J-BOND with 𝛾 = 0, 𝜂 ∈ {0.01,0.05,0.1}: the larger the 𝜂 the faster the average reward increases. The middle and right plots show J-BOND with 𝜂 = 0.05 and 𝛾 ∈ {0,0.5,1,2}: the larger the regularization 𝛾, the slower the policy moves away from 𝜋ref, improving the reward/KL trade-off.

[Figure 16]

[Figure 17]

[Figure 18]

- Figure 7 | J-BOND (𝜂 = 0.02) for Gemma 7B, compared to standard REINFORCE (with KL-regularized

objective of Equation (1)) with regularization strength 𝛽RL ∈ {0.001,0.01,0.1,1}. J-BOND does not require committing to a specific regularization strength, but it continuously improves the reward displaying a stable and linear KL increase and a better reward/KL trade-off.

EMA vs. hard anchor updates. We ablate the benefits of using an EMA moving anchor (Equation (18)) compared to the periodically updated anchor used in Section 4.3. For this, we run J-BOND with 𝛾 = 0 and EMA coefficient 𝜂 = 0.02 on Gemma 7B, and compare it with its variant where the anchor is only updated every 50 steps. In Figure 5, we report the average reward of the policy during training (left plot), the KL from the reference policy 𝜋ref (middle plot), and the resulting reward/KL trade-off for the two runs. Both runs produce the same reward increase profile (this is not surprising since an EMA with 𝜂 = 0.02 roughly corresponds to an update period of 50 steps) but, crucially, J-BOND with an EMA anchor displays a significantly lower KL increase and, as a result, a better reward/KL trade-off.

Anchor speed and KL regularization. Here we illustrate the effect of the anchor mixing parameter 𝜂 ∈ [0, 1] and the benefits of the additional KL regularization parameter 𝛾 ≥ 0 introduced in Equation (19). For this, we consider Gemma 2B and first run J-BOND with 𝛾 = 0 and 𝜂 ∈ {0.01,0.05,0.1}. In the left plot of Figure 6 we report the average reward of the policy along training. This illustrates that the larger the mixing parameter 𝜂 (i.e., the faster the anchor moves), the faster the reward increases, as one could intuitively expect. Second, we fix 𝜂 = 0.05 and run J-BOND with different regularization strengths 𝛾 ∈ {0, 0.5, 1, 2}. We plot the results in the middle and rightmost plots of Figure 6. As expected, the larger the regularization 𝛾, the more constrained are the policy updates and thus, the slower the policy moves away from 𝜋ref (middle plot). Importantly, the right plot shows that such a regularization has a positive effect since it can ultimately improve the reward/KL trade-off.

Comparison with standard RLHF. We compare J-BOND against standard RLHF algorithms that aim at maximizing the KL-regularized objective of Equation (1). To optimize Equation (1), we use

REINFORCE (Williams, 1992) with 2 policy samples per-prompt and a leave-one-out baseline (Ahmadian et al., 2024) for policy gradient advantages. For J-BOND we set the anchor mixing coefficient to 𝜂 = 0.02. For REINFORCE, we test possible regularization strengths 𝛽RL ∈ {0.001, 0.01, 0.1, 1}. In Figure 7 we plot the average reward of the training policy (left plot) and its KL divergence from the reference 𝜋ref (middle plot). As presumed, REINFORCE is quite sensitive to the regularization coefficient 𝛽RL: the larger the regularization, the lower the reward achieved by REINFORCE (and the lower the KL from 𝜋ref). This highlight a key advantage of J-BOND: it does not require committing to a specific regularization level, but it continuously improves the reward displaying a stable and linear KL increase. Moreover, in the rightmost plot we plot the corresponding reward/KL trade-off showing that J-BOND produces a better reward/KL than all of the REINFORCE baselines.

J-BOND for Gemma open models. J-BOND was also the algorithm used to fine-tune open-weight models such as Gemma 1.1 2B and 7B (Gemma Team, 2024), RecurrentGemma 2B and 9B (Botev et al., 2024) as well as CodeGemma 1.1 (CodeGemma Team, 2024). This led to competitive performance. For example, the Gemma 1.1 IT 7B model outperforms Mistral 7B v0.2 Instruct in both safety and instruction following (see Gemma Team (2024, Table 5) for more details).

#### 7. Related Work

Best-of-N was introduced in Stiennon et al. (2020) as a straightforward but costly inference method to optimize language generation against a given reward function. Further works established and refined an analytical form for the KL divergence against the reference (i.e., Bo1) policy (Hilton, 2023; Beirami et al., 2024), provided an estimator for the average Best-of-N reward (Nakano et al., 2021), made theoretical connections with KL-constrained RL (Yang et al., 2024) and provided scaling laws for Best-of-N alignment (Gao et al., 2023).

Matching Best-of-N for improved alignment is a strategy that was studied in different flavors in the literature. Dong et al. (2023) and Touvron et al. (2023) propose to fine-tune LLMs in a supervised fashion on Best-of-N data actually applying forward KL minimization. Concurrently to ours, Gui et al. (2024) proposes to mimic the Best-of-N policy by applying a combination of supervised fine-tuning on best responses and direct preference optimization on best-and-worst response pairs. The latter is similar to a common strategy in online preference optimization methods: Guo et al. (2024) use pairwise AI feedback on online generations to obtain online preferences that are then optimized, while Calandriello et al. (2024) use a dedicated preference reward model instead. Concurrently and closest to our work, Amini et al. (2024) also apply distribution matching in order to get the benefits of Best-of-N sampling with amortized cost. While their formalization is identical, we opt for a different divergence (i.e., Jeffreys) than the one they use (i.e., only backward KL), and propose an iterative procedure with dynamic anchor, which we show critical for optimal results. Best-of-N can also be used for self-improvement in reward modeling, as evidenced in Pace et al. (2024).

Using a contrastive advantage is an option of J-BOND studied in prior works as well, which replaced a value estimate by the average Monte Carlo return of other samples. This was applied in the context of REINFORCE (Kool et al., 2019; Pinto et al., 2023), for online RLHF (Ahmadian et al., 2024), offline RLHF (Flet-Berliac et al., 2024) and preference optimization (Wu et al., 2024).

Exponential moving average (EMA) of policy as reference in regularization, which we use in J-BOND, is an increasingly popular option. While most alignment approaches use a static anchor, dynamic anchors bring the benefit of improving the flexibility of the policy space being explored (Munos et al., 2023; Gorbatovski et al., 2024; Ramé et al., 2024), with the caveat that too slow updates limit optimization and too fast updates hinder stability.

Scaling post-training and iterated amplification. BOND hinges on the idea of investing more resources during training to ensure that computational demands during inference remain low, a factor often overlooked in traditional scaling laws (Hoffmann et al., 2022). Specifically, BOND incorporates the principles of iterated amplification (Christiano et al., 2018; Cotra, 2018), where amplification in this context consists of producing multiple generations, comparing their rewards, and using these to iteratively improve the policy performance. In this regard, BOND is complementary to WARM (Ramé

- et al., 2024) and WARP (Ramé et al., 2024), which previously scaled post-training by training multiple reward models and policies, respectively.

#### 8. Conclusion

We introduce BOND, a novel RLHF method that fine-tunes the policy via online distillation of the Best-of-N sampling distribution. We propose a concrete algorithm, J-BOND, that integrates multiple components to enhance its practicality and efficiency; Monte-Carlo quantile estimation, a combination between forward and backward KL divergence objectives, and an iterative procedure with an exponential moving average anchor. J-BOND improves the KL-reward Pareto front of solutions, and compares favorably against state-of-the-art baselines. We hope this work can help improve alignment of AI systems, making them safer and more reliable.

#### Acknowledgements

We thank Daniele Calandriello for insightful comments, as well as Gil Shamir, Bilal Piot, and Remi Munos for helpful discussions.

#### References

- R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In ICLR, 2024. (p. 6)

A. Ahmadian, C. Cremer, M. Gallé, M. Fadaee, J. Kreutzer, A. Üstün, and S. Hooker. Back to basics: Revisiting REINFORCE style optimization for learning from human feedback in LLMs. arXiv preprint,

2024. (pp. 3 and 12) A. Amini, T. Vieira, and R. Cotterell. Variational Best-of-N alignment. arXiv preprint, 2024. (p. 12)

A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones, N. Joseph, B. Mann, N. DasSarma, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, J. Kernion, K. Ndousse, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, and J. Kaplan. A general language assistant as a laboratory for alignment. arXiv preprint, 2021. (p. 1)

A. Beirami, A. Agarwal, J. Berant, A. D’Amour, J. Eisenstein, C. Nagpal, and A. T. Suresh. Theoretical guarantees on the Best-of-N alignment policy. arXiv preprint, 2024. (p. 12)

A. Botev, S. De, S. L. Smith, A. Fernando, G.-C. Muraru, R. Haroun, L. Berrada, R. Pascanu, P. G. Sessa, R. Dadashi, L. Hussenot, J. Ferret, S. Girgin, O. Bachem, A. Andreev, K. Kenealy, T. Mesnard, C. Hardin, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivière, M. S. Kale, J. Love, P. Tafti, A. Joulin, N. Fiedel, E. Senter, Y. Chen, S. Srinivasan, G. Desjardins, D. Budden, A. Doucet, S. Vikram, A. Paszke, T. Gale, S. Borgeaud, C. Chen, A. Brock, A. Paterson, J. Brennan, M. Risdal, R. Gundluru, N. Devanathan, P. Mooney, N. Chauhan, P. Culliton, L. G. Martins, E. Bandy, D. Huntsperger,

- G. Cameron, A. Zucker, T. Warkentin, L. Peran, M. Giang, Z. Ghahramani, C. Farabet, K. Kavukcuoglu,

D. Hassabis, R. Hadsell, Y. W. Teh, and N. de Frietas. Recurrentgemma: Moving past transformers for efficient open language models. arXiv preprint, 2024. (p. 12)

D. Calandriello, D. Guo, R. Munos, M. Rowland, Y. Tang, B. A. Pires, P. H. Richemond, C. L. Lan, M. Valko, T. Liu, et al. Human alignment of large language models through online preference optimisation. In ICML, 2024. (p. 12)

- S. Casper, X. Davies, C. Shi, T. K. Gilbert, J. Scheurer, J. Rando, R. Freedman, T. Korbak, D. Lindner, P. Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. TMLR, 2023. (p. 1)

P. Christiano, B. Shlegeris, and D. Amodei. Supervising strong learners by amplifying weak experts. arXiv preprint, 2018. (p. 13)

P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei. Deep reinforcement learning from human preferences. In NeurIPS, 2017. (p. 1)

J. Clark and D. Amodei. Faulty Reward Functions in the Wild. https://openai.com/research/ faulty-reward-functions, 2016. (p. 1)

CodeGemma Team. Codegemma: Open code models based on gemma. arXiv preprint, 2024. (p. 12) A. Cotra. Iterated distillation and amplification. https://ai-alignment.com/iterated-dis

tillation-and-amplification-157debfd1616, 2018. (p. 13)

- T. M. Cover. Elements of information theory. 1999. (p. 21)

W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos. Distributional reinforcement learning with quantile regression. In AAAI Conference on Artificial Intelligence, 2017. (p. 21)

- H. Dong, W. Xiong, D. Goyal, Y. Zhang, W. Chow, R. Pan, S. Diao, J. Zhang, K. SHUM, and T. Zhang. RAFT: Reward ranked finetuning for generative foundation model alignment. TMLR, 2023. (p. 12)

- Y. Flet-Berliac, N. Grinsztajn, F. Strub, E. Choi, C. Cremer, A. Ahmadian, Y. Chandak, M. G. Azar, O. Pietquin, and M. Geist. Contrastive policy gradient: Aligning llms on sequence-level scores in a supervised-friendly fashion. arXiv preprint, 2024. (p. 12)

- R. M. French. Semi-distributed representations and catastrophic forgetting in connectionist networks. Connection Science, 1992. (pp. 1 and 3)

L. Gao, J. Schulman, and J. Hilton. Scaling laws for reward model overoptimization. In ICML, 2023.

(pp. 1, 7, and 12)

- M. Geist, B. Scherrer, and O. Pietquin. A theory of regularized markov decision processes. In ICML,

2019. (p. 3) Gemini Team. Gemini: A family of highly capable multimodal models. 2023. (p. 1) Gemma Team. Gemma: Open models based on gemini research and technology. arXiv preprint, 2024.

(pp. 2, 10, and 12)

A. Gorbatovski, B. Shaposhnikov, A. Malakhov, N. Surnachev, Y. Aksenov, I. Maksimov, N. Balagansky,

and D. Gavrilov. Learn your reference model for real good alignment. arXiv preprint, 2024. (p. 12) L. Gui, C. Gârbacea, and V. Veitch. Bonbon alignment for large language models and the sweetness

of Best-of-N sampling. arXiv preprint, 2024. (p. 12)

- S. Guo, B. Zhang, T. Liu, T. Liu, M. Khalman, F. Llinares, A. Rame, T. Mesnard, Y. Zhao, B. Piot, J. Ferret, and M. Blondel. Direct language model alignment from online ai feedback. arXiv preprint,

2024. (p. 12) J. Hilton. KL divergence of max-of-n, 2023. (p. 12) J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A.

Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. In NeurIPS,

2022. (p. 13)

H. Jeffreys. An invariant form for the prior probability in estimation problems. Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 186(1007):453–461, 1946.

(p. 5)

A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas,

E. B. Hanna, F. Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024. (p. 2) D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In ICLR, 2015. (p. 10) W. Kool, H. van Hoof, and M. Welling. Buy 4 REINFORCE samples, get a baseline for free! 2019.

(p. 12) S. Kullback. Information Theory and Statistics. New York, 1959. (p. 4) A. Lazaridou, A. Potapenko, and O. Tieleman. Multi-agent communication meets natural language:

Synergies between functional and structural language learning. In ACL, 2020. (p. 3)

- Z. Li, T. Xu, Y. Zhang, Y. Yu, R. Sun, and Z.-Q. Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint, 2023. (p. 3)

R. Munos, M. Valko, D. Calandriello, M. G. Azar, M. Rowland, Z. D. Guo, Y. Tang, M. Geist, T. Mesnard, A. Michi, et al. Nash learning from human feedback. arXiv preprint, 2023. (p. 12)

- R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint, 2021. (pp. 1 and 12)
- S. Narayan, S. B. Cohen, and M. Lapata. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, Conference on Empirical Methods in Natural Language Processing, 2018. (pp. 2 and 6)

R. Ngo, L. Chan, and S. Mindermann. The alignment problem from a deep learning perspective. arXiv

preprint, 2022. (p. 1) OpenAI. Gpt-4 technical report. 2023. (p. 1) L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama,

A. Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022.

(p. 1)

A. Pace, J. Mallinson, E. Malmi, S. Krause, and A. Severyn. West-of-n: Synthetic preference generation for improved reward modeling. arXiv preprint, 2024. (p. 12)

A. Pan, K. Bhatia, and J. Steinhardt. The effects of reward misspecification: Mapping and mitigating misaligned models. In ICLR, 2022. (p. 1)

A. S. Pinto, A. Kolesnikov, Y. Shi, L. Beyer, and X. Zhai. Tuning computer vision models with task rewards. In ICML, 2023. (p. 12)

J. Qiping Yang, S. Salamatian, Z. Sun, A. Theertha Suresh, and A. Beirami. Asymptotics of language model alignment. arXiv, 2024. (p. 3)

A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever. Improving language understanding by generative pre-training. 2018. (p. 1)

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. 2019. (p. 1)

R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint, 2023. (p. 4)

C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 2020. (p. 1)

A. Ramé, N. Vieillard, L. Hussenot, R. Dadashi, G. Cideron, O. Bachem, and J. Ferret. WARM: On the benefits of weight averaged reward models. In ICML, 2024. (p. 13)

A. Ramé, J. Ferret, N. Vieillard, R. Dadashi, L. Hussenot, P.-L. Cedoz, P. G. Sessa, S. Girgin, A. Douillard, and O. Bachem. WARP: On the benefits of weight averaged rewarded policies. arXiv preprint, 2024.

(pp. 10, 12, and 13)

- M. Reid, N. Savinov, D. Teplyashin, D. Lepikhin, T. Lillicrap, J.-b. Alayrac, R. Soricut, A. Lazaridou,

- O. Firat, J. Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint, 2024. (p. 1)
- P. Roit, J. Ferret, L. Shani, R. Aharoni, G. Cideron, R. Dadashi, M. Geist, S. Girgin, L. Hussenot, O. Keller, et al. Factually consistent summarization via reinforcement learning with textual entailment feedback. In ACL, 2023. (p. 6)

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint, 2017. (p. 3)

J. M. V. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger. Defining and characterizing reward gaming. In NeurIPS, 2022. (pp. 1 and 3)

- N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano. Learning to summarize with human feedback. NeurIPS, 2020. (pp. 1 and 12)

R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. MIT Press, 1998. (p. 6) Y. Tang, D. Z. Guo, Z. Zheng, D. Calandriello, Y. Cao, E. Tarassov, R. Munos, B. Ávila Pires, M. Valko,

Y. Cheng, and W. Dabney. Understanding the performance gap between online and offline alignment algorithms. arXiv preprint, 2024. (p. 3)

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint, 2023. (pp. 1, 3, and 12)

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In NeurIPS, 2017. (p. 2)

- N. Vieillard, T. Kozuno, B. Scherrer, O. Pietquin, R. Munos, and M. Geist. Leverage the average: an analysis of kl regularization in reinforcement learning. NeurIPS, 2020. (p. 4)

J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le. Finetuned language models are zero-shot learners. In ICLR, 2022. (p. 1)

R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Reinforcement learning, 1992. (pp. 1, 3, 6, 12, and 19)

Y. Wu, Z. Sun, H. Yuan, K. Ji, Y. Yang, and Q. Gu. Self-play preference optimization for language model alignment. arXiv preprint, 2024. (p. 12)

J. Q. Yang, S. Salamatian, Z. Sun, A. T. Suresh, and A. Beirami. Asymptotics of language model alignment. arXiv preprint, 2024. (p. 12)

D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences. arXiv preprint, 2019. (p. 1)

#### A. Supporting results and derivations

###### A.1. Proof of Theorem 1

Consider 𝑁 random generations 𝑦1, 𝑦2, ..., 𝑦𝑁 from 𝜋ref and an arbitrary generation 𝑦 among them. Let 𝐴𝑖(𝑦) denote the event that 𝑦 is the best sample (i.e., 𝑟(𝑦) ≥ 𝑟(𝑦𝑖) for all 𝑖) and that 𝑖 is the lowest index for which 𝑦𝑖 = 𝑦. It is trivial to see that the the events {𝐴𝑖(𝑦)}𝑖=1,2,...,𝑁 are disjoint and that their union corresponds to 𝑦 being selected by Best-of-N sampling.

The event 𝐴𝑖(𝑦) occurs if and only if three conditions are met: 𝑟(𝑦𝑗) < 𝑟(𝑦) for all 𝑗 < 𝑖, 𝑦𝑖 = 𝑦, and 𝑟(𝑦𝑗) < 𝑟(𝑦) for all 𝑗 < 𝑖. This allows us to derive the likelihood of the event 𝐴𝑖(𝑦):

𝑖−1

𝑁

ℙ[𝐴𝑖(𝑦)] =

ℙ 𝑟(𝑦𝑗) < 𝑟(𝑦) × 𝜋ref(𝑦) ×

ℙ 𝑟(𝑦𝑗) ≤ 𝑟(𝑦)

𝑗=1

𝑗=𝑖+1

= 𝑝<(𝑦)𝑖−1 × 𝜋ref(𝑦) × 𝑝≤(𝑦)𝑁−𝑖−1.

The likelihood that Best-of-N sampling selects the generation 𝑦 is then given by

∑︁𝑁

𝜋BoN(𝑦) =

ℙ[𝐴𝑖(𝑦)]

𝑖=1

∑︁𝑁

𝑝<(𝑦)𝑖−1 × 𝜋ref(𝑦) × 𝑝≤(𝑦)𝑁−𝑖

=

𝑖=1

∑︁𝑁

𝑝<(𝑦)𝑖−1 × 𝑝≤(𝑦)𝑁−𝑖

= 𝜋ref(𝑦) ×

𝑖=1

∑︁𝑁

𝑖−1

𝑝<(𝑦) 𝑝≤(𝑦)

= 𝜋ref(𝑦) × 𝑝≤(𝑦)𝑁−1 ×

. □

𝑖=1

###### A.2. Link to the continuous case

- A noteworthy observation is that we can relate the Best-of-N expression to the case of a continuous distribution, in which case the term (B) in Equation (4) is constant and equal to 𝑁 (which is intuitively natural as 𝑝<(𝑦) and 𝑝≤(𝑦) have the same value in this case). Indeed, recall that the probability for a sequence 𝑦 to be drawn from the Best-of-N distribution is

∑︁𝑁

𝑖−1

𝑝<(𝑦) 𝑝≤(𝑦)

𝜋BoN(𝑦) = 𝜋ref(𝑦) × 𝑝≤(𝑦)𝑁−1

. (20)

×

𝑖=1

(A)

(B)

Here, 𝑦 is a discrete variable, as it lives in ⟦1;𝑇⟧𝐿 where 𝑇 is the number of tokens and 𝐿 is the maximum length of a sequence.

Now, we show why Equation (20) matches the classic formula for the max of 𝑁 continuous variables. Formally, let 𝑋 be a real valued random variable with density 𝑓𝑋 and a cumulative distribution function 𝐹𝑋. Taking 𝑌1, ...𝑌𝑁 i.i.d. variables with the same density, define 𝑋𝑁 = max{𝑌1, ...𝑌𝑁} as the maximum over the 𝑁 variables. Then, we have that

𝐹𝑋𝑁 (𝑦) = ℙ(𝑌1 ≤ 𝑦, . . .𝑌𝑁 ≤ 𝑦) = 𝐹𝑋(𝑦)𝑁, (21)

and thus

𝑓𝑋𝑁 (𝑦) = 𝑓𝑋(𝑦)𝐹𝑋(𝑦)𝑁−1𝑁. (22) In Equation (22), we recognize the Best-of-N formula in the case where the correction factor (B) is 𝑁. For the term (A), 𝐹𝑋(𝑦) plays the role of 𝑝≤(𝑦), as by definition 𝐹𝑋(𝑦) = ℙ(𝑋 ≤ 𝑦). Finally, 𝑓𝑋(𝑦) is the density of 𝑋, which is analogous to the probability 𝜋ref(𝑦) in the discrete case.

###### A.3. Backward KL and policy gradient equivalence

We formally show the analogy between the gradient of the backward KL divergence of Equation (14) and the standard (e.g., REINFORCE (Williams, 1992)) policy gradient of a KL-regularized RLHF problem with equivalent reward 𝑟BOND and regularization 𝛽BOND.

The exact backward KL gradient can be derived as:

∇𝜋 KL(𝜋 || 𝜋BoN) = ∇𝜋𝔼𝑦∼𝜋[log𝜋(𝑦) − log𝜋BoN(𝑦)]

= ∇𝜋 ∑︁

𝜋(𝑦)(log𝜋(𝑦) − log𝜋BoN(𝑦))

= ∑︁

𝑦

∇𝜋𝜋(𝑦)(log𝜋(𝑦) − log𝜋BoN(𝑦)) + 𝜋(𝑦)∇𝜋 log𝜋(𝑦)

= ∑︁

𝑦

𝜋(𝑦)∇𝜋 log𝜋(𝑦)(log𝜋(𝑦) − log𝜋BoN(𝑦)) + 𝜋(𝑦)∇𝜋 log𝜋(𝑦)

𝑦

= 𝔼𝑦∼𝜋[∇𝜋 log𝜋(𝑦)(log𝜋(𝑦) − log𝜋BoN(𝑦)) + ∇𝜋 log𝜋(𝑦)]

= 𝔼𝑦∼𝜋[∇𝜋 log𝜋(𝑦)(log𝜋(𝑦) − log𝜋BoN(𝑦))].

Above, we have used the product rule of gradient, the rule ∇𝜋𝜋(𝑦) = 𝜋(𝑦)∇𝜋 log𝜋(𝑦) and the fact that 𝔼𝑦∼𝜋∇𝜋 log𝜋(𝑦) = 0.

Equivalence with Policy Gradient RL. As anticipated, one can verify that descending the above gradient is equivalent – up to a constant scaling – to running the RL policy gradient REINFORCE algorithm on the RL objective of Equation 1 with 𝑟 = 𝑟BOND and 𝛽RL = 𝛽BOND. Indeed, we can use the expression for 𝜋BoN to break down the above gradient into:

𝔼𝑦∼𝜋[∇𝜋 log𝜋(𝑦)(log𝜋(𝑦) − log𝜋BoN(𝑦))]

∑︁𝑁

𝑝<(𝑦) 𝑝≤(𝑦)

= 𝔼𝑦∼𝜋 ∇𝜋 log𝜋(𝑦) log𝜋(𝑦) − log𝜋ref(𝑦) − (𝑁 − 1) log 𝑝≤(𝑦) − log

𝑖=1

𝑟BOND(𝑦) 𝑁 − 1

= 𝔼𝑦∼𝜋 ∇𝜋 log𝜋(𝑦) log𝜋(𝑦) − log𝜋ref(𝑦) −

= −(𝑁 − 1) 𝔼𝑦∼𝜋[∇𝜋 log𝜋(𝑦)(𝑟BOND(𝑦) − 𝛽BOND(log𝜋(𝑦) − log𝜋ref(𝑦)))]

.

gradient used by REINFORCE

𝑖−1

##### A.4. Derivation of J-BOND reward

Here we provide a theoretical explanation behind the design of the J-BOND reward function discussed in Section 5:

𝑟J-BOND(𝑦) = −log(16) if 𝑟(𝑦) ≤ min{𝑟(𝑦′

1), 𝑟(𝑦′

2)} 0 otherwise

.

Recall that 𝑟J-BOND(·) is meant to approximate the the true reward function 𝑟BOND(𝑦) = log 𝑝≤(·) which is unknown since 𝑝≤(·) in general requires knowing the reward distribution of 𝜋𝑡anchor (in J-BOND, we only take 2 samples 𝑦′

2 ∼ 𝜋𝑡anchor).

1, 𝑦′

As mentioned in Section 5, we designed 𝑟J-BOND(·) to assign a negative reward only if sample 𝑦 is worse than both the anchor samples, to mimic the concavity of the log quantile log 𝑝≤(·). In practice, we did not observe gains when rewarding also the intermediate case. The particular choice of value −log(16) is motivated by the following main reason.

We want that, when sample 𝑦 has median reward compared to the anchor rewards’ distribution (i.e., 𝑝≤(𝑦) = 0.5), then – in expectation – 𝑟J-BOND(𝑦) coincides with the true reward 𝑟BOND(𝑦) = log 𝑝≤(·) = log(0.5). For this purpose, let us consider the parametrized function:

𝛼 if 𝑟(𝑦) < min{𝑟(𝑦′

1), 𝑟(𝑦′

2)} 0 otherwise

𝑟J-BOND𝛼 (𝑦) =

.

Note that the stochasticity of 𝑟J-BOND𝛼 (𝑦) is due to the 2 random anchor samples 𝑦′

2 and its expectation can be computed as:

1, 𝑦′

2∼𝜋𝑡anchor 𝑟J-BOND𝛼 (𝑦) = 𝛼 · ℙ {𝑟(𝑦1′) > 𝑟(𝑦)} ∩ {𝑟(𝑦2′) > 𝑟(𝑦)} + 0 · ℙ["otherwise"] (23) = 𝛼 · (1 − 𝑝≤(𝑦))2, (24)

𝔼𝑦′

1,𝑦′

where we have used the definition of 𝑝≤(𝑦) = ℙ𝑦′∼𝜋𝑡anchor[𝑟(𝑦′) ≤ 𝑟(𝑦)]. Using the expression above, we can find the 𝛼 for which 𝔼𝑦′

2∼𝜋𝑡anchor 𝑟J-BOND𝛼 (𝑦) = 𝑟BOND(𝑦) when 𝑝≤(𝑦) = 0.5: 𝛼 · (1 − 0.5)2 = log(0.5) → 𝛼 = −log(16).

1,𝑦′

We illustrate this in Figure 8, where we plot the expected 𝑟J-BOND(𝑦) reward and the true reward 𝑟BOND(𝑦) as a function of 𝑝≤(·).

[Figure 19]

- Figure 8 | Expected value of the J-BOND reward, i.e., 𝔼𝑦′

2∼𝜋𝑡anchor[𝑟J-BOND(𝑦)] (see Equation (23)), compared to the true (unknown) reward log 𝑝≤(𝑦), as a function of the quantile 𝑝≤(𝑦). By design of 𝑟J-BOND, the two curves coincide when 𝑝≤(𝑦) = 0.5, i.e., when 𝑦 has median reward w.r.t. the anchor distribution.

1,𝑦′

#### B. Additional Experiments

###### B.1. Learned quantile models

Monte-Carlo quantile estimation (Section 4.1) approximates the reward quantiles by sampling multiple times from the reference policy 𝜋ref, for each observed context. While we found it to be very simple

and effective, it may require many sample for an accurate quantile estimation and, in addition, it does not exploit any information about the given context. For instance, assuming we have a good quantile estimation for context 𝑥 and are presented a new context 𝑥′. MC quantile estimates treat 𝑥′ independently from 𝑥, although they may have very similar reward quantiles.

Motivated by this, in this section we explore an alternative approach that aims at learning a contextdependent quantile estimator 𝑝≤𝜃(·), parametrized by parameter 𝜃. The idea is to view quantile 𝑝≤(𝑦) as the parameter of a Binomial random variable 𝑍 where 𝑍 = 𝕀{𝑟(𝑦𝑖) ≤ 𝑟(𝑦)} for 𝑦𝑖 ∼ 𝜋ref. Under such a view, we can interpret 𝑝≤𝜃(·) as the output of a binary classifier and train it via maximum likelihood estimation using the standard binary cross-entropy loss (Cover, 1999):

𝐿(𝜃, 𝑥, 𝑦) = −𝔼𝑦′∼𝜋ref(𝑥) log 𝑝≤𝜃(𝑥, 𝑦)𝕀{𝑟(𝑥,𝑦′)≤𝑟(𝑥,𝑦)} + log(1 − 𝑝≤𝜃(𝑥, 𝑦))𝕀{𝑟(𝑥,𝑦′)>𝑟(𝑥,𝑦)} . (25)

We test such an approach in the abtractive summarization task considered in Section 4. We parametrize 𝑝≤𝜃(·) with a LLM initialized as 𝜋ref and fine-tuned using the loss of Equation (25). Simultaneously, the policy 𝜋𝑡 is fine-tuned using BOND and utilizing 𝑝≤𝜃𝑡(·) as quantile estimator at each step. Notably, we approximate the expectation in Equation (25) via a single sample from 𝜋ref for each prompt in the batch. In Figure 9 we report the backward and forward KL divergences between the training policy and the 𝜋BoN distribution as well as the average log quantiles, and compare them to the ones obtained when running BOND with MC quantile estimation. Note that in both cases, 𝜋BoN is approximated by 32 MC samples during evaluation. When using the learned quantile model 𝑝≤𝜃(·), we observe BOND achieves very comparable KL divergences and log quantiles compared to using MC quantile estimaton. This illustrates that the use of learned quantiles is valid and promising, potentially offering interesting computational advantages in situations where, e.g., 𝑝≤𝜃(·) can be re-used or learned offline with a fixed sample budget.

[Figure 20]

[Figure 21]

[Figure 22]

- Figure 9 | BOND (with 𝑁 = 8) using MC quantile estimates vs. a Learned quantile model.

Finally, we remark that the explored approach is quite naive, and alternative learned quantile models can definitely be derived, e.g., further enforcing ordering in the predicted quantiles, using quantile regression (Dabney et al., 2017), or assuming a pre-specified (e.g., Gaussian) rewards’ distributions.

##### B.2. Additional plots for BOND with Jeffreys divergence objective

We provide additional experiments that complement the ablation of Section 4.2 when running BOND with a Jeffreys divergence objective 𝐽effreys𝛽 for different values of 𝛽. In particular, in Figure 10 we show BOND results when using 𝑁 = 4 (top plots) and 𝑁 = 16 (bottom plots). Results are consistent with what discussed in Section 4.2: When using 𝛽 = 0.5, BOND minimizes both backward and forward KL divergences from 𝜋BoN compared to when solely one of them is used as BOND objective. Moreover, it optimizes the log quantiles similarly to 𝛽 = 1.

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- Figure 10 | BOND with 𝑁 = 4 (top plots) and 𝑁 = 16 (bottom plots), and different values of 𝛽 for the Jeffreys divergence objective (cf. Equation (11)). When using 𝛽 = 0.5 (Jeffreys divergence), BOND

minimizes both backward (left plots) and forward (middle plots) KL divergences from 𝜋BoN, achieving best of both objectives (𝛽 = 0 and 𝛽 = 1). Moreover, it optimizes the reward quantiles (right plots) significantly more than when using 𝛽 = 0.

