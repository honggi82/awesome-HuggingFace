## Optimizing Anytime Reasoning via Budget Relative Policy Optimization

##### Penghui Qi12, Zichen Liu12, Tianyu Pang1, Chao Du1, Wee Sun Lee2, Min Lin1 1Sea AI Lab 2National University of Singapore

###### https://github.com/sail-sg/AnytimeReasoner

350 This completes the proof.

# arXiv:2505.13438v3[cs.LG]7Nov2025

[r(x,y)] (14)

Janytime(ω) = E

###### E

346 This completes the proof.

Anytime reasoning Budget Relative Policy Optimization

b→pB,x→pX ,z→ωω(·|x)

y→ωω(·|x,z→b)

bm → b1 bm · V2 (15)

b1 bm · V1 +

Aˆ>b

[r(x,y)] (14)

Janytime(ω) = E

###### E

1,↑b2 = R →

b→pB,x→pX ,z→ωω(·|x)

y→ωω(·|x,z→b)

#### V1 R

</think> </think>

- 0
- 1

<think> </think>

- r11 = 0 r21 = 0.33

⋯

⋯

⋯

</think>

- r12 = 0.33 r22 = 0.66

- rm1 = 1
- rm2 = 1

Optimize anytime reasoning by maximizing this area

Successprobability

###### V2

</think> </think>

question

</think>

<think>

⋮

</think> </think>

<think>

b1 b2 b3 ⋯ bm

r1n = 0.33 r2n = 1 rmn = 0

Thinking budget (token)

b1 b2 ⋯ bm

Figure 1: Left: We optimize anytime reasoning by sampling thinking budgets from a prior distribution pB and maximizing the rewards at sampled budgets to push up the area under the curve. This objective naturally introduces verifiable dense rewards into the thinking process. Right: Budget Relative Policy Optimization (BRPO) leverages these dense rewards to improve advantage estimation via the Monte Carlo return (R) and an interpolated baseline that combines current progress (V1) and the average return within the rollout group (V2).

### Abstract

Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency and flexibility in both training and deployment. In this work, we present AnytimeReasoner, a novel framework for optimizing reasoning performance under varying thinking budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency.

14

14

39th Conference on Neural Information Processing Systems (NeurIPS 2025).

### 1 Introduction

OpenAI o1 [OpenAI, 2024] and DeepSeek-R1 [Guo et al., 2025] have shown that scaling test-time compute via RL is crucial for LLM reasoning. This involves an extensive thinking process using the chain of thought (CoT) [Wei et al., 2022] before producing an answer. RL is then employed to maximize the outcome reward provided by a rule-based verifier to check the correctness of the generated answer. While RL for LLM reasoning is an active area of research, most existing work focuses on optimizing final performance based on the complete thinking process. This approach can be inefficient in both training and deployment, as long CoTs are costly, especially for online services.

In our work, we focus on optimizing anytime reasoning for LLMs via RL. This is conceptually similar to the anytime algorithms introduced in Dean and Boddy [1988], Zilberstein and Russell [1995], where the system can be interrupted at any point during computation, providing the best possible solution so far and is expected to improve the solution quality when more resources are allocated. Concretely in LLM reasoning, we assume the thinking process can be interrupted at any time, and the model should be able to summarize the best solution from incomplete thinking. This capability can significantly extend the serving capacity for online services with limited computing resources. When there are too many requests to handle, the service can choose to interrupt in-progress requests once the thinking length is able to give sufficient accuracy, reserving longer thinking with better accuracy when resources are available. Moreover, users may want to control the thinking budget

- as in Gemini 2.5[Comanici et al., 2025], but the optimal budget is often agnostic. Compared to budgetaware reasoning[Han et al., 2024], our design supports an economical strategy by incrementally increasing the budget, as it allows for continued thinking and reuses the computation already spent.

To achieve optimal performance for anytime reasoning, we propose sampling the thinking budget from a prior distribution while learning, rather than using a fixed, large budget as in prior work [Liu et al., 2025, Zeng et al., 2025, Luo et al., 2025]. This approach makes the model performance robust to potential interruptions in the thinking process, while incentivizing it to reach correct answers more efficiently. By achieving a balance between token efficiency and thorough exploration [Qu et al., 2025], these models are also able to obtain better performance when given larger budgets.

We investigate how to efficiently train LLMs with RL under sampled thinking budgets. By forcing the model to summarize the answers at predefined thinking budgets (drawn from the support of the prior distribution), we introduce verifiable dense rewards into the reasoning process. These rewards provide richer signals and better credit assignment during training [Qu et al., 2025, Cui et al., 2025a]. We also propose a novel variance reduction technique termed Budget Relative Policy Optimization (BRPO) that advances beyond GRPO [Shao et al., 2024] to improve training stability and efficiency under this dense reward framework. As illustrate in Figure 1 (right), we leverage rewards at previous budgets to compute the advantage function, combining with the average return of a group of reasoning trajectories. Empirically, we observe that generating a high-quality summary is critical for both final and anytime performance. Thus, we decouple the optimization of the thinking and summary policies, always sampling from a uniform distribution to derive a better summary policy, thereby improving training efficiency.

We term our overall framework as AnytimeReasoner. Experimental results demonstrate that AnytimeReasoner consistently surpasses GRPO in both final and anytime performance. We conduct extensive ablation studies to evaluate the impact of each component. By independently incorporating decoupled optimization, variance reduction, and budget sampling into GRPO, we observe significant performance enhancements, underscoring the effectiveness of our methods. Notably, even when merely using the maximum token budget (without budget sampling), our method still outperforms GRPO in both standard and anytime reasoning, highlighting the robustness of our approach.

### 2 Methodology

In a training paradigm similar to R1-Zero [Guo et al., 2025], the model is tasked with generating a comprehensive CoT within a designated "thinking box" upon receiving a question. Subsequently, the model summarizes the answer based on this thinking process. A rule-based reward is then calculated according to the summarized answer. The RL objective is to maximize the expected reward:

###### [r(x,y)] (1)

###### Ez ∼ πθ(·|x)

###### Ey ∼ πθ(·|x,z)

J (θ) = Ex ∼ pX

question

answer

thinking process

where x represents the question, z denotes the thinking process, y is the summarized answer, and r(x,y) is the reward function.

In previous studies [Zeng et al., 2025, Liu et al., 2025, Luo et al., 2025], the generation of thinking process and summary are typically sampled together. If the thinking process exceeds the predefined generation limit, the response is considered a negative sample. We contend that this approach is impractical, particularly in online services where a valid summary should be provided even if the thinking process is incomplete. We propose decoupling the generation of the thinking process and its summary, allocating separate token budgets for each. When the thinking process is halted due to budget constraints, we insert ellipses followed by a </think> to prompt the model to produce a summary (see Appendix A), similar to Muennighoff et al. [2025] and Qu et al. [2025].

To differentiate between the thinking and summary policies, we denote the thinking policy as πθ and the summary policy as πϕ. By defining rϕ(x,z) = E

[r(x,y)], the objective can be expressed as:

y∼πϕ(·|x,z)

[rϕ(x,z)]. (2)

J (θ,ϕ) = E

x∼pX ,z∼πθ(·|x)

Given that |y| ≪ |z|, multiple summaries can be sampled to better estimate the expected reward for each thinking process, while incurring only a small computational overhead.

##### 2.1 Optimizing Anytime Reasoning

Test-time scaling [OpenAI, 2024] is crucial for enhancing the reasoning capabilities of LLMs. This concept operates on the premise that increased computational effort during the reasoning process generally leads to better performance. However, in typical RL training setups like R1-Zero-like [Guo et al., 2025], the performance on anytime reasoning is not guaranteed. The reward evaluation is based on the entire thinking process, lacking insight into whether incremental thinking consistently improves performance [Qu et al., 2025].

To optimize anytime reasoning, we propose sampling the thinking budget from a prior distribution rather than using a fixed token budget. Let b represent the token budget for thinking, sampled from a prior distribution pB over a set of increasing budgets {b1,...,bm} (Pj = pB(b = bj) for simplicity). The anytime reasoning objective is:

 

 , (3)

m

Janytime(θ,ϕ) = E

[rϕ(x,z≤b)] = E

Pjrϕ(x,z≤b

)

j

b∼pB,x∼pX ,z∼πθ(·|x)

x∼pX ,z∼πθ(·|x)

j=1

where z≤b is the truncated thinking process at length of the token budget b,

z, if b ≥ |z| truncate(z,b), if b < |z|

z≤b =

.

Instead of focusing solely on the final score based on the entire thinking process as in standard reasoning task, we maximize the expected score over all possible budgets with distribution pB. As illustrated in Figure 1, this is akin to maximizing the area under the score curve when pB is a uniform distribution across every token budget. However, evaluating for all token budgets is impractical and unnecessary, so we evaluate the score only at a small predefined budget support (with m ≤ 8 in our experiments).

It is important to note that this approach transforms the problem into a dense reward framework, introducing verifiable dense rewards for each thinking budget. This facilitates better credit assignment during RL training and enhances the identification of each component’s contribution to a successful reasoning process. As illustrated in Figure 2, the dense rewards for budgets prior to reaching a correct answer are low. However, the cumulative return is relatively higher if the reasoning process ultimately arrives at a correct answer. In contrast, the cumulative return after the first correct answer is relatively low, localizing and highlighting the tokens that contributed to the initial correct answer. This approach is distinct from typical sparse reward RL training for standard reasoning tasks, where all tokens receive the same return. Such sparse reward structures typically lead to unstable and inefficient RL training, while our dense reward approach provides more informative learning signals throughout the entire reasoning process.

Reasoning

𝑥 𝑧 𝑧 𝑧 𝑧

process 𝑥 question

Intermediate reward

𝑟 = 1.0

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

𝑧 thinking

CumulativeReturn 𝑅 = 1.9 𝑅 = 1.9

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

- Figure 2: By introducing dense rewards, we achieve better credit assignment during RL training. We assume a uniform distribution over thinking budgets and omit the probability for simplicity.

Relation to Standard Reasoning Tasks A larger thinking budget is supposed to yield better performance in expectation. Since z≤b is always a prefix of z, the optimal summary policy πϕ∗ should satisfy:

[rϕ∗(x,z)], (4) for any b and x. Then we have:

###### E

[rϕ∗(x,z≤b)] ≤ E

z∼πθ(·|x)

z∼πθ(·|x)

Janytime(θ,ϕ∗) ≤ J (θ,ϕ∗) (5) This justifies the anytime reasoning objective as a lower bound of the standard reasoning objective. Therefore, maximizing performance in anytime reasoning should also enhance performance in standard reasoning tasks. In an extreme case where Pm = 1 (training only with full reasoning length), Janytime falls back to the standard reasoning objective J . For detailed proof, refer to Appendix C.

##### 2.2 Budget Relative Policy Optimization

By defining jt = arg minj bj ≥ t, which represents the nearest token budget after t, the gradient for the thinking policy can be computed as follows:

 

  , (6)

|z|

∇θJanytime(θ, ϕ) = E

∇θ log πθ(zt|x, z<t) (R(x, z, jt) − V (x, z<t))

x∼pX ,z∼πθ(·|x)

t=1

where

m

R(x,z,jt) =

Pjrϕ(x,z≤b

),

j

j=jt

and V (x,z<t) is the variance reduction term, which should be a function correlated to R(x,z,jt) but invariant with respect to zt.

Typically, we set V (x,z<t) = E

[R(x,[z<t,z≥t],jt)], representing the expected future

z≥t∼πθ(·|x,z<t)

return [Sutton and Barto, 2018]. In traditional RL, GAE[Schulman et al., 2015] is often used by estimating this value with a critic model. However, training a critic model for LLM can be both costly and noisy [Guo et al., 2025]. An alternative is sampling-based approach, as in VinePPO [Kazemnejad et al., 2024] and Remax [Li et al., 2023], but this requires significant additional computation across all thinking budgets. Group-based methods, such as GRPO [Shao et al., 2024] and RLOO [Ahmadian

- et al., 2024], treat generation as a bandit and use the average score of multiple responses for variance reduction. However, they are unsuitable in our scenario due to the presence of dense rewards.

In LLM generation, newly sampled tokens (actions) are consistently appended to the existing context (states). This implies that the current context (z<t) always serves as a prefix for any future context ([z<t,z≥t]). This unique property distinguishes it from traditional RL but is often overlooked. Assuming a perfect summary policy that consistently extracts the best answer from the thinking process, the reward should increase monotonically with the number of generated tokens, satisfying rϕ(x,z<t) ≤ rϕ(x,[z<t,z≥t]). Consequently, the current reward rϕ(x,z<t) is correlated with any future reward rϕ(x,[z<t,z≥t]), particularly when t is large enough to yield a correct answer or when |z<t| ≫ |z≥t|. This correlation justifies its use as a suitable baseline for variance reduction.

Building on this insight, we introduce Budget Relative Policy Optimization (BRPO) for efficient variance reduction. Specifically, we employ the following variance reduction term:

jt−1 j=1 λj

m

t−jrϕ(x,z≤b

)

j

Pj, (7)

V1 =

jt−1 j=1 λjt−j

j=jt

Normalized Covariance

###### Normalized Variance

1.0

1.0

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | |C C<br><br>|orr[V1, R orr[V , R|] ]| |
| | | | | | | |C|2<br><br>orr[V, R]| | |
| | | | | | | | | | | |
| | | | | | | | | | | |

V1only (Var[R V1]/Var[R]) V2only (Var[R V2]/Var[R]) BRPO (Var[R V]/Var[R])

0.8

0.8

AMC2022

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

1000 2000 3000 4000 5000 6000 7000 8000

1000 2000 3000 4000 5000 6000 7000 8000

1.0

1.0

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | |C C<br><br>|orr[V1, R orr[V , R|] ]| |
| | | | | | | |C|2<br><br>orr[V, R]| | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.8

0.8

###### AIME2024

0.6

0.6

0.4

0.4

V1only (Var[R V1]/Var[R]) V2only (Var[R V2]/Var[R]) BRPO (Var[R V]/Var[R])

0.2

0.2

0.0

0.0

1000 2000 3000 4000 5000 6000 7000 8000 Token Steps (t)

1000 2000 3000 4000 5000 6000 7000 8000 Token Steps (t)

- Figure 3: Left: The correlation coefficient of V1 and V2 with R(x,z,jt). Right: The normalized variance of our BRPO. We evaluate the R1-Distill-1.5B model under the scenario where λ = 0.5, and pB is a uniform distribution over {1000,2000,...,8000}.

where the evaluated scores at previous budgets, weighted by a discount factor λ, serve as the reward baseline (highlighted in red), and are multiplied by the sum of probabilities after jt to align with the scale of R(x,z,jt).

As illustrated in Figure 3, when t is small, the effectiveness of V1 may diminish because a short thinking process z<t provides limited information. In such cases, we apply a variant of GRPO as a complement. We sample a set of thinking processes {z1,z2,...,zG} and compute:

G

1 G

R(x,zi,jt), (8)

V2 =

i=1

which represents the expected return after jt given the question x. Note that the correlation between V2 and R(x,z,jt) decreases as t increases, as shown in Figure 3, due to differing prefixes (z<t) in these thinking processes.

By combining V1 and V2, the overall variance reduction term is:

jt − 1 m

m − jt + 1 m

V2. (9)

V (x,z<t) =

V1 +

As demonstrated in Figure 3, our BRPO significantly outperforms GRPO in reducing variance, especially when the thinking is long.

##### 2.3 Decoupled Optimization for Thinking and Summary

In a rigorous derivation, the optimization of thinking and summary policies should share the same prior budget distribution pB. However, an optimal summary policy is crucial when the thinking process is incomplete, and its effectiveness is significantly influenced by pB. An imbalanced prior distribution can lead to suboptimal summary policy. To achieve a robust anytime reasoning performance, we decouple the optimization of thinking and summary policies by using a different budget distribution, p′B, for the summary policy. The decoupled gradient of the summary policy with respect to the anytime reasoning objective 3 can be computed as follows:

m

Pj′ E

∇ϕ log(πϕ(y|x, z≤bj))r(x, y) . (10)

∇ϕJanytime(θ, ϕ) = E

x∼pX ,z∼πθ(·|x)

y∼πϕ(·|x,z≤bj )

j=1

In our experiments, we set p′B as a uniform distribution over the budget support {b1,...,bm}. We employ a distinct approach to optimize the summary policy. Specifically, for each question x and

thinking process z≤b

, we sample a group of summaries and use GRPO to stabilize the optimization.

j

Typically, a shared model (ϕ = θ) is used for both thinking and summary policies. In such cases, the overall gradient is:

∇θJanytime(θ) = ∇θJanytime(θ,ϕ) ϕ=θ + ∇ϕJanytime(θ,ϕ) ϕ=θ.

Average

###### AMC 2022

###### AIME 2024

70

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

30

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

60

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

25

| |
|---|

| |
|---|

| |
|---|

Accuracy(%)

45

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

20

50

| |
|---|

| |
|---|

| |
|---|

40

| |
|---|

| |
|---|

| |
|---|

15

| |
|---|

R1-Distill-1.5B

R1-Distill-1.5B

R1-Distill-1.5B

| |
|---|

| |
|---|

35

40

GRPO

GRPO

GRPO

10

| |
|---|

| |
|---|

AnytimeReasoner-base

AnytimeReasoner-base

AnytimeReasoner-base

| |
|---|

AnytimeReasoner-linear

AnytimeReasoner-linear

AnytimeReasoner-linear

30

| |
|---|

5

30

AnytimeReasoner-uniform

AnytimeReasoner-uniform

AnytimeReasoner-uniform

| |
|---|

1000 2000 3000 4000 5000 6000 7000 8000

1000 2000 3000 4000 5000 6000 7000 8000

1000 2000 3000 4000 5000 6000 7000 8000

MATH

###### MINERVA

###### OLYMPIAD_BENCH

30

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

85

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

28

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

80

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Accuracy(%)

40

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

26

| |
|---|

| |
|---|

| |
|---|

| |
|---|

75

35

| |
|---|

| |
|---|

| |
|---|

24

R1-Distill-1.5B

R1-Distill-1.5B

R1-Distill-1.5B

| |
|---|

70

GRPO

GRPO

GRPO

| |
|---|

30

22

AnytimeReasoner-base

AnytimeReasoner-base

AnytimeReasoner-base

| |
|---|

AnytimeReasoner-linear

AnytimeReasoner-linear

AnytimeReasoner-linear

65

25

AnytimeReasoner-uniform

AnytimeReasoner-uniform

AnytimeReasoner-uniform

20

1000 2000 3000 4000 5000 6000 7000 8000

1000 2000 3000 4000 5000 6000 7000 8000

1000 2000 3000 4000 5000 6000 7000 8000

Token Budget

Token Budget

Token Budget

- Figure 4: The comparison of anytime reasoning performance between GRPO and our AnytimeReasoner with various prior budget distributions. Notably, the accuracies at the maximum token budget

(8000) reflect the performance in the standard reasoning task.

### 3 Experiments

We implement our algorithms based on the Verl framework [Sheng et al., 2024], incorporating several key modifications as detailed in Appendix B. We employ Proximal Policy Optimization (PPO) [Schulman et al., 2017] to optimize both thinking and summary policies. For the thinking policy, we use BRPO to compute the advantage function, as detailed in Section 2.2. During training, we allocate four token budgets (m = 4) for thinking: {2000, 4000, 6000, 8000}. For each question, we sample a group of 8 complete thinking processes (stopped either by </think> or when exceeding 8000 tokens). We sample 4 answers to calculate the average score at each thinking budget, which is used to compute the advantage function as in Dr.GRPO [Liu et al., 2025]. The summary length is restricted to 128 tokens. We extract the first answer and use a rule-based verifier to determine the 0/1 outcome reward. As detailed in Section 2.3, we employ different prior distributions for the thinking and summary policies. Unless otherwise specified, the prior distribution p′B for the summary policy is set to a uniform distribution.

We fine-tuned DeepSeek-R1-Distill-Qwen-1.5B [Guo et al., 2025] on 40,315 math problems from DeepScaleR [Luo et al., 2025] for a single epoch, using a batch size of 64 questions per policy iteration. Our experiments were conducted on 8 NVIDIA A100 80G GPUs, with each experiment taking approximately 30 hours to complete (less than 10% overhead in total compared to GRPO). During training, we evaluate the average scores of AIME2024 and AMC2022 every 20 steps and report their performance curves, sampling 32 responses for each question. After training, we assess the final model using five benchmarks: AIME2024 [Li et al., 2024a], AMC2022 [Li et al.,

- 2024a], MATH500 [Hendrycks et al., 2021], Minerva Math [Lewkowycz et al., 2022], and Olympiad Bench [He et al., 2024], with 32 uniform token budgets ranging from 0 to 8000. We compare our methods with GRPO [Shao et al., 2024], incorporating the corrections introduced in Dr.GRPO [Liu

- et al., 2025].

- 3.1 Main Results We consider the following prior distributions pB when optimizing the thinking policy by equation 3:

- • Base: We only optimize the final performance as in standard reasoning task, namely Pm = 1.
- • Uniform: We set pB as a uniform distribution.
- • Linear: We assign probability proportional to the budget length, such that pB(b) ∝ b.

We evaluate the final models after training and plot the score curves under varying thinking budgets in Figure 4. For each question in AMC and AIME, we sample 320 thinking processes to compute the average score. For other datasets, we sample 80 thinking processes per question.

As shown in Figure 4, all variants of our method consistently outperform GRPO by a large margin across varying prior distributions. With small budgets, AnytimeReasoner-uniform excels by prioritizing optimization of these budgets. When the thinking budget is large, AnytimeReasoner with different prior distributions tends to converge to similar performance, demonstrating the robustness of our approach. Notably, even for AnytimeReasoner-base, where we optimize performance only under the maximum thinking budget as in the GRPO baseline, we still achieve significant better performance

- at all thinking budgets. This improvement is due to the decoupled optimization and our variance reduction technique (discussed further in Section 3.2.3). More details and additional evaluations on longer context can be found in Appendix D.

##### 3.2 Ablations

To further investigate which aspects of our framework contribute to performance improvements, we conduct detailed ablations considering three factors: verifiable dense rewards (Section 3.2.1), decoupled optimization (Section 3.2.2), and variance reduction (Section 3.2.3). We report three metrics during training. Anytime Accuracy: the average accuracy over thinking budgets at {2000, 4000, 6000, 8000}. Final Accuracy: the accuracy at the maximum budget (8000). Average Thinking Length: the average thinking length under the maximum budget (8000).

##### 3.2.1 Verifiable Dense Rewards

Anytime Accuracy (%)

###### Final Accuracy (%)

###### Average Thinking Length

5400

GRPO

68

62

GRPO+length_penalty_v1 GRPO+length_penalty_v2 GRPO+linear

5200

66

5000

60

AMC2022

64

4800

58

62

4600

4400

60

GRPO

GRPO

56

4200

GRPO+length_penalty_v1 GRPO+length_penalty_v2 GRPO+linear

GRPO+length_penalty_v1 GRPO+length_penalty_v2 GRPO+linear

58

4000

54

56

3800

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

32

7000 GRPO

- 19

- 20

- 21

- 22

- 23

- 24

GRPO+length_penalty_v1 GRPO+length_penalty_v2 GRPO+linear

30

6800

###### AIME2024

28

6600

6400

26

GRPO

GRPO

6200

24

GRPO+length_penalty_v1 GRPO+length_penalty_v2 GRPO+linear

GRPO+length_penalty_v1 GRPO+length_penalty_v2 GRPO+linear

6000

22

5800

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Step

Training Step

Training Step

- Figure 5: Ablation on verifiable dense rewards. For GRPO+length_penalty_v1, we follow Aggarwal

and Welleck [2025], assigning reward 1 − 0.b2|z|

for the correct answer and 0 for wrong answer. For GRPO+length_penalty_v2, we follow Arora and Zanette [2025] with α as 0.2.

m

We investigate the effectiveness of verifiable dense rewards by modifying the objective of the thinking policy in equation 3 with a linear prior distribution, while keeping the summary policy training consistent with GRPO. Specifically, we use V2 as the variance reduction term to align with GRPO and eliminate the influence of enhanced variance reduction. To demonstrate our method’s superior token efficiency, we compare it against reward shaping, which uses a length penalty on correct answers to encourage concise reasoning [Aggarwal and Welleck, 2025, Arora and Zanette, 2025].

As illustrated in Figure 5, incorporating dense rewards improves both the anytime and final performance. Notably, since our objective diverges from directly optimizing final performance as in the GRPO baseline, the observed improvements can be attributed to enhanced credit assignment facilitated by dense rewards. Another prominent observation is that the average thinking length is clearly shorter than the GRPO baseline under the maximum budget. This is because the thinking policy is encouraged to arrive at a correct answer as quickly as possible, making the model favor

shorter, correct responses. Although reward shaping with length penalty can also reduce the thinking length, it sacrifices the performance and is unstable during training.

##### 3.2.2 Decoupled Optimization

Anytime Accuracy (%)

###### Final Accuracy (%)

###### Average Thinking Length

5400

GRPO

GRPO

GRPO

66

GRPO+decouple

GRPO+decouple

GRPO+decouple

60

5200

64

AMC2022

58

62

5000

60

56

4800

58

54

56

4600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

32

7100

GRPO

GRPO

GRPO

- 19

- 20

- 21

- 22

- 23

- 24

GRPO+decouple

GRPO+decouple

GRPO+decouple

7000

30

6900

AIME2024

28

6800

26

6700

6600

24

6500

22

6400

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Step

Training Step

Training Step

Figure 6: Ablation on decoupled optimization for summary policy. To study the impact of decoupled optimization for thinking and summary policies (detailed in Section

- 2.3), we modify the training of summary policy in GRPO to align with AnytimeReasoner, while keeping the thinking policy training unchanged. Specifically, we sample 4 answers for each thinking budget in {2000, 4000, 6000, 8000}, applying GRPO within each summary group. This approach trains a summary policy under uniformly distributed thinking budgets, while the thinking policy optimizes performance only under the maximum budget (8000).

As shown in Figure 6, the decoupled GRPO clearly outperforms the vanilla GRPO, especially in the AMC benchmark. Notably, the significant improvement in anytime accuracy (the average score under sampled thinking budgets) indicates that decoupled optimization results in a better summary policy for anytime reasoning.

- 3.2.3 Variance Reduction

Anytime Accuracy (%)

###### Final Accuracy (%)

###### Average Thinking Length

5400

GRPO

68

GRPO+vr

62

5300

GRPO+vr+decouple

66

5200

60

AMC2022

64

5100

58

62

5000

4900

60

56

4800

GRPO

GRPO

| |
|---|

58

| |
|---|

| |
|---|

GRPO+vr

GRPO+vr

4700

54

GRPO+vr+decouple

GRPO+vr+decouple

56

4600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

GRPO

34 GRPO

7100 GRPO

26

GRPO+vr

GRPO+vr

GRPO+vr

32

7000

GRPO+vr+decouple

GRPO+vr+decouple

GRPO+vr+decouple

| |
|---|

| |
|---|

24

| |
|---|

6900

30

AIME2024

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

6800

28

22

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

6700

| |
|---|

26

| |
|---|

| |
|---|

| |
|---|

| |
|---|

20

6600

24

6500

22

18

6400

20

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Step

Training Step

Training Step

Figure 7: Ablation on variance reduction.

To evaluate the effectiveness of our BRPO variance reduction (as detailed in Section 2.2), we modified the training of the thinking policy by incorporating BRPO’s variance reduction techniques, while maintaining the summary policy training consistent with GRPO. Specifically, we set m = 4 and P(bm) = 1 in equation 7, aligning the objective exactly with GRPO.

Figure 7 shows that our approach enhances performance on the AIME benchmark. As discussed in Section 3.2.2, the suboptimal summary policy in GRPO may constrain the potential of BRPO’s effectiveness. To address this, we introduced decoupled optimization (detailed in Section 2.3) to improve the summary policy, resulting in further performance gains.

##### 3.3 Evaluation on 7B Model

We also evaluated our approach on a larger model, DeepSeek-R1-Distill-Qwen-7B. For this experiment, we modified the training setup by running two epochs on the DeepScaleR dataset with a batch size of 128 questions per iteration. We incorporated the clip higher technique from DAPO[Yu et al., 2025] to prevent entropy collapse[Cui et al., 2025b] observed in the training. As shown in Figure 8, our AnytimeReasoner framework achieves clearly superior performance in anytime reasoning, with a maximum improvement of about 5 absolute points. For standard reasoning, our methods outperform the GRPO baseline for most of the time during training, despite high variance in the final accuracy..

Anytime Accuracy (%)

Final Accuracy (%)

Average Thinking Length

4750

80

- 78

- 79

- 80

- 81

- 82

- 83

- 84

4500

78

4250

AMC2022

76

4000

74

3750

3500

72

GRPO

GRPO

GRPO

3250

AnytimeReasoner-uniform

AnytimeReasoner-uniform

AnytimeReasoner-uniform

70

AnytimeReasoner-linear

AnytimeReasoner-linear

AnytimeReasoner-linear

3000

AnytimeReasoner-base

AnytimeReasoner-base

AnytimeReasoner-base

68

2750

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

- 47

- 48

- 49

- 50

- 51

- 52

- 53

- 54

46

6500

6250

44

6000

AIME2024

42

5750

5500

40

5250

GRPO

GRPO

GRPO

38

5000

AnytimeReasoner-uniform

AnytimeReasoner-uniform

AnytimeReasoner-uniform

AnytimeReasoner-linear

AnytimeReasoner-linear

AnytimeReasoner-linear

36

4750

AnytimeReasoner-base

AnytimeReasoner-base

AnytimeReasoner-base

4500

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Step

Training Step

Training Step

Figure 8: The training curves for DeepSeek-R1-Distill-Qwen-7B.

### 4 Related Works

Reinforcement Learning with Verifiable Rewards Since the introduction of DeepSeek-R1 [Guo et al., 2025], a growing body of research has adopted the reinforcement learning with verifiable rewards (RLVR) paradigm [Lambert et al., 2024] to improve the reasoning capabilities of large language models (LLMs). SimpleRL [Zeng et al., 2025] provides the first open-source replication of R1-Zero in mathematical domains and analyzes RL dynamics across various base models. Hu et al. [2025] demonstrate that removing the KL regularization used in RLHF [Christiano et al., 2017] improves both RL efficiency and asymptotic performance. Liu et al. [2025] identify an optimization bias in GRPO [Shao et al., 2024] and propose Dr.,GRPO, which applies a Monte Carlo policy gradient method with a baseline [Sutton and Barto, 2018]. While these works improve our understanding of R1-Zero-style training, they still depend on sparse outcome-based rewards, which pose challenges for credit assignment and learning efficiency [Kazemnejad et al., 2024]. In contrast, our method introduces a novel policy optimization framework that leverages cheaply estimated verifiable dense rewards to improve sample efficiency and learning stability.

Token Budget Efficiency of Reasoning Models Previous efforts have studied budgeted reasoning by reducing response length through prompting [Jin et al., 2024, Nayab et al., 2024, Lee et al., 2025, Ma et al., 2025] or adaptive sampling [Yang et al., 2025]. While these training-free approaches can shorten outputs, they often entail a trade-off between conciseness and task performance. More recent work explores token efficiency within online RL frameworks, enabling models to jointly optimize for accuracy and brevity. Yeo et al. [2025] observe that the output lengths on harder questions tend to grow during RL training, and propose a cosine-shaped reward to constrain length. Liu et al. [2025] trace this issue to optimization bias in GRPO and show that correcting it enhances token efficiency.

Further, Arora and Zanette [2025] and Aggarwal and Welleck [2025] apply explicit reward shaping to target shortened or fixed outputs. Our work differs by operating in an anytime reasoning framework, where the reasoning process can be interrupted at anytime and the best-effort solution should be provided [Dean and Boddy, 1988, Zilberstein and Russell, 1995]. Despite not explicitly enforcing conciseness, our objective naturally encourages efficient reasoning, as demonstrated empirically.

Connection to MRT An independent work to ours, MRT [Qu et al., 2025], optimizes test-time compute by minimizing cumulative regret relative to an oracle. Since the oracle is unknown, they employ meta-RL [Xiang et al., 2025, Beck et al., 2023] as an approximation, aiming to maximize the "progress" of each newly generated episode. Despite sharing a similar high-level goal, our formulation fundamentally differs. Rather than minimizing regret, we optimize anytime performance by sampling the thinking budget from a prior distribution, remaining tractable with standard RL techniques. These foundational distinctions lead to significant methodological differences. Firstly, our approach operates on a per-token basis, instead of on episode which is ambiguous and can be hackable in RL if not well handled. Secondly, our method is grounded in principled RL, explicitly accounting for long-term returns. In contrast, MRT adopts a greedy strategy, optimizing the progress of immediate next episode only. Our experimental results also significantly outperform their reported outcomes. We achieve an accuracy of 32.7% compared to their reported 30.3% on AIME 2024.

### 5 Conclusion

The effectiveness of test-time scaling in LLM reasoning is commonly attributed to the generationverification gap [Xiang et al., 2025], where verifying solutions is substantially easier than generating them. During reasoning, the model engages in an iterative search process, exploring potential solutions until a valid one is found. Once generated, the solution is verified for correctness, and this search-verification loop continues until a confident answer is produced.

In this work, we present a framework that systematically exploits this generation-verification gap. Our approach is based on the key observation that verifying answers and extracting them from partial reasoning traces is easy and computationally cheap. Building on this insight, we design our framework to produce answers at some predefined thinking budgets, thereby introducing verifiable dense rewards to enhance RL training. Furthermore, we utilize these additional rewards to construct a more effective variance reduction baseline than GRPO, significantly improving the stability and efficiency of RL training. By integrating these techniques, our framework achieves superior performance in both standard and anytime reasoning tasks.

### References

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv: 2503.04697, 2025.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Daman Arora and Andrea Zanette. Training language models to reason efficiently. arXiv preprint arXiv: 2502.04463, 2025.

Jacob Beck, Risto Vuorio, Evan Zheran Liu, Zheng Xiong, Luisa Zintgraf, Chelsea Finn, and Shimon Whiteson. A survey of meta-reinforcement learning. arXiv preprint arXiv:2301.08028, 2023.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025a.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025b.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in neural information processing systems, 35: 16344–16359, 2022.

Thomas L Dean and Mark S Boddy. An analysis of time-dependent planning. In AAAI, volume 88, pages 49–54, 1988.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Tokenbudget-aware llm reasoning. arXiv preprint arXiv:2412.18547, 2024.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Openreasoner-zero: An open source approach to scaling reinforcement learning on the base model. https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero, 2025.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. The impact of reasoning step length on large language models. In ACL (Findings), 2024.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment. arXiv preprint arXiv:2410.01679, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\" ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Ayeong Lee, Ethan Che, and Tianyi Peng. How well do llms compress their own chain-of-thought? a token complexity approach. arXiv preprint arXiv: 2503.01141, 2025.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13:9, 2024a.

Junyan Li, Delin Chen, Tianle Cai, Peihao Chen, Yining Hong, Zhenfang Chen, Yikang Shen, and Chuang Gan. Flexattention for efficient high-resolution vision-language models. In European Conference on Computer Vision, pages 286–302. Springer, 2024b.

Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://github.com/agentica-project/ deepscaler, 2025.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858, 2025.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Sania Nayab, Giulio Rossolini, Marco Simoni, Andrea Saracino, Giorgio Buttazzo, Nicolamaria Manes, and Fabrizio Giacomelli. Concise thoughts: Impact of output length on llm reasoning and cost. arXiv preprint arXiv: 2407.19825, 2024.

OpenAI. Learning to reason with llms, 2024. URL https://openai.com/index/

###### learning-to-reason-with-llms/.

Yuxiao Qu, Matthew YR Yang, Amrith Setlur, Lewis Tunstall, Edward Emanuel Beeching, Ruslan Salakhutdinov, and Aviral Kumar. Optimizing test-time compute via meta reinforcement finetuning. arXiv preprint arXiv:2503.07572, 2025.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Violet Xiang, Charlie Snell, Kanishk Gandhi, Alon Albalak, Anikait Singh, Chase Blagden, Duy Phung, Rafael Rafailov, Nathan Lile, Dakota Mahan, et al. Towards system 2 reasoning in llms: Learning how to think with meta chain-of-though. arXiv preprint arXiv:2501.04682, 2025.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models. arXiv preprint arXiv: 2504.15895, 2025.

Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

Shlomo Zilberstein and Stuart Russell. Approximate reasoning using anytime algorithms. In Imprecise and approximate computation, pages 43–62. Springer, 1995.

## Appendix

### Table of Contents

- A Implementation Details 14
- B Tree-like Generation and Training 14
- C Relation Between Standard and Anytime Reasoning 15
- D Experimental Results 16

- D.1 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- D.2 Evaluation on 16k Context Length . . . . . . . . . . . . . . . . . . . . . . . . . . 16

### A Implementation Details

We illustrate the implementation details about how we truncate the reasoning process and prompt the model to output an answer.

|<｜User｜>For how many values of the constant $k$ will the polynomial $x^{2}+kx+36$ have two distinct integer roots? Let's think step by step and output the final answer within \boxed{}. <｜Assistant｜><think><br><br>Okay, so I have this problem here: For how many values of the constant \( k \) will the polynomial \( x^{2} + kx + 36 \) have two distinct integer roots? Hmm, let me try to figure this out step by step.<br><br>... So, we have 4 positive and 4 negative pairs, making 8 pairs in total. </think><br><br>**Final Answer** The number of values of \( k \) is \boxed{8}.|
|---|

(a) Thinking is stopped by </think>.

|<｜User｜>For how many values of the constant $k$ will the polynomial $x^{2}+kx+36$ have two distinct integer roots? Let's think step by step and output the final answer within \boxed{}. <｜Assistant｜><think><br><br>Okay, so I have this problem here: For how many values of the constant \( k \) will the polynomial \( x^{2} + kx + 36 \) have two distinct integer roots? Hmm, let me try to figure this out step by step.<br><br>... So, we have 4 positive and 4 negative pairs, making ...<br><br>... </think><br><br>**Final Answer** The number of values of \( k \) is \boxed{8}.|
|---|

(b) Thinking is stopped due to out of budget.

Figure 9: We decouple the generation of thinking and its summary. Given the question, the model first generates the thinking, which can be stopped by a special token </think> or the budget limit. Then we insert ∗ ∗ Final Answer ∗ ∗ (and two ellipsis ··· plus </think> for out of budget cases) to prompt the model to summarize the answer. In training, these inserted tokens will be ignored when calculating the loss.

### B Tree-like Generation and Training

Unlike previous methods with sequential question-response generation and training, our approach employs a tree-like structure. In this section, we introduce how to address implementation challenges for efficient training.

During generation, we use the prefix caching feature of vLLM [Kwon et al., 2023] to reuse computations. We sample a complete thinking process z for a question x, then split it based on predefined token budgets ({i,j,k} in Figure 10). Each partial thinking process is appended with a special endof-think token (</think>), and the model is prompted to output the answer directly (see Appendix A for more details).

𝑦 𝑦 𝑦 𝑦 𝑦 𝑦

𝑥 question

incorrect Answer

Generation correct Answer

𝑥 𝑧 𝑧 𝑧 𝑧

𝑧 thinking

𝑟 = 0.0 𝑟 = 0.5 𝑟 = 1.0

Training

𝑥 𝑧 𝑧 𝑧 𝑧 𝑦 𝑦 𝑦 𝑦 𝑦 𝑦

Figure 10: Our methods utilize a tree-like structure for generation and training.

During training, each response is typically concatenated with its corresponding question using FlashAttention [Dao et al., 2022] for speed. However, this introduces significant duplicated computation for tree-like structures, making it impractical due to high computational demands for LLM training. We implement a tree structure attention mask based on FlexAttention [Li et al., 2024b]. As shown in Figure 10, we append all summaries at the end of the thinking process and record their connection positions in a 1D tensor. This tensor is converted to a block mask by FlexAttention, avoiding 2D tensors that can cause out-of-memory issues for long generation lengths.

### C Relation Between Standard and Anytime Reasoning In this section, we provide a proof for the inequality below:

1

Pm Janytime(θ,ϕ∗). According to equation 4, we have:

Janytime(θ,ϕ∗) ≤ J (θ,ϕ∗) ≤

###### E

[rϕ∗(x,z≤b)] ≤ E

[rϕ∗(x,z)],

z∼πθ(·|x)

z∼πθ(·|x)

Thus, it follows that:

Janytime(θ,ϕ∗) = E

[rϕ(x,z≤b)] ≤ E

###### E

b∼pB

x∼pX ,z∼πθ(·|x)

(11)

[rϕ(x,z)]

x∼pX ,z∼πθ(·|x)

= J (θ,ϕ∗).

Assuming r(x,y) ≥ 0, which is always achievable by adding a constant to each reward, we also have:

Janytime(θ,ϕ∗) = E

[rϕ(x,z≤b)] ≥ E

###### E

b∼pB

x∼pX ,z∼πθ(·|x)

[Pmrϕ(x,z≤b

)]

(12)

m

x∼pX ,z∼πθ(·|x)

= E

[Pmrϕ(x,z)]

x∼pX ,z∼πθ(·|x)

= PmJ (θ,ϕ∗).

Combining 11 and 12, we can get

1

Pm Janytime(θ,ϕ∗). (13) This completes the proof.

Janytime(θ,ϕ∗) ≤ J (θ,ϕ∗) ≤

Algorithm AMC22 AIME24 MATH500 Minerva OlympiadBench Avg. R1-Distill-1.5B 56.4 22.3 81.1 26.3 42.0 45.6 GRPO 65.0 28.9 84.7 28.9 45.9 50.7 AR-base 68.4 32.7 85.5 29.6 47.3 52.7 AR-linear 68.6 32.1 85.6 29.6 47.3 52.6 AR-uniform 68.5 32.2 85.6 29.2 47.2 52.5

- Table 1: The Final Accuracy by evaluating the maximum budget (8000) for the final models.

Algorithm AMC22 AIME24 MATH500 Minerva OlympiadBench Avg. R1-Distill-1.5B 48.2 16.3 74.5 24.1 36.0 39.8 GRPO 53.4 19.0 77.2 26.6 38.8 43.0 AR-base 57.0 21.9 78.2 27.3 40.2 44.9 AR-linear 58.2 22.3 79.0 27.7 40.9 45.6 AR-uniform 58.8 22.9 79.4 27.5 41.2 46.0

- Table 2: The Anytime Accuracy by evaluating 32 budgets (every 250 tokens) for the final models.

### D Experimental Results

##### D.1 Main Results

We present the training curves of our AnytimeReasoner in Figure 11, corresponding to the experiments in Section 3.1. We also evaluate the performance of the models at training step of 600, and report the final accuracy in Table 1 and the anytime accuracy in Table 2.

Anytime Accuracy (%)

Final Accuracy (%)

Average Thinking Length

70

GRPO

5400

64

AnytimeReasoner-uniform

68

AnytimeReasoner-linear

5200

62

66

AnytimeReasoner-base

AMC2022

64

60

5000

62

58

4800

60

GRPO

GRPO

56

AnytimeReasoner-uniform

AnytimeReasoner-uniform

4600

58

AnytimeReasoner-linear

AnytimeReasoner-linear

54

AnytimeReasoner-base

AnytimeReasoner-base

56

4400

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

7200

GRPO

28

34

AnytimeReasoner-uniform

AnytimeReasoner-linear

7000

32

26

AnytimeReasoner-base

AIME2024

30

24

6800

28

22

26

6600

GRPO

GRPO

24

20

AnytimeReasoner-uniform

AnytimeReasoner-uniform

AnytimeReasoner-linear

AnytimeReasoner-linear

6400

22

18

AnytimeReasoner-base

AnytimeReasoner-base

20

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Step

Training Step

Training Step

Figure 11: The training curves for main results.

##### D.2 Evaluation on 16k Context Length

We also verified the effectiveness of our method on longer context lengths by fine-tuning DeepSeekR1-Distill-Qwen-1.5B up to a 16k maximum context. We used a uniform prior over eight thinking budgets: {2k, 4k, 6k, 8k, 10k, 12k, 14k, 16k}. As shown in Figure 12, our approach consistently achieves stronger performance in both anytime and standard reasoning.

Anytime Accuracy (%)

###### Final Accuracy (%)

###### Average Thinking Length

GRPO

GRPO

GRPO

8000

68

70.0

Uniform

Uniform

Uniform

66

67.5

7500

64

AMC2022

65.0

62

7000

62.5

60

60.0

6500

58

57.5

56

6000

55.0

54

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

12000

34

GRPO

36 GRPO

GRPO

Uniform

Uniform

Uniform

11500

32

34

30

32

AIME2024

11000

30

28

10500

28

26

26

10000

24

24

9500

22

22

0 100 200 300 400 500 600

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Step

Training Step

Training Step

Figure 12: The training curves for DeepSeek-R1-Distill-Qwen-1.5B with maximum 16k context length.

