# arXiv:2505.02391v1[cs.LG]5May2025

## Optimizing Chain-of-Thought Reasoners via Gradient Variance Minimization in Rejection Sampling and RL

Jiarui Yao1∗ Yifan Hao1 Hanning Zhang1 Hanze Dong2 Wei Xiong1 Nan Jiang1 Tong Zhang1

1University of Illinois Urbana-Champaign 2Salesforce AI Research

### Abstract

Chain-of-thought (CoT) reasoning in large language models (LLMs) can be formalized as a latent variable problem, where the model needs to generate intermediate reasoning steps. While prior approaches such as iterative reward-ranked fine-tuning (RAFT) have relied on such formulations, they typically apply uniform inference budgets across prompts, which fails to account for variability in difficulty and convergence behavior. This work identifies the main bottleneck in CoT training as inefficient stochastic gradient estimation due to static sampling strategies. We propose GVM-RAFT, a prompt-specific Dynamic Sample Allocation Strategy designed to minimize stochastic gradient variance under a computational budget constraint. The method dynamically allocates computational resources by monitoring prompt acceptance rates and stochastic gradient norms, ensuring that the resulting gradient variance is minimized. Our theoretical analysis shows that the proposed dynamic sampling strategy leads to accelerated convergence guarantees under suitable conditions. Experiments on mathematical reasoning show that GVM-RAFT achieves a 2-4× speedup and considerable accuracy improvements over vanilla RAFT. The proposed dynamic sampling strategy is general and can be incorporated into other reinforcement learning algorithms, such as GRPO, leading to similar improvements in convergence and test accuracy. Our code is available at https://github.com/RLHFlow/GVM.

### 1 Introduction

We consider mathematical reasoning with large language models (LLMs): given a prompt x ∈ X, and aims to produce a correct final answer z ∈ Z. A prevalent approach in this area is the chain-ofthought (CoT) reasoning (Wei et al., 2022), in which the model generates a step-by-step rationale y ∈ Y before outputting the final answer z. In practice, we are typically given a pre-trained and instruction fine-tuned LLM parameterized by θ0 and training samples {(xi,zi)} of prompt-answer pairs. Additionally, we assume access to a verifier r⋆(x,z) → {0,1} that indicates whether a predicted answer is correct or not. This is standard in practice and is particularly popular by the recent DeepSeek-R1 project (DeepSeek-AI et al., 2025), which suggests using only a symbolic verifier rather than training a neural reward model. The goal is to improve model performance by training it to generate high-quality CoT rationales that improve the final answer accuracy on unseen prompts.

We formalize CoT reasoning as a latent variable problem, treating the rationale y as hidden. From this perspective, we propose a new algorithmic framework based on the expectation-maximization (EM)

∗The first two authors contributed equally with random author order, detailed contributions deferred to Appendix A. Emails: {jiarui14, yifanh12, hanning5, wx13, nanjiang, tozhang}@illinois.edu, hanze.dong@salesforce.com.

Preprint. Under review.

algorithm, which we formalize in Section 2. Prior works such as Sordoni et al. (2023); Singh et al. (2023) have shown that this EM framework can be implemented as a variant of iterative reward-ranked fine-tuning (RAFT) (Dong et al., 2023; Touvron et al., 2023), also known as rejection sampling fine-tuning in the literature. Specifically, RAFT iteratively alternates between the expectation and maximization steps:

- 1. E-step: Prompt LLMs to generate n responses per prompt, and keep responses with the highest reward only (with the correct final answers). This process can be thought to be approximating the posterior distribution of the latent variable and the evidence lower bound (ELBO). See Section 2 for a formal presentation.
- 2. M-step: Fine-tune the LLMs on the selected responses from the E-step. The fine-tuned model is used for the next-iteration E-step.

𝜋𝑇 (Final Policy)

GVM Iterations

{𝑛𝑖}𝑖=1𝑛

𝜋𝑡

{𝑝𝑖}𝑖=1𝑛 {𝐺𝑖}𝑖=1𝑛

Accept

Lipschitz Coefficients

Rates

x T times

Stage 2 Budget Assignment

Stage 1 Estimation

𝜋𝑡−1

t=1,2,…,T

𝜋0 (SFT Model)

Figure 1: The demonstration of the whole pipeline for GVM. Starting from π0, which a supervised fine-tuning (SFT) model, GVM will iteratively conduct the estimation and budget assignment process for T times according to the description in Algorithm 1. Each iteration could be decomposed into two stages, and the final policy model will be returned after those iterations.

Although RAFT and its variants have been widely applied to tasks in the post-training stage of LLMs, they are primarily motivated by the idea of imitating the best-of-n sampling rather than variance reduction. As a result, theoretical understanding is often lacking. For instance, these RAFT-style approaches typically adopt a uniform sampling strategy that treats all prompts equally, ignoring differences in sample efficiency or convergence behavior

In this work, we revisit CoT reasoning under the EM framework and provide a deeper theoretical analysis. We identify the key bottleneck as the Monte Carlo estimation of the ELBO gradient during the E-step, which arises due to the intractability of going over all possible CoT rationales. Previous works mainly use the best-of-n sampling to allocate the inference budget uniformly (Sordoni et al., 2023; Singh et al., 2023), which overlook the differences in the convergence rate under different prompts. To address this, we propose a dynamic sample budget allocation strategy that adaptively assigns computational resources across prompts based on theoretical insights. This leads to a more efficient Monte Carlo estimation of the ELBO gradient. Our resulting algorithm, a refined RAFT variant with dynamic inference budget scheduling through Gradient Variance Minimization (GVM-RAFT), achieves substantial performance improvements and even sometimes outperforms state-of-the-art deep RL methods such as GRPO (Shao et al., 2024) in our Qwen-based case studies. In particular, the sample budget allocation schedule itself can be of independent interests and we also extend our techniques to the RL algorithm, where it also brings notable improvements. We summarize our contributions as follows:

- 1. We revisit the EM framework and RAFT in the context of CoT reasoning, and identify that a major limitation of current approaches lies in inefficient stochastic gradient estimation

- caused by uniform and static sampling strategies (i.e., best-of-n sampling), which fail to account for prompt-specific difficulty and convergence behavior.
- 2. Motivated by the goal of minimizing the variance of stochastic gradient, we propose a dynamic sampling strategy that adaptively allocates computational resources based on prompt hardness and gradient norms. Our approach provides both intuitive theoretical insight and rigorous convergence guarantees, establishing a principled framework for efficient onpolicy sampling under computational budget constraints.
- 3. We apply our method to both RAFT++ and GRPO algorithms with real-world experiments on mathematical reasoning tasks. Our results demonstrate that the proposed approach achieves 2-4× speedup in convergence rate and also considerably improve the final test accuracy.

#### 1.1 Related Work

Iterative rejection sampling fine-tuning. A line of works studies the RAFT-type algorithms (iterative rejection sampling fine-tuning) (Dong et al., 2023; Touvron et al., 2023; Gulcehre et al., 2023; Yuan et al., 2023). These algorithms iteratively generate n responses per prompt, discard those with incorrect answers, and fine-tune the LLM on the remaining accepted samples. Another closely related method is STaR (Zelikman et al., 2022), which also involves iterative training on self-generated CoT responses. In comparison, STaR uses greedy decoding rather than sampling, which deviates from the EM principle and suffers from severe bias across prompts of different difficulty levels (see Section 2 for details). STaR also proposes to provide the answer in the prompt to generate CoT responses for harder problems. However, our preliminary experiments suggest that LLMs often generate wrong reasoning path and output the correct answer regardless of the context. Additionally, STaR retrains from the original pre-trained model in each iteration, rather than bootstrapping from the updated model. These algorithms have been widely applied in the post-training stage of LLMs since then (Touvron et al., 2023; Xiong et al., 2025b; AI@Meta, 2024; Yang et al., 2024a). We also mention in passing that this algorithm framework can further date back to the expert iteration in the RL literature Anthony et al. (2017). Lastly, Tong et al. (2024) study inference budget allocation in the context of synthetic data construction. Their approach uses a fixed model to repeatedly sample responses until m correct CoT rationales are obtained per prompt (m is a hyper-parameter). While this setup also studies budget-aware sampling, they only study the offline setting and their method lacks a dynamic scheduling mechanism so cannot be applied to more general iterative online algorithms.

Chain-of-thought reasoning under EM framework. Our work is also closely related with the line of work studying EM framework in CoT reasoning. Singh et al. (2023) connects RAFT with the EM algorithm under the CoT reasoning framework. We will cover such a connection in Section 2 for completeness and also discuss the limitation of the vanilla RAFT algorithm, and motivate our algorithm designs. Another related work is Sordoni et al. (2023), which proposes a gradient-free EM-based approach for prompt selection, emphasizing a different direction. Our work is also closely related to Hoffman et al. (2023), which proposes TRICE that optimizes an ELBO-inspired objective using MCMC to approximate the posterior over the latent CoT rationales. TRICE maintains a table of prompt–CoT pairs and, in each iteration, samples a new CoT y per prompt, and replace the old CoT record yold with y if y leads to correct final answer. They also propose to add a mean-zero baseline to reduce the variance of the gradient estimator (Owen and Zhou, 2000). In comparison, we design a dynamic inference budget allocation scheduling for a more effective estimation of the ELBO gradient by rejection sampling so the algorithm framework and techniques diverge significantly. Moreover, we implement our algorithm in a highly online and on-policy way (see Section 3.3 for details) to achieve the best performance. The lazy update of TRICE in maintaining the table of prompt-CoT pairs may not fit in this scenario. Beyond algorithmic differences, we also provide a theoretical analysis under standard conditions such as smoothness, which is missing in all these prior works. Hu et al. (2024) offers a more detailed statistical perspective on CoT reasoning as latent variable modeling. However, they mainly focus on in-context learning rather than iterative fine-tuning. Lastly, recent work Zhong et al. (2025) proposes to use a neural network to approximate the posterior distribution of latent CoT rationales and train the network by PPO. They then use the network to generate responses for the SFT or DPO algorithms, whose techniques are completely different from ours. They also provide a theoretical convergence analysis of EM algorithm under the KL-regularized Markov Decision Process

(MDP) framework using the techniques from RL theory literature (Zhong et al., 2024). This type of analysis typically ignores the optimization complexity and requires to find the exact minimizer of E-step and M-step at each iteration, thus differing from our techniques and results.

RL algorithms for LLM post training. Reinforcement learning has played a central role in the post-training of LLMs. The most prominent example is reinforcement learning from human feedback (RLHF), which brings the first generation of Chat-GPT3.5 (Ouyang et al., 2022; Bai et al., 2022). These methods typically optimize a learned Bradley-Terry reward model (Bradley and Terry, 1952) using the Proximal Policy Optimization (PPO) algorithm (Schulman et al., 2017). To simplify the RLHF, a long line of works has proposed simplified contrastive learning objectives based on pairwise comparisons, such as Slic-HF (Zhao et al., 2023), DPO (Rafailov et al., 2023), IPO (Azar et al., 2023), GPO (Tang et al., 2024), and their online variants (Xiong et al., 2023; Dong et al., 2024). After the release of OpenAI’s O1 (Jaech et al., 2024) and DeepSeek’s R1 (DeepSeek-AI et al., 2025), RL methods based on verifier scores, which evaluate the correctness of the final answer, have attracted significant attention in building reasoning models (Yu et al., 2025; Shi et al., 2025). This project also follows the recent convention and falls into this framework. In particular, to avoid the critic in PPO, both industry and academic researchers have revisited REINFORCE-style algorithms (Williams and Peng, 1991), including ReMax (Li et al., 2023), RLOO (Ahmadian et al., 2024; Kool et al., 2019), GRPO (Shao et al., 2024), Reinforce++ (Hu, 2025), and Reinforce-rej (Xiong et al., 2025a). We also mention in passing that the RAFT-type algorithms can also be viewed as a special variant of Reinforce-style algorithms, where reward is either 1 or 0. Throughout the rest of this paper, we primarily present our methods with RAFT, since it aligns most naturally with the EM objective. However, the presented dynamic inference budget scheduling can be of independent interests beyond the RAFT algorithm. We will show that integrating it into other RL-style methods, such as PPO, REINFORCE, or GRPO, can also bring notable improvements.

### 2 Problem Formulation and EM Framework

In this section, we formally define the problem, review existing approaches, and motivate our algorithm designs. We consider the chain-of-thought (CoT) reasoning process as:

x → y → z, x ∈ X,y ∈ Y,z ∈ Z,

where x is a prompt, y is the intermediate CoT and z is the final predicted answer. We fit the data [x,y,z] within the following distribution function class:

Ξ := {P(y,z|x,θ) = P(y|x,θ) · P(z|y,θ) | θ ∈ Rp}. (1)

Our target is to learn a good model M(θ) ∈ Ξ, which can minimize the negative log-likelihood of predicting the correct answer:

lnP(z|x,θ), (2)

L(θ) = −Ex∼d

0

where d0 is a prompt distribution and P(·|θ) denotes the distribution induced by the model with parameters θ. While many math datasets include reference CoT rationales, we do not include these reference rationales y here. This is because recent practice typically does not fine-tune LLMs on these CoT rationales after the pre-training/SFT stages (DeepSeek-AI et al., 2025; Team et al., 2025).

The process from x to z can be complex, making it difficult to directly estimate the distribution P(z | x) from the observed data [x,z]. However, by introducing a latent intermediate variable y, the conditional distributions P(y | x) and P(z | y) are often easier to estimate, thereby simplifying the problem:

L(θ) = −Ex∼d

P(y|x,θ)P(z|x,y,θ).

ln

0

y∈Y

Introducing the intermediate CoT y naturally motivates the use of the expectation-maximization (EM) algorithm, which iteratively approximates the posterior over latent variables.

Derivation of the EM algorithm. We consider a training set B = {(xi,zi)}mi=1 drawn from d0 with zi being the labeled ground-truth answer to illustrate the idea. Following the standard derivation

of EM algorithm, we can bound the L(θ) as follows:

 

 

m

m

P(y,zi|xi,θ) Qi(y)

lnP(zi|xi,θ) = −

L(θ) = −

ln

Qi(y)

y∈Y

i=1

i=1

m

P(y,zi|xi,θ) Qi(y)

≤ −

Qi(y)ln

(3)

i=1 y∈Y

m

m

Ey∼Q

i(y) lnP(y,zi|xi,θ)

= −

H(Qi(y))

+

i=1

i=1

JQ(θ)

where Qi(·) is a probability distribution over Y so that Qi(y) ≥ 0 and y∈Y Qi(y) = 1. The last inequality is from the convexity of −ln(·) and Jensen’s inequality (Lemma 2) and H(p) := −Et∼p(t) lnp(t) is the entropy.

In the E-step, our goal is to find a Qi(y) to get a sharper upper bound for L(θ). In particular, the equality is reached with the posterior distribution of y:

P(y|xi,θ) · P(zi|y,θ) y∈Y P(y|xi,θ)P(zi|y,θ)

P(y|xi,θ) · P(zi|y,θ) Z(xi,zi,θ)

P(y,zi|xi,θ) P(zi|xi,θ)

Qi(y) = P(y|xi,zi,θ) =

=

:=

,

(4) where Z(xi,zi,θ) denotes the normalization constant and the second equality uses Equation (1). Here −JQ(θ) − mi=1 H(Qi(y)) is referred to as the evidence lower bound (ELBO)∗ in the literature (Bishop and Nasrabadi, 2006; Kingma et al., 2013), and P(y|xi,zi,θ) is the induced posterior distribution, which is not the equivalent to the autoregressive distribution of CoT when we present the LLMs with prompts and answers.

In the M-step, to minimize L(θ), we can fix Qi(y) as in Equation (4) and indirectly minimize JQ(θ). Note that we do not include the entropy loss because it is a constant in the M-step since we will fix

Qi(y). To summarize, the EM algorithm will alternate between the following two steps: at iteration t,

- 1. E-step: Update the posterior distribution of the latent CoT Qt according to Equation (4) and obtain the JQt defined in Equation (3).
- 2. M-step: The goal of the M-step is to update LLMs M(θt) to minimize JQt:

m

1 m

Ey∼Qt

i(y) lnP(y,zi|xi,θ).

−

i=1

The updated model M(θt+1) is used for the next-iteration E-step.

To apply the EM algorithm, we need to sample y ∼ Qti(y) to approximate the objective. A central challenge is that computing Qti(y) and JQt(θ) requires summing over all possible latent CoT reasoning y ∈ Y because we need to get the normalization constant in Equation (4). This process is usually computationally intractable. Therefore, the objective JQt(θ) must be approximated via sampling.

To approximate the posterior Qi(y), one standard approach is rejection sampling (Neal, 2003). We remark that we refer rejection sampling to the one in statistics, which is used to approximate a

target distribution Qi(y) by a proposal distribution P(y|x,θ), which we can sample from. In the literature of RLHF or LLM, rejection sampling is often referred to as the best-of-n sampling (Bai

et al., 2022; Ouyang et al., 2022). Specifically, to approximate Qi(y) by P(y|xi,θ), the rejection sampling proceeds as follows:

- 1. Sample y ∼ P(y|xi,θ);
- 2. Draw u ∼ Uniform([0,1]);
- 3. Accept y if u ≤ Q

i(y)

P(y|xi,θ)·M , where M > 0 satisfies Qi(y) ≤ M · P(y|xi,θ) for all y ∈ Y.

∗We consider the negative log-likelihood here so it becomes an upper bound

We notice that a valid choice of M is 1/Z(xi,zi,θ), leading to an accept probability of P(zi|y,θ). In mathematical reasoning, given the CoT rationale, the final predicted answer is typically of low randomness. Then, the rejection sampling in statistics aligns well with the best-of-n sampling, where we only keep all the responses with the correct answer. This eventually leads to the RAFTtype algorithms. We remark that this connection between the EM framework and the RAFT-type algorithms has been previously observed in Singh et al. (2023); Sordoni et al. (2023).

However, we argue that the current best-of-n sampling implementation is overly coarse and insufficiently faithful to the true E-step. Specifically, if a prompt xi is very difficult for current M(θt), the density of P(y|xi,θ) will concentrate on the wrong CoT rationales so that P(zi|y,θ) is close to zero. In this case, the accept probability will be very low and we need many samples before we can accept one valid y. In contrast, if our current model M(θt) can already output correct CoT rationale y in most of the time, we will accept most of the generated responses. Eventually, with a fixed inference budget, this uniform allocation (n responses per prompt) tends to bias the accepted samples toward easier prompts with higher acceptance rates.

- Algorithm 1 Meta Algorithm: GVM-EM

- 1: Input: Initial parameters θ0, training samples D = {(xi, zi)}ni=1, number of epochs T, initial posterior Q0 = P(· | θ0).
- 2: for t = 0, . . . , T do
- 3: ▷ E-step (Expectation):
- 4: Sample a set of samples Bt = {xi, zi}mi=1. Update the posterior distribution over latent CoT rationales Qt(·) using Equation (4).
- 5: For each prompt xi, compute the required number of samples nti according to (1) Theoretical Proposition 1 or (2) Practical Algorithm 2.
- 6: Perform rejection sampling to obtain accepted responses y. Collect corresponding (xi, zi, y) into Dit, such that y ∼ Qti(·).
- 7: ▷ M-step (Maximization):
- 8: Update model parameters via gradient descent using:

∇θM(θt) = −

1 m

m

i=1

1 |Dit|

yj∈Dit

∇θ log P(yj, zi | xi, θ).

- 9: end for
- 10: Output: Final model M(θT).

#### 2.1 Notation

The true marginal negative log-likelihood is denoted as L(θ). The negative ELBO at iteration t is denoted as Lt, which is equal to

Lt(θ) = JQt(θ) +

m

H(Qi(y)).

i=1

Our derivation in the next section will focus on JQt(θ) since entropy term is considered to be a constant in M-step when fixing Qt. We also present a notation table in Table 3 to improve readability.

### 3 Gradient Variance Minimization by Dynamic Sample Allocation

To address the limitations of best-of-n sampling, we propose a dynamic inference budget allocation strategy that adapts to the acceptance rates of rejection sampling for each prompt xi. The overall meta-algorithm is presented in Algorithm 1, and in what follows, we describe the budget allocation mechanism in detail.

#### 3.1 Dynamic Inference Budget Scheduling to Minimize Gradient Variance

Unbiased gradient estimation. We begin by formulating the true gradient at iteration t under the EM objective JQt:

m

m

Qti(y)∇lnP(y,zi|xi,θ) = −

∇JQt(θ) = −

i=1 y∈Y

i=1

∇lnP(y,zi|xi,θ), (5)

Ey∼Qt

i

where Qti(y) = P(y|xi,zi,θt−1) is the posterior distribution of y. However, this distribution is intractable to compute exactly. Therefore, we approximate Qti(y) via rejection sampling by drawing nti times from current LLMs P(y|xi,θt−1). This leads to the following unbiased estimator for Equation (5).

Lemma 1 (Unbiased Gradient Estimator). In the iteration t, denoting Dit as the set of accepted samples on y related to (xi,zi), we have the following unbiased gradient estimator for JQt:

m

1 ntipti

∇lnP(yj,zi|xi,θt−1), (6)

−

yj∈Dit

i=1

where pti = Ey∼P(·|x

i,θ)P(zi|y,θ) is the average accept rate of rejection sampling.

Proof. Taking the expectation of Equation (6), we obtain:

 −

 

m

1 ntipti

E

∇lnP(yj,zi|xi,θt−1)

yj∈Dit

i=1

nti

m

1 ntipti

j∼P(·|xi,θt−1) [1(accept yj) · ∇lnP(yj,zi|xi,θt−1)]

Ey

= −

i=1

i=1

nti

m

1 ntipti

ptiEy

j∼Qti∇lnP(yj,zi|xi,θt−1)

= −

j=1

i=1

m

Ey∼Qt

∇lnP(y,zi|xi,θt−1),

= −

i

i=1

which is the same as Equation (5).

| |
|---|

Variance-aware sampling allocation. While the estimator above is unbiased, its variance can vary significantly across prompts. Prompts with low acceptance rates introduce high variance due to the small number of accepted samples. Reducing variance is crucial for efficient training with stochastic gradient, as also emphasized in prior works in statistics and optimization, including Roux et al. (2012); Johnson and Zhang (2013); Defazio et al. (2014); Chen et al. (2018). To design a more efficient sampling strategy, we analyze the variance of the gradient estimator and optimize the

allocation.

 

 

m

1 ntipti

∇(lnP(yj,zi|xi,θ))

V

yj∈Dit

i=1

nti

m

1 (ntipti)2

j∼P(·|xi,θt−1) (1(accept yj)∇(lnP(yj,zi|xi,θ)))

Vy

=

i=1

j=1

nti

m

1 (ntipti)2

j∼P(·|xi,θt−1) ∥1(accept yj)∇(lnP(yj,zi|xi,θ))∥2

Ey

≤

j=1

i=1

nti

m

1 (ntipti)2

j∼Qti ∥∇(lnP(yj,zi|xi,θ))∥2

ptiEy

=

i=1

j=1

m

1 ntipti

∥∇(lnP(y,zi|xi,θ))∥2

Ey∼Qt

=

##### .

i

i=1

G2i

Given a fixed total sampling budget N, we seek to allocate {nti} to minimize this upper bound:

m

m

G2i ptinti

nti = N.

, s.t.

min

i=1

i=1

In practice, some prompts are totally beyond the ability of the current LLMs or cannot be evaluated by the verifier (e.g., due to some label error). This leads to extremely low acceptance rates and unstable gradient estimations. To mitigate this, we introduce a regularization term that penalizes sampling on such prompts. The revised objective becomes:

min

m

G2i ptinti

1 1 + α/(pti)β

i=1

, s.t.

m

nti = N, (7)

i=1

where α > 0,β ≥ 2 are hyperparameters that control the regularization strength. For example, as the accept rate pti → 0, the weight (1+α/(pti)β)−1 → 0, which leads to the fact that sample size nti → 0 and prevents excessive sampling on uninformative prompts. Recent studies such as Xiong et al. (2025a) have also demonstrated the critical role it plays for stable training of online RL algorithms. Solving the regularized optimization problem in Equation (7) yields the following closed-form solution to the optimal sampling allocation: Proposition 1. The optimal number of samples allocated to each prompt is:

Gi/ pti + (ptα

i)β−1

nti = N ·

∝

n l=1 Gl/ ptl + (ptα

l)β−1

Gi pti + (ptα

##### , ∀i = 1,...,m.

i)β−1

The proof is deferred to Appendix D. Remark 1. Accepted sample size has a lower bound as:

m

√

2(α(β − 1))1/(2β) ·

N

i=1

pti pti + (ptα

Gi m l=1 Gl ·

##### .

i)β−1

With Remark 1, our proposed budget scheduling method is not only efficient but also guarantees a sufficient number of accepted samples during the training process, even in the presence of informative prompts. Algorithm 2 shows a practical implementation of the GVM algorithm.

- Algorithm 2 GVM: Practical Implementation

- 1: Input: Model parameter θ, training samples {xi, zi}ni=1, total inference budget N, parameter estimation sample size N′, penalty parameter {α, β}.
- 2: for i = 1, . . . , m do
- 3: For each prompt xi, sample N′ times to get {xi, yij, zij}Nj=1′ .
- 4: Estimate the accept rate pti and Lipschitz bounds Gi on each prompt xi as

pi =

N′ j=1 1(zij = zi)

N′ , Gi =

1≤j≤N′,zij=zi

1 N′pi ∥∇θ ln P(yij, zi|xi, θ)∥2.

- 5: Calculate sample size {ni} as:

ni = N ·

Gi/ pi + (p α

i)β−1

n l=1 Gl/ pl + (p α

l)β−1

.

- 6: end for
- 7: Output: {ni}mi=1.

#### 3.2 Theoretical Result

In this section, we present the theoretical guarantee of loss convergence. Without loss of generalization, we assume that each E-step is followed by k M-steps. For the t-th E-step, let bri denote the batch size for prompt xi at the r-th M-step, where kt − k < r ≤ kt, and the corresponding sample batch is denoted by Bir. The upper bound loss function we construct is denoted by Lt(θ):∗

Lt(θ) := −Ex∼d

Ey∼Qt

i(y) lnP(y,z|x,θ).

0

In further analysis, we take the following notations for simplifying the statement:

T

∆1(k,T) :=

t=1

T

Ω(k,T) :=

t=1

k−1

T

E∥∇θLt(θkt−k+r)∥2 > 0, ∆2(k,T) :=

r=0

t=1

k−1

EV (gkt−k+r) > 0,

r=0

k−1

∇Lt(θkt−k+r)∥2 > 0,

E∥

r=0

where

 −

 .

m

1 m

1 bikt−k+r

∇θ (lnP(yj,zi|xi,θkt−k+r))

V (gkt−k+r) = V

i=1

yj∈Bikt−k+r

Under mild smoothness conditions, we can derive the following result.

- Theorem 1 (Decreasing rate with smoothness condition.). Suppose −lnP(y,z|x,θ) is 1/γ-smooth with respect to θ. If 0 < η ≤ γ, then the proposed algorithm satisfies that

η2 2γ

η 2

E[L(θkT) − L(θ∗)] − E[L(θ0) − L(θ∗)] ≤ −

∆1(k,T) +

Ω(k,T).

- In Theorem 1, with sufficiently large enough sample size, Ω(k,T) will be small enough, which ensures that the right-hand side of the inequality is negative. This guarantees that the loss function decreases at each iteration. Furthermore, if the loss function exhibits convexity, we can derive the following result:

∗The expectation is taken over all sources of randomness, including the sampled data {Dit} and the selected batches {Bir}.

- Theorem 2 (Decreasing rate with smooth and convex condition.). Suppose −lnP(y,z|x,θ) is 1/γ-smooth and convex with respect to θ. If 0 < η ≤ γ/2, then the proposed algorithm satisfies that

η 2k

η 4k

η 4k

E[L(θkT) − L(θ∗)] − E[L(θ0) − L(θ∗)] ≤ −

∆2(k,T) −

∆1(k,T) +

Ω(k,T).

- In Theorem 2, the right-hand side includes an additional negative term, −∆2(k,T), which indicates a faster rate of decrease in the loss function. Specifically, during each E-step, the gradients across M-steps vary only slightly, this implies that:

k−1

k−1

∇Lt(θkt−k+r)∥2 ≈ k2E∥∇Lt(θkt−k)∥22 ≈ k

E∥∇Lt(θkt−k+r)∥2 =⇒ ∆2(k,T) ≈ k∆1(k,T), which further induces that

E∥

r=0

r=0

η 2

η 2k

η 4k

η 2

η 4k

∆1(k,T) ≈

∆1(k,T) >

∆1(k,T), η 4k

∆2(k,T) +

+

η2 2γ

Ω(k,T), as well as

Ω(k,T) <

η2 2γ

η 4k

η 4k

η 2

η 2k

∆2(k,T) −

Ω(k,T) < −

−

Ω(k,T). All of the proofs are deferred to Appendix D.

∆1(k,T) +

∆1(k,T) +

Guaranteed Decrease in Our Proposed Method. From the theorems above, we know that with a sufficiently large sample size, Ω(k,T) becomes small, and the upper bound for E[L(θkT) − L(θ∗)]− E[L(θ0) − L(θ∗)] is strictly negative. This implies that our proposed method consistently decreases the objective loss, thereby ensuring an effective training process.

Reduction of the variance term Ω(k,T). According to the upper bounds derived above, a smaller Ω(k,T) leads to more efficient training. On one hand, increasing the sample size reliably reduces this term. On the other hand, under a finite budget in practice, an effective budget scheduling strategy can also reduce Ω(k,T) and thereby improve training efficiency. Our algorithm achieves this via the scheduling scheme defined in Algorithm 2.

Trade-off in Budget Scheduling Update Frequency. In our algorithm, the budget scheduling is updated every k optimization steps. Updating more frequently (i.e., using a smaller k) improves training efficiency but increases computational overhead. Conversely, updating less frequently reduces computation, but as optimization proceeds, the loss gradient norm ∥∇θLt∥2 tends to become small, which can slow down training. Therefore, choosing an appropriate value of k requires balancing this trade-off between efficiency and computational cost.

#### 3.3 Practical Implementation

Practical implementation of GVM. In this subsection, we describe how to implement the inference budget allocation strategy in practice, as summarized in Proposition 1, and the implementation is referred to GVM - Gradient Variance Minimization. Although the optimal sample sizes are of closed form, the expression involves the average accept rate pti and the gradient norm Gi,t, both of which depend on the posterior Qti(·) and are therefore not available directly. Specifically, suppose that we are given the training set {xi,zi}ni=1 and current LLM M(θ). We can write:

pi = Ey∼P(·|x

i,θ)P(zi|y), Gi = Ey∼Qt

i(·)∥∇lnP(y,zi|xi,θ)∥. We also use sampling to estimate these quantities. We can first generate N′ samples per prompt to get {xi,yij,zij}N

′

j=1. Then, we can compute the following empirical estimators:

N′ j=1 1(zij = zi)

1 N′pi ∥∇θ lnP(yij,zi|xi,θ)∥2.

pi =

N′ , Gi =

1≤j≤N′,zij=zi

Then, we simply plug these empirical estimators into Proposition 1 to get the sample sizes. The entire procedure is summarized in Algorithm 2.

Sample size vs accepte rate

Iteration 2

300 Iteration 3

400 Iteration 4

350

300

250

300

250

200

SampleSize

200

150

200

150

100

100

100

50

50

0

0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Accepted Rate

Accepted Rate

Accepted Rate

- Figure 2: The assigned sample budget in GVM-RAFT++ with N′ = 32,N = 32n for three iterations.

Visualization We present the relationship between the final sample budget and the accept rates of different prompts for GVM-RAFT++ in Figure 2. From the trend, it could be inferred that the sample budget allocated to prompts increases smoothly as the accept rates decrease. However, other affecting factors together determine the trend of the sample budget besides the hardness, where the Lipschitz coefficients measuring the contribution of the samples by the gradients play a role. If we do not take the impact of Lipschitz coefficients {Gi} into account, then we could have the allocated sample budget to prompts with different levels of hardness measured by accept rates as shown in Figure 6.

GVM-RAFT++. We implement GVM and the meta EM Algorithm 1 in a highly online fashion, building on the RAFT++ framework. In each iteration t, we draw a set of prompts {xi,zi} ∼ d0 and use the current model θold to collect Dt = ∪iDit as the replay buffer where the inference budget allocation is determined via Algorithm 2. RAFT++ then uses these samples to compute a stochastic gradient estimator of the objective JQt. To accelerate training, we perform multiple gradient steps per iteration in a mini-batch way. This causes the model distribution to shift away from the distribution used to generate the data. To address this mismatch, RAFT++ incorporates importance sampling and clipping strategies from PPO (Schulman et al., 2017) into the original RAFT, arriving at the following loss function on the prompt-response pair (x,a):

|a|

1 |a|

LRAFT++(θ) =

t=1

r(x,ai) , (8)

min st(θ),clip(st(θ),1−ϵ,1+ϵ) ·I r(x,a) = argmax

i

θ(at|x,a1:t−1)

where st(θ) = π

πθold(at|x,a1:t−1) and at is the t-th token of a. Here, the indicator ensures that we only train on accepted responses: those approximating the posterior via rejection sampling.

Extension to the RL algorithms. While we focus primarily on RAFT-like algorithms, the proposed GVM strategy can be readily adapted to other RL-style fine-tuning algorithms. We focus on the GRPO, which receives significant attention recently due to its successful application to training DeepSeek-R1. Specifically, for each prompt x, GRPO will sample m > 1 responses and compute the following advantage for the t-th token of the i-th response:

ri − mean(r1,···rm) std(r1,··· ,rm)

At(x,ai) =

,

where ri denotes the final reward of the i-th response. This leads to the following loss function for GRPO,

|yi|

m

πθ(yi,t|x,yi,<t) πθ

1 |yi|

1 m

Aˆi,t,

JGRPO(θ) = Ex,{y

min

i}mi=1∼πθold(y|x)

(yi,t|x,yi,<t)

old

t=1

i=1

πθ(yi,t|x,yi,<t) πθ

,1 − ϵ,1 + ϵ A ˆi,t − βDKL[πθ∥πref] .

clip

(yi,t|x,yi,<t)

old

### 4 Experiments and Results

In this section, we validate the effectiveness of the proposed algorithm by real-world experiments. We will implement our approach with both RAFT++ and GRPO.

#### 4.1 Experiments Setup

Datasets and Models To validate the effectiveness of the proposed methods, we conduct experiments with Qwen2.5-Math-1.5B and Qwen2.5-Math-7B (Yang et al., 2024b). We focus on the mathematical reasoning task and use Math-Verify∗ as the verifier.

Hyperparameters We utilize verl (Sheng et al., 2024) as the training framework, and implement the RAFT++ as Xiong et al. (2025a) show that the additional importance sampling and clipping improve over the vanilla RAFT (Dong et al., 2023). For each iteration, we use a prompt batch size of 1024, and use a mini-batch size 256 for gradient update. The max prompt length is set to be 1024, and the models are allowed to generate at most 3072 tokens so that they do not exceed the context window of 4096 tokens. There is no warmup stage and the learning rate is chosen to be a constant 1e − 6. For the training dataset, we use the Numina-Math (LI et al., 2024). For simplicity of implementation, we update the posterior distribution (i.e., the GVM in Algorithm 2) every 10 steps. In other words, we will compute the inference buget allocation for the next 10 M-steps and take 10 M-steps before entering another E-step. The 10 M-steps and the 1 E-step are together referred to as one iteration of GVM-RAFT++. We also experiment with the clip higher (Yu et al., 2025) to use an asymmetric clip threshold design. Specifically, we clip the importance sampling ratio πθ/πθ

to

old

the range (1 − εlow,1 + εhigh), where εlow = 0.2, and εhigh = 0.28,0.4 then 0.35 for RAFT++ and GVM-RAFT++ respectively. We omit this technique in GRPO training as we observe it leads to worse performance in our experiments.

For detailed experiments setup, please refer to Appendix B.

#### 4.2 Main Results

In this section, we summarize the results from integrating GVM into both RAFT++ and GRPO algorithms as a sample budget rebalancing strategy. The performance is measured by Average @ 8, which means we randomly sample 8 instances from the model with a non-zero temperature, and take the average accuracy as the final result. For Qwen2.5-Math-1.5B, we use a temperature of 1.0 in evaluation, while for Qwen2.5-Math-7B, we use a temperature of 0.7 as the entropy loss is higher after integrating the clip higher trick, which leads to more diverse outputs.

Table 1: Performance of different algorithms across five benchmarks including Math500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), Olympiad Bench (He et al., 2024), AIME24, and AMC23. From the results, we could observe that after reweighting the sample size of prompts, GVM-RAFT++ and GVM-GRPO could indeed outperform both RAFT++ without sample size reweighting and GRPO.

|Method|Math500 Minerva Math Olympiad Bench AIME24 AMC23 5 Average<br><br>|
|---|---|
|Qwen2.5 Math-1.5B<br><br>Base GRPO RAFT++ GVM-GRPO GVM-RAFT++<br><br>|56.35 17.00 25.22 3.33 37.81 27.94 70.78 29.00 33.56 10.00 47.19 38.11 69.02 27.71 31.74 9.58 44.06 36.42 73.92 29.96 36.26 12.92 49.06 40.42 72.90 29.04 36.20 9.17 51.88 39.64<br><br>|
|Qwen2.5 Math-7B<br><br>Base GRPO RAFT++ GVM-GRPO GVM-RAFT++<br><br>|42.00 12.82 19.20 12.92 30.00 23.39 81.20 36.03 44.15 20.83 63.12 49.07 81.68 35.85 43.83 20.83 63.12 49.06 81.55 36.26 43.56 22.92 65.00 49.86 81.00 36.67 43.48 22.92 61.56 49.13<br><br>|

GVM Improves Efficiency with Comparable Performance From Table 1, we could conclude that GVM could indeed improve the final performance of both RAFT++ and GRPO when applied on

∗https://github.com/huggingface/Math-Verify

Average accuracy: GVM-RAFT++ vs RAFT++

N = 8, N = 8n

N = 32, N = 32n

0.450

0.425

| |
|---|

AverageAccuracy

| |
|---|

| |
|---|

0.400

| |
|---|

0.375

| |
|---|

0.350

0.325

GVM-RAFT++

GVM-Raft++

0.300

RAFT++

Raft++

| |
|---|

0 20 40 60 80 100 120

0 5 10 15 20 25 30 35 40

Steps

Steps

- Figure 3: The average accuracy of RAFT++ and GVM-RAFT++ with sample sizes 8 and 32, respectively, on Math500, Minerva Math, and Olympiad Bench, with base model Qwen2.5-Math1.5B.

Qwen2.5-Math-1.5B. For Qwen2.5-Math-7B, the final performance is comparable to baselines, under a faster convergence rate. This verifies the effectiveness of sample budget rebalancing using both the accept rates (standing for the hardness) and the Lipschitz coefficients (standing for the gradients) of different prompts.

In Figure 3, we display the step-wise performance of both RAFT++ and GVM-RAFT++ based on Qwen2.5-Math-1.5B with sample sizes per prompt of 8 and 32, respectively. From the results, we could infer that GVM firstly could enhance the convergence rate evidently, with about 2× speedup for N′ = 8,N = 8n and 4× speed up for N′ = 32,N = 32n measured in update steps compared to its vanilla versions. Besides, upon convergence, GVM-RAFT++ could achieve around 1.25% and

- 5% performance improvement for both configurations, respectively.

In addition, we observed that for Qwen2.5-Math-1.5B, when N′ = 8, the training rewards of GVMRAFT++ are higher than RAFT++ itself, and when N′ = 16, the training rewards of GVM-RAFT++ are approximately the same as RAFT++, while when N′ = 32, the training rewards are lower than that of RAFT++. This implies that using a larger N′ for GVM will bias the sample budget to harder prompts, which may be due to the fact that with a larger N′, harder problems are more likely to be solved at least once compared to when N′ is smaller.

GVM Could Be Generalized to RL Algorithms Though our derivation and theoretical proofs are based on EM algorithm, the sampling strategy itself could be disentangled from the pipeline and utilized solely. Take the GRPO algorithm proposed in Shao et al. (2024) as an example. Figure 5 demonstrates that with the same sample budget rebalancing strategy as in RAFT++, GVM-GRPO could achieve similar performance convergence to GVM-RAFT++. This further verifies the usefulness of GVM as a single sampling strategy compared to being deployed in the EM pipeline. Zhong et al. (2025) makes extra assumptions (please refer to Example 3.5) on the reward structure, which enables their framework’s natural generalization to RL from a theoretical perspective. The similar ideas shed light on the generalization of our method to RL algorithms, and we verified the empirical effectiveness of GVM in our experiments, indicating the success of budget reweighting through gradient variance minimization.

#### 4.3 Ablation Study

In the section of the ablation study, we focus on several aspects that might affect the performance of GVM, including the sample sizes N′ and N, together with different policy update algorithms, etc.

GVM Balances the Sampling Budget in a Stable Way Figure 4 displays the performance trend concerning the training steps, with each training step corresponding to exactly 1024 × 4 samples, which means that the definition of “step” here may not be mapped to the actual training step. Take the case N′ = 16,N = 16n for example, under this setting, each iteration is optimized over 1024 prompts, with a total sampling budget 1024 × 16 = 4 × (1024 × 4), therefore corresponding to

- 4 steps in the figure. Though increasing the number of samples in the first stage, N′, could make more prompts be accepted at least once, thus leading to more useful prompts during the training procedure, the convergence rate does not change evidently. This means GVM could possibly estimate the difficulties of the prompts with a relatively small sampling budget to a sufficient threshold, and we could choose small N′ and N in realistic applications.

Average accuracy for different sample sizes of GVM-RAFT++

0.46

0.44

AverageAccuracy

0.42

0.40

0.38

Setting

N = 8, N = 4n N = 8, N = 8n

0.36

N = 16, N = 16n N = 32, N = 32n

0.34

0 5 10 15 20 25 30

Step

- Figure 4: The average accuracy of GVM-RAFT++ with different sample sizes for both stage one and stage two on Math500, Minerva Math and Olympiad Bench. According to the trend, though the number of sample sizes in both stage one and stage two increases, the convergence rate stays almost the same.

0 20 40 60 80 100 120 140

Steps

0.36

0.38

0.40

0.42

0.44

0.46

AverageAccuracy

Qwen2.5-Math-1.5B

GVM-GRPO

GRPO

GVM-RAFT++

RAFT++

0 20 40 60 80 100 120

Steps

0.44

0.46

0.48

0.50

0.52

0.54

Qwen2.5-Math-7B

GVM-GRPO

GRPO

GVM-RAFT++

RAFT++

- Figure 5: The average accuracy of RAFT++, GVM-RAFT++ and GRPO, GVM-GRPO with sample sizes N′ = 8,N = 8n respectively on Math500, Minerva Math and Olympiad Bench. Applying the GVM sample strategy to RL algorithms like GRPO achieves similar results to GVM-RAFT++ compared to vanilla GRPO. Here for vanilla RAFT++ and GRPO, the rollout number per prompt is set to 8 as well.

### 5 Conclusion and Discussion

In summary, we propose a novel dynamic sampling and updating strategy - GVM, which could assign prompt-specific sampling budget in a fine-grained manner. The algorithm consists of two stages: a pre-sampling phase that estimates the difficulty of each instance and computes promptspecific Lipschitz coefficients, followed by an update phase that performs parameter optimization. Our experiments have demonstrated the effectiveness of GVM, achieving faster convergence and even superior final performance under suitable settings compared to other baselines. GVM could improve the convergence for both rejection sampling backed pipelines and then be generalized to RL algorithms like GRPO, which demonstrates the significant potential for adaptively reweighting the sampling and update budget. Finally, we also provide rigorous theoretical analysis and establish performance guarantees for this class of two-stage algorithms.

The experiments are conducted with Qwen series base models, while the effectiveness of GVM still awaits a broader verification on other base models. Besides, we believe GVM could generalize to other RL algorithms like PPO and Reinforce, while more experiments need to be performed to support the hypothesis. These could serve as the directions for possible further explorations.

### References

Ahmadian, A., Cremer, C., Gallé, M., Fadaee, M., Kreutzer, J., Pietquin, O., Üstün, A., and Hooker, S.

(2024). Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740.

AI@Meta (2024). Llama 3 model card. Anthony, T., Tian, Z., and Barber, D. (2017). Thinking fast and slow with deep learning and tree

search. Advances in neural information processing systems, 30.

Azar, M. G., Rowland, M., Piot, B., Guo, D., Calandriello, D., Valko, M., and Munos, R. (2023). A general theoretical paradigm to understand learning from human preferences. arXiv preprint arXiv:2310.12036.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Bishop, C. M. and Nasrabadi, N. M. (2006). Pattern recognition and machine learning, volume 4. Springer.

Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345.

Chen, J., Zhu, J., Teh, Y. W., and Zhang, T. (2018). Stochastic expectation maximization with variance reduction. Advances in Neural Information Processing Systems, 31.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X.,

- Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y.,
- Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y.,

- Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren,

Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z.,

- Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z.

(2025). Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.

Defazio, A., Bach, F., and Lacoste-Julien, S. (2014). Saga: A fast incremental gradient method with support for non-strongly convex composite objectives. Advances in neural information processing systems, 27.

Dong, H., Xiong, W., Goyal, D., Zhang, Y., Chow, W., Pan, R., Diao, S., Zhang, J., SHUM, K., and Zhang, T. (2023). RAFT: Reward ranked finetuning for generative foundation model alignment. Transactions on Machine Learning Research.

Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. (2024). Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863.

Gulcehre, C., Paine, T. L., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., et al. (2023). Reinforced self-training (rest) for language modeling. arXiv preprint arXiv:2308.08998.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al.

(2024). Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021). Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Hoffman, M. D., Phan, D., Dohan, D., Douglas, S., Le, T. A., Parisi, A., Sountsov, P., Sutton, C., Vikram, S., and Saurous, R. A. (2023). Training chain-of-thought via latent-variable inference. In NeurIPS.

Hu, J. (2025). Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262.

Hu, X., Zhang, F., Chen, S., and Yang, Z. (2024). Unveiling the statistical foundations of chain-ofthought prompting methods. arXiv preprint arXiv:2408.14511.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. (2024). Openai o1 system card. arXiv preprint arXiv:2412.16720.

Johnson, R. and Zhang, T. (2013). Accelerating stochastic gradient descent using predictive variance

reduction. Advances in neural information processing systems, 26. Kingma, D. P., Welling, M., et al. (2013). Auto-encoding variational bayes. Kool, W., van Hoof, H., and Welling, M. (2019). Buy 4 reinforce samples, get a baseline for free! Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A.,

Anil, C., Schlag, I., Gutman-Solo, T., et al. (2022). Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857.

LI, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S. C., Rasul, K., Yu, L., Jiang, A., Shen, Z., Qin, Z., Dong, B., Zhou, L., Fleureau, Y., Lample, G., and Polu, S. (2024). Numinamath. https://github.com/project-numina/aimo-progress-prize/blob/main/ report/numina_dataset.pdf.

Li, Z., Xu, T., Zhang, Y., Yu, Y., Sun, R., and Luo, Z.-Q. (2023). Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv e-prints, pages arXiv–2310.

Neal, R. M. (2003). Slice sampling. The annals of statistics, 31(3):705–767.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Owen, A. and Zhou, Y. (2000). Safe and effective importance sampling. Journal of the American Statistical Association, 95(449):135–143.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290.

Roux, N., Schmidt, M., and Bach, F. (2012). A stochastic gradient method with an exponential

convergence _rate for finite training sets. Advances in neural information processing systems, 25. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy

optimization algorithms. arXiv preprint arXiv:1707.06347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y., Wu, Y., and Guo, D. (2024). Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. (2024). Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Shi, T., Wu, Y., Song, L., Zhou, T., and Zhao, J. (2025). Efficient reinforcement finetuning via adaptive curriculum learning. arXiv preprint arXiv:2504.05520.

Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Liu, P. J., Harrison, J., Lee, J., Xu, K., Parisi, A., et al. (2023). Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585.

Sordoni, A., Yuan, E., Côté, M.-A., Pereira, M., Trischler, A., Xiao, Z., Hosseini, A., Niedtner, F., and Le Roux, N. (2023). Joint prompt optimization of stacked llms using variational inference. Advances in Neural Information Processing Systems, 36:58128–58151.

Tang, Y., Guo, Z. D., Zheng, Z., Calandriello, D., Munos, R., Rowland, M., Richemond, P. H., Valko, M., Pires, B. Á., and Piot, B. (2024). Generalized preference optimization: A unified approach to offline alignment. arXiv preprint arXiv:2402.05749.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al.

(2025). Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599. Tong, Y., Zhang, X., Wang, R., Wu, R., and He, J. (2024). Dart-math: Difficulty-aware rejection

tuning for mathematical problem-solving. arXiv preprint arXiv:2407.13690.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Williams, R. J. and Peng, J. (1991). Function optimization using connectionist reinforcement learning algorithms. Connection Science, 3(3):241–268.

Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., and Zhang, T. (2023). Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint.

Xiong, W., Yao, J., Xu, Y., Pang, B., Wang, L., Sahoo, D., Li, J., Jiang, N., Zhang, T., Xiong, C., et al. (2025a). A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343.

Xiong, W., Zhang, H., Ye, C., Chen, L., Jiang, N., and Zhang, T. (2025b). Self-rewarding correction for mathematical reasoning. arXiv preprint arXiv:2502.19613.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. (2024a). Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., Lu, K., Xue, M., Lin, R., Liu, T., Ren, X., and Zhang, Z. (2024b). Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Fan, T., Liu, G., Liu, L., Liu, X., et al.

(2025). Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Yuan, Z., Yuan, H., Li, C., Dong, G., Tan, C., and Zhou, C. (2023). Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825.

Zelikman, E., Wu, Y., Mu, J., and Goodman, N. (2022). Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488.

Zhang, T. (2023). Mathematical analysis of machine learning algorithms. Cambridge University Press.

Zhao, P. and Zhang, T. (2014). Accelerating minibatch stochastic gradient descent using stratified sampling. arXiv preprint arXiv:1405.3080.

Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., and Liu, P. J. (2023). Slic-hf: Sequence likelihood calibration with human feedback. arXiv preprint arXiv:2305.10425.

Zhong, H., Feng, G., Xiong, W., Zhao, L., He, D., Bian, J., and Wang, L. (2024). Dpo meets ppo: Reinforced token optimization for rlhf. arXiv preprint arXiv:2404.18922.

Zhong, H., Yin, Y., Zhang, S., Xu, X., Liu, Y., Zuo, Y., Liu, Z., Liu, B., Zheng, S., Guo, H., et al.

(2025). Brite: Bootstrapping reinforced thinking process to enhance language model reasoning. arXiv preprint arXiv:2501.18858.

### A Author Contributions

This work stems from all authors’ valuable contributions and close collaborations.

JY implements the workflow based on the algorithm pipeline; conducts preliminary experiments and verifies the effectiveness of GVM in the early stage; scales up the experiments; mainly writes the experiments-relevant sections of this paper.

YH initializes the project; contributes to the algorithm proposal and the clarification of the pipeline procedure; derives the theoretical analysis for GVM and establishes corresponding bounds for the algorithm; mainly writes the sections on methodology and theoretical analysis.

HZ devotes to the conduct of experiments; contributes to the ablation studies of the tradeoff between entropy loss and model performance; reviews related works on the techniques to mitigate fast entropy loss in the early stage of RL training and provides insightful suggestions on how to balance the tradeoff.

HD implements baselines and provides practical implementation suggestions; conducts a series of experiments, including baselines comparisons, GVM iterations in both RAFT++ and RL (GRPO) algorithmic settings; provides discussions and suggestions about experiment design; contributes to writing.

WX provides insightful advice through the project timeline on theoretical analysis, experiment design, and paper writing; contributes to the initial codebase and baseline training; mainly writes the introduction, related works, and problem formulation sections; and revises the draft version of this paper.

NJ and TZ support the work and provide computational resources, guidance, and suggestions for theoretical analysis, experiment design and paper writing.

### B Experiment Details

To understand how to choose the hyperparameters α and β, we first simplify the question and omit the impacts of gradients here. Then the sample weight of function f(x) = √ 1

x+xβα−1 with different choices of α and β is visualized as the figure below. From Figure 6, we could see that in order to assign more sample weights on problems with lower accept rates, we should choose both α and β as low as possible within a reasonable region. Therefore, in our experiments, we fix α = 0.001 and β = 2.0. For all the hyperparameters, please refer to Table 2.

Table 2: Full hyperparameters.

#### Parameter Value

α 1e − 3 β 2

batch size 1024 mini batch size 256

max prompt length 1024 max response length 3072

learning rate 1e − 6 KL loss coefficient 0.001

As for the choices of α and β is GVM stage two sample budget assignment, we visualize the effects of both parameters in Figure 6. From the figure, we could see that when β is fixed, the smaller α is, the more sample budget will be allocated to harder prompts. Similarly, when α is fixed, the smaller β is, the more sample budget will be allocated to harder prompts.

Effects of and on f(x) = 1/ x + x1

- =1.0e-04, =2.0

- =1.0e-04, =3.5

- 0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

=1.0e-04, =5.0

- =1.0e-03, =2.0

- =1.0e-03, =3.5

=1.0e-03, =5.0

- =1.0e-02, =2.0

- =1.0e-02, =3.5

x f(x)

=1.0e-02, =5.0

0.0 0.2 0.4 0.6 0.8 1.0

√

x + αx1−β.

Figure 6: Visualization of different choices of α and β for f(x) = 1/

To analyze the impacts of which layers are used in the gradients calculation and the reduction methods of gradients, we conduct a series of experiments using Qwen2-math-7B-Instruct (Yang et al., 2024a). For the layers from which we retrieve the gradients, we choose the top one, lm_head, and the bottom one, embed_tokens, as the activated layers for loss backpropagation and gradients calculation. For gradients reduction, we compare both summation and average by sequence length. From the results shown in Figure 7, we could see that different kinds of layers for gradient calculation do not make a huge difference on the final sample sizes, while the summation of gradients biases the sample sizes to

problems with lower accept rates, which we speculate the correct answers for problems with lower accept rates could steer the model more than easier problems. Therefore, in our experiments, we choose summation as the method for gradients reduction. According to the relationship between the sample sizes and accept rates. The relationship between gradients and accept rates, it could be inferred that in a high level, the gradients are larger for prompts with lower accept rates if we use the average reduction. At the same time, it is not always the case, as prompts with higher accept rates may also have large gradients.

In the second stage of GVM, to make the total sample budget exactly N, we sort the prompts according to their accept rates in descending order, and add more samples to the prompts with lower accept rates. In our experiments, for GVM, each iteration consists of nine steps of update as the amount of training data per iteration is slightly less than 10 × 1024 = 10240, which is the amount of data needed for ten steps of update with train batch size 1024.

For the implementation of GVM-GRPO, we divide the sample budget calculated in the second stage by a fixed size (for example, 4 in our experiments), and copy the prompts multiple times by ni/4. Then we set the rollout number to 4 during the training procedure.

For compute resources, we mainly conduct the experiments on NVIDIA RTX A6000 and H100 GPUs, and each iteration in GVM typically takes 90 minutes with sample sizes N′ = 8,N = 8n on a 4×H100 GPU server.

- C Notation Table All of the notations are listed in Table 3.

Table 3: The table of notations used in this paper. Notation Description

x,y,z Prompt, CoT rationale, and predicted answer X,Y,Z the space of prompt, CoT rationale, and predicted answer m the number of training samples (xi,zi)

Qti(y) the posterior distribution P(y|xi,zi,θ) on the t-th E-step Dit the collected sample set for (xi,zi,y) on the t-th E-step nti the size of scheduling budget for prompt xi on the t-th E-step pti the average accept rate for prompt xi on the t-th E-step N the whole size of sampling budget (α,β) tuning parameters in budget scheduling Bir the sample batch corresponding to xi on the r-th M-step bir the batch size of Bir nti required number of samples for prompt xi on the t-th E-step

k the number of M-step after each E-step θkt−k+r the model parameter on the r-th M-step within the t-th E-step η learning rate Lt(θ) upper bound loss function on the t-th E-step

- D More Theoretical Results and Missing Proofs

If we consider further strongly convex loss function, there will be:

Theorem 3 (Decreasing rate with smooth and strongly convex condition.). Suppose −lnP(y,z|x,θ) is 1/γ-smooth and H-strongly convex with respect to θ. If 0 < η ≤ min{γ/2,1/(4kH)}, then the proposed algorithm satisfies that

Hη2 2k

η 4k

η 8k

E[L(θkT) − L(θ∗)] − E[L(θ0) − L(θ∗)] ≤ −

∆3(k,T) −

Ω(k,T), where

∆1(k,T) +

r−1

T

k

∇Lt(θkt−k+m)∥2 > 0.

E∥

∆3(k,T) :=

t=1

r=1

m=0

Proof of Proposition 1. Based on the objective function in Equation (7), we can define the Lagrange objective function as

m

G({nti},λ) :=

i=1

m

G2i ptinti − λ(

1 1 + α/(pti)β

i=1

Taking derivative on the variables, we could obtain that

nti − N).

G2i pti(nti)2 − λnti, i ∈ [m],

∂G ∂nti

2 1 + α/(pti)β

= −

m

∂G ∂λ

nti.

= N −

i=1

To achieve the quations as

∂G ∂λ

∂G ∂nti

= 0,

= 0, we can obtain the result as

Gi/ pti + (ptα

i)β−1

nti = N ·

∝

n l=1 Gl/ ptl + (ptα

l)β−1

Gi pti + (ptα

##### , ∀i ∈ [m],

i)β−1

which finishes the proof.

| |
|---|

- Proof for Theorem 1. To simplify the analysis, we denote

m

1 m

1 bikt−k+r

##### ∇θ (lnP(yj,zi|xi,θkt−k+r)).

gkt−k+r = −

i=1

yj∈Bikt−k+r

With the fact that −lnP(y,z|x,θt) are 1/γ-smooth, we can obtain

η2 2γ

E[Lt(θkt−k+r+1) − Lt(θkt−k+r)] ≤ −ηE∥∇θLt(θkt−k+r)∥2 +

E∥gkt−k+r∥2,

for any r = 0,1,...,k − 1 on iter-t. Summing the above inequality over all r, and with the fact that EL(θkt) ≤ ELt(θkt) ≤ ELt(θkt−k) = EL(θkt−k), we have

E[L(θkt) − L(θkt−k)] ≤ −η

= −η

η 2

≤ −

k−1

k−1

E∥∇θLt(θkt−k+r)∥2 +

r=0

r=0

k−1

k−1

E∥∇θLt(θkt−k+r)∥2 +

r=0

r=0

k−1

k−1

E∥∇θLt(θkt−k+r)∥2 +

r=0

t=0

η2 2γ

E∥gkt−k+r∥2

η2 2γ

EV (gkt−k+r) + E∥∇θLt(θkt−k+r)∥2

η2 2γ

EV (gkt−k+r),

where the last inequality is due to η/γ ≤ 1. Further summing t over 1,...,T, we can obtain that

k−1

k−1

T

T

η2 2γ

η 2

E∥∇θLt(θkt−k+r)∥2 +

E[L(θkT) − L(θ0)] ≤ −

EV (gkt−k+r).

t=1

r=0

t=1

r=0

- Proof for Theorem 2. The proof techniques are mainly from Zhao and Zhang (2014). To be specific, for each iteration t, we denote that

δkt−k+r = −

1 m

+

m

1 m

i=1

m

i=1

which implies that

1 bikt−k+r

∇θ (lnP(yj,zi|xi,θkt−k+r)),θkt−k+r − θkt−k

yj∈Bikt−k+r

1 bkti −k+r

##### (lnP(yj,zi|xi,θkt−k+r) − lnP(yj,zi|xi,θkt−k)),

yj∈Bikt−k+r

Eδkt−k+r = E⟨∇Lt(θkt−k+r),θkt−k+r − θkt−k⟩ − ELt(θkt−k+r) + ELt(θkt−k) ≥ 0. Based on these notations, we can obtain that ∥θkt−k+r − θkt−k∥2 − ∥θkt−k+r+1 − θkt−k∥2 = 2⟨ηgkt−k+r,θkt−k+r − θkt−k⟩ − ∥ηgkt−k+r∥2

m

1 bikt−k+r

2η m

(lnP(yj,zi|xi,θkt−k+r) − lnP(yj,zi|xi,θkt−k)) − ∥ηgkt−k+r∥2,

= 2ηδkt−k+r −

i=1

yj∈Bikt−k+r

taking expectation of both sides, we have E∥θkt−k+r−θkt−k∥2−E∥θkt−k+r+1−θkt−k∥2 ≥ 2η (ELt(θkt−k+r) − ELt(θkt−k))−η2E∥gkt−k+r∥2. Summing over r = 0,1,...,k − 1, we have

k−1

k−1

η 2

- 1

- 2η

E∥gkt−k+r∥2.

E∥θkt − θkt−k∥2 +

E(Lt(θkt−k+r) − Lt(θkt−k)) ≤ −

r=0

r=0

Also, considering the fact obtained in previous analysis as:

k−1

E[Lt(θkt−k+r+1) − Lt(θkt−k+r)] ≤ −η

r=0

k−1

k−1

E∥∇θLt(θkt−k+r)∥2 +

r=0

r=0

we can combine the two inequalities above, and obtain that

η2 2γ

E∥gkt−k+r∥2,

k−1

1 k

E[Lt(θkt−k+r+1) − Lt(θkt−k)]

E[L(θkt) − L(θkt−k)] ≤ E[Lt(θkt) − Lt(θkt−k)] ≤

r=0

k−1

k−1

η2 2γ

- 1

- 2ηk

η k

1 k

η 2

E∥gkt−k+r∥2

E∥θkt − θkt−k∥2 −

E∥∇θLt(θkt−k+r)∥2 +

≤ −

+

r=0

r=0

k−1

k−1

k−1

η2 2γk

η2 2γ

η 2k

η 2 −

1 k

EV (gkt−k+r)2

E∥∇θLt(θkt−k+r)∥2 +

∇Lt(θkt−k+r)∥2 −

E∥

= −

r=0

r=0

r=0

k−1

k−1

k−1

η 4k

η 4k

η 2k

EV (gkt−k+r)2 ,

∇Lt(θkt−k+r)∥2 −

E∥∇θLt(θkt−k+r)∥2 +

E∥

≤ −

r=0

r=0

r=0

where the last inequality is from η/γ ≤ 1/2. Summing over t = 1,...,T, we have

T

E[L(θkT) − L(θ0)] ≤ −

t=1

T

η 4k

+

t=1

k−1

T

η 2k

∇Lt(θkt−k+r)∥2 −

E∥

r=0

t=1

k−1

EV (gkt−k+r)2 .

r=0

k−1

η 4k

E∥∇θLt(θkt−k+r)∥2

r=0

- Proof for Theorem 3. For each iteration t, we denote that

m

1 m

1 bkti −k+r

∇θ (lnP(yj,zi|xi,θkt−k+r)),θt − θkt−k

δkt−k+r = −

i=1

yj∈Bikt−k+r

m

1 m

1 bkti −k+r

(lnP(yj,zi|xi,θkt−k+r) − lnP(yj,zi|xi,θkt−k))

+

i=1

yj∈Bikt−k+r

H 2 ∥θt − θkt−k∥22,

−

which implies that Eδkt−k+r = ⟨∇Lt(θkt−k+r),θkt−k+r−θkt−k⟩−Lt(θkt−k+r)+Lt(θkt−k)−

H

2 ∥θkt−k+r−θkt−k∥22 ≥ 0. Based on these notations, we can obtain that

∥θkt−k+r − θkt−k∥2 − ∥θkt−k+r+1 − θkt−k∥2

= 2⟨ηsgkt−k+r,θkt−k+r − θkt−k⟩ − ∥ηsgkt−k+r∥2

 

 

m

H 2 ∥θkt−k+r − θkt−k∥2

2η m

1 bkti −k+r

(lnP(yj,zi|xi,θkt−k+r) − lnP(yj,zi|xi,θkt−k)) +

= 2ηδkt−k+r −

i=1

yj∈Bikt−k+r

− ∥ηgkt−k+r∥2, taking expectation of both sides, we have E∥θkt−k+r − θkt−k∥2 − E∥θkt−k+r+1 − θkt−k∥2 ≥2η ELt(θkt−k+r) − Lt(θkt−k) +

H

2 ∥θkt−k+r − θkt−k∥2 − η2E∥gkt−k+r∥2.

Summing over r = 0,1,...,k − 1, we have

k−1

k−1

k−1

1 2η

η 2

H 2

E∥gkt−k+r∥2. Also, considering the fact obtained in previous analysis as:

E∥θkt−k+r−θkt−k∥2+

E∥θkt−k−θkt−k∥2−

E(Lt(θkt−k+r) − Lt(θkt−k)) ≤ −

r=0

r=0

r=0

k−1

k−1

k−1

η2 2γ

E∥∇θLt(θkt−k+r)∥2 +

E∥gkt−k+r∥2, we can combine the two inequalities above, and obtain that E[L(θkt) − L(θkt−k)] ≤ E[Lt(θkt) − Lt(θkt−k)] ≤

E[Lt(θkt−k+r+1) − Lt(θkt−k+r)] ≤ −η

r=0

r=0

r=0

k−1

1 k

E[Lt(θkt−k+r+1) − Lt(θkt−k)]

r=0

k−1

k−1

- 1

- 2ηk

H 2k

η k

E∥θkt − θkt−k∥2 −

E∥θkt−k+r − θkt−k∥2 −

E∥∇θLt(θkt−k+r)∥2

≤ −

r=0

r=0

k−1

η2 2γ

1 k

η 2

E∥gkt−k+r∥2

+

+

r=0

k−1

k−1

k−1

η2 2γ

η2 2γk

η 2k

1 k

η 2 −

∇Lt(θkt−k+r)∥2 −

E∥∇θLt(θkt−k+r)∥2 +

E∥

EV (gkt−k+r)

= −

r=0

r=0

r=0

r−1

r−1

k

k

Hη2 2k

Hη2 2k

∇Lt(θkt−k+m)∥2 −

E∥

EV(gkt−k+m)

−

r=1

m=0

r=1

m=0

k−1

r−1

k−1

k

Hη2 2k

η 2k

η 4k

∇Lt(θkt−k+r)∥2 −

∇Lt(θkt−k+m)∥2 −

E∥∇θLt(θkt−k+r)∥2

E∥

E∥

≤ −

r=0

r=1

m=0

r=0

k−1

η 8k

EV (gkt−k+r),

+

r=0

where the last inequality is from η/γ ≤ 1/2. Summing over t = 1,...,T, we have

E[L(θkT) − L(θ∗)]

r−1

T

k

Hη2 2k

≤ E[L(θ0) − L(θ∗)] −

E∥

t=1

r=1

m=0

k−1

T

η 8k

EV (gkt−k+r).

+

r=0

t=1

T

∇Lt(θkt−k+m)∥2 −

t=1

k−1

η 4k

E∥∇θLt(θkt−k+r)∥2

r=0

| |
|---|

### E Technical Lemmas

Lemma 2 (Jensen’s Inequality). Suppose that ϕ(w) is a convex function on Ω. Consider w1,··· ,wm ∈ Ω, and non-negative numbers α1,··· ,αm ∈ R so that mi=1 αi = 1. Then,

m

m

αiwi) ≤

ϕ(

αiϕ(wi).

i=1

i=1

More generally, let p be a probability measure on Ω, then ϕ(Ew∼pw) ≤ Ew∼pϕ(w). In particular, since ∥ · ∥ is convex (by triangle inequality of the norm), we know that

∥Ez∥ ≤ E∥z∥. Proof. See Proposition A.9 of Zhang (2023) for a proof.

| |
|---|

### F More about Rejection Sampling

A general algorithm pipeline for rejection sampling is displayed in Algorithm 3, while in practice, we directly use the outcome-based reward signal to decide whether a prompt will be chosen or not.

Algorithm 3 Rejection sampling

- 1: Input: probability with current model {P(y|xi, θt−1), P(zi|y, θt−1)}, auxiliary probability q(y|xi, zi), sample size nti, accept rate pti, empty set D
- 2: for j = 1, . . . , nti do
- 3: Obtain a sample point yj from the distribution q(y|xi, zi).
- 4: Sample a random point uj ∼ U(0, 1).
- 5: Calculate Qi(yj) as:

Qi(yj) =

P(yj|xi, θt−1)P(zi|yj, θt−1) y P(y|xi, θt−1)P(zi|y, θt−1)/ y z P(y|xi, θt−1)P(z|y, θt−1)

- 6: if uj ≤ p

t iQi(yj)

q(yj|xi,zi) then

- 7: Enlarge D = D ∪ {yj}.
- 8: end if
- 9: end for
- 10: Output: set D.

embed_tokens gradients mean

0.8

200

0.6

SampleSize

150

Gradients

0.4

100

0.2

50

0

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Accept Rate

Accept Rate

embed_tokens gradients sum

300

200

250

200

SampleSize

150

Gradients

150

100

100

50

50

0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Accept Rate

Accept Rate

lm_head gradients mean

- 0

- 1

- 2

- 3

- 4

- 5

200

150

SampleSize

Gradients

100

50

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Accept Rate

Accept Rate

lm_head gradients sum

200

3000

2500

150

SampleSize

2000

Gradients

100

1500

1000

50

500

0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Accept Rate

Accept Rate

Figure 7: Comparison among different choices of the layers for gradient calculation and the methods of gradients reduction on Qwen2.5-Math-7B.

### G Case Study

Here we display several instances from several benchmarks we used respectively. In these cases, the model trained by vanilla RAFT++ did not generate a correct answer within a fixed budget of attempts, while models trained by GVM-RAFT++ or GVM-GRPO managed to find the correct answers.

Math500 Example

Problem The expression 2 · 3 · 4 · 5 + 1 is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing

(2 · (3 · 4)) · (5 + 1) = 144.

In total, how many values can be obtained from the expression 2 · 3 · 4 · 5 + 1 by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).

#### Answer 4

RAFT++ Output Given the expression 2 · 3 · 4 · 5 + 1, we can get different results by inserting parentheses in various ways. Let’s denote the expression with parentheses as E. The expression 2 · 3 · 4 · 5 + 1 without parentheses is evaluated as:

2 · 3 · 4 · 5 + 1 = 120 + 1 = 121. Next, let’s consider the possible ways to insert parentheses:

1. (2 · 3) · 4 · (5 + 1) 2. 2 · (3 · 4) · (5 + 1) 3. (2 · 3 · 4) · 5 + 1 4. 2 · (3 · (4 · 5) + 1) 5. (2 · 3 · 4 · 5) + 1 However, not all of the above are valid expressions because we need to evaluate the expression step by step. For simplicity, let’s evaluate each possible corrected expression:

- 1. (2 · 3) · 4 · (5 + 1) = 6 · 4 · 6 = 144 2. 2 · (3 · 4) · (5 + 1) = 2 · 12 · 6 = 144 3. (2 · 3 · 4) · 5 + 1 = 24 · 5 + 1 = 120 + 1 = 121 4. 2 · (3 · (4 · 5) + 1) = 2 · (3 · 20 + 1) =
- 2 · (60 + 1) = 2 · 61 = 122 5. (2 · 3 · 4 · 5) + 1 = 120 + 1 = 121 From the above evaluations, we see that we get the values 144, 144, 121, 122, and 121, which means the distinct values are 121, 122, and 144. So the total number of distinct values we can obtain is 3. The answer is:

|3|
|---|

GVM-RAFT++ Output Given the expression 2 · 3 · 4 · 5 + 1, we need to determine how many distinct values can be obtained by inserting parentheses in different ways. First, let’s note that without parentheses, the expression evaluates to 2·3·4·5+1 = 120+1 = 121. When we insert parentheses, the expression can take different forms based on where the parentheses are placed. Let’s evaluate all possible ways of inserting parentheses:

- 1. (2 · 3 · 4) · (5 + 1) = (24) · 6 = 144 2. 2 · (3 · 4 · 5) + 1 = 2 · 60 + 1 = 120 + 1 = 121

3. 2 · 3 · (4 · 5 + 1) = 2 · 3 · 21 = 126 4. (2 · 3) · (4 · 5 + 1) = (6) · 21 = 126 5.

- 2 · (3 · 4 · 5 + 1) = 2 · 61 = 122 By evaluating these different forms, we see that the possible values are 121, 122, 126, and 144. So the distinct values that can be obtained are 121, 122, 126, and 144. Therefore, the total number of distinct values is 4.

|4|
|---|

The final answer is

.

Minerva Math Example

Problem Preamble: A spectrum is taken of a single star (i.e., one not in a binary). Among the observed spectral lines is one from oxygen whose rest wavelength is 5007. The Doppler shifted oxygen line from this star is observed to be at a wavelength of 5012. The star is also observed to have a proper motion, µ, of 1 arc second per year (which corresponds to ∼ 1.5 × 10−13 radians per second of time). It is located at a distance of 60pc from the Earth. Take the speed of light to be 3 × 108 meters per second. What is the component of the star’s velocity parallel to its vector to the Earth (in kilometers per second)? Answer To find this longitudinal velocity component, we use the Doppler shift, finding

|300|
|---|

km/s.

Vr = ∆λλc = 50005 c =

RAFT++ Answer To solve this problem, we need to follow these steps:

1. Calculate the radial velocity of the star using the Doppler effect. 2. Use the radial velocity and the proper motion to find the component of the star’s velocity parallel to its vector to the Earth.

- Step 1: Calculate the radial velocity The radial velocity, v, of the star can be found using the formula for the Doppler shift of a photon:

v = c

λobs − λrest λrest

where λobs is the observed wavelength and λrest is the rest wavelength. Given: - λrest = 5007Å - λobs = 5012Å - c = 3 × 108 m/s We first need to convert the wavelengths from to meters:

1Å = 10−10 m So:

λrest = 5007 × 10−10 m λobs = 5012 × 10−10 m

Now we can calculate v:

v = 3 × 108

5012 × 10−10 − 5007 × 10−10 5007 × 10−10

v = 3 × 108

5 × 10−10 5007 × 10−10 v = 3 × 108

5 5007 v =

15 × 108

5007 ≈ 29956m/s v ≈ 29.96km/s

- Step 2: Calculate the parallel velocity component The proper motion µ is given as 1.5 × 10−13 rad/s, and the distance to the star is 60pc. We convert 1pc = 3.086 × 1016 m:

##### distance = 60 × 3.086 × 1016 m = 1.8516 × 1018 m

The radial velocity v is the component of the star’s velocity parallel to its vector to the Earth plus the effect of the proper motion:

##### v = v∥ + µ × d

where v∥ is the component of the star’s velocity parallel to its vector to the Earth, µ is the proper motion, and d is the distance.

Since we calculated v ≈ 29.96km/s = 29960m/s, we can solve for v∥:

##### v∥ = v − µ × d

v∥ = 29960 − (1.5 × 10−13 × 1.8516 × 1018) v∥ = 29960 − 2777.4 v∥ ≈ 27182.6m/s

v∥ ≈ 27.18km/s So the component of the star’s velocity parallel to its vector to the Earth is approximately 27.18km/s. The final answer is:

|27.18|
|---|

GVM-GRPO Answer To solve this problem, we need to determine the component of the star’s velocity parallel to its vector to the Earth. We can break this into two parts:

- 1. **Rest-wavelength shift using the Doppler effect:** The Doppler effect tells us that the

observed wavelength λobs of a spectral line with rest wavelength λrest and velocity v relative to the observer is given by:

λobs = λrest 1 +

v c

Here, λobs = 5012 Å, λrest = 5007 Å, and c = 3 × 108 m/s. We can rearrange the equation to solve for v:

λobs λrest

= 1 +

v c

5012 5007

= 1 +

v c

5012 5007 − 1 =

v c

5012 − 5007 5007

=

v c

5 5007

=

v c

v =

5

5007 × 3 × 108 m/s Converting the velocity from m/s to km/s:

v =

5 5007 × 3 × 108 ×

1 1000

km/s =

5 × 3 × 105 5007

km/s ≈ 299.5 km/s

- 2. **Proper motion effect:** The proper motion µ of the star is given as 1 arc second per year. This can be converted to an angular velocity ω using the formula:

1 arc second 1 year ×

1 year 3.154 × 107 seconds

1 3.154 × 107

arc seconds per second The proper motion µ is in arc seconds per year, so we need to convert it to radians per second:

ω =

=

1 radian 206265 arc seconds ×

1 year 3.154 × 107 seconds

µ =1 arc second/year ×

1 206265 × 3.154 × 107

radians/second ≈ 1.5 × 10−13 radians/second The component of the star’s velocity parallel to its vector to the Earth (v∥) is given by:

=

##### v∥ = µ × d where d is the distance to the star in meters (60 pc = 60×3.086×1016 m = 1.8516×1018 m): v∥ = 1.5 × 10−13 × 1.8516 × 1018 m/s = 277.74 km/s

Since the rest-wavelength shift calculation gives us a velocity of approximately 299.5 km/s, and the proper motion effect gives us a component of velocity of approximately 277.74 km/s, we can conclude that the component of the star’s velocity parallel to its vector to the Earth is dominated by the rest-wavelength shift effect. Therefore, the final answer is:

|300 km/s|
|---|

Olympiad Bench Example

Problem A cubic sequence is a sequence of integers given by an = n3 +bn2 +cn+d, where b,c and d are integer constants and n ranges over all integers, including negative integers.

Determine the possible values of a2015 · a2016 for a cubic sequence satisfying the condition in part (a). Answer 0

RAFT++ Output First, let’s understand the cubic sequence an = n3 + bn2 + cn + d. We need to find the possible values of a2015 · a2016. Given that an is a cubic sequence, it can be expressed in the form an = n3 + bn2 + cn + d. Therefore,

- a2015 = 20153 + b · 20152 + c · 2015 + d
- a2016 = 20163 + b · 20162 + c · 2016 + d

The product of these two terms can be expressed as:

##### a2015 · a2016 = (20153 + b · 20152 + c · 2015 + d) · (20163 + b · 20162 + c · 2016 + d)

Since the sequence terms are given by a polynomial, and the polynomial’s coefficients can take any integer value, the product a2015 · a2016 could theoretically take any integer value depending on the integers b, c, and d. Hence, the possible values of a2015 · a2016 span all integers Z. Thus, the set of possible values of a2015 · a2016 is all integers Z. The final answer is:

|Z|
|---|

GVM-GRPO Output The general form of a cubic sequence is given by an = n3 + bn2 + cn + d. We need to find the possible values of a2015 · a2016. First, let’s write down the expressions for a2015 and a2016:

- a2015 = 20153 + b · 20152 + c · 2015 + d
- a2016 = 20163 + b · 20162 + c · 2016 + d

To find a2015 · a2016, let’s look at the difference between consecutive terms of the sequence: an+1 − an = [(n + 1)3 + b(n + 1)2 + c(n + 1) + d] − [n3 + bn2 + cn + d] = [(n + 1)3 − n3] + [b(n + 1)2 − bn2] + [c(n + 1) − cn] = [(n + 1 − n)((n + 1)2 + (n + 1)n + n2)] + [b((n + 1)2 − n2)] + [c(n + 1 − n)] = [1 · (n2 + 2n + 1 + n2 + n + n2)] + [b(n2 + 2n + 1 − n2)] + [c] = 3n2 + 3n + 1 + b(2n + 1) + c = 3n2 + 3n + 1 + 2bn + b + c = 3n2 + (3 + 2b)n + (1 + b + c)

Since b,c are constants, the difference an+1 − an is a quadratic function of n. This means an itself must be a cubic function, which confirms the form an = n3 + bn2 + cn + d. Now, let’s look at the product a2015 · a2016:

- a2015 = 20153 + b · 20152 + c · 2015 + d
- a2016 = 20163 + b · 20162 + c · 2016 + d

The product a2015 · a2016 can be simplified by noting that:

##### a2015 · a2016 = (20153 + b · 20152 + c · 2015 + d) · (20163 + b · 20162 + c · 2016 + d)

This can be simplified further by noting that:

a2016 = a2015 + (20163 − 20153) + b(20162 − 20152) + c(2016 − 2015)

= a2015+(2016−2015)(20162+2016·2015+20152+2016+2015)+b(2016−2015)(2016+2015)+c

= a2015 + (20162 + 2016 · 2015 + 20152 + 2016 + 2015) + b(2016 + 2015) + c

Since the terms are cubic, the exact values of a2015 and a2016 depend on the specific values of b,c, and d, but the general form shows that a2015 and a2016 are very close together. Specifically, since 20163 is very large compared to 20153, the terms involving b,c, and d may cancel out in the product.

Thus, the product a2015 · a2016 is essentially the square of a2015 plus some constant terms. This suggests that the only possible values for a2015 · a2016 are those that can be expressed as the square of an integer. Therefore, the possible values of a2015 · a2016 are the squares of the product of integers. The final answer is:

|0|
|---|

This is because the only integers a2015 and a2016 can satisfy are 0, given the cubic nature of the sequence and the sequence behavior around n = 2015 and n = 2016.

AIME24 Example Problem A list of positive integers has the following properties:

- • The sum of the items in the list is 30.
- • The unique mode of the list is 9.
- • The median of the list is a positive integer that does not appear in the list itself. Find the sum of the squares of all the items in the list. Answer 236

RAFT++ Output Given the conditions of the problem, we can determine the structure of the list step by step. Let’s denote the list of positive integers as a1,a2,...,an.

1. The sum of the items in the list is 30. 2. The unique mode of the list is 9. Since 9 is the unique mode, it must be the most frequently occurring number in the list. 3. The median of the list is a positive integer that does not appear in the list itself. Let’s start by considering the minimum possible values for n. Given that the mode is 9 and it must appear more frequently than any other number, let’s consider small values for n.

- If n = 5, the list would have to include 9 three or more times to be the mode. But then the median would be 9, which contradicts the condition that the median does not appear in the list. So n ≥ 6.
- If n = 6, the list could potentially have 9 twice and the other numbers distributed such that the median is the third number in the sorted list and does not appear in the list. If the median is 6, the list [6,9,9,a,b,c] would satisfy the conditions, where a,b,c ≤ 6 and a,b,c ̸= 9. The sum of the numbers should be 30:

##### 6 + 9 + 9 + a + b + c = 30 =⇒ a + b + c = 6

Since a,b,c ≤ 6 and they are distinct integers, the only possible values for a,b,c are 1,2,3. Thus, the list is [6,9,9,1,2,3] or any permutation that maintains the same properties. The sum of the squares of the numbers is:

##### 62 + 92 + 92 + 12 + 22 + 32 = 36 + 81 + 81 + 1 + 4 + 9 = 212

We can now verify that this is correct by checking the conditions. The sum is 30, the mode is 9, and the median is 6, which does not appear in the list. The final answer is:

|212|
|---|

GVM-GRPO Output To solve this problem, we need to carefully consider the properties of the list of positive integers:

1. The sum of the items in the list is 30. 2. The unique mode of the list is 9. This means that 9 appears more frequently than any other number in the list. 3. The median of the list is a positive integer that does not appear in the list itself.

Let’s denote the list of integers as [a1,a2,...,an] where a1 ≤ a2 ≤ ... ≤ an. Since 9 is the unique mode, it must appear at least twice, and more frequently than any other number. The median is the middle value when the list is sorted. Since the median is not in the list, n must be odd. Let’s assume n = 2k + 1. The median is then ak+1. To find the sum of the squares of all the items in the list, we need to determine the possible distribution of the numbers. Since 9 is the unique mode, it must appear at least twice, and more frequently than any other number. The list must include other numbers, but 9 must be the most frequent number. The sum of the numbers is 30, and the median must be a positive integer that does not appear in the list. Let’s start by assuming the simplest case where 9 appears exactly twice. This means a1 = a2 = 9. The list then looks like [9,9,a3,a4,...,a2k+1]. The sum of these numbers is 30, so 9 + 9 + a3 + a4 + ... + a2k+1 = 30. This simplifies to 18 + a3 + a4 + ... + a2k+1 = 30, or a3 + a4 + ... + a2k+1 = 12. The median ak+1 must be a positive integer that does not appear in the list. If k = 1, then n = 3, and the list is [9,9,a3]. The median is a2 = 9, which is not a valid solution since 9 must appear more frequently than any other number. If k = 2, then n = 5, and the list is [9,9,a3,a4,a5]. The median is a3. Since 9 appears twice, a3 must be different from 9. The sum of the other three numbers is 12. The possible combinations are (1,1,10), (1,2,9), (1,3,8), (1,4,7), (1,5,6), (2,2,8), (2,3,7), (2,4,6), (3,3,6), (3,4,5). However, since 9 must be the mode, a3 cannot be 9. Also, a3 must be a positive integer that does not appear in the list. The only valid combination is (1,2,9), but 9 is not a valid median. The next valid combination is (1,3,8), where the median a3 = 3. This means the list is [9,9,3,a4,a5], and the remaining sum is 12 − 3 = 9. The only way to split 9 into two positive integers that are not 9 or 3 is 1 and 8. So the list is [9,9,3,1,8]. The sum of the squares of these numbers is 92 + 92 + 32 + 12 + 82 = 81 + 81 + 9 + 1 + 64 = 236. Therefore, the sum of the squares of all the items in the list is

|236|
|---|

.

AMC23 Example Problem Rows 1, 2, 3, 4, and 5 of a triangular array of integers are shown below.

- 1 1 1 1 3 1 1 5 5 1 1 7 11 7 1 Each row after the first row is formed by placing a 1 at each end of the row, and each interior entry is 1 greater than the sum of the two numbers diagonally above it in the previous row. What is the units digits of the sum of the 2023 numbers in the 2023rd row?

#### Answer 5

RAFT++ Output We need to determine the units digit of the sum of the 2023rd row in the triangular array. Observing the Behavior of Units Digits 1. **First Row:** 1 (units digit is 1) 2. **Second Row:** 1 1 (units digit is 1 + 1 = 2) 3. **Third Row:** 1 3 1 (units digit is 3) 4. **Fourth Row:** 1 5 5 1 (units digit is 1 + 5 = 6) 5. **Fifth Row:** 1 7 11 7 1 (units digit is

- 7 + 11 = 18, units digit is 8) 6. **Sixth Row:** 1 (7+19) 1 (units digit is 9 + 9 = 18, units digit is 6) If we continue observing this pattern, we notice that the units digits of the sums of the diagonals in each row follow a specific pattern which repeats every 2 rows: - First row: 1 Second row: 2 - Third row: 1 - Fourth row: 6 - Fifth row: 8 - Sixth row: 6

We notice that the units digits of the sums for rows 3,5,7,9,... are 1, 8, 1, 8, ... and for rows

- 2,4,6,8,... are 2, 6, 6, 6, ... Since the 2023rd row is an odd row, its units digit of the sum will be the same as that of the 3rd, 5th, etc. rows, which is 1. Therefore, the units digit of the sum of the 2023rd row is

|1|
|---|

.

GVM-RAFT++ Output To solve this problem, we need to analyze the triangular array and find a pattern. Let’s start by examining the structure of the array.

1. The first row is: 1 2. The second row is: 1 1 3. The third row is: 1 3 1 4. The fourth row is: 1 5 5 1 5. The fifth row is: 1 7 11 7 1 The pattern for each interior entry is that it is 1 greater than the sum of the two numbers diagonally above it in the previous row. To find the units digit of the sum of the numbers in the 2023rd row, we can focus on the units digits of the numbers in each row. Notice that each row starts and ends with 1. The units digits of the interior numbers follow a specific pattern. Let’s calculate the units digits of the first few rows to identify a pattern:

- Row 1: 1 (units digit: 1) - Row 2: 1 1 (units digits: 1, 1) - Row 3: 1 3 1 (units digits: 1,

- 3, 1) - Row 4: 1 5 5 1 (units digits: 1, 5, 5, 1) - Row 5: 1 7 11 7 1 (units digits: 1, 7, 1, 7, 1) We can see that the units digits repeat a pattern. Let’s look at the units digits of the interior numbers more closely:

- Row 1: 1 (units digit: 1) - Row 2: 1 1 (units digits: 1, 1) - Row 3: 1 3 1 (units digits: 1,

- 3, 1) - Row 4: 1 5 5 1 (units digits: 1, 5, 5, 1) - Row 5: 1 7 11 7 1 (units digits: 1, 7, 1, 7, 1) It seems that the units digits of the interior numbers in each row follow a pattern where the units digit of each interior number is determined by the sum of the two units digits diagonally above it, incremented by 1. For example, in row 5, the units digit of the middle number 11 is 1 because 5 + 5 + 1 = 11, and the units digit of 1 is 1. Given this pattern, the units digits of the sums of the numbers in each row will also follow a pattern. Specifically, the units digits of the sums of the numbers in the rows will be 2, 4, 6, 8,

- 0, and then repeat (since the sum of the units digits in each row is always 2 more than the sum of the previous row’s units digits). The units digits of the sums of the numbers in the rows are as follows: - Row 1: 1 (units digit of sum is 1) - Row 2: 1 + 1 = 2 (units digit of sum is 2) - Row 3: 1 + 3 + 1 = 5 (units digit of sum is 5) - Row 4: 1 + 5 + 5 + 1 = 12 (units digit of sum is 2) - Row 5:
- 1 + 7 + 11 + 7 + 1 = 27 (units digit of sum is 7) We can see that the units digits of the sums repeat every 4 rows: 1, 2, 5, 2. Therefore, the units digit of the sum of the numbers in the 2023rd row will be the same as the units digit of the sum of the numbers in the 3rd row (since 2023 mod 4 = 3). So, the units digit of the sum of the numbers in the 2023rd row is 5. Let’s write the final answer:

|5|
|---|

