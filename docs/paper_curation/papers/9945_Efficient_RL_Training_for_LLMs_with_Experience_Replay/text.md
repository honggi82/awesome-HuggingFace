### arXiv:2604.08706v1[cs.LG]9Apr2026

# Efficient RL Training for LLMs with Experience Replay

Charles Arnal1,∗, Vivien Cabannes1,∗, Taco Cohen1, Julia Kempe1,2, Remi Munos1,† 1FAIR at Meta, 2NYU Courant Institute and CDS ∗Equal contribution, †Supervising author

While Experience Replay—the practice of storing rollouts and reusing them multiple times during training—is a foundational technique in general RL, it remains largely unexplored in LLM post-training due to the prevailing belief that fresh, on-policy data is essential for high performance. In this work, we challenge this assumption. We present a systematic study of replay buffers for LLM post-training, formalizing the optimal design as a trade-off between staleness-induced variance, sample diversity and the high computational cost of generation. We show that strict on-policy sampling is suboptimal when generation is expensive. Empirically, we show that a well-designed replay buffer can drastically reduce inference compute without degrading – and in some cases even improving – final model performance, while preserving policy entropy.

Date: April 13, 2026 Correspondence: Charles Arnal at charlesarnal@meta.com

#### 1 Introduction

Reinforcement Learning (RL) has emerged as the key driver behind the reasoning capabilities of modern Large Language Models (LLMs), enabling breakthroughs in complex tasks such as mathematics and coding (DeepSeek et al., 2025; OpenR1 et al., 2025). However, this performance comes at a prohibitive computational cost. Unlike pre-training, where data is static, RL requires the continuous generation of new training trajectories. In state-of-the-art pipelines, this inference cost often dominates the training budget, and may consume more than 80% of post-training GPU hours. Standard approaches exacerbate this issue through extreme sample inefficiency: methods like PPO or GRPO typically operate as on-policy as possible, meaning rollouts are generated, used for a single gradient update, and immediately discarded.

This “generate-then-discard” paradigm stands in stark contrast to classical Reinforcement Learning, where Experience Replay, i.e. storing and reusing past trajectories in a buffer, is a foundational tool for sample efficiency (Mnih et al., 2015; Lin, 1992). While Experience Replay is standard in sample-limited robotics or gaming environments, it has been largely overlooked in LLM training, where the prevailing consensus suggests that the performance degradation from off-policy data outweighs the computational benefits.

In this work, we challenge this consensus. We demonstrate that discarding trajectories after a single use is computationally suboptimal. By incorporating a replay buffer into asynchronous training pipelines, we trade a controlled increase in data off-policiness (staleness) and a decrease in data diversity for a dramatic reduction in inference costs. We formalize this trade-off through a theoretical analysis of the bias-variance decomposition in stochastic gradient descent, proving that optimal compute efficiency is achieved not by being strictly on-policy, but by balancing the freshness and diversity of data against its generating cost. Our contributions are as follows:

• Theoretical Analysis: We detail the implementation of replay buffers in asynchronous LLM training and provide a mathematical framework quantifying the trade-off between compute efficiency, sample diversity, and gradient bias. We derive theoretical bounds for the optimal buffer size and replay ratio, showing that as the relative cost of inference increases, the optimal strategy shifts further towards experience replay.

0.77

0.76

Accuracy(pass@1)

0.75

0.74

0.73

(W, T) = (6, 2), N = 84

Baseline

0.72

0 5000 10000 15000 20000 25000 30000 35000

Compute

- Figure 1 Experience Replay improves LLM RL Training. Accuracy on MATH as a function of compute spent when training Qwen2.5-7B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and a buffer of size 84 with (W, T) = (5, 3). We report the median and IQR over 10 seeds. Compute is calibrated so that a single weight update for the baseline costs 1 unit. Baseline runs display increased instability.

- • Empirical Analysis: Through extensive experiments, we provide an in-depth analysis of how buffer hyperparameters influence the training process. We show that while aggressive reuse of samples can degrade performance, a well-sized buffer acts as a regularizer that stabilizes training and preserves model output diversity (improving pass@k metrics).
- • Empirical Gains: We validate those conclusions on larger models and show that simple, easy-toimplement buffer strategies can save up to 40% of the compute budget while maintaining, and sometimes surpassing, the same final accuracy as the on-policy baseline, as shown e.g. in Figure 1. We further explore how more sophisticated sampling strategies (e.g., prioritizing positive trajectories) and alternative losses can extend the stability of replay buffers, allowing for even greater efficiency gains.

Through this study, we present a straightforward approach for high-efficiency RL fine-tuning, shifting the focus from maximizing performance per step to maximizing performance per unit of compute.

#### 2 Related Work

Experience Replay in RL. The use of a replay buffer is a cornerstone of deep RL, famously enabling stability and sample efficiency in algorithms like DQN (Mnih et al., 2015), Soft Actor-Critic (Haarnoja et al., 2018), and DDPG (Lillicrap et al., 2015). Techniques such as Prioritized Experience Replay (Schaul et al., 2015) and Hindsight Experience Replay (Andrychowicz et al., 2017) further optimized how agents learn from past data. Despite this rich history, modern LLM reasoning pipelines (DeepSeek et al., 2025; OpenR1 et al., 2025) have largely defaulted to on-policy training (e.g., GRPO, PPO), discarding trajectories immediately after a gradient update to avoid off-policy degradation, though, in practice, implementation constraints typically lead to some unavoidable off-policiness.

Replay Buffers for LLMs. Very recently, several works have re-introduced replay buffers to LLM training, though with different motivations. Wang et al. (2025) and Bartoldson et al. (2025) utilize buffers primarily to enhance exploration and final model performance, often requiring specialized loss functions or complex filtering. Similarly, Lu et al. (2025) and Zhang et al. (2025) propose dynamic sampling or multi-phase training to maximize data quality. In contrast, our work focuses strictly on compute efficiency. We do not propose a new training paradigm to beat state-of-the-art accuracy; rather, we systematically analyze the trade-off between off-policiness and efficiency in standard asynchronous pipelines, demonstrating that simple experience replay can drastically reduce the compute budget while maintaining accuracy.

A more detailed discussion of off-policy algorithms and related theoretical works is provided in Appendix A.

#### 3 Experience Replay for Off-Policy RL

We present how experience replay can be efficiently implemented in an LLM post-training pipeline and discuss the role of various hyperparameters and their impact on compute efficiency.

##### 3.1 Reinforcement Learning and Replay Buffers

In modern, compute-efficient RL pipelines for LLMs, the GPUs are often split between W inference workers and T trainers (Noukhovitch et al., 2024; Gehring et al., 2024; Wu et al., 2025; Bartoldson et al., 2025; FAIR CodeGen Team et al., 2025). At any given time, each of the two groups maintains its own (possibly stale) copy of the model weights. The inference workers continuously generate trajectories (also called rollouts) using their set of weights, then pass them to the trainers, usually via a transfer queue. Concurrently, trainers pull trajectories from the queue, perform forward-backward passes over them and update their weights. Trajectories are discarded after having been used once (Schulman et al., 2017; Shao et al., 2024; DeepSeek et al., 2025). Every few gradient steps, the inference worker’s weights are updated with the current value of the trainers’ weights. This setting, which corresponds to our experimental implementation, is sometimes referred to as asynchronous training. Synchronous setups also exist (von Werra et al., 2020; Sheng et al., 2024); we discuss them in Section 4.

A replay buffer can be implemented as follows: instead of adding their rollouts to a queue, the inference workers add them to a list of trajectories, the replay buffer. In parallel, trainers continuously sample from this replay buffer; sampling from the buffer does not remove the sampled trajectories from it.1 This allows for the re-using of samples, which in turn reduces the amount of overall compute needed by amortizing the cost of rollout generation, as detailed further below. Pseudo-code is provided in Appendix B.

The replay buffer can be sampled by the trainers to assemble their training batches following several strategies that pick samples based on characteristics such as their recency, their associated rewards, the norm of past gradients computed using them, or how many times the rollout has already been sampled. One might also want to define a decay rule for the buffer, e.g. making the buffer a first-in, first-out list by keeping only the N freshest samples.

Buﬀer size: N = 84 N = 252 N = 756 N = 2268

| |0.2<br><br>0.3|0.3<br><br>0.4| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
|0.0<br><br>0.1| |0.0<br><br>0.1<br><br>0.2| | | | | | | | | |

0.15

Probability

0.10

0.05

0.00

0 10 20 30

0 2 4 6 8 10 new 0 1 2 3 4 5 6 7 8+

(W, T) = (6, 2) (W, T) = (5, 3) (W, T) = (4, 4)

0.3

0.100

0.4

Probability

0.2

0.075

0.3

0.050

0.2

0.1

0.025

0.1

0.000

0.0

0.0

0 10 20 30

0 2 4 6 8 10

new 0 1 2 3 4 5 6 7 8+

Oﬀ-policiness

Replay ratio

Steps since last use

- Figure 2 Effect of Experimental Design on Off-Policiness and Diversity Statistics. Top row: Distribution of off-policiness, replay ratio and steps-since-last-use over all samples and uses of samples during a training run for buffer size N ∈ {84, 252, 756, 2268} and (W, T) = (6, 2). See also Appendix D.3 for details on the steps-since-last-use metric. Bottom row: Same statistics for N = 252 and (W, T) ∈ {(6, 2), (5, 3), (4, 4)}. The average replay ratio is 1.78, 3.42 and 7.0 for (W, T) equal to (6, 2), (5, 3) and (4, 4) respectively.

1In our specific implementation, the buffer is sharded across trainers; see Appendix D for details.

##### 3.2 Off-Policiness, Diversity, and Compute Efficiency

The design of the buffer and the ratio W/T of inference workers to trainers directly impact three major aspects of training: the compute efficiency, the degree of off-policiness, and the diversity of the samples.

To illustrate these concepts, we consider throughout this subsection a buffer configuration with T ∈ {1,2,...,7} trainer GPUs, W := 8 − T inference worker GPUs, and a first-in, first-out buffer (i.e. that contains the last N samples generated by the inference workers). Training samples are drawn uniformly at random from the buffer at each step. We train Qwen2.5-7B Qwen et al. (2025) model on the OpenR1-Math-220k reasoning dataset OpenR1 et al. (2025) (see Subsection 5.1 for experimental details).

Compute Efficiency The compute spent on an RL training run, which we think of in terms of active GPU seconds2, can be decomposed roughly as the sum of the trainer compute, spent on forward-backward passes and weight updates, and the inference compute, spent on generating rollouts, i.e. compute ∼= trainer compute + inference compute. In the asynchronous setting and without a buffer, the ratio W/T of inference worker GPUs to trainer GPUs admits an optimal value µ that minimizes GPU downtime. Indeed, as a first order approximation, we can assume that the trainer compute C needed for a step (including forward and backward passes and weight update) depends only on the (fixed) batch size, and not on the number of trainer GPUs. Let µ > 0 be the factor such that producing a batch of rollouts of the same size costs C · µ compute for the inference workers.3 The total compute needed for each parameter update is then roughly

compute without buffer ≈ C(1 + µ). (1)

In that case, the optimal ratio of inference worker GPUs to trainer GPUs, i.e. the ratio such that trainer GPUs process generated rollouts exactly at the speed at which inference worker GPUs produce them, so that neither have any downtime, is precisely µ: if generating rollouts is µ times more costly than training on them, one needs µ times more inference GPUs than trainer GPUs.

By contrast, when using a replay buffer, the inference compute is decoupled from the trainer compute: inference workers can always continuously add trajectories to the buffer from which the trainers can freely pull, independently from how many inference workers and trainers there are. As in the case without buffer, each backward pass costs C trainer compute. On the other hand, the inference compute spent during a backward pass depends on the number of inference worker GPUs that are concurrently working. Hence, the total compute spent for each parameter update is roughly equal to

total compute with buffer ≈ C(1 + W/T).

As reflected in this formula, when using a buffer, increasing the number of trainers relative to the number of inference workers makes each gradient step cheaper; intuitively, this is simply because rollouts are re-used more times on average, meaning that for a given number of optimization steps, fewer rollouts need to be generated.

We define the compute ratio of a buffer configuration to be

1 + W/T 1 + µ

, (2)

γ :=

that is, the ratio of the compute cost of a parameter update with and without a buffer.

|(W, T)|(7,1)<br><br>|(6,2)|(5,3)<br><br>|(4,4)|(2,6)<br><br>|(1,7)|
|---|---|---|---|---|---|---|
|γ<br><br>|1.29|0.65<br><br>|0.43|0.32<br><br>|0.22|0.18|

Table 1 γ for various values of (W, T) and an estimated µ = 5.28 for Qwen2.5-7B.

2We explain in greater details our simplifying assumptions in Appendix D. 3In our experiments, we find that µ ranges from ∼4 to ∼10 depending on the model, task and implementation considered.

Degree of Off-Policiness The design of the replay buffer and the ratio (W,T) directly impact the off-policiness of the training distribution. We define the off-policiness (or staleness) of a sample used in a gradient update as the difference between the step at which the sample was created and the current step. The average off-policiness over all samples is influenced by both the size N of the buffer (the larger the buffer, the greater the average off-policiness of the samples that it contains) and the ratio W/T of inference worker to trainer GPUs: the more trainer GPUs there are, the faster weight updates occur and the faster samples become outdated. This can be observed on the left of Figure 2, where the distribution of off-policiness over all the samples used through a training run is represented for various pairs (W,T) and buffer sizes N.

Diversity of Samples The use of a replay buffer may deteriorate training dynamics: as the same samples are reused, the training distribution seen by the policy gradient algorithm becomes less diverse, and less information regarding the true objective function is utilized. This notion of sample diversity arises at two distinct levels. First, the global diversity of samples, which we measure using the replay ratio of the samples, defined as the number of times a sample has been used for a gradient step over the entire training run. The average replay ratio will be chiefly conditioned by the ratio W/T: the more trainer GPUs there are relative to the number of inference worker GPUs, the more passes they will do on average on each data point. This is illustrated in the middle of Figure 2.

Second, the local diversity of samples which is the degree to which samples are repeatedly used in close succession. We measure local diversity using the time-since-last-use of the samples in the current trainer batch, i.e. the number of gradient steps since the last gradient update to which they contributed. We expect a loss in local diversity to be more harmful than a loss in global diversity. At a fixed ratio W/T, one can trade off-policiness for local diversity: by increasing the size of the buffer, the training distribution’s degree of off-policiness will increase (as discussed earlier), but the empirical training distribution will be locally more diverse: though samples are just as likely to be reused over the entire training run, they are less likely to be reused in close succession (due to the greater number of candidate samples in the buffer). This can be seen on the right side of Figure 2.

Goal: Increased Efficiency, Preserved Accuracy The primary motivation behind the use of a replay buffer is to save inference compute by reusing trajectories. As explained above, each gradient step (including the required sampling) can be made computationally cheaper by letting the ratio W/T decrease. However, we have also seen that letting the ratio W/T decrease makes the training distribution more off-policy and less diverse. It is usually assumed that high off-policiness and low sample diversity should be avoided (see however Tang et al. (2025); Arnal et al. (2025) and Charton and Kempe (2024)). Hence there is a trade-off : re-using samples from the buffer makes each gradient step cheaper, but resampling too aggressively might end up hurting the expected accuracy gain from each step. In our experiments, we explore the efficiency/accuracy optimality curve; in other words, we want to maximize the accuracy achievable at a given compute cost by selecting the best buffer configuration. To ensure our conclusions are readily applicable to production environments, we also deliberately prioritize simple implementations that require only modest departures from current SOTA pipelines.

#### 4 Mathematical Analysis

While the previous section and our experiments focus on the more compute-efficient asynchronous RL setting, we choose to conduct our mathematical analysis in the conceptually simpler synchronous setting, in which the training alternates between two clearly distinct modes: a generating phase, in which new trajectories are created, and a training phase, during which a gradient descent step is performed using the new rollouts. We consider a simple first-in, first-out replay buffer: at each training step t, we (i) generate R new rollouts using the current policy and insert them at the beginning of a buffer of capacity N (evicting the oldest samples), and (ii) sample a minibatch of size B uniformly from the buffer to form a gradient update

1 B

θt+1 = θt − η gt, gt =

B

G(θt,zt,i

).

j

j=1

Here, θ denotes the policy parameters, zt,i denotes the i-th element of the buffer at step t, ij the j-th sampled index, and G(θ,z) denotes the corresponding gradient estimate of ∇F(θ), where F is the objective we wish to minimize. The compute cost of such an update, expressed in arbitrary units, is given by c = B + µR, where µ denotes the compute cost ratio between a forward-backward pass and one rollout generation, matching the definition above.

The goal of our theoretical analysis is to characterize how the design of the replay buffer affects learning efficiency from a theoretical standpoint. We adopt the classical non-convex stochastic optimization framework and study the convergence of the training dynamics toward stationary points, as measured by the decay of the expected squared norm of the gradient. Unless stated otherwise, all norms are Euclidean.

- Assumption 4.1 (Target Smoothness). The function F is non-negative, differentiable, and L-smooth, i.e. ∀x,y ∥∇F(y) − ∇F(x)∥ ≤ L∥y − x∥.

Let Ft represent the information available from the parameter iterates up to time t, i.e. the σ-field associated to the sequence (θs)s≤t. Define the per-sample and minibatch gradient noises by

εt,i = G(θt,zt,i) − ∇F(θt), and εt =

1 B

B

j=1

εt,i

j

.

In contrast to usual SGD analysis, experience replay introduces a bias in the gradient estimate through the correlation introduced by the buffer, even with importance ratio correction.4 We expect this bias to be larger when trajectories presently in the buffer have had a strong influence on the subsequent updates leading to the current parameter θt, and to be small when the parameters have moved little over the time span covered by the buffer. This intuition motivates the following assumption, discussed further in Appendix C.

- Assumption 4.2 (Bias). There exists a constant κ ≥ 0 such that for all (t,i),

∥E[εt,i | Ft]∥ ≤ κ∥θt − θt

i∥, where ti = t + 1 − ⌈i/R⌉ is the time at which the i-th element of the buffer was added to the buffer.

The variance of our gradient estimates depends on both the per-sample variance, and the correlation between different samples drawn within the same minibatch. The per-sample variance typically increases with offpoliciness, reflecting the growing variance of importance ratio as off-policiness increases. In addition, samples within a batch can be statistically dependent, since some may have influenced the sequence of parameter updates that produced the others. This coupling is mediated by how strongly any individual rollout can affect subsequent iterates. At time ti, a rollout generated at time tj < ti will have contributed, on average, (ti − tj) · B/N times to the gradient updates between ti and tj. As each update averages over B samples, we expect the dependency to scale in O(|ti − tj|/N). This motivates the following assumption.

- Assumption 4.3 (Variance). There exists a non-decreasing function σ : R → R+ and a coefficient ρ ∈ [0,1], such that for any (t,i),

E[∥εt,i∥2] ≤ σ2(t − ti), and for j ̸= i,

ρ|ti − tj| N

correlation(εt,i,εt,j) ≤

.

We are now ready to state the main convergence theorem, proven in Appendix C.

- Theorem 4.4. Under Assumptions 4.1, 4.2 and 4.3, when the learning rate satisfies η ≤ min(R/(2√2κN),L/2)

T−1

4N2κ2η R2

1 T

12F(θ0) ηT

∥∇F(θt)∥2 ≤

+ L V

+ 8η

t=1

4While importance sampling corrects the marginal distribution mismatch between πθt and πθt−τ , experience replay forces us to reason about previous distributions conditioned on the current parameters, i.e. the distribution πθt−τ (· | θt) at time θt−τ conditioned on the fact that the training trajectory that followed (which was influenced by the samples drawn at θt−τ through the policy gradient algorithm) ended up at θt. The distribution πθt−τ (· | θt) is typically not computable.

for any T > 1, where V is a variance parameter defined as

N R

1 B

1 N

ρ R

V = σ¯2

+

+

.

and σ¯(H) is the average of σ(1),...,σ(H).

- Theorem 4.5 (Optimal Design). Given an asymptotically large compute budget C, related to the number T of iterations by C = (B + µR)T, we optimize over (η,N,R,B) the bound in Theorem 4.4. Assuming R divides N, and relaxing integer constraints, it yields the optimal ratios

σ¯2(x)( 1/µ + ρ + 1/x)2,

N/R = x∗ := arg min

x>0

B/R = r∗ := µ/(ρ + 1/x∗).

Here, N/R denotes the off-policiness horizon, i.e. the maximum off-policiness of rollouts in the buffer, and B/R the replay ratio, i.e. the average number of times a sample is replayed over the full run.5

We also provide a closed-form expression for x∗ in Appendix C under a power-law assumption on σ, as well as further empirical illustrations in Figure 6.

Theorem 4.5 characterizes the optimal replay-buffer design in terms of the staleness horizon N/R and the replay ratio B/R. These ratios serve as key design levers, allowing practitioners to systematically configure the replay buffer for peak algorithmic performance. Theorem 4.5 reveals a three-way trade-off between staleness-induced noise growth (σ¯2), coupling between replayed samples and the parameter iterates (ρ), and the rollout-vs-training compute imbalance (µ). When the compute cost of rollouts is small (small µ), or when off-policy induced variance (σ¯ increases fast) and correlation (ρ) are high, the optimal staleness horizon x∗ approaches zero. This suggests that in such regimes, it is more effective to remain on-policy than to utilize a replay buffer. Conversely, when rollout generation is expensive (large µ) or off-policy effects are negligible (σ¯ and ρ are small), a replay buffer becomes optimal, characterized by a large staleness horizon and a high replay count. Overall, our theory formalizes the central trade-off studied in our experiments: replay can substantially reduce inference compute, but only up to the point where staleness-induced variance and samples-iterate correlations begin to dominate the benefit of reusing trajectories.

#### 5 Experimental results

We explore how experience replay impacts accuracy and compute efficiency when training small and mid-size models with asynchronous RL fine-tuning on reasoning datasets.

##### 5.1 Experimental setup

We evaluate replay buffers in the asynchronous setting described in Section 3, with W inference workers generating rollouts and T trainers performing optimization steps from a shared buffer. Unless otherwise specified, we sample from the buffer uniformly. In our primary experiments, we fine-tune Qwen3-0.6B and Qwen2.5-7B (Qwen et al., 2025) with GRPO (Shao et al., 2024) on OpenR1-Math-220k (OpenR1 et al., 2025), and evaluate on either OpenR1-Math-220k or MATH (Hendrycks et al., 2021). Unless stated explicitly, we use a learning rate of 3.37 · 10−7 for Qwen3-0.6B and 6 · 10−8 for Qwen2.5-7B. We plot accuracy w.r.t. either the number of gradient steps, the compute spent (estimated with (2)) or the wall-time. All our experiments are run with at least 4 random seeds, and we report the median and the interquartile range. See Appendix E for ablations on the learning rate, and Appendix D for additional details on the setup, including the estimation of the optimal ratio µ.

5Note that R/N is the ratio of fresh samples in the buffer, thus by contraposition N/R is the number of rounds a sample will stay in the buffer. Moreover, since each sample in the buffer is associated with a sampling probability 1/N, we sample B of them in a batch, and a sample stays for N/R round in the buffer, their average use over their shelf-life is (1/N)B(N/R) = B/R.

###### Best Accuracy

[Figure 1]

64 256 512 768

Buﬀer size N

0.56

0.05

64

[Figure 2]

BestAccuracy(pass@k)

0.54

128 256 512 768

0.04

BuﬀerSizeN

Accuracy(pass@1)

0.54

0.52

1536 2304 4608 9216

0.03

0.50

1536 2304 4608 9216

0.02

0.52

0.48

0.01

20736 62208

0.46

0.50

0.00

20736 62208

0.44

186624 559872

- -0.02

- -0.01

0.42

186624 559872 Baseline

2) (5, 3) (4, 4)

0.40

(6,

1 2 4 8 16 32

0 5000 10000 15000

k (pass@k)

(W, T)

Compute

- Figure 3 Accuracy and Pass@k with respect to Buffer Size. Left: Test accuracy as a function of compute spent when training Qwen3-0.6B on OpenR1-Math-220k for (W, T) = (6, 2) and various buffer sizes N ∈ {64, 128, 256, 512, 768, 1536, 2304, 6912, 20736}, as well as for a no-buffer baseline. We report the median and IQR over more than 4 seeds. Compute is normalized so that each weight update costs 1 unit for buffer configurations and 1.96 for the baseline. Middle: Pass@k increase after training for a representative subset of these buffer configurations, relative to the baseline. Right: Best Accuracy achieved over entire runs for various buffer sizes and W/T ratios. The compute needed to reach those accuracies is reported in Figure 14 in Appendix E.

##### 5.2 Main results

Figure 1 summarizes our central finding: for a good choice of buffer configuration, one may save up to 40% of compute to reach a given accuracy. For all compute budget, the accuracy achievable using experience replay is superior to that achievable with strictly on-policy training, contradicting the current paradigm. Moreover, we observe an additional benefit not predicted by our theory: using a buffer stabilizes training, preventing crashes and sometimes enabling a higher peak accuracy. These findings are confirmed on other buffer configurations, other models and other tasks in Figures 15, 16 and 17 of Appendix E.

We now run more comprehensive experiments on a smaller model to further analyze the impact of various buffer hyperparameters and better understand these phenomena.

Buffer Size and Off-Policiness. The left side of Figure 3 shows the test accuracy of Qwen3-0.6B for (W,T) = (6,2) and various buffer sizes as a function of compute. We first observe that all training trajectories (with or without buffer) culminate in a global maximum accuracy, followed by a decline in performance–this is not an uncommon phenomenon in RL (see, e.g., Zheng et al., 2025).6 We further observe that increasing the size of the buffer, hence increasing the average off-policiness of the samples, has two marked effects: it slows down the training, and it stabilizes it, leading to a potentially higher maximal accuracy that is reached later in the training run. As a secondary exploration, we trained the same model without a replay buffer and introduced various levels of off-policiness in the training distribution. The results, reported in Figure 12 in the Appendix, align with our findings and show that moderate levels of off-policiness can have a stabilizing effect on the training (independently from the use of experience replay). We hypothesize that reusing rollouts sampled from older policies regularizes the (evolving) objective function by increasing the diversity of the training distribution, and thus helps prevent overfitting. As larger models take much longer to overfit, the same effect is not visible in Figure 1.

Replay ratio. As the ratio W/T between inference workers and trainers decreases, the compute cost of each gradient update drops (Table 1), but the average replay ratio rises, going from 2.2 for (W,T) = (6,2) to 5.6 and 17.6 for (W,T) = (5,3) and (4,4) respectively. We see on the heatmap in Figure 3 that while moderate replay ratios do not adversely affect the maximal accuracy, aggressive replay eventually degrades performances (most likely due to the associated reduced local sample diversity, see Section 3). As shown on the more

6Looking at the training accuracy (Figure 13 in Appendix E), we see that it peaks later than the test accuracy, then crashes as well, indicating that the models initially overfit before ultimately collapsing into a nonsensical policy.

exhaustive plots for (W,T) ∈ {(5,3),(4,4)} in Figure 13 of Appendix E, more extreme configurations can nonetheless remain attractive due to their high compute efficiency.

Output diversity. One can see in Figure 3 that training with experience replay can also improve the pass@k (for k > 1). This is true in absolute terms (i.e. the pass@k is improved), but also comparatively: using a buffer helps the pass@k for large k even more than it helps the pass@1. This shows that while the loss in diversity of the model’s output distribution is a major concern in RL (Cui et al., 2025; Yue et al., 2025), experience replay can help preserve it. We attribute this phenomenon to the increased diversity of the training distribution which results from the use of older samples.

To summarize, our experiments suggest that reducing the ratio W/T improves compute efficiency but worsens learning dynamics, whereas increasing the buffer size slows training while stabilizing it and helping preserve output diversity. Under suitable configurations, these effects combine to yield a net improvement across all metrics relative to strictly on-policy RL.

##### 5.3 Wall-time Speed

We found compute (as defined through (2)), which isolates the algorithmic effect of experience replay (fewer rollouts per update), to be a more informative metric than wall-time, which is influenced by implementationdependent scheduling and queuing effects. That said, we have observed that the gains in wall-speed from using a buffer in our particular setup either match or exceed the gains in compute efficiency (see Figures 10 and 11 in Appendix E).

Indeed, in asynchronous settings (described in Section 3), inference workers often stall when the transfer queue is full, and trainers stall when the queue is empty—both effects are exacerbated when reward computation introduces variable latency (as also noted by Lu et al. (2025)). This can occur even when the optimal ratio µ of trainer GPUs to inference GPUs is achieved, and is exacerbated when it is not. A replay buffer attenuates these stalls by decoupling production from consumption: trainers can continue optimizing even when rollout generation temporarily slows, and inference workers can continue generating rollouts even when trainers are temporarily back-pressured. This smoothing effect brought by the buffer is independent from the increase in compute efficiency discussed at length above. It can be leveraged to streamline an asynchronous RL pipeline with a ratio W/T set to precisely µ while keeping the expected replay ratio equal to 1.

##### 5.4 Controlling for the learning rate: optimality curves

We performed preliminary ablations (reported in Figures 8 and 9 in Appendix E) to ensure that we selected for each model the optimal learning rate for the baseline, i.e. that which led to the highest maximum accuracy and the greatest training stability. As experience replay changes the optimization dynamics,7 we ran further control experiments to ensure that the efficiency gains reported cannot be attributed to inadequate hyperparameters tuning. Namely, we performed an extensive sweep across learning rates and buffer configurations. For both buffer and non-buffer setups, we plot for each compute budget the best achievable accuracy (over learning rates and buffer parameters) for that budget, resulting in two optimality curves reported in Figure 4. We find that the best buffer configurations consistently outperform the best non-buffer configurations.

##### 5.5 Further optimization: refining replay buffer design

So far, we have intentionally focused on the simplest replay buffer implementation, requiring the least deviation from the standard SOTA pipelines. We now extend our study to more exotic designs in search of further improvements, and consider two refinements. Firstly, we replace the basic sampling strategy used hitherto with a modified strategy, that we call positive-bias sampling: instead of keeping the freshest N generated rollouts in the buffer, we keep the freshest (1 − δ)N generated rollouts along with the freshest δN correct rollouts not included in those (1 − δ)N trajectories (an example is given in Appendix D.4), and uniformly sample from these N samples. Our intuition is that the utility of correct rollouts is less affected by off-policiness. Secondly, we replace GRPO with the AsymRE loss from Arnal et al. (2025), which has shown promises in

7E.g., a (statistically unlikely) scenario where the exact same training batch is reused twice in a row would in fact be equal, up to second order terms, to a single gradient step with a learning rate twice as large.

0.56

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | |ba wi<br><br>|seline th buffer|
| | | | | | | |

0.54

Accuracy(pass@1)

0.52

0.50

0.48

0.46

0.44

0 1k 2k 3k 4k 5k 6k

Compute

- Figure 4 Pareto Frontier across Hyperparameters Sweep. Test accuracy as a function of compute spent when

training Qwen3-0.6B on OpenR1-Math-220k for various learning rates ({1.5i · 10−7}5i=0) and buffer configurations: no buffer (blue curves), buffer of size {64, 128, 256, 512, 768, 2304, 6912, 20736} with (W, T) ∈ {(6, 2), (5, 3), (4, 4)} (orange curves). Each curve is the median over at least 4 seeds. The two boldfaced curves delineate the Pareto frontier of each family of runs. Compute is normalized so that each weight update costs 1 unit for baseline configurations and and 0.51 for buffer configurations.

such settings (see Appendix D). Unlike GRPO, AsymRE does not feature importance ratio correction, which is known to increase variance when off-policiness is high and does not account for subtle dependency effects when sampling from a buffer.

As showcased in Figure 5, we find that both variants lead to substantial improvements over the basic buffer implementation; larger-scale experiments are now needed to validate the robustness of these findings.

As a third refinement, we also tried sampling from a (standard) buffer uniformly without replacement in order to increase local diversity, but the results were inconclusive (see Figure 18).

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | |δ=0.0 (G δ=0.2 (G|RPO) RPO)|
| | | | | | | |
| | | | | |δ=0.5 (G δ=0.0 (A|RPO) symRE)|
| | | | | | | |
| | | | | |δ=0.2 (A δ=0.5 (A|symRE) symRE)|
| | | | | | | |
| | | | | | | |

0 2000 4000 6000 8000 10000

Step

0.42

0.44

0.46

0.48

0.50

0.52

0.54

Accuracy(pass@1)

- Figure 5 Alternative Loss, Positive-Bias Sampling Rule. Test accuracy as a function of training steps when training Qwen3-0.6B on OpenR1-Math-220k with a buffer of size N = 4608 and (W, T) = (6, 2). We use either GRPO or AsymRE, and apply positive-bias sampling with coefficient δ ∈ {0, 0.2, 0.5} (note: δ = 0 corresponds to standard uniform sampling).

#### 6 Conclusion

In this work, we challenged the "generate-then-discard" paradigm that currently dominates LLM reinforcement learning. Through a combination of theoretical analysis and extensive empirical evaluation, we show that a well-configured replay buffer serves as a powerful lever for compute efficiency. Our theoretical framework characterizes a fundamental three-way trade-off between staleness, sample diversity, and the relative cost of inference. We show that as the computational burden of rollout generation grows, the optimal strategy shifts decisively toward experience replay. Empirically, we find that these gains are not merely theoretical: a simple replay buffer can reduce the compute budget by up to 40% while maintaining or even surpassing the accuracy of on-policy baselines. These findings suggest that maximizing performance per unit of compute, rather than per gradient step, is a more practical objective for RL pipelines, and that experience replay is a key component in achieving this.

While our results are consistent for the model scales evaluated in this study, further work is needed to validate these efficiency gains on larger frontier models. Additionally, we believe that the Pareto frontier can be pushed further by moving beyond uniform buffers toward more sophisticated sampling rules and off-policy corrections, as well as other losses.

#### References

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. Advances in neural information processing systems, 30, 2017.

Charles Arnal, Gaëtan Narozniak, Vivien Cabannes, Yunhao Tang, Julia Kempe, and Remi Munos. Asymmetric REINFORCE for off-policy reinforcement learning: Balancing positive and negative rewards, 2025. https://arxiv. org/abs/2506.20520.

Brian R. Bartoldson, Siddarth Venkatraman, James Diffenderfer, Moksh Jain, Tal Ben-Nun, Seanie Lee, Minsu Kim, Johan Obando-Ceron, Yoshua Bengio, and Bhavya Kailkhura. Trajectory balance with asynchrony: Decoupling exploration and learning for fast, scalable LLM post-training, 2025. https://arxiv.org/abs/2503.18929.

François Charton and Julia Kempe. Emergent properties with repeated examples, 2024. https://arxiv.org/abs/ 2410.07041.

Taco Cohen, David W. Zhang, Kunhao Zheng, Yunhao Tang, Remi Munos, and Gabriel Synnaeve. Soft policy optimization: Online off-policy RL for sequence models, 2025. https://arxiv.org/abs/2503.05453.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. https://arxiv.org/abs/2505.22617.

DeepSeek, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

FAIR CodeGen Team, Jade Copet, Quentin Carbonneaux, Gal Cohen, Jonas Gehring, Jacob Kahn, Jannik Kossen, Felix Kreuk, Emily McMilin, Michel Meyer, Yuxiang Wei, David Zhang, Kunhao Zheng, Jordi Armengol-Estapé, Pedram Bashiri, Maximilian Beck, Pierre Chambon, Abhishek Charnalia, Chris Cummins, Juliette Decugis, Zacharias V. Fisches, François Fleuret, Fabian Gloeckle, Alex Gu, Michael Hassid, Daniel Haziza, Badr Youbi Idrissi, Christian Keller, Rahul Kindi, Hugh Leather, Gallil Maimon, Aram Markosyan, Francisco Massa, Pierre-Emmanuel Mazaré, Vegard Mella, Naila Murray, Keyur Muzumdar, Peter O’Hearn, Matteo Pagliardini, Dmitrii Pedchenko, Tal Remez, Volker Seeker, Marco Selvi, Oren Sultan, Sida Wang, Luca Wehrstedt, Ori Yoran, Lingming Zhang, Taco Cohen, Yossi Adi, and Gabriel Synnaeve. CWM: An open-weights LLM for research on code generation with world models, 2025. https://arxiv.org/abs/2510.02387.

Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Taco Cohen, and Gabriel Synnaeve. RLEF: Grounding code LLMs in execution feedback with reinforcement learning. arXiv preprint arXiv:2410.02089, 2024.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. NeurIPS, 2021.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

Long-Ji Lin. Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine learning, 8(3):293–321, 1992.

Fanbin Lu, Zhisheng Zhong, Shu Liu, Chi-Wing Fu, and Jiaya Jia. Arpo: End-to-end policy optimization for GUI agents with experience replay, 2025. https://arxiv.org/abs/2505.16282.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems, pages 1054–1062, 2016.

Michael Noukhovitch, Shengyi Huang, Sophie Xhonneux, Arian Hosseini, Rishabh Agarwal, and Aaron Courville. Asynchronous RLHF: Faster and more efficient off-policy RL for language models. arXiv preprint arXiv:2410.18252,

2024. https://arxiv.org/abs/2410.18252.

OpenR1, :, Loubna Ben Allal, Lewis Tunstall, Anton Lozhkov, Elie Bakouch, Guilherme Penedo, Hynek Kydlicek, and Gabriel Martin Blazquez. OpenR1-math-220k: A large-scale dataset for mathematical reasoning. https: //huggingface.co/blog/open-r1/update-2, February 2025.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. https://arxiv.org/abs/2412.15115.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2023.

Pierre Harvey Richemond, Yunhao Tang, Daniel Guo, Daniele Calandriello, Mohammad Gheshlaghi Azar, Rafael Rafailov, Bernardo Avila Pires, Eugene Tarassov, Lucas Spangher, Will Ellsworth, et al. Offline regularised reinforcement learning for large language models alignment. arXiv preprint arXiv:2405.19107, 2024.

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. DeepSeek-Math: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. arXiv preprint arXiv: 2409.19256, 2024.

Yuda Song, Gokul Swamy, Aarti Singh, Andrew Bagnell, and Wen Sun. The importance of online data: Understanding preference fine-tuning via coverage, 2024. https://arxiv.org/abs/2406.01462.

Yunhao Tang, Taco Cohen, David W. Zhang, Michal Valko, and Rémi Munos. RL-finetuning LLMs from on- and off-policy data with a single algorithm, 2025. https://arxiv.org/abs/2503.19612.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, and Nathan Lambert. TRL: Transformer reinforcement learning. https://github.com/huggingface/trl, 2020. Accessed: 2025-09-03.

Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Yue Wang, and Yuzhi Zhang. Eframe: Deeper reasoning via exploration-filter-replay reinforcement learning framework, 2025. https://arxiv. org/abs/2506.22200.

Bo Wu, Sid Wang, Yunhao Tang, Jia Ding, Eryk Helenowski, Liang Tan, Tengyu Xu, Tushar Gowda, Zhengxing Chen, Chen Zhu, Xiaocheng Tang, Yundi Qian, Beibei Zhu, and Rui Hou. LlamaRL: A distributed asynchronous reinforcement learning framework for efficient large-scale LLM training. arXiv preprint arXiv:2505.24034, 2025. https://arxiv.org/abs/2505.24034.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source LLM reinforcement learning system at scale, 2025. https://arxiv.org/abs/2503.14476.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025. https://arxiv.org/abs/2504. 13837.

Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, and Guorui Zhou. RLEP: Reinforcement learning with experience replay for LLM reasoning, 2025. https://arxiv.org/abs/2507.07451.

Shangtong Zhang and Richard S Sutton. A deeper look at experience replay. arXiv preprint arXiv:1712.01275, 2017. Haizhong Zheng, Jiawei Zhao, and Beidi Chen. Prosperity before collapse: How far can off-policy rl reach with stale

data on llms?, 2025. https://arxiv.org/abs/2510.01161.

## Appendix

#### A Extended Related Work

We provide here a more comprehensive overview of experience replay in reinforcement learning, ranging from foundational deep RL works to the most recent applications in Large Language Models.

##### A.1 Experience Replay in Classical Deep RL

The concept of improving computational efficiency by storing and reusing past transitions is standard in general RL but has historically been difficult to stabilize.

- • Foundations: Mnih et al. (2015) (DQN) demonstrated that training on samples drawn randomly from a replay buffer breaks temporal correlations in data, stabilizing the training of value functions. This became a standard component of off-policy learning.
- • Prioritized Sampling: Schaul et al. (2015) introduced Prioritized Experience Replay (PER), improving upon uniform sampling by prioritizing transitions with high temporal-difference (TD) error, effectively focusing learning on "surprising" or difficult examples.
- • Hindsight Replay: Andrychowicz et al. (2017) proposed Hindsight Experience Replay (HER) for goal-oriented tasks. By re-labeling failed trajectories as successful attempts towards the state they did reach, HER allows agents to learn from failure, significantly boosting sample efficiency in sparse-reward settings.
- • Theoretical Analysis: Zhang and Sutton (2017) provided an early theoretical analysis of experience replay, investigating the relationship between buffer size, replay ratio, and performance, a line of inquiry we extend to the LLM setting in Section 4.

##### A.2 Off-Policy Algorithms

Using a replay buffer inherently introduces off-policiness—the discrepancy between the data-generating policy and the current policy. Various algorithms have been designed to handle this:

- • Actor-Critic Methods: DDPG (Lillicrap et al., 2015) and Soft Actor-Critic (SAC) (Haarnoja et al.,

2018) are off-policy algorithms that update the policy using samples from a buffer. SAC, in particular, maximizes both expected return and entropy, stabilizing training in complex environments.

- • Off-Policy Corrections: The Retrace algorithm (Munos et al., 2016) utilizes truncated importance sampling to safely learn from multi-step returns generated by behavioral policies. Addressing the instability of stale updates in LLMs, Zheng et al. (2025) propose second-moment constraints (M2PO) to stabilize off-policy training.
- • Recent Theoretical Advances: More recent approaches derive consistency conditions from KLregularized policy optimization problems (Rafailov et al., 2023; Richemond et al., 2024; Tang et al., 2025; Cohen et al., 2025), analyze the role of dataset coverage (Song et al., 2024), or propose additive renormalization of baselines (Arnal et al., 2025) to handle distribution shifts mathematically.

##### A.3 Experience Replay in Modern LLM Training

While on-policy methods like PPO (Schulman et al., 2017) and GRPO (Shao et al., 2024) dominate the current LLM landscape, a wave of very recent works (2025) has begun to explore replay mechanisms. However, their goals differ significantly from ours:

- • Improving Performance via Exploration: Bartoldson et al. (2025) use a replay buffer combined with a dedicated loss function specifically to increase exploration in sparse reward settings. Similarly, Wang et al. (2025) focus on saving successful solutions to challenging prompts ("gold samples") to facilitate reasoning breakthroughs.

- • Complex Training Pipelines: Zhang et al. (2025) propose a two-phase training procedure where samples from an initial exploration phase are reused, while Lu et al. (2025) use a buffer to implement dynamic sampling strategies.

Unlike these works, which often introduce complex new objectives to maximize final accuracy, our work conducts a rigorous analysis of the efficiency trade-offs in standard pipelines with the addition of a simple replay buffer. We aim to answer how much compute can be saved by reusing data in a standard asynchronous setup without degrading performance?

#### B Pseudo-Code Implementation

This section provides a peudo-code implementation of the asynchronous Reinforcement Learning pipeline. This code utilizes Python’s asyncio library to simulate the concurrent execution of inference workers (W) and trainers (T). It highlights the transition from a standard stream-based approach to the replay Buffer architecture discussed in our work.

##### B.1 Queue-based Data Transfer

In baseline asynchronous RL, data typically flows through a last-in, first-out (LIFO) pipe, prioritizing the freshest samples for training. As noted in Section 3, this structure forces a tight coupling between rollout generation and consumption, where trajectories are discarded after a single update.

- 1 import asyncio

- 2 import random

- 3

- 4 class QueueStructure:

- 5 """Standard FIFO storage for on-policy rollouts."""

- 6 def __init__(self):

- 7 self.queue = asyncio.LifoQueue()

- 8

- 9 async def push(self , data):

- 10 await self.queue.put(data)

- 11

- 12 async def sample(self , batch_size):

- 13 # Strictly consumes data: items are removed once sampled

- 14 return [await self.queue.get() for _ in range(batch_size)] Listing 1 LIFO Queue implementation for on-policy streaming

##### B.2 Inference Worker

The Sampler represents one of the W inference workers. It operates in a loop, generating trajectories to be pushed into the storage structure. While the pseudo-code suggests a sequential weight update, in efficient implementations, such as the one used for this work, weights are typically updated concurrently to rollout generation to maximize throughput (i.e., the policy may change during a rollout, with later tokens generated under a different set of weights than earlier ones).

- 1 class Sampler:

- 2 def __init__(self , dump_struct):

- 3 self.dump_struct = dump_struct

- 4

- 5 async def run(self , dataset):

- 6 for data in dataset:

- 7 await self.receive_weights()

- 8 rollout = await self.generate_rollout(data)

- 9 await self.dump_struct.push(rollout)

- 10

- 11 # Signal completion to the Trainer

- 12 await self.dump_struct.push("DONE")

- 13

- 14 async def receive_weights(self):

- 15 """Pull latest parameters from Trainer to stay as ’on-policy’ as possible." ""

- 16 ...

- 17

- 18 async def generate_rollout(self , data):

- 19 """Standard LLM inference step."""

- 20 ... Listing 2 Inference Worker (Sampler) logic

##### B.3 The Consumer: Trainer

The Trainer represents one of the T optimization units. It pulls batches of size B and performs gradient updates. This loop runs concurrently with the Sampler.

- 1 class Trainer:

- 2 def __init__(self , dump_struct):

- 3 self.dump_struct = dump_struct

- 4 self.is_running = True

- 5

- 6 async def run(self , batch_size):

- 7 while self.is_running:

- 8 batch = await self.dump_struct.sample(batch_size)

- 9

- 10 if "DONE" in batch:

- 11 self.is_running = False

- 12 break

- 13

- 14 await self.forward_backward(batch)

- 15 await self.update_weights()

- 16

- 17 async def forward_backward(self , batch):

- 18 """Compute GRPO/PPO loss and gradients."""

- 19 ...

- 20

- 21 async def update_weights(self):

- 22 """Apply optimizer step and broadcast new weights."""

- 23 ... Listing 3 Optimization Worker (Trainer) logic

##### B.4 Main Orchestration

The main loop instantiates the workers. While this pseudo-code implementation uses W = T = 1 for simplicity, in practice more workers would operate in parallel, with a ratio of worker to trainer GPUs set to maximize GPU utilization by minimizing idle time.

- 1 async def main():

- 2 dataset = ...

- 3 batch_size = ...

- 4 dump_struct = ...

- 5

- 6 sampler = Sampler(dump_struct)

- 7 trainer = Trainer(dump_struct)

- 8

- 9 # Launching inference and training concurrently

- 10 await asyncio.gather(

- 11 sampler.run(dataset),

- 12 trainer.run(batch_size)

- 13 )

- 14

- 15 if __name__ == "__main__":

- 16 asyncio.run(main()) Listing 4 Asynchronous execution entry point

##### B.5 The Replay Buffer: BufferStructure

A replay buffer can be implemented with minimal changes to the pipeline above. Indeed, one only needs to replace the transfer queue with a new data structure to implement experience replay. We present it below as the BufferStructure which will store up to N buffered trajectories. Unlike the queue, this structure enables multiple samples of the same trajectory.

- 1 class BufferStructure:

- 2 """Experience Replay Buffer supporting random sampling."""

- 3 def __init__(self , buffer_size):

- 4 self.buffer = []

- 5 self.buffer_size = buffer_size

- 6 self.lock = asyncio.Lock()

- 7

- 8 async def push(self , data):

- 9 async with self.lock:

- 10 # FIFO eviction policy for the buffer

- 11 if len(self.buffer) >= self.buffer_size:

- 12 self.buffer.pop(0)

- 13 self.buffer.append(data)

- 14

- 15 async def sample(self , batch_size):

- 16 async with self.lock:

- 17 # Sampling does not remove items from the buffer

- 18 return random.sample(self.buffer , batch_size) Listing 5 Circular Replay Buffer with Random Sampling

#### C Mathematical Details

We provide additional details regarding the mathematical analysis in Section 4.

##### C.1 Modeling Details

Bias Assumption. Assumption 4.2 can be motivated by writing the bias explicitly, using the duality bracket and any dual norms

E[εt,i | Ft] = Ez∼π

[G(θt,z) | Ft] − Ez∼π

,G(θt,·) ≤ πθ

(· | Ft) − πθ

[G(θt,z)] = πθ

t−ti

t

θt−ti

θt

(· | Ft) − πθ

t ∥G(θt,·)∥∗ . Here πθ

t−ti

knowing that the future iterates up to θt. If z in position i in the buffer at time t was never sampled in the batches leading from θt−t

(· | Ft) denote the distribution of the samples under θt−t

t−ti

i

to θt, πθ

(· | Ft) would be equal to πθ

t−ti

i

, as knowing the iterates Ft would not help us reconstruct that sample. However, the more the sample was used, the more these distributions would be dissimilar. With κ0 the average repetition of a sample in training batch from time ti to t, one may posit

t−ti

(· | Ft) − πθ

t ≤ κ0 πθ

πθ

t−ti

t−ti

− πθ

t

,

where κ0 capture the measure of local diversity discussed in Section 3: the more a sample is reused on average between time ti and t, the bigger κ.8 Assuming G is bounded by some constant G∞, we get a bound on the bias of the form

E[εt,i | Ft] ≤ κ0G∞ πθ

###### − πθ

. Finally assuming the policy is parameterized in some Lipschitz way for some constant C, we get

t−ti

t

E[εt,i | Ft] ≤ κ0G∞C ∥θt−t

i − θt∥. This motivates formally Assumption 4.2.

Variance Assumption. When using z the i-th element of the buffer to estimate ∇F(θt), the per-sample estimator typically includes some form of off-policy correction (explicit importance weights, clipped ratios as in PPO-style objectives, or implicit reweighting through an advantage estimator). Abstractly, one may write the estimator as G(θt,zt,i) = wt,t

###### is a (possibly clipped) importance-ratio weighting between πθ

###### (z)G0(θt,z), where wt,t

i

i

###### ), and G0 is a bounded-variance on-policy quantity (e.g. a score-function term times an advantage). As τ := t − ti grows, the mismatch between πθ

###### and πθ

###### (recall that z was generated z by πθ

t

ti

ti

###### and πθ

t

ti

and thus the variance of G(θt,z). This motivates an upper bound of the form E[∥εt,i∥2] ≤ σ2(τ) for some increasing function σ2, which captures (in aggregate) the growth of off-policy noise with staleness.

###### typically increases, which in turn increases the variability of importance weight wt,t

i

Dependencies Assumption. In standard SGD analyses, the samples zt are i.i.d., and minibatching yields a 1/B variance reduction. With experience replay, however, the buffer at time t is “endogenous”: trajectories currently stored in the buffer may have been used in past updates, and those updates affected the parameters that later generated other trajectories that are now in the buffer. Concretely, εt,i = G(θt,zt,i) − ∇F(θt) depends on θt, while θt itself is a function of past minibatch draws; hence two buffer elements can become statistically coupled through the update trajectory that produced θt. At a given step, a fixed element of a buffer of size N is selected in expectation B/N times (sampling with replacement). Over h steps, it is therefore used about hB/N times. Since each update is an average over B samples, each occurrence contributes a factor 1/B to the update. Now consider two distinct buffer elements i ≠ j with insertion times ti ≤ tj. The updates in the interval [ti,tj) can transmit information from zt,i to later iterates that enter εt,j (since zt,j is only generated at time tj). Thus the strength of the coupling should generally increase with the temporal separation |ti −tj|. Aggregating algorithm-specific constants (e.g. clipping, advantage normalization, optimizer state) into a function ρ, this motivates

ρ N |ti − tj|,

corr(εt,i,εt,j) ≤

for ρ a value in [0,1]. Note that even when ti = tj, a residual dependence may remain because both trajectories can jointly influence the subsequent parameter path and hence θt. While we omit it for simplicity, adding it would not change much the derivations.

- C.2 Proof of Convergence Combining the L-smoothness Assumption 4.1 with one of Taylor expansion formulas yields

Applied in θt+1 and θt

L 2 ∥y − x∥2 .

F(y) ≤ F(x) + ⟨∇F(x),y − x⟩ +

L 2 ∥θt+1 − θt∥2 .

F(θt+1) ≤ F(θt) + ⟨∇F(θt),θt+1 − θt⟩ +

With

θt+1 − θt = −ηgt = −η(∇F(θt) + εt),

8As such, one may want to refine κ0 to be a function of the average number of time a sample was used between time ti and t, which is (min(t − ti, N/R) − 1)B/N.

we get

Lη2

2 ∥∇F(θt) + εt∥2 . Developing and rearranging leads to

F(θt+1) ≤ F(θt) − η ⟨∇F(θt),∇F(θt) + εt⟩ +

Lη2 2 ∥∇F(θt)∥2 − η − Lη2 ⟨∇F(θt),εt⟩ +

Lη2

2 ∥εt∥2 . Summing over t and rearranging with get

F(θt+1) ≤ F(θt) − η −

Lη2 2

η −

T−1

F(θ0) − F(θT) T − η − Lη2

1 T

∥∇F(θt)∥2 ≤

t=0

T−1

Lη2 2

1 T

⟨∇F(θt),εt⟩ +

t=0

T−1

1 T

∥εt∥2 .

t=0

Assuming

Lη < 1/2, (3) we get, with ξ the sign of ⟨F(θt),εt⟩,

T−1

T−1

T−1

F(θ0) − F(θT) ηT

ξ T

Lη 2

1 T

- 3

- 4T

∥∇F(θt)∥2 ≤

∥εt∥2 .

⟨∇F(θt),εt⟩ +

+

t=0

t=0

t=0

Taking the expectation with respect to FT, we bound, using Cauchy-Schwarz and a Young’s inequality, E[⟨∇f(θt),εt⟩ | FT] = E[⟨∇f(θt),εt⟩ | Ft] = ⟨∇f(θt),E[εt | Ft]⟩ ≤ ∥∇f(θt)∥∥E[εt | Ft]∥ ≤

1 4 ∥∇f(θt)∥2 + ∥E[εt | Ft]∥2 .

Hence,

T−1

T−1

1 4T

ξ T

⟨∇F(θt),εt⟩ ≤

t=0

t=0

Plugging this into the previous inequality, we get

1 T

∥∇f(θt)∥2 +

T−1

∥E[εt | Ft]∥2 .

t=0

T−1

F(θ0) − F(θT) ηT

- 1

- 2T

E[∥∇F(θt)∥2] ≤

+ E

t=0

T−1

1 T

Lη 2

1 T

∥E[εt | Ft]∥2 +

t=0

T−1

E[∥εt∥2].

t=0

We need to bound the last two quantities, which we identify as the “bias”, and the “variance” part.

###### C.2.1 Bound on the Bias.

Under Assumption 4.2, with uniform sampling over the buffer, assuming R divides N for simplicity, with H = N/R the staleness horizon,

∥E[εt | Ft]∥2 = Ei[E[εt,i | Ft]]

2

≤ Ei E[εt,i | Ft] 2 ≤ κ2Ei ∥θt − θt

i∥2 =

The drift is controlled by the magnitude of the gradient updates,

κ2 H 0≤τ<H ∥θt − θt−τ∥2 .

θt − θt−τ = η

gs = η

t−τ≤s<t

We proceed with the following bound

∇F(θs) + η

t−τ≤s<t

∥θt − θt−τ∥2 ≤ 2τη2

∥∇F(θs)∥2 + ∥εs∥2 ≤ 2Hη2

t−τ≤s<t

t−H≤s<t

εs.

t−τ≤s<t

∥∇F(θs)∥2 + ∥εs∥2 .

Summing over t, we get

T−1

1 T

t=0

1 T

∥E[εt | Ft]∥2 ≤ 2H2η2κ2

T−1

E[∥∇F(θt)∥2 | Ft] + E[∥εs∥2 | Ft].

t=0

When

2H2κ2η2 ≤ 1/4, (4) plugging our bound on the bias into the main inequality gives

T−1

F(θ0) − F(θT) ηT

1 4T

E[∥∇F(θt)∥2] ≤

+

t=0

2N2κ2η2 R2

Lη 2

+

T−1

1 T

E[∥εt∥2].

t=0

- C.2.2 Rearrangement between the Variance and the Second Moment We need to bound the second-moment E[∥εt∥2]. Let introduce

ξt,i = εt,i − E[εt,i], ξt =

1 B

j∈[B]

ξt,i

j

.

We have, reusing the previous bound on the bias, together with Eq. (4),

1 T

T−1

t=0

E[∥εt∥2] = E

1 T

T−1

t=0

E[∥εt∥2 | Ft] = E

1 T

T−1

t=0

∥E[εt | Ft]∥2 +

1 T

T−1

t=0

E[∥ξt∥2]

≤

1 4T

T−1

t=0

E[∥∇F(θt)∥2 | Ft] +

1 4T

T−1

t=0

E[∥εs∥2 | Ft] +

1 T

T−1

t=0

E[∥ξt∥2] Hence,

1 T

T−1

t=0

E[∥εt∥2] ≤

1 3T

T−1

t=0

E[∥∇F(θt)∥2 | Ft] +

4 3T

T−1

t=0

E[∥ξt∥2] Plugging this into the main bound, and rearranging,

1 12T

T−1

t=0

E[∥∇F(θt)∥2] ≤

F(θ0) − F(θT) ηT

+

4 3

2N2κ2η2 R2

+

Lη 2

1 T

T−1

t=0

E[∥ξt∥2].

- C.2.3 Bound on the Variance Using Assumption 4.3, we bound, with γ(|ti − tj|) the correlation between εt,i and εt,j,

⟨ξt,i,ξt,j⟩] = P(i = j)E[∥ξt,i∥2] + P(i ̸= j)E[⟨ξt,i,ξt,j⟩ | i ̸= j]

≤ P(i = j)E[σ(t − ti)2] + P(i ̸= j)E γ(ti − tj)σ(t − ti)σ(t − tj) Assuming R divides N for simplicity, we have, with H = N/R,

together with

E[σ(t − ti)2] =

H−1

1 H

σ(s)2 =: σ¯2(H),

s=0

E[γ(|ti − tj|)σ(t − ti)σ(t − tj)] =

=

H−1

1 H2

s,s′=0

H−1

1 H2

s=0

1 H

= σ¯2(H)

H−1

- 1

- 2H2

γ(|s − s′|)σ(s)σ(s′) ≤

s,s′=0

H−1

H−1

1 H2

γ(|s − s′|) ≤

σ(s)2

s′=0

s=0

H−1

2γ(τ) =: σ¯2(H)¯γ(H).

τ=0

γ(|s − s′|)(σ(s)2 + σ(s′)2)

H−1

σ(s)2

2γ(τ)

τ=0

We deduce that

1 B2

1 B2

1 B2

2] +

E[∥ξt∥2] =

E ξt,i

E[ ξt,i

E ξt,i

,ξi

,ξi′

=

j′

j

j

j

j

j,j′∈[B]

j̸=j′∈[B]

j∈[B]

(B − 1) B

1 B

E[∥ξt,i∥2] +

E ξt,i

=

,ξt,i

j′

j

(B − 1)¯σ2(H) B

σ¯2(H) B

(P(ij = ij′) + (1 − P(ij ̸= ij′))¯γ(H)) In the case with replacement, we get

≤

+

P(ij = ij′) = 1/N, hence

E[∥ξt∥2] ≤ σ¯2(H)

B − 1 B

1 N

1 B

+

B − 1 B

N − 1 N

γ¯(H) ≤ σ¯2

+

N R

1 B

1 N

+

+ γ¯

N R

With γ(|ti − tj|) = ρ|ti − tj|/N, we get

H−1

2 H

ρτ N

γ¯(H) =

=

τ=0

Plugging this into the main bound, we get

H(H − 1) 2

2ρ HN

=

ρ(H − 1) N ≤

ρH N

ρ R

=

.

T−1

F(θ0) − F(θT) ηT

1 T

E[∥∇F(θt)∥2] ≤ 12

+ 8ησ¯2

t=0

N R

4N2κ2η R2

+ L

1 B

1 N

ρ R

+

+

.

##### C.3 Design Trade-Off.

###### C.3.1 Solution with Specifying σ

Using that the number of gradient steps T is a direct function of the compute C = cT, where c = (B + µR), aiming to minimize the right hand-side in convergence bound of Theorem 4.4 provides design consideration for the buffer parameter R, B, N. As the compute goes to infinity, we notice that the optimal learning rate (solving a third degree polynomial equation) goes to zero. As such, the term in κ2η becomes negligible in front of L asymptotically. In this case, our analysis suggests to design of the buffer by optimizing for R, N and B in order to minimize

12F(θ0)(B + µR) Cη

+ 8Lησ¯2

J0(R,B,N,η;µ,σ,ρ,κ,C¯ ) =

N R

1 N

ρ R

1 B

+

+

Optimizing in η gives

4√6 LF(θ0) √

J0(R,B,N,η∗;µ,σ,ρ,κ,C¯ ) =

C

1 B

1 N

ρ R

N R

(B + µR)¯σ2

+

+

.

Hence, we can simplify our minimization goal by aiming to minimize

J (R,B,N;µ,σ,ρ¯ ) = (B + µR)¯σ2

N R

1 B

1 N

ρ R

+

+

Let us introduce the staleness horizon x = N/R, which corresponds to the maximum staleness of trajectories in the buffer.

1 B

1 xR

ρ R

J = σ¯2(x)(B + µR)

+

+

Let us introduce the replay ratio y, which is the average number of time a sample will be replayed during the SGD trajectory. Since a sample z stays for x iteration in the buffer, that at each iteration B samples are

extracted from the buffer, and that the sampling are independent between steps, the replay ratio is expressed as

N R × E[

N R

B N

B R

I{zt,i = z}] =

y =

=

.

i∈[B]

Using this ratio, we get

1 y

1 x

J = σ¯2(x)(1 + y/µ)

+

+ ρ

We aim to minimize J (x,y) over the domain D = (0,+∞)2. Since J is continuous, and tends to infinity on the border of the domain D, it achieves its minimum for some (x∗,y∗) ∈ D (not necessarily unique). Moreover, since J is infinitely differentiable on its domain, y∗ is characterized by ∂yJ (y,x∗) = 0, which leads to

1 y∗

1 x∗

1 x∗

1 y∗2

1 y∗2

+ ρ /µ − (1 + y∗/µ)

+ ρ /µ −

0 =

=

.

+

Hence,

√µ

y∗ =

.

ρ + x1

∗

Plugging this expression back into J gives a one-dimensional objective in x,

I(x) := J (x,√µ/ (ρ + 1/x)) = σ¯2(x) 1 +

1 µ ρ + x1

1 x

1 x

ρ +

/µ +

+ ρ

2

1 x

1 √µ

= σ¯2(x)

+ ρ +

.

where the last equality follows from the fact that for a = (ρ + 1/x),

1 √µa

1 +

a µ

+ a =

Hence the remaining design choice is

a µ

1 µ

+ a +

+

a µ

=

+ √a

1 µ

2

.

σ¯2(x)

x∗ ∈ arg min

x

1 √µ

1 x

+ ρ +

2

.

Note that we omitted integers and divisibility constraints, which would leads to a constrained version of the solution provided above.

Remark on “Under Specifications” Note our analysis reduces the parameterization of J from three variables (B,N,T) to only two ratios (x,y). This reduction follows from the homogeneity of J under the scaling transformation (B,N,R)  → (αB,αN,αR) for any α > 0. As a consequence, Theorem 4.5 characterizes optimal ratios rather than prescribing absolute values (e.g., a specific batch size), which may at first appear puzzling.

However, the scale invariance only holds in the asymptotic regime. Entering this regime requires the number of gradient steps T = C/(B + µR) to be sufficiently large, thus imposing an upper bound on B + µR for a fixed compute budget C. Similarly, integer constraints and divisibility assumptions imposes lower bounds on B, N, and R. In practice, precise finite-time bound would introduce additional quantities, which would break the homogeneity, and provide clear indications on the optimal batch size.

- C.3.2 Closed-Form Solution with Power-Law Variance Let us specify the variance profile as a power law:

σ(x) =

x τ

α

p+1

for some coefficients τ and α. Using the integral approximation Hs=0−1 sp ≈ H

p+1 , we compute the average variance σ¯2(H):

H−1

2α

H2α+1 2α + 1

2α

s τ

1 Hτ2α

1 H

- 1

- 2α + 1

H τ

σ¯2(H) =

≈

.

=

s=0

Recall that x = N/R = H. Substituting this into the design objective I(x), we aim to minimize

2

x2α (2α + 1)τ2α

1 √µ

1 x

I(x) =

.

+ ρ +

Dropping constant multiplicative factors, this is equivalent to minimizing the simplified function

K(x) = xα

1 √µ

1 x

+ ρ +

. (5)

The first-order optimality condition K′(x) = 0 yields

αxα−1

1 x

1 √µ

+ ρ +

1 2 ρ + x1

+ xα

1 x2

−

= 0.

Multiplying by 2x2−α ρ + 1/x, we obtain the algebraic equation

2αx

1 √µ

1 x

1 x

ρ +

+ ρ +

= 1 ⇐⇒ 2α

ρx2 + x µ

= 1 − 2α − 2αρx.

Squaring both sides leads to a quadratic equation Ax2 + Bx + C = 0 governing the optimal staleness x∗:

4α2(ρx2 + x)/µ = (1 − 2α − 2αρx)2 ⇐⇒ 4α2ρ(1/µ − ρ)

x2 + 4α(α/µ + ρ(1 − 2α))

x−(1 − 2α)2

= 0.

A

B

C

Solving for the positive root, and noting that the discriminant ∆ = B2 −4AC simplifies to ∆ = 16α2µ(α2/µ+ ρ(1 − 2α)), the optimal staleness horizon is given explicitly by

x∗ = −(α/µ + ρ(1 − 2α)) + α2/µ2 + ρ(1 − 2α)/µ 2αρ(1/µ − ρ)

.

To find the optimal replay ratio y∗, we avoid substituting the complex closed-form of x∗ and instead exploit the optimality conditions directly. Recall the relationship characterizing y∗:

1 (ρ + 1/x∗)/µ

1 x∗

µ y∗2

=⇒ ρ +

y∗ =

=

.

This allows us to express the staleness x∗ strictly as a function of y∗:

µ − ρy∗2 y∗2

y∗2 µ − ρy∗2

µ y∗2 − ρ =

1 x∗

=⇒ x∗ =

=

.

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

OptimalStaleness(x)∗

- 100
- 101

100 101

Rollout cost (µ)

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

OptimalReplayRatio(y)∗

- 100
- 101

100 101

Rollout cost (µ)

- ρ=0.1, α=0.1

- ρ=1.0, α=0.1

- ρ=0.1, α=0.2

- ρ=1.0, α=0.2

On-policy regime

- Figure 6 Optimal Staleness and Replay Ratio as a function of Rollout Cost (µ). As µ increase, we see that it is better to increase the staleness horizon x∗ = N/R, and the replay ratio (y∗ = B/R). This also the case when the variance α or the correlation ρ decreases.

√µ y∗ into the first-order optimality condition derived for x∗:

Substituting the term ρ + 1/x∗ =

1 x∗

1 x∗

1 y∗

µ y∗2

1 √µ

).

1 = 2αx∗

ρ +

+ ρ +

= 2αx∗(

+

Simplifying the term in the parenthesis yields = 1, or equivalently:

y∗2 2α(µ + y∗)

x∗ =

.

Equating the two characterization of x∗ gives

y∗2

y∗2 µ − ρy∗2

2α(µ + y∗) ⇐⇒ µ − ρy∗2 = 2α(µ + y∗). Rearranging terms yields a quadratic equation in y∗:

=

ρy∗2 + 2αy∗ + µ(2α − 1) = 0. Assuming α < 1/2, the constant term µ(2α − 1) is negative, guaranteeing a unique positive solution:

y∗ = −α + α2 + µρ(1 − 2α) ρ

.

An illustration of the formula for x∗ and y∗ is provided in Figure 6, and the function x  → K(x) (as well as the optimum value x∗) is shown in Figure 7.

#### D Experimental details

We provide additional details regarding our experimental setup.

##### D.1 Hardware and parallelism We use Nvidia H100 and H200 GPUs.

Our experiments are run on either 1,2 or 4 8-GPUs nodes, with data parallelism and without tensor parallelism. When describing a buffer experiment ran on more than 1 node, we report T and W divided by the number of nodes; in other words, we describe an experiment run on 16 GPUs with 4 trainer GPUs and 12 inference GPUs as (6,2) rather than (12,4). We do so to simplify notations, and because increasing the number of nodes while keeping the same ratio W/T does not impact any of the relevant quantities (size of the buffer,

4.0

α = 0.00 α = 0.10 x∗(α = 0.10)

α = 0.30

x∗(α = 0.30)

3.5

α = 0.50

α = 0.20

x∗(α = 0.50)

x∗(α = 0.20)

3.0

K(x)

2.5

2.0

1.5

1.0

0 2 4 6 8 10

x

- Figure 7 Function x  → K(x) (which is defined in Eq. (5) and corresponds to J for the specific choice σ(x) = (x/τ)α) as a function of the staleness horizon x = N/R, for different values of α ∈ [0, 1/2], and the corresponding optimal values of x∗.

replay ratio, off-policiness): the training dynamics remain the same (up to essentially random effects linked to inter-nodes communications), and the training is accelerated with respect to wall-time, which we do not take into account when estimating compute (in other words, we consider that the cost of a gradient step is not affected by the number of nodes).

Our non-buffer experiments are run with (W,T) ∈ {(4,4),(5,3),(6,2)}: though we find that the theoretical optimal ratio µ is closer to W/T = 5, ratios closer to 1 are in practice better when training on a small number of GPUs (e.g. 8 or 16). This is because letting T be very small (e.g. T ∈ {1,2}) forces the maximum micro-batch size to also be very small, while large micro-batch sizes are needed to leverage parallelism-based optimizations.

##### D.2 Optimization and general hyperparameters

We train using the Adam (Kingma and Ba, 2014) optimizer with constant learning rates. We use a batch size of 60, except in the few runs for which T = 7, for which we let the batch size be 63 (as it must be divisible by the number of trainer GPUs). Unless otherwise specified, we use a learning rate of 6.8 · 10−8 for Qwen2.5-7B and of 3.37 · 10−7 for Qwen3-0.6B.

We use the following GRPO implementation (see Shao et al. (2024)):

πθ(z|q) πθ

πθ(z|q) πθ

A,clip

JGRPO(θ) = Eq∼D,z min

,1 − εlow,1 + εhigh A ,

(z|q)

(z|q)

old

old

where q is a prompt sampled from a training distribution D and z is a rollout sampled from the buffer following the chosen sampling strategy. Both the probability πθ

(z|q) and the advantage A of z are computed at the time when z is first generated. More specifically, a group of G rollouts z1,...,zG is generated by the inference workers for each prompt q, and the advantage Ai of zi is defined as

old

ri − mean({r(z1,q),r(z2,q),··· ,r(zG,q)}) std({r(z1,q),r(z2,q),··· ,r(zG,q)})

. (6)

Ai =

In other words, the advantage is computed when the rollout is generated (and not when it is used to compose a gradient update).

In particular, we do not include a KL regularization term, as recent research suggests that it does not improve performance (see e.g. Yu et al. (2025)). We let εlow = εhigh = 0.2, and we let the group size G be equal to 16. Note that when this loss is combined with a buffer, it can be shown that the joint distribution over the

current training batch (which is assembled by sampling from the replay buffer) is not corrected in expectation by the importance sampling factor π

θ(z|q)

πθold(z|q) (even without taking the clipping into account). We also consider the AsymRE objective function from Arnal et al. (2025), expressed using the same notations as

G

1 G

(r(z,q) − (Vˆ + δV ))log(πθ(z|q)) ,

JAsymRE(θ) = Eq∼D,z

i=1

where Vˆ := mean({r(z1,q),r(z2,q),··· ,r(zG,q)}) if z1,...,zG is the group of generated rollouts to which z belongs (see above) and δV = −0.1.

We train Qwen3-0.6B without weight tying. We use a temperature of 1 when generating training samples, of 0.1 when evaluating pass@1 (with top_p = 0.95), and of 1 when evaluating pass@k with k > 1 (with top_p = 0.95).

##### D.3 Metrics

Compute Our abstract measure of compute is in closest correspondence to the notion of FLOPS, but we make throughout the text the following implicit assumptions, which are never completely realized in practice:

- • We are in an optimized settings in which there is a direct correspondence between FLOPS and GPU work time, except when a GPU is idle because it is waiting on the work of other GPUs,
- • Tasks can be continuously parallelized; in other words, there are no boundaries effects due to the discrete nature of the number of samples and GPUs, and
- • When parallelizing a task between K GPUs, the total compute spent is not a function of K.

In particular, we ignore the effects of important implementation details, such as tensor parallelism, data parallelism, sharding, etc.

Steps-since-last-use We define in greater detail the steps-since-last-use metric reported in Figure 2. In the context of this paragraph, we use the term "rollout" to refer to a given data point, and the term "sample" to refer to a data point as it appears in a gradient descent batch. Each rollout (a given sequence of tokens) can correspond to zero, one or several samples belonging to one or several batches depending on how often it was sampled from the buffer.

- • We order all samples used during a training trajectory:

- – For every batch B, we pick a random ordering of the samples of B.
- – If batch B was processed before batch B′, then z < z′ for any z ∈ B,z′ ∈ B′.

- • Whenever a rollout appears as a sample for the first time according to this global order, we associate the value "new" to the sample.
- • If a sample z corresponds to a rollout that has already given rise to an earlier sample z′, then we associate to z the number of gradient steps taken since z′.

As an illustration, let us assume that a rollout gives rise to exactly four samples: z1 ∈ B3 and z2,z3,z4 ∈ B5, where the numbering of the samples reflect their ordering and the batch Bi was used at time i. Then z1 is mapped to "new", z2 to 2, z3 to 0 and z4 to 0. In Figure 2, we plot the histogram of the values taken by steps-since-last over all samples of each trajectory considered.

Pass@k The pass@k curve from Figure 3 is computed as follows: for a given k and a given choice of hyperparameters (with or without buffer, etc.), the median over the random seeds of the pass@k training curve is computed. We then report the maximum of this median curve over the training trajectory, as well as the IQR at the step where the maximum is reached. In particular, the corresponding training step is in general not the same for distinct choices of k.

##### D.4 Buffer-specific aspects Compute ratio γ

To estimate the compute cost of a parameter update in a buffer configuration with T trainer GPUs and W inference GPUs, we use the compute ratio

1 + W/T 1 + µ

γ =

,

defined in Equation (2). This quantity depends in turn on the optimal ratio µ, defined as the compute cost of generating a rollout divided by the compute cost of processing it through a gradient update; equivalently, µ is the exact number of inference GPUs for each trainer GPUs required so that there is no downtime.

This quantity depends on the model, dataset, implementation details and hardware, as well as on some parameter choices (such as the batch size). Consider a training run using a replay buffer, and the following quantities:

- • Ktraining the number of non-unique rollouts processed through backpropagation over the entire run, i.e. the number of gradient steps multiplied by the batch size,
- • Kinference the number of unique rollouts generated by the inference GPUs over the entire run,
- • T the number of trainer GPUs, and
- • W the number of inference GPUs.

Each trainer GPU will have processed Ktraining/T rollouts, and each inference GPU will have generated Kinference/W rollouts on average. As inference and trainer GPUs work independently from each other when using a replay buffer and do not suffer any downtime, we can consider that this is a fair measure of their relative speed (or equivalently of the relative compute cost of training vs inference), and use it to estimate µ:

Ktraining/T Kinference/W

µ ∼=

.

To make this estimate more precise, we use the median value over several random seeds. We report in the table below our estimates of the coefficients µ for the various models featured in our experiments.

|Model| |Median µ<br><br>|IQR|# Independent runs|
|---|---|---|---|---|
|Qwen3-0.6B| |6.84|[6.45,7.07]<br><br>|242|
|Qwen2.5-7B| |5.28|[5.12,5.48]<br><br>|129|

- Table 2 Estimates of µ for various models, computed as the median over several independent runs. We also provide the interquartile range over those runs.

We provide in Table 3 the γ values corresponding to Qwen3-0.6B.

|(W, T)<br><br>|(7,1)|(6,2)<br><br>|(5,3)|(4,4)<br><br>|(2,6)|(1,7)|
|---|---|---|---|---|---|---|
|γ|1.02<br><br>|0.41|0.34<br><br>|0.26|0.17<br><br>|0.15|

- Table 3 γ for various values of (W, T) and an estimated µ = 6.84 for Qwen3-0.6B.

Sharded buffers In our concrete implementation, each trainer GPU maintains its own separate replay buffer. In other words, newly generated rollouts are added to the replay buffers of the various trainer GPUs (each a list of trajectories) in a balanced way, with each rollout being added to the replay buffer of a single trainer GPU. Each trainer GPU, when creating a sub-batch from which it will compute a gradient (which will then be averaged with the gradients of other trainer GPUs), samples only from its own buffer. We always report the total buffer size N, which is the sum over the T trainer GPUs of the sizes N/T of their separate buffers. Our preliminary experiments suggest that this design choice has little impact.

Positive-bias sampling We introduced in Section 5 an alternative buffer strategy, which we call positive-bias sampling: instead of keeping the freshest N generated rollouts in the buffer, we keep the freshest (1 − δ)N generated rollouts along with the freshest δN correct rollouts not included in those (1 − δ)N trajectories. As an example, if N = 8, δ = 0.75 and the last rollouts to be produced are

...zt−9 zt−8 zt−7 zt−6 zt−5 zt−4 zt−3 zt−2 zt−1 zt,

where incorrect and correct rollouts are shown in red and green respectively, then the buffer at time t is equal to

zt−8 zt−7 zt−5 zt−4 zt−3 zt−2 zt−1 zt

#### E Additional Experimental Results

We provide various additional experimental results:

- • In Figures 8 and 9, we run ablations to select the best learning rate for Qwen3-0.6B and Qwen2.5-7B. We see that 3.37 · 10−7, respectively 6.810−8, achieve the best balance between speed and stability.
- • We report accuracy with respect to wall-time for Qwen3-0.6B and Qwen2.5-7B in Figures 10 and 11.
- • We study the impact of off-policiness in no-buffer configurations in Figure 12.
- • We report accuracy with respect to compute and training dynamics for Qwen3-0.6B with various buffer sizes and W/T ratios in 13. We also report the best accuracies achieved and the corresponding compute costs in Figure 14.
- • We test our methods on other models and tasks in Figures16 (Qwen3-8B on Lean coding tasks) and 17 (Llama 3.2 3B on OpenR1-Math-220k).
- • In complement to Figure 1, we report in Figure 15 the results for additional buffer configurations for Qwen2.5-7B.
- • We show that alternative sampling strategies based on sampling without replacement do not have a clear impact in Figure 18.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.55

0.50

Accuracy(pass@1)

- LR=6.67e-08

LR=1e-07

- LR=1.5e-07

- LR=2.25e-07

- LR=3.37e-07

LR=5.06e-07

- LR=7.59e-07

0.45

0.40

0.35

0.30

0.25

0.20

0 2k 4k 6k 8k 10k

0 2k 4k 6k 8k 10k

Step

Step

- Figure 8 Learning Rate Ablations for Qwen3-0.6B. Test accuracy as a function of the number of steps when training Qwen3-0.6B on OpenR1-Math-220k with various learning rates LR with at least 4 seeds per configuration. We show the median and IQR over the seeds on the left, and all seeds separately on the right.

0.76

Accuracy(pass@1)

0.74

LR=6e-08

0.72

- LR=1e-07

- LR=2e-07

0.70

- LR=4e-07

- LR=5e-07

- LR=6e-07

0.68

0.66

0 2k 5k 7k 10k 12k 15k 17k 20k

0 2k 5k 7k 10k 12k 15k 17k 20k

Step

Step

- Figure 9 Learning Rate Ablations for Qwen2.5-7B. Test accuracy on MATH as a function of the number of steps when training Qwen2.5-7B on OpenR1-Math-220k with various learning rates LR with at least 4 seeds per configuration. We show the median and IQR over the seeds on the left, and all seeds separately on the right. Note the frequent crashes when LR > 6.8 · 10−8.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 1 2 3 4

Wall Time (hours)

0.40

0.43

0.45

0.48

0.50

0.53

0.55

Accuracy(pass@1)

(W,T) = (6,2)

0 1 2 3 4

Wall Time (hours)

(W,T) = (5,3)

0 1 2 3 4

Wall Time (hours)

(W,T) = (4,4)

Buﬀer = 64

Buﬀer = 256 Buﬀer = 512 Buﬀer = 768

Buﬀer = 1536 Buﬀer = 2304 Buﬀer = 4608 Buﬀer = 9216 Buﬀer = 20736

Baseline

- Figure 10 Wall-time efficiency for Qwen3-0.6B Test accuracy as a function of wall-time when training Qwen3-0.6B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and various buffer configurations. We report the median and the IQR over at least 4 seeds per curve.

0 5 10 15 20 25 30 35 40

Wall Time (hours)

0.67

0.68

0.69

0.70

0.71

0.72

0.73

0.74

0.75

0.76

Accuracy(pass@1)

(W,T) = (5,3)

Buﬀer = 84

Buﬀer = 2268

Buﬀer = 20412

Baseline

- Figure 11 Wall-time efficiency for Qwen2.5-7B Test accuracy on MATH as a function of wall-time when training Qwen2.5-7B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and various buffer configurations. We report the median and the IQR over at least 4 seeds per curve.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.5

Accuracy(pass@1)

0.4

0.3

Oﬀ-policiness:

0.2

3

13 28 98

0.1

0 2k 4k 6k 8k 10k 12k 14k

Step

- Figure 12 Impact of off-policiness. We train Qwen3-0.6B on OpenR1-Math-220k without a buffer and we artificially introduce various levels of off-policiness by reducing the frequency at which the model’s weights used by the inference workers to generate rollouts are updated. We label each curve with the median level of off-policiness over all rollouts used, and plot the median test accuracy and its IQR as a function of the number of training steps over at least 4 seeds per curve.

Buﬀer size N

64

512 768 1536

2304 4608 9216

20736 62208 186624

559872

128 256

Baseline (step)

Baseline (compute)

(W,T) = (6,2)

(W,T) = (5,3) (W,T) = (4,4)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.54

Accuracy(pass@1)

0.52

0.50

0.48

0.46

0.44

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.50

TrainAccuracy(pass@1)

0.45

0.40

0.35

0.30

0.25

0.60

0.50

TrainingEntropy

0.40

0.30

0.20

0.10

0 2500 5000 7500 10000 12500 15000 17500

0 5000 10000 15000

0 5000 10000 15000

Step

Step

Step

- Figure 13 Test, Train and Entropy Dynamics. We train Qwen3-0.6B on OpenR1-Math-220k with a buffer for (W, T) ∈ {(6, 2), (5, 3), (4, 4)} and various buffer sizes. We report the test accuracy (top), the training accuracy (middle, smoothed using a sliding window), and the training entropy (bottom) as a function of the number of training steps. Note that the training entropy is computed over the batches used by the trainers to compute gradient updates; as using a buffer implies reusing samples generated by outdated policies, it is expected that the reported entropy would be much higher. We also report two baseline curves, corresponding to non-buffer configurations: one is plotted with respect to the number of steps, while the other is rescaled to be at compute-parity with the buffer configurations (i.e. so that an x-axis unit represents the same amount of compute). Each curve is the median (along with its IQR) over at least 4 seeds.

Best Accuracy

###### Compute to 98% of Best Accuracy

NumberofTrainers

|[Figure 3]| |
|---|---|
| | |
| | |
| | |

|[Figure 4]| |
|---|---|
| | |
| | |
| | |

[Figure 5]

[Figure 6]

15000

4

4

0.54

10000

3

3

0.52

5000

2

2

0.50

6425651276815362304460892162073662208186624559872

6425651276815362304460892162073662208186624559872

Buﬀer Size

Buﬀer Size

- Figure 14 Accuracy and Speed with respect to Design Choices. We train Qwen3-0.6B on OpenR1-Math-220k for (W, T) ∈ {(6, 2), (5, 3), (4, 4) and various buffer sizes. We report on the right the median best test accuracy achieved over each training run (in other words, the median over the seeds of the best accuracy achieved for each seed), and on the left the median amount of compute that was needed to first reach 98% of that score.

0 5000 10000 15000 20000 25000 30000 35000

Compute

0.72

0.73

0.74

0.75

0.76

0.77

Accuracy(pass@1)

N = 84

N = 512

N = 2268

N = 20412

Baseline

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 5000 10000 15000 20000 25000 30000 35000

Compute

0.72

0.73

0.74

0.75

0.76

0.77

Accuracy(pass@1)

N = 84 N = 512 N = 2268 N = 20412 Baseline

- Figure 15 Additional results for Qwen2.5-7B. Accuracy on MATH as a function of compute spent when training Qwen2.5-7B on OpenR1-Math-220k for the no-buffer baseline (orange curve) and a buffer of size N ∈ {84, 512, 2268, 20412} with (W, T) equal to (6, 2) (left) or (5, 3) (right). We report the median and IQR over more than 4 seeds. Compute is calibrated so that a single weight update for the baseline costs 1 unit.

0 200 400 600 800 1000

Compute

0.15

0.20

0.25

0.30

0.35

Accuracy(pass@1)

N = 128

N = 10000

Baseline

- Figure 16 Accuracy with respect to Buffer Size for Qwen3-8B on Lean coding tasks. Test accuracy as a function of compute spent when training Qwen3-8B on miniF2F for (W, T) = (6, 2) and various buffer sizes N ∈ {128, 10000}, as well as for a no-buffer baseline. We report the median and IQR over 4 seeds. Compute is normalized so that each weight update costs 0.55 unit for buffer configurations and 1 for the baseline.

0.325

0.300

Accuracy(pass@1)

0.275

0.250

0.225

0.200

(W, T) = (6, 2), N = 128

0.175

(W, T) = (6, 2), N = 2048

(W, T) = (6, 2), N = 16384

0.150

Baseline

0.125

0 2000 4000 6000 8000 10000 12000

Compute

- Figure 17 Accuracy with respect to Buffer Size for Llama 3.2 3B. Test accuracy as a function of compute spent when training Llama 3.2 3B on OpenR1-Math-220k for (W, T) = (6, 2) and various buffer sizes N ∈ {128, 2048, 16384}, as well as for a no-buffer baseline. We report the median and IQR over 4 seeds. Compute is normalized so that each weight update costs 0.58 unit for buffer configurations and 1 for the baseline.

0 2000 4000 6000 8000

Step

0.44

0.46

0.48

0.50

0.52

Accuracy(pass@1)

(W,T) = (6,2), N = 64

0 2000 4000 6000 8000

Step

(W,T) = (5,3), N = 768

With replacement (vanilla)

No replacement

No replacement, at least once

- Figure 18 We compared our standard buffer implementation, in which the buffer is sampled uniformly by the trainer GPUs ("vanilla"), with two variants: one in which the sampling is done uniformly without replacement ("No replacement"), and one in which samples that have never been used are sampled in priority, after what the remainder of the batch is filled without replacement ("No replacement, at least once"). We did not find any strong signal, as exemplified by these two representative buffer configurations (with which we trained Qwen3-0.6B on OpenR1-Math220k).

