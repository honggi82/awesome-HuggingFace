## Blockwise Advantage Estimation for Multi-Objective RL with Verifiable Rewards

Kirill Pavlenko*1 Alexander Golubev*1 Simon Karasik1 Boris Yangel2

# arXiv:2602.10231v1[cs.LG]10Feb2026

### Abstract

Group Relative Policy Optimization (GRPO) assigns a single scalar advantage to all tokens in a completion. For structured generations with explicit segments and objectives, this couples unrelated reward signals across segments, leading to objective interference and misattributed credit. We propose Blockwise Advantage Estimation, a family of GRPO-compatible methods that assigns each objective its own advantage and applies it only to the tokens in the corresponding text block, reducing reliance on hand-designed scalar rewards and scaling naturally to additional objectives. A key challenge is estimating advantages for later blocks whose rewards are conditioned on sampled prefixes; standard unbiased approaches require expensive nested rollouts from intermediate states. Concretely, we introduce an OutcomeConditioned Baseline that approximates intermediate state values using only within-group statistics by stratifying samples according to a prefixderived intermediate outcome. On math tasks with uncertainty estimation, our method mitigates reward interference, is competitive with a state-ofthe-art reward-designed approach, and preserves test-time gains from confidence-weighted ensembling. More broadly, it provides a modular recipe for optimizing sequential objectives in structured generations without additional rollouts.

### 1. Introduction

Reinforcement Learning (RL) has gained significant traction as the primary paradigm for post-training Large Language Models (LLMs), enabling them to solve complex tasks that are difficult to supervise with static datasets (Ouyang et al., 2022). Among recent methods, Group Relative Policy Optimization (GRPO) (Shao et al., 2024) and its mod-

*Equal contribution 1Nebius 2The Humanoid, Work done while at Nebius. Correspondence to: Alexander Golubev <alex golubev@nebius.com>.

Preprint. February 12, 2026.

ifications (DAPO (Yu et al., 2025), GSPO (Zheng et al., 2025), CISPO (MiniMax et al., 2025), ScaleRL (Khatri et al., 2025)) have become particularly popular due to their stability and memory efficiency. By normalizing rewards within a sampled group of trajectories rather than relying on a separate value network, these methods avoid the need for maintaining a critic model, significantly reducing the computational footprint of training inherent to classical PPO.

Despite these advantages, standard GRPO typically treats each generated completion as a monolithic trajectory: it computes a single scalar advantage and applies it uniformly to all tokens. This formulation is often suboptimal for complex tasks involving sequential subgoals, independent rubrics, or multi-step refinements. Given a sequence of rewards r1,...,rn, the standard approach is to construct a scalar function R = f(r1,r2,...,rn) that maps all rewards into one number and then apply a GRPO-like algorithm (Shao et al., 2025; Feng et al., 2025a). This exacerbates credit assignment and requires careful handcrafting of R to avoid score degradation described in RLCR (Damani et al., 2025), mode collapse, and reward hacking (Feng et al., 2025a).

Many LLM tasks, however, are naturally segmented: each rk can be assigned to non-overlapping text blocks (e.g., a solution to a problem gets r1, and the following self-reflection block gets r2). Our core insight is that, in this case, optimization does not have to be framed as a reward design problem. Instead, we can update each block independently based on signals tied specifically to its local objective rk. Motivated by this observation, we introduce Blockwise Advantage Estimation, a family of GRPO-compatible methods that, using only the initial set of rollouts, compute separate proxy-advantages for different text segments and update each segment using only its local objective signal.

A central technical challenge is baseline estimation for later segments. Consider a generation partitioned into segments X1,X2,..., where Xi is sampled conditioned on the prefix up to Xi−1. To form a low-variance advantage for segment Xi, one ideally normalizes the segment-level reward relative to the value of the intermediate state at the end of segment Xi−1. A naive solution is to perform Monte Carlo rollouts from the intermediate state, similar to VinePPO (Kazemne-

jad et al., 2025) or SPO (Guo et al., 2025), yielding unbiased estimates but incurring prohibitive compute for long-context LLMs and multi-thousand-token generations.

We therefore focus on scalable baseline estimators that use only data already produced within the current GRPO group. We study a series of progressively more informative estimators and find that an Outcome-Conditioned Baseline performs best in practice. This allows us to approximate the state-value function and reduce variance without additional inference costs. Our contributions are as follows:

- • We introduce Blockwise Advantage Estimation, a critic-free, GRPO-compatible framework for multiobjective RL on structured generations that routes objective-specific advantages to the segments of tokens that control them, reducing reliance on hand-tuned scalar reward combinations.
- • We formalize the conditional baseline problem that arises because later segments are sampled conditioned on earlier sampled segments, and show that the Group Mean Baseline proposed in vanilla GRPO (normalizing rewards for all turns by the within-group mean and variance) can be suboptimal for credit assignment in multisegment generations. We then propose a computeefficient Outcome-Conditioned Baseline that approximates boundary-state values using only within-group statistics, reducing variance for later segments without Monte-Carlo rollouts or extra inference.
- • We demonstrate the efficacy of our method on the task of joint reasoning and uncertainty estimation across three models. Our approach performs on par with a state-of-the-art (SoTA) reward-designed baseline like RLCR in terms of Expected Calibration Error (ECE) and reasoning accuracy.
- • We demonstrate broader applicability in the additional setting of two-attempt RL for math, and discuss implications for long-horizon agentic tasks where segmentlevel decoupling enables efficient optimization over extremely long contexts.

### 2. Related Work

Credit assignment beyond trajectory-level advantages. A classical response to coarse trajectory-level credit assignment is to move toward finer-grained advantage estimates. Token-level methods (e.g., PPO with GAE) can in principle provide per-token learning signals, but require fitting a value function, which is empirically difficult in LLM settings due to the diversity of prompt-conditioned states and limited per-prompt data. Recent critic-free approaches instead estimate intermediate advantages by Monte Carlo rollouts from

selected intermediate states. VinePPO partitions generations into heuristic steps (e.g., line breaks) and resamples continuations to estimate step-level advantages. Segment Policy Optimization generalizes this idea to arbitrary segment partitions and estimates segment-level advantages via MC. These methods primarily address credit assignment for a single terminal objective and still rely on additional sampling from intermediate prefixes to estimate intermediate values. In contrast, our setting is multi-objective with rewards naturally attached to disjoint text blocks, and our goal is a compute-efficient GRPO-compatible method: to estimate per-block baselines using only the rollouts already present in the group.

Stratified and conditional advantage estimation. Several recent works show that advantage normalization can be systematically miscalibrated when trajectories are heterogeneous in ways that affect reward distributions. Stratified GRPO (Zhu et al., 2025) identifies cross-stratum bias in search agents: using a global baseline across trajectories with different tool-use structures induces an “apples-tooranges” offset that distorts credit assignment; their Stratified Advantage Normalization computes normalized advantages within homogeneous strata and can be blended with a global estimator for finite-sample stability. CANON (Chen et al., 2025) proposes conditional regrouping based on intrinsic rollout metrics (e.g., entropy or response length), computing both inter-group and intra-group advantages. Both lines use trajectory-level stratification to sharpen learning signals for a single reward. Our conditionality problem is different: in segmented generations, later blocks are sampled from different intermediate prefixes inside the same prompt group, so the appropriate baseline for block k is a conditional value E[rk | x,X<k].

Reward shaping, process supervision, and selfverification objectives. Another common approach to multi-signal training is reward design: one constructs a single scalar reward that mixes multiple desiderata, often requiring careful tuning to avoid trade-offs or exploitation. Posterior-GRPO (Fan et al., 2025) incorporates processbased rewards for code reasoning by training a reasoningquality reward model and then applying process rewards only to successful outcomes to mitigate reward hacking; their posterior assignment also helps distinguish among uniformly successful samples where GRPO would otherwise yield near-zero learning signal. GRPO-Verif (Wang et al., 2025) optimizes both solution generation and verification responses by computing separate group-normalized advantages for solutions and verifications and combining them with a weighting hyperparameter. This approach is one representative of our proposed family of methods; we additionally show that our Outcome-Conditioned Baseline improves credit assignment for later blocks without requiring extra

rollouts.

### 3. Preliminaries

##### 3.1. Group Relative Policy Optimization

We briefly review GRPO and introduce notations used throughout the paper. Let x ∼ D denote a prompt sampled from a dataset, and let πθ be a language-model policy that generates a completion y = (y1,...,yT) in an autoregressive way:

πθ(y | x) =

T

πθ(yt | x,y<t).

t=1

Given a scalar reward function R(x,y) ∈ R (e.g., a score from verifier or preference model), GRPO estimates a relative advantage using a group of samples from the same prompt. Concretely, for each prompt x, GRPO draws a group of G completions {y(i)}Gi=1 ∼ πθ

(· | x) and evaluates rewards r(i) = R(x,y(i)). For each completion, it then forms a group-normalized advantage,

old

Aˆ(i) =

G

r(i) − µr σr + ϵ

1 G

r(j),

, µr =

j=1

G

1 G

(r(j) − µr)2,

σr =

j=1

where ϵ > 0 is a small constant for numerical stability.

GRPO optimizes a PPO-style clipped policy-gradient objective, applying the same scalar advantage Aˆ(i) to all tokens in the completion:

how well a model’s reported confidence matches empirical correctness. For each example i ∈ {1,...,N}, let ci ∈ {0,1} denote the correctness of a solution determined by a verifier, and let qi ∈ [0,1] denote the model’s confidence, interpreted as a predicted probability that the solution is correct.

Brier score. The Brier score measures the squared error between predicted probability and the binary outcome:

N

1 N

(qi − ci)2.

Brier =

i=1

For binary events, the Brier score is a strictly proper scoring rule: it is minimized (in expectation) when qi equals the true conditional probability of correctness.

Expected Calibration Error (ECE). ECE compares average confidence to empirical accuracy within confidence bins. Let {Bm}Mm=1 be a partition of [0,1] into M bins, and let Bm = {i : qi ∈ bin m}. Define

1 |Bm| i∈B

acc(Bm) =

ci,

m

1 |Bm| i∈B

conf(Bm) =

###### qi.

m

Then

M

|Bm| N

acc(Bm) − conf(Bm) .

ECE =

m=1

ECE is widely used in practice but depends on the binning scheme (choice of M and bin boundaries).

1 G

LGRPO(θ) = −

G

i=1

Ti

min ρi,t(θ)Aˆ(i),

t=1

clip(ρi,t(θ),1 − η,1 + η)Aˆ(i) − βDKL[πθ||πref] ,

(1)

where η is the PPO clip parameter and

πθ yt(i) | x,y<t(i) πθ

(2)

ρi,t(θ) =

yt(i) | x,y<t(i)

old

is the per-token likelihood ratio between the updated policy and the behavior policy used to generate the samples.

##### 3.2. Calibration Metrics for Confidence

As will be discussed in Section 4.1, we study models that optimize for both correctness and calibration. We evaluate

### 4. Method

##### 4.1. Self-Confidence Estimation for Math

We study joint reasoning and uncertainty estimation in math problem solving. Each example consists of a prompt x (a math question) and a ground-truth final answer y⋆. Given x, the model generates a structured completion X that contains (i) a solution/answer segment and (ii) a self-analysis segment that includes a numerical confidence score:

X = [Xsol; Xconf ], q ∈ [0,1] parsed from Xconf. The illustrative example of completion from a model is demonstrated in Appendix E. In our implementation, blocks are defined by deterministic parsing rules based on the structured tags (Appendix A): we treat the full solution/answer content as Xsol and the self-analysis plus the numeric confidence report as Xconf. We extract the predicted final answer yˆ from Xsol and compute a verifiable correctness indicator c = 1{yˆ ≡ y⋆} ∈ {0,1},

where ≡ denotes equivalence under a task-specific verifier (e.g., exact match up to formatting).

In standard RL with verifiable rewards (RLVR), the reward depends only on correctness:

###### RRLVR(ˆy,y⋆) = c.

While effective for improving accuracy, this reward provides no incentive for calibrated confidence and can encourage overconfident guessing.

Adding an additional reward to incentivize uncertainty estimation makes it possible to do in a number of ways, where it is not obvious which one to choose. RLCR work claims SoTA performance in the described task and suggests augmenting the binary correctness reward with a calibration term based on a proper scoring rule. Concretely, RLCR uses the Brier score for the reported confidence q:

RRLCR(ˆy,q,y⋆) =

= RRLVR(ˆy,y⋆) − RBrier(q,c) = c − (q − c)2.

This reward penalizes being confidently wrong (large q when c = 0) and also penalizes under-confidence when correct (small q when c = 1), encouraging q to track the empirical probability of correctness while still rewarding accurate solutions.

The reward design for the available segments represents a crucial part of training a model. In Section 5.4, we observe that when the confidence objective is implemented via a binary cross-entropy reward, vanilla GRPO can suffer markedly, motivating the segment-aware credit assignment and baselines we develop.

In the remainder of this section, we use this task as a running example to motivate our approach: correctness is primarily controlled by Xsol, whereas calibration is primarily controlled by Xconf.

##### 4.2. Blockwise Decomposition

We consider completions that decompose into K contiguous blocks,

y = [X1; X2; ...; XK ], where each block Xk is associated with an objective-specific reward rk.1 For a prompt x, GRPO samples a group {y(i)}Gi=1 and yields block rewards {rk(i)}Gi=1.

The modification we explore in this paper is to replace the single completion-level advantage with blockwise proxy-

advantages {Aˆ(ki)}Kk=1, and apply Aˆ(ki) only to tokens that

1The reward rk may depend on the full prefix up to block k (e.g., a confidence reward depends on whether the preceding solution is correct), but it is primarily controlled by the tokens in block k.

belong to block k. Concretely, let Tk(i) denote the token indices in block k of sample i. Using the standard token-

level importance ratio ρ(ti)(θ), defined in Equation 2, we optimize the sum of PPO-style clipped objectives across blocks:

G

K

1 G

1 |Tk(i)| t∈T (i)

LBAE(θ) = −

min

i=1

k=1

k

ρ(ti)(θ)Aˆ(ki), clip(ρ(ti)(θ),1 − ε,1 + ε)Aˆ(ki) ,

Block weighting. We average the token loss within each block by |Tk(i)| so that blocks with very different lengths (e.g., a short confidence report vs. a long solution) have comparable per-token influence. More generally, one could introduce explicit block weights wk and optimize k wk · |Tk(i)|−1 t∈T (i)

ℓk,t; we use the uniform choice wk = 1 throughout to avoid additional tuning.

k

For the first block X1, all group samples share the same starting state (the prompt x), and GRPO’s group normalization provides a natural baseline. For blocks k > 1, however, the starting state includes the sampled prefix [X1;...;Xk−1], which differs across group members. A statistically appro-

priate advantage for block k should therefore normalize rk(i) relative to the value of the intermediate state at the end of

block (k−1), i.e., a conditional baseline of the form

bk x,X<k(i) ≈ E rk | x,X<k(i) . (3) Estimating this quantity naively would require Monte Carlo rollouts from each intermediate prefix X<k(i), which is prohibitively expensive for long-context LLM training and multi-objective scenarios.

In the next subsection, we therefore study practical estimators that approximate these conditional baselines using only the samples already available within the GRPO group, avoiding additional rollouts. In Section 5.1, we show how such baselines correspond to true Monte-Carlo estimates.

See Appendix B for pseudocode.

##### 4.3. Conditional Baselines

We consider the following estimators for block k > 1, and apply them to form proxy-advantages Aˆ(ki) = rk(i) − ˆb(ki) (with optional variance normalization as in GRPO):

##### 1. No baseline (raw reward).

###### ˆb(ki) = 0, Aˆ(ki) = rk(i).

This is an unbiased estimator for state-action function which typically has high variance.

- 2. Batch Mean Baseline. Using all samples in the current batch (across prompts),

ˆb(ki) = Ej∼batch[rk(j) ].

This reduces variance, but may be poorly matched to a given prompt’s difficulty and to the specific prefix

X<k(i).

- 3. Group Mean Baseline. Using the GRPO group for the same prompt x, ignoring that prefixes differ,

ˆb(ki) =

1 G

G

j=1

rk(j).

This is prompt-adaptive, but can be miscalibrated for k > 1 because rewards are averaged across trajectories with different intermediate prefixes. This is the original proposition from GRPO work how it can be applied in multi-turn scenarios.

- 4. Outcome-Conditioned Baseline (OCB). Partitioning the group into subgroups based on a discrete

intermediate outcome o(i) of the prefix X<k(i). Let Go = {j ∈ {1,...,G} : o(j) = o}. We set

ˆb(ki) =

1 |Go(i)| j∈G

o(i)

rk(j).

In our main experiments,

o ∈ {correct,incorrect},

yielding a simple two-bin baseline. This formulation allows for a very broad range of applications which we cover in Sec 6.1.

See Appendix B for a discussion of when OCB is unbiased.

- 5. Experiments and Results

In this section, we ablate components of the proposed approach and evaluate our methods against baselines.

##### 5.1. How Accurate Are Conditional Baselines?

The baseline designs described in Section 4.3 naturally raise the question of how close they are to MC estimates.

To investigate this, we focus on the second block in our structured generation, whose reward depends on the sampled solution prefix. We start from a strong, well-calibrated reference policy obtained with RLCR for Qwen2.5-7B (Qwen et al., 2025), and collect trajectories of the form: y = [Xsol;Xconf]. For each prompt x and sampled solution block Xsol(i), the calibration reward (a Brier score) is denoted

rcal(i). The ideal advantage for the confidence block subtracts the intermediate-state value

A(MCi) = rcal(i) − E rcal | x,Xsol(i) ,

which depends on the particular sampled solution prefix. To approximate this quantity, we select random 100 prompts from MATH500 (Hendrycks et al., 2021) and perform 32 rollouts for the confidence block from each fixed prefix

(x,Xsol(i)), producing a reference estimate of A(MCi) . We then compute proxy-advantages Aˆ(i) using the estimators from Sec 4.3.

Figure 1 compares our baselines to the Monte-Carlo reference. To quantify fidelity, we measure the RMSE between

Aˆ(i) and A(MCi) , split by whether the solution is correct. The Outcome-Conditioned Baseline achieves the lowest error in

both strata and shows unbiased estimation.

##### 5.2. Main Results

We now evaluate Blockwise Advantage Estimation on joint reasoning and uncertainty estimation for math across Qwen2.5-7B-Base, Qwen2.5-7B-Instruct, and Qwen2.53B-Instruct.

GRPO training setup. Unless otherwise specified, we use the same GRPO backbone across methods, with following implementation choices:

- • No advantage standardization. We center advantages but do not divide by the within-group standard deviation, following common GRPO variants such as Dr. GRPO (Liu et al., 2025);
- • Dynamic entropy regularization. We adapt the entropy coefficient over training to avoid premature entropy collapse (He et al., 2025);
- • No KL regularization. We rely on conservative PPO clipping and entropy control for stability and set β = 0 in Eq. (1), following Dr.GRPO.

The models are trained using a mixture of MATH and DAPO training datasets having in total 25k training prompts (Feng et al., 2025b). All methods use the same prompts, verifier, hyperparameters and parsing logic for extracting answers and confidences. All prompts can be found in Appendix A and hyperparameters – in Table 4. We report Pass@1 accuracy and confidence quality using AUROC, ECE, and Brier Score. We report standard errors of the mean (SEM) for all evaluation metrics; details on the SEM computation procedure are provided in Appendix I and in Table 7.

Tables 1, 2, 6 summarize the main outcomes on MATH500 (in-distribution), GSM8K (Cobbe et al., 2021) (easy OOD),

[Figure 1]

- Figure 1. The error distribution comparing to true advantage MC estimates for three methods: Batch Mean, Group Mean, OCB w.r.t. correct/incorrect solutions. Batch Mean shows skewed approximation unlike Group Mean, while OCB demonstrates the lowest RMSE error across all groups.

Table 1. Effect of conditional-baseline approximations for the confidence block for Qwen2.5-3B-Instruct trained for 1024 steps, evaluated on MATH500

Method Acc (%) AUROC ECE Brier RLCR 68.7 0.881 0.059 0.115 Group Mean 66.4 0.878 0.069 0.124 OCB 67.9 0.893 0.030 0.110 Batch Mean 66.2 0.858 0.078 0.128 None (raw reward) 66.1 0.882 0.099 0.125

and AIME23–25 (Mathematical Association of America) (hard OOD). Across model sizes and base/instruct variants, we observe three consistent trends.

- • OCB is the strongest compute-free conditional baseline. On Qwen2.5-3B-Instruct, OCB yields a large calibration gain on MATH500, improving ECE (0.030 vs. 0.059 for RLCR) as well as AUROC/Brier, with only a modest accuracy gap. On Qwen2.5-7B-Base and Qwen2.5-7B-Instruct, OCB matches RLCR.
- • Unconditioned means can be brittle. The Group Mean baseline can reduce ECE (e.g., MATH500 for 7B-Base: 0.020 vs. 0.032 for OCB), but may degrade accuracy and proper scoring (Brier). Under distribution shift, it can fail severely (e.g., GSM8K for 7BInstruct: ECE 0.125, Brier 0.146, AUROC 0.641; Table 6). Batch Mean and no-baseline variants are consistently worse on the 3B setting (Table 1), aligning with our advantage-quality analysis described in Section 5.

Table 2. Main results for Qwen2.5-7B-Base trained for 512 steps on self-confidence estimation for math.

Method Acc (%) AUROC ECE Brier MATH500 (in-distribution)

RLCR 75.0 0.914 0.043 0.093 OCB 75.1 0.917 0.032 0.094 Group Mean 72.7 0.912 0.020 0.099

###### GSM8K (easy OOD)

RLCR 88.9 0.835 0.037 0.075 OCB 89.7 0.847 0.039 0.073 Group Mean 87.8 0.849 0.030 0.079

###### AIME23–25 (hard OOD)

RLCR 9.4 0.942 0.092 0.052 OCB 9.5 0.902 0.126 0.064 Group Mean 8.2 0.911 0.094 0.056

• RLCR remains a strong baseline, while BAE+OCB is competitive without reward scalarization. RLCR is often best on accuracy and/or hard-OOD discrimination (e.g., AUROC on AIME23–25 for 7B-Base; Table 2), but BAE+OCB remains competitive and often yields a better accuracy–calibration trade-off on in-domain and easy OOD regimes, while avoiding reward scalarization across segments.

##### 5.3. Test-Time Scaling Evaluation

A practical benefit of self-confidence estimation is test-time scaling (TTS): at inference time we can sample multiple candidate solutions and use the model’s reported confidence

to select a final answer. This can significantly improve accuracy without any additional training.

For each prompt x, we sample n independent completions {y(i)}ni=1 ∼ πθ(· | x) at fixed decoding settings, specified in Appendix Section D. From each completion we extract (i) a final answer aˆ(i) and (ii) a reported confidence q(i) ∈ [0,1]. We then aggregate the n candidates into a single prediction aˆ using the following strategies: Pass@1, Pass@k, Majority Vote (Maj), Max confidence selection (MaxConf), Confidence-weighted majority (WeightedMaj). WeightedMaj is defined by the following selection rule:

aˆwmaj = arg max

a

n

q(i) 1{aˆ(i) = a}.

i=1

Intuitively, Maj exploits consensus across samples, while MaxConf exploits self-reported uncertainty. Confidenceweighted majority combines both: answers that appear repeatedly and with consistently high confidence receive the largest total weight. When q(i) is well calibrated, these rules become principled: q(i) approximates the probability that a candidate is correct, so higher-confidence candidates should be preferred, and confidence-weighting acts as a natural soft aggregation.

[Figure 2]

- Figure 2. TTS evaluation for RLCR and OCB methods across three datasets: (a) – MATH500, (b) – GSM8K, (c) – AIME23–25

- Figure 2 shows test-time scaling curves. Across datasets, increasing the number of samples improves accuracy for realizable selection rules, while Pass@k indicates substantial remaining headroom from generating diverse candidates. Crucially, OCB preserves the key downstream benefit of calibrated confidence. Performance tracks closely with RLCR, indicating that our segment-level credit assignment does not merely improve calibration metrics in isolation, but produces actionable confidence estimates that translate into better inference-time decision making. We also notice a

Table 3. Qwen2.5-7B-Base trained BAE with Brier score and BCE as reward for the confidence block

Method Acc (%) AUROC ECE Brier MATH500 (in-distribution)

OCB (Brier, 512 st.) 75.1 0.92 0.03 0.09 OCB (BCE, 512 st.) 69.7 0.92 0.03 0.10 OCB (BCE, 1024 st.) 76.7 0.93 0.03 0.09

###### GSM8K (easy OOD)

OCB (Brier, 512 st.) 89.7 0.85 0.04 0.07 OCB (BCE, 512 st.) 86.9 0.84 0.02 0.08 OCB (BCE, 1024 st.) 90.2 0.86 0.02 0.07

###### AIME23–25 (hard OOD)

OCB (Brier, 512 st.) 9.5 0.90 0.13 0.06 OCB (BCE, 512 st.) 8.1 0.87 0.08 0.06 OCB (BCE, 1024 st.) 10.4 0.92 0.09 0.06

consistent empirical pattern how accuracy and calibration metrics contribute to overall TTS performance, the details and discussion of which can be found in Appendix G.

##### 5.4. Alternative Rewards for Confidence

So far, our confidence block has been trained with a Brierstyle reward:

rBrier(q,c) = −(q − c)2.

A key advantage of Blockwise Advantage Decomposition is that it does not depend on any particular scalarization of objectives; as long as the confidence objective is computable from (q,c), it can be optimized locally on the confidence block.

To demonstrate this, we also consider the Bernoulli loglikelihood, another strictly proper scoring rule:

rBCE(q,c) = clog q + (1 − c)log(1 − q) , with q clipped to [ϵ,1 − ϵ] for numerical stability.

A naive adaptation of RLCR replaces the Brier term inside a single scalar reward with BCE. In our experiments, this frequently converges to a degenerate local optimum in which the model avoids the calibration penalty by emitting extremely low confidence and refusing to answer, rather than improving solution quality. The example of such behavior can be found in Appendix F.

In contrast, training OCB using BCE for a confidence block gives solid results. Empirically, we find that training the confidence block with BCE converges more slowly than Brier in terms of accuracy, but reaches strong performance once trained for sufficiently long. The corresponding evaluation results are demonstrated in Table 3.

##### 5.5. Beyond Self-Confidence

Our framework is not specific to uncertainty estimation. To illustrate broader applicability, we consider a simple sequential refinement setting in which the model produces two

consecutive solution attempts for the same math problem. This mirrors common agentic workflows with reflection and subsequent adjustments, but in the simplest possible form with a fixed number of refinement steps.

Given a prompt x, the model generates two attempts in a single completion y = [X1; X2], where X1 contains a first reasoning trace and answer, and X2 contains a reflection followed by a second answer. We refer to this setup as Two-Attempt Refinement.2 The exact prompt for this task is provided in Appendix A. The reward structure for each attempt segment is described in Appendix C.

To illustrate the applicability of our method, we train Qwen2.5-3B-Instruct under the BAE+OCB scheme. Figure 3 shows that the second attempt consistently improves over the first attempt across all sampling budgets.

outcome-conditioned grouping provides a simple and effective variance-reduction mechanism for later-block advantages without requiring additional rollouts.

##### 6.2. Limitations

Outcome-conditioned baselines require populated strata. The Outcome-Conditioned Baseline relies on estimating subgroup means within each GRPO group (e.g., correct vs. incorrect). When one stratum is rare, subgroup sizes can become small, increasing variance and potentially degrading calibration metrics. In practice, this suggests that OCB benefits from either sufficiently large groups, sufficiently high diversity of the model being trained, or additional smoothing or shrinkage toward the unconditioned Group Mean. We study how the number of completions influences OCB performance in Appendix H.

[Figure 3]

- Figure 3. TTS evaluation within Two-Attempt Refinement on MATH500 dataset. Second attempt demonstrates higher performance across all aggregations.

### 6. Discussion, Limitations and Future Work

##### 6.1. Discussion

The motivation behind OCB is that this baseline imposes an equivalence relation on prefixes that share the same taskrelevant outcome and can be used for efficient approximation. When the block-k reward depends primarily on such an outcome (as in uncertainty estimation, where calibration is chiefly determined by whether the preceding solution is correct), conditioning the baseline on o yields a closer approximation to the true conditional value E[rk | x,X<k] while retaining the compute benefits of group-based normalization. More broadly, many objectives only care about what was achieved rather than how it was achieved: for instance, in code generation there may be many distinct patches that fix a bug, but downstream evaluation typically only depends on whether the tests pass. In these settings,

2We avoid the term “two-shot” to prevent confusion with fewshot prompting.

Segment boundaries are assumed known and stable. BAE assumes that block boundaries are well defined (e.g., solution vs. confidence report), so that “which tokens control which objective” is unambiguous. For tasks with fuzzy boundaries, defining segments may require additional design choices. Incorrect segmentation could reintroduce crossobjective interference.

##### 6.3. Future Work

Richer conditioning and broader evaluation. Beyond binary correctness, one can define multi-bin outcomes (e.g., verifier margin, intermediate goal achievement, tool feedback) or learn outcome clusters, yielding a higher-fidelity conditional baseline without additional rollouts; evaluating such variants beyond verifiable rewards is a natural next step.

Overall, we view our method as a step toward more modular and scalable multi-objective RL for LLMs: replacing handtuned reward scalarization with structured credit assignment aligned to how models actually generate text.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, specifically improving credit assignment in multi-objective reinforcement learning tasks. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

### References

Chen, G., Li, Y., Jiang, Y., Qian, C., Ren, Q., Yang, J., Cheng, Y., Liu, D., and Shao, J. Conditional advantage estimation for reinforcement learning in large reasoning models, 2025. URL https://arxiv.org/abs/

2509.23962.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems, 2021. URL https://arxiv.

org/abs/2110.14168.

Damani, M., Puri, I., Slocum, S., Shenfeld, I., Choshen, L., Kim, Y., and Andreas, J. Beyond binary rewards: Training lms to reason about their uncertainty, 2025. URL https://arxiv.org/abs/2507.16806.

Fan, L., Zhang, Y., Chen, M., and Liu, Z. Posterior-grpo: Rewarding reasoning processes in code generation, 2025. URL https://arxiv.org/abs/2508.05170.

- Feng, X., Veeriah, V., Chiam, M., Dennis, M., Pachauri, R., Tumiel, T., Barbero, F., Obando-Ceron, J., Shi, J., Singh, S., Hou, S., Tomaˇsev, N., and Zahavy, T. Generating creative chess puzzles, 2025a. URL https://arxiv. org/abs/2510.23881.
- Feng, Y., Jain, P., Hartshorn, A., Duan, Y., and Kempe, J. Don’t waste mistakes: Leveraging negative rl-groups via confidence reweighting, 2025b. URL https:// arxiv.org/abs/2510.08696.

Guo, Y., Xu, L., Liu, J., Ye, D., and Qiu, S. Segment policy optimization: Effective segment-level credit assignment in rl for large language models, 2025. URL https: //arxiv.org/abs/2505.23564.

He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P., Zhang, X., Zhang, F., Xu, J., Shen, W., Li, S., Zeng, L., Wei, T., Cheng, C., An, B., Liu, Y., and Zhou, Y. Skywork open reasoner 1 technical report, 2025. URL https://arxiv.org/abs/2505.22312.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Kazemnejad, A., Aghajohari, M., Portelance, E., Sordoni, A., Reddy, S., Courville, A., and Roux, N. L. Vineppo: Refining credit assignment in rl training of llms, 2025. URL https://arxiv.org/abs/2410.01679.

Khatri, D., Madaan, L., Tiwari, R., Bansal, R., Duvvuri,

- S. S., Zaheer, M., Dhillon, I. S., Brandfonbrener, D., and Agarwal, R. The art of scaling reinforcement learning compute for llms, 2025. URL https://arxiv.org/ abs/2510.13786.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding r1-zero-like training: A critical perspective, 2025. URL https://arxiv. org/abs/2503.20783.

Mathematical Association of America. American invitational mathematics examination (aime). MAA American Mathematics Competitions. URL https://maa. org/maa-invitational-competitions/.

MiniMax, :, Chen, A., Li, A., Gong, B., Jiang, B., Fei, B., Yang, B., Shan, B., Yu, C., Wang, C., Zhu, C., Xiao, C., Du, C., Zhang, C., Qiao, C., Zhang, C., Du, C., Guo, C., Chen, D., Ding, D., Sun, D., Li, D., Jiao, E., Zhou, H., Zhang, H., Ding, H., Sun, H., Feng, H., Cai, H., Zhu, H., Sun, J., Zhuang, J., Cai, J., Song, J., Zhu, J., Li, J., Tian, J., Liu, J., Xu, J., Yan, J., Liu, J., He, J., Feng, K., Yang, K., Xiao, K., Han, L., Wang, L., Yu, L., Feng, L., Li, L., Zheng, L., Du, L., Yang, L., Zeng, L., Yu, M., Tao, M., Chi, M., Zhang, M., Lin, M., Hu, N., Di, N., Gao, P., Li, P., Zhao, P., Ren, Q., Xu, Q., Li, Q., Wang, Q., Tian, R., Leng, R., Chen, S., Chen, S., Shi, S., Weng, S., Guan, S., Yu, S., Li, S., Zhu, S., Li, T., Cai, T., Liang, T., Cheng, W., Kong, W., Li, W., Chen, X., Song, X., Luo,

- X., Su, X., Li, X., Han, X., Hou, X., Lu, X., Zou, X., Shen, X., Gong, Y., Ma, Y., Wang, Y., Shi, Y., Zhong,
- Y., Duan, Y., Fu, Y., Hu, Y., Gao, Y., Fan, Y., Yang, Y., Li, Y., Hu, Y., Huang, Y., Li, Y., Xu, Y., Mao, Y., Shi,

- Y., Wenren, Y., Li, Z., Li, Z., Tian, Z., Zhu, Z., Fan, Z., Wu, Z., Xu, Z., Yu, Z., Lyu, Z., Jiang, Z., Gao, Z., Wu,
- Z., Song, Z., and Sun, Z. Minimax-m1: Scaling test-time compute efficiently with lightning attention, 2025. URL https://arxiv.org/abs/2506.13585.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback, 2022. URL https:

//arxiv.org/abs/2203.02155.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025. URL https:

//arxiv.org/abs/2412.15115.

Shao, R., Asai, A., Shen, S. Z., Ivison, H., Kishore, V., Zhuo, J., Zhao, X., Park, M., Finlayson, S. G., Sontag, D., Murray, T., Min, S., Dasigi, P., Soldaini, L., Brahman, F., tau Yih, W., Wu, T., Zettlemoyer, L., Kim, Y., Hajishirzi, H., and Koh, P. W. Dr tulu: Reinforcement learning with evolving rubrics for deep research, 2025. URL https:

//arxiv.org/abs/2511.19399. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X.,

Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Wang, X., Liu, B., Jiang, S., Liu, J., Qi, J., Chen, X., and He, B. From solving to verifying: A unified objective for robust reasoning in llms, 2025. URL https://arxiv. org/abs/2511.15137.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M., Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang, C., Yu, H., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W.Y., Zhang, Y.-Q., Yan, L., Qiao, M., Wu, Y., and Wang, M. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/ abs/2503.14476.

Zheng, C., Liu, S., Li, M., Chen, X.-H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., Zhou, J., and Lin, J. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.

Zhu, M., Chen, X., Yu, B., Zhao, H., and Jia, J. Stratified grpo: Handling structural heterogeneity in reinforcement learning of llm search agents, 2025. URL https:// arxiv.org/abs/2510.06214.

- A. System Prompts All self-confidence experiments on Math were run with the following prompt.

## Self-Confidence System Prompt

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and provides the user with its confidence level. The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags. The final answer is enclosed between <answer> </answer> tags. The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags. The Assistant should reason about its confidence in the solution and its uncertainty in the solution within these tags. The final format that must be followed is: <reasoning> reasoning process here </reasoning><answer> final answer here </answer><analysis> analysis about confidence and uncertainty here </analysis><confidence> confidence level here (number between 0 and 1) </confidence>

EXAMPLES Q: Who was awarded the Oceanography Society’s Jerlov Award in 2018?

<reasoning> The question asks who was awarded the Oceanography Society’s Jerlov Award in 2018. The Jerlov Award is given by the Oceanography Society, and it is named after the late Professor Gunnar Jerlov, a pioneer in the field of ocean optics. To find the winner of the 2018 award, I would need to look up the most recent information on the Oceanography Society’s website or other reliable sources.</reasoning>

<answer> Dr. David W. Schiedek </answer>

<analysis> The Oceanography Society’s website or other reliable sources do not provide information on the winner of the 2018 Jerlov Award. It is possible that the information is not readily available or that the award was not given in 2018. Additionally, the name ”Dr. David W. Schiedek” is not a well-known figure in the field of oceanography, which raises uncertainty about the accuracy of this answer. </analysis>

<confidence> 0.25 </confidence> END OF EXAMPLES

The Two-Attempt Refinement experiment was run with the following system prompt.

## Two-Attempt Refinement

A conversation between User and Assistant. The user asks a question, and the Assistant solves it using multiple attempts. For each attempt, the Assistant thinks about the reasoning process and provides an answer. Each attempt consists of reasoning enclosed in <reasoning> </reasoning> tags followed by an answer enclosed in <answer> </answer> tags. After the first attempt, the Assistant reflects on the previous solution and may

provide an improved answer. The format for 2 attempts is: <reasoning> first attempt reasoning </reasoning> <answer> first attempt answer </answer> <reasoning> reflection and second attempt reasoning </reasoning> <answer> second attempt answer (same or corrected) </answer> EXAMPLES Q: What is 2 + 2? <reasoning> This is a simple arithmetic problem. 2 + 2 equals 4. </reasoning> <answer> 4 </answer> <reasoning> Let me verify: 2 + 2 = 4. This is correct. </reasoning> <answer> 4 </answer> Q: Who was awarded the Oceanography Society’s Jerlov Award in 2018?

<reasoning> The question asks who was awarded the Oceanography Society’s Jerlov Award in 2018. The Jerlov Award is given by the Oceanography Society, and it is named after the late Professor Gunnar Jerlov, a pioneer in the field of ocean optics. To find the winner of the 2018 award, I would need to look up the most recent information on the Oceanography Society’s website or other reliable sources. </reasoning>

<answer> Dr. David W. Schiedek </answer>

<reasoning> Let me reconsider. The Oceanography Society’s website or other reliable sources do not provide information on the winner of the 2018 Jerlov Award under the name Dr. David W. Schiedek. However, there is information that the award was given to Dr. Annick Bricaud in 2018, which sounds more plausible given her contributions to ocean optics. </reasoning>

<answer> Dr. Annick Bricaud </answer> END OF EXAMPLES

Table 4. Canonical configuration shared by all runs (YAML-like rendering for readability).

|# Common settings (all GRPO-like runs) sequence_length: 4096 temperature: 1.0 batch_size: 2048 group_size: 32 # unless stated otherwise<br><br>optimizer: AdamW<br><br>adam_beta1: 0.9<br><br>adam_beta2: 0.999 weight_decay: 0.1 lr: 1e-6 warmup: linear(64 steps)<br><br><br># Dynamic entropy controller target_entropy: 0.2 eta_delta: 5e-4<br><br>kl_beta: 0.0 # we don’t use KL regularization weights_dtype: bfloat16<br><br>|
|---|

- B. Additional Method Details Algorithm 1 BAE with outcome-conditioned baselines (for one prompt x)

- 1: Sample a GRPO group {y(i)}Gi=1 ∼ πθ

old

(· | x) and parse each completion into blocks y(i) = [X1(i);...;XK(i)].

- 2: Compute per-block rewards {rk(i)} and intermediate outcomes o(ki) = ok(x,X<k(i)) for k > 1.
- 3: for k = 1 to K do
- 4: if k = 1 then
- 5: Set baseline ˆb(1i) ← G1 Gj=1 r1(j).

- 6: else
- 7: For each outcome value o, compute stratum mean µk,o ← |G1

o| j∈Go rk(j) where Go = {j : o(kj) = o}.

- 8: Set ˆb(ki) ← µk,o(i)

k

.

- 9: end if
- 10: Form proxy-advantage Aˆ(ki) ← rk(i) − ˆb(ki).
- 11: end for
- 12: Update θ by optimizing the blockwise clipped objective, applying Aˆ(ki) only to tokens in block k.

When is OCB unbiased? For a fixed prompt x and block k, suppose there exists a discrete outcome variable o = o(x,X<k) such that

E[rk | x,X<k] = E[rk | x,o].

Equivalently, conditioning on o captures all dependence of rk on the sampled prefix X<k relevant for the conditional value. In this case, the within-stratum mean used by OCB is an unbiased estimator of E[rk | x,X<k] (up to finite-sample noise). When this assumption is violated, OCB introduces bias; empirically, it is most effective when rk depends primarily on a coarse task-relevant outcome (e.g., correctness) rather than on fine-grained properties of the prefix.

- C. Reward Structure for 2 Attempts Experiment

Each attempt yields a verifiable final answer aˆ1 and aˆ2, scored by the same correctness verifier, and a⋆ is the ground-truth answer. Let c1 and c2 be the correctness of the corresponding attempts. In this experiment, we use a shaped, attempt-specific reward scheme:

###### r1 = c1,

and for the second attempt,

 

1, if c2 = 1, 0.1, if c2 = 0 and aˆ2 ̸≡ aˆ1, 0, if c2 = 0 and aˆ2 ≡ aˆ1.

r2 =



Intuitively, r2 assigns full credit for solving the problem, but when the model remains incorrect it mildly rewards trying a different answer rather than repeating the same mistake. This discourages degenerate behavior where the second attempt simply restates the first attempt verbatim, while keeping correctness as the dominant learning signal.

### D. Evaluation Hyperparameters

Inference protocol. We evaluate each model by sampling n = 16 independent completions per prompt with temperature T = 1.0 using vLLM v0.10.0. We allow long generations with a maximum generation length of 32k tokens to avoid truncation on difficult instances.

We evaluate reasoning accuracy by extracting a final answer from the model output and checking it with a math verifier math-verify. After extraction we put each answer in \boxed{...}. Empirically, enforcing \boxed{} substantially reduces parsing failures.

Calibration metrics and benchmarks. For Expected Calibration Error (ECE), we use fixed-width bins

[0.0,0.1),[0.1,0.2),...,[0.9,1.0),[1.0,1.1), i.e., 11 bins total.

[Figure 4]

Figure 4. Expected Calibration Error (ECE) computed with fixed-width bins across three datasets: (a)–MATH500, (b)–GSM8K, (c)– AIME23–25

### E. Completion Example with Reasoning and Self-Analysis

To illustrate the format of generations in our task, we present a simple completion from Qwen2.5-7B-Base before any training:

## Completion Example

<reasoning> To find the length of segment BN, I can use the properties of similar triangles. Since MN is parallel to AB, triangles CAN and CBM are similar, where side AC corresponds to side BC and side AN corresponds to side BM. Using the property of similar triangles, we have the proportion: AC/BC = AN/BM

We know that AC is 5 units long, BC is 2.4 units long, and AN is 3 units long. We can find the length of BM by using the fact that AC + AN = AP (where AP is the sum of the lengths of segments AC and AN). So, BM = AP - AM - MC. Knowing these values, I can substitute them into the proportion and solve for the length of BM.</reasoning>

<answer>11.5</answer>

<analysis>Given that MN is parallel to AB, triangles CAN and CBM are similar, and the proportions can be set up with the given lengths. Substituting the values and solving for BM gives the required length of BN. However, since I used approximate values for AC and BM, there may be some uncertainty in the final answer, and thus the confidence in the solution is relatively low.</analysis>

<confidence>0.6</confidence>

- F. GRPO Calibration Trained with BCE Reward Here we illustrate that a naive reward scheme for standard GRPO,

R = rRLVR + rBCE,

can lead to a local optimum where the model learns to refuse to answer while minimizing its confidence. This behavior can be interpreted as a form of reward hacking. Mitigating it would require hyperparameter tuning or the introduction of additional terms into the reward.

## RLVR + BCE Reward

Let a and b be positive integers satisfying 3a < b and a2 + ab + b2 = (b + 3)2 + 27. Find the minimum possible value of a + b.

Remember to put your answer on its own line after “Answer:”.

<reasoning> I am not familiar with the equation presented in the question, involving positive integers a and b. I know what positive integers are, but I do not recognize this specific equation or how to solve it. Additionally, I am unsure about the context of the question, as it involves a symbol (∃) and a square that looks like an algebraic expression, but it is not clear to me what it means. </reasoning>

<answer> Unable to determine </answer>

<analysis> This question involves a complex equation with a symbol and a square that I am unfamiliar with. There are no guidelines or previous examples to follow, and the equation does not resemble any standard mathematical format that I am aware of. The lack of context and familiarity with the equation make it impossible to provide an accurate answer or even a plausible guess. </analysis>

<confidence> 0.001 </confidence>

In contrast, OCB with BCE optimizing the confidence block shows no signs of reward hacking.

## OCB with BCE Reward For Confidence Block

In the circle with center Q, radii AQ and BQ form a right angle. The two smaller regions are tangent semicircles, as shown. The radius of the circle with center Q is 14 inches. What is the radius of the smaller semicircle? Express your answer as a common fraction.

<reasoning> To find the radius of the smaller semicircles in the given geometric configuration, we need to follow these steps:

##### 1. Understand the Initial Setup:

- - There’s a circle centered at Q with radius 14 inches.
- - Radii AQ and BQ form a right angle.
- - Two smaller regions are tangent semicircles, as shown. 2. Visualize the Problem:
- - The circle with center Q provides a structural guide for the configuration.
- - Radii AQ and BQ are perpendicular to each other and hence form a right angle at Q.
- - The two semicircles are tangent to each other and tangential to the larger circle’s boundary. 3. Establish Relationships and Names:
- - Let the radius of the larger circle be R = 14 inches.
- - Let the radius of each of the smaller semicircles be r. 4. Analyze the Geometry:
- - The right angle formed by AQ and BQ divides the larger circle into four sectors.
- - Each smaller semicircle is tangent to the larger circle and each other at points along the radius of the larger circle. 5. Use Geometric Properties:
- - The combined diameter of the two smaller semicircles is equal to the diameter of the larger circle minus the diameter of each semicircle.
- - The part of the radius of Q occupied by the semicircles traces the perimeter of the square inscribed within the circle. 6. Set Up the Equations:
- - The diagonal of the inscribed square is the diameter of the circle.
- - The diameter of the circle is 2R = 2 × 14 = 28 inches.
- - For an inscribed square within the circle, the length of one side of the square can be denoted as a. The diagonal of the square forms a right triangle with two sides of length a each:

a

√

2 = 2R =⇒ a

√

2 = 28 =⇒ a =

28 √2

= 14

√

2.

- - The area of the square is a2 = (14√2)2 = 392 square inches. 7. Apply Tangency Conditions:
- - The tangency condition requires the semicircles to align appropriately within the larger circle.
- - Considering the sum of the diameters of the two semicircles along with the right angle setup mentioned, the combined diameter of the two smaller semicircles equals the remaining radius of the larger circle after accounting for the diameter of the square portion. 8. Calculate the Radius of the Smaller Semicircles:
- - Each semicircle with radius r has a diameter of 2r.
- - The combined diameter of the two semicircles is 2r + 2r = 4r.
- - This combined diameter needs to fit into the space remaining within the circle as its larger circle portion:

√

4r = 14 − (One side of square per radius) = 14 + 14 − 14 = 14

2 − 14 =⇒ 4r = 14.

- - Solving for r:

14 4

7 2

r =

=

.

Therefore, the radius of the smaller semicircle is 72 inches. </reasoning> <answer> 72 </answer> <analysis> First, let’s verify and refine the geometric configuration and relationships we have worked with.

- 1. Circle and Radii Setup: We have a circle centered at Q with radius R = 14 inches. Radii AQ and BQ are perpendicular, splitting the circle into four quadrants.
- 2. Semicircles Positioning: The smaller semicircles are tangent to each other and tangent to the larger circle’s boundary. However, the radius ratio and placements need clear geometric consideration.
- 3. Geometry Revisit: Let the radius of the smaller semicircles be r. The two tangent semicircles form a width equal to their combined diameters along AQ and BQ. The radius R of the larger circle is composed of segments r + r along with another segment equal to r in one direction.
- 4. Right Triangle Insight: Given that the two semicircles form a right angle at Q, and the semicircles’ radius sum along with the square inscribed within provides the key geometric property:

- - The diameters 2r of each semicircle sum up to 2r + 2r = 4r.
- - The diagonal of the square inscribed in the circle of radius 14 inches is 2R = 28 inches, forming the right triangle with a√2 = 28, giving a = 14√2. The area calculation as a2 = 392 square inches in this context adds insignificant value for finding the smaller semicircles’ radii.

- 5. Mistakes and Clarifications: The linking of squared terms and square side lengths probably needs better connection. The right triangle and square properties were considered but not correctly tied to the radius r.

Taking a simpler approach, realizing the relationship directly between the inscribed squares and semicircles, let’s re-center:

- - The diameters of the semicircles and their tangency precisely dictate the relationship with the large circle.
- - Confirmation through an initial insight that 2r equates to R, r =

###### R 3

. Finally, upon thorough review and correction of the geometric relationships and calculations:

14 3

r =

.

</analysis> <confidence> 0.15 </confidence>

### G. Accuracy/Calibration Contribution into TTS Performance

When calibration is “good enough”, accuracy dominates TTS. A consistent empirical pattern is that once a model is already reasonably calibrated (e.g., ECE on the order of a few percent), marginal improvements in calibration metrics do not necessarily translate into better test-time scaling unless they also improve (or at least preserve) solution accuracy and sample quality. For example, in our 7B-Base experiments (Table 2), the group-mean baseline achieves lower ECE than OCB on MATH500, yet it yields worse TTS accuracy under confidence-weighted ensembling because its Pass@1 accuracy is

Table 5. Effect of GRPO group size G for BAE with Outcome-Conditioned Baselines (OCB) on Qwen2.5-7B-Base (512 steps). Larger groups slightly reduce ECE but can trade off against accuracy and discrimination (AUROC).

Setting Acc (%) AUROC ECE Brier MATH500 (in-distribution)

OCB (G=32) 75.1 0.92 0.03 0.09 OCB (G=64) 75.1 0.90 0.03 0.09

GSM8K (easy OOD)

OCB (G=32) 89.7 0.85 0.04 0.07 OCB (G=64) 88.4 0.81 0.02 0.08

AIME23–25 (hard OOD)

OCB (G=32) 9.5 0.90 0.13 0.06 OCB (G=64) 8.6 0.89 0.12 0.07

Table 6. Main results for Qwen2.5-7B-Instruct trained for 512 steps on self-confidence estimation for math.

Method Acc (%) AUROC ECE Brier MATH500 (in-distribution)

RLCR 78.6 0.922 0.031 0.086 OCB 77.3 0.929 0.040 0.087 Group Mean 76.6 0.893 0.032 0.098

###### GSM8K (easy OOD)

RLCR 89.8 0.760 0.032 0.081 OCB 89.1 0.827 0.032 0.076 Group Mean 82.4 0.641 0.125 0.146

###### AIME23–25 (hard OOD)

RLCR 11.9 0.932 0.132 0.072 OCB 12.9 0.907 0.096 0.074 Group Mean 12.1 0.937 0.094 0.058

substantially lower. This suggests that in the “well-calibrated” regime, the dominant driver of TTS is the quality/diversity of sampled solutions (accuracy and Pass@k headroom), while confidence primarily serves as a tie-breaker and robustness mechanism.

### H. Group Size Ablation for OCB Method

Outcome-Conditioned Baselines estimate the conditional value for later blocks by averaging rewards within outcome-defined subgroups of a GRPO group (e.g., correct vs. incorrect solutions). This estimator therefore depends on the group size G: larger groups typically provide more samples per outcome class, reducing variance of the subgroup mean but increasing sampling cost.

We study the effect of group size on OCB using Qwen2.5-7B-Base trained for 512 steps with the same batch size of 2048 completions. All settings are identical except for the GRPO group size, comparing G = 32 and G = 64.

Table 5 shows that increasing the group size from 32 to 64 yields slightly improved calibration (lower ECE) on all three benchmarks, while causing a reduction in accuracy and AUROC in our runs. We attribute this to a twofold lower number of distinct examples seen during training.

#### I. Full Results with Standard Errors Table 7 reports evaluation metrics in mean±SEM format.

Uncertainty estimation for evaluation metrics. We report standard errors of the mean (SEM) for all evaluation metrics to quantify estimation uncertainty. For Pass@1, we compute SEM directly from the empirical distribution of metric values across sampled combinations. the SEM is computed as σ/√n, where σ is the sample standard deviation and n is the total number of combination-level evaluations. For calibration metrics (Expected Calibration Error, Brier score, and AUROC),

Table 7. Evaluation metrics with standard errors (mean±SEM). Model Dataset Method Acc (%) AUROC ECE Brier

RLCR 68.65±1.04 0.8814±0.0085 0.0591±0.0071 0.1147±0.0046 Group Mean 66.35±1.06 0.8781±0.0086 0.0688±0.0074 0.1241±0.0056 OCB 67.90±1.04 0.8927±0.0081 0.0297±0.0064 0.1104±0.0046 Batch Mean 66.15±1.06 0.8584±0.0090 0.0775±0.0074 0.1283±0.0044 None (raw reward) 66.10±1.06 0.8818±0.0083 0.0990±0.0076 0.1248±0.0061

MATH500 (in-dist.)

Qwen2.5-3B Instruct

RLCR 74.99±0.48 0.9140±0.0037 0.0426±0.0033 0.0932±0.0021 OCB 75.10±0.48 0.9169±0.0035 0.0322±0.0030 0.0935±0.0020 Group Mean 72.71±0.50 0.9117±0.0037 0.0199±0.0032 0.0991±0.0021

MATH500 (in-dist.)

RLCR 88.86±0.22 0.8345±0.0047 0.0368±0.0018 0.0753±0.0013 OCB 89.70±0.21 0.8473±0.0046 0.0394±0.0019 0.0729±0.0012 Group Mean 87.77±0.23 0.8487±0.0044 0.0297±0.0019 0.0793±0.0013

Qwen2.5-7B Base

GSM8K (easy OOD)

RLCR 9.41±0.77 0.9416±0.0115 0.0920±0.0053 0.0518±0.0031 OCB 9.48±0.78 0.9018±0.0180 0.1263±0.0055 0.0639±0.0035 Group Mean 8.22±0.73 0.9105±0.0141 0.0941±0.0055 0.0563±0.0033

AIME23–25 (hard OOD)

RLCR 78.55±0.46 0.9215±0.0036 0.0305±0.0030 0.0864±0.0022 OCB 77.30±0.47 0.9293±0.0031 0.0395±0.0031 0.0867±0.0023 Group Mean 76.59±0.47 0.8926±0.0043 0.0323±0.0031 0.0979±0.0024

MATH500 (in-dist.)

RLCR 89.78±0.21 0.7595±0.0055 0.0318±0.0019 0.0813±0.0016 OCB 89.11±0.21 0.8269±0.0050 0.0315±0.0019 0.0759±0.0016 Group Mean 82.43±0.52 0.6413±0.0084 0.1247±0.0047 0.1459±0.0043

Qwen2.5-7B Instruct

GSM8K (easy OOD)

RLCR 11.94±0.86 0.9320±0.0103 0.1319±0.0061 0.0718±0.0035 OCB 12.85±0.89 0.9073±0.0124 0.0957±0.0065 0.0739±0.0042 Group Mean 12.08±1.73 0.9365±0.0219 0.0936±0.0110 0.0581±0.0067

AIME23–25 (hard OOD)

which are computed globally across all predictions, we employ bootstrap resampling with 1,000 iterations. In each bootstrap iteration, we resample the (q,c) pairs (confidence, correctness) with replacement and recompute each calibration metric; the SEM is estimated as the standard deviation of the resulting bootstrap distribution.

