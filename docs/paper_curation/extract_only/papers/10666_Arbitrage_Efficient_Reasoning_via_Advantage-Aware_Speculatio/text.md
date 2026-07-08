## arXiv:2512.05033v2[cs.CL]9Dec2025

### ARBITRAGE: Efficient Reasoning via Advantage-Aware Speculation

Monishwaran Maheswaran*1 Rishabh Tiwari*1 Yuezhou Hu*1 Kerem Dilmen1 Coleman Hooper1 Haocheng Xi1 Nicholas Lee1 Mehrdad Farajtabar2 Michael W. Mahoney1,3,4 Kurt Keutzer1 Amir Gholami1,3

1 UC Berkeley 2 Apple 3 ICSI 4 LBNL

Abstract

Modern Large Language Models achieve impressive reasoning capabilities with long Chain of Thoughts, but they incur substantial computational cost during inference, and this motivates techniques to improve the performance-cost ratio. Among these techniques, Speculative Decoding accelerates inference by employing a fast but inaccurate draft model to auto-regressively propose tokens, which are then verified in parallel by a more capable target model. However, due to unnecessary rejections caused by token mismatches in semantically equivalent steps, traditional tokenlevel Speculative Decoding struggles in reasoning tasks. Although recent works have shifted to step-level semantic verification, which improve efficiency by accepting or rejecting entire reasoning steps, existing step-level methods still regenerate many rejected steps with little improvement, wasting valuable target compute. To address this challenge, we propose ARBITRAGE, a novel step-level speculative generation framework that routes generation dynamically based on the relative advantage between draft and target models. Instead of applying a fixed acceptance threshold, ARBITRAGE uses a lightweight router trained to predict when the target model is likely to produce a meaningfully better step. This routing approximates an ideal ARBITRAGE ORACLE that always chooses the higher-quality step, achieving near-optimal efficiency–accuracy trade-offs. Across multiple mathematical reasoning benchmarks, ARBITRAGE consistently surpasses prior step-level SD baselines, reducing inference latency by up to ∼ 2× at matched accuracy. Our code is available at https://github.com/SqueezeAILab/Arbitrage

#### 1 Introduction

Large Language Models (LLMs) have achieved remarkable progress across a wide range of domains, with mathematical reasoning standing out as a particularly transformative frontier [45, 29, 35, 43, 34]. This rapid advancement has been fueled by the release of high-quality, reasoning-centric datasets [53, 4, 26, 6] as well as innovations in reinforcement learning algorithms tailored for complex reasoning tasks [29, 52, 48]. As a result, state-of-the-art LLMs now rival, or even surpass, human performance on challenging mathematics and coding benchmarks, including international olympiad-level tasks [10, 33, 58, 14]. Despite these impressive capabilities, the inference efficiency of transformer-based LLMs remains a critical bottleneck. While training is typically compute-bound, auto-regressive decoding is fundamentally memory-bound: each token generation step relies on matrix-vector multiplications rather than the more hardware-efficient matrix-matrix operations. Consequently, the throughput of modern GPUs is limited not by raw compute power, but by global memory bandwidth, a phenomenon widely known as the “memory wall” [7]. This constraint severely limits token generation speed and end-to-end latency, and the problem is exacerbated in long chain-of-thought reasoning, where solutions often span hundreds or thousands of tokens, leading to many decoding steps and substantial latency. In this work, we focus on when to spend expensive compute, rather than how to make each forward pass cheaper. A more holistic strategy for accelerating end-to-end generation is Speculative Decoding (SD) [17, 3], which leverages parallelism by pairing a small, fast draft model with a larger, more accurate target model. The draft model auto-regressively generates multiple tokens in sequence, while the target model verifies them in parallel via a single forward pass; accepted tokens are appended to the output, and rejected tokens trigger fallback

*Equal Contribution

Arbitrage (Ours)

Step 1: Draft Model's answer is accepted fi target model is not expected to be better than the draft

Step 2: Draft Model's answer is rejected fi target model is expected to propose a better answer

|Draft Model|
|---|

###### Question

Draft Model

Target Model

Router Router

Move on to next step

Move on to next step

Generate Again

Generate

Generate

Score

Score

Step 1

Step 2

Step 2

The Model does not contribute to the final answer

The Model contributes to the final answer

- Figure 1: ARBITRAGE overview. At each reasoning step, the draft proposes a candidate. The router produces a score yˆ, which is the estimated probability that the target will outperform the draft on this step, and accepts the draft if yˆ ≤ τ, otherwise escalates to the target to regenerate (yˆ > τ). The selected step is appended to the context. The threshold τ governs the compute–quality trade-off.

to the target model. In this paradigm, the central question is: when is it actually beneficial to switch from the draft to the target model? Ideally, we should only invoke the target on those reasoning steps where it is expected to provide a meaningfully better continuation than the draft.

Classical SD methods answer this question at the token level, accepting or rejecting tokens based on exact agreement between draft and target distributions. Although effective in some settings, token-level SD suffers from low acceptance rates in complex reasoning tasks, particularly in chain-of-thought generation, where minor token-level discrepancies can lead to premature rejection of otherwise high-quality, semantically-equivalent reasoning steps [20, 27], thus wasting computation. To address this challenge, recent work has shifted toward step-level Speculative Decoding. Notably, Reward-guided Speculative Decoding (RSD) [20] evaluates entire reasoning steps (e.g., delimited by \n\n) using a Process Reward Model (PRM) and accepts a draft step if its absolute PRM score exceeds a global threshold, otherwise discarding it and regenerating with the target model. This thresholding rule depends only on whether the draft looks “good enough” in isolation, rather than on whether the target is actually expected to produce a better step, making the routing decision fundamentally advantage-blind and often triggering costly target regenerations that yield little or no quality gain.

Arbitrage (Ours)

Step 1: Draft Model's answer is accepted

Step 2: Draft Model's answer is rejected

###### Question Router

Draft Model

Target Model

Router

Move on to next step

Move on to next step

Generate

Generate

Advantage

Advantage

Sentence 1

Sentence 2

In this work, we address this issue by proposing ARBITRAGE, a novel step-level speculative generation framework that dynamically routes between the draft and target models to minimize redundant computation while preserving, or even enhancing, output quality; see Figure 1 for an overview. Our key insight is to base the acceptance decision not just on the quality of the draft output (independent of the target model), but on the expected quality difference between the draft and target models for a particular reasoning step. Conceptually, we define an ARBITRAGE ORACLE that, for each step, compares the draft- and target-generated continuations and greedily selects the higher-quality one according to a given metric. Since running this oracle naively would require executing the target on every step, we instead train a lightweight ARBITRAGE ROUTER that, given partial context, predicts how much a target-generated step is likely to outperform its draft counterpart. By anchoring decisions on the expected quality advantage, ARBITRAGE avoids unnecessary target generations when the target model is expected to yield a similar-quality output as that of the draft model.

RSD

In summary, our main contributions are as follows.

- • We systematically analyze the computational waste inherent in existing step-level SD methods, showing that a large fraction of regenerations fail to improve output quality (see Section 3.2).
- • Based on our analysis, we propose ARBITRAGE, a step-level speculative generation framework that routes decisions based on the expected quality advantage between the draft and target models, rather than an absolute quality threshold (see Section 4).
- • We introduce a formal ARBITRAGE ORACLE, which, at each reasoning step, compares the draft and target

- generated continuations and selects the higher-quality one according to a given metric. This defines a myopic (greedy) policy that is locally optimal for the per-step routing decision (see Section 4.1).
- • We introduce a principled training pipeline for the ARBITRAGE ROUTER, which is a practical lightweight model that can mimic ARBITRAGE ORACLE during inference (see Section 4.2).
- • We evaluate ARBITRAGE on different benchmarks and model settings, showing that it reduces inference latency by up to ∼ 2× over baselines for a given accuracy target (see Sections 5.2 and 5.3).

#### 2 Related Work

- 2.1 Chain-of-Thought Reasoning

Chain-of-Thought (CoT) reasoning has been shown to improve model performance without requiring additional training. The simplest approach uses prompt engineering, e.g., instructing models to “think step by step” to elicit multi-step reasoning in instruction-following LLMs [41]. More advanced methods employ search algorithms [32, 50, 25] or reward models [21, 42, 51] to guide generation toward higher-quality reasoning paths. To directly enhance inherent reasoning capabilities of models, recent work applies Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)[28, 47, 34, 29, 15, 45, 4]. These training-based approaches significantly boost model capacity, enabling smaller models to match or even surpass larger counterparts in reasoning performance.

- 2.2 Speculative Decoding for Reasoning

Speculative Decoding (SD) [17] accelerates autoregressive generation by using a small, fast draft model to propose tokens that are then verified in parallel by a larger, more accurate target model. Traditional SD methods operate at the token level [16, 23, 54, 36, 1, 19, 2, 13]. However, token-level alignment results in premature repeated rejections in reasoning tasks where semantic level equivalence is enough. To address this, recent work shifts verification from tokens to entire reasoning steps. This is typically delimited by structural markers such as double newlines (\n\n) in CoT outputs [20] or phrases like “wait” or “let me rethink” in reasoning chains [46, 27]. This step-level approach avoids discarding an entire high-quality reasoning trajectory due to a single rejected intermediate token, thereby improving both efficiency and robustness.

- 2.3 Reward Models for Reasoning Guidance

Reward Models (RMs) play a central role in evaluating and guiding speculative generation by scoring output quality. Outcome-supervised Reward Models (ORMs) assign rewards based solely on the correctness of the final answer [55, 22]. In contrast, Process-supervised Reward Models (PRMs) [11, 40, 56, 37, 21] provide fine-grained, per-step quality assessments of the reasoning process itself. By evaluating intermediate logical steps, PRMs enable more nuanced feedback and are particularly well-suited for step-level SD frameworks. In this work, we focus on CoT-style reasoning, where PRMs provide reliable step-level signals.

#### 3 Preliminaries

We begin by formalizing reward-based speculative generation through the lens of a simple guiding principle: we should only route from the draft model to the target model when doing so is expected to help—that is, when the target is likely to provide a strictly better reasoning step than the draft under the same evaluation signal. Concretely, we consider step-level speculative generation, in which a Process Reward Model (PRM) scores entire candidate reasoning steps and these scores are used to decide whether to accept the draft or regenerate with the target. Prior work, illustrated by RSD [20], fits into this framework but makes routing decisions based solely on the draft step’s PRM score compared against a fixed global threshold. Whenever the draft score falls below this threshold, the system invokes the target model, regardless of whether the target is actually likely to improve the step. This absolute-score rule often triggers costly target regenerations that yield little or no quality gain. In Section 3.1 , we review this baseline in detail; in Section 3.2, we analyze its computational and behavioral limitations, quantifying the resulting wasted compute. These

observations motivate a relative, advantage-based routing strategy (discussed in Section 4), in which the draft-to-target switch is explicitly tied to the target model’s expected advantage over the draft.

##### 3.1 Overview of RSD

Like classical SD, RSD employs a lightweight draft model in conjunction with a more powerful target model during text generation. A key distinguishing feature of RSD is its integration of a PRM that provides step-level supervision. By operating at a larger granularity through the use of reward signals from a PRM, RSD can adaptively accept highvalue draft outputs, rather than rejecting them due to token-level mismatch. At each generation step, the draft model proposes a candidate reasoning step, which is then scored by the PRM. Based on this score, the system decides whether to accept the draft output or fallback to the target model for regenerating the step. However, unlike classical SD, because the PRM evaluates the entire reasoning step holistically rather than token-by-token, it avoids the computational overhead of fine-grained token-wise verification, thereby maintaining efficiency.

Formally, let the input context be a sequence of tokens x = [x1, x2, . . . , xt]. The draft model θdraft or target model

θtarget can autoregressively generate a candidate reasoning step z = {zi}iγ=1 until it produces a designated separator token (e.g., \n\n), which marks the end of the step. This generation process is defined as:

zi ∼ qθmodel(zi | x,z<i), for i = 1, . . . , γ,

where z<i = [z1, . . . , zi−1] and zγ is the separator token. Here, θmodel can be either θdraft or θtarget. We use zd to denote the draft generated step and zt be the target generated step. We first generate zd, which is then evaluated by the PRM, defined as hθPRM : X × Z → R. The PRM model outputs a scalar reward:

sd = hθPRM(x,zd).

This score is used to decide whether to accept the speculation or reject it. The binary accept indicator is

A(x,zd) = I{sd > τ}, τ ∈ R.

That is, the step zd is accepted if sd > τ (where τ is a fixed threshold); otherwise, zd is discarded and zt is generated by the target model. For a question that requires n reasoning steps, the empirical acceptance rate is α = n1 ∑in=1 Ai, where Ai is the accept indicator of step i. Crucially, α is inversely related to τ: a higher threshold yields lower acceptance (fewer steps accepted), trading speed for quality. By tuning τ, RSD controls the efficiency–accuracy trade-off. This PRM-guided mechanism dynamically balances computational cost and output quality.

##### 3.2 Limitations of Current Approach

While [20] demonstrates the effectiveness of reward-based speculative generation, we identify a fundamental inefficiency in its rejection mechanism. When the PRM rejects a draft step based on its reward score, regenerating that step with the target model often provides little or no improvement in quality. To build intuition for how this mechanism behaves, Figure 2 presents a schematic example of step-level decisions under RSD and our method. Crucially, this regeneration incurs the full computational cost, despite yielding negligible quality gains, and this leads to substantial wasted compute. To illustrate this, let st = hθPRM(x,zt) be the corresponding reward for the target generated step if zd is rejected (A(x,zd)) = 0). The regeneration incurs cost Ct, while the previous draft generation cost Cd is paid regardless.

When does rejection help? Quality improves only if the target step outperforms the draft step under the same PRM: benefit = I{A = 0} I{st > sd}.

Otherwise, the expensive regeneration is wasted. We formalize the expected wasted computation of current approaches as:

###### WRSD ≜ E Ct · I{A = 0} · I{st ≤ sd} ,

which grows both with the rejection rate Pr(sd ≤ τ) and with the failure probability Pr(st ≤ sd | sd ≤ τ). Observe that this inefficiency comes from two key factors.

###### WASTED REGENERATION

STEP 3 REJECTED STEP 4 REJECTED STEP 7

{

STEP 1

p/q

2 = p2/q2 ⇒ p2 = 2q2 p which implies p = (2k)2 = 2q2 ⇒ 4k2 = 2q2 q2 = 2k2

…

…

Contradicting in lowest terms. is irrational.

Assume, for contradiction, that

which implies is odd

→ 2

2 is rational.

Reward : 0.75 > 0.7 Reward : 0.35 < 0.7 Reward : 0.4 < 0.7

RSD

STEP 3 STEP 4

p = (2k)2 = 2q2 ⇒ 4k2 = 2q2 q2 = 2k2

2 = p2/q2 ⇒ p2 = 2q2 p

which implies is even

which implies

Reward : 0.4

Reward : 0.65

ACCEPTED

REJECTED

Arbitrage{

###### STEP 4 STEP 7

STEP 3

###### STEP 1

p = (2k)2 = 2q2 ⇒ 4k2 = 2q2 q2 = 2k2

p/q

2 = p2/q2 ⇒ p2 = 2q2 p

…

…

Contradicting in lowest terms. is irrational.

Assume, for contradiction, that

which implies is odd

2 is rational.

→ 2

which implies

Reward : 0.75 | ∀ Reward : 0 ∈ 0 Reward : 0.35 | ∀ Reward : 0.3 > 0 Reward : 0.4 | ∀ Reward : 0 ∈ 0

###### STEP 4

STEP 3

STEP 1

p = (2k)2 = 2q2 ⇒ 4k2 = 2q2 q2 = 2k2

2 = p2/q2 ⇒ p2 = 2q2 p

Assume, for contradiction, that

which implies is even

2 is rational.

which implies

Reward : 0.65 Reward : 0.4

Reward : 0.75

NOT GENERATED

DRAFT MODEL

TARGET MODEL

- Figure 2: ARBITRAGE vs. baseline step-level SD approaches. Comparison of Reward-guided Speculative Decoding (RSD, top, which we use as a baseline) and our ARBITRAGE algorithm (bottom). RSD accepts or rejects draft steps using an absolute PRM reward threshold: when the PRM score of a draft-generated step falls below this threshold, the step is discarded and the target model is invoked to regenerate it. This absolute criterion can trigger unnecessary target regenerations (e.g., Step 4), where the target does not significantly improve the quality of the step. ARBITRAGE instead estimates the expected quality gain from escalating a step, i.e., invoking the larger target model to regenerate the step rather than keeping the draft step. It only calls the target when this predicted gain is positive, thereby avoiding wasted target calls (e.g., Steps 1 and 4).

- 1. On some difficult steps, the PRM assigns a low score to a draft that is essentially correct, causing a deferral to the target. The target then produces a step of comparable quality, yielding no gain, but incurring the full computational cost.
- 2. For the hardest instances, the draft and target may exhibit the same logical gaps or reasoning errors, so regeneration is unlikely to improve quality.

We empirically validate this phenomenon. Figure 3 summarizes the wasted compute due to useless deferrals to the target model for regeneration. We observe that this wasted compute w.r.t. total cost increases with higher deferral rates. For example, at a deferral rate of 70%, approximately 40% total steps were regenerated with the target model without any gain in output quality. See Appendix A for further analysis.

#### 4 Method

In this section, we formalize our proposed framework: ARBITRAGE, which aims to minimize redundant computation in step level speculative decoding methods. Building on the inefficiencies we identified in Section 3.2, our goal is to design a routing policy that dynamically decides at each reasoning step whether invoking the target model will meaningfully improve the step quality. We begin in Section 4.1 by defining the ARBITRAGE ORACLE, which establishes the theoretical upper bound for routing efficiency by comparing the counterfactual rewards of draft and target generations. We then in Section 4.2 introduce the ARBITRAGE ROUTER, a lightweight predictive model that approximates this oracle using only the draft’s context, thereby enabling near-optimal routing decisions, without executing the expensive target model.

60

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

WastedDeferralCalls(%)

50

40

30

20

0.3 0.4 0.5 0.6 0.7 0.8 0.9

Deferral Rate

- Figure 3: Wasted target calls vs. deferral rate. In reward-based step-level speculation, the deferral rate (x-axis) is the fraction of reasoning steps that are escalated to the target model. The wasted deferral rate (y-axis) is the percentage of those escalations where the target’s step is no better than the draft (equal or lower PRM score), relative to total number of steps. Wasted compute increases steadily with deferral rate, indicating many unnecessary target invocations under absolute-score rejection.

##### 4.1 ARBITRAGE ORACLE: The Ideal Routing Strategy

Building on the identified limitations of absolute-score-based rejection, we formalize the ARBITRAGE ORACLE. Unlike conventional approaches that accept or reject draft steps based solely on the draft’s PRM score, sd, the ARBITRAGE ORACLE makes routing decisions by comparing the draft and target models’ counterfactual PRM scores for the same reasoning step. Specifically, given a draft step zd and its target-generated counterpart zt for context x, the oracle selects the sequence with the higher PRM score:

z∗ = argmax z∈{zd,zt}

hθPRM(x,z). (4.1)

This strategy establishes an upper bound for routing performance: it achieves greedy-maximal output quality for any given acceptance rate by always selecting the highest-quality step available.

To make this principle concrete while controlling the acceptance rate, we introduce two simple notions. First, we define the advantage of the target over the draft on a given step:

Definition 4.1: Step-level advantage

Given a context x, a draft step zd with PRM score sd = hθPRM(x,zd) and a target step zt with score st = hθPRM(x,zt), the advantage of the target over the draft on this step is

∆ = st − sd. (4.2) A positive ∆ indicates that the target is strictly preferred to the draft under the PRM.

This mirrors the standard “advantage” terminology from reinforcement learning, and a positive ∆ indicates that the target is strictly preferred to the draft under the PRM. Second, we define an escalation decision, which encodes the routing choice of whether to accept the draft step or escalate to the target model:

Definition 4.2: Escalation decision and realized score

An escalation decision is a binary variable a ∈ {0,1}, where a = 1 means we escalate to the target model (regenerate the step with the target) and a = 0 means we do not escalate and accept the draft step. The corresponding realized PRM score under decision a is

S(a) = (1 − a) sd + a st = sd + a ∆. (4.3)

A routing policy π maps available information to a ∈ {0,1}. When controlling the escalation rate via a scalar threshold τ ∈ R, the oracle uses the thresholding policy

a⋆τ = I{∆ > τ}. (4.4)

The threshold τ regulates the trade-off between accuracy and how often we escalate to the target, and thus directly controls the computational cost. Appendix B proves that, under a greedy per-step objective, this thresholding rule is optimal: among all policies with the same escalation rate, it maximizes the expected PRM score S(a).

While the ARBITRAGE ORACLE establishes the upper bound for routing performance, it remains computationally infeasible in practice. Evaluating st requires executing the target model to obtain zt, precisely the costly operation that SD aims to avoid. Consequently, our central challenge is to approximate the oracle’s routing decision without invoking the target model itself. Formally, this reduces to estimating the expected advantage, conditioned on the observed context and draft step:

Can we estimate how much better the target would score than the draft on the current step, without actually running the target model?

A practical solution to this question would enable near-oracle routing decisions while preserving the speedups of SD. To this end, the next subsection introduces the ARBITRAGE ROUTER, a lightweight predictive model trained to estimate this conditional advantage from partial context, thus approximating oracle-level routing at minimal additional cost.

##### 4.2 ARBITRAGE ROUTER: The Practical Routing Strategy

We address the above challenge of the impracticality of computing the ARBITRAGE ORACLE with a lightweight ARBITRAGE ROUTER that, given only the draft-side context and step: (x,zd), predicts whether escalating will improve quality. Concretely, the router outputs a score yˆ = hθrouter(x,zd) interpreted as the likelihood that the target outperforms the draft. For routing during inference, we use a threshold τ (reject iff yˆ > τ), enabling advantage-aware decisions without executing the target during routing. The ARBITRAGE ROUTER is trained offline using data labeled by the oracle (see Section 4.1) to learn its routing decisions.

Oracle-free routing objective Given context x and draft step zd, predict the oracle advantage

∆(x,zd) = st − sd, or equivalently its sign y = I[∆ > 0], using only draft-side information, without executing the target model.

###### Dataset Construction

Using the step-level advantage ∆ (Definition 4.1) and the oracle label y, we construct a step-level dataset as follows. The oracle label for escalation is

y = I[∆ > 0], indicating whether invoking the target yields higher quality. For each context x, we decode both the draft and target models from the same prefix to obtain paired steps (zd, zt), and compute their PRM scores sd and st with the fixed PRM hθPRM, as in Definition 4.1. The resulting tuples

(x, zd, zt, sd, st, ∆, y)

form the supervised training data for the router. To reduce variance in the oracle signal, we optionally draw multiple target samples per context and average their PRM scores to obtain s¯t and the corresponding averaged advantage ∆¯ ; the oracle label y is then computed from this averaged advantage.

###### Data Preprocessing

We sample 30K questions via stratified sampling from the NuminaMath-CoT dataset [18], to be used as a seed dataset for our fine-tuning. Because the majority of draft steps are acceptable, the oracle labels y exhibit strong class imbalance (y = 0 dominates y = 1). For example, with the Llama family configuration, the label distribution was 62.27% for y = 0 and 37.73% for y = 1. For better performance, we add annotations to each history step indicating the invoked models. We apply random downsampling of the majority class to balance the training set and prevent bias toward over-accepting draft steps. We also normalize sequence lengths and enforce a consistent step separator token (\n\n) to align PRM scoring and router inputs.

Training ARBITRAGE ROUTER

The router is initialized from a compact PRM checkpoint (reasoning-aware, step-level pretraining) to provide an inductive bias for evaluating intermediate reasoning quality. We finetune the model with the AdamW optimizer [24], using linear warmup and decay. The classification head is applied on the final token embedding, and the model is optimized via standard cross-entropy loss. The training hyperparameters are presented in the Table C.1 in Appendix C.

###### Routing Quality Evaluation.

Evaluating the quality of the router during training is non-trivial. Unlike standard classifiers, we cannot afford to perform full end-to-end evaluation of the entire SD pipeline across a range of thresholds τ, as each such evaluation would require executing both the draft and target models, which would be prohibitively expensive. Consequently, we need a reliable proxy metric that reflects how well the router aligns with the oracle’s routing decisions, without repeatedly running the full algorithm. Traditional metrics such as accuracy, precision, or recall at a fixed threshold can serve as a simple evaluation metric to judge the quality of the router. Since y = 0 and y = 1 leads to generations from different models, we observe that:

- 1. Accuracy of class y = 0 (accept draft) ensures high acceptance rates and fast generation.
- 2. Accuracy of class y = 1 (reject draft) ensures that challenging steps—where the draft model is likely to fail—are delegated to the target model, preserving output quality and overall accuracy.

However, these traditional metrics are heavily dependent on the chosen τ, and they do not capture how the router’s predictions correlate with the true advantage signal that drives the oracle. Thus, they provide limited insight: a router may achieve high accuracy at a specific threshold, and yet still mis-rank steps by their expected improvement, leading to suboptimal routing behavior when τ changes. To solve this, and to use a fixed, global metric to evaluate overall quality, we adopt Spearman’s rank correlation [49] between the predicted routing probabilities yˆ and the oracle advantage scores ∆:

ρ = corrS(yˆ, ∆). (4.5) This rank-based correlation serves as a threshold-invariant proxy for router quality, measuring how well the predicted ordering of steps matches the oracle’s ground-truth ordering of advantages. A high ρ indicates that the router consistently assigns higher probabilities to steps where escalation truly helps, implying robust performance across a wide range of thresholds, even without explicitly re-running the full decoding algorithm.

###### ARBITRAGE Speculative Generation.

Figure 1 shows the overall ARBITRAGE speculative generation workflow during inference with ARBITRAGE ROUTER. The process is formulated as a step-level speculation: (i) the draft model proposes a candidate step; and (ii) the router decides whether to accept it or escalate to the target model for regeneration. Algorithm 1 in Appendix D summarizes the complete decoding procedure. The router introduces a small overhead of one forward pass per step, while offering fine-grained control over the compute–quality trade-off through the threshold τ.

#### 5 Evaluation

In this section, we evaluate the performance of our framework, ARBITRAGE, across different model configurations and reasoning benchmarks. We first analyze accuracy under varying acceptance rates (as defined in Section 3.1), which

Llama3 (1B/8B) | Math500

50

| |
|---|

45

Accuracy(%)

| |
|---|

40

Arbitrage Oracle Arbitrage Router RSD

35

| |
|---|

Target Accuracy

30

| |
|---|

Draft Accuracy

0 20 40 60 80 100

Acceptance Rate (%)

- 4

8

12

16

Accuracy(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Llama3 (1B/8B) | OlympiadBench

Arbitrage Oracle Arbitrage Router RSD

Target Accuracy

Draft Accuracy

(a)

0 20 40 60 80 100

Acceptance Rate (%)

50

55

60

65

70

75

Accuracy(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Llama3 (8B/70B) | Math500

0 20 40 60 80 100

Acceptance Rate (%)

20

25

30

35

40

Accuracy(%)

| |
|---|

| |
|---|

| |
|---|

Llama3 (8B/70B) | OlympiadBench

(b)

0 20 40 60 80 100

Acceptance Rate (%)

76

78

80

82

84

Accuracy(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Qwen2.5-Math (3bit-7B/7B) | Math500

0 20 40 60 80 100

Acceptance Rate (%)

32

34

36

38

40

Accuracy(%)

| |
|---|

| |
|---|

Qwen2.5-Math (3bit-7B/7B) | OlympiadBench

(c)

Figure 4: ARBITRAGE improves the compute–quality trade-off. Accuracy vs. acceptance rate for ARBITRAGE ORACLE, ARBITRAGE ROUTER, and RSD across two benchmarks (MATH500 and OlympiadBench) and three model configurations. The top row shows results on MATH500 and the bottom row on OlympiadBench. Columns (a), (b), and (c) correspond to LLaMA3 (1B/8B), LLaMA3 (8B/70B), and Qwen2.5-Math (3bit-7B/7B), respectively. In all cases, ARBITRAGE consistently yields higher accuracy at comparable acceptance rates, demonstrating superior compute–quality efficiency. Additional results are provided in Appendix E.

control the balance between using the draft and target models. We then present end-to-end latency results, showing that ARBITRAGE achieves substantial speedups by performing fewer, more effective escalations to the target model.

- 5.1 Setup

0 20 40 60 80 100

Acceptance Rate (%)

All experiments were conducted on NVIDIA A6000 GPUs using SGLang as the inference backend, each model occupying a dedicated GPU. For speedup measurements, we fix the batch size to 1. We evaluate ARBITRAGE on two representative model families, LLaMA3 [8] and Qwen2.5-Math [44], under two practical draft–target regimes:

- 1. A large target with a smaller draft from the same family.
- 2. A large target with a weight-quantized version of itself as the draft.

We use llama.cpp for quantizing Qwen models and GPTQ [5] for quantizing Llama models. All the models chosen are instruction finetuned versions from their respective family. For the PRM, unless stated otherwise, we use Skywork-

- o1-Open-PRM (1.5B) [31], and we fine-tune Qwen2.5-Math-1.5B-Instruct-PRM-0.2 [38] to obtain an ARBITRAGE ROUTER tailored to each target–draft pair. For brevity, from here on we use notations like ARBITRAGE (4bit-7B/7B) to denote ARBITRAGE based inference using a 4bit weight quantized 7B draft model, a bf16 7B target model and 1.5B router.

- 5.2 Results

- Figure 4 plots accuracy (y-axis) as a function of acceptance rate (x-axis) on MATH500 [12] and OlympiadBench [9] for three routing configurations: LLaMA3 (1B/8B), LLaMA3 (8B/70B), and Qwen2.5-Math (3bit-7B/7B). The top row shows MATH500 and the bottom row OlympiadBench; in each panel we also include the draft-only and target-

- only accuracies as horizontal reference lines, along with ARBITRAGE ORACLE as an upper bound. Across all six model–dataset combinations, the ARBITRAGE ROUTER curve lies strictly above RSD at nearly all acceptance rates,

Llama3 (4bit-8B/8B) | Math500

Llama3 (1B/8B) | OlympiadBench

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | |1|.97× sp|eedup| | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

16

50

1.62× speedup

Accuracy(%)

Accuracy(%)

48

12

46

Arbitrage Router

8

RSD

44

Target Model

Draft Model

42

4

6 8 10 12 14 16

5 10 15 20

Average Latency (sec)

Average Latency (sec)

(a)

(b)

- Figure 5: ARBITRAGE improves the compute–quality trade-off. Accuracy–time curves for ARBITRAGE ROUTER and RSD on two LLaMA3 routing configurations. Subplot (a) reports results for a quantized-draft / full-precisiontarget setting (Q4-bit-8B/8B/1.5B) on MATH500, and subplot (b) for a small-draft / large-target setting (1B/8B/1.5B) on OlympiadBench. Across both configurations, ARBITRAGE ROUTER consistently achieves higher accuracy at a given wall-clock time than RSD, yielding a better Pareto frontier. Each marker corresponds to a different threshold operating point; moving right indicates increased target-model invocations (and thus higher latency).

indicating that advantage-aware routing extracts more accuracy per unit of target usage. In these regimes, the ARBITRAGE ROUTER curve tracks the oracle upper bound closely, while RSD remains noticeably below it. These empirical trends indicate that ARBITRAGE ROUTER closely tracks the oracle across model scales, datasets, and acceptance regimes.

Looking more closely at individual configurations, we observe that the gains from ARBITRAGE ROUTER are most pronounced when the draft model is substantially weaker than the target. For example, in the LLaMA3 (1B/8B) setting on both MATH500 and OlympiadBench, RSD remains much closer to the draft-only baseline over a wide range of acceptance rates, whereas ARBITRAGE ROUTER rapidly climbs toward the target-only line. This suggests that when the potential advantage of escalation is large but unevenly distributed across queries, advantage-aware routing is particularly effective at identifying the subset of examples where the target model is truly needed. In contrast, the LLaMA3 (8B/70B) and Qwen2.5-Math (3bit-7B/7B) settings exhibit smaller draft–target gaps, yet ARBITRAGE ROUTER still secures a consistent margin over RSD, demonstrating that our method remains beneficial even when the draft is already relatively strong. Near the high-acceptance regime, ARBITRAGE ROUTER remains closer to the oracle than RSD, implying that it continues to allocate target computation to challenging instances rather than escalating indiscriminately. Finally, the consistent ordering of the methods across both MATH500 and OlympiadBench suggests that the advantages of relative-advantage routing are robust to dataset difficulty and distributional shift, rather than being tied to a particular benchmark.

##### 5.3 Speedup Analysis

We quantify the compute savings of ARBITRAGE by measuring the average end-to-end wall-clock time per problem as we sweep the routing threshold, yielding the accuracy–latency Pareto curves in Figure 5. Each point on a curve corresponds to a different operating threshold, and therefore a different fraction of queries that are escalated from the draft to the target model. Moving to the right along a curve increases the escalation rate and thus the overall latency, since more problems are solved (partially or fully) by the slower target model. Across all model and dataset configurations, ARBITRAGE strictly dominates RSD: for any fixed latency budget, ARBITRAGE attains higher accuracy, and for any desired accuracy, it achieves a lower latency. On MATH500 under the quantized-draft regime (Q4-8B/8B), ARBITRAGE delivers up to 1.62× lower latency at comparable accuracy relative to RSD. On OlympiadBench with the small-draft regime (1B/8B), it reaches up to 1.97× speedup at matched accuracy. Qualitatively, we observe that the largest gains occur in the mid-latency regime. Here, RSD tends to over-escalate on examples for which the target offers marginal benefit, incurring substantial latency without commensurate accuracy gains. In contrast, ARBITRAGE concentrates target compute on instances where the estimated advantage over the draft is highest, and relies on the

draft model elsewhere. This selective escalation yields curves that rise more steeply with latency, reflecting a more favorable compute–quality trade-off.

##### 5.4 Case Studies

QUESTION

STEP 3

REGENERATED by TARGET PRM Score: 0.14

[Figure 1]

A square and a regular heptagon are coplanar and share a common side , as shown. What is the degree measure of angle

Next, let's consider the regular heptagon. The measure of each interior angle of a regular heptagon can be calculated using the formula for the interior angle of a regular polygon:

AD BAC

? Express your answer as a common fraction.

(n − 2) × 180∘ n

Interior angle =

n n = 7

where is the number of

sides. For a heptagon, :

REJECTED by RSD | REJECTED by ARBITRAGE PRM Score: 0.20 | ARBITRAGE Score: 0.84

STEP 1

(7 − 2) × 180∘ 7

5 × 180∘ 7

900∘ 7

Interior angle =

=

=

∠BAC

To determine the degree measure of angle , we need to analyze the properties of the square and the regular heptagon. Let's start by understanding the angles formed by their sides.

# . . .

REGENERATED by TARGET PRM Score: 0.19

- STEP 1

First, let's consider the square. Since it is a square, all its internal angles are 90∘. Therefore, ∠CAD = 90∘.

- STEP 2 STEP 7

∠BAC AD

To determine the measure of angle , we need to analyze the properties of the square and the regular heptagon that share the common side .

REJECTED by RSD | ACCEPTED by ARBITRAGE PRM Score: 0.22 | ARBITRAGE Score: 0.77

ACCEPTED by RSD & ARBITRAGE PRM Score 0.49 | ARBITRAGE Score: 0.55

To subtract these angles, we need a common denominator:

90 × 7 7

630∘ 7

90∘ =

=

REGENERATED by TARGET PRM Score: 0.22

STEP 2

First, let's consider the square. Since it is a square, each interior angle is 90∘. Therefore, ∠CAD = 90∘.

STEP 8

ACCEPTED by RSD & ARBITRAGE PRM Score 0.60 | ARBITRAGE Score: 0.55

Now, we can subtract:

900∘ 7

630∘ 7

900∘ − 630∘ 7

270∘ 7

REJECTED by RSD/ACCEPTED by ARBITRAGE PRM Score: 0.15 | ARBITRAGE Score: 0.73

STEP 3

∠BAC =

−

=

=

Next, let's consider the regular heptagon. The sum of the interior angles of a heptagon is given by the formula: Since the heptagon is regular, each interior angle is:

STEP 9

ACCEPTED by RSD & ARBITRAGE PRM Score 0.25 | ARBITRAGE Score: 0.36

(7 − 2) × 180∘ = 5 × 180∘ = 900∘

|270 7<br><br>|
|---|

Thus, the degree measure of angle ∠BAC is:

900∘ 7

- Figure 6: Qualitative example from MATH500. We first obtain a full solution using RSD and then rescore every step with ARBITRAGE ROUTER under the same acceptance budget. Steps kept by RSD are highlighted in blue, those it rejects are in red or orange, and steps regenerated by the target model are shown in green. Notably, steps 2 and

- 3 (orange) are accepted by ARBITRAGE ROUTER but rejected by RSD: their regenerated versions are essentially identical in content and still lead to the correct final answer of 270/7. The example shows how RSD’s global reward cutoff can discard valid intermediate reasoning, whereas ARBITRAGE ’s advantage-based, relative routing preserves such steps and eliminates redundant regeneration.

To further illustrate the practical advantages of ARBITRAGE over the baseline, we conduct a qualitative case study on the MATH500 benchmark using Qwen2.5-Math 3-bit-7B/7B as the draft and target models. Starting from an RSD-generated trajectory, we re-score each step with ARBITRAGE ROUTER while enforcing the same 80% overall acceptance rate for both approaches. As visualized in Figure 6, RSD rejects several intermediate steps (shown in red

and orange) because their individual PRM scores fall below a fixed global cutoff. Many of these steps are in fact correct and encapsulate nontrivial reasoning. For example, determining the interior angles of the regular heptagon and establishing the necessary geometric relationships. When these steps are regenerated by the target model (green), the resulting text is semantically almost identical and still leads to the correct final answer. In contrast, ARBITRAGE keeps such steps considering the predicted advantage of the target over the draft: routing decisions are based on the relative benefit of escalation rather than an absolute reward threshold. This advantage-aware view avoids redundant regenerations, preserves coherent reasoning trajectories, and allocates target compute only where a substantially better step is likely.

#### 6 Ablations

We conduct ablation studies to understand how different design choices affect the performance of ARBITRAGE ROUTER. We focus on two main dimensions: (1) model architecture, and (2) data-related design choices. On the architecture side, we compare multi-class classification against regression-based variants. On the data side, we evaluate the impact of stepwise annotation and class-balanced downsampling. For each setting, we report the Spearman correlation between the router’s predictions and the oracle advantage scores ∆ on the evaluation set, as well as accuracy for both classes. In every ablation, we modify only the factor under study and keep all other components fixed. Some ablations were run mid-development, so their final checkpoints may not be strictly comparable to the main model. Unless otherwise noted, we finetune each variant for 3 epochs.

##### 6.1 Model Architecture Label Granularity

Prior work suggests that supervision with fine-grained labels can improve performance on coarser downstream decisions [57, 30]. Motivated by this, we explore multi-class, ordinal variants of the routing objective. Specifically, we partition the training and evaluation data into 2, 4, or 10 classes based on quantiles of the true advantage score ∆. Table 1 summarizes the results. The 2-class setting achieves the highest overall Spearman correlation and maintains balanced accuracy across the two labels (accept: label 0 vs. escalate: label 1), indicating that it provides a good trade-off between acceptance rate and end-task quality. When we increase the number of classes to 4 or 10, overall correlation drops or the accuracy becomes skewed between label 0 and label 1, leading to worse routing behavior. Empirically, the advantage scores are densely concentrated around zero with a long positive tail. In this regime, aggressive discretization into many bins introduces ambiguous decision boundaries and effectively blurs neighboring classes. Given these observations, we adopt the 2-class formulation as our default.

###### Classification vs. Regression

Because the advantage score ∆ is continuous, a natural alternative to classification is to train a regression model. For ordered labels, ordinal regression is often used to exploit the underlying ranking structure [39]. We therefore compare a standard 4-way classification model with an ordinal-regression model defined over the same 4 ordinal levels. As shown in Table 1, the ordinal regression variant underperforms the classification model across all metrics, despite using richer ranking information during training. In contrast, the classification architecture yields stronger empirical performance and is easier to calibrate and deploy. Based on this combination of accuracy and practical simplicity, we use the standard classification formulation in the final ARBITRAGE ROUTER.

##### 6.2 Data-related Design Choices

###### Step Annotation

Another design choice is whether to incorporate explicit signals about the reasoning trajectory at each step. A baseline approach treats each step independently, predicting the routing decision from the local step context alone. Our hypothesis is that previous routing decisions provide valuable context: if the target model has been invoked repeatedly in earlier steps, the current step is more likely to belong to a difficult portion of the problem and thus may benefit from escalation. To test this hypothesis, we introduce a simple annotation scheme that encodes the history of model

- Table 1: Ablation results for different model architectures, varying label granularity and including an ordinal regression variant. The standard 2-class formulation (accept vs. escalate) attains the best trade-off between Spearman correlation and per-label accuracy, indicating that binary routing is the most robust choice.

Variant Spearman (ρ) Acc 0 Acc 1 2-Classification 0.1508 70.04 72.96 4-Classification 0.1417 64.24 80.53 4-Class Ordinal 0.1355 62.00 79.77 10-Classification 0.1337 71.29 71.77

choices. For each step i, we prepend a tag such as Model 0 or Model 1 to the step input, indicating which model was used at that step. This produces contextualized inputs whose annotations summarize both the overall difficulty of the question and the local difficulty pattern along the chain of thought. Table 2 shows that including these annotations improves label-1 accuracy (correctly escalating when the target model is beneficial) and increases Spearman correlation, which correlates with end-to-end task accuracy. Because these gains are consistent and the mechanism is simple to implement, we enable step annotations by default in ARBITRAGE ROUTER.

- Table 2: Effect of incorporating historical routing context. The annotated setting, which conditions each step on prior routing decisions, improves both Spearman correlation and label-1 accuracy, indicating that leveraging routing history enhances end-to-end routing performance.

Variant Spearman (ρ) Acc 0 Acc 1 Annotated 0.1508 70.04 72.96 Not Annotated 0.1305 73.06 69.58

Data Downsampling

- As discussed in Section 4.2, the training data for ARBITRAGE ROUTER is highly imbalanced: negative examples (steps where the draft is accepted, y = 0) occur roughly twice as often as positive examples (steps where escalation is preferred, y = 1). This reflects the fact that most steps are sufficiently handled by the draft model, with only a minority requiring escalation. We evaluate two simple preprocessing strategies to address this imbalance: (1) No downsampling, where we train directly on the raw, imbalanced dataset; and (2) Balanced downsampling, where we randomly subsample accepted steps (y = 0) so that the counts of y = 0 and y = 1 are approximately equal. Results are reported in Table 3. Without downsampling, the router becomes overconfident in predicting yˆ = 0 and assigns very low probability to yˆ = 1, leading to under-escalation and suboptimal performance. With balanced downsampling, even though we train on fewer total examples, the model exhibits better calibration and produces more balanced predictions over the two classes. In our final training pipeline, we therefore apply class-balanced downsampling to mitigate label skew and improve routing quality.

- Table 3: Effect of class-balanced downsampling. Balancing the training data reduces bias toward the majority (accept) class, improving Spearman correlation and label-1 accuracy and leading to more stable routing behavior.

Variant Spearman (ρ) Acc 0 Acc 1 Balanced DS 0.1673 63.28 64.80 No DS 0.1587 77.34 49.62

#### 7 Conclusions

We study step-level speculative generation as a paradigm for accelerating LLM inference on reasoning-intensive tasks, and we identify a key source of inefficiency in existing approaches. In current methods, when a draft step is rejected

and recomputed by the target model, the regenerated step often provides limited or no improvement, while still incurring the full latency cost of the target model. To address this, we introduce ARBITRAGE, a speculative generation framework that explicitly reasons about the relative quality of draft and target steps. ARBITRAGE has two components: (1) the ARBITRAGE ORACLE, an idealized routing policy that selects the better of the two steps using ground-truth advantage signals; and (2) the ARBITRAGE ROUTER, a lightweight, trainable predictor that approximates this oracle using only draft-generated context, enabling practical deployment without querying the target model during routing. Across multiple mathematical reasoning benchmarks, ARBITRAGE consistently improves over prior baselines in both inference speed and answer quality. For example, on MATH500 with a quantized-draft regime (Q4-8B/8B), ARBITRAGE achieves up to 1.62× lower latency at comparable accuracy, and on OlympiadBench with a small-draft regime (1B/8B), it reaches up to 1.97× speedup at matched accuracy. More broadly, ARBITRAGE reduces end-to-end latency by up to roughly 2× over step-level speculative decoding baselines at fixed accuracy targets. By replacing absolute, draft-only acceptance rules with an expected-advantage estimate, ARBITRAGE achieves a stronger efficiency–accuracy trade-off and establishes a new baseline for step-level speculative decoding.

#### Acknowledgements

We acknowledge gracious support from the Furiosa AI, Intel, Apple, NVIDIA, Macronix, and Mozilla team. Furthermore, we appreciate support from Google Cloud, the Google TRC team and Prof. David Patterson. Prof. Keutzer’s lab is sponsored by the Intel corporation, UC Berkeley oneAPI Center of Excellence, Intel VLAB team, as well as funding through BDD and BAIR. We also acknowledge support by the Director, Office of Science, Office of Advanced Scientific Computing Research, of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231. MWM would also like to acknowledge DARPA, DOE, NSF, and ONR. DOE SciGPT grant. Our conclusions do not necessarily reflect the position or the policy of our sponsors, and no official endorsement should be inferred.

#### References

- [1] Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon. Hydra: Sequentially-dependent draft heads for medusa decoding, 2024.
- [2] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024.
- [3] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling, 2023.
- [4] Etash Guha et. al. Openthoughts: Data recipes for reasoning models, 2025.
- [5] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers, 2023.
- [6] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-math: A universal olympiad level mathematic benchmark for large language models, 2024.
- [7] Amir Gholami, Zhewei Yao, Sehoon Kim, Coleman Hooper, Michael W Mahoney, and Kurt Keutzer. Ai and memory wall. IEEE Micro, 44(3):33–39, 2024.
- [8] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [9] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

- [10] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- [11] Jujie He, Tianwen Wei, Rui Yan, Jiacai Liu, Chaojie Wang, Yimeng Gan, Shiwen Tu, Chris Yuhao Liu, Liang Zeng, Xiaokun Wang, Boyang Wang, Yongcong Li, Fuxiang Zhang, Jiacheng Xu, Bo An, Yang Liu, and Yahui Zhou. Skywork-o1 open series, November 2024.
- [12] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [13] Yuezhou Hu, Jiaxin Guo, Xinyu Feng, and Tuo Zhao. Adaspec: Selective knowledge distillation for efficient speculative decoders, 2025.
- [14] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024.
- [15] Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Sai Surya Duvvuri, Manzil Zaheer, Inderjit S Dhillon, David Brandfonbrener, and Rishabh Agarwal. The art of scaling reinforcement learning compute for llms. arXiv preprint arXiv:2510.13786, 2025.
- [16] Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael W. Mahoney, Amir Gholami, and Kurt Keutzer. Speculative decoding with big little decoder, 2023.
- [17] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding, 2023.
- [18] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-CoT](https: //github.com/project-numina/aimo-progress-prize/blob/main/report/numina_ dataset.pdf), 2024.
- [19] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test, 2025.
- [20] Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-guided speculative decoding for efficient llm reasoning, 2025.
- [21] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023.
- [22] Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms, 2024.
- [23] Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Online speculative decoding, 2024.
- [24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.
- [25] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025.
- [26] Open R1 Project. Openr1-math-220k, 2025.
- [27] Rui Pan, Yinwei Dai, Zhihao Zhang, Gabriele Oliaro, Zhihao Jia, and Ravi Netravali. Specreason: Fast and accurate inference-time compute via speculative reasoning. arXiv preprint arXiv:2504.07891, 2025.

- [28] Bo Pang, Hanze Dong, Jiacheng Xu, Silvio Savarese, Yingbo Zhou, and Caiming Xiong. Bolt: Bootstrap long chain-of-thought in language models without distillation, 2025.
- [29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.
- [30] Carlos Silla and Alex Freitas. A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22:31–72, 01 2011.
- [31] Skywork Team. Skywork-o1 open series. https://huggingface.co/Skywork, 2024.
- [32] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024.
- [33] Haoxiang Sun, Yingqian Min, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. Challenging the boundaries of reasoning: An olympiad-level math benchmark for large language models, 2025.
- [34] DeepSeek-AI Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [35] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
- [36] Rishabh Tiwari, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. Quantspec: Self-speculative decoding with hierarchical quantized kv cache, 2025.
- [37] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback, 2022.
- [38] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.
- [39] Jinhong Wang, Jintai Chen, Jian Liu, Dongqi Tang, Danny Z. Chen, and Jian Wu. A survey on ordinal regression: Applications, advances and prospects, 2025.
- [40] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Mathshepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024.
- [41] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023.
- [42] Yuancheng Xu, Udari Madhushani Sehwag, Alec Koppel, Sicheng Zhu, Bang An, Furong Huang, and Sumitra Ganesh. Genarm: Reward guided generation with autoregressive reward model for test-time alignment, 2025.
- [43] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.
- [44] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

- [45] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement, 2024.
- [46] Wang Yang, Xiang Yue, Vipin Chaudhary, and Xiaotian Han. Speculative thinking: Enhancing small-model reasoning with large model guidance at inference time, 2025.
- [47] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms, 2025.
- [48] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025.
- [49] Jerrold Zar. Spearman Rank Correlation, volume 5. 07 2005.
- [50] Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, and Wanli Ouyang. Accessing gpt-4 level mathematical olympiad solutions via monte carlo tree self-refine with llama-3 8b, 2024.
- [51] Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian, Biqing Qi, Xiu Li, and Bowen Zhou. Genprm: Scaling test-time compute of process reward models via generative reasoning, 2025.
- [52] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025.
- [53] Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, and Eric P. Xing. Megamath: Pushing the limits of open math corpora, 2025.
- [54] Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agarwal. Distillspec: Improving speculative decoding via knowledge distillation, 2024.
- [55] Banghua Zhu, Evan Frick, Tianhao Wu, Hanlin Zhu, Karthik Ganesan, Wei-Lin Chiang, Jian Zhang, and Jiantao Jiao. Starling-7b: Improving helpfulness and harmlessness with RLAIF. In First Conference on Language Modeling, 2024.
- [56] Jiachen Zhu, Congmin Zheng, Jianghao Lin, Kounianhua Du, Ying Wen, Yong Yu, Jun Wang, and Weinan Zhang. Retrieval-augmented process reward model for generalizable mathematical reasoning, 2025.
- [57] Lei Zhu. Boosting image classification accuracy leveraging finer grained labels. pages 42–45, 11 2023.
- [58] Yaoming Zhu, Junxin Wang, Yiyang Li, Lin Qiu, ZongYu Wang, Jun Xu, Xuezhi Cao, Yuhuai Wei, Mingshi Wang, Xunliang Cai, and Rong Ma. Oibench: Benchmarking strong reasoning models with olympiad in informatics, 2025.

#### A Analyzing Compute Wastage in RSD

Here we present a detailed analysis of RSD rejections; see Figure A.1. The figure characterizes how RSD’s rejection behavior varies with the draft-score threshold τ on sd. Among rejected steps, the probability that the target improves the PRM score, Pr(step quality improves | step rejected by draft, τ), stabilizes at roughly 0.4 across a wide range of τ (Figure A.1a). This means that in approximately 60% of target invocations there is no gain (the step is equal or worse). At the same time, the reject rate Pr(sd ≤ τ) increases sharply with τ (Figure A.1b), so absolute-score thresholding rapidly saturates the system with expensive target calls without increasing the likelihood of improvement.

We capture the resulting compute inefficiency via an explicit “wasted target calls” metric, derived from our wastedcomputation formalization:

reject_rate(τ) · 1 − Pr(step quality improves | step rejected by draft, τ) ,

which grows monotonically with τ (Figure A.1c). Taken together, these trends indicate that when the draft fails due to shared error modes with the target (e.g., hard reasoning instances), regenerating with the target often cannot salvage the step. Thus, RSD’s absolute-score rejection pays full target cost while delivering little to no quality benefit, motivating routing policies that depend on the relative advantage (st − sd) rather than thresholding sd in isolation.

#### B Optimality of ARBITRAGE ORACLE

Theorem B.1: ARBITRAGE ORACLE optimality under Budgeted Escalation

Given a budget B ∈ {0, . . . , N} on the number of escalations, consider policies a = (a1, . . . , aN) ∈ {0,1}N with ∑iN=1 ai ≤ B, where ai = 1 means “escalate to target” and ai = 0 means “accept draft.” Then there exists a threshold τ ∈ R such that the policy a⋆τ with a⋆τ,i = I{∆i > τ} attains

N

∑

sd,i + ai∆i .

max

S(a) = max

a: ∑ ai≤B

a: ∑ ai≤B

i=1

Because ∑i sd,i is independent of a, this problem reduces to

N

N

∑

∑

ai∆i s.t.

max

ai ≤ B. (⋆)

a∈{0,1}N

i=1

i=1

Proof. Suppose ∆j < 0 and aj = 1; then setting aj ← 0 strictly increases the objective while preserving feasibility, since the constraint is an upper bound. Thus some optimal solution has aj = 0 for all j with ∆j < 0. Now, among indices with ∆i > 0, let a be any feasible solution that includes j but excludes k with ∆k > ∆j > 0. Swapping aj ← 0 and ak ← 1 increases the objective by ∆k − ∆j > 0 and keeps feasibility, contradicting optimality. Repeating exchanges yields an optimal solution that selects the k = min{B, #{i : ∆i > 0}} largest positive ∆i’s.

In the constrained case, order ∆(1) ≥ · · · ≥ ∆(N) and choose τ with ∆(k) > τ ≥ ∆(k+1) (using ∆(0) := +∞,

∆(N+1) := −∞). Then a⋆τ,i = I{∆i > τ} selects exactly those k indices; ties at ∆i = τ can be broken arbitrarily to satisfy the budget. This achieves the maximum of (⋆), hence of S(a).

| |
|---|

- Corollary B.2 (Pareto Frontier). Let C(a) ∈ R+ denote computational cost (e.g., # target calls = ∑i ai). As τ varies,

the threshold policies a⋆τ trace a cost–quality curve C(a⋆τ), S(a⋆τ) that majorizes any other policy’s cost–quality pair: for any target cost c,

S(a⋆τ(c)) = sup

S(π),

π: C(π)≤c

i.e., the oracle threshold family achieves the Pareto frontier of cost versus quality.

Proof. Fix any cost budget c corresponding to some integer B. By Theorem B.1, the optimal value among policies with ∑i ai ≤ B is attained by a⋆τ for a threshold τ that selects the B largest ∆i’s. Thus no policy of cost at most c can exceed S(a⋆τ), establishing the frontier property.

| |
|---|

- Corollary B.3 (Zero Wasted Computation under Unconstrained Oracle). In the unconstrained case, the oracle uses ai⋆ = I{∆i > 0}. Then the expected wasted computation, target calls that fail to improve quality, is

###### E 1{ai⋆ = 1} 1{st,i ≤ sd,i} = 0.

Proof. By definition, ai⋆ = 1 only if ∆i > 0, which implies st,i > sd,i. Hence 1{ai⋆ = 1} 1{st,i ≤ sd,i} = 0 and the expectation is zero.

| |
|---|

Remark B.4 (ARBITRAGE ORACLE Assumption). The theorem and corollaries optimize the sum of step-local PRM scores, assuming a decision at step i does not affect future steps’ scores.

#### C Implementation Details

Here, we include the training hyperparameters for the router; see Table C.1.

Table C.1: Experimental hyperparameters. Hyperparameter Value Batch size 1024 Learning rate 1e-5 Epochs 3

#### D ARBITRAGE Speculative Generation

The algorithm 1 details the inference procedure for ARBITRAGE, our step-level speculative generation framework. Given a question q, the algorithm iteratively constructs a reasoning trace by alternating between draft generation and

dynamic routing decisions. At each iteration, the draft model qθd auto-regressively generates a candidate reasoning step zd until it emits a step separator token ⟨sep⟩ (e.g., \n\n) or the end-of-sequence token ⟨eos⟩. This step is then fed, along with the current context x, into the lightweight ARBITRAGE ROUTER hθrouter, which outputs a scalar yˆ estimating the likelihood that the target model produces a significantly better step. The router’s prediction is compared with a tunable threshold τ: if yˆ ≤ τ, the draft step zd is deemed sufficient and attached to the context; otherwise, the system falls back to the target model qθt, which regenerates the step from the same prefix x, yielding zt. ARBITRAGE minimizes redundant computation while preserving (or even enhancing) output correctness. The entire process is fully compatible with standard auto-regressive decoding pipelines and introduces minimal overhead due to the lightweight architecture of the router.

#### E Additional Results

We provide additional evaluation results that complement Section 5.2; see Figure E.1.

Algorithm 1: ARBITRAGE Inference

Input: question q; draft model qθd; target model qθt; router hθrouter; threshold τ; step separator ⟨sep⟩;

end-of-sequence token ⟨eos⟩; maximum number of steps N Output: final decoded sequence x x ← q

// Initialize context with the question for t ← 1 to N do

if last token of x is ⟨eos⟩ then

###### break

// finished early

// Draft step Sample draft step zd ∼ qθd(· | x) until the first of ⟨sep⟩ or ⟨eos⟩ // Routing decision yˆ ← hθrouter(x,zd) if yˆ ≤ τ then

z⋆ ← zd

// accept draft else

Sample target step zt ∼ qθt(· | x) until the first of ⟨sep⟩ or ⟨eos⟩ z⋆ ← zt

// escalate to target

// Append chosen step x ← x ∥ z⋆ if last token of x is ⟨eos⟩ then

###### break return x

Target outcomes among rejected cases vs threshold ( )

100%

| | | | | | | | | | | | | | | | | | | | | | |P(|ste|p|qu|al|ity|im|p|rov|es|||st|ep|re|je|cte|d|by|d|ra|ft)| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |P(|ste|p|qu|al|ity|eq|u|al|| s|te|p r|eje|ct|ed|b|y|dra|ft|)| | |
| | | | | | | | | | | | | | | | | | | | | | |P(|ste|p|qu|al|ity|w|or|se|| s|te|p|rej|ec|te|d|by|dr|aft|)| | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

80%

60%

Probability

40%

20%

0%

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

(threshold on sd)

(a) Outcomes among rejected cases vs. τ

Reject rate vs threshold ( )

100%

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

80%

60%

Rejectrate

40%

20%

0%

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

(b) Reject rate vs. τ

Critical Limitation: wasted target calls under absolute-score rejection

100%

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

80%

Relativewastedtargetcalls

60%

40%

20%

0%

0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

(c) Wasted target calls vs. τ

Figure A.1: Performance of the RSD method as a function of threshold τ.

Llama3 (1B/8B) | Math500

50

| |
|---|

45

Accuracy(%)

| |
|---|

40

Arbitrage Oracle Arbitrage Router RSD

35

| |
|---|

Target Accuracy

30

| |
|---|

Draft Accuracy

0 20 40 60 80 100

Acceptance Rate (%)

Llama3 (8B/70B) | Math500

75

| |
|---|

70

Accuracy(%)

| |
|---|

65

60

Arbitrage Oracle Arbitrage Router RSD

55

Target Accuracy

| |
|---|

Draft Accuracy

| |
|---|

50

0 20 40 60 80 100

Acceptance Rate (%)

Llama3 (4bit-8B/8B) | Math500

52

| |
|---|

Accuracy(%)

48

| |
|---|

Arbitrage Oracle Arbitrage Router RSD

44

Target Accuracy

| |
|---|

Draft Accuracy

40

0 20 40 60 80 100

Acceptance Rate (%)

Qwen2.5-Math (3bit-7B/7B) | Math500

84

| |
|---|

82

| |
|---|

Accuracy(%)

| |
|---|

80

Arbitrage Oracle Arbitrage Router RSD

78

| |
|---|

Target Accuracy

Draft Accuracy

76

0 20 40 60 80 100

Acceptance Rate (%)

Llama3 (1B/8B) | OlympiadBench

| |
|---|

16

| |
|---|

Accuracy(%)

12

| |
|---|

| |
|---|

8

4

0 20 40 60 80 100

Acceptance Rate (%)

Llama3 (8B/70B) | OlympiadBench

40

35

Accuracy(%)

| |
|---|

| |
|---|

30

| |
|---|

25

20

0 20 40 60 80 100

Acceptance Rate (%)

Llama3 (4bit-8B/8B) | OlympiadBench

- 12

- 13

- 14

- 15

- 16

- 17

Accuracy(%)

| |
|---|

0 20 40 60 80 100

Acceptance Rate (%)

Qwen2.5-Math (3bit-7B/7B) | OlympiadBench

40

38

Accuracy(%)

36

| |
|---|

| |
|---|

34

32

0 20 40 60 80 100

Acceptance Rate (%)

Figure E.1: Additional compute–quality results. Accuracy versus acceptance rate for ARBITRAGE ORACLE, ARBITRAGE ROUTER, and RSD on Math500 (left column) and OlympiadBench (right column) across the LLaMA3 and Qwen2.5-Math model families. In all settings, ARBITRAGE achieves higher accuracy for a given acceptance rate, indicating a more favorable compute–quality trade-off.

