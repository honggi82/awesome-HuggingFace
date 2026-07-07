# arXiv:2605.15529v1[cs.CL]15May2026

## Process Rewards with Learned Reliability

Jinyuan Li1, Langlin Huang1, Chengsong Huang1, Shaoyang Xu2, Donghong Cai1, Yuyi Yang1, Wenxuan Zhang2, Jiaxin Huang1∗

1Washington University in St. Louis 2Singapore University of Technology and Design {ljinyuan,jiaxinh}@wustl.edu

### Abstract

Process Reward Models (PRMs) provide step-level feedback for reasoning, but current PRMs usually output only a single reward score for each step. Downstream methods must therefore treat imperfect step-level reward predictions as reliable decision signals, with no indication of when these predictions should be trusted. We propose BETAPRM, a distributional PRM that predicts both a step-level success probability and the reliability of that prediction. Given step-success supervision from Monte Carlo continuations, BETAPRM learns a Beta belief that explains the observed number of successful continuations through a Beta-Binomial likelihood, rather than regressing to the finite-sample success ratio as a point target. This learned reliability signal indicates when a step reward should be trusted, enabling downstream applications to distinguish reliable rewards from uncertain ones. As one application, we introduce Adaptive Computation Allocation (ACA) for PRMguided Best-of-N reasoning. ACA uses the learned reliability signal to stop when a high-reward solution is reliable and to spend additional computation on uncertain candidate prefixes. Experiments across four backbones and four reasoning benchmarks show that BETAPRM improves PRM-guided Best-of-N selection while preserving standard step-level error detection. Built on this signal, ACA improves the accuracy–token tradeoff over fixed-budget Best-of-16, reducing token usage by up to 33.57% while improving final-answer accuracy.

Code: https://github.com/JinyuanLi0012/Beta-Binomial-PRM

### 1 Introduction

Process Reward Models (PRMs) [14, 19, 31, 41, 43, 59, 61, 72, 73] provide step-level feedback for reasoning by scoring the intermediate steps of a solution. Because these step-level scores can guide candidate selection [5, 21, 35] and policy optimization [12, 36], PRMs have become a useful interface for both test-time scaling [2, 10, 30] and reinforcement learning [34, 70]. However, existing PRMs typically expose this interface as a single point estimate of step correctness, such as the probability that a step is correct. Downstream methods [17, 42, 71] often have to treat this imperfect score as a reliable decision signal, because no additional signal is available. A single PRM score tells us which step or candidate the model prefers, but not whether that preference should be trusted. As a result, an unreliable score can directly affect downstream decisions without being identified as uncertain.

As shown in Fig. 1, this classic interface mismatches both test-time usage and training supervision: First, a single scalar reward cannot capture the predictive uncertainty of intermediate steps. At inference time, a causal PRM judges a step from the problem and current prefix, without seeing future continuations [54, 57, 48]. Even when no local error is obvious, it is uncertain whether a seemingly correct prefix will lead to a correct final answer. A more natural PRM output should capture both the estimated probability of success and the uncertainty of that estimate.

∗Corresponding author.

Preprint.

[Figure 1]

Monte Carlo supervision

Reasoning Prefix

from the same prefix Standard PRM

###### BetaPRM

[Figure 2]

Point-target Regression

###### Beta Belief over Prefix Success

[Figure 3]

- Run 1
- Run 2

5/8

High Kappa

###### Low Kappa

- Step 1

- Step 2

- Step 3

[Figure 4]

Run 1 Run 2 Run 3 3/8 4/8 5/8

- Run 1
- Run 2

4/8

Run 2 Run 3

Run 1

Run 3

0 Prefix Success Rate 1 Monte Carlo Variation Becomes Label Variation

###### ...

...

0 Prefix Success Rate 1

0 Prefix Success Rate 1

Model Success Rate as a Beta Distribution Learned Concentration Captures Uncertainty

Run n

3/8

- Figure 1: Motivation of BETAPRM. Repeated Monte Carlo continuations from the same prefix can produce different empirical success ratios. Standard PRMs treat these ratios as point targets, whereas BETAPRM models the prefix success probability as a Beta belief. The Beta mean µ gives the process reward, while the concentration κ captures the reliability of the estimate, allowing the model to assign likelihood to the observed count K out of N rather than treating K/N as an exact point label.

Second, step-level PRM labels are often noisy finite-sample estimates. A common source of supervision [61, 63, 65, 72] samples N continuations from a reasoning prefix and counts how many reach the correct final answer. If K continuations succeed, the empirical ratio K/N is only a Monte Carlo estimate of the prefix success probability, not the true underlying probability. Repeating the procedure from the same prefix could yield a different K due to sampling randomness. Standard PRM training [13, 14, 31] nevertheless regresses to this observed ratio as a point label, forcing the model to fit a noisy finite-sample outcome with a single scalar prediction. A better objective should keep the supervision in counting form: the model should assign high probability to observing K successes out of N continuations, rather than only regress to the single ratio K/N.

In this paper, we address both limitations by giving the PRM a way to express uncertainty about its own prediction. A step-level reward supported by a confident belief should not be treated the same as one produced under ambiguity. This motivates BETAPRM, a distributional PRM that predicts both how promising a reasoning prefix is and how reliable that prediction is. As illustrated in Figure 2, BETAPRM predicts a Beta distribution over the prefix success probability, and is trained so that this distribution can explain the Monte Carlo observations from sampled continuations. This distribution is parameterized by (1) the predicted success probability µ, which serves as the usual PRM score, and (2) the concentration κ, which controls how tightly the belief is centered around that prediction. High concentration gives a sharp belief, while low concentration gives a flattened belief that can explain a wider range of Monte Carlo observations.

The learned concentration changes how PRM scores can be used. Rather than treating every scalar reward as equally trustworthy, downstream algorithms can distinguish confident rewards from uncertain ones. It is broadly useful for PRM-guided decision making; in this paper, we demonstrate one concrete test-time use case: Adaptive Computation Allocation (ACA) for Best-of-N reasoning. Fixed-budget Best-of-N [11, 33] spends the same rollout budget on every problem, even when the current pool already contains a high-scoring candidate whose PRM judgment is reliable. ACA spends the budget through progressive batches: it stops when the selected answer is reliably ahead, and otherwise continues from uncertain prefixes where more computation may change the decision.

Empirically, BETAPRM improves PRM-guided Best-of-N selection across four backbones and four benchmarks (e.g., +3.37 points on average on InternVL2.5-8B), while preserving standard step-level error detection ability. Further analyses show that the learned concentration provides a nontrivial reliability signal. Built on this reliability signal, ACA improves the inference-time accuracy-token tradeoff compared with vanilla Best-of-16, where it reduces token usage by up to 33.57% and even pushes final-answer accuracy higher.

### 2 Related Work

Process Reward Models. PRMs [13, 56, 47, 29] provide step-level feedback for reasoning, unlike outcome reward models [11, 67] that score only final answers. Prior work trains PRMs either as step judges for local error detection [63, 14], or as Q-value-style models that estimate whether a prefix can be completed correctly [13, 31]. We focus on a limitation of the latter view: Monte Carlo continuations provide finite-sample evidence about prefix success, yet existing methods often collapse

this evidence into a single point label. Our approach instead makes reliability part of the PRM output, so downstream methods can use not only the predicted reward but also how trustworthy it is.

Test-Time Scaling. Test-time scaling [22, 53, 3, 58, 66] improves reasoning by spending more inference compute, including voting [64], verifier-guided selection [74], and search over reasoning paths [17]. A common and simple instance is Best-of-N [33]: sample multiple candidate solutions and select one using a verifier or reward model. Most Best-of-N methods use a fixed budget [3], allocating the same number of samples to every problem despite large variation in difficulty. Recent methods [49] calibrate PRM success estimates to choose instance-specific budgets for sampling complete solutions. In contrast, our method uses BETAPRM’s reward and learned reliability during generation to decide when to stop and which uncertain prefix to continue.

### 3 Preliminaries

- 3.1 Prefix-Conditioned Process Rewards

Given an input problem x, let s1:T = (s1,...,sT) denote a step-by-step solution. We insert a special process marker <prm> after each step, and the PRM produces a score at each marker position:

x,s1,<prm>,s2,<prm>,...,sT,<prm>.

Since the reward model is a causal language model, the score at the t-th marker is computed from the prefix ct = (x,s≤t), without access to future steps st+1:T. This matches the online use of PRMs in generation or search, where a partial reasoning state is evaluated before its continuation is observed.

We therefore interpret process rewards as prefix-level quantities. Instead of assigning an isolated correctness label to step t, we define its quality as the prefix success probability qt = Pr(final answer is correct | x,s≤t). Since qt is a latent variable, the next subsection describes how finite continuation samples provide supervision to learn this variable.

- 3.2 Monte Carlo Step Supervision

The prefix success probability qt is an unobserved latent variable. A widely used way to construct step-level supervision is to sample N continuations from a prefix ct = (x,s≤t) and count how many reach the correct final answer. Let Kt denote the number of successful continuations. The empirical ratio qˆt = Kt/N is a Monte Carlo estimate of qt.

Standard PRM objectives [13, 31, 61, 62, 65, 72] often reduce this observation to a single point target by optimizing cross-entropy against qˆt:

LCE = −qˆt log pt − (1 − qˆt)log(1 − pt),

where pt is the predicted step score. This treats the empirical ratio as if it were the latent prefix success probability itself. Because qˆt is computed from a small number of continuations, repeating the same procedure could produce a different Kt. Thus, forcing the model to learn the single point estimate qˆt might lead to overfitting to sample noise. Instead, it is more natural to treat the supervision as a count observation (Kt success out of N trials).

### 4 BETAPRM

#### 4.1 Beta-Binomial Count Model

To formalize the count-based supervision, we assume a binomial generative process for the successful continuations: Kt | qt ∼ Binomial(N,qt). Because qt is an unknown latent success probability in [0,1], we model it with a Beta belief, qt ∼ Beta(αt,βt), which naturally pairs with the Binomial count observation above. For better interpretability, we reparameterize the Beta distribution by its mean µt = αt/(αt + βt) and concentration κt = αt + βt. Under this formulation, µt acts as the expected success probability (the standard PRM output score), while κt controls how sharply the belief is concentrated around that mean. Marginalizing out the latent qt yields a Beta-Binomial distribution over Kt, providing a likelihood for count observations rather than a point target for qˆt.

[Figure 5]

- Figure 2: Intuition of Beta-Binomial supervision. A predicted Beta belief over prefix success induces a distribution over possible observed success ratios K/N. The green curve is concentrated and aligned with the observed count, the orange curve is concentrated but misaligned and thus penalized, and the gray curve has lower concentration and allows a wider range of finite-sample observations.

#### 4.2 BETAPRM Parameterization

BETAPRM instantiates the Beta belief by predicting its mean and concentration at each process marker. At the t-th <prm> marker, the language model produces a hidden state ht and vocabulary logits zt. Let ztYes and ztNo denote the logits of the two reward tokens Yes and No. We define the predicted success probability by applying a softmax only over these two logits:

exp(ztYes) exp(ztYes) + exp(ztNo)

µt =

.

This preserves the standard PRM interpretation of the Yes probability as the scalar reward. To estimate reliability, BETAPRM predicts a separate concentration parameter κt:

κt = softplus(gϕ(ht)) + κmin,

where gϕ is a lightweight linear head and κmin is a small fixed lower bound for numerical stability. This separates the reward from the reliability channel: the reward-token logits determine µt, while the additional head determines how concentrated the model’s belief should be.

The Beta parameters are then derived using αt = µtκt and βt = (1 − µt)κt. Here µt centers the belief over prefix success and serves as the scalar PRM score, while κt controls the concentration, allowing prefixes with similar scores to carry different reliability estimates.

#### 4.3 Beta-Binomial Training Objective

We train the predicted Beta belief by maximizing the likelihood of the observed count Kt. As shown in Figure 2, a concentrated belief centered near the observed ratio assigns high probability to the count, while a concentrated but misaligned belief receives a large loss. A lower-concentration belief spreads probability mass over a wider range of possible finite-sample observations, reflecting lower confidence.

Using the Beta-Binomial formulation, the predictive probability of the observed count is

p(Kt | N,αt,βt) =

N Kt

B(Kt + αt,N − Kt + βt) B(αt,βt)

,

where B(·,·) is the Beta function. Let P be the set of supervised process markers in a mini-batch. We define the Beta-Binomial loss, LBeta-Binomial, as the negative log-likelihood of the observed counts:

1 |P| t∈P

LBeta-Binomial = −

log p(Kt | N,αt,βt).

Minimizing this loss encourages the model to assign high probability to the observed count. We add an auxiliary regularization loss to explicitly encourage calibrated reliability estimates. If µt disagrees with the observed ratio Kt/N, it contradicts with a large κt that indicates high confidence.

###### Early Stopping Evaluation

###### Progressive Batch Generation

LCB UCB

Start with a small candidate pool

- Y1

Adaptive Allocation

[Figure 6]

Y2 Step 1 Step 2 Step 3 Step 4

Reliable Winner

Y1 Step 1 Step 2 Step 4

Repair Uncertain Prefix Step 3

[Figure 7]

Step 3 Step 4 Step 5

[Figure 8]

Y3

- Y2 Overlap: Keep Search

- Y1

Y3

- Y2

Step 1 Step 2 Step 3 Step 4

Step 3 Step 4 Step 5 Step 3 Step 4 Step 5

Step 1 Step 2 Step 3 Step 4

###### Y1

Disjoint: Stop Early

###### Y2 gap

Step 1 Step 2 Step 3 Step 4

Y3

Budget

Low Candidate Score High

- Figure 3: Overview of Adaptive Computation Allocation (ACA). ACA generates candidates in batches, uses BETAPRM scores and reliability estimates to test whether the current winner remains reliably ahead, and otherwise samples new continuations from uncertain prefixes. We therefore penalize the product of disagreement and concentration:

Kt N

1 |P| t∈P

sg(µt) −

Lreg = λreg

κt,

where sg(·) denotes the stop-gradient operation. The stop-gradient operation prevents this auxiliary term from pulling µt toward the noisy ratio, which would make it another point-label regression loss. Instead, it mainly calibrates the concentration parameter: high κt is discouraged when µt disagrees with the count evidence, and encouraged when they are consistent.

The overall training objective is

L = LBeta-Binomial + Lreg.

### 5 Reliability-Aware Inference: Adaptive Computation Allocation

BETAPRM outputs both a reward mean and a reliability estimate. As shown in Figure 3, we study a straightforward inference-time use case: allocating computation in PRM-guided Best-of-N reasoning. In standard practices [11, 33], Best-of-N improves inference by sampling multiple candidate solutions and selecting one according to a scoring rule, which can be a process reward model. In addition, every query receives the same number of sampled rollouts. We introduce Adaptive Computation Allocation (ACA) that saves computation when the current sampled pool may already contain a high-scoring answer. ACA utilizes BETAPRM to estimate uncertainty and mainly works by two logic: (1) stop early when a reliable answer is found, and (2) redirect computation for uncertain prefixes.

Risk-Adjusted Candidate Score. ACA compares complete candidates using both reward and reliability. We convert the Beta belief into a step-level uncertainty, σt = µt(1 − µt)/(κt + 1), the standard deviation of the predicted Beta distribution. Larger κt gives smaller σt, indicating a more reliable reward estimate. We then define a risk-adjusted step score rt = µt − λσt, where λ controls the uncertainty penalty, and aggregate into a candidate-level uncertainty for y = s1:T as

1 T

S(y) =

T

(µt − λσt).

t=1

Thus, candidates are ranked by predicted process quality discounted by uncertainty.

Progressive Batch Generation and Early Stopping. Standard Best-of-N generates all N candidates in one shot. ACA instead spends the budget in a progressive way: it first samples a small pool of n0 candidates, scores them with BETAPRM, and then either stops or allocates another batch, up to the maximum budget N.

At each stage, ACA selects the highest-scoring candidate y⋆ = arg maxy S(y) for the stopping test, where we construct lower and upper confidence bounds (LCB and UCB):

T

1 T

LCB(y) = S(y) − cstopU(y), UCB(y) = S(y) + cstopU(y), U(y) =

σt,

t=1

where cstop scales the width of the confidence bounds. ACA terminates the allocation process for the current problem and returns y⋆ if

LCB(y⋆) > max y̸=y⋆

UCB(y).

This criterion means that the highest-scoring candidate dominates the current pool: even its pessimistic score exceeds the optimistic score of every competitor. In this case, further expanding the pool with additional continuations is unlikely to change the PRM-guided selection.

Uncertainty-Guided Prefix Repair. If the stopping criterion is not met, ACA spends the next batch on a competitive existing response, chosen as the non-winner candidate with the highest UCB, where additional computation is most likely to change the current decision. To choose where to repair this response, ACA uses a deterministic cutpoint rule over reasoning steps. It first computes a conservative step score µt − ccutσt and selects the earliest step whose value falls below a low-quality threshold pbad. If no such step exists, ACA falls back to the most uncertain eligible reasoning step, i.e., the step with the largest σt. The selected step is treated as a cutpoint: ACA keeps the prefix before the cutpoint, discards the subsequent generation, and samples new continuations from that prefix. The procedure repeats until the confidence condition holds or the budget N is reached.

### 6 Experiments

#### 6.1 Experimental Setup

We evaluate our proposed methods from two aspects. First, we evaluate BETAPRM as a PRM on PRM-guided Best-of-N selection and step-level error detection. Second, we evaluate whether its uncertainty estimates improve Adaptive Computation Allocation (ACA) in Best-of-N reasoning.

We train on VisualPRM400K-v1.12 [63], the available dataset that reports K successful continuations out of N = 16 Monte Carlo samples for each prefix. The standard PRM baseline is trained with cross-entropy using the empirical ratio K/N as a single-point target, while BETAPRM uses the Beta-Binomial objective on (K,N). We evaluate BETAPRM as a PRM with four backbones: InternVL2.5-8B [9], InternVL3-8B [75], InternVL3-14B [75], and Qwen2.5-VL-7B [1]. Bestof-N selection uses candidate pools generated by InternVL2.5-8B [9] and reports final-answer accuracy on MathVision [60], OlympiadBench [18], MathVerse [68], and MathVista [40]. Step-level error detection is evaluated on VisualProcessBench [63]. ACA is evaluated on two representative backbones, InternVL2.5-8B [9] and Qwen2.5-VL-7B [1], against fixed-budget Best-of-N under the same maximum budget N = 16, reporting accuracy and generated tokens. Full training, evaluation, and ACA implementation details are provided in Appendix A.

#### 6.2 BETAPRM Evaluation

BETAPRM improves Best-of-N selection across four backbones and four benchmarks. Table 1 evaluates PRMs as solution selectors under the same candidate pools. Standard PRM selects the candidate with the highest average process reward. BETAPRM exposes both a reward mean and a learned reliability estimate, so we use its full output through a risk-budget selector:

1 T

SRB(y) =

T

1 T

µt − λ

t=1

T

1[σt > τ], σt =

t=1

µt(1 − µt) κt + 1

.

It keeps the average reward term, but discounts rollouts that contain many high-uncertainty steps.

BETAPRM achieves the best accuracy in every backbone–benchmark block. Its average gains over standard PRM are +1.29, +1.46, +3.37, and +2.66 points for InternVL3-14B, InternVL3-8B, InternVL2.5-8B, and Qwen2.5-VL-7B, respectively. These gains reflect that our Beta-Binomial objective effectively learns µt to explain the success ratio Kt/N as a deterministic soft label and concentration κt to down-weight high-scoring traces whose rewards are uncertain.

2https://huggingface.co/datasets/OpenGVLab/VisualPRM400K-v1.1-Raw

- Table 1: PRM-guided Best-of-16 final-answer accuracy. All PRMs select from the same candidate pools generated by InternVL2.5-8B. Values with gray ↑ / ↓ indicate improvement/decline over the single-pass baseline, and Avg. ∆ averages this improvement/decline over the four benchmarks.

Selector MathVision OlympiadBench MathVerse MathVista Avg. ∆ Single Pass 18.08 8.65 35.31 52.77 –

InternVL3-14B

- +Base (w/o training) 19.74 ↑1.66 11.33 ↑2.68 36.17 ↑0.86 52.50 ↓0.27 ↑1.23

+Standard PRM 23.03 ↑4.95 16.67 ↑8.02 45.41 ↑10.10 60.70 ↑7.93 ↑7.75

+BETAPRM 25.66 ↑7.58 16.67 ↑8.02 46.35 ↑11.04 62.30 ↑9.53 ↑9.04 (+1.29) InternVL3-8B

+Base (w/o training) 18.75 ↑0.67 13.33 ↑4.68 37.21 ↑1.90 52.40 ↓0.37 ↑1.72 +Standard PRM 22.69 ↑4.61 15.33 ↑6.68 44.80 ↑9.49 60.00 ↑7.23 ↑7.00

- +BETAPRM 24.34 ↑6.26 18.00 ↑9.35 45.20 ↑9.89 61.10 ↑8.33 ↑8.46 (+1.46) InternVL2.5-8B

+Base (w/o training) 20.72 ↑2.64 9.33 ↑0.68 36.83 ↑1.52 51.90 ↓0.87 ↑0.99 +Standard PRM 21.38 ↑3.30 11.33 ↑2.68 42.81 ↑7.50 57.60 ↑4.83 ↑4.58

- +BETAPRM 25.66 ↑7.58 15.33 ↑6.68 44.31 ↑9.00 61.30 ↑8.53 ↑7.95 (+3.37) Qwen2.5-VL-7B

+Base (w/o training) 15.46 ↓2.62 8.00 ↓0.65 35.84 ↑0.53 50.70 ↓2.07 ↓1.20 +Standard PRM 21.38 ↑3.30 14.00 ↑5.35 44.92 ↑9.61 60.30 ↑7.53 ↑6.45 +BETAPRM 24.34 ↑6.26 17.33 ↑8.68 45.99 ↑10.68 63.60 ↑10.83 ↑9.11 (+2.66)

- Table 2: Step-level error detection on VisualProcessBench. Overall denotes micro-F1 over all annotated steps, while each source column reports macro-F1 on that subset.

Model Overall MathVision MathVerse MMMU DynaMath WeMath InternVL3-14B

Base (w/o training) 49.40 47.61 50.97 48.86 50.19 47.34 Standard PRM 61.90 59.90 61.93 63.12 62.93 63.79 BETAPRM 61.90 60.94 62.74 59.18 61.67 64.59

###### InternVL3-8B

- Base (w/o training) 48.39 47.21 48.91 47.01 49.41 49.00

- Standard PRM 60.69 60.20 60.41 58.73 61.63 63.23

- BETAPRM 61.85 59.65 62.17 62.42 62.71 64.23 InternVL2.5-8B

Base (w/o training) 52.28 52.40 52.04 50.21 54.85 49.95 Standard PRM 61.54 60.78 60.47 62.05 62.91 64.38 BETAPRM 60.97 60.43 60.70 60.48 63.00 59.98

Qwen2.5-VL-7B

Base (w/o training) 49.68 50.22 49.58 49.85 49.62 48.51 Standard PRM 62.23 62.17 61.25 61.44 62.88 65.55

- BETAPRM 62.91 62.19 62.91 59.49 63.75 66.69

BETAPRM preserves standard PRM error-detection ability. Table 2 reports results on VisualProcessBench [63], a benchmark for step-level error detection. Each reasoning trace has human step-wise correctness labels, and a PRM score is thresholded into a binary prediction of whether each step is correct or erroneous.

BETAPRM remains competitive with standard PRM under this thresholding setting. Across the evaluated backbones, its overall micro-F1 remains comparable to PRM: it matches PRM on InternVL314B, improves slightly on InternVL3-8B and Qwen2.5-VL-7B, and is slightly lower on InternVL2.58B. Together with the Best-of-16 results, this shows that Beta-Binomial training improves the relative ranking of candidate solutions without degrading the PRM’s ability to separate correct and erroneous steps under a standard decision threshold.

The auxiliary evidence regularizer improves concentration calibration. Table 3 isolates the effect of Lreg. Adding it to the Beta-Binomial likelihood improves all four Best-of-16 benchmarks, with an average gain of +1.02 points. This matches its intended role: when the predicted mean µt disagrees with the observed Monte Carlo ratio Kt/N, the regularizer penalizes high concentration. The stop-gradient is crucial here: it avoids pulling µt toward Kt/N and making the auxiliary term another soft-label regression objective. With stop-gradient, the term instead focuses on updating the

- Table 3: Ablation of the auxiliary evidence regularizer on InternVL2.5-8B under PRM-guided Best-of-16 selection. Removing Lreg consistently reduces accuracy.

Method MathVision OlympiadBench MathVerse MathVista Avg. BETAPRM 25.66 15.33 44.31 61.30 36.65 BETAPRM w/o Lreg 24.67 ↓ 0.99 14.00 ↓ 1.33 43.63 ↓ 0.68 60.20 ↓ 1.10 35.63 ↓ 1.02

[Figure 9]

- Figure 4: Training dynamics of the learned concentration κt. The mean and the 90th percentile both decrease early in training and later recover, showing that BETAPRM first becomes conservative and then learns to assign higher confidence to prefixes whose reward estimates are better supported.

concentration parameter κt. The consistent gains suggest that explicitly calibrating concentration improves the reliability signal used in candidate ranking.

BETAPRM learns adaptive confidence. Figure 4 tracks the learned concentration κt during training. Across 4 different backbones, both the mean and the 90th percentile of κt drop sharply at the beginning and then gradually increase. This is the expected behavior: early in training, the reward mean µt is still unreliable, so the model lowers its confidence instead of making sharp predictions.

- As training progresses, the model assigns higher concentration to prefixes whose predicted reward is better supported by the observed number of successful continuations.

The upper-tail behavior is also important. After the initial drop, the 90th percentile recovers more strongly than the mean and remains clearly separated from it. This suggests that the model is not simply raising κt uniformly; it forms an upper tail of prefixes with substantially higher confidence. This separation is useful for reliability-aware use: if all prefixes had similar concentration, κt would provide little guidance about which rewards are trustworthy. A high-confidence upper tail lets downstream methods treat some predictions as more strongly supported while staying conservative on ordinary or low-confidence predictions.

#### 6.3 ACA Improves the Inference-Time Accuracy-Token Tradeoff

ACA uses fewer tokens while improving Best-of-N accuracy. Table 4 compares ACA with fixed-budget PRM-guided Best-of-N. All methods use the same maximum budget of N = 16 candidate generations and the same BETAPRM risk-budget selector SRB for final selection; they differ only in how the budget is spent. Vanilla Best-of-N generates all candidates from scratch, whereas ACA spends the budget in stages and uses BETAPRM uncertainty to decide whether to stop or repair uncertain prefixes.

ACA improves the accuracy–token tradeoff across both backbones. Across both InternVL2.58B and Qwen2.5-VL-7B, ACA improves all four benchmarks, saving 16.76%–33.57% tokens on InternVL2.5-8B and 19.39%–33.00% on Qwen2.5-VL-7B. The ablation without early stopping shows that adaptive expansion alone mainly reduces computation, but can keep spending budget even after the current selected answer is already reliable, introducing additional candidates that may distract an imperfect PRM selector. The full ACA gives the strongest tradeoff by combining uncertainty-guided expansion with confidence-based stopping.

BETAPRM provides a distinct learned uncertainty signal. We investigate whether the explicit uncertainty modeling in BETAPRM is actually necessary for ACA, or if a standard PRM would suffice to acheive the same efficiency. We compare BETAPRM with ACA variants using a standard

- Table 4: ACA improves the accuracy–token tradeoff in PRM-guided Best-of-16. Token counts are reported in thousands. Percentages indicate token reduction relative to Vanilla BoN.

Method

Adaptive Allocation

Early Stopping

MathVision OlympiadBench MathVerse MathVista

Acc. ↑ Tokens ↓ Acc. ↑ Tokens ↓ Acc. ↑ Tokens ↓ Acc. ↑ Tokens ↓

InternVL2.5-8B

Vanilla BoN × × 25.00 1383k 15.33 1151k 44.47 17932k 60.90 2790k

ACAw/o EarlyStop ✓ × 24.01 1237k (↓10.56%) 15.33 1028k (↓10.69%) 42.99 14692k (↓18.07%) 60.20 2462k (↓11.76%) ACA ✓ ✓ 26.32 965k (↓30.24%) 16.67 958k (↓16.76%) 45.58 11912k (↓33.57%) 62.20 1949k (↓30.14%)

Qwen2.5-VL-7B Vanilla BoN × × 24.67 1383k 16.67 1151k 45.74 17932k 63.30 2790k ACAw/o EarlyStop ✓ × 24.34 1205k (↓12.87%) 16.67 1084k (↓5.82%) 44.34 16075k (↓10.36%) 62.80 2551k (↓8.57%) ACA ✓ ✓ 26.65 988k (↓28.57%) 18.00 928k (↓19.39%) 46.40 12015k (↓33.00%) 64.00 2030k (↓27.22%)

- Table 5: ACA ablation under a Best-of-16 budget. Learned uncertainty from BETAPRM gives a stronger accuracy–token tradeoff than Standard PRM with proxy uncertainty or reward-only allocation.

MathVision OlympiadBench MathVerse MathVista Acc. Tokens Acc. Tokens Acc. Tokens Acc. Tokens InternVL2.5-8B

Method

- ACA w. BETAPRM (Learned Uncertainty) 25.99 965k 16.67 958k 45.58 11912k 62.10 1949k ACA w. Standard PRM (Proxy Uncertainty) 22.37 1225k 14.00 994k 44.29 14551k 61.40 2304k ACA w. Standard PRM (Reward-Only) 21.38 738k 14.67 527k 43.02 10783k 58.80 1799k

Qwen2.5-VL-7B

- ACA w. BETAPRM (Learned Uncertainty) 26.65 988k 18.00 928k 46.40 12015k 64.00 2030k ACA w. Standard PRM (Proxy Uncertainty) 24.67 1133k 15.33 915k 45.41 13595k 62.30 2072k ACA w. Standard PRM (Reward-Only) 21.38 604k 14.67 499k 44.47 9138k 60.30 1618k

PRM. The first baseline, ACA with Standard PRM (Reward Only), removes uncertainty entirely: it ranks candidates by the average process reward from standard PRM, uses a score-margin stopping rule, and repairs the lowest-scoring step when allocating more computation. The second baseline, ACA with Standard PRM (Proxy Uncertainty), uses σt = µt(1 − µt), as an uncertainty proxy for ACA. Our full variant, ACA with BETAPRM (Learned Uncertainty), uses the uncertainty induced by its learned concentration κt. For fair comparison, all variants use the same linear risk-adjusted score Slin(y) = T1 Tt=1(µt − λσt). For the reward-only baseline, we set σt = 0, so this reduces to the average process reward. This shared form keeps the comparison well-defined across variants and focuses the ablation on the source of uncertainty.

- As shown in Table 5, BETAPRM with learned uncertainty gives the best accuracy–token tradeoff. Across the evaluated backbones, it improves over proxy uncertainty in both dimensions, achieving higher accuracy while using fewer tokens in all evaluated settings. In contrast, the µ-only variant often uses the fewest tokens, but at a clear accuracy cost. Without a reliability estimate, it treats low reward as the only reason to repair and large reward margins as sufficient evidence to stop. This misses the key cases ACA is designed for: prefixes whose score is not necessarily low, but whose PRM score is uncertain enough that additional continuations could change the selected answer. Thus, reward-only allocation saves computation by reducing exploration, but loses accuracy because it cannot identify where uncertainty still matters.

### 7 Conclusion

We study how PRMs can score reasoning steps while also indicating when those scores should be trusted. We propose BETAPRM, a distributional PRM that represents each reasoning prefix with a Beta belief over its success probability and trains it from Monte Carlo observations using a Beta-Binomial objective. This gives the model both a predicted prefix success probability and a learned reliability estimate for that prediction. Experiments show that BETAPRM improves PRMguided Best-of-N selection without sacrificing step-level error detection. Using this reliability signal, Adaptive Computation Allocation further improves final-answer accuracy while reducing inference tokens by up to 33.57%. Overall, BETAPRM turns scalar process rewards into reliability-aware signals for test-time selection and computation allocation.

### Acknowledgement

This research was supported in part by the NVIDIA Academic Grant Program and WashU Ignite Interdisciplinary Grants.

### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv e-prints, pages arXiv–2502, 2025.
- [2] Ahsan Bilal, Ahmed Mohsin, Muhammad Umer, Ali Subhan, Hassan Rizwan, Ayesha Mohsin, and Dean Hougen. What if we allocate test-time compute adaptively? arXiv preprint arXiv:2602.01070, 2026.
- [3] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.
- [4] Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th international conference on computational linguistics, pages 1511–1520, 2022.
- [5] Hyungjoo Chae, Sunghwan Kim, Junhee Cho, Seungone Kim, Seungjun Moon, Gyeom Hwangbo, Dongha Lim, Minjin Kim, Yeonjun Hwang, Minju Gwak, Dongwook Choi, Minseok Kang, Gwanhoon Im, ByeongUng Cho, Hyojun Kim, Jun Hee Han, Taeyoon Kwon, Minju Kim, Beong woo Kwak, Dongjin Kang, and Jinyoung Yeo. Web-shepherd: Advancing PRMs for reinforcing web agents. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=G2kMroO9UV.
- [6] Shuaichen Chang, David Palzer, Jialin Li, Eric Fosler-Lussier, and Ningchuan Xiao. MapQA: A dataset for question answering on choropleth maps. In NeurIPS 2022 First Table Representation Workshop, 2022. URL https://openreview.net/forum?id=znKbVjeR0yI.
- [7] Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3313–3323, 2022.
- [8] Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu, and Wanxiang Che. M3cot: A novel benchmark for multi-domain multi-step multi-modal chain-of-thought. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8199–8221, 2024.
- [9] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [10] Zhengyu Chen, Yudong Wang, Teng Xiao, Ruochen Zhou, Xusheng Yang, Wei Wang, Zhifang Sui, and Jingang Wang. From mathematical reasoning to code: Generalization of process reward models in test-time scaling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 30368–30376, 2026.
- [11] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [12] Ning Dai, Zheng Wu, Renjie Zheng, Ziyun Wei, Wenlei Shi, Xing Jin, Guanlin Liu, Chen Dun, Liang Huang, and Lin Yan. Process supervision-guided policy optimization for code generation. arXiv preprint arXiv:2410.17621, 2024.

- [13] Lingxiao Du, Fanqing Meng, Zongkai Liu, Zhixiang Zhou, Ping Luo, Qiaosheng Zhang, and Wenqi Shao. Mm-prm: Enhancing multimodal mathematical reasoning with scalable step-level supervision. arXiv preprint arXiv:2505.13427, 2025.
- [14] Keyu Duan, Zichen Liu, Xin Mao, Tianyu Pang, Changyu Chen, Qiguang Chen, Michael Qizhe Shieh, and Longxu Dou. Efficient process reward model training via active learning. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id= CJ2FmPmoDE.
- [15] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing HONG, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. G-LLaVA: Solving geometric problem with multi-modal large language model. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=px1674Wp3C.
- [16] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.
- [17] Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. rstar-math: Small LLMs can master math reasoning with self-evolved deep thinking. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=5zwF1GizFa.
- [18] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.
- [19] Mingqian He, Yongliang Shen, Wenqi Zhang, Zeqi Tan, and Weiming Lu. Advancing process verification for large language models via tree-based preference learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 2086–2099, 2024.
- [20] Vlad Hosu, Hanhe Lin, Tamas Sziranyi, and Dietmar Saupe. Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing, 29:4041–4056, 2020.
- [21] Pengfei Hu, Zhenrong Zhang, Qikai Chang, Shuhang Liu, Jiefeng Ma, Jun Du, Jianshu Zhang, Quan Liu, Jianqing Gao, Feng Ma, et al. Prm-bas: Enhancing multimodal reasoning through prm-guided beam annealing search. arXiv preprint arXiv:2504.10222, 2025.
- [22] Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. Efficient test-time scaling via self-calibration. arXiv preprint arXiv:2503.00031, 2025.
- [23] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019.
- [24] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.
- [25] Kushal Kafle, Brian Price, Scott Cohen, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5648–5656, 2018.
- [26] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Ákos Kádár, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017.

- [27] Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. Geomverse: A systematic evaluation of large models for geometric reasoning. In AI for Math Workshop@ ICML 2024, 2024.
- [28] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer, 2016.
- [29] Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, Jaekyeom Kim, Hao Peng, Moontae Lee, Honglak Lee, and Lu Wang. Process reward models that think. arXiv preprint arXiv:2504.16828, 2025.
- [30] Seungone Kim, Ian Wu, Jinu Lee, Xiang Yue, Seongyun Lee, Mingyeong Moon, Kiril Gashteovski, Carolin Lawrence, Julia Hockenmaier, Graham Neubig, et al. Scaling evaluation-time compute with reasoning models as process evaluators. arXiv preprint arXiv:2503.19877, 2025.
- [31] Jinyuan Li, Chengsong Huang, Langlin Huang, Shaoyang Xu, Haolin Liu, Wenxuan Zhang, and Jiaxin Huang. Training data efficiency in multimodal process reward models. arXiv preprint arXiv:2602.04145, 2026.
- [32] Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, and Alan L Yuille. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14963–14973, 2023.
- [33] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=v8L0pN6EOi.
- [34] Haolin Liu, Dian Yu, Sidi Lu, Yujun Zhou, Rui Liu, Zhenwen Liang, Haitao Mi, Chen-Yu Wei, and Dong Yu. Save the good prefix: Precise error penalization via process-supervised rl to enhance llm reasoning. arXiv preprint arXiv:2601.18984, 2026.
- [35] Runze Liu, Junqi Gao, Jian Zhao, Kaiyan Zhang, Xiu Li, Biqing Qi, Wanli Ouyang, and Bowen Zhou. Can 1b LLM surpass 405b LLM? rethinking compute-optimal test-time scaling. In Workshop on Reasoning and Planning for Large Language Models, 2025. URL https: //openreview.net/forum?id=CvjX9Lhpze.
- [36] Wei Liu, Junlong Li, Xiwen Zhang, Fan Zhou, Yu Cheng, and Junxian He. Diving into selfevolving training for multimodal reasoning. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=X3ikghfWwD.
- [37] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6774–6786, Online, August 2021. Association for Computational Linguistics. doi: 10. 18653/v1/2021.acl-long.528. URL https://aclanthology.org/2021.acl-long.528/.
- [38] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum? id=uXa9oBDZ9V1.
- [39] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35: 2507–2521, 2022.

- [40] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=KUNzEQMWU7.
- [41] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Meiqi Guo, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.
- [42] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Xinzhe Ni, Zicheng Lin, Songtao Jiang, Yiyao Yu, Chufan Shi, Lei Wang, Ruihang Chu, et al. Unlocking multimodal mathematical reasoning via process reward model. arXiv preprint arXiv:2501.04686, 2025.
- [43] Qianli Ma, Haotian Zhou, Tingkai Liu, Jianbo Yuan, Pengfei Liu, Yang You, and Hongxia Yang. Let’s reward step by step: Step-level reward model as the navigators for reasoning. arXiv preprint arXiv:2310.10080, 2023.
- [44] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022.
- [45] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [46] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C.V. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1697–1706, January 2022.
- [47] Tej Deep Pala, Panshul Sharma, Amir Zadeh, Chuan Li, and Soujanya Poria. Error typing for smarter rewards: Improving process reward models with error-aware hierarchical supervision. arXiv preprint arXiv:2505.19706, 2025.
- [48] xu Zhao Pan, Pengfei Zhou, Jiaxin Ai, Wangbo Zhao, Kai Wang, Xiaojiang Peng, Wenqi Shao, Hongxun Yao, and Kaipeng Zhang. MPBench: A comprehensive multimodal reasoning benchmark for process errors identification. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 21586–21606, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1112. URL https://aclanthology.org/2025.findings-acl.1112/.
- [49] Young-Jin Park, Kristjan Greenewald, Kaveh Alim, Hao Wang, and Navid Azizan. Know what you don’t know: Uncertainty calibration of process reward models. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2026. URL https: //openreview.net/forum?id=hzMkfIrdDT.
- [50] Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 conference on empirical methods in natural language processing, pages 1466–1476, 2015.
- [51] Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4663–4680, 2024.
- [52] Shweta Singh, Aayan Yadav, Jitesh Jain, Humphrey Shi, Justin Johnson, and Karan Desai. Benchmarking object detectors with coco: A new path forward. In European Conference on Computer Vision, pages 279–295. Springer, 2024.
- [53] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

- [54] Mingyang Song, Zhaochen Su, Xiaoye Qu, Jiawei Zhou, and Yu Cheng. Prmbench: A finegrained and challenging benchmark for process-level reward models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25299–25346, 2025.
- [55] Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for reasoning about natural language grounded in photographs. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 6418–6428, 2019.
- [56] Lin Sun, Chuang Liu, Xiaofeng Ma, Tao Yang, Weijia Lu, and Ning Wu. Freeprm: Training process reward models without ground truth process labels. arXiv preprint arXiv:2506.03570, 2025.
- [57] Haoqin Tu, Weitao Feng, Hardy Chen, Hui Liu, Xianfeng Tang, and Cihang Xie. Vilbench: A suite for vision-language process reward modeling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 6775–6790, 2025.
- [58] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.
- [59] Jun Wang, Meng Fang, Ziyu Wan, Muning Wen, Jiachen Zhu, Anjie Liu, Ziqin Gong, Yan Song, Lei Chen, Lionel M Ni, et al. Openr: An open source framework for advanced reasoning with large language models. arXiv preprint arXiv:2410.09671, 2024.
- [60] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.
- [61] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024.
- [62] Shuai Wang, Zhenhua Liu, Jiaheng Wei, Xuanwu Yin, Dong Li, and Emad Barsoum. Athena: Enhancing multimodal reasoning with data-efficient process reward models. arXiv preprint arXiv:2506.09532, 2025.
- [63] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, and Wenhai Wang. VisualPRM400k: An effective dataset for training multimodal process reward models. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=IHyY6vdYZw.
- [64] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw.
- [65] Wei Xiong, Wenting Zhao, Weizhe Yuan, Olga Golovneva, Tong Zhang, Jason Weston, and Sainbayar Sukhbaatar. Stepwiser: Stepwise generative judges for wiser reasoning. arXiv preprint arXiv:2508.19229, 2025.
- [66] Runyang You, Yongqi Li, Meng Liu, Wenjie Wang, Liqiang Nie, and Wenjie Li. Parallel test-time scaling for latent reasoning models. arXiv preprint arXiv:2510.07745, 2025.
- [67] Fei Yu, Anningzhe Gao, and Benyou Wang. Ovm, outcome-supervised value models for planning in mathematical reasoning. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 858–875, 2024.
- [68] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.

- [69] Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Shanghang Zhang, Peng Gao, and Hongsheng Li. MAVIS: Mathematical visual instruction tuning with an automatic data engine. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=MnJzJ2gvuf.
- [70] Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, and Xiangyu Zhao. Process vs. outcome reward: Which is better for agentic RAG reinforcement learning. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2026. URL https: //openreview.net/forum?id=h3LlJ6Bh4S.
- [71] Yuxin Zhang, Meihao Fan, Ju Fan, Mingyang Yi, Yuyu Luo, Jian Tan, and Guoliang Li. Rewardsql: Boosting text-to-sql via stepwise reasoning and process-supervised rewards. arXiv preprint arXiv:2505.04671, 2025.
- [72] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. The lessons of developing process reward models in mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 10495–10516, 2025.
- [73] Chujie Zheng, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Processbench: Identifying process errors in mathematical reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1009–1024, 2025.
- [74] Tong Zheng, Chengsong Huang, Runpeng Dai, Yun He, Rui Liu, Xin Ni, Huiwen Bao, Kaishen Wang, Hongtu Zhu, Jiaxin Huang, et al. Parallel-probe: Towards efficient parallel thinking via 2d probing. arXiv preprint arXiv:2602.03845, 2026.
- [75] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

Table 6: Source coverage of VisualPRM400K-v1.1 used for PRM training.

Group Representative sources Diagram / Synthetic Reasoning AI2D [28], CLEVR [24], Super-CLEVR [32], NLVR2 [55], FigureQA [26],

IconQA [38] Chart / Document / OCR QA ChartQA [44], DocVQA [45], DVQA [25], InfographicVQA [46], SROIE [23] General VQA and Visual Reasoning VQAv2 [16], COCO-ReM [52], KonIQ-10k [20], M3CoT [8], MAPQA-

SUV [6] Science and Math Reasoning ScienceQA [39], MathV360K [51], MAVIS variants [69] Geometry Reasoning Geo170K [15], Geometry3K [37], GeoQA+ [4], GEOS [50], Geom-

Verse [27], UniGeo [7]

### A Experimental Setup and Implementation Details

#### A.1 Training Data and Backbones

We train all PRMs on VisualPRM400K-v1.1 [63]. We use this version because it exposes the raw Monte Carlo supervision needed by our objective: for each supervised reasoning prefix, the dataset reports the number K of successful continuations among N = 16 sampled continuations. This allows BETAPRM to train on the count pair (K,N), while the standard PRM baseline is trained with cross-entropy using the empirical ratio K/N as a soft label.

After validity filtering, the training split contains 565,096 rollouts and 3,174,394 annotated steps. The average solution contains 5.62 reasoning steps, with 27.8 words per step on average. The dataset is broad in source coverage, containing 38 subsets across diagram understanding, chart and document QA, general visual question answering, science reasoning, and mathematical/geometry reasoning.

We instantiate both the standard PRM and BETAPRM with four multimodal backbones: InternVL2.58B [9], InternVL3-8B [75], InternVL3-14B [75], and Qwen2.5-VL-7B [1]. For each backbone, we insert a <prm> marker after every reasoning step and supervise only the marker positions. The reward mean µt is computed from the Yes/No reward-token logits. For BETAPRM, we additionally attach a lightweight linear head on the marker hidden state to predict the concentration κt. Unless otherwise specified, we freeze the vision encoder and fine-tune the language model together with the multimodal projection modules. Training takes about 48 hours on 4 A100 GPUs for the 7B/8B backbones and about 48 hours on 8 A100 GPUs for the 14B backbone.

#### A.2 Optimization and Model Hyperparameters

We use the same optimization recipe for the standard PRM baseline and BETAPRM whenever they share the same backbone. The standard PRM is trained with the cross-entropy objective over the Yes/No reward tokens, while BETAPRM replaces this objective with the Beta-Binomial loss and adds the concentration head. The InternVL2.5-8B, InternVL3-8B, and InternVL3-14B experiments use the same hyperparameters; Qwen2.5-VL-7B uses the same optimization settings with its native image preprocessing. Table 7 summarizes the hyperparameters needed to reproduce training.

For InternVL backbones, we use dynamic image resolution with image size 448, at most 6 image patches, down-sampling ratio 0.5, and drop-path rate 0.4. For Qwen2.5-VL-7B, we use the native Qwen2.5-VL preprocessing with minimum and maximum pixel counts 784 and 200,704, respectively. All models insert <prm> after each reasoning step and compute µt from the Yes and No logits at these marker positions.

#### A.3 Best-of-N Evaluation Protocol

All PRM selectors use the same candidate pools, so the comparison isolates the effect of the reward model and selection rule.

Each candidate y = s1:T is formatted by inserting a <prm> marker after every reasoning step:

##### Question: x Process: s1,<prm>,...,sT,<prm>.

Table 7: Optimization and model hyperparameters. Beta-Binomial-specific rows apply only to BETAPRM.

Item Value

Optimizer AdamW Learning Rate 1 × 10−5 Weight Decay 0.05 LR Schedule Cosine Decay with Warmup Warmup Ratio 0.05 Epochs 1 Global Batch Size 512 Max Sequence Length 8192 Trainable Modules LLM + multimodal projector; vision encoder frozen

ϵ in Beta Parameters 1 × 10−6 κmin 1 × 10−3 Initial κ 4.0

Lreg Coefficient 5 × 10−2 Concentration-head LR Multiplier 10.0

- At each marker, the model scores the step using the normalized Yes probability over the reward tokens Yes and No. For the Standard PRM baseline, candidates are ranked by the average reward,

T

1 T

SPRM(y) =

µt.

t=1

For BETAPRM, we additionally extract the concentration κt and compute the Beta standard deviation

µt(1 − µt) κt + 1

σt =

.

We rank candidates with the risk-budget selector used in the main experiments:

T

T

1 T

1 T

µt − λ

1[σt > τ].

SRB(y) =

t=1

t=1

The penalty weight λ and uncertainty threshold τ are selected from the same small fixed grid for all reported BETAPRM runs: λ ∈ {0.2,0.5,0.7,1.0,1.5}, with τ set by the q-th percentile of step-level σt, q ∈ {0.7,0.8,0.9}.

#### A.4 VisualProcessBench Evaluation Protocol

We evaluate step-level error detection on VisualProcessBench [63]. For each instance, we concatenate the question with the provided step-by-step rationale and insert a <prm> marker after every step, using the same input format as PRM training. The model produces one score at each marker, which is then converted into a binary prediction of whether the corresponding step is correct or erroneous. Neutral labels are ignored when computing metrics.

For the Standard PRM baseline, the step score is the normalized Yes probability µt. For BETAPRM, we use the same reward mean together with the learned concentration to compute

σt = µt(1 − µt)/(κt + 1), and evaluate the risk-adjusted step score st = µt − λσt, with λ = 0.5. This uses the reliability signal in the same direction as our downstream selection experiments: uncertain positive-looking steps are scored more conservatively.

Given a threshold τcls, steps with st ≥ τcls are classified as correct and those below the threshold are classified as erroneous. Following the benchmark protocol [63], we choose a single global threshold

per model by sweeping τcls and maximizing the overall validation F1. We report the overall score and the per-source macro-F1 breakdown on VisualProcessBench.

#### A.5 Adaptive Computation Allocation Details

ACA is evaluated under the same maximum Best-of-16 budget as the fixed-budget baseline. For each problem, ACA first samples n0 = 4 complete candidates from scratch. If the stopping criterion is not

satisfied, it allocates another batch of m = 4 candidates, up to the maximum budget N = 16. All new candidates are generated by InternVL2.5-8B with the same decoding parameters as the fixed-budget Best-of-16 baseline: temperature 0.7, top-p = 0.9, top-k = 30 and maximum new tokens 2048.

- At each stage, candidates are scored by the linear risk-adjusted score used in the ACA stopping rule,

1 T

Slin(y) =

T

(µt − λσt),

t=1

with λ = 0.5. The lower and upper confidence bounds use U(y) = T−1 t σt with cstop = 0.3. When ACA continues, it expands the highest-UCB non-winner competitor. For prefix repair, we use

pbad = 0.3 as the low-quality threshold and ccut = 1.0 in the conservative step score µt − ccutσt.

For the main ACA results, final candidate selection uses the same risk-budget selector SRB as the Best-of-16 evaluation. For the ACA ablation in Table 5, all variants instead use the shared linear

score Slin, with σt = 0 for the reward-only Standard PRM baseline. This keeps the ablation focused on the source of uncertainty.

### B Limitations

BETAPRM requires supervision that preserves the Monte Carlo count used to estimate prefix success, rather than only binarized step labels. Our experiments therefore use VisualPRM400K-v1.1 [63], which, to our knowledge, is the only publicly available PRM training dataset that reports the number of successful continuations for each prefix. This availability constraint is why our experiments focus on multimodal PRMs, although the Beta-Binomial formulation itself is not tied to multimodal inputs.

### C Broader Societal Impact

BETAPRM improves the reliability and efficiency of reasoning systems by enabling PRMs to report both reward estimates and learned reliability. Such signals can help downstream methods avoid over-trusting uncertain judgments, allocate computation more adaptively, and reduce unnecessary inference cost. More broadly, reliability-aware reward modeling may make AI reasoning systems easier to audit and more useful for research, education, and other reasoning-intensive applications.

Care should still be taken when applying BETAPRM beyond the evaluated benchmarks. Learned reliability is an additional signal rather than a guarantee of correctness, so high-stakes uses should involve human oversight, calibration checks, and domain-specific evaluation.

