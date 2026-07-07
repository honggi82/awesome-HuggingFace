### Training Data Efficiency in Multimodal Process Reward Models

Jinyuan Li1 Chengsong Huang1 Langlin Huang1 Shaoyang Xu2 Haolin Liu3 Wenxuan Zhang2 Jiaxin Huang1

## arXiv:2602.04145v2[cs.LG]5Feb2026

#### Abstract

Multimodal Process Reward Models (MPRMs) are central to step-level supervision for visual reasoning in MLLMs. Training MPRMs typically requires large-scale Monte Carlo (MC)annotated corpora, incurring substantial training cost. This paper studies the data efficiency for MPRM training. Our preliminary experiments reveal that MPRM training quickly saturates under random subsampling of the training data, indicating substantial redundancy within existing MC-annotated corpora. To explain this, we formalize a theoretical framework and reveal that informative gradient updates depend on two factors: label mixtures of positive/negative steps and label reliability (average MC scores of positive steps). Guided by these insights, we propose the Balanced-Information Score (BIS), which prioritizes both mixture and reliability based on existing MC signals at the rollout level, without incurring any additional cost. Across two backbones (InternVL2.5-8B and Qwen2.5-VL-7B) on VisualProcessBench, BIS-selected subsets consistently match and even surpass the full-data performance at small fractions. Notably, the BIS subset reaches full-data performance using only 10% of the training data, improving over random subsampling by a relative 4.1%. Our code is released Balanced-Info-MPRM.

#### 1. Introduction

Process Reward Models (PRMs) (Ma et al., 2023; Zhu et al., 2025; Tan et al., 2025) provide step-level supervision for reasoning by scoring intermediate steps instead of only the final answer. In multimodal reasoning, Multimodal PRMs (MPRMs) are increasingly used for Multimodal Large

1Washington University in St. Louis 2Singapore University of Technology and Design 3University of Virginia. Correspondence to: Jinyuan Li <ljinyuan@wustl.edu>, Jiaxin Huang (Corresponding Author) <jiaxinh@wustl.edu>.

Preprint. February 6, 2026.

[Figure 1]

Figure 1. Overall micro-F1 on VisualProcessBench using InternVL2.5-8B trained via full training set or different subsets. Our BIS-10% effectively matches the final performance of FullData setting using 10× fewer rollouts.

Language Models (MLLMs) (Wang et al., 2024e;f; Bai et al., 2023; 2025; Liu et al., 2023; Team et al., 2024; 2025) to conduct complex visual reasoning tasks both during training and at test time (Wang et al., 2025a;b; Zhang et al., 2025a; Luo et al., 2024; Du et al., 2025; Tu et al., 2025; Cao et al., 2025; Dong et al., 2025). Common practices for training MPRMs generally rely on large-scale Monte Carlo (MC)-annotated rollouts (e.g., VisualPRM400K-v1.1 (Wang et al., 2025b), with 565K rollouts and 3.17M annotated steps), which makes training computationally expensive. In this paper, we study the practical bottleneck in training data efficiency for MPRMs: how does MPRM performance scale with the rollout budget, and how can we select informative subsets that preserve full-data performance?

Our preliminary study suggests substantial redundancy in MC-annotated MPRM training data. We randomly subsample the training data at varying fractions ρ and find that performance quickly saturates at small ρ, with a moderate gap to the full dataset. This trend persists even when the subset is trained longer to match full-data training steps. We further compare several size-matched heuristic subsets and find that selecting rollouts that mix correct and incorrect steps is more informative than random selection, whereas rollouts with the lowest average MC scores tend to contain noisy pseudopositive labels and hurt performance. This suggests two key criteria for high-quality rollouts: mixture and reliability.

To substantiate this intuition, we formalize a teacher–student abstraction framework for theoretical analysis, and connect the interplay between gradient signal, label noise and data redundancy. We model MC estimation noise via a probabilistic label-flip model and show how it affects training gradients. This modeling supports the view that MPRM training is primarily limited by gradient noise rather than data scarcity. Moreover, our theory explains why mixture and reliability capture rollout quality: mixture tracks model uncertainty, while reliability, measured by MC scores, captures the noise level in positive steps. Their contributions interact multiplicatively in shaping informative gradients.

Building on these insights, we introduce the BalancedInformation Score (BIS), a rollout-level criterion that instantiates the “mixed but reliable” principle. BIS quantifies both label mixture (of positive and negative steps) and reliability (average MC score over positive steps). It is modelagnostic and only uses the MC signals stored in the dataset, without requiring extra model calls. Extensive experiments with two backbones (InternVL2.5-8B (Chen et al., 2024c) and Qwen2.5-VL-7B (Bai et al., 2025)) on VisualProcessBench (Wang et al., 2025b) show that BIS recovers full-data performance at small subset ratios, with the largest gain over random sub-sampling in low-budget regimes. In particular,

- Figure 1 shows that a BIS-selected 10% subset trained for only 50 steps suffices to reach and even surpass the full-data performance on InternVL2.5-8B, saving 95.5% computational cost. Taken together, these findings provide a practical recipe with grounded analysis for reducing training compute for MPRMs without sacrificing model performance.

#### 2. Preliminary Study

##### 2.1. Background and General Setup

Previous MPRM research mainly improves supervision pipelines or training frameworks (detailed in Appendix A). In contrast, we study post-hoc rollout selection with no extra supervision or compute. We adopt the standard MPRM training setup and keep it fixed throughout. Following prior works (Wang et al., 2024d; Zhang et al., 2025g), we use the VisualPRM400K-v1.1 dataset (Wang et al., 2025b), where each reasoning step is annotated with an MC-estimated success rate from N=16 sampled continuations. Step labels are binarized, so yt=1 if the MC score > 0 (at least one continuation reaches correct final answer), and yt=0 otherwise. Specifically, for a reasoning rollout with T steps, a special token <prm> is appended after each step t. The model is trained to predict the step-level probability (“Yes”/“No”) tokens using the cross-entropy loss L = − Tt=1(yt log pt + (1 − yt)log(1 − pt)), where pt is the probability of predicting the token “Yes”. We use InternVL2.5-8B (Chen et al., 2024c) in preliminary study,

Table 1. Dataset statistics for different training-set settings. “Steps” denote reasoning steps with annotated labels.

Metric Full-Data Random-25% Low-MC-25% Mixed-25% # rollouts 565,096 141,288 141,210 141,253 # reasoning steps 3,174,394 794,756 796,940 795,752 Avg. steps/rollout 5.62 5.63 5.64 5.63 Avg. words/step 27.8 27.8 29.9 27.6 Error-step ratio 3.57% 3.61% 12.57% 11.02% Avg. MC/step 0.8566 0.8590 0.6010 0.7160

and evaluate the MPRM performance on VisualProcessBench (Wang et al., 2025b), a human-annotated step-level benchmark spanning five sources (MathVision (Wang et al., 2024c), MathVerse (Zhang et al., 2024), MMMU (Yue et al., 2024), DynaMath (Zou et al., 2025), and WeMath (Qiao et al., 2025)), and follow its protocol to report per-source macro-F1 and micro-averaged F1 over all sources. Training details, data statistics are provided in Appendix C, B and H.

##### 2.2. Random Sub-Sampling: Evidence of Redundancy

Empirical Finding 1: MPRM performance quickly saturates under random subsampling, indicating strong redundancy in the training data.

To assess how MPRM performance scales with processsupervision data, we use the full training corpus as FullData and evaluate random subsampling. For any keep ratio ρ, Random-ρ retains a fraction ρ of rollouts from each of the 38 source subsets, preserving their relative composition.

We train with single-pass fine-tuning and report micro-F1 for different Random-ρ subsets in Figure 2a. Performance improves with ρ but quickly plateaus, exhibiting pronounced diminishing returns. This suggests substantial redundancy in the MC-annotated rollouts, as discarding a large fraction of rollouts only modestly degrades performance.

To probe the plateau at moderate ρ, we take ρ = 25% and compare Random-25% with Full-Data under a matched compute budget. We match the number of training steps by training Random-25% for four epochs, making its training cost comparable to one epoch of Full-Data. Table 1 summarizes the corpus statistics for these settings.

We compare their learning curves in Figure 2b. Although the Full-Data model eventually performs better, the gap to Random-25% remains moderate, confirming substantial redundancy in the training data. We also report per-source results in Appendix D. Additionally, under matched updates, Random-25% slightly overfits and its performance degrades late in training. In the remaining experiments, we use singlepass fine-tuning, where each rollout is seen exactly once.

Given the redundancy above, a natural next question is: is there a principled data selection method that substantially filters training data while preserving full-data performance?

[Figure 2]

[Figure 3]

[Figure 4]

(a) (b) (c)

- Figure 2. Overall VisualProcessBench micro-F1 under different data regimes. (a) Single-pass scaling with random sub-sampling; per-source macro-F1 curves are shown in Figure 4. (b) Training on Full-Data vs. Random-25% under matched updates; per-source macro-F1 curves are shown in Figure 5. (c) Training on three 25% subsets for one epoch; per-source macro-F1 curves are shown in Figure 6. Full-Data† denotes the best checkpoint of a one-epoch Full-Data run (4× more optimization steps than 25% subsets).

##### 2.3. Characterizing Informative Rollouts

Empirical Finding 2: Effective supervision comes from mixed rollouts that contain both correct and incorrect steps while maintaining reliable positive labels.

We now shift focus from how many rollouts to use to which rollouts to keep. To study the impact of increased exposure to negative steps, we construct three subsets of VisualPRM400K: Random-25%, Low-MC-25%, and Mixed-25%.

Random-25% randomly samples 25% of rollouts from each source to preserve the original dataset distribution.

Low-MC-25% is constructed by ranking rollouts within each source by their average MC score per step and retaining the bottom 25%. As a result, the average MC per step drops to 0.601 and the incorrect-step ratio rises to 12.57%, far higher than in Random-25% (3.61%). Many low-MC steps have only a few successful continuations out of N = 16, yet are still labeled as positive under the standard binarization rule, making them prone to pseudo-positive labels.

Mixed-25% prioritizes rollouts with both positive and negative steps. Since mixed rollouts are only 7.67%, when a source has fewer than 25% mixed rollouts, we fill the remainder by randomly sampling from the rest. It has a similar incorrect-step ratio to Low-MC-25% (11.02% vs. 12.57%) but a higher average MC score (0.716 vs. 0.601), exposing the model to many negative steps while still anchoring them with reasonable amount of reliable positive labels.

- Table 1 summarizes the statistics for these subsets. Using the same training protocol, we fine-tune each 25% subset for one epoch and plot the overall VisualProcessBench microF1 over training steps in Figure 2c. From this comparison, we observe two patterns as follows:

First, at their best checkpoints, the three subsets satisfy Mixed-25% > Low-MC-25% > Random-25%. Both LowMC-25% and Mixed-25% outperform Random-25%, indicating that, under a fixed data budget, exposing the model to more incorrect steps is beneficial.

Second, Mixed-25% consistently yields the strongest per-

formance even though its incorrect-step ratio is comparable to Low-MC-25% while its average MC score is notably higher. This suggests that neither maximizing negative steps nor minimizing average MC scores alone is sufficient. Extremely low-MC steps tend to be noisy pseudo-positives (labeled positive despite very low success rates), whereas rollouts that combine reasonably reliable positive steps with clear errors provide more useful supervision.

These observations motivate us to find a rollout-scoring mechanism that prioritizes two aspects: (1) emphasizing mixed rollouts (containing both correct and incorrect steps) while (2) avoiding noisy rollouts that contain many extremely low MC-score steps.

#### 3. Theoretical Analysis

Before introducing our scoring mechanism, we provide a theoretical analysis to explain empirical findings 1 and 2. We formalize the interplay among data redundancy, label noise, and gradient behavior, which also guides the design of an effective data-selection score. 3.1. Teacher–Student Abstraction

We model MPRM training using a linear teacher–student framework: the teacher represents the ideal model that knows true step-level correctness, while the student model learns from noisy MC-annotated labels. We model steplevel label prediction as logistic regression on the representation space ϕ for simplicity. For the j-th step in rollout x, let ϕx,j ∈ Rd denote the hidden representation at the <prm> token position and Yx,jtrue ∈ {0,1} its binary label. An ideal “teacher” MPRM is

q∗(ϕ) = Pr(Y true = 1 | ϕ) = σ ⟨w∗,ϕ⟩ , (1)

where w∗ ∈ Rd is the optimal parameter and σ is the sigmoid function. The student MPRM (our learned model) is

qw(ϕ) = σ ⟨w,ϕ⟩ , (2) trained by minimizing the expected logistic loss

L(w) = E(ϕ,Y )[−Y log qw(ϕ) − (1 − Y ) log(1 − qw(ϕ))] . (3)

In the MC-annotated training set, each step is associated with an MC score sx,j ∈ [0,1] from N sampled continuations and a binary label Yx,jmc = I[sx,j > 0]. For the theoretical analysis, we do not model the MC sampling explicitly; instead, we model (ϕx,j,Yx,jtrue) as i.i.d. samples from the teacher model in Eq. (1). The observed MC score sx,j (and the corresponding binarized training label Yx,jmc) provides a noisy estimate of the underlying correctness probability q∗(ϕx,j). Under this formulation, the student qw is trained with the logistic loss in Eq. (3) on the observed MC labels Yx,jmc, matching the objective in Section 2.1.

##### 3.2. Understanding the Plateau of Random Subsets

Theoretical Finding 1: MPRM training mostly suffers from noisy gradients instead of insufficient training data.

In this part, we aim to explain why randomly sub-sampled subsets across varying ρ recover much of the full-dataset performance. In the teacher–student setup of Section 3.1, we consider training the student model to minimize a logistic loss L(w), with w∗ being the optimal parameter (achieved with infinite data and infinite training), and wT the parameters after T finite stochastic gradient descent (SGD) steps.

Under the assumptions in Appendix F.1, standard nonasymptotic analyses of SGD for logistic regression (Bach & Moulines, 2013; Bottou & Bousquet, 2007) yield a bound on the excess risk (the gap in expected loss between wT and w∗) of the form

E L(wT) −L(w∗) ≲ Cdata Neff−1/2 + Copt T−1/2 (4) This bound contains two components: (1) a data complexity term Cdata Neff−1/2, which decays with the effective sample size Neff, and (2) an optimization error term Copt T−1/2, which decays with the number T of SGD updates. Cdata,Copt > 0 are problem-dependent constants that do not scale with Neff or T. A detailed derivation of Eq. (4) is given in Appendix F.1.

Why larger datasets help less than expected. In MCannotated training data, many steps receive noisy labels, especially when only a few out of N continuations succeed. This label noise increases the stochastic-gradient noise level, which effectively enlarges the constant Copt in the optimization term (Moulines & Bach, 2011). Meanwhile, for VisualPRM400K-v1.1 with 3.17M annotated steps, the data complexity term CdataNeff−1/2 is relatively small. Taken together, the optimization term dominates the total error, and further increasing Neff yields only marginal gains.

Now consider a random subset Random-γ that keeps each data point with probability γ ∈ (0,1), so the effective sample size becomes γNeff while the problem-dependent constants remain comparable. Let Tγ denote the number of

SGD updates. In the matched-update setting we keep the update budget fixed, Tγ = T, and Eq. (4) gives

- 1

- 2.

- 1

- 2CdataNeff−1/2 + CoptT−

)] − L(w∗) ≲ γ−

E[L(wT

γ

Random sub-sampling therefore amplifies only the (already small) data term by γ−1/2, while leaving the optimization

term unchanged; once CdataNeff−1/2 ≪ CoptT−1/2, changing γ has only a modest effect on the total error, explaining

why Random-25% closely tracks Full-Data in Figure 2b. In the single-pass setting we have Tγ = γT, so Eq. (4) gives E[L(wT

- 1

- 2 + Copt(γT)−

- 1

- 2.

)] − L(w∗) ≲ Cdata(γNeff)− Let B := CdataN−

γ

- 1

- 2

- 1

- 2. The right-hand side can

eff + CoptT− be written as Bγ := γ− the γ−1/2. For the full-data configuration with ∼ 3.17M annotated steps and ∼ 1.1k updates, we operate in a lowerror regime where B ≪ ϵtar. Here ϵtar is the target error level. When B is already far below ϵtar, multiplying it by the constant factor γ−1/2 still gives Bγ ≤ ϵtar, so both Full-Data and Random-γ single-pass training remain within the desired accuracy range. In such a regime, a constant factor γ−1/2 in the bound is not enough to induce a large performance gap, which matches the small empirical difference we observe in Figure 2c and reinforces that improving gradient quality, rather than merely enlarging Neff, is key to escaping the current optimization floor.

- 1

- 2B, so both terms are scaled by

3.3. Why Mixed but Reliable Rollouts Are Informative? Here, we explain why mixture and reliability characterize informative rollouts: label mixture tracks teacher uncertainty, while MC scores quantify the reliability of step labels.

##### Step-level Information from Teacher Uncertainty

Theoretical Finding 2: Ideal teacher-model uncertainty q∗(ϕ) 1 − q∗(ϕ) quantifies per-step information.

Under the teacher–student framework in Section 3.1, the gradient of logistic loss for a step with representation ϕ ∈ Rd and label Y ∈ {0,1} is

g(ϕ,Y ;w) = qw(ϕ) − Y ϕ, qw(ϕ) = σ(⟨w,ϕ⟩). Since we study offline data selection, which is fixed throughout training, we measure how informative a step is under the teacher distribution rather than an evolving student, yielding a student-independent criterion. At the teacher’s optimal parameter w∗, the second moment of the gradient has the form (derivation in Appendix F.2)

E ∥g(ϕ,Y ;w∗)∥2 | ϕ = q∗(ϕ) 1 − q∗(ϕ) ∥ϕ∥2, (5)

where q∗(ϕ) = qw∗(ϕ). E ∥g(ϕ,Y ;w∗)∥2 |ϕ quantifies the expected per-step learning signal. Thus, for step-level MPRM training, the most informative steps are those where

the teacher is most uncertain (q∗(ϕ) ≈ 1/2), when the term q∗(ϕ) 1 − q∗(ϕ) reaches its maximum.

##### Effect of Label Noise at Step Level

Theoretical Finding 3: Extremely low-MC positive steps behave like label-reversed samples and produce gradients that carry little true signal and harm training.

To analyze how MC label noise affects learning, we adopt a symmetric label noise approximation: the true label Y is flipped independently with probability η ∈ [0,1/2) to produce a noisy label Y˜, where restricting to η < 1/2 is without loss of generality since larger rates can be reduced to 1−η by flipping the label semantics. In practice the effective noise is step-dependent, and we adopt this constant noise distribution to derive the explicit formula for the second moment of the noisy gradient at w∗, where g˜(ϕ,Y˜;w∗) = (q∗(ϕ) − Y˜)ϕ. A direct computation (Appendix F.3) yields

E ∥g˜(ϕ,Y˜;w∗)∥2 | ϕ

= (1 − 4η)q∗(ϕ) 1 − q∗(ϕ) + η ∥ϕ∥2. (6)

Relative to the clean case in Eq. (5), the uncertainty term q∗(ϕ) 1 − q∗(ϕ) is shrunk by (1 − 4η), with an additional q∗-independent noise η∥ϕ∥2. When η is large, gradients are increasingly noise-dominated and carry little useful signal. Empirically, steps with extremely low-MC scores but positive labels are typically unstable: only a few out of all continuations succeed for incidental reasons (e.g., later selfcorrection), so these steps act like label-flipped negative steps rather than real positive steps.

conditioning on {qx,j} yields E[ˆpx(1−pˆx) | {qx,j}] = q¯x(1−q¯x)−

n

1 n2

qx,j(1−qx,j).

j=1

(7) By Jensen’s inequality for t  → t(1 − t), we have

n

1 n

qx,j 1 − qx,j ≤ q¯x(1 − q¯x) =: θx. (8) Combining (7) and (8) yields the sandwich bound

A(x) =

j=1

1 n ≤ E p ˆx(1 − pˆx) | {qx,j} ≤ θx. (9)

θx 1 −

Thus pˆx(1 − pˆx) is an observable estimate for the teacherlevel uncertainty θx, with only O(1/n) bias, and A(x) ≤ θx by construction. Consequently, in this noise-free setting, rollouts that are nearly all-positive or all-negative have small pˆx(1 − pˆx), suggesting smaller θx (and hence smaller expected A(x)). In contrast, rollouts with mixed labels tend to have larger θx, which allows A(x) to be larger and can yield gradient updates with stronger learning signal.

Now consider the symmetric flip noise model: Y˜x,j is obtained by flipping Yx,j with a constant rate η. Then Y˜x,j | qx,j ∼ Bernoulli(˜qx,j) with q˜x,j = (1 − 2η)qx,j +η (Appendix F.4 Eq. (22)). Averaging over steps gives q¯˜x = (1−2η)¯qx+η, and the induced rollout-level mixture satisfies

θ˜x := q¯˜x(1 − q¯˜x) = (1 − 2η)2 θx + η(1 − η). (10)

Eq. (10) decomposes θ˜x into a scaled uncertainty term (1 − 2η)2θx plus an offset η(1 − η). Since θ˜x ≈ θx only for small η (Appendix F.4 Eq. (24)), we next complement mixture with a reliability signal to identify low-noise rollouts.

##### Rollout Label Mixture Estimates Teacher Uncertainty

Theoretical Finding 4: Rollout label mixture pˆx(1−pˆx) is an O(1/n)-biased estimator of the unobserved teacher-level uncertainty θx = q¯x(1 − q¯x) under noise-free labels.

We now relate rollout label mixture to the teacher’s positivelabel probabilities as follows. For a rollout x with n steps we define average step-wise information:

n

1 n

qx,j 1 − qx,j .

A(x) :=

j=1

Under the bounded-norm assumption in Appendix F.4, the unweighted and norm-weighted quantities differ by at most global multiplicative constants. For exposition, we begin with the teacher-consistent idealization Yx,j | qx,j ∼ Bernoulli(qx,j), under which {Yx,j}nj=1 are unbiased samples from {qx,j}. Let pˆx := n1 nj=1 Yx,j be the empirical positive-label fraction, and let q¯x := n1 nj=1 qx,j be the step-average teacher probability. Then pˆx(1 − pˆx) is maximized when labels are balanced; it becomes 0 when the steps are all-positive or all-negative, so it directly measures label mixture within rollout x. By Lemma 2 in Appendix F.4,

##### MC Scores as Effective Noise Indicators

Theoretical Finding 5: MC scores monotonically reflect label reliability: low-MC positives exhibit high effective noise.

Mixture alone is not sufficient because positive labels can be noisy. We now model the MC annotation process and connect MC scores to the label-noise level. For step j in rollout x, let rx,j ∈ [0,1] be the probability that a single continuation from this step reaches the correct final answer, given its representation ϕx,j. The MC annotator generates N independent continuations, records the number of successful ones Kx,j, and stores the score sx,j = Kx,j/N. Under the Binomial model Kx,j | rx,j ∼ Binomial(N,rx,j) we have

1 N

E[sx,j | rx,j] = rx,j, Var(sx,j | rx,j) =

rx,j(1−rx,j),

so sx,j is an unbiased estimator of rx,j and concentrates around it as N grows. Under the standard binarization rule Yx,j = I[Kx,j > 0], the resulting step-level probability of the positive label is

Pr(Yx,j = 1 | ϕx,j) = 1 − (1 − rx,j)N which is strictly increasing in rx,j. Since rx,j is deter-

mined by ϕx,j, we model this probability using the teacher q∗(ϕx,j) = σ(⟨w∗,ϕx,j⟩) in Eq. (1). These observations link the MC score sx,j, the binarized label Yx,j, and the teacher probability q∗(ϕx,j) through the underlying success probability rx,j. We next quantify how this link translates into an effective noise level for positive labels.

To formalize reliability, we fix a threshold τ ∈ (0,1) and define a step to be τ-reliable if its one-shot success probability is at least τ, i.e., Zx,j := I[rx,j ≥ τ]. We then relate this reliability notion to what the MC annotator actually observes. Specifically, we let the unobserved success probability rx,j vary across steps and model it with a Beta distribution rx,j ∼ Beta(a,b). Conditional on rx,j, the number of successful continuations among the N MC samples satisfies Kx,j | rx,j ∼ Binomial(N,rx,j). Under this Beta–Binomial model, observing Kx,j = k yields the posterior rx,j | Kx,j = k ∼ Beta(a + k,b + N − k). This induces an effective noise level for positive steps, defined as ηeff(k) := Pr(Zx,j = 0 | Kx,j = k)

= Pr(rx,j < τ | Kx,j = k) = Iτ(a + k,b + N − k) where Iτ(·,·) denotes the regularized incomplete beta function. For Kx,j > 0, ηeff(k) is exactly the posterior probability that a positive step with Kx,j = k is pseudo-positive, i.e., τ-unreliable. Moreover, ηeff(k) is strictly decreasing in k (Lemma 3 in Appendix F.5). Consequently, lowMC positives (Kx,j > 0 but small sx,j = Kx,j/N) have large ηeff and are likely to be τ-unreliable, so they behave like high-noise samples. Under the label-flipping noise model of Eq. (6), this corresponds to operating at a larger noise rate η ≈ ηeff(Kx,j), which increases the noise term η∥ϕx,j∥2 and decreases the signal term q∗(ϕx,j)(1 − q∗(ϕx,j))∥ϕx,j∥2.

Averaging sx,j over positive steps in a rollout thus yields a natural rollout-level reliability estimate: since ηeff(k) decreases monotonically with the MC score sx,j = Kx,j/N, rollouts whose positive steps have higher average MC scores also have smaller average ηeff(Kx,j), so their gradient updates are less affected by label noise.

##### Mixture and Reliability Couple Multiplicatively

Theoretical Finding 6: Rollouts are most informative when both label mixture and reliability are high.

Finally, we combine the previous two parts and analyze how label mixture and reliability jointly shape the rolloutlevel signal term in Eq. (6). We average Eq. (6) over n steps in rollout x, let qx,j := q∗(ϕx,j) and the rollout-level q-dependent signal term takes the form

n

1 n

(1 − 4ηx,j)qx,j(1 − qx,j)∥ϕx,j∥2.

S(x) :=

j=1

where ηx,j is the step-dependent noise rate, and can be

approximated by ηeff(Kx,j). The signal term S(x) is determined by the multiplication of two factors: the term qx,j(1 − qx,j) favors rollouts with uncertain steps; while (1 − 4ηx,j) favors smaller label noise ηx,j (larger Kx,j or sx,j based on the monotonicity). Therefore, S(x) of a rollout is large only when the multiplication of uncertainty and reliability is large. In practice, we approximate the teacher uncertainty θx := q¯x(1 − q¯x) using the observable label mixture pˆx(1−pˆx). Section 4 distills this into the BalancedInformation Score used in our data selection method.

#### 4. Balanced-Information Score

Motivated by the empirical results above and the theoretical analysis, we introduce the Balanced-Information Score (BIS) as a rollout-level scoring mechanism that prioritizes informative rollouts for MPRM training.

Setting. Consider an MPRM training set where each rollout x contains n annotated steps with MC scores {sj}nj=1 ⊂ [0,1]. Following the standard binarization rule, each step is assigned a hard label yj = I[sj > 0].

Rollout-level Quantities. For a rollout x, define the positive-step ratio ppos(x) = n1 nj=1 yj, used to quantify label mixture, and a rollout-level reliability measure R(x):

 

1 npos j:y

j=1 sj, npos > 0, 1, npos = 0,

R(x) =



where npos = nj=1 yj. Rollouts with npos = 0 contain no positive labeled steps to average over, so we fix R(x) = 1.

Balanced-Information Score. We define the BalancedInformation Score (BIS) of rollout x as:

BIS(x) = ppos(x) 1 − ppos(x) + α R(x), where the hyperparameter α > 0 is a small smoothing constant that assigns a non-zero weight to low-mixture rollouts. The term ppos(1 − ppos) favors mixed rollouts that contain both correct and incorrect steps, while R(x) favors rollouts whose positive steps are reliably correct under MC estimation. Therefore, BIS(x) is highest for rollouts that provide both clear negative signals and trustworthy positive anchors.

Subset Construction with Keep Ratio ρ. Given a global keep-ratio ρ ∈ (0,1) applied uniformly across all sources, we assign BIS to every MC-annotated rollout to build a dataefficient training set. Within each source dataset, we rank rollouts by BIS(x) in descending order and keep the top ρ fraction. We then concatenate the selected rollouts over all sources to form the BIS-selected subset. This procedure only relies on existing step-level MC scores and does not require additional supervision or extra model calls. The

- Table 2. Overall micro-F1 and per-source macro-F1 on VisualProcessBench for full-data training and sub-sampled rollouts under different keep ratios ρ. “Soft” uses the raw MC scores as continuous soft targets; “Hard (τ)” uses binary labels with threshold τ. Bold numbers denote the column-wise maximum within each subset group. “Base” denotes the original backbone model without any additional training.

Dataset Overall MathVision MathVerse MMMU DynaMath WeMath InternVL2.5-8B

Dataset Overall MathVision MathVerse MMMU DynaMath WeMath Qwen2.5-VL-7B

Base 52.28 52.40 52.04 50.21 54.85 49.95

Base 49.68 50.22 49.58 49.85 49.62 48.51

###### Full-Data (565k – 1,100 steps)

###### Full-Data (565k – 1,100 steps)

Hard (τ=0) 65.12 65.77 65.43 61.84 66.17 63.56

Hard (τ=0) 65.57 66.05 65.29 63.23 66.24 66.40

- Hard (τ=1/N) 64.26 64.91 63.91 61.07 65.44 64.80

- Hard (τ=2/N) 63.02 61.61 62.92 59.70 65.83 65.09 Soft 61.54 60.78 60.47 62.05 62.91 64.38

- Hard (τ=1/N) 65.27 66.17 64.53 63.83 66.60 64.55

- Hard (τ=2/N) 62.72 62.33 62.18 62.56 63.37 64.65 Soft 62.23 62.17 61.25 61.44 62.88 65.55

###### 5% Subsets (28k – 55 steps)

###### 5% Subsets (28k – 55 steps)

Random-5% 63.34 64.95 62.42 58.94 65.91 62.57 BIS-5% 64.51 66.66 64.53 60.30 63.80 64.40

Random-5% 53.54 54.38 52.82 55.85 52.80 53.04 BIS-5% 64.42 66.69 64.20 62.72 63.34 63.12

###### 10% Subsets (56k – 110 steps)

###### 10% Subsets (56k – 110 steps)

Random-10% 62.86 64.65 62.14 60.25 62.98 63.25 BIS-10% 65.46 66.90 65.07 63.35 65.56 65.10

Random-10% 61.99 63.42 61.85 58.37 62.24 61.97 BIS-10% 64.63 66.05 64.14 61.33 64.64 66.05

###### 15% Subsets (84k – 165 steps)

###### 15% Subsets (84k – 165 steps)

Random-15% 63.27 65.06 63.00 57.23 64.17 63.96 BIS-15% 64.98 67.09 64.44 61.80 64.58 65.40

Random-15% 59.82 59.32 60.58 54.97 61.32 60.36 BIS-15% 65.29 66.50 64.03 66.40 64.80 66.58

###### 25% Subsets (141k – 275 steps)

###### 25% Subsets (141k – 275 steps)

Random-25% 63.37 64.49 62.60 58.32 65.83 63.67 BIS-25% 65.46 67.98 64.86 60.49 65.72 65.59

Random-25% 64.44 65.87 63.38 61.99 66.32 63.44 BIS-25% 65.53 66.84 64.48 63.60 66.19 66.66

###### 35% Subsets (198k – 385 steps)

###### 35% Subsets (198k – 385 steps)

Random-35% 63.52 65.50 63.49 58.01 64.00 63.08 BIS-35% 64.98 67.25 64.47 59.61 65.79 64.82

Random-35% 64.77 66.45 64.43 60.47 65.34 64.84 BIS-35% 65.69 66.77 65.50 63.37 65.65 65.99

###### 50% Subsets (283k – 550 steps)

###### 50% Subsets (283k – 550 steps)

Random-50% 64.02 64.55 63.94 60.25 65.38 64.14 BIS-50% 65.00 65.84 63.79 63.03 66.66 66.03

Random-50% 64.54 66.52 64.27 59.74 65.73 62.89 BIS-50% 65.02 67.22 64.30 60.80 65.62 64.99

downstream MPRM can be trained from this ρ-subset with the same training setup of Section 2.1.

#### 5. Experiments

##### 5.1. Training Setup

We conduct experiments with two backbones: InternVL2.58B (Chen et al., 2024c) and Qwen2.5-VL-7B (Bai et al., 2025), and evaluate different methods on VisualProcessBench (Wang et al., 2025b). To study data efficiency, we sub-sample rollouts with keep ratios ρ ∈ {5,10,15,25,35,50}% and compare BIS-ρ against Random-ρ under matched budgets. We additionally include heuristic subset baselines as ablations at selected budgets. All models are trained for a single pass over their retained rollouts, with training steps and learning-rate scheduling scaled proportionally to ρ. We also report Best-of-N evaluation with MPRM reranking. Full training details are provided in Appendix C.

##### 5.2. Main Results

BIS recovers full-data performance at small ratios and consistently outperforms random sub-sampling. Table 2 compares BIS-ρ to Random-ρ under identical rollout budgets. Across both backbones, BIS reaches full-data performance at small ρ and remains consistently stronger than random sub-sampling, with the largest gains at small rollout budgets. For InternVL2.5-8B, BIS already matches the full-data performance at ρ=10%, reaching an overall

micro-F1 of 65.46%, a +2.6 points gain over the random baseline Random-10%, while using only one tenth of the rollouts and updates. For Qwen2.5-VL-7B, BIS shows even larger advantages in the extremely low-budget regime: it improves over random sub-sampling by +10.9 points at ρ=5% and +5.5 points at ρ=15%, and already reaches the full-data reference at ρ=25%. We report the complete training dynamics for all keep ratios in Appendix G, including both overall micro-F1 and per-source macro-F1, and show that BIS maintains clear advantages over random sub-sampling throughout training. BIS also shows a clear scaling trend with ρ: performance improves rapidly at small budgets, peaks at a moderate keep ratio, and can slightly drop afterwards, as increasing ρ mainly adds lower-BIS rollouts that are less informative under our “mixture × reliability” criterion. Since ρ=25% performs strongly for both backbones, we use it for analysis in the following experiments.

Effect of labeling scheme. We further study the impact of the threshold for binarizing MC scores into hard labels and show the results in Table 2. First, training with soft labels is clearly inferior to default hard-label scheme. This is consistent with MC scores being noisy and coarsely discretized estimates, thus encouraging the MPRM to fit sampling noises. Second, increasing the binarization threshold to above 0 consistently degrades performance, suggesting that low-MC labels conflate hard cases with noisy pseudo-positives and that stricter thresholding can mislabel hard cases as negatives. BIS therefore avoids this ambiguous low-MC regime by prioritizing mixed yet reliable rollouts.

- Table 3. Best-of-N evaluation on four benchmarks with MPRM reranking trained on different training sets.

Model MM-K12 OlympiadBench MathVerse MathVista InternVL2.5-8B 33.13 8.65 35.31 52.77

+MPRMFull-Data 39.00 ↑5.87 12.00 ↑3.35 39.41 ↑4.10 57.50 ↑4.73 +MPRMRandom-25% 39.40 ↑6.27 11.33 ↑2.68 39.41 ↑4.10 58.20 ↑5.43 +MPRMBIS-25% 41.00 ↑7.87 12.67 ↑4.02 40.89 ↑5.58 59.00 ↑6.23

Table 4. Ablations of BIS under a 25% rollout budget.

Subset Overall MathVision MathVerse MMMU DynaMath WeMath

BIS-25% 65.46 67.98 64.86 60.49 65.72 65.59 Mixed-25% 64.70 66.32 64.78 58.65 65.66 64.51

∆ ↓0.76 ↓1.66 ↓0.08 ↓1.84 ↓0.06 ↓1.08 Reliable-25% 62.75 62.12 63.14 60.52 63.85 63.14

∆ ↓2.71 ↓5.86 ↓1.72 ↑0.03 ↓1.87 ↓2.45 Low-MC-25% 64.18 66.31 64.31 59.40 64.16 62.97

∆ ↓1.28 ↓1.67 ↓0.55 ↓1.09 ↓1.56 ↓2.62

BIS improves best-of-N reranking. We further evaluate MPRM in a practical best-of-N reranking setting with N = 16 candidates per problem on four different benchmarks (MM-K12 (Du et al., 2025), OlympiadBench (He et al., 2024), MathVerse (Zhang et al., 2024), and MathVista (Lu et al., 2024)) in Table 3. The full evaluation protocol is described in Appendix C. Consistent with the main results, MPRMs trained on BIS-selected subsets achieve the strongest best-of-N performance across all benchmarks, outperforming both Random-25% and the full-data MPRM. This suggests that BIS yields robust improvements in MPRM effectiveness beyond benchmark-specific behavior.

BIS favors moderate-R(x) rollouts. Figure 3 shows the distribution of the reliability term R(x) on VisualPRM400Kv1.1 for all rollouts and the BIS-25% subset. Additional per-source statistics are provided in Appendix H. The black curve demonstrates the selected data coverage over the full data. BIS strongly suppresses the low-reliability rollouts where R(x) are small which tend to include noisy step labels. Meanwhile, the coverage peaks at moderate R(x) (around 0.2–0.6) and decreases as R(x) becomes very large, showing that BIS does not simply maximize R(x). This is because BIS jointly considers reliability and mixture: rollouts with large R(x) may still have low mixture, resulting in a low overall BIS score. Consistently, the right panel shows that BIS favors rollouts with higher mixture, explaining why high-R(x) rollouts are not always preferred.

[Figure 5]

Figure 3. Distributions of the reliability term R(x) (left) and mixture term p(x)(1 − p(x)) (right) on VisualPRM400K-v1.1, comparing all rollouts and the BIS-25% subset. The black curve shows the coverage (Selected / All).

Table 5. Sensitivity of BIS to the smoothing constant α under a fixed 25% rollout budget.

Subset Overall MathVision MathVerse MMMU DynaMath WeMath

α = 0.02 64.84 66.63 64.69 60.49 64.78 65.27 α = 0.05 65.46 67.98 64.86 60.49 65.72 65.59 α = 0.08 64.86 67.10 64.47 59.96 64.88 65.35

separate sources. Low-MC-25% shows a similar trend and remains weaker than BIS, indicating that heuristic filtering alone is insufficient to match BIS selection. In contrast, using only the reliability score (Reliable-25%) is clearly insufficient: it lags behind BIS on nearly all benchmarks and shows an advantage only on MMMU where performances are very close. This aligns with the analysis in Section 3.3 that the best performance is obtained only when both mixture and reliability are considered, where the mixture term provides contrast between positive and negative labels, and the reliability term steers away from noisy low-MC positive labels. We also plot the complete learning curves in Appendix E, and further demonstrate that BIS-25% yields the highest or near-highest performance at almost all steps.

BIS is robust to α. We ablate the smoothing constant α for the mixture term, which controls the lower-bound when ppos(1 − ppos) is small. Table 5 indicates that performance is broadly stable across different α values, with a consistent best choice at α = 0.05. Overly small α can underweight low-mixture yet reliable trajectories and reduce the diversity of retained supervision, whereas overly large α weakens the mixture term and shifts selection toward reliability-only ranking. The intermediate value best balances these effects.

Ablation study: both mixture and reliability matter. Table 4 ablates the two components of BIS under the same 25% rollout budget. Mixed-25% and Low-MC-25% are heuristic subsets of Section 2.3, while Reliable-25% retains the top 25% of rollouts ranked by R(x). We observe that only using the mixture score (Mixed-25%) is competitive but still consistently weaker than the BIS both on average and on

#### 6. Conclusion

We study how to select Monte Carlo–annotated multimodal reasoning rollouts for training MPRMs. We found that randomly discarding most rollouts only mildly degrades performance, indicating that current training sets contain substantial redundancy. Our theoretical analysis explains

that informative gradient updates concentrate on uncertain yet reliably labeled steps, while low-MC pseudo-positives mainly add variance. We propose the Balanced-Information Score (BIS), which ranks rollouts by label mixture and reliability using only the MC signals already stored in the dataset. Empirical results demonstrate that BIS-selected subsets match or surpass full-data MPRM performance with as little as 10% of rollouts. Overall, our study provides a data-centric principle for curating future MPRM corpora.

#### Impact Statement

This work aims to improve the data and compute efficiency of training multimodal process reward models. Successful adaptation of our method to practical model training will benefit the society by reducing energy cost.

#### Acknowledgments

We would like to thank Han Li and Zhangchen Xu for their valuable insights. This research was supported in part by the NVIDIA Academic Grant Program and WashU Ignite Interdisciplinary Grants.

#### References

Anthropic. The claude 3 model family: Opus, sonnet, haiku. Available at: https://www.anthropic.com/ne ws/claude-3-family, 2024. Accessed: 2025-1223.

Bach, F. and Moulines, E. Non-strongly-convex smooth stochastic approximation with convergence rate o (1/n). Advances in neural information processing systems, 26, 2013.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large visionlanguage model with versatile abilities. arXiv preprint arXiv:2308.12966, 1(2):3, 2023.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Bottou, L. and Bousquet, O. The tradeoffs of large scale learning. Advances in neural information processing systems, 20, 2007.

Cao, J. and Xiao, J. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th international conference on computational linguistics, pp. 1511–1520, 2022.

Cao, Q. and Xie, P. Dreamprm-1.5: Unlocking the potential

of each instance for multimodal process reward model training. arXiv preprint arXiv:2509.05542, 2025.

Cao, Q., Wang, R., Zhang, R., Somayajula, S. A., and Xie, P. Dreamprm: Domain-reweighted process reward model for multimodal reasoning. arXiv preprint arXiv:2505.20241, 2025.

Chang, S., Palzer, D., Li, J., Fosler-Lussier, E., and Xiao, N. MapQA: A dataset for question answering on choropleth maps. In NeurIPS 2022 First Table Representation Workshop, 2022. URL https://openreview.net /forum?id=znKbVjeR0yI.

Chen, G., Liao, M., Li, C., and Fan, K. Alphamath almost zero: Process supervision without process. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. URL https: //openreview.net/forum?id=VaXnxQ3UKo.

Chen, J., Li, T., Qin, J., Lu, P., Lin, L., Chen, C., and Liang, X. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 3313–3323, 2022.

Chen, Q., Qin, L., Zhang, J., Chen, Z., Xu, X., and Che, W. M3cot: A novel benchmark for multi-domain multi-step multi-modal chain-of-thought. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8199–8221, 2024b.

Chen, X., Liu, B., Wang, X., Wang, Y., and Lu, C. Vrprm: Process reward modeling via visual reasoning. arXiv preprint arXiv:2508.03556, 2025.

Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024c.

Cui, G., Yuan, L., Wang, Z., Wang, H., Zhang, Y., Chen, J., Li, W., He, B., Fan, Y., Yu, T., et al. Process reinforcement through implicit rewards. arXiv preprint arXiv:2502.01456, 2025.

Ding, Y., Shi, X., Li, J., Tu, Z., Zhang, M., et al. Scan: Self-denoising monte carlo annotation for robust process reward learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Dong, G., Zhang, C., Deng, M., Zhu, Y., Dou, Z., and Wen, J.-R. Progressive multimodal reasoning via active retrieval. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume

1: Long Papers), pp. 3579–3602, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.180. URL https://aclanthology.org/2025.ac l-long.180/.

Du, L., Meng, F., Liu, Z., Zhou, Z., Luo, P., Zhang, Q., and Shao, W. Mm-prm: Enhancing multimodal mathematical reasoning with scalable step-level supervision. arXiv preprint arXiv:2505.13427, 2025.

Duan, K., Liu, Z., Mao, X., Pang, T., Chen, C., Chen, Q., Shieh, M. Q., and Dou, L. Efficient process reward model training via active learning. arXiv preprint arXiv:2504.10559, 2025.

Fan, K., Feng, K., Lyu, H., Zhou, D., and Yue, X. Sophiavlr1: Reinforcing mllms reasoning with thinking reward. arXiv preprint arXiv:2505.17018, 2025.

Gao, J., Pi, R., Zhang, J., Ye, J., Zhong, W., Wang, Y., HONG, L., Han, J., Xu, H., Li, Z., and Kong, L. GLLaVA: Solving geometric problem with multi-modal large language model. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=px16 74Wp3C.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Han, J., Buntine, W., and Shareghi, E. Uncertainty-based methods for automated process reward data construction and output aggregation in mathematical reasoning. arXiv preprint arXiv:2508.01773, 2025.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Hosu, V., Lin, H., Sziranyi, T., and Saupe, D. Koniq-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing, 29:4041–4056, 2020.

Hu, P., Zhang, Z., Chang, Q., Liu, S., Ma, J., Du, J., Zhang, J., Liu, Q., Gao, J., Ma, F., et al. Prm-bas: Enhancing multimodal reasoning through prm-guided beam annealing search. arXiv preprint arXiv:2504.10222, 2025.

Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., and Jawahar, C. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pp. 1516–1520. IEEE, 2019.

Johnson, J., Hariharan, B., Van Der Maaten, L., Fei-Fei, L., Lawrence Zitnick, C., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017.

Kafle, K., Price, B., Cohen, S., and Kanan, C. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5648–5656, 2018.

Kahou, S. E., Michalski, V., Atkinson, A., K´ad´ar, A.,´ Trischler, A., and Bengio, Y. Figureqa: An annotated figure dataset for visual reasoning. arXiv preprint arXiv:1710.07300, 2017.

Kazemi, M., Alvari, H., Anand, A., Wu, J., Chen, X., and Soricut, R. Geomverse: A systematic evaluation of large models for geometric reasoning. In AI for Math Workshop@ ICML 2024, 2024.

Kembhavi, A., Salvato, M., Kolve, E., Seo, M., Hajishirzi, H., and Farhadi, A. A diagram is worth a dozen images. In European conference on computer vision, pp. 235–251. Springer, 2016.

Khalifa, M., Agarwal, R., Logeswaran, L., Kim, J., Peng, H., Lee, M., Lee, H., and Wang, L. Process reward models that think. arXiv preprint arXiv:2504.16828, 2025.

Kuang, P., Wang, X., Liu, W., Dong, J., Xu, K., and Wang, H. Tim-prm: Verifying multimodal reasoning with toolintegrated prm. arXiv preprint arXiv:2511.22998, 2025.

Li, Z., Wang, X., Stengel-Eskin, E., Kortylewski, A., Ma, W., Van Durme, B., and Yuille, A. L. Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14963– 14973, 2023.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum ?id=v8L0pN6EOi.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Liu, W., Li, J., Zhang, X., Zhou, F., Cheng, Y., and He, J. Diving into self-evolving training for multimodal reasoning. arXiv preprint arXiv:2412.17451, 2024.

Lu, P., Gong, R., Jiang, S., Qiu, L., Huang, S., Liang, X., and Zhu, S.-C. Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning. In Zong, C., Xia, F., Li, W., and Navigli, R. (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 6774–6786, Online, August 2021a. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.528. URL https: //aclanthology.org/2021.acl-long.528/.

Lu, P., Qiu, L., Chen, J., Xia, T., Zhao, Y., Zhang, W., Yu, Z., Liang, X., and Zhu, S.-C. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021b. URL https:

//openreview.net/forum?id=uXa9oBDZ9V1.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=KUNz EQMWU7.

Luo, L., Liu, Y., Liu, R., Phatale, S., Guo, M., Lara, H., Li, Y., Shu, L., Zhu, Y., Meng, L., et al. Improve mathematical reasoning in language models by automated process supervision. arXiv preprint arXiv:2406.06592, 2024.

Luo, R., Zheng, Z., Wang, L., Wang, Y., Ni, X., Lin, Z., Jiang, S., Yu, Y., Shi, C., Chu, R., et al. Unlocking multimodal mathematical reasoning via process reward model. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Ma, Q., Zhou, H., Liu, T., Yuan, J., Liu, P., You, Y., and Yang, H. Let’s reward step by step: Step-level reward model as the navigators for reasoning. arXiv preprint arXiv:2310.10080, 2023.

Masry, A., Do, X. L., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of

the association for computational linguistics: ACL 2022, pp. 2263–2279, 2022.

Mathew, M., Karatzas, D., and Jawahar, C. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., and Jawahar, C. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1697–1706, January 2022.

Moulines, E. and Bach, F. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. Advances in neural information processing systems, 24, 2011.

Nesterov, Y. et al. Lectures on convex optimization, volume

137. Springer, 2018.

Ong, B., Pala, T. D., Toh, V., Tjhi, W. C., and Poria, S. Training vision-language process reward models for testtime scaling in multimodal reasoning: Key insights and lessons learned. arXiv preprint arXiv:2509.23250, 2025.

OpenAI. Gpt-4o system card, 2024. URL https://op enai.com/index/gpt-4o-system-card/.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

Qiao, R., Tan, Q., Dong, G., MinhuiWu, M., Sun, C., Song, X., Wang, J., Gongque, Z., Lei, S., Zhang, Y., et al. Wemath: Does your large multimodal model achieve humanlike mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 20023–20070, 2025.

Qwen Team. QVQ: To see the world with wisdom. Available at: https://qwenlm.github.io/blog/qvq-7 2b-preview/, December 2024. Accessed: 2025-1223.

Rasley, J., Rajbhandari, S., Ruwase, O., and He, Y. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 3505–3506, 2020.

Seo, M., Hajishirzi, H., Farhadi, A., Etzioni, O., and Malcolm, C. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015

conference on empirical methods in natural language processing, pp. 1466–1476, 2015.

Shalev-Shwartz, S. and Ben-David, S. Understanding machine learning: From theory to algorithms. Cambridge university press, 2014.

Shi, W., Hu, Z., Bin, Y., Liu, J., Yang, Y., Ng, S. K., Bing, L., and Lee, R. K.-W. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 4663–4680, 2024.

Singh, S., Yadav, A., Jain, J., Shi, H., Johnson, J., and Desai, K. Benchmarking object detectors with coco: A new path forward. In European Conference on Computer Vision, pp. 279–295. Springer, 2024.

Suhr, A., Zhou, S., Zhang, A., Zhang, I., Bai, H., and Artzi, Y. A corpus for reasoning about natural language grounded in photographs. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 6418–6428, 2019.

Sun, W., Du, Q., Cui, F., and Zhang, J. An efficient and precise training data construction framework for process-supervised reward model in mathematical reasoning. arXiv preprint arXiv:2503.02382, 2025.

Tan, X., Yao, T., Qu, C., Li, B., Yang, M., Lu, D., Wang, H., Qiu, X., Chu, W., Xu, Y., et al. Aurora: Automated training framework of universal process reward models via ensemble prompting and reverse verification. arXiv preprint arXiv:2502.11520, 2025.

Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ram´e, A., Rivi`ere, M., et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

Tu, H., Feng, W., Chen, H., Liu, H., Tang, X., and Xie, C. Vilbench: A suite for vision-language process reward modeling. arXiv preprint arXiv:2503.20271, 2025.

Wang, H., Xiong, W., Xie, T., Zhao, H., and Zhang, T. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. arXiv preprint arXiv:2406.12845, 2024a.

Wang, J., Fang, M., Wan, Z., Wen, M., Zhu, J., Liu, A., Gong, Z., Song, Y., Chen, L., Ni, L. M., et al. Openr: An open source framework for advanced reasoning with

large language models. arXiv preprint arXiv:2410.09671,

- 2024b.

Wang, K., Pan, J., Shi, W., Lu, Z., Ren, H., Zhou, A., Zhan, M., and Li, H. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169,

- 2024c.

Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426–9439, 2024d.

Wang, S., Liu, Z., Wei, J., Yin, X., Li, D., and Barsoum, E. Athena: Enhancing multimodal reasoning with data-efficient process reward models. arXiv preprint arXiv:2506.09532, 2025a.

Wang, W., Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Zhu, J., Zhu, X., Lu, L., Qiao, Y., et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024e.

Wang, W., Lv, Q., Yu, W., Hong, W., Qi, J., Wang, Y., Ji, J., Yang, Z., Zhao, L., XiXuan, S., et al. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475– 121499, 2024f.

- Wang, W., Gao, Z., Chen, L., Chen, Z., Zhu, J., Zhao, X., Liu, Y., Cao, Y., Ye, S., Zhu, X., et al. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291, 2025b.
- Wang, X., Wang, P., Pei, J., Shen, W., Peng, Y., Hao, Y., Qiu, W., Jian, A., Xie, T., Song, X., et al. Skywork-vl reward: An effective reward model for multimodal understanding and reasoning. arXiv preprint arXiv:2505.07263, 2025c.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pp. 38–45, 2020.

Xiong, W., Zhao, W., Yuan, W., Golovneva, O., Zhang, T., Weston, J., and Sukhbaatar, S. Stepwiser: Stepwise generative judges for wiser reasoning. arXiv preprint arXiv:2508.19229, 2025.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and

reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhang, J., Yan, Y., Zheng, K., Zou, X., Dai, S., and Hu, X. Gm-prm: A generative multimodal process reward model for multimodal mathematical reasoning. arXiv preprint arXiv:2508.04088, 2025a.

Zhang, L., Hosseini, A., Bansal, H., Kazemi, M., Kumar, A., and Agarwal, R. Generative verifiers: Reward modeling as next-token prediction. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id=Ccwp 4tFEtE.

Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Qiao, Y., et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024.

Zhang, R., Wei, X., Jiang, D., Guo, Z., Zhang, Y., Tong, C., Liu, J., Zhou, A., Zhang, S., Gao, P., and Li, H. MAVIS: Mathematical visual instruction tuning with an automatic data engine. In The Thirteenth International Conference on Learning Representations, 2025c. URL https:// openreview.net/forum?id=MnJzJ2gvuf.

Zhang, Y., Wu, Y., Zhang, H., Li, W., Chen, H., Li, G., Han, Z., and Tresp, V. Groundedprm: Tree-guided and fidelityaware process reward modeling for step-level reasoning. In NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning, 2025d.

Zhang, Y.-F., Lu, X., Hu, X., Fu, C., Wen, B., Zhang, T., Liu, C., Jiang, K., Chen, K., Tang, K., et al. R1-reward: Training multimodal reward model through stable reinforcement learning. arXiv preprint arXiv:2505.02835, 2025e.

- Zhang, Y.-F., Yang, H., Zhang, H., Shi, Y., Chen, Z., Tian, H., Fu, C., Wang, H., Wu, K., Cui, B., et al. Basereward: A strong baseline for multimodal reward model. arXiv preprint arXiv:2509.16127, 2025f.
- Zhang, Z., Zheng, C., Wu, Y., Zhang, B., Lin, R., Yu, B., Liu, D., Zhou, J., and Lin, J. The lessons of developing process reward models in mathematical reasoning. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 10495–10516, Vienna, Austria, July 2025g. Association for Computational Linguistics. ISBN 979-889176-256-5. doi: 10.18653/v1/2025.findings-acl.547. URL https://aclanthology.org/2025.fi ndings-acl.547/.

Zheng, C., Zhu, J., Ou, Z., Chen, Y., Zhang, K., Shan, R., Zheng, Z., Yang, M., Lin, J., Yu, Y., et al. A survey of process reward models: From outcome signals to process supervisions for large language models. arXiv preprint arXiv:2510.08049, 2025.

Zhu, J., Zheng, C., Lin, J., Du, K., Wen, Y., Yu, Y., Wang, J., and Zhang, W. Retrieval-augmented process reward model for generalizable mathematical reasoning. arXiv preprint arXiv:2502.14361, 2025.

Zou, C., Guo, X., Yang, R., Zhang, J., Hu, B., and Zhang, H. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. In The Thirteenth International Conference on Learning Representations, 2025.

# Appendix

### Table of Contents

- A Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

- A.1 Multimodal Process Reward Models under MC Process Supervision . . . . . . . 15
- A.2 Data-efficient Process Supervision . . . . . . . . . . . . . . . . . . . . . . . . 15

- B VisualProcessBench Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C Experimental Setup and Implementation Details . . . . . . . . . . . . . . . . . . . . 17
- D Extended Results for Random Sub-sampling . . . . . . . . . . . . . . . . . . . . . . 19
- E Extended Results for 25% Subsets . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- F Theoretical Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- F.1 A Canonical Logistic Case for the Scaling Decomposition . . . . . . . . . . . . 22
- F.2 Step-wise Gradient Variance . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- F.3 Symmetric Label Noise . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- F.4 Rollout-Level Mixture and Representation Norms . . . . . . . . . . . . . . . . 26
- F.5 MC-induced Pseudo-positive Probability and Monotonicity . . . . . . . . . . . . 28

- G Training Dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- H Per-source BIS Histograms . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- I Case Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41

#### A. Related Work

##### A.1. Multimodal Process Reward Models under Automated Monte Carlo Process Supervision

Recent work (Zheng et al., 2025) shows that MPRMs improve multimodal reasoning both as dense rewards for reinforcement learning fine-tuning (Luo et al., 2025; Wang et al., 2025c; Liu et al., 2024; Fan et al., 2025) and as stepwise verifiers for inference-time trajectory ranking (Zhang et al., 2025a; Cao et al., 2025; Cao & Xie, 2025; Wang et al., 2025a; Tu et al., 2025; Hu et al., 2025). Unlike outcome rewards that score only the final answer (Lightman et al., 2024; Zhang et al., 2025b;f;e; Wang et al., 2024a), an MPRM (Chen et al., 2025; Ong et al., 2025; Kuang et al., 2025) provides dense, step-level supervision by mapping each intermediate multimodal reasoning state to a real-valued “on-track” score conditioned on the input images and text. Most standard MPRM corpora are built from Monte Carlo (MC) estimates computed on reasoning prefixes, with VisualPRM400K (Wang et al., 2025b) as a representative example. One common approach samples multiple continuations from each prefix and uses the empirical success rate to score step correctness (Wang et al., 2024d). A complementary line replaces plain sampling with structured search via Monte Carlo Tree Search, improving the stability of error localization and supervision signals (Luo et al., 2024; Wang et al., 2024b; Chen et al., 2024a). Despite their differences, MC-based annotators are inherently noisy under finite sampling and long-horizon multimodal reasoning, yielding unstable labels and low-success “pseudo-positives” that tightening the binarization threshold cannot simply fix and can even hurt MPRM performance (Wang et al., 2025b). Taken together, MC-annotated supervision is plentiful but highly uneven in information content, making rollout-level prioritization crucial for efficient and stable MPRM training.

##### A.2. Data-efficient Process Supervision

Recent work on data-efficient process supervision for PRMs can be broadly grouped into three complementary categories. The first line of work optimizes the annotation pipeline itself (Han et al., 2025; Sun et al., 2025; Zhang et al., 2025d; Wang et al., 2025a; Zhang et al., 2025g). These methods improve Monte Carlo or search-based annotators with techniques such as Monte Carlo tree search, tool grounding, and consensus-based filtering, enabling each trajectory to yield higher-quality step labels from fewer or cheaper model calls. A second line of work focuses on learning robust process supervision from weak, noisy, or indirect feedback (Ding et al., 2025; Xiong et al., 2025; Khalifa et al., 2025; Cui et al., 2025; Chen et al., 2024a). These approaches design objectives and model forms that allow PRMs to learn effectively from imperfect MC labels and, in some cases, even from outcome-only signals, thereby reducing reliance on expensive, high-quality process annotations. Our work belongs to a third line that focuses on data selection and supervision allocation. DreamPRM (Cao et al., 2025) and DreamPRM-1.5 (Cao & Xie, 2025) adjust dataset and example weights via bi-level optimization, while ACTPRM (Duan et al., 2025) and SCAN (Ding et al., 2025) select which samples to query or refine with expensive Monte Carlo estimation under a limited annotation budget. In contrast, our rollout-level BIS is a scalar score computed post hoc from existing MC statistics, enabling data selection with no extra model calls, relabeling, or changes to the underlying PRM architecture or training objective.

#### B. VisualProcessBench Statistics

Table 6 summarizes the composition of VisualProcessBench (Wang et al., 2025b), including the number of problems per source dataset, the distribution of source solutions across base models, the breakdown of correct/incorrect/neutral steps, and basic length statistics. In total, the benchmark contains 2,866 solution trajectories with 26,950 annotated steps, providing a reasonably large and diverse testbed for step-level evaluation.

Table 6. Statistics of VisualProcessBench (Wang et al., 2025b).

Item Value Dataset Composition Total Samples 2866

- – MMMU (Yue et al., 2024) 267
- – MathVision (Wang et al., 2024c) 712
- – MathVerse (Zhang et al., 2024) 1026
- – DynaMath (Zou et al., 2025) 570
- – WeMath (Qiao et al., 2025) 291

Source Solutions Source Solutions 2866

- – GPT-4o (OpenAI, 2024) 870
- – Claude-3.5-Sonnet (Anthropic, 2024) 865
- – QvQ-72B-Preview (Qwen Team, 2024) 825
- – InternVL2.5-78B (Chen et al., 2024c) 306

Steps Total Steps 26950

- – Correct Steps 16585
- – Incorrect Steps 7691
- – Neutral Steps 2674

Length Statistics Query Word Length Quartile (22, 24, 50) Response Word Length Quartile (137, 193, 552) Step Word Length Quartile (13, 31, 67) Number of Steps per Solution 9.4

#### C. Experimental Setup and Implementation Details

This appendix provides the full experimental setup omitted from the main text for brevity, including the model and training data, the MPRM training objective, optimization and hardware settings, and the evaluation protocol.

Table 7. Training, model, and hardware hyperparameters (shared across all data-selection conditions).

##### Item Value

Optimization

Optimizer AdamW Learning Rate 1 × 10−5 Weight Decay 0.05 AdamW Betas (0.9,0.999) AdamW ϵ 1 × 10−8 LR Schedule Linear Warmup + Cosine Decay Warmup Ratio 0.05 Gradient Clipping Enabled via DeepSpeed (gradient clipping=auto, using the Trainer

default max grad norm) Precision bf16

Batching and Training Budget

Per-device Batch Size 2 Gradient Accumulation Steps 64 Global Batch Size B = 512 Epochs 1 (single pass; default for all experiments unless otherwise noted) Max Sequence Length 8192 (truncate from the end) Optimization Steps T = ⌈N/B⌉ for a pool of N rollouts

Model, Input, and Hardware Backbone InternVL2.5-8B & Qwen2.5-VL-7B Trainable Modules LLM + multimodal fusion MLP (vision backbone frozen) Image Size InternVL2.5-8B: 448, dynamic resolution enabled; max 6 patches.

Qwen2.5-VL-7B: dynamic resizing (min/max=784/200704). GPUs 4 × NVIDIA H100 80GB

Model and Training Data We use InternVL2.5-8B (Chen et al., 2024c) as the default backbone, following prior MPRM work (Wang et al., 2025b; Du et al., 2025). In the main experiments, we additionally report results with a second backbone, Qwen2.5-VL-7B (Bai et al., 2025). For both models, we freeze the vision encoder and fine-tune the language model together with the multimodal projector modules. We use the default vision setup and input preprocessing of each backbone; the corresponding model and input hyperparameters are summarized in Table 7. We train on VisualPRM400K-v1.11 (Wang et al., 2025b), choosing v1.1 because it exposes per-step MC scores, whereas the v1 release only provides binarized labels. This dataset was sampled from 38 different data sources. Detailed training-data statistics are reported in Table 8.

MPRM Training Objective Each training example consists of the question, the associated images, and a step-by-step solution, where every step is followed by a special token <prm>. The tokenizer inserts this special token into the text stream and the data loader attaches the corresponding binary label to its position. During training, the model is supervised only on these <prm> positions: the logits at each placeholder are restricted to the two reward tokens (‘‘Yes’’, ‘‘No’’) and optimized with a two-way cross-entropy loss, so that the probability of ‘‘Yes’’ serves as the score for that step.

Optimization and Implementation Table 7 summarizes the hyperparameters and implementation details shared by all experiments. The learning rate is linearly warmed up over the first 5% of optimization steps and then cosine-decayed to zero over the remaining steps. Under our default single-pass protocol (i.e., one epoch over the selected training pool), the total number of optimization steps is recomputed for each data regime as T = ⌈N/B⌉, where N is the number of rollouts in the

1https://huggingface.co/datasets/OpenGVLab/VisualPRM400K-v1.1-Raw

pool and B is the global batch size. Training is implemented in PyTorch (Paszke et al., 2019) with HuggingFace Trainer from Transformers (Wolf et al., 2020) and DeepSpeed (Rasley et al., 2020) ZeRO-3 for memory efficiency.

Evaluation Protocol For each VisualProcessBench (Wang et al., 2025b) instance, we concatenate the question with the provided step-by-step rationale and insert <prm> after every step, mirroring training. The model produces a scalar score per step (the ‘‘Yes’’ probability at the corresponding placeholder). Given a threshold τ, we classify steps with scores ≥ τ as positive and those < τ as negative, ignoring neutral labels. Following the benchmark protocol, we select a single global threshold per model on a held-out split by sweeping τ and maximizing the micro-averaged F1 across all sources; we then report the overall micro-F1 in the main text and provide per-source macro-F1 breakdowns in the appendix.

Best-of-N Reranking Protocol We report best-of-N reranking results with N = 16 for all benchmarks in Table 3. For each problem, we first sample 16 candidate step-by-step rollouts using InternVL2.5-8B with standard stochastic decoding (temperature = 0.7, top-p = 0.9, top-k = 30, and max new tokens = 2048). Each candidate is formatted as a sequence of reasoning steps followed by a final answer. To rerank candidates, we apply the MPRM to obtain a scalar score at the step level. Given a candidate rollout τ with T reasoning steps and step scores {st}Tt=1, we compute a trajectory-level score by averaging over steps:

T

1 T

st.

S(τ) =

t=1

We then select the candidate with the highest S(τ) as the final prediction for evaluation.

#### D. Extended Results for Random Sub-sampling (Section 2.2)

[Figure 6]

- Figure 4. Single-pass scaling with random sub-sampling (Random-ρ) on VisualProcessBench. The top-left panel reports Overall micro-F1; the remaining panels show macro-F1 on each source dataset. The dashed horizontal line marks the Full-Data (ρ = 100%) model.

- Figure 4 extends the single-pass scaling plot in Figure 2a by breaking down the Random-ρ behavior on VisualProcessBench by source. The top-left panel reproduces the Overall micro-F1 curve, while the remaining panels show macro-F1 on each benchmark. Across sources, performance rises sharply when ρ increases from 0 to a small fraction (e.g., 5%), and then quickly saturates: further enlarging the random pool beyond the low two-digit range yields only mild additional gains. This per-source view mirrors the redundancy-dominated scaling discussed in Section 2.2.
- Figure 5 focuses on the ρ = 25% working point and compares Full-Data and Random-25% under a matched update budget. The top-left panel shows Overall micro-F1, and the remaining panels report macro-F1 on each VisualProcessBench source. Across sources, the full-data run has a systematic but moderate edge over the Random-25% subset, and the gap is smaller than one might expect after discarding 75% of the rollouts. This pattern is consistent with a regime in which additional rollouts yield diminishing returns.

[Figure 7]

###### Figure 5. VisualProcessBench performance vs. training step when training on the Full-Data and Random-25% settings of VisualPRM400Kv1.1. The top-left panel shows the Overall micro-F1 aggregated over all sources, while the remaining panels show macro-F1 on each individual VisualProcessBench source.

#### E. Extended Results for 25% Subsets (Sections 2.3 and 5.2)

[Figure 8]

- Figure 6. Evaluation performance vs. training step on VisualProcessBench for four 25% subsets of VisualPRM400K-v1.1. The top-left panel shows overall micro-F1; the remaining panels show macro-F1 on each source dataset. All 25% subset models are trained for a single pass over their respective training pools. Full-Data† shows the best checkpoint from a one-epoch Full-Data run (4× more optimization steps), shown only as a reference. BIS-25% consistently outperforms other subsets on overall and on most individual sources across the training trajectory, not only at a single checkpoint.

Table 4 in the main text reports overall micro-F1 and per-source macro-F1 on VisualProcessBench under the same 25% rollout budget, comparing BIS-25% against three baselines: Mixed-25%, Reliable-25%, and Low-MC-25%. Here we complement Table 4 with the full training curves of these 25% subsets on VisualProcessBench, and additionally include Random-25% as a standard sub-sampling baseline. Figure 6 plots overall micro-F1 and per-source macro-F1 as a function of training step. These curves provide a dynamic view of how BIS re-allocates the fixed update budget compared with Random-25%, Low-MC-25%, Mixed-25%, and Reliable-25%. Across sources, BIS-25% yields the highest or near-highest curve at almost all steps. Combined with the aggregate scores in Table 4, these extended results corroborate that BIS is strictly more effective than these 25% baselines under the same rollout and update budget.

- F. Theoretical Details F.1. A Canonical Logistic Case for the Scaling Decomposition This section makes the decomposition in Eq. (4) precise in the logistic teacher–student setting of Section 3.1, and derives the O(Neff−1/2) data term and O(T−1/2) optimization term step by step. Setup. Each training step is a pair (ϕ,Y ) with ϕ ∈ Rd and Y ∈ {0,1}. The population logistic loss is

L(w) = E(ϕ,Y ) −Y log qw(ϕ) − (1 − Y )log 1 − qw(ϕ) , qw(ϕ) = σ(⟨w,ϕ⟩),

and w∗ denotes its minimizer. Let Neff denote the number of i.i.d. training steps after thinning the pool, and

LNeff

(w) =

Neff

1 Neff

ℓ(w;ϕi,Yi), ℓ(w;ϕ,Y ) = −Y log qw(ϕ) − (1 − Y )log 1 − qw(ϕ)

i=1

be the empirical logistic loss. We write

eff ∈ arg min

LNeff

(w) for an empirical minimizer, and wT for the SGD iterate after T updates on this finite sample. Assumptions. We assume:

wN

w∈W

- (A1) (Bounded features) There exists B > 0 such that ∥ϕ∥2 ≤ B almost surely.
- (A2) (Well-specified logistic teacher) There exists w∗ ∈ Rd such that Pr(Y = 1 | ϕ) = σ(⟨w∗,ϕ⟩) almost surely.
- (A3) (Strong convexity and smoothness on a bounded domain) Assume L is µ-strongly convex and L-smooth on a closed,

convex, bounded set W ⊂ Rd containing w∗, with supw∈W ∥w∥2 ≤ R. Such a condition can be obtained, for example, by adding a small ℓ2 penalty on w.

- (A4) (SGD with decaying steps) We run projected SGD so that wt ∈ W for all t. Stochastic gradients are computed on i.i.d. samples and have bounded second moment, and the step sizes satisfy ηt = η0/√t with η0 small enough so that ηt ≤ 1/L.

Goal. We bound the excess population loss

E L(wT) − L(w∗) where the expectation is over both the draw of the training set and the randomness of SGD, and show that it decomposes into a O(Neff−1/2) data term and a O(T−1/2) optimization term.

##### Step 1: Decomposition into data and optimization terms. Insert and subtract wN

: E L(wT) − L(w∗) = E L(wT) − L( wN

eff

) − L(w∗)

) + E L( wN

eff

eff

=: Opt(T,Neff) + Data(Neff). (11)

The first term measures optimization error after T SGD updates on a fixed finite sample; the second term measures the gap between the empirical and population optima due to finite data.

##### Step 2: Bounding the finite-data term. By definition,

) − L(w∗) . Using the standard optimism of empirical risk minimization (ERM) argument,

Data(Neff) = E L( wN

eff

(w∗) − L(w∗) ≤ L( wN

(w∗) + LNeff

) − L(w∗) = L( wN

) − LNeff

) + LNeff

) − LNeff

L( wN

( wN

( wN

eff

eff

eff

eff

(w∗) − L(w∗) ≤ 2 sup

) − LNeff

) + LNeff

( wN

eff

eff

L(w) − LNeff

(w) .

w∈W

Taking expectations and applying uniform convergence then yields

(w) . (12)

Data(Neff) ≤ 2E sup w∈W

L(w) − LNeff

Under (A1)–(A3), the logistic loss ℓ(w;ϕ,Y ) is Lipschitz in w on W and the class {ℓ(w;·,·) : w ∈ W} has bounded Rademacher complexity. Standard uniform convergence bounds for Lipschitz losses in generalized linear models (ShalevShwartz & Ben-David, 2014) then imply the existence of a constant Cdata > 0 such that

Cdata √Neff

E sup

L(w) − LNeff

(w) ≤

w∈W

Combining (12) and (13) and absorbing the factor 2 into the constant gives

. (13)

Data(Neff) ≤

Cdata √Neff

. (14)

Here Cdata depends on the feature bound B and the domain radius R (and thus grows with BR).

##### Step 3: Bounding the optimization term. We now control the optimization error Opt(T,Neff) = E[L(wT) − L( wN

)]. Conditioned on the fixed sample {(ϕi,Yi)}N

eff

i=1 , let

eff

Neff

1 Neff

F(w) := LNeff

eff ∈ arg min

(w) =

ℓ(w;ϕi,Yi), wN

F(w).

w∈W

i=1

We assume F is µ-strongly convex and L-smooth on W. Moreover, we assume that the empirical minimizer wN

lies in the interior of W, so that ∇F( wN

eff

) = 0. The stochastic gradients gt used by SGD satisfy E[gt | wt] = ∇F(wt) and E[∥gt∥2 | wt] ≤ G2 for some G > 0. The SGD recursion on the empirical loss is

eff

wt+1 = ΠW(wt − ηtgt), ηt = η0t−1/2,

with η0 small enough so that ηt ≤ 1/L for all t. Define the mean squared distance to the empirical minimizer as

eff∥2 . A standard one-step expansion of ∥wt+1 − wN

Dt := E ∥wt − wN

eff∥2, combined with the µ-strong convexity and L-smoothness of F and the bounded-variance assumption on gt, implies that the sequence (Dt) satisfies a recursion of the form

Dt+1 ≤ 1 − 2µηt + 2L2ηt2 Dt + 2G2ηt2,

see Moulines & Bach (2011) for a detailed derivation. Since ηt is non-increasing and wt ∈ W for all t, and W is bounded so that Dt ≤ diam(W)2, we may absorb the 2L2ηt2Dt term and 2σ2 into a single constant G2, yielding the simplified recursion

Dt+1 ≤ (1 − 2µηt)Dt + ηt2G2. (15) Specializing to the step-size schedule ηt = η0t−1/2, (15) can be rewritten as

c √t

Dt+1 ≤ 1 −

Dt +

C t

, (16)

for some constants c,C > 0 depending only on µ,η0,G.

- Lemma 1 (One-dimensional SGD recursion). Let (Dt)t≥1 be a nonnegative sequence satisfying (16) for all t ≥ 1, with c,C > 0. Then there exists a constant C′ > 0, depending only on c,C and D1, such that

DT ≤

C′ √

T

for all T ≥ 1.

This lemma is a direct corollary of standard results for stochastic approximation recursions; see, e.g., the standard meansquare error recursion and the corresponding non-asymptotic bound in Moulines & Bach (2011) (with α = 1/2) for an explicit derivation of the O(T−1/2) rate. Applying Lemma 1 to (16) yields

C′ √

eff∥2 ≤

. (17)

DT = E ∥wT − wN

T

By L-smoothness of F (see Lemma 1.2.3 in Nesterov et al. (2018)), we have for any w

L 2 ∥w − wN

eff∥2. Since wN

F(w) ≤ F( wN

) + ∇F( wN

), w − wN

+

eff

eff

eff

is a minimizer of F, ∇F( wN

) = 0, and thus

eff

eff

L 2 ∥wT − wN

eff∥2. Taking expectations and combining with (17) gives

F(wT) − F( wN

) ≤

eff

E LNeff

(wT) − LNeff

( wN

eff

) ≤

Copt √

, (18)

T

for some constant Copt > 0 depending only on µ,L,G and the initialization. In particular, higher Monte Carlo noise typically increases the second-moment bound G2, which increases Copt and can make the optimization term dominant in a noise-limited regime.

To relate this bound on the empirical loss to the population loss, we insert and subtract LNeff

and decompose E L(wT)−L( wN

)) . (19) By the triangle inequality,

) = E LNeff

) +E (L(wT)−LNeff

(wT)−LNeff

(wT))−(L( wN

)−LNeff

( wN

( wN

eff

eff

eff

eff

(w) . (20)

E (L(wT) − LNeff

)) ≤ 2E sup

(wT)) − (L( wN

) − LNeff

L(w) − LNeff

( wN

eff

eff

w∈W

Applying the uniform-convergence bound (13) to (20) and combining with (18) and (19) yields

Opt(T,Neff) = E L(wT) − L( wN

eff

) ≤

Copt √

T

Cdata′ √Neff

, (21)

+

for some constant Cdata′ > 0. When we combine (21) with the finite-sample term in (14), the two O(Neff−1/2) contributions can be aggregated into a single constant, leading to the overall decomposition in (4).

- Step 4: Putting the pieces together. Substituting (14) and (21) into the decomposition (11) yields

E L(wT) − L(w∗) = Opt(T,Neff) + Data(Neff) ≤

Cdata √Neff

+

Copt √

.

T

These O(Neff−1/2) and O(T−1/2) rates are conservative but sufficient for the scaling decomposition used in Section 3.2.

##### F.2. Step-wise Gradient Variance

We derive the step-wise gradient variance expression used in Equation (5). Recall the teacher–student setup in Section 3.1. For a step with representation ϕ ∈ Rd and clean label Y ∈ {0,1}, the logistic-loss gradient at parameter w is

g(ϕ,Y ;w) = qw(ϕ) − Y ϕ, qw(ϕ) = σ(⟨w,ϕ⟩). At the teacher parameter w∗, write

q∗(ϕ) := qw∗(ϕ), Y | ϕ ∼ Bernoulli q∗(ϕ) .

Then

E Y | ϕ = q∗(ϕ), E Y 2 | ϕ = q∗(ϕ). The conditional mean of the gradient at w∗ is

E g(ϕ,Y ;w∗) | ϕ = q∗(ϕ) − E[Y | ϕ] ϕ = 0, so the conditional second moment equals the conditional variance:

E ∥g(ϕ,Y ;w∗)∥2 | ϕ = E (q∗(ϕ) − Y )2 | ϕ ∥ϕ∥2

= q∗2(ϕ) − 2q∗(ϕ)E[Y | ϕ] + E[Y 2 | ϕ] ∥ϕ∥2

= q∗(ϕ) − q∗2(ϕ) ∥ϕ∥2

= q∗(ϕ) 1 − q∗(ϕ) ∥ϕ∥2, which is exactly the expression in Equation (5).

##### F.3. Symmetric Label Noise

We now derive the noisy-gradient expression used in Equation (6). Fix ϕ ∈ Rd and write q∗(ϕ) = qw∗(ϕ). For brevity, set q := q∗(ϕ) ∈ (0,1).

Let the clean label Y | ϕ ∼ Bernoulli(q) be flipped independently with probability η ∈ [0,1/2) to form a noisy label Y˜. The noisy gradient at w∗ is

g˜(ϕ,Y˜;w∗) = (q − Y˜)ϕ.

Conditioned on ϕ, we can express the distribution of the noisy label Y˜ by conditioning on the clean label Y and applying the law of total probability:

Pr(Y˜ = 1 | ϕ) = Pr(Y˜ = 1,Y = 1 | ϕ) + Pr(Y˜ = 1,Y = 0 | ϕ)

= Pr(Y˜ = 1 | Y = 1,ϕ)Pr(Y = 1 | ϕ) + Pr(Y˜ = 1 | Y = 0,ϕ)Pr(Y = 0 | ϕ)

= (1 − η)q + η (1 − q) = q(1 − 2η) + η =: p1,

where q = q∗(ϕ) and Y | ϕ ∼ Bernoulli(q). Thus

Pr(Y˜ = 0 | ϕ) = 1 − p1 =: p0. Consequently,

q − 1, Y˜ = 1, q, Y˜ = 0.

q − Y˜ =

and the conditional second moment of the noisy gradient is

E ∥g˜(ϕ,Y˜;w∗)∥2 | ϕ = E (q − Y˜)2 | ϕ ∥ϕ∥2

= p1(q − 1)2 + p0q2 ∥ϕ∥2. Substituting p1 = q(1 − 2η) + η and p0 = 1 − p1 and expanding gives

p1(q − 1)2 + p0q2 = (1 − 4η)q(1 − q) + η. Therefore

E ∥g˜(ϕ,Y˜;w∗)∥2 | ϕ = (1 − 4η)q(1 − q) + η ∥ϕ∥2. Reinstating q = q∗(ϕ) yields

E ∥g˜(ϕ,Y˜;w∗)∥2 | ϕ = (1 − 4η)q∗(ϕ) 1 − q∗(ϕ) + η ∥ϕ∥2, which is the form stated in Equation (6).

##### F.4. Rollout-Level Mixture and Representation Norms

We formalize two facts used in Section 3.3: (i) rollout-level label mixture is an approximately unbiased proxy for the latent teacher mixture, and (ii) under bounded representation norms, average q(1 − q) and average q(1 − q)∥ϕ∥2 differ only by constant factors.

Label variance decomposition. Fix a rollout x with n steps. For step j let qj := qx,j∗ ∈ [0,1] and Yj | qj ∼ Bernoulli(qj), independently conditioned on {qj}. Define

n

n

n

1 n

1 n

1 n

qj(1 − qj).

pˆ :=

Yj, q¯ :=

qj, A(x) :=

j=1

j=1

j=1

- Lemma 2 (Label variance decomposition). Conditioned on {qj}, the empirical label variance satisfies

n

1 n2

E p ˆ(1 − pˆ) {qj} = q¯(1 − q¯) −

qj(1 − qj).

j=1

Proof. We have

n

n

1 n2

2 n2 1≤j<k≤n

1 n

Yj, pˆ2 =

Yj2 +

YjYk.

pˆ =

j=1

j=1

Conditioned on {qj} the Yj are independent with E[Yj | qj] = qj and E[Yj2 | qj] = qj, so

n

1 n2

2 n2 1≤j<k≤n

E[ˆp | {qj}] = q,¯ E[ˆp2 | {qj}] =

qj +

qjqk.

j=1

Using

n

n

2

1 n2

2 n2 1≤j<k≤n

1 n

qj2 +

q¯2 =

qj

=

qjqk

j=1

j=1

to eliminate the cross terms yields

1 n2

E[ˆp2 | {qj}] = q¯2 +

n

j=1

1 n2

qj −

n

1 n2

qj2 = q¯2 +

j=1

n

qj(1 − qj).

j=1

Finally,

as claimed.

1 n2

E p ˆ(1 − pˆ) {qj} = E[ˆp | {qj}] − E[ˆp2 | {qj}] = q¯(1 − q¯) −

n

qj(1 − qj),

j=1

| |
|---|

Since t  → t(1 − t) is concave on [0,1], Jensen’s inequality gives

n

1 n

qj(1 − qj) ≤ q¯(1 − q¯) =: θx.

A(x) =

j=1

Using Lemma 2 and the identity

we can rewrite

n

1 n2

qj(1 − qj) =

j=1

1 n

A(x),

E p ˆ(1 − pˆ) {qj} = θx −

1 n

A(x).

Since 0 ≤ A(x) ≤ θx, this immediately yields the sandwich bound

1 n ≤ E p ˆ(1 − pˆ) {qj} ≤ θx.

θx 1 −

Thus the conditional bias of pˆ(1 − pˆ) as an estimator of the teacher-level mixture θx is at most θx/n ≤ 1/(4n), and pˆ(1 − pˆ) is an approximately unbiased proxy for θx. In particular, rollouts with larger pˆ(1 − pˆ) tend to have larger teacher-level mixture θx (in expectation, up to an O(1/n) bias). Since A(x) ≤ θx, a larger θx simply provides more headroom for A(x) to be large, and therefore for the rollout to contain more informative steps.

Symmetric flip noise and induced mixture. In Section 3.3 we also consider a symmetric flip noise approximation: the observed label Y˜j is obtained by independently flipping the clean label Yj with probability η ∈ [0,1/2). Let Bj ∼ Bernoulli(η) be independent of Yj and define Y˜j := Yj ⊕ Bj. Conditioned on qj, we have

q˜j := P(Y˜j = 1 | qj) = P(Yj = 1,Bj = 0 | qj) + P(Yj = 0,Bj = 1 | qj)

= (1 − η)qj + η(1 − qj) = (1 − 2η)qj + η. (22) Thus Y˜j | qj ∼ Bernoulli(˜qj). Averaging (22) over steps gives

n

1 n

q¯˜ =

q˜j = (1 − 2η)¯q + η.

j=1

Defining θ˜x := q¯˜(1 − q¯˜) and θx := q¯(1 − q¯), a direct expansion yields

θ˜x = (1 − 2η)¯q + η 1 − (1 − 2η)¯q − η

= (1 − 2η)¯q + η (1 − η) − (1 − 2η)¯q

= (1 − 2η)2 q¯(1 − q¯) + η(1 − η)

= (1 − 2η)2 θx + η(1 − η).

##### Closeness to the noise-free analysis for small η. Eq. (23) implies

(23)

θ˜x − θx = (1 − 2η)2 − 1 θx + η(1 − η), so using 0 ≤ θx ≤ 1/4 we obtain the uniform bound

|θ˜x − θx| ≤ 4η θx + η(1 − η) ≤ 2η. (24)

Moreover, since Y˜x,j | q˜x,j ∼ Bernoulli(˜qx,j) independently conditioned on {q˜x,j}, applying Lemma 2 with qx,j replaced by q˜x,j shows that pˆ∼,x(1 − pˆ∼,x) estimates θ˜x up to an additional O(1/n) bias. Therefore, when η is small (and n is not too small), the mixture computed from observed labels is within O(η) + O(1/n) of the noise-free target in expectation, so the analysis in the noise-free setting applies up to a small additive perturbation.

Effect of bounded representation norms. Define the full average step-wise information

1 n

Afull(x) :=

n

qj(1 − qj)∥ϕx,j∥2.

j=1

Assume the representations are uniformly bounded: there exist constants 0 < cmin ≤ cmax < ∞ such that cmin ≤ ∥ϕx,j∥2 ≤ cmax for all steps. Then

cmin A(x) ≤ Afull(x) ≤ cmax A(x). Hence, up to global multiplicative constants, A(x) and Afull(x) measure the same notion of step-wise information. Qualitative comparisons between rollouts can therefore be phrased in terms of A(x). In particular, increasing A(x) increases a corresponding lower bound on Afull(x), and A(x) serves as a constant-factor proxy for Afull(x) under this boundedness assumption.

##### F.5. MC-induced Pseudo-positive Probability and Monotonicity

Posterior and closed form. Let r ∈ [0,1] denote the one-shot success probability of a step, and let K | r ∼ Binomial(N,r) be the number of successful continuations. Fix τ ∈ (0,1) and define τ-reliability by Z := I[r ≥ τ]. We place a Beta prior r ∼ Beta(a,b) with density p(r) ∝ ra−1(1 − r)b−1 for a,b > 0. The Binomial likelihood is

N k

rk(1 − r)N−k.

p(K = k | r) =

By Bayes’ rule,

p(r | K = k) ∝ p(K = k | r)p(r) ∝ ra+k−1(1 − r)b+N−k−1,

which is the density of Beta(a + k, b + N − k). Writing αk := a + k and βk := b + N − k, the normalized posterior density is

1 B(αk,βk)

k−1(1 − r)β

k−1, r ∈ (0,1),

rα

fk(r) := p(r | K = k) =

where B(α,β) = 0 1 tα−1(1 − t)β−1 dt is the Beta function. Define the conditional pseudo-positive probability (effective noise level)

τ

ηeff(k) := Pr(Z = 0 | K = k) = Pr(r < τ | K = k) =

fk(r)dr.

0

Let B(τ;α,β) := 0 τ tα−1(1 − t)β−1 dt be the incomplete Beta function. Then

B(τ;αk,βk) B(αk,βk)

=: Iτ(αk,βk),

ηeff(k) =

where Iτ(α,β) is the regularized incomplete beta function. Monotonicity.

- Lemma 3 (Monotonicity of ηeff). For any a,b > 0, N ≥ 1, and τ ∈ (0,1), the map k  → ηeff(k) = Pr(r < τ | K = k) is strictly decreasing on {0,1,...,N}. Equivalently, Pr(r ≥ τ | K = k) = 1 − ηeff(k) is strictly increasing in k.

Proof. Let fk be the posterior density above with parameters αk = a + k and βk = b + N − k, i.e.,

1 B(αk,βk)

k−1(1 − r)β

k−1, r ∈ (0,1).

rα

fk(r) =

Fix k ∈ {0,...,N − 1} and compare consecutive posteriors. A direct calculation gives, for r ∈ (0,1),

fk+1(r) fk(r)

B(αk,βk) B(αk+1,βk+1) · rα

k+1−αk(1 − r)β

k+1−βk

=

r 1 − r

B(αk,βk) B(αk + 1,βk − 1) ·

.

=

Since k ≤ N − 1 and b > 0, we have βk = b + N − k ≥ b + 1 > 1, so βk − 1 > 0 and the Beta-function identity B(α + 1,β − 1) = β−α1B(α,β) applies. Thus we obtain the explicit form

βk − 1 αk

b + N − k − 1 a + k

fk+1(r) fk(r)

r 1 − r

= Ck

, Ck :=

=

,

which is strictly increasing in r on (0,1) since r/(1 − r) is strictly increasing. Fix τ ∈ (0,1) and define c := f

k+1(τ) fk(τ) = Ck 1−ττ . Because f

k+1(r) fk(r) is strictly increasing in r, we have

fk+1(r) fk(r)

< c for r ∈ (0,τ),

fk+1(r) fk(r)

> c for r ∈ (τ,1),

and f

k+1(τ) fk(τ) = c at r = τ. Multiplying by fk(r) and integrating yields

τ

fk+1(r)dr ≤ c

0

τ

fk(r)dr,

0

1

fk+1(r)dr ≥ c

τ

1

fk(r)dr.

τ

Let Ak := 0 τ fk(r)dr = Pr(r < τ | K = k). Since 0 1 fk = 1, the second inequality becomes 1 − Ak+1 ≥ c(1 − Ak). If c ≤ 1, then the first inequality gives Ak+1 ≤ cAk ≤ Ak. If c ≥ 1, then the second inequality implies 1−Ak+1 ≥ 1−Ak, i.e. Ak+1 ≤ Ak. Thus in all cases Ak+1 ≤ Ak, proving monotonic non-increase. Finally, since the ratio f

k+1(r) fk(r) is strictly increasing in r, the inequalities f

###### k+1(r) fk(r) < c on (0,τ) and f

k+1(r)

fk(r) > c on (τ,1) are strict on sets of positive Lebesgue measure. Moreover, fk(r) > 0 for all r ∈ (0,1) when αk,βk > 0. Hence the integral inequalities above are strict, yielding Ak+1 < Ak for any τ ∈ (0,1). Therefore ηeff(k) = Ak is strictly decreasing in k on {0,1,...,N}.

| |
|---|

#### G. Training Dynamics

Figure 7,8 and 9 report the training dynamics of BIS-ρ and Random-ρ under keep ratios ρ ∈ {5,10,15,25,35,50}%. For each ρ, we track both the overall micro-F1 and the per-source macro-F1 on VisualProcessBench over training steps, and include the full-data reference as horizontal dashed lines. Across ratios and backbones, BIS not only achieves stronger final performance, but also improves faster: it reaches high accuracy in substantially fewer steps and maintains clear advantages over random sub-sampling throughout training. This gap is most pronounced in the low-budget regime, where Random-ρ often learns slowly and remains far below the full-data reference, while BIS-ρ rapidly closes the gap and frequently approaches full-data performance early in training.

[Figure 9]

[Figure 10]

- Figure 7. Training dynamics on VisualProcessBench for ρ ∈ {5%, 10%}, comparing BIS-ρ and Random-ρ for both InternVL2.5-8B and Qwen2.5-VL-7B (overall micro-F1 and per-source macro-F1).

[Figure 11]

[Figure 12]

###### Figure 8. Training dynamics on VisualProcessBench for ρ ∈ {15%, 25%}, comparing BIS-ρ and Random-ρ for both InternVL2.5-8B and Qwen2.5-VL-7B (overall micro-F1 and per-source macro-F1).

[Figure 13]

[Figure 14]

###### Figure 9. Training dynamics on VisualProcessBench for ρ ∈ {35%, 50%}, comparing BIS-ρ and Random-ρ for both InternVL2.5-8B and Qwen2.5-VL-7B (overall micro-F1 and per-source macro-F1).

#### H. Per-source BIS Histograms

Interpreting low-BIS sources via step-level MC difficulty. Figure 10 reports, for each of the 38 source subsets, the BIS score distribution over all rollouts and the per-source top-25% rollouts selected by BIS. The per-source shapes vary substantially, and interpreting them requires care: because BIS(x) = (ppos(x)(1−ppos(x))+α)R(x),rollouts that are nearly pure (almost all-positive or almost all-negative) receive small mixture weight ppos(1 − ppos) and thus tend to concentrate at the lower end of the BIS range, even though these two purity regimes correspond to opposite difficulty profiles.

To disambiguate them, we pair the histograms with the step-level average MC score s¯ for each source (Table 8), where s¯ averages MC scores over all annotated steps and serves as a proxy for how often the MC solver can reach a correct completion from intermediate states. Under this lens, sources whose full-pool histogram is concentrated near low BIS scores split into two qualitatively different categories:

High-s¯ sources (easy for the current MC solver). When s¯ is high, low BIS scores mainly reflect trajectories that are dominated by positive steps (ppos ≈ 1), hence little label mixture. This is exemplified by ScienceQA (s¯ = 0.9723) and NLVR2 (s¯ = 0.9672), as well as other high-s¯ sources such as DVQA (0.9453). In these sources, naturally mixed trajectories are rarer, so the available high-BIS tail is intrinsically thinner.

Low-s¯ sources (hard for the current MC solver). When s¯ is low, low BIS scores instead reflect trajectories with few positive steps under MC sampling (ppos ≈ 0). Representative examples include GeoQA+ (Open) (s¯ = 0.5720) and GeomVerse (0.6819). Here the bottleneck is the scarcity of reliable positive anchors: even when errors are plentiful, many steps rarely lead to successful continuations, so the pool contains fewer rollouts that simultaneously provide informative negatives and trustworthy positives.

Between these extremes, medium-s¯ sources tend to provide the richest substrate for BIS, because they naturally generate rollouts with non-trivial label mixture while still retaining a meaningful fraction of reasonably reliable positives. For example, FigureQA (s¯ = 0.7615) and VQAv2 (s¯ = 0.8253) exhibit visibly heavier mass at moderate-to-high BIS scores, indicating a larger supply of mixed-but-reliable trajectories that BIS can exploit.

Across all sources, BIS selection produces a consistent within-source reallocation: the orange histograms shift weight toward higher scores relative to the blue ones, meaning that under the same per-source budget (top-25% per file) BIS preferentially keeps the most mixed and reliable rollouts available in that source. This provides a distributional explanation for the main experimental trend in Table 2: compared with Random-25%, which spends budget proportionally to the original pool, and Low-MC-25%, which can over-focus on unreliable pseudo-positives, BIS-25% systematically extracts the highest-information tail when such a tail exists (e.g., medium-s¯ sources), while avoiding wasting budget on uninformative near-pure trajectories in very easy sources or on noise-dominated trajectories in very hard sources. As a result, BIS allocates the fixed update budget to a subset with higher effective signal per update, aligning with its consistent gains over Random / Mixed / Low-MC under the same rollout and optimization budgets.

Consistent selection patterns in mixture and reliability. Figure 11 and 12 further reports, for each source, the distributions of the reliability term R(x) and the mixture term ppos(x)(1 − ppos(x)) for all rollouts and the selected top-25% subset, together with the coverage curve (Selected / All). Across sources, BIS follows two stable behaviors: it strongly suppresses low-R(x) rollouts, and it consistently favors higher mixture by shifting coverage toward moderate-to-high ppos(1 − ppos). Importantly, the preference over R(x) is not monotonic: coverage often peaks at moderate R(x) and then saturates or drops as R(x)→1, since near-pure trajectories can have large R(x) but still receive small mixture weight and thus low BIS scores. The plots also highlight substantial source-level heterogeneity in the available mixed-and-reliable tail, complementing the step-level MC difficulty analysis in Table 8.

Table 8. Per-source step-level MC score statistics on VisualPRM400K-v1.1. Avg. MC is the mean MC score over all annotated steps; Avg./16 equals 16×Avg. MC (expected successes out of N =16 continuations). Step/Rollout counts follow the same validity filtering as in our histogram analysis.

Subset Avg. MC Avg./16 # Steps # Rollouts AI2D (Kembhavi et al., 2016) 0.9409 15.0539 146,419 28,147 ChartQA (Masry et al., 2022) 0.9078 14.5245 136,655 28,049 CLEVR (Johnson et al., 2017) 0.9106 14.5689 109,317 24,004 COCO-ReM (Y/N) (Singh et al., 2024) 0.8933 14.2922 55,628 18,450 DocVQA (Mathew et al., 2021) 0.8663 13.8604 85,589 21,049 DVQA (Kafle et al., 2018) 0.9453 15.1255 177,352 35,367 FigureQA (Kahou et al., 2017) 0.7615 12.1843 57,601 12,345 Geo170K (Gao et al., 2025) 0.9023 14.4373 61,314 8,205 Geometry3K (Lu et al., 2021a) 0.9243 14.7880 94,624 11,756 Geometry3K (Open) (Lu et al., 2021a) 0.8707 13.9318 58,569 7,090 GeomVerse (Kazemi et al., 2024) 0.6819 10.9106 38,156 7,765 GeoQA+ (Open) (Cao & Xiao, 2022) 0.5720 9.1514 45,287 7,025 GeoQA+ (MC) (Cao & Xiao, 2022) 0.8486 13.5772 67,715 9,216 GEOS (Seo et al., 2015) 0.9061 14.4970 16,847 1,827 GEOS (Open) (Seo et al., 2015) 0.8217 13.1478 1,563 178 IconQA (Lu et al., 2021b) 0.8908 14.2536 121,873 25,811 InfographicVQA (Mathew et al., 2022) 0.7645 12.2323 78,408 17,996 KonIQ-10k (Hosu et al., 2020) 0.8822 14.1148 72,430 13,337 M3CoT (Chen et al., 2024b) 0.8811 14.0971 115,649 19,476 MAPQA-SUV (Chang et al., 2022) 0.7260 11.6155 48,368 12,366 MathV360K (Shi et al., 2024) 0.8759 14.0142 107,503 20,132 MAVIS-Abs (Zhang et al., 2025c) 0.8104 12.9670 166,756 16,530 MAVIS-Cos (Zhang et al., 2025c) 0.8057 12.8916 86,393 9,174 MAVIS-Log (Zhang et al., 2025c) 0.7357 11.7719 162,713 15,422 MAVIS-Poly (Zhang et al., 2025c) 0.8836 14.1375 206,293 23,461 MAVIS-Sin (Zhang et al., 2025c) 0.8231 13.1701 88,130 9,675 MAVIS-Tan (Zhang et al., 2025c) 0.8027 12.8428 60,391 6,180

- MAVIS-Geo-0 (Zhang et al., 2025c) 0.9280 14.8473 78,764 13,149

- MAVIS-Geo-1 (Zhang et al., 2025c) 0.8765 14.0243 37,760 6,728

- MAVIS-Geo-2 (Zhang et al., 2025c) 0.7990 12.7848 25,851 4,801

- MAVIS-Geo-3 (Zhang et al., 2025c) 0.7719 12.3508 19,080 3,706 NLVR2 (Suhr et al., 2019) 0.9672 15.4753 96,072 26,439 ScienceQA (Lu et al., 2022) 0.9723 15.5574 186,894 34,523 SROIE (Huang et al., 2019) 0.8708 13.9325 83,389 18,074 Super-CLEVR (Int) (Li et al., 2023) 0.7325 11.7205 37,279 12,070 Super-CLEVR (Y/N) (Li et al., 2023) 0.7713 12.3404 33,171 10,440 UniGeo (Chen et al., 2022) 0.6015 9.6239 41,523 6,299 VQAv2 (Goyal et al., 2017) 0.8253 13.2043 67,166 18,887

[Figure 15]

- Figure 10. Per-source BIS score distributions on VisualPRM400K-v1.1. Each panel corresponds to one source dataset; light purple bars show all rollouts from that source and dark purple bars show the per-file top-25% rollouts selected by BIS. Across sources, the full-pool distributions differ markedly—with some datasets exhibiting sharp spikes near the baseline α = 0.05 and others showing heavier high-score tails—but BIS consistently suppresses low-score mass and enriches higher-score regions within each source.

[Figure 16]

[Figure 17]

[Figure 18]

##### Question.

The graph of the function f(x) = a ∗ x + b is shown, where the condition that a is non-zero is satisfied. The function f(x) takes the values −10 and −6 at x = −5 and x = −3, respectively. Given the graph and the aforementioned conditions, identify the zeros of the function.

##### Ground truth. 0

###### Step scores (hard label yj = I[scorej > 0]).

Step Content Score yj

- Step-0 To determine the zeros of the function f(x) = ax + b, we need to follow these steps: 0.875 1
- Step-1 1. Identify the equation of the line:

- - Given that f(x) = −10 at x = −5 and f(x) = −6 at x = −3, we can use these points to find the slope a and the y-intercept b.
- - Substitute x = −5, y = −10 into the equation: −10 = a(−5) + b (1)
- - Substitute x = −3, y = −6 into the equation: −6 = a(−3) + b (2)

0.875 1

- Step-2 2. Solve the system of equations:

- - From equation (1): −10 = −5a + b
- - From equation (2): −6 = −3a + b

0.9375 1

- Step-3 3. Simplify the equations:

- - Equation (1) becomes: b = −10 + 5a
- - Equation (2) becomes: −6 = −3a + b

0.875 1

- Step-4 4. Substitute b values from equation (1) into equation (2):

- - Substitute b = −10 + 5a into −6 = −3a + b: −6 = −3a + (−10 + 5a)
- - Simplify: −6 = −3a − 10 + 5a
- - Combine like terms: −6 = 2a − 10
- - Solve for a: 6 = 2a =⇒ a = 3

0.0 0

- Step-5 5. Find b using the value of a:

- Substitute a = 3 back into b = −10 + 5a: b = −10 + 5(3) = −10 + 15 = 5

0.0 0

- Step-6 6. Write the final equation of the function:

- The equation of the function is f(x) = 3x + 5.

0.0 0

- Step-7 7. Setting f(x) = 0 to find the zeros: 5 3

- - 0 = 3x + 5
- - Solve for x: 3x = −5 =⇒ x = −

0.0 0

- Step-8 Thus, the zeros of the function f(x) = 3x + 5 are x = −53. 0.0 0

- Step-9 Final answer: −35 0.0 0

##### BIS computation (α = 0.05).

Valid step scores are the 10 entries above. Hard labels: yj = 1 iff scorej > 0.

4 10

n = 10, npos = 4, ppos =

= 0.4,

0.875 + 0.875 + 0.9375 + 0.875 4

= 0.890625, BIS(x) = ppos(1 − ppos) + 0.05 · R

R =

= (0.4 · 0.6 + 0.05) · 0.890625 = 0.25828125.

Figure 13. Case study (MAVIS-Function (Zhang et al., 2025c)): step-level MC scores, hard labels, and BIS computation for one rollout.

[Figure 19]

##### Question.

Which part of the diagram best depicts mineralisation?

- – D
- – F
- – M
- – A Please answer the question based on the options mentioned before. Ground truth. M

###### Step scores (hard label yj = I[scorej > 0]).

Step Content Score yj

- Step-0 To determine which part of the diagram depicts mineralization, let’s break down the relevant concepts.

0.5625 1

- Step-1 1. Mineralization: It refers to the process by which organic matter is converted into minerals. This often involves bacteria breaking down organic materials and releasing inorganic salts such as nitrates and phosphates back into the soil, accessible for plant uptake.

0.6875 1

- Step-2 2. Flowchart Analysis:

- - Identify the main components in the diagram that are involved with mineralization.
- - Typically, mineralization follows the decomposition pathway, so look for arrows leading to outputs that could be soil minerals.

0.6875 1

- Step-3 3. Options:

- - D: This point may involve decomposers like bacteria and fungi that convert organic matter to minerals.
- - F: This appears to be involved in detritus or initial organic matter feed.
- - M: This might be part of the nutrient cycle but may not directly depict mineralization.
- - A: This point seems more related to primary producers or initial growth stage.

0.0625 1

- Step-4 4. Evaluation of Options:

- - D shows arrows indicating a transformation from organic to inorganic content, implying the breakdown and conversion into minerals.
- - F, M, A do not visibly indicate this mineralization process.

0.0 0

- Step-5 ### Answer: 0.0 0
- Step-6 Based on these analyses, D is the most consistent signifier of the mineralization process in this diagram. Therefore, the answer is:

0.0 0

- Step-7 Final answer: D 0.0 0

##### BIS computation (α = 0.05).

Valid step scores are the 8 entries above. Hard labels: yj = 1 iff scorej > 0.

4 8

n = 8, npos = 4, ppos =

= 0.5,

0.5625 + 0.6875 + 0.6875 + 0.0625 4

= 0.5, BIS(x) = ppos(1 − ppos) + 0.05 · R

R =

= (0.5 · 0.5 + 0.05) · 0.5 = 0.15.

Figure 14. Case study (AI2D (Kembhavi et al., 2016)): step-level MC scores, hard labels, and BIS computation for one rollout.

##### Question.

[Figure 20]

Angle ABC is equivalent to π/2. Side DE extending into an equilateral triangle shape beyond the rectangle. EFG forms a sector. Side HI continues into an equilateral triangle inside the rectangle. AB is equal to 72, Angle ACB measures 60◦, GI measures 71. What is the total surface area of the shape GFHJI?

- A. 954*√3

- B. 3816*√3/5

- C. 70
- D. 1272*√3 Ground truth. D

###### Step scores (hard label yj = I[scorej > 0]).

Step Content Score yj

- Step-1 To find the total surface area of the complex shape GFHJI, let’s break it down step by step.

0.0625 1

- Step-2 1. Identify the components of the shape GFHJI: - Rectangle GFHI - Equilateral triangle HIJ - Segments GH and HI, which are parts of the equilateral triangle - The sector EFG

0.0625 1

- Step-3 2. Calculate the area of rectangle GFHI: - Side GF = 71 (given) - Determine the width (HI) of the rectangle: - HI is part of the equilateral triangle HIJ. - The side of the equilateral triangle is GH, which is the same as CI (71 units). - The height of the

equilateral triangle HIJ can be calculated as

√3

2 × 71 = 60.98.

0.0625 1

- Step-4 3. Calculate the area of the equilateral triangle HIJ: - Area =

√3

4 × 712 = 865.48. 0.0625 1

- Step-5 4. Calculate the area of the sector EFG: - The radius EF = FG = 71 units. - Angle GFE is not given, so assume it’s part of the sector attached to the rectangle. - Calculate the area of the sector 360θ π × r2.

0.0 0

- Step-6 5. Combine the areas: - Rectangle GFHI: 71 × 60.98 ≈ 4326.68 - Equilateral triangle HIJ: 865.48 - Sector EFG: Assume π6 × 712 = 2716.23 (half of the sector for simplicity)

0.0625 1

- Step-7 6. Summarize all areas and subtract the overlapping areas: - Total area = Area of rectangle + Area of equilateral triangle + Area of sector.

0.0625 1

- Step-8 Given calculations involve approximations based on assumed values. 0.0 0
- Step-9 Answer:

|B|
|---|

(Based on approximations and calculated components.) 0.0 0

##### BIS computation (α = 0.05).

Valid step scores are the 9 entries above. Hard labels: yj = 1 iff scorej > 0.

- 2

- 3

6 9

=

,

n = 9, npos = 6, ppos =

0.0625 + 0.0625 + 0.0625 + 0.0625 + 0.0625 + 0.0625 6

R =

= 0.0625, BIS(x) = ppos(1 − ppos) + 0.05 · R

- 2

- 3 ·

1 3

+ 0.05 · 0.0625 = (0.222222 . . . + 0.05) · 0.0625 ≈ 0.0170138889.

=

Figure 15. Case study (MAVIS-Function (Zhang et al., 2025c)): step-level MC scores, hard labels, and BIS computation.

#### I. Case Studies

Figures 13, 14 and 15 illustrate why BIS is a reasonable rollout selection rule when step labels are obtained by thresholding MC scores.

Figure 13 is a representative high-quality rollout. It contains clear step-level variation and a meaningful mix of positive and negative signals, while the positive steps receive consistently high MC scores. As a result, BIS assigns a relatively large score, matching our motivation that the most informative rollouts should be both mixed and reliable.

Figure14 is a more borderline case. Although the rollout still contains mixed signals, some “positive” steps have low scores and the reasoning later collapses to an incorrect final choice, leading to only moderate R(x). BIS therefore assigns a moderate score, reflecting that mixture alone is not sufficient if the positive supervision is not reliable.

Finally, Figure 15 shows a more diagnostic failure mode. Several steps contain clear, checkable geometry/number mistakes (e.g., mis-identifying given lengths or applying invalid equalities), yet their MC scores remain non-zero but very small (often at the 1/16 level). Under hard labeling, these low-but-nonzero scores are still binarized to yj=1, creating unreliable “pseudo-positive” supervision. This illustrates that non-zero low MC scores do not necessarily imply correctness, and can introduce noisy supervision during training. BIS mitigates this issue via the reliability term, which down-weights such low-confidence rollouts and prevents them from dominating the selected subset.

