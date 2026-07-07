## arXiv:2604.03993v1[cs.LG]5Apr2026

# Can LLMs Learn to Reason Robustly under Noisy Supervision?

Shenzhi Yang1,2, Guangcheng Zhu1,2, Bowen Song2, Sharon Li3, Haobo Wang1‡, Xing Zheng2, Yingfan Ma2, Zhongqi Chen2, Weiqiang Wang2, Gang Chen1

1Zhejiang University 2Ant Group 3University of Wisconsin-Madison

‡Corresponding authors.

Reinforcement Learning with Verifiable Rewards (RLVR) effectively trains reasoning models that rely on abundant perfect labels, but its vulnerability to unavoidable noisy labels from expert scarcity remains critically underexplored. In this work, we take the first step toward a systematic analysis of noisy label mechanisms in RLVR. In contrast to supervised classification, most RLVR algorithms incorporate a rollout-based condition: a label’s influence on training is contingent on whether the current policy can generate rollouts that realize it, a property that naturally extends to noisy labels. Based on this observation, we distinguish two types of noise: inactive noisy labels, which reduce data efficiency, and active noisy labels, which are reinforced and risk skewing the model toward incorrect distributions. From experiments on training with noisy samples, we identify a Early Correctness Coherence phenomenon: although noisy samples begin to lag behind in later stages, accuracy on both clean and noisy samples increases similarly in early training. Motivated by this dynamic, we propose Online Label Refinement (OLR), which progressively corrects potentially noisy labels with majority-voted answers when two conditions hold: a positive slope in the majority answer’s rollout pass rate and stable historical consistency across updates, enabling gradual self-correction as the policy improves. We evaluate OLR on six in-distribution mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three out-of-distribution tasks (ARC-c, GPQA-diamond, and MMLU-pro). Across noise ratios from 0.1 to 0.9, OLR consistently improves robustness under both inactive and active noisy-label settings, achieving average gains of 3.6%–3.9% on in-distribution benchmarks and 3.3%–4.6% on out-of-distribution evaluations. The code is available via https://github.com/ShenzhiYang2000/OLR.

[Figure 1]

[Figure 2]

#### 1 Introduction

Following the success of DeepSeek-R1 (Guo et al., 2025), reinforcement learning with verifiable rewards (RLVR) has rapidly emerged as a powerful paradigm for training reasoning-oriented language models. While algorithms such as GRPO (Shao et al., 2024) and its variants (Liu et al., 2025; Zheng et al., 2025; Yu et al., 2025) eliminate the need for critic models via intra-group advantage estimation, this framework relies on large amounts of carefully curated data. In practice, however, expert scarcity and weak verifiers (Yan et al., 2025b) make imperfect labeling unavoidable, resulting in noisy training labels. Although noisy label learning (Natarajan et al., 2013; Sukhbaatar et al., 2014) has been extensively studied in traditional supervised classification (Hendrycks et al., 2018; Collier et al., 2021; Li and Zhu, 2024), these studies predominantly focus on static datasets and limited output label spaces. However, this perspective overlooks a critical characteristic of most RLVR algorithms arising from its near-infinite label space and on-policy data

generation: a label’s influence on training is contingent on whether the current policy can generate rollouts that realize it, a property that naturally extends to noisy labels. Unlike traditional classification, where all wrong labels contribute to loss, in RLVR, a noisy label is reinforced only if the model can generate it. If the model cannot generate the rollouts to realize the noisy label (inactive noisy labels), all generations have equal advantage. These do not actively mislead but waste rollouts and reduce data efficiency. More harmful are noisy labels the model can generate (active noisy labels), which receive positive advantage and steer the policy toward incorrect distributions (Fig.1). To date, robust training of large language models (LLMs) for reasoning under such noisy supervision remains largely unexplored.

In this work, we take the first step toward a systematic analysis of noisylabel learning in RLVR. We observe a key training dynamic, which we term the Early Correctness Coherence: although noisy samples begin to lag behind in later stages, accuracy on both clean and noisy samples increases similarly in early training. Interestingly, even with noisy supervision, the chance that the majority answer (Zuo et al., 2025) matches the correct answer in-

###### 𝑞: Solve for𝑥 = 4

|Inactive Noisy Label|
|---|
|Noisy Example: 𝑦noise: 𝑥 = sin 𝑥 Policy Probability: 𝝅𝜽 𝒚noise 𝒒 = 𝟎 Reward and Advantage: R = 0, A = 0|

|Active Noisy Label|
|---|
|Noisy Example: 𝑦noise: 𝑥 = 2 Policy Probability: 𝝅𝜽 𝒚noise 𝒒 > 𝟎 Reward and Advantage: R = 1, A > 0|

Figure 1 Two types of noisy labels in RLVR. Inactive noisy label (left): an incorrect initial label that the model cannot trigger the corresponding reasoning path for and reinforce, so it remains inactive. Active noisy label (right): an incorrect label that the model has a probability of triggering the corresponding reasoning path for and reinforcing, so it remains active.

creases. This indicates that the training process already produces real correct answers for noisy samples (Fig. 2). Motivated by this observation, we propose Online Label Refinement (OLR). Instead of relying on externally provided labels, OLR selectively corrects potentially noisy labels, whether inactive or active, with the model’s own policy-generated majority answers, which are more likely to reflect these emerging correct answers. Intuitively, this resembles a student learning: early coherence learning builds reasoning ability, and answers that repeatedly emerge from the student’s own practice with increasing confidence are likely to be correct. In our method, we formalize this intuition by monitoring two signals:

- 1. The slope of the majority answer’s pass rate across rollouts. A positive slope indicates that as the model repeatedly attempts the problem, more rollouts converge to the same answer, suggesting it provides a positive expected advantage and meaningful learning signal as the policy improves.
- 2. Historical consistency of the majority answer. By tracking whether the same majority answer remains dominant over consecutive updates, we filter out transient or accidental majority answers, preventing them from misleading the model.

In summary, once a majority answer exhibits both a positive slope and historical consistency, we replace the original label with it. This replacement is guided by two complementary signals: the model’s improving capability (evidenced by convergent rollouts) and reinforcement dynamics (the answer consistently yields higher expected advantage). As reasoning improves and correct majorities emerge, noisy labels are gradually overwritten, enabling the policy to continuously refine its own training targets. We evaluate OLR on six mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three outof-distribution tasks (ARC-c, GPQA-diamond, and MMLU-Pro). Across noise ratios from 0.1 to 0.9, OLR achieves consistent gains. Under inactive noise, it yields average improvements of 3.6% on in-distribution benchmarks and 3.3% on out-of-distribution tasks. Under active noise, the gains increase to 3.9% and 4.6%, respectively. These results demonstrate that OLR robustly handles both noise types across diverse task distributions.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(a) Inactive Noise w/o OLR (b) Inactive Noise w/ OLR (c) Active Noise w/o OLR (d) Active Noise w/ OLR

Figure 2 Early Correctness Coherence. Training accuracy of Qwen3-4B-Base (Yang et al., 2025) under noisy supervision (noise ratio 0.5). For each sample, we take the majority answer as the model’s prediction. Clean and noisy samples exhibit similar learning dynamics early in training but gradually diverge as training progresses. (i) In the initial phase, the accuracy of correct answers from both groups increases steadily, suggesting that the model already contains latent correct answers for noisy samples that are not fully exploited. (ii) In later stages, accuracy on clean samples continues to improve while performance on noisy samples lags behind. Our method, Online Label Refinement (OLR), utilizes this early coherence and significantly improves reasoning performance.

#### 2 Related Work

Reinforcement Learning with Verifiable Rewards (RLVR) has proven highly effective for training reasoning models in domains with clear ground truth, such as math and code (Hu et al., 2025; Team et al., 2025; Mroueh, 2025; Li et al., 2025b). By replacing learned reward models with rule-based verifiers (Jaech

- et al., 2024; Guo et al., 2025), RLVR avoids the complexities of human preference modeling (Christiano

- et al., 2017; Ouyang et al., 2022), enabling stable training pipelines that have produced powerful models like DeepSeek-R1 (Guo et al., 2025) via algorithms such as GRPO (Shao et al., 2024; Liu et al., 2025; Yu

et al., 2025; Zheng et al., 2025). Parallel work has investigated alternative unsupervised methods, including self-judgment (Wu et al., 2024; Yuan et al., 2024; Xiong et al., 2025), ensemble heads (Wang et al., 2024b; Zhou et al., 2025), and heuristics like entropy and majority voting (Agarwal et al., 2025; Li et al., 2025a; Zuo et al., 2025), to enable scalable online learning (Zhang et al., 2025a; Zhao et al., 2025). However, despite the growing body of RLVR research, the critical issue of learning under noisy labels remains largely unaddressed. This paper presents a first exploration of noisy label mitigation in RLVR.

Noisy Label Learning is well-established, but its techniques rarely accommodate RLVR’s generation and on-policy dynamics. Existing methods fall into three categories. 1) Noise Transition Matrix Estimation (Xiao et al., 2015; Chen and Gupta, 2015; Srivastava et al., 2014; Sukhbaatar et al., 2014; Collier et al., 2021; Bucarelli et al., 2023) learns flipping probabilities between labels, but is unsuitable for RLVR, where noisy labels form an open set without clear class boundaries. 2) Loss Correction (Goodfellow et al., 2014; Pereyra et al., 2017; Zhang et al., 2017; Menon et al., 2020; Cheng et al., 2021) uses regularization to reduce overfitting to noise, but these classification-focused methods have limited applicability to RLVR’s generative tasks. 3) Small-loss-based Sample Selection (Malach and Shalev-Shwartz, 2017; Wang et al., 2018; Han

- et al., 2018b; Jiang et al., 2018; Wu et al., 2021; Song et al., 2021) treats low-loss samples as clean. However, in RLVR, small-loss samples are not necessarily beneficial (Zhan et al., 2025).

#### 3 Method

- Section 3.1 formulates the standard RLVR problem and provides the first definition of noisy labels in RLVR.
- Section 3.2 presents a theoretical analysis of the Early Correctness Coherence phenomenon observed in practice. Building on this analysis, Section 3.3 introduces our proposed method, Online Label Refinement (OLR), the first noisy label mitigation approach for RLVR.

###### 3.1 Problem Formulation

We consider reinforcement learning with verifiable rewards (RLVR), where a policy πθ(y | x) is trained to generate a solution y for a given input prompt x. Unlike supervised classification, training data are generated

on-policy: for each prompt x, the model samples K rollouts Y(x) = {y(1), . . . , y(K)} ∼ πθ(· | x). A verifier provides a scalar reward signal r(x, y) ∈ R, which is typically binary, indicating whether the generated solution satisfies a certain criterion (e.g., correctness or format adherence). Among the relevant methods, the most popular is Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which computes the advantage for each rollout by normalizing rewards within the rollout group:

###### r(x, y(k)) − µ(x) σ(x) + ϵ

A(x, y(k)) =

, (1)

where µ(x) = K1 ∑kK=1 r(x, y(k)), σ(x) = K1 ∑kK=1(r(x, y(k)) − µ(x))2, ϵ is the minimum value to avoid division by zero. The policy is updated by maximizing the expected advantage-weighted log-likelihood:

πθ(y | x) πθold(y | x)

, ϵ) − β · DKL[πθ∥πref] (2) where CLIP is the clipped surrogate objective, and DKL denotes the KL divergence.

LGRPO(θ) = ExEy∼πθ(·|x) CLIP(A(x, y)

Although RLVR algorithms like GRPO have achieved great success, they typically require large amounts of perfectly annotated data. In practice, however, due to the scarcity of experts, noisy labels are unavoidable. Moreover, despite their prevalence, in-depth analysis of noisy labels in RLVR is almost nonexistent. Therefore, we take the first step toward understanding their mechanism. To establish a formal understanding, we first define noisy labels within the RLVR framework. Specifically, for a prompt x, let y⋆(x) denote the groundtruth solution and y˜(x) the potentially corrupted training label. Furthermore, whether a noisy label will be reinforced depends critically on whether the policy can roll it out, and we define this property as Rollout Feasibility as follows:

- Definition 3.1 (Rollout Feasibility). A solution y is rollout-feasible under πθ if πθ(y | x) > 0.

Then, based on whether the noisy label is rollout-feasible, we classify it into the following two distinct types of noisy labels1:

- Definition 3.2 (Inactive Noisy Label). y˜(x) ̸= y⋆(x) is inactive if it is not rollout-feasible: πθ(y˜ | x) = 0.
- Definition 3.3 (Active Noisy Label). y˜(x) ̸= y⋆(x) is active if it is rollout-feasible: πθ(y˜ | x) > 0.

With this formalization in place, a fundamental question arises: can models learn to reason robustly under such noisy supervision? To answer this question, we analyze the training behavior of noisy RLVR and the phenomenon, Early Correctness Coherence, observed in practice.

###### 3.2 Early Correctness Coherence Phenomenon in Noisy RLVR

As shown in Figure 2, we observe the phenomenon of Early Correctness Coherence in noisy RLVR training experiments: although noisy samples begin to lag behind in later stages, accuracy on both clean and noisy samples increases similarly in early training. This intriguing phenomenon reveals that, despite noisy supervision, the training process already produces real correct answers for noisy samples. We attribute this phenomenon to positive cross-sample coupling enabled by shared parameters, where updates from clean samples inadvertently benefit noisy ones. This mechanism explains why both groups improve in unison during early training, even under noisy supervision. We formalize this mechanism below.

1It is worth noting that the correct labels for all samples should be rollout-feasible. If this condition is not met, the sample is invalid and requires external knowledge for the model to learn it, though this falls outside the scope of our research.

- Theorem 3.4 (Early Correctness Coherence in Noisy RLVR). (Informal) Let D = Dclean ∪ Dnoise with noise ratio ρ = |Dnoise|/|D|. For each prompt x at epoch t, let pt(y|x) = πθt(y|x) and define log-ratio

⋆(x)|x)

Lt(x) = log pt(y

pt(y˜(x)|x) . Assume: (i) p0(y⋆) > p0(y˜); (ii) positive cross-sample coupling between the clean sample xc and the noisy sample xn:

Exc,xn ∇θ log π(y⋆(xc)|xc) · ∇θ log π(y⋆(xn)|xn) ≥ γ > 0 (3)

(iii) mean drift ∆s = γ(1 − ρ)Gc − ρGn, where Gc and Gn are the mean clean and noisy advantage magnitudes. Let ρc = γGγGc

c+Gn. If ρ < ρc and K ≳ log(T/δ), then with probability at least 1 − δ, for all t ≤ T,

log(T/δ) K

Lt(x) ≥ L0(x) + ηt ∆s − O

, (4)

implying that pt(y⋆(x)|x) ≫ pt(y˜(x)|x).

See Appendix A for the formal Theorem A.1 and the full proof, and Appendix Figure 5 for experimental verification of Theorem 3.4’s assumptions.

Remark Theorem 3.4 suggests that in the early phase, correct answers, with increasing probability, can gradually emerge even on noisy samples through cross-sample coupling.

This indicates that correct answers may be recovered from the model’s rollouts as training progresses, motivating our method, which leverages these emerging answers for label refinement.

###### 3.3 Online Label Refinement (OLR)

The analysis of Early Correctness Coherence above suggests that correct solutions tend to emerge with progressively higher probability during early training, even on noisy samples. Thus, answers whose rollout probability consistently increases are likely to be correct; if such answers differ from the provided label, the original supervision is likely noisy and can be replaced, which forms the foundation of our Online Label Refinement (OLR).

Formally, at epoch t, the policy generates K rollouts Yt(x) = {yt(1), . . . , y(tK)} ∼ πθ(· | x). To estimate solution probabilities, we use the empirical pass rate from these rollouts. Since enumerating all possible answers

is infeasible, we track only the majority answer ymajt (x) = argmaxc |{y ∈ Yt(x) : y = c}|, with pass rate pmajt (x) = K1 |{y ∈ Yt(x) : y = ymajt (x)}|. We maintain a trajectory Ht(x) = {(t′, ymajt′ (x), pmajt′ (x)) : t′ ∈ [1, t]} to evaluate the reliability of the candidate answer.

- Criterion 1: Positive Convergence Slope. To detect reliable improvement, we compute the linear regression slope of the pass-rate trajectory. Let pt(x) = [pmaj1 (x), . . . , pmajt (x)]⊤ and t = [1, . . . , t]⊤. The slope is

St(x) =

(t − t¯1)⊤(pt(x) − p¯1) (t − t¯1)⊤(t − t¯1)

, (5)

where t¯ = 1t ∑it=1 i and p¯ = 1t ∑it=1 pmaji (x). A positive slope St(x) > δslope indicates increasing model confidence.

- Criterion 2: Historical Consistency. To filter stochastic fluctuations, we check whether the current majority answer matches the historical majority yhistt (x) = argmaxy |{(t′, ymajt′ (x)) ∈ Ht(x) : ymajt′ (x) = y}|. The

consistency indicator is

Ct(x) = I ymajt (x) = yhistt (x) . (6)

After an initial early learning phase of T epochs to accumulate rollout statistics, the effective label yˆt(x) used for reward computation is

yˆt(x) =

ymajt (x) if St(x) > δslope and Ct(x), y˜(x) otherwise.

(7)

We adopt GRPO as the base optimizer; incorporating OLR enhances the model’s tolerance to noisy labels during training. See below for theoretical analysis and Fig. 3 (f) for experimental validation.

- Theorem 3.5 (OLR Improves Label Noise Tolerance). Let ∆ denote the probability that a noisy prompt satisfies the OLR replacement criteria. When OLR replaces a label, the selected label equals the correct

solution with probability at least Pr(yˆt(x) = y⋆(x)) ≥ 1 − ϵ, where ϵ = O(exp −K∆2p ) and ∆p denotes the probability gap between the correct solution and competing answers. Consequently, the effective noise ratio becomes ρeff = ρ(1 − ∆) < ρ, which increases the tolerable noise threshold from ρc to ρOLRc = 1−ρc∆ > ρc

For a detailed proof, please refer to Appendix A.8. Overall, OLR dynamically selects reliable majority answers to rewrite noisy labels by analyzing the slope of the pass rate trend for majority answers and their historical consistency, thereby achieving effective denoising. Compared to model inference and update, the additional time overhead introduced by OLR is almost negligible (see Appendix Table 9). Below, we demonstrate OLR’s effectiveness through experimental results.

#### 4 Experiment

In Section 4.1, we introduce the dataset used in the experiments, the design for generating noise labels, and the baseline methods we compare. In Section 4.2, we analyze the performance gains of OLR under various noise ratios, compare it with baselines, examine training dynamics, present parameter sensitivity analysis, and ablation results. Appendix C presents additional experiments on model sizes, LLM types, Early Correctness Coherence, and training dynamics under inactive noise.

###### 4.1 Setup

Dataset and Benchmarks. To investigate how varying proportions of active and inactive noisy labels affect RLVR, we sample 800 instances from DAPO-Math (Yu et al., 2025) (default setting) and inject noise at ratios of {0.1, 0.3, 0.5, 0.7, 0.9}. Additionally, to examine scalability, we sample 4,000 instances with a noise ratio of 0.5. For inactive noise, we replace correct labels with strings that the model is unlikely to output. However, static construction of active noise is impractical, as the probability that noisy labels actually appear during rollout depends on both the noise ratio and the model’s output probability2. Therefore, we dynamically construct active noise labels during inference by replacing correct labels with incorrect answers generated by the model in an on-policy manner. For evaluation, we use six in-distribution math reasoning benchmarks: AIME 2024, AIME 2025, AMC (Li et al., 2024), Minerva (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), and MATH-500 (Hendrycks et al., 2021). We report avg@32 on AIME 2024/2025 and AMC (due to small test sets) and pass@1 on the others. For out-of-distribution generalization, we evaluate on

2For example, with 50% active noise proportion (400/800 samples) and 50% model confidence on these noisy labels, rolling out each sample 8 times gives only (1 − 0.58)400 ≈ 0.21 probability that all noisy samples output noisy labels.

Table 1 Results on Qwen3-4B-Base under different noise ratios. Each numerical subscript indicates the absolute improvement or degradation compared with naive GRPO without OLR. Bold indicates the better.

In-Distribution Out-of-Distribution

Method

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Qwen3-4B-Base 9.6 4.2 34.2 52.6 19.5 28.4 24.8 35.8 14.1 33.3 27.7

###### Inactive Noisy Label

ρ = 0.1 21.3 20.4 52.7 83.6 39.7 49.3 44.5 86.3 37.4 59.0 60.9

- w/ OLR 22.5 19.6 55.0 83.0 41.2 48.3 44.9+0.4 88.9 37.9 59.5 62.1+1.2

- ρ = 0.3 12.9 8.3 50.2 75.8 42.0 43.0 38.7 88.1 29.8 56.2 58.0

w/ OLR 23.8 16.7 53.5 82.6 40.8 48.7 44.4+5.7 88.1 36.4 57.6 60.7+2.7 ρ = 0.5 12.5 5.0 45.3 76.0 34.9 42.7 36.1 86.7 28.8 56.0 57.2

w/ OLR 21.7 18.3 53.9 84.2 42.3 48.9 44.9+8.8 86.8 38.4 57.8 61.0+3.8

- ρ = 0.7 9.6 8.8 45.9 76.4 37.5 42.7 36.8 83.4 26.3 54.5 54.7 w/ OLR 14.6 11.3 48.0 79.2 37.1 43.4 38.9+2.1 87.4 29.8 57.3 58.2+3.5

ρ = 0.9 12.5 9.6 44.3 73.4 37.5 40.0 36.2 76.5 27.3 51.5 51.8 w/ OLR 10.8 9.6 48.2 76.6 37.1 42.2 37.4+1.2 85.9 30.3 55.3 57.2+5.4 Active Noisy Label

ρ = 0.1 13.8 9.6 48.0 77.8 37.5 42.4 38.2 84.6 30.3 55.5 56.8

w/ OLR 19.2 19.2 52.0 81.6 39.7 52.0 44.0+5.8 86.8 35.4 59.1 60.4+3.6 ρ = 0.3 13.8 9.6 47.1 73.6 36.0 41.8 37.0 80.8 25.3 51.7 52.6

w/ OLR 13.8 12.9 50.3 77.4 36.4 41.3 38.7+1.7 85.4 33.3 56.7 58.5+5.9 ρ = 0.5 10.4 8.8 45.8 74.6 33.1 40.4 35.5 73.9 24.7 48.6 49.1

w/ OLR 20.4 15.4 49.7 81.4 36.4 48.1 41.9+6.4 78.5 28.3 54.8 53.9+4.8

- ρ = 0.7 10.4 9.2 45.5 74.2 31.6 39.0 35.0 66.2 24.7 49.2 46.7 w/ OLR 10.8 10.0 47.1 76.2 37.5 41.5 37.2+2.2 80.1 31.8 52.4 54.8+8.1

ρ = 0.9 8.3 5.8 27.0 43.0 21.7 24.3 21.7 34.2 12.6 30.5 25.8

w/ OLR 7.1 5.8 35.7 47.8 25.7 28.6 25.1+3.4 31.1 16.2 32.1 26.5+0.7

ARC-c (Clark et al., 2018), GPQA-diamond (Rein et al., 2024) (denoted GPQA∗), and MMLU-Pro (Wang et al., 2024a), covering open-domain reasoning, graduate-level science, and academic reasoning.

Baseline Methods. We compare two categories of baseline methods. The first category consists of unsupervised approaches that do not rely on ground-truth labels: (1) TTRL (Zuo et al., 2025); (2) Co-Reward (Zhang et al., 2025b); (3) Self-Certainty (Zhao et al., 2025); (4) Token-Level Entropy (Agarwal et al., 2025); and (5) Sentence-Level Entropy (Agarwal et al., 2025). The second category includes noise-robust learning methods and transfer-friendly regularization techniques: (6) Confidence Penalty (Pereyra et al., 2017); (7) Label Smoothing (Lukasik et al., 2020); (8) Small-loss Selection (Gui et al., 2021); and (9) Random Selection. Detailed description is in Appendix B.2.

###### 4.2 Experimental Results

How does OLR perform under different noise ratios? The results in Table 1 demonstrate that OLR consistently improves post-training model performance across a range of noise ratios (0.1, 0.3, 0.5, 0.7, and

- 0.9) under both active and inactive noise labeling scenarios. Under inactive noise, OLR achieves average gains of 3.6% on six in-distribution mathematical benchmarks and 3.3% on three out-of-distribution benchmarks. Notably, even at a high noise ratio of 50%, OLR maintains performance comparable to that observed at 10% noise, highlighting its robustness in inactive noise settings. In contrast, active noise labeling leads to a sharper decline in performance as noise increases, with the model collapsing and final performance falling below that of the initial model. In this more challenging active noise scenario, OLR still yields substantial improvements: average gains of 3.9% on the mathematical benchmarks and 4.6% on the out-of-distribution benchmarks. These results demonstrate that OLR not only enhances robustness under inactive noise but also delivers significant gains in complex active noise environments.

Table 2 Results on Qwen3-4B-Base under a 50% noise ratio with different baselines. Each numerical subscript indicates the absolute improvement or degradation compared with naive GRPO. Bold indicates the best in inactive noise scenarios, and underline indicates the best in active noise scenarios.

In-Distribution Out-of-Distribution

Method

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Qwen3-4B-Base 9.6 4.2 34.2 52.6 19.5 28.4 24.8 35.8 14.1 33.3 27.7

###### Unsupervised Methods

TTRL 12.1 8.3 48.2 76.4 36.0 40.3 36.9 86.9 30.3 56.0 57.7 Co-reward 12.1 7.1 44.9 76.0 38.9 40.7 36.6 86.1 29.3 55.8 57.1 Self-certainty 5.8 5.0 28.9 41.8 21.7 20.6 20.6 24.3 12.1 30.6 22.3 Token-entropy 3.8 2.1 22.3 44.2 15.8 15.7 17.3 17.0 7.6 28.9 17.8 Seq-entropy 4.6 2.1 25.2 41.0 19.5 19.4 18.6 18.6 6.6 28.4 17.9

###### Active Noisy Label

GRPO 10.4 8.8 45.8 74.6 33.1 40.4 35.5 73.9 24.7 48.6 49.1 w/ Confidence Penalty 12.5 9.6 47.0 76.4 34.2 39.6 36.6+1.1 82.9 32.3 53.3 56.2+7.1 w/ Label Smoothing 14.6 7.1 44.1 73.2 33.5 38.1 35.1-0.4 55.5 24.2 47.4 42.4-6.7 w/ Samll-loss Select 6.7 1.7 22.4 43.8 16.5 17.6 18.1-17.4 32.3 9.6 26.7 22.9-26.2 w/ Random Select 9.2 8.8 43.7 74.6 33.1 39.6 34.8-0.7 45.6 14.6 46.3 35.5-13.6 w/ OLR 20.4 15.4 49.7 81.4 36.4 48.1 41.9+6.4 78.5 28.3 54.8 53.9+4.8

Inactive Noisy Label

GRPO 12.5 5.0 45.3 76.0 34.9 42.7 36.1 86.7 28.8 56.0 57.2

w/ Confidence Penalty 13.3 7.9 48.6 76.6 39.0 43.1 38.1+2.0 87.4 31.3 56.6 58.4+1.2 w/ Label Smoothing 11.7 7.9 47.3 78.8 39.0 42.8 37.9+1.8 86.5 29.8 57.4 57.9+0.7 w/ Samll-loss Select 12.1 9.6 46.4 76.8 36.4 38.5 36.6+0.5 73.9 30.3 50.7 51.6-5.6 w/ Random Select 11.3 12.1 47.4 77.8 38.2 41.3 38.0+1.9 78.1 31.8 55.0 55.0-2.2 w/ OLR 21.7 18.3 53.9 84.2 42.3 48.9 44.9+8.8 86.8 38.4 57.8 61.0+3.8

Table 3 Results on Qwen3-4B-Base under a 50% noise ratio with 4K samples. Bold indicates the better.

In-Distribution Out-of-Distribution

Method

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Active Noisy Label

w/o OLR 12.9 7.5 49.4 76.4 39.0 41.5 37.8 85.1 26.8 54.1 55.3

- w/ OLR 20.0 18.3 50.0 78.8 37.1 43.9 41.4 82.9 28.8 55.4 55.7

Inactive Noisy Label

w/o OLR 17.9 12.9 50.2 78.4 41.5 41.9 40.5 88.4 33.3 57.3 59.7

- w/ OLR 21.7 20.8 55.6 85.2 44.9 48.3 46.1 90.4 38.9 60.6 63.3

How does OLR perform compared to baseline methods? We present OLR’s performance relative to baseline methods in Table 2 under a 50% noise-ratio scenario from two perspectives: unsupervised methods and traditional noisy label learning approaches. (i) Unsupervised methods. Most unsupervised methods lead to severe model collapse. The best among them, TTRL and Co-Reward, only marginally outperform naive noisy training on in-distribution tasks, suggesting that abandoning labels entirely is suboptimal. On out-of-distribution tasks, TTRL shows stronger robustness to active noise. Nevertheless, OLR achieves superior overall performance: average gains of 5.0% and 8.0% over unsupervised SOTA in ID settings under active and inactive noise, respectively, and outperforms the best unsupervised method by 3.3% on OOD tasks. (ii) Traditional noisy label learning methods. We also find most noisy label methods ineffective in RLVR. Among active noisy label settings, only Confidence Penalty improves over naive training; others harm performance. Notably, small-loss selection declines most sharply, underperforming both baseline and random selection. This is because in RLVR, small loss frequently signals uninformative samples (e.g., rollouts entirely correct or incorrect) rather than high label quality, offering little improvement signal. In ID benchmarks, OLR outperforms the best of these methods by an average of 6.1%.

How does OLR behave during the training process? Figure 3 illustrates training dynamics under 50% active noise. As shown in (a), during the early learning phase, the majority answer accuracy for both clean and noisy samples rises to over 60%, thereby exceeding the initial 50% correct label ratio. This early learning

[Figure 7]

[Figure 8]

[Figure 9]

Early Learning Phase Label Refining Phase

(a) Training Accuracy (b) Selected vs. Unselected Accuracy (c) Passrate Slope

[Figure 10]

[Figure 11]

[Figure 12]

(d) Selection Ratio (e) Ablation Accuracy (f) Real vs. Initial Noise Figure 3 Training dynamics of OLR with Qwen3-4B-Base under an active noise setting (noise ratio = 0.5).

[Figure 13]

[Figure 14]

[Figure 15]

(a) Sensitivity analysis (b) Ablation study Figure 4 Results under a 50% noise ratio on Qwen-3-4B-Base.

stage enables initial gains; however, accuracy plateaus around 60% afterward. Once OLR is applied, accuracy further increases to over 80%, achieving a 20% improvement, demonstrating that OLR builds upon the early learning stage. In (b), for samples selected by OLR, the majority answer accuracy exceeds 90% and approaches 100%, whereas unselected samples remain below 70%. Since unselected samples retain their original labels, the remaining noise consists only of original noisy labels (reduced by 30%) and a small fraction (below 5%) of erroneous majority answers from selected samples. Furthermore, (c) shows that the slope of the pass rate for the majority answers remains positive throughout training for both clean and noisy samples. This persistently positive slope, combined with the rising accuracy in (a), indicates that correct answers can gradually emerge with increasing probability even on noisy samples through cross-sample coupling. (d) shows that OLR selects over 40% of samples from both clean and noisy sets, indicating it chooses sufficiently without favoring only simple samples. (e) demonstrates that combining both criteria achieves nearly 100% accuracy in sample selection, while removing either reduces accuracy by about 20%, highlighting the indispensability of both criteria. Finally, (f) demonstrates that OLR reduces the proportion of complex active noise by approximately 30% during the refinement phase, thereby showcasing its robustness and effectiveness. Figure 6 in the Appendix illustrates the training dynamics in an inactive noise scenario.

Does OLR’s effectiveness scale with more data? To evaluate OLR’s performance with larger datasets, we conduct experiments using a training set of 4,000 samples with a 50% noise ratio. As shown in Table 3, OLR continues to significantly enhance the model’s reasoning performance, yielding average improvements of 4.6% on the ID benchmarks and 2.0% on the OOD benchmarks across both noise scenarios.

Parameter sensitivity analysis. We analyze OLR’s sensitivity to slope threshold δslope and early learning phase duration T. Across all settings, OLR consistently outperforms naive noisy training (Figure 4a).

Sweeping δslope ∈ {0.0,0.025,0.05,0.075,0.1,0.2} shows optimal performance at 0.05 for in-distribution (both noise types) and out-of-distribution under inactive noise. Too low δslope introduces erroneous majority answers; too high yields insufficient refinement samples. No clear trend emerges for out-of-distribution under active noise. Varying T ∈ {2,4,6,8,10} also yields optimal performance at T = 5 for the same settings. A smaller T causes inaccurate slope estimation; a larger T prolongs noisy training. Again, no clear pattern is observed for out-of-distribution under active noise, which we aim to address in future research. Table 6 and 7 in the Appendix show the specific experimental results.

Ablation experiments. We conduct ablation studies to evaluate the contribution of each criterion in OLR. As shown in Figure 4b, removing either the slope or the historical consistency criterion consistently leads to a substantial performance drop, under both active and inactive noise and across ID and OOD benchmarks. Table 8 in the Appendix shows the specific experimental results. These results confirm that the two criteria are complementary and both essential to the effectiveness of OLR.

#### 5 Conclusion

In this paper, we take the first step toward a systematic analysis of noisy label mechanisms in RLVR. We first classify noisy labels based on rollout feasibility into active and inactive categories, establishing a taxonomy for future research. Second, we identify Early Correctness Coherence: clean and noisy samples improve similarly in early training, despite later divergence. Finally, building on this insight, we propose Online Label Refinement (OLR), the first method in RLVR that denoises labels during training. Experiments show OLR achieves significant denoising effectiveness and improves reasoning performance across various noise ratios. Beyond RLVR, our research holds promise for broader applications, including VLM and agent domains, where noisy label learning remains an underexplored direction, providing a foundation for future research across diverse paradigms.

#### References

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. The unreasonable effectiveness of entropy minimization in llm reasoning. arXiv preprint arXiv:2505.15134, 2025.

Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, and Tongliang Liu. Understanding and improving early stopping for learning with noisy labels. Advances in Neural Information Processing Systems, 34:24392–24403, 2021.

David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. Advances in neural information processing systems, 32, 2019.

Maria Sofia Bucarelli, Lucas Cassano, Federico Siciliano, Amin Mantrach, and Fabrizio Silvestri. Leveraging inter-rater agreement for classification in the presence of noisy labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3439–3448, 2023.

Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj, and Marios Savvides. Softmatch: Addressing the quantity-quality trade-off in semi-supervised learning. arXiv preprint arXiv:2301.10921, 2023.

Pengfei Chen, Ben Ben Liao, Guangyong Chen, and Shengyu Zhang. Understanding and utilizing deep neural networks trained with noisy labels. In International conference on machine learning, pages 1062–1070. PMLR, 2019.

Xinlei Chen and Abhinav Gupta. Webly supervised learning of convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 1431–1439, 2015.

Hao Cheng, Zhaowei Zhu, Xing Sun, and Yang Liu. Mitigating memorization of noisy labels via regularization between representations. arXiv preprint arXiv:2110.09022, 2021.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

Mark Collier, Basil Mustafa, Efi Kokiopoulou, Rodolphe Jenatton, and Jesse Berent. Correlated input-dependent label noise in large-scale image classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1551–1560, 2021.

Chuanwen Feng, Yilong Ren, and Xike Xie. Ot-filter: An optimal transport filter for learning with noisy labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16164–16174, 2023.

Jacob Goldberger and Ehud Ben-Reuven. Training deep neural-networks using a noise adaptation layer. In International conference on learning representations, 2017.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Xian-Jin Gui, Wei Wang, and Zhang-Hao Tian. Towards understanding deep learning from noisy labels with small-loss criterion. arXiv preprint arXiv:2106.09291, 2021.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi,

et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. Bo Han, Jiangchao Yao, Gang Niu, Mingyuan Zhou, Ivor Tsang, Ya Zhang, and Masashi Sugiyama. Masking: A new perspective of

noisy supervision. Advances in neural information processing systems, 31, 2018a.

Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. Advances in neural information processing systems, 31, 2018b.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.

Dan Hendrycks, Mantas Mazeika, Duncan Wilson, and Kevin Gimpel. Using trusted data to train deep networks on labels corrupted by severe noise. Advances in neural information processing systems, 31, 2018.

Dan Hendrycks, Kimin Lee, and Mantas Mazeika. Using pre-training can improve model robustness and uncertainty. In International conference on machine learning, pages 2712–2721. PMLR, 2019.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model, 2025. https://arxiv.org/abs/2503.24290.

Jinchi Huang, Lie Qu, Rongfei Jia, and Binqiang Zhao. O2u-net: A simple noisy label detection approach for deep neural networks. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3326–3334, 2019.

Zhizhong Huang, Junping Zhang, and Hongming Shan. Twin contrastive learning with noisy labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11661–11670, 2023.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Simon Jenni and Paolo Favaro. Deep bilevel learning. In Proceedings of the European conference on computer vision (ECCV), pages 618–633, 2018.

Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In International conference on machine learning, pages 2304–2313. PMLR, 2018.

Nazmul Karim, Mamshad Nayeem Rizve, Nazanin Rahnavard, Ajmal Mian, and Mubarak Shah. Unicon: Combating label noise through uniform selection and contrastive learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9676–9686, 2022.

Taehyeon Kim, Jongwoo Ko, JinHwan Choi, Se-Young Yun, et al. Fine samples for learning with noisy labels. Advances in Neural Information Processing Systems, 34:24137–24149, 2021a.

Youngdong Kim, Juseung Yun, Hyounguk Shon, and Junmo Kim. Joint negative and positive learning for noisy labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9442–9451, 2021b.

Kimin Lee, Sukmin Yun, Kibok Lee, Honglak Lee, Bo Li, and Jinwoo Shin. Robust inference via generative classifiers for handling noisy labels. In International conference on machine learning, pages 3763–3772. PMLR, 2019.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q. Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. https://huggingface.co/datasets/Numinamath, 2024. Hugging Face repository, 13:9.

Junnan Li, Richard Socher, and Steven CH Hoi. Dividemix: Learning with noisy labels as semi-supervised learning. arXiv preprint

arXiv:2002.07394, 2020. Mengting Li and Chuang Zhu. Noisy label processing for classification: A survey. arXiv preprint arXiv:2404.04159, 2024. Pengyi Li, Matvey Skripkin, Alexander Zubrey, Andrey Kuznetsov, and Ivan Oseledets. Confidence is all you need: Few-shot rl

fine-tuning of language models. arXiv preprint arXiv:2506.06395, 2025a. Shikun Li, Xiaobo Xia, Shiming Ge, and Tongliang Liu. Selective-supervised contrastive learning with noisy labels. In Proceedings

of the IEEE/CVF conference on computer vision and pattern recognition, pages 316–325, 2022. Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling. arXiv preprint arXiv:2502.11886, 2025b. Yifan Li, Hu Han, Shiguang Shan, and Xilin Chen. Disc: Learning from noisy labels via dynamic instance-specific selection and

correction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24070–24079, 2023. Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like

training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025. Michal Lukasik, Srinadh Bhojanapalli, Aditya Menon, and Sanjiv Kumar. Does label smoothing mitigate label noise? In International Conference on Machine Learning, pages 6448–6458. PMLR, 2020. Eran Malach and Shai Shalev-Shwartz. Decoupling" when to update" from" how to update". Advances in neural information processing systems, 30, 2017. Aditya Krishna Menon, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar. Can gradient clipping mitigate label noise? In International conference on learning representations, 2020. Youssef Mroueh. Reinforcement learning with verifiable rewards: Grpo’s effective loss, dynamics, and success amplification. arXiv preprint arXiv:2503.06639, 2025. Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with noisy labels. Advances in neural information processing systems, 26, 2013. Duc Tam Nguyen, Chaithanya Kumar Mummadi, Thi Phuong Nhung Ngo, Thi Hoai Phuong Nguyen, Laura Beggel, and Thomas Brox. Self: Learning to filter noisy labels with self-ensembling. arXiv preprint arXiv:1910.01842, 2019.

Diego Ortego, Eric Arazo, Paul Albert, Noel E O’Connor, and Kevin McGuinness. Multi-objective interpolation training for robustness to label noise. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6606–6615, 2021.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1944–1952, 2017.

Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548, 2017.

Yuntao Qu, Shasha Mo, and Jianwei Niu. Dat: Training deep networks robust to label-noise by matching the feature distributions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6821–6829, 2021.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. https://openreview.net/forum?id=Ti67584b98.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. https: //arxiv.org/abs/2402.03300.

Yanyao Shen and Sujay Sanghavi. Learning with bad training data via iterative trimmed loss minimization. In International conference on machine learning, pages 5739–5748. PMLR, 2019.

Hwanjun Song, Minseok Kim, and Jae-Gil Lee. Selfie: Refurbishing unclean samples for robust deep learning. In International conference on machine learning, pages 5907–5915. PMLR, 2019.

Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, and Jae-Gil Lee. Robust learning by self-transition for handling noisy labels. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pages 1490–1500, 2021.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.

Sainbayar Sukhbaatar, Joan Bruna, Manohar Paluri, Lubomir Bourdev, and Rob Fergus. Training convolutional networks with noisy labels. arXiv preprint arXiv:1406.2080, 2014.

Ryutaro Tanno, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C Alexander, and Nathan Silberman. Learning from noisy labels by regularized estimation of annotator confusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11244–11253, 2019.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Yikai Wang, Xinwei Sun, and Yanwei Fu. Scalable penalized regression for noise detection in learning with noisy labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 346–355, 2022.

Yisen Wang, Weiyang Liu, Xingjun Ma, James Bailey, Hongyuan Zha, Le Song, and Shu-Tao Xia. Iterative learning with open-set noisy labels. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8688–8696, 2018.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024a.

Zhaoyang Wang, Weilei He, Zhiyuan Liang, Xuchao Zhang, Chetan Bansal, Ying Wei, Weitong Zhang, and Huaxiu Yao. Cream: Consistency regularized self-rewarding language models. arXiv preprint arXiv:2410.12735, 2024b.

Hongxin Wei, Lei Feng, Xiangyu Chen, and Bo An. Combating noisy labels by agreement: A joint training method with coregularization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13726–13735, 2020.

Hongxin Wei, Lue Tao, Renchunzi Xie, and Bo An. Open-set label noise can improve robustness against inherent label noise. Advances in Neural Information Processing Systems, 34:7978–7992, 2021.

Pengxiang Wu, Songzhu Zheng, Mayank Goswami, Dimitris Metaxas, and Chao Chen. A topological filter for learning with label noise. Advances in neural information processing systems, 33:21382–21393, 2020.

Tianhao Wu, Weizhe Yuan, Olga Golovneva, Jing Xu, Yuandong Tian, Jiantao Jiao, Jason Weston, and Sainbayar Sukhbaatar.

Meta-rewarding language models: Self-improving alignment with llm-as-a-meta-judge. arXiv preprint arXiv:2407.19594, 2024. Zhi-Fan Wu, Tong Wei, Jianwen Jiang, Chaojie Mao, Mingqian Tang, and Yu-Feng Li. Ngc: A unified framework for learning with

open-world noisy data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 62–71, 2021.

Xiaobo Xia, Tongliang Liu, Bo Han, Chen Gong, Nannan Wang, Zongyuan Ge, and Yi Chang. Robust early-learning: Hindering the memorization of noisy labels. In International conference on learning representations, 2020.

Tong Xiao, Tian Xia, Yi Yang, Chang Huang, and Xiaogang Wang. Learning from massive noisy labeled data for image classification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2691–2699, 2015.

Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, and Tong Zhang. Self-rewarding correction for mathematical reasoning. arXiv preprint arXiv:2502.19613, 2025.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025a.

Yuchen Yan, Jin Jiang, Zhenbang Ren, Yijun Li, Xudong Cai, Yang Liu, Xin Xu, Mengdi Zhang, Jian Shao, Yongliang Shen, et al.

Verifybench: Benchmarking reference-based reward systems for large language models. arXiv preprint arXiv:2505.15801, 2025b. An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv,

et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. Jiangchao Yao, Jiajie Wang, Ivor W Tsang, Ya Zhang, Jun Sun, Chengqi Zhang, and Rui Zhang. Deep learning from noisy image labels with quality embedding. IEEE Transactions on Image Processing, 28(4):1909–1922, 2018.

Yazhou Yao, Zeren Sun, Chuanyi Zhang, Fumin Shen, Qi Wu, Jian Zhang, and Zhenmin Tang. Jo-src: A contrastive approach for combating noisy labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5192–5201, 2021.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor Tsang, and Masashi Sugiyama. How does disagreement help generalization against label corruption? In International conference on machine learning, pages 7164–7173. PMLR, 2019.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. Self-rewarding language models. arXiv preprint arXiv:2401.10020, 3, 2024.

Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F Wong, and Yu Cheng. Exgrpo: Learning to reason from experience. arXiv preprint arXiv:2510.02245, 2025.

Chang-Bin Zhang, Peng-Tao Jiang, Qibin Hou, Yunchao Wei, Qi Han, Zhen Li, and Ming-Ming Cheng. Delving deep into label smoothing. IEEE Transactions on Image Processing, 30:5984–5996, 2021a.

Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.

Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. arXiv preprint arXiv:2504.05812, 2025a.

Yikai Zhang, Songzhu Zheng, Pengxiang Wu, Mayank Goswami, and Chao Chen. Learning with feature-dependent label noise: A progressive approach. arXiv preprint arXiv:2103.07756, 2021b.

Zizhuo Zhang, Jianing Zhu, Xinmu Ge, Zihua Zhao, Zhanke Zhou, Xuan Li, Xiao Feng, Jiangchao Yao, and Bo Han. Co-reward: Selfsupervised reinforcement learning for large language model reasoning via contrastive agreement. arXiv preprint arXiv:2508.00410, 2025b.

Xuandong Zhao, Zhewei Kang, Aosong Feng, Sergey Levine, and Dawn Song. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Tianyi Zhou, Shengjie Wang, and Jeff Bilmes. Robust curriculum learning: from clean label detection to noisy label self-correction. In International conference on learning representations, 2020.

Xin Zhou, Yiwen Guo, Ruotian Ma, Tao Gui, Qi Zhang, and Xuanjing Huang. Self-consistency of the internal reward models improves self-rewarding language models. arXiv preprint arXiv:2502.08922, 2025.

Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, and Bowen Zhou. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

# Appendix

#### A Theoretical Proof

Theorem A.1 (Early Correctness Coherence in Noisy RLVR). (Formal) Consider a dataset D = Dclean ∪ Dnoise with noise ratio ρ = |Dnoise|/|D|. For each prompt x, let the policy have rollout probability pt(y|x) = πθt(y|x) with K rollouts per step. Assume small learning rate η and early-phase dynamics (ignoring clipping effects). Suppose (i) initial rollout bias p0(y⋆) > p0(y˜) creating a signal gap, (ii) positive cross-sample coupling

Γ(xc, xn) = ∇θ log π(y⋆(xc)|xc) · ∇θ log π(y⋆(xn)|xn), Exc∈Dclean,xn∈Dnoise[Γ(xc, xn)] ≥ γ > 0,

(8) so that reinforcing clean samples increases correct probability on noisy samples, and (iii) average deterministic drift of the log-ratio ∆s := γ(1 − ρ)Gc − ρGn, where Gc and Gn are the mean clean and noisy advantage magnitudes. Define the log-ratio

pt(y⋆(x)|x) pt(y˜(x)|x)

γGc γGc + Gn

Lt(x) = log

. (9)

and ρc =

If ρ < ρc and K ≳ log(T/δ), then with probability at least 1 − δ simultaneously for all t ≤ T, where T denotes the early convergence stage,

Lt(x) ≥ L0(x) + ηt ∆s − O

implying that pt(y⋆(x)|x) ≫ pt(y˜(x)|x).

- A.1 Notation and Setup We consider a dataset D = Dclean ∪ Dnoise with noise ratio

ρ = |Dnoise|

.

|D|

log(T/δ) K

, (10)

For each prompt x ∈ D, let y⋆(x) denote the correct solution and y˜(x) denote a active noisy label. Both are assumed rollout-feasible. The policy is πθ(y|x), and we define

pt(y|x) = πθt(y|x), with K rollouts per step. Binary reward:

###### r(x, y) − µ(x) σ(x) + ϵ

r(x, y) ∈ {0,1}, A(x, y) =

.

###### A.2 Finite-Rollout Concentration Bounds

- Lemma A.2 (Advantage Concentration). For any prompt x, let µˆ(x) and σˆ(x) be the empirical mean and standard deviation computed from K rollouts. Then, for any δ > 0, with probability at least 1 − δ:

log(2/δ) 2K

|µˆ(x) − E[µ(x)]| ≤

, (11)

log(2/δ) K

|σˆ(x) − σ(x)| ≤ C

, (12)

|Aˆ(x, y) − A(x, y)| ≤ C′ log(1/δ)

, (13)

K

where C, C′ are constants depending on reward bounds. Proof. We start with the empirical mean of K rollouts for a fixed prompt x:

µˆ(x) =

K

1 K

r(x, y(k)),

### ∑

k=1

where each r(x, y(k)) ∈ [0,1] and y(k) ∼ πθ(·|x).

- Step 1: Concentration of the empirical mean. Since the r(x, y(k)) are independent conditioned on the policy, we can apply Hoeffding’s inequality:

Pr |µˆ(x) − E[µˆ(x)]| > ϵ ≤ 2exp − 2Kϵ2 ,

where E[µˆ(x)] = µ(x) is the true expected reward for this prompt. Solving for ϵ with confidence level 1 − δ gives:

|µˆ(x) − µ(x)| ≤

log(2/δ) 2K

with probability at least 1 − δ. (14)

- Step 2: Concentration of the empirical variance. The empirical variance is

1 K

σˆ2(x) =

K

(r(x, y(k)) − µˆ(x))2.

### ∑

k=1

Since 0 ≤ r(x, y(k)) ≤ 1, we have bounded deviations:

###### |(r(x, y(k)) − µˆ(x))2 − (r(x, y(k)) − µ(x))2| ≤ |µˆ(x) − µ(x)|. (15)

Thus, the empirical variance is a 1-Lipschitz function of the empirical mean. Combining this with Hoeffding’s inequality for bounded r gives

log(2/δ) K for some constant C.

|σˆ(x) − σ(x)| ≤ C

###### Step 3: Concentration of normalized advantage. The normalized advantage is

r(x, y) − µˆ(x) σˆ(x) + ϵ

Aˆ(x, y) =

.

Consider the mapping (r, µˆ, σˆ)  → Aˆ(x, y). This mapping is Lipschitz in each argument: a deviation δr, δµˆ, δσˆ induces a change

|δr| + |δµˆ| σ + ϵ

###### + |r − µ|

|δAˆ| ≤

(σ + ϵ)2|δσˆ|. Since r, µ, σ are bounded in [0,1], the total Lipschitz constant is finite. Therefore, combining the bounds from Steps 1–2 and applying union bound over numerator and denominator terms, we obtain

|Aˆ(x, y) − A(x, y)| ≤ C′ log(1/δ)

,

K

with probability at least 1 − δ, for some constant C′ depending on the reward bounds and ϵ.

| |
|---|

- A.3 Log-Ratio Dynamics and Martingale Decomposition Define the log-ratio for a active-noise sample xn:

pt(y⋆(xn)|xn) pt(y˜(xn)|xn)

Lt(xn) = log

.

⋆(xn)|xn) pt(y˜(xn)|xn)

- Lemma A.3 (Finite-Rollout Log-Ratio Dynamics with High-Probability Bound). Let Lt(xn) = log pt(y

be the log-ratio of clean vs active-noise solutions for sample xn, and let the one-step update satisfy

∆Lt(xn) = η A ˆt(xn, y⋆) − Aˆt(xn, y˜) + ηΞt(xn), where Ξt(xn) is the cross-sample coupling contribution. Define the martingale difference

Mt(xn) = ∆Lt(xn) − E[∆Lt(xn) | Ft−1]. Assume the finite-rollout advantage satisfies Lemma A.2. Then for rollout size K sufficiently large and learning rate η small, with probability at least 1 − δ,

t−1

log(1/δ) K

### ∑

E[∆Ls(xn) | Fs−1] ≤ tηC

for all t ≤ T, (16)

Lt(xn) − L0(xn) − η

s=0

where C > 0 is a constant depending on reward bounds. In particular, {Mt(xn), Ft} is a martingale difference sequence with bounded increments.

Proof. Recall the log-ratio for a active-noise sample xn:

pt(y⋆(xn)|xn) pt(y˜(xn)|xn)

Lt(xn) = log

.

- Step 1: One-step stochastic update. Under GRPO, the parameter update is

Aˆt(xn, y)∇θ log πθ(y|xn),

θt+1 = θt +η ∑

y∈Yt(xn)

where Aˆt is computed from K rollouts. Using first-order approximation for small η, the log-ratio change is

∆Lt(xn) := Lt+1(xn) − Lt(xn) ≈ η A ˆt(xn, y⋆) − Aˆt(xn, y˜) + ηΞt(xn), (17) where Ξt is the cross-sample coupling term arising from shared parameters across prompts.

- Step 2: Conditional expectation and martingale decomposition. Define the filtration Ft−1 as the sigmaalgebra generated by all previous rollouts and policy parameters up to step t − 1. Conditional on Ft−1, the stochastic gradient Aˆt(xn, y) is unbiased:

E[Aˆt(xn, y) | Ft−1] = At(xn, y),

since the rollouts y(k) ∼ πθt(·|xn) are independent given θt. Define the martingale difference sequence:

Mt(xn) := ∆Lt(xn) − E[∆Lt(xn) | Ft−1].

By construction:

1. E[Mt(xn) | Ft−1] = 0, i.e., {Mt(xn), Ft} is a martingale difference sequence. 2. The magnitude is bounded due to bounded rewards and Lemma A.2:

|Mt(xn)| = |∆Lt(xn) − E[∆Lt(xn) | Ft−1]| ≤ ηC

log(1/δ) K

with probability 1 − δ, (18) where the constant C comes from the Lipschitz dependence of the log-ratio on normalized advantage.

- Step 3: Martingale decomposition. With this definition, we can write

Lt+1(xn) = L0(xn) + η

t

∑

s=0

∆s(xn) +

t

∑

s=0

Ms(xn),

where ∆s(xn) = E[∆Ls(xn) | Fs−1] is the deterministic drift term. This decomposition separates the deterministic drift from the stochastic fluctuation (martingale), allowing application of Azuma-Hoeffding

inequality to bound deviations of Lt(xn) around its mean trajectory.

- Step 4: High-probability bound. Applying Azuma-Hoeffding to ∑ts=0 Ms(xn), since each |Ms(xn)| ≤ ηC log(1/δ)/K, we obtain

Pr

t

ϵ2 2tη2C2 log(1/δ)/K

### ∑

Ms(xn) ≥ ϵ ≤ 2exp −

s=0

. (19)

Setting ϵ = tηC log(1/δ)/K gives probability at most δ.

| |
|---|

- A.4 Cross-Sample Coupling Define cross-sample coupling:

Γ(xi, xj) = ∇θ log π(y⋆(xj)|xj) · ∇θ log π(y⋆(xi)|xi). (20)

Assume:

Exc∼Dclean,xn∼Dnoise[Γ(xc, xn)] ≥ γ > 0. Then the deterministic drift for noise sample correct log-probability is:

∆t = γ(1 − ρ)Gc − ρGn,

where Gc and Gn denote average advantage magnitudes over clean and noisy samples.

###### A.5 High-Probability Early Correctness Coherence

- Theorem A.4 (High-Probability Early Correctness Coherence). Suppose p0(y⋆) > p0(y˜), γ > 0, and K ≳ log(T/δ). Define

ρc =

γGc Gn + γGc

. If ρ < ρc, then with probability at least 1 − δ,

Lt(xn) ≥ L0(xn) + ηt γ(1 − ρ)Gc − ρGn − O(

log(T/δ) K

) (21) for all t ≤ T. In particular, Lt(xn) increases monotonically in early phase, implying

pt(y⋆(xn)|xn) ≫ pt(y˜(xn)|xn). Proof. Sum over martingale differences:

Lt = L0 +η ∑

s<t

∆s + ∑

s<t

Ms.

By Azuma-Hoeffding inequality, with probability 1 − δ, | ∑ Ms| ≤ tηC log(T/δ)/K. The deterministic drift term is positive under ρ < ρc, yielding the stated high-probability bound.

| |
|---|

A.6 Active Noise Collapse Condition

- Theorem A.5 (Active-Noise Collapse). If ρ > ρc + O( log(T/δ)/K), then with high probability Lt(xn) → −∞, i.e., active-noise solutions dominate. Proof. Recall the log-ratio for a active-noise sample xn:

pt(y⋆(xn)|xn) pt(y˜(xn)|xn)

Lt(xn) = log

.

- Step 1: Martingale decomposition. From Lemma A.3, we have the decomposition

Lt(xn) = L0(xn) + η

t−1

∑

s=0

∆s(xn) +

t−1

∑

s=0

Ms(xn),

where ∆s(xn) = E[∆Ls(xn) | Fs−1] is the deterministic drift and {Ms(xn)} is a martingale difference sequence with bounded increments

|Ms(xn)| ≤ ηC

log(1/δ) K

w.p. 1 − δ.

- Step 2: Drift under active-noise dominance. By definition, the deterministic drift is ∆s(xn) = E[∆Ls(xn) | Fs−1] = γ(1 − ρ)Gc − ρGn,

where Gc and Gn are the average clean and noisy advantage magnitudes, and γ > 0 is the cross-sample coupling coefficient. Under the condition

log(T/δ) K

γGc Gn + γGc

, with ρc =

,

ρ > ρc + O

we have

###### ∆s(xn) ≤ −ϵ < 0

for some positive ϵ depending on (ρ − ρc) and finite-rollout corrections. This ensures that the deterministic drift is negative at every step.

- Step 3: Finite-rollout martingale corrections. The martingale sum ∑ts−=10 Ms(xn) introduces stochastic fluctuations. By Azuma-Hoeffding inequality, for all t ≤ T:

Pr

t−1

∑

s=0

Ms(xn) ≥ tηC

log(1/δ) K ≤ δ.

For sufficiently large K, the martingale fluctuation is smaller in magnitude than the deterministic negative drift:

tηC

log(1/δ) K

< tηϵ, so the stochastic perturbation does not reverse the sign of the drift.

- Step 4: Log-ratio monotonic decrease. Combining Steps 2 and 3, we have with probability at least 1 − δ:

t−1

t−1

log(1/δ) K

### ∑

### ∑

∆s(xn)

). (22)

Lt(xn) = L0(xn) +

≤ L0(xn) − tη(ϵ − C

Ms(xn)

+

s=0

s=0

negative drift

bounded fluctuation

As t → ∞, the negative drift dominates and the log-ratio decreases monotonically, hence

Lt(xn) → −∞, implying that

pt(y⋆(xn)|xn) ≪ pt(y˜(xn)|xn), i.e., active-noise solutions dominate the policy with high probability.

| |
|---|

- A.7 KL-Regularized Dynamics With KL regularization term βDKL(πθ∥πref), the replicator drift becomes

∆tKL = γ(1 − ρ)Gc − ρGn − β∆ref, where

pref(y⋆) pref(y˜)

∆ref = log

.

- Theorem A.6 (KL-Regularized Phase Boundary). With KL regularization, the critical noise ratio shifts to

γGc − β∆ref Gn + γGc

ρcKL =

. The high-probability Early Correctness Coherence and collapse conditions are modified accordingly. Proof. We consider the same log-ratio for a active-noise sample xn:

pt(y⋆(xn)|xn) pt(y˜(xn)|xn)

.

Lt(xn) = log

###### Step 1: One-step update with KL regularization. Under GRPO with KL regularization, the surrogate gradient includes a term:

πθ(y|xn) πref(y|xn)

]. (23)

−β∇θDKL(πθ∥πref) = −βEy∼πθ[∇θ log πθ(y|xn) log

Using the same first-order approximation as before for small η, the deterministic drift of the log-ratio becomes

∆tKL := E[∆Lt(xn) | Ft−1] = γ(1 − ρ)Gc − ρGn − β∆ref, (24) where

pref(y⋆) pref(y˜) captures the bias introduced by the reference policy.

∆ref = log

###### Step 2: Modified critical noise ratio. In the non-KL case, the phase boundary is

γGc γGc + Gn

.

ρc =

Including the KL drift, we require the total deterministic drift to be zero at the boundary:

∆tKL = γ(1 − ρcKL)Gc − ρcKLGn − β∆ref = 0.

Solving for ρcKL gives

γGc − β∆ref Gn + γGc

γ(1 − ρcKL)Gc − ρcKLGn − β∆ref = 0 ⇒ ρcKL =

. (25)

###### Step 3: Martingale decomposition and high-probability bound. Define the martingale difference as in Lemma A.3:

Mt(xn) = ∆Lt(xn) − E[∆Lt(xn) | Ft−1]. Its magnitude is still bounded by ηC log(1/δ)/K due to Lemma A.2. Thus, the log-ratio evolves as

Lt(xn) = L0(xn) + η

t−1

t−1

∆sKL +

∑

∑

s=0

s=0

and applying Azuma-Hoeffding gives with probability 1 − δ:

t−1

∆sKL ≤ tηC

∑

Lt(xn) − L0(xn) − η

s=0

Ms(xn),

log(1/δ) K

. (26)

###### Step 4: Implications for Early Correctness Coherence and collapse. - If ρ < ρcKL, the deterministic drift ∆tKL > 0, so Lt(xn) increases, and Early Correctness Coherence occurs. - If ρ > ρcKL, the deterministic drift ∆tKL < 0, so Lt(xn) → −∞ with high probability, i.e., active-noise collapse occurs. Finite-rollout stochastic fluctuations do not change the sign of the drift if K is sufficiently large, preserving the phase boundary. This completes the proof.

| |
|---|

Inactive Noise Active Noise

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Clean labels

Wrong labels Clean labels Wrong labels

Figure 5 Results obtained with 50% inactive or active noisy labels (800 samples) using the Qwen-3-4B-Base model (Yang et al., 2025). For each sample, we take the majority vote across multiple model rollouts as the model’s prediction. The left and right pairs of columns illustrate the model’s predictions on clean and noisy samples, respectively, under both inactive and active noise. For clean samples, we display the proportion of correct (green) and incorrect (blue) predictions. For noisy samples, we show the proportion of predictions that are correct (green), that match the noisy label (red), or that match neither the true nor the noisy label

(blue). The early training phase reveals that the model learns to predict true labels even on noisy examples, indicating a preference for fitting correctly labeled samples and an increasing likelihood of producing correct answers on noisy ones over time.

###### A.8 OLR Improves Label Noise Tolerance

Proof. We prove the theorem 3.5 in three steps: correctness of the selected label, reduction of the effective noise ratio, and improvement of the tolerable noise threshold.

- Step 1: Correctness of the selected label. For prompt x with ground-truth y⋆(x), let πθ generate K rollouts Yt(x). Denote p⋆ = πθ(y⋆(x) | x) and pc = πθ(c | x) for c ̸= y⋆(x), with probability gap ∆p = p⋆ − maxc̸=y⋆(x) pc > 0 under early correctness dominance (Theorem A.4). Let Nc ∼ Binomial(K, pc), and define the majority answer ymajt (x) = argmaxc Nc. An incorrect selection occurs only if Nc ≥ Ny⋆ for some c ̸= y⋆, which by Chernoff and union bound gives

Pr ymajt (x) ̸= y⋆(x) ≤ ϵ, ϵ = O(exp −K∆2p ). Hence, whenever OLR replaces a label,

Pr(yˆt(x) = y⋆(x)) ≥ 1 − ϵ.

- Step 2: Effective noise ratio reduction. Let ρ be the original noise ratio and ∆ the probability that a noisy sample satisfies the OLR replacement criteria. Then the probability that a noisy sample remains incorrect is approximately 1 − ∆, ignoring the exponentially small ϵ, so

ρeff ≈ ρ(1 − ∆) < ρ.

- Step 3: Increased tolerable noise threshold. Let ρc be the critical noise ratio in Theorem A.4. Stability now requires

ρc 1 − ∆

=: ρOLRc . Since ∆ > 0, we have ρOLRc > ρc.

ρeff = ρ(1 − ∆) < ρc ⇒ ρ <

Conclusion. Combining the above shows that OLR (i) selects correct labels with high probability, (ii) reduces the effective noise ratio, and (iii) increases the maximum tolerable noise level, completing the proof.

| |
|---|

[Figure 20]

[Figure 21]

[Figure 22]

(a) Training Accuracy vs. Steps (b) Selected vs. Unselected Accuracy (c) Passrate Slope

[Figure 23]

[Figure 24]

[Figure 25]

(d) Selection Ratio (e) Ablation Accuracy (f) Real vs. Initial Noise Figure 6 Training dynamic of OLR with Qwen3-4B-Base under the inactive noisy (0.5 ratio) scenario. Table 4 Results on Qwen3-8B-Base under 50% noise ratio. Bold indicates the better.

In-Distribution Out-of-Distribution

Method

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Qwen3-8B-Base 10.0 7.5 38.0 61.2 27.2 33.3 29.5 28.8 13.6 46.9 29.8

###### Active Noisy Label

w/o OLR 14.6 12.5 52.3 80.2 39.7 43.9 40.5 59.6 19.2 60.3 46.4 w/ OLR 16.3 15.0 50.3 80.4 42.6 45.5 41.7 75.5 25.3 60.1 53.6

###### Inactive Noisy Label

w/o OLR 24.6 20.4 58.3 85.0 39.7 52.7 46.8 67.2 34.3 62.9 54.8

- w/ OLR 24.6 18.8 56.8 86.6 44.1 51.6 47.1 75.5 31.3 62.2 56.3

Table 5 Results on Deepseek-R1-Distill-Llama-8B under 50% noise ratio. Bold indicates the better.

Method

In-Distribution Out-of-Distribution

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Deepseek-R1-Distill-Llama-8B 15.8 17.9 48.2 72.2 18.8 35.7 34.8 24.9 10.1 38.7 24.6

Active Noisy Label

w/o OLR 21.7 56.2 19.6 72.2 18.4 40.4 38.1 20.6 13.1 41.7 25.1

- w/ OLR 25.8 17.5 60.5 73.8 18.4 39.3 39.2 27.2 14.6 41.5 27.8

Inactive Noisy Label

w/o OLR 28.9 22.9 65.8 80.0 22.8 45.2 44.2 31.0 18.7 45.0 31.6 w/ OLR 34.2 20.4 65.1 80.0 24.3 45.6 44.9 33.8 19.2 45.0 32.7

#### B Experiment Details

###### B.1 Detailed Setup

All experiments were conducted on 8 × NVIDIA A100 (80G) GPUs. The validation and test sets followed those used in prior work (Yan et al., 2025a). We set the hyperparameters to δslope = 0.05 and T = 5, with a maximum response length of 4096 and 8 rollouts. During training, we used a learning rate of 1e-6 and a KL loss weight of 1e-3. The temperature coefficient was set to 1.0 during training, and to 0.6 for both validation

and testing. Our implementation is built upon veRL3, which uses vLLM4 as the rollout generator. We thank the authors for making these open-source repositories available.

###### B.2 Baseline Description

We compare two categories of baseline methods. The first category consists of unsupervised approaches that do not rely on ground-truth labels:

- • TTRL (Zuo et al., 2025): uses majority-voted outputs as pseudo-labels;
- • Co-Reward (Zhang et al., 2025b): leverages contrastive consistency to alleviate collapse in unsupervised RLVR training;
- • Self-Certainty (Zhao et al., 2025): promotes high-confidence predictions by maximizing KL divergence;
- • Token-Level Entropy (Agarwal et al., 2025): enforces consistency by minimizing token-level entropy;
- • Sentence-Level Entropy (Agarwal et al., 2025): encourages coherent predictions by maximizing sentence-level likelihood.

The second category includes noise-robust learning methods and transfer-friendly regularization techniques:

- • Confidence Penalty (Pereyra et al., 2017): discourages overconfident predictions to reduce overfitting to noisy labels (implemented at the token level in our setting);
- • Label Smoothing (Lukasik et al., 2020): replaces one-hot targets with smoothed distributions to mitigate overconfidence and noise sensitivity (also applied at the token level);
- • Small-loss Selection (Gui et al., 2021): treats low-loss samples as clean and prioritizes them for training;
- • Random Selection: randomly samples training instances as a control variant.

#### C More Experiments

OLR performance on models of different sizes and types. We run OLR on Qwen3-8B-Base. The experiments in Table 4 show that OLR also delivers performance gains, particularly on OOD benchmarks, where it achieves an average improvement of 4.4% over models trained without OLR across both noise scenarios. At the same time, the results shown in Table 5 demonstrate that OLR improves performance on Deepseek-R1-Distill-Llama-8B (Grattafiori et al., 2024) across both ID and OOD benchmarks. This clearly demonstrates the versatility of OLR.

Detailed analysis on Early Correctness Coherence. To verify the Early Correctness Coherence phenomenon, we conduct a detailed analysis of the model’s predictions on clean and noisy samples under both noise scenarios (see Figure 5). The results substantiate the two key assumptions of Theorem 3.4: (i) correct labeling probability is higher for clean samples than for noisy ones, reflecting the quality of pre-training; and (ii) even under noisy supervision, the probability of correct answers for noisy samples increases during early training. These empirical observations lay a solid foundation for the proposed OLR method.

- 3https://github.com/volcengine/verl
- 4https://github.com/vllm-project/vllm

Table 6 Results on Qwen3-4B-Base under 50% noise ratios with different slope threshold δslope. Bold indicates the best.

In-Distribution Out-of-Distribution

Method

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Qwen3-4B-Base 9.6 4.2 34.2 52.6 19.5 28.4 24.7 35.8 14.1 33.3 27.7

Active Noisy Label

GRPO 10.4 8.8 45.8 74.6 33.1 40.4 35.5 73.9 24.7 48.6 49.1

- OLR w/ δslope = 0.0 13.3 9.2 46.8 76.4 38.2 40.7 37.4 85.4 30.8 56.8 57.7 OLR w/ δslope = 0.025 10.0 10.0 43.1 76.0 34.2 38.8 35.4 73.5 27.3 50.2 50.3

- OLR w/ δslope = 0.050 20.4 15.4 49.7 81.4 36.4 48.1 41.9 78.5 28.3 54.8 53.9 OLR w/ δslope = 0.075 12.5 11.3 45.8 75.4 36.2 39.4 36.8 82.6 31.3 56.4 56.8

OLR w/ δslope = 0.1 10.8 8.8 44.3 76.4 37.5 41.8 36.6 80.0 26.8 54.1 53.6 OLR w/ δslope = 0.2 9.6 8.8 46.1 76.2 40.1 38.8 36.6 73.0 26.8 54.2 51.3

Inactive Noisy Label

GRPO 12.5 5.0 45.3 76.0 34.9 42.7 36.1 86.7 28.8 56.0 57.2

- OLR w/ δslope = 0.0 12.9 9.2 48.0 78.2 40.1 41.8 38.4 88.0 28.3 56.1 57.5 OLR w/ δslope = 0.025 12.9 15.0 47.1 79.8 43.4 43.1 40.2 85.6 30.8 56.6 57.7

OLR w/ δslope = 0.050 21.7 18.3 53.9 84.2 42.3 48.9 44.9 86.8 38.4 57.8 61.0 OLR w/ δslope = 0.075 14.2 9.6 48.5 79.2 40.4 44.4 39.4 82.6 31.3 56.4 56.8

- OLR w/ δslope = 0.1 10.8 12.1 48.3 78.0 39.7 41.5 38.4 85.9 29.3 56.2 57.1
- OLR w/ δslope = 0.2 15.8 5.4 47.1 77.8 39.3 40.7 37.7 85.8 27.8 54.9 56.2

Table 7 Results on Qwen3-4B-Base under 50% noise ratios with different lengths of the early learning stage T. Bold indicates the best.

In-Distribution Out-of-Distribution

Method

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Qwen3-4B-Base 9.6 4.2 34.2 52.6 19.5 28.4 24.7 35.8 14.1 33.3 27.7

###### Active Noisy Label

GRPO 10.4 8.8 45.8 74.6 33.1 40.4 35.5 73.9 24.7 48.6 49.1 OLR w/ T = 2 11.3 9.6 46.0 75.8 37.1 42.5 37.1 83.4 28.3 55.4 55.7

- OLR w/ T = 4 20.4 15.4 49.7 81.4 36.4 48.1 41.9 78.5 28.3 54.8 53.9 OLR w/ T = 6 9.6 10.0 45.9 77.2 39.7 40.0 37.1 79.7 26.8 53.8 53.4 OLR w/ T = 8 12.1 7.1 43.7 76.2 37.5 41.2 36.3 84.7 31.3 53.2 56.4

Inactive Noisy Label

GRPO 12.5 5.0 45.3 76.0 34.9 42.7 36.1 86.7 28.8 56.0 57.2 OLR w/ T = 2 11.7 13.3 47.1 79.4 40.1 42.1 39.0 87.7 31.3 57.3 58.8

- OLR w/ T = 4 21.7 18.3 53.9 84.2 42.3 48.9 44.9 86.8 38.4 57.8 61.0 OLR w/ T = 6 19.2 17.9 52.9 85.0 40.1 49.0 44.0 82.3 33.8 57.9 58.0 OLR w/ T = 8 12.1 7.5 47.1 79.6 41.5 42.5 38.4 85.2 33.8 55.9 58.3

Table 8 Ablation results on Qwen3-4B-Base under 50% noise ratio. Bold indicates the best.

In-Distribution Out-of-Distribution

Method

AIME24 AIME25 AMC MATH-500 Minerva Olympiad Avg. ARC-c GPQA⋆ MMLU-Pro Avg. Active Noisy Label

- OLR 20.4 15.4 49.7 81.4 36.4 48.1 41.9 78.5 28.3 54.8 53.9

- - Positive Convergence Slope 10.8 7.9 46.7 79.0 35.7 42.1 37.0 73.9 24.7 55.5 51.4
- - Historical Consistency 13.8 10.4 46.2 77.2 35.3 40.4 37.2 78.5 26.8 54.5 53.3

Inactive Noisy Label

- OLR 21.7 18.3 53.9 84.2 42.3 48.9 44.9 86.8 38.4 57.8 61.0

- - Positive Convergence Slope 10.0 10.8 47.4 79.6 39.7 42.1 38.3 86.4 25.3 56.2 56.0
- - Historical Consistency 10.0 6.7 47.1 78.0 38.6 41.0 36.9 85.1 31.3 54.0 56.8

How does OLR behave during the RLVR with inactive noisy labels? Figure 6 shows training dynamics under 50% inactive noise. In (a), early learning lifts majority accuracy for both clean/noisy samples from 50% to over 70%. Accuracy then plateaus until OLR boosts it further to over 80%, a 10% gain showing OLR builds on early learning. In (b), OLR-selected samples exceed 90% accuracy (approaching 100%), while unselected ones stay below 80%. (c) shows positive pass rate slopes throughout training for both sample types, indicating correct answers emerge with increasing probability on noisy samples via cross-sample

##### Phase Operation Compute Complexity Time

RLVR Training Policy rollout and GRPO update O(NKL) ∼3.8 h

OLR Update Majority estimation + statistics update O(NK) negligible Total RLVR + OLR O(NKL) ∼3.8 h

Table 9 Training time and compute complexity of OLR with 800 samples. The additional overhead introduced by OLR is negligible compared with rollout generation.

coupling. (d) shows OLR selects over 40% of both clean/noisy sets, avoiding bias toward simple samples. (e) shows both criteria together achieve near 100% selection accuracy; removing either reduces accuracy by 20%, highlighting their indispensability. Finally, (f) shows OLR reduces inactive noise by 25% during refinement, demonstrating robustness.

Time Complexity. Let N denote the number of prompts, K the number of rollouts per prompt, and L the average rollout length. The dominant cost in RLVR training is rollout generation, which requires O(NKL) time per epoch due to autoregressive decoding. OLR introduces two additional operations: majority-answer counting over K rollouts and maintaining statistics for the convergence slope and historical consistency checks. Majority counting costs O(K) per prompt, while the statistics can be updated incrementally in O(1) time. Therefore, the additional overhead of OLR is O(NK) per epoch, which is negligible compared to the rollout cost O(NKL). As shown in Table 9, the empirical runtime is consistent with this analysis: training on 800 prompts takes about 3.8 hours, with negligible overhead from OLR.

#### D More Related Work

Noisy Label Learning is a mature field, but its techniques rarely accommodate RLVR’s open-ended generation space and on-policy dynamics. In traditional classification scenarios, methods for mitigating noisy labels are abundant and typically fall into the following categories:

Noise Transition Matrix Estimation (Xiao et al., 2015; Chen and Gupta, 2015; Srivastava et al., 2014; Sukhbaatar et al., 2014; Patrini et al., 2017; Goldberger and Ben-Reuven, 2017; Lee et al., 2019; Yao et al., 2018; Han et al., 2018a; Hendrycks et al., 2018; Collier et al., 2021; Bucarelli et al., 2023): These approaches generally involve learning the flipping probabilities between true and noisy labels, mimicking the noise process. However, this is unsuitable for RLVR scenarios because noisy labels in such contexts often form an open set without inherent class boundaries, making it difficult to derive a noise transition matrix.

Loss Correction (Goodfellow et al., 2014; Pereyra et al., 2017; Zhang et al., 2017; Menon et al., 2020; Hendrycks et al., 2019; Tanno et al., 2019; Jenni and Favaro, 2018; Zhang et al., 2021a; Wei et al., 2021; Lukasik et al., 2020; Xia et al., 2020; Qu et al., 2021; Cheng et al., 2021): These methods typically employ regularized algorithms to reduce model overfitting to noisy labels. While primarily designed for classification tasks, RLVR involves generative tasks, which limits the direct applicability of most existing methods.

Small-loss-based Sample Selection (Malach and Shalev-Shwartz, 2017; Wang et al., 2018; Han et al., 2018b; Jiang et al., 2018; Huang et al., 2019; Chen et al., 2019; Shen and Sanghavi, 2019; Yu et al., 2019; Wu et al., 2020; Wei et al., 2020; Wu et al., 2021; Song et al., 2021): These approaches identify samples with low training loss as potentially clean, exploiting the phenomenon of early memorization. However, in RLVR, small-loss samples are not necessarily beneficial. For instance, when a model’s rollout is either entirely

correct or entirely incorrect for a sample, its loss is zero. Yet such samples offer no training value and should be avoided (Zhan et al., 2025).

Semi-supervised Learning-based Methods (Berthelot et al., 2019; Song et al., 2019; Nguyen et al., 2019; Zhou

- et al., 2020; Li et al., 2020; Kim et al., 2021b; Zhang et al., 2021b; Bai et al., 2021; Gui et al., 2021; Kim
- et al., 2021a; Ortego et al., 2021; Yao et al., 2021; Wang et al., 2022; Karim et al., 2022; Li et al., 2022; Chen et al., 2023; Li et al., 2023; Feng et al., 2023; Huang et al., 2023): While abundant in traditional classification, semi-supervised methods are task-specific and scarce in RLVR, where pre-selecting clean samples from noisy data remains an unsolved challenge that prevents direct transfer.

- E Pseudo Code We provide the pseudo code 1.

Algorithm 1 Online Label Refinement (OLR)

Require: Dataset D = {(x, y˜(x))}, policy πθ, rollouts per prompt K, early learning epoch T, slope

threshold δslope

- 1: Initialize rollout history H[x] ← ∅ for all x ∈ D
- 2: for training step t = 1 to MaxSteps do
- 3: for each prompt x ∈ D do
- 4: Generate K rollouts: Yt(x) ∼ πθ(· | x)
- 5: Compute majority answer:

ymajt (x) = argmax

c

|{y ∈ Yt(x) : y = c}|

- 6: Compute pass rate:

pmajt (x) =

1 K|{y ∈ Yt(x) : y = ymajt (x)}|

- 7: Update history: H[x] ← H[x] ∪ {(t, ymajt (x), pmajt (x))}
- 8: if t ≤ T then
- 9: yˆt(x) ← y˜(x)
- 10: continue
- 11: end if
- 12: Compute slope St(x) of pass-rate trajectory via linear regression
- 13: Determine historical majority:

yhistt (x) = argmax

y

|{ymajt′ (x) ∈ H[x] : ymajt′ (x) = y}|

- 14: if St(x) > δslope and ymajt (x) = yhistt (x) then
- 15: yˆt(x) ← ymajt (x)
- 16: else
- 17: yˆt(x) ← y˜(x)
- 18: end if
- 19: end for
- 20: Update policy πθ with GRPO using {(x, yˆt(x))}
- 21: end for

