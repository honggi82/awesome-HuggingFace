# arXiv:2605.29156v1[cs.LG]27May2026

## Rubric-ARROW: Alternating Pointwise Rubric Reward Modeling for LLM Post-training in Non-verifiable Domains

Haoxiang Jiang1 Zihan Dong2 Tianci Liu3 Wanying Wang4 Ran Xu5 Tony Yu6 Linjun Zhang2 Haoyu Wang1

1University at Albany 2Rutgers University 3Purdue University 4Independent Researcher 5Emory University 6Georgia Institute of Technology

Pointwise reward modeling offers critical signals for LLM post-training, yet struggles with absolute scoring in subjective, non-verifiable settings. Rubric-based methods address this by decomposing evaluation into explicit criteria, but existing approaches typically depend on frontier LLMs and suffer from ties caused by hard Boolean aggregation. We present Rubric-ARROW, an alternating framework that jointly trains a rubric generator and a rubric-conditioned judge, with its RL stage using only pairwise preference data. Our method couples a probability-based scoring rule that reduces ties with phase-specific preference-based rewards and an alternating GRPO scheme that together train the pointwise evaluator. Extensive experiments show that Rubric-ARROW achieves competitive reward-modeling accuracy and yields consistent gains for downstream policy post-training.

Keywords: Rubrics-as-Rewards, Reward Modeling, LLM Alignment

Date: 2026-05-29

Model Weights & Checkpoints: https://huggingface.co/collections/OpenRubrics/rubricarrow

Datasets: https://huggingface.co/datasets/OpenRubrics/RubricARROW-Judge-SFT

Contact: hjiang2@albany.edu; hwang28@albany.edu

#### 1. Introduction

Reward modeling is a core component of LLM post-training (Stiennon et al., 2020, Ouyang et al., 2022, Bai et al., 2022). In particular, pointwise reward modeling is attractive, as they assign each response an independent scalar score (Wang et al., 2024), enabling efficient evaluation and direct policy optimization. Yet in non-verifiable tasks such as open-ended instruction following and chat, a single holistic score is often unreliable (Ying et al., 2025, Hashemi et al., 2024, Lee et al., 2025, Song et al., 2024). This has motivated rubric-based reward modeling, which decomposes evaluation into explicit criteria and aggregates criterion-level judgments into a pointwise reward score.

However, rubric-based reward modeling introduces a new challenge: a reliable evaluator requires not only high-quality criteria, but also a judge that can apply them consistently. A common solution is to prompt frontier LLMs to generate rubrics and judgments (Viswanathan et al., 2026, Chen et al., 2025a, Gunjal et al., 2025, Liu et al., 2023). While effective, this paradigm largely outsources evaluation to external models, making reward estimation expensive, difficult to control, and hard to deploy as a standalone reward model.

Recent learning-based approaches take steps toward trainable rubric evaluators, but they remain limited in three important ways. First, several methods train either the rubric generator or the judge, without jointly optimizing the two components altogether (Liu et al., 2025a, Rezaei et al., 2025). Second, many approaches rely on frontier LLMs as the judge to provide supervision for reward modeling (Zhou

- et al., 2025, Li et al., 2026, He et al., 2025), preserving the dependence on external evaluators. Third, the supervision most commonly available for alignment is pairwise preference data, which does not directly specify either absolute pointwise scores or criterion-level labels. This mismatch makes it unclear how to train a rubric-conditioned pointwise reward model from preferences alone. Compounding this issue, hard Boolean aggregation over rubric criteria often collapses distinct responses into tied scores, weakening the discriminative signal needed for both reward modeling and downstream policy optimization (Zhang et al., 2025).

Motivated by challenges above, we introduce Rubric-ARROW1, an alternating training scheme for pointwise rubric reward models from pairwise preferences. Rubric-ARROW treats rubric-based evaluation as a two-module learning problem: a rubric generator proposes criteria, and a rubric-conditioned judge assigns criterion-level scores that are aggregated into a pointwise reward. Crucially, both modules are optimized with reinforcement learning using only pairwise preference data in the RL stage, avoiding reliance on frontier-LLM-labeled judge supervision. To obtain informative rewards, we replace hard Boolean aggregation with a probability-based scoring rule that yields finer-grained criterion scores and reduces ties. We then couple this scoring rule with phase-specific preference rewards and an alternating RL procedure, allowing the rubric generator and judge to iteratively improve each other. The resulting reward model is deployable as a standalone pointwise evaluator and can be used directly for downstream policy optimization with standard algorithms such as DPO (Rafailov et al., 2023) and GRPO (Shao et al., 2024).

In summary, our contributions are as follows:

- • We present Rubric-ARROW, a pointwise rubric reward modeling framework for non-verifiable domains that introduces a probability-based scoring rule and an alternating RL procedure to jointly train a rubric generator and a rubric-conditioned judge, with the RL stage relying on pairwise preference data alone.
- • We provide theoretical analysis establishing the preference consistency of our judge-side reward, variance reduction from opposite-side averaging, and phase-wise convergence guarantees for our designed alternating training pipeline.
- • Extensive experiments on reward modeling benchmarks and downstream policy post-training show that Rubric-ARROW consistently outperforms strong baselines by 3.0%.

#### 2. Related Work

Reward modeling and LLM judges. Reward modeling is a central component of RLHF and LLM post-training (Ouyang et al., 2022). Early methods typically learn scalar reward models from pairwise preferences using Bradley–Terry-style objectives (Liu et al., 2024, Wang et al., 2024). More recent works shifted toward generative and judge-based evaluators, including models that produce rationales or chain-of-thought judgments (Chen et al., 2025b, Guo et al., 2026, Chen et al., 2025a, Mahan et al., 2024, Yu et al., 2025, Hong et al., 2026), and LLM-as-a-judge methods for open-ended evaluation (Zheng

1Short for alternating pointwise rubric reward modeling framework.

- et al., 2023, Li et al., 2024a, 2025, Xu et al., 2025). Such judges can be pointwise, scoring responses independently, or pairwise, comparing responses directly. While pointwise judges are efficient and directly reusable as reward models, their scores are often too coarse to capture fine-grained preference differences in subjective tasks.

Rubric-based reward modeling. Rubric-based reward modeling improves open-ended evaluation by replacing holistic judgments with structured criteria or checklists (Gunjal et al., 2025, Viswanathan

- et al., 2026, Pan et al., 2026). Prior work has explored rubric extraction and synthesis (Liu et al., 2025a, Li et al., 2026), rubric-guided policy learning (Zhou et al., 2025, Rezaei et al., 2025, He et al., 2025, Gunjal et al., 2025), and rubric-conditioned judges (Xu et al., 2026, Jia et al., 2026). Most related are pointwise rubric reward models (Xie et al., 2025, Zhang et al., 2025), which often depend on frontier LLMs for rubric or judge supervision and use hard aggregation over criteria. Rubric-ARROW differs by jointly training the rubric generator and judge from pairwise preferences during alternating RL, while using probability-based criterion scores to obtain more discriminative pointwise rewards.

#### 3. Preliminaries

We consider a pointwise rubric reward model for non-verifiable tasks. Given an instruction-response pair, the evaluator first generates a rubric and then queries a rubric-conditioned judge to produce criterion-level judgments, which are aggregated into a scalar score.

Rubric generator. For an instruction x, the rubric generator πr with parameters θr samples a rubric set r = {rk}kK=1 ∼ πr(· | x; θr), where K is the number of rubric items and rk denotes the k-th rubric item.

Rubric-conditioned judge. Given an instruction x, a candidate response y, and a rubric set r, the judge πj with parameters θj produces criterion-level judgments a = {ak}kK=1 with ak ∈ {0,1} and corresponding rationales e = {ek}kK=1, drawn as (a, e) ∼ πj(· | x, y,r; θj). Here, ak = 1 if y satisfies rk and 0 otherwise.

Pointwise score. The evaluator maps (x, y) to a scalar score by aggregating the criterion-level judgments. We use the weighted average

s(x, y) = ∑︀K

k=1 wkak ∑︀K

, (1)

k=1 wk

where wk > 0 is a predefined prior weight assigned to rubric item rk.

#### 4. Method

We propose a framework that trains a pointwise rubric reward model in two stages: a brief SFT warmup followed by alternating RL on pairwise preference data alone. The RL stage alternates between optimizing the rubric-conditioned judge and the rubric generator, using phase-specific rewards derived from pairwise preference data. The remainder of this section introduces a probability-based scoring rule that yields smoother pointwise scores, the phase-specific RL objectives, the alternating GRPO procedure, and finally how the learned evaluator drives downstream policy training.

###### 4.1. Probability-based scoring

The pointwise score in Eq. (1) aggregates Boolean judgments ak ∈ {0,1}, which can lead to many tied scores in practice. To obtain a fine-grained signal, we follow prior LLM-based evaluators that exploit output distributions rather than only hard decisions (Liu et al., 2023, Hashemi et al., 2024). Specifically, for each rubric item rk, let ptruek and pfalsek denote the judge’s probabilities of the “true” and “false” tokens at the verdict position. We define the criterion-level score as the probability margin dk = ptruek − pfalsek . The probability-based pointwise score is then obtained by replacing ak with dk in Eq. (1). This preserves the original aggregation structure while converting discrete decisions into continuous, more discriminative reward signals.

###### 4.2. Phase-specific RL objectives

Unlike prior RL-based judge training methods that continuously rely on frontier LLMs to synthesize judge labels in the RL loop, our RL stage uses only widely available pairwise preference data. Let D = {(xi, y1i , y2i , oi⋆)}iN=1 be a preference dataset with oi⋆ ∈ {1,2} indicating the preferred response.

Pairwise preference data does not directly supervise a pointwise rubric reward model because the evaluator scores each response independently. In our alternating RL stage, we therefore use different rollout estimators for the judge and the generator phase, while both phases aim to make the pointwise evaluator recover the human preference ordering.

Judge-side preference-consistency reward. Fix the rubric generator πr, and let qi = +1 if oi⋆ = 1 and qi = −1 otherwise. For each response yib, b ∈ {1,2}, the current judge produces n stochastic evaluation rollouts, yielding scores {sib,m}nm=1 and rollout averages s¯ib = n1

∑︀n

m=1 sib,m. We compare each rollout on one side against the opposite side’s average score. For the m-th rollout of y1i ,

R1i,m = I[︁qi

> 0]︁, (2)

(︀

)︀

s1i,m − s¯2i

and for the m-th rollout of y2i ,

R2i,m = I[︁qi

> 0]︁, (3)

(︀

)︀

s¯1i − s2i,m

When oi⋆ = 1, the first reward encourages judge rollouts of y1i to score above s¯2i , while the second encourages judge rollouts of y2i to score below s¯1i ; the direction reverses automatically when oi⋆ = 2. The corresponding judge-side pair reward is

- 1

- 2n

Ri =

and the judge-side population objective is

∑︁n

(︁R1i,m + R2i,m)︁ ∈ [0,1], (4)

m=1

[Ri] . (5)

Jj(θj; θr) = E

(xi,y1i ,y2i ,oi⋆)∼D

This phase converts pairwise preference labels into a pointwise supervision signal for the judge, even though each response is scored independently.

Rubric generator-side preference-ranking reward. Fix the judge πj. For each instruction xi, the rubric generator samples n rubric rollouts {ri,m}nm=1. Each sampled rubric is then used by the frozen judge to evaluate both responses under the same rubric, producing s1i,m and s2i,m. We assign each sampled rubric the reward

Ri(,rm) = I[︁qi

> 0]︁ − λΩ(ri,m), (6)

(︀

)︀

s1i,m − s2i,m

where Ω(ri,m) penalizes unnecessarily long or semantically redundant rubrics; the exact length-aware bonus structure is given in Appendix A.4. The corresponding generator-side population objective is

[︃1 n

Ri(,rm)]︃. (7)

∑︁n

Jr(θr; θj) = E

(xi,y1i ,y2i ,oi⋆)∼D

m=1

The generator is rewarded for producing concise, non-redundant rubrics that enable the frozen judge to rank the preferred response higher. We optimize the judge and generator in alternating update phases until convergence.

###### 4.3. Alternating optimization

SFT warm-up. Before alternating RL, we initialize both modules with a brief supervised stage. We adopt the high-quality rubrics from OpenRubrics (Liu et al., 2025a) as-is and generate judge labels for each (instruction, response) pair with GPT-5-mini. We then filter the warm-up data to keep only pairs whose chosen response receives a higher aggregated pointwise score than the rejected response, and use the resulting data to SFT both the rubric generator πr and the rubric-conditioned judge πj. After this one-time warm-up, the alternating RL stage uses only pairwise preference data and no further frontier-LLM annotation.

Judge phase. We fix πr and optimize πj with GRPO on Jj(θj; θr). For each preference pair, the GRPO group contains the 2n rollout rewards {R1i,m, R2i,m}nm=1. Since πr is frozen, the rubrics for each training instance can be pre-generated once and reused throughout the judge update, which substantially reduces wall-clock cost.

Rubric-generator phase. We then freeze the judge πj and optimize the rubric generator πr with GRPO on Jr(θr; θj). For each preference pair, the GRPO group consists of the n sampled rubrics and their

rewards {Ri(,rm)}nm=1. The reward is assigned to each rubric rollout, as a rubric defines the evaluation criteria for both responses and is scored by whether the frozen judge ranks the preferred response

higher under those criteria.

###### 4.4. Policy Training

After training, the pointwise rubric reward model serves as a standalone evaluator for policy optimization. In the offline setting, we use it to construct DPO training pairs (Rafailov et al., 2023) by scoring candidate responses for each instruction and selecting the highest- and lowest-scoring responses as the chosen and rejected samples, respectively. In the online setting, we use the pointwise score as the reward signal for GRPO (Shao et al., 2024).

#### 5. Theoretical Analysis

We summarize four theoretical properties of our method and defer all proofs to Appendix B. The first two results analyze the judge-side reward Jj. The last two are phase-wise GRPO guarantees: they apply to either a judge phase or a generator phase with the other module held fixed.

###### 5.1. Preference consistency of the judge-side reward

Fix an instruction x and two candidate responses y+ and y− such that the human prefers y+ over y−. Let the rollout-level scalar scores satisfy S+ = µ+ + ξ+ and S− = µ− + ξ−, and define ∆ := µ+ − µ−. Let Jn denote the judge-side preference recovery probability induced by our rollout comparison rule.

- Theorem 5.1 (Preference consistency). Under independent, continuous, zero-mean, symmetric rollout noise, Jn > 21 if and only if ∆ > 0, and Jn = 12 when ∆ = 0.

- Theorem 5.1 states that the judge-side reward is calibrated for recovering the human preference.

- 5.2. Variance reduction from opposite-side averaging

Theorem 5.2 (Variance reduction from opposite-side averaging). Under Gaussian rollout noise and ∆ > 0, comparing each rollout against the opposite response’s rollout average yields a strictly higher preference recovery probability than a single pairwise comparison whenever n > 1. In the homoscedastic case σ+ = σ− = σ,

Jn = Φ(︂

∆ σ√1 + 1/n

)︂, (8)

where Φ is the standard normal CDF. Thus, Jn increases monotonically with n. The exact heteroscedastic expression for σ+ ̸= σ− is given in Appendix B.

Theorem 5.2 shows that opposite-side averaging improves judge reliability by reducing comparison variance.

- 5.3. Optimization guarantees

- Theorem 5.3 (Phase-wise surrogate stationary-point convergence of GRPO). Fix a phase b ∈ {j,r} and hold the other module constant. Under the smoothness, unbiased-gradient, bounded-variance, and diminishing-step-size assumptions in Appendix B,

For the next two results, let b ∈ {j,r} denote a phase of the alternating algorithm: b = j corresponds to judge updates with θr frozen, and b = r corresponds to generator updates with θj frozen. Let Jb be the corresponding population objective (Jj or Jr), and let LG,b be the one-step on-policy GRPO surrogate for that phase. The effective rollout group sizes are Gj = 2n for the judge phase and Gr = n for the generator phase. Here T denotes the number of GRPO updates performed in phase b, θbt is the active-module parameter at phase step t, and θ−b denotes the frozen parameter of the other module.

∑︁T−1

1 T

E⃦∇θbLG,b(θbt; θ−b)⃦2 = O(T−1/2). (9) Hence GRPO attains first-order stationarity for the phase-specific surrogate.

t=0

- Theorem 5.4 (Phase-wise stationary neighborhood for the true objective). Fix a phase b ∈ {j,r} and hold the other module constant. If, in addition, the surrogate-to-true gradient gap for that phase is bounded

- as specified in Appendix B, then

∑︁T−1

1 T

E⃦∇θbJb(θbt; θ−b)⃦2 = O(T−1/2)

t=0

b ) + O(Bstale(b) ) + O(Bclip(b) ).

+ O(G−1

Here Gb is the effective rollout group size in phase b, with Gj = 2n for the judge phase and Gr = n for the generator phase. The O(G−1

b ) term absorbs the finite-group error Bgrp(b) /Gb, while Bstale(b) and Bclip(b) denote stale-policy and clipping bias terms. Hence the rollout-dependent term is O(n−1) in both phases.

#### 6. Experiment

###### 6.1. Datasets and Experiment Settings

Training data. We train Rubric-ARROW on the general-domain pairwise preference data from OpenRubrics (Liu et al., 2025a). We divide the dataset into equally sized, disjoint subsets and assign one subset to each outer iteration of the alternating procedure.

Backbone and variants. Both the rubric generator and the judge are initialized from Qwen-3-8B (Yang

- et al., 2025). At inference time, Rubric-ARROW follows the two-stage pointwise rubric-judging pipeline described in Section 4. We additionally report two variants: a voting@5 ensemble, where five independent rubric-judging runs are aggregated by majority voting; and Rubric-ARROW w/o RL, a pointwise rubric reward model that uses only the SFT warm-up checkpoint without the alternating RL stage, included to isolate the contribution of RL.

Baselines. For reward-model evaluation, we compare Rubric-ARROW with generative judges (RMR1 (Chen et al., 2025b), RRM (Guo et al., 2026), JudgeLRM (Chen et al., 2025a)), prompted Qwen-3-8B rubric+judge pipelines, and trained rubric-based baselines (Rubric-RM (Liu et al., 2025a), RubricARROW w/o RL, RIFL (He et al., 2025)), with black-box APIs included for reference. For downstream policy training on Qwen2.5-7B-Instruct (Yang et al., 2024), we compare against scalar reward models (Skywork (Liu et al., 2024), ArmoRM (Wang et al., 2024)), AI feedback data (UltraFeedback (Cui et al.,

- 2023)), and rubric/checklist baselines (RLCF (Viswanathan et al., 2026), Rubric-RM, Rubric-ARROW w/o RL, RIFL). Full baseline descriptions are given in Appendix A.3.

Evaluation benchmarks and metrics. We evaluate Rubric-ARROW in two settings. For reward-model evaluation, we use RewardBench (Chat/Chat-Hard) (Lambert et al., 2025), FollowBench (Jiang et al.,

- 2024), PPE-IFEval (Frick et al., 2025), InfoBench (Qin et al., 2024), IFBench (Peng et al., 2025), RMBench (Liu et al., 2025b), RewardBench2 (Precise-IF/Focus) (Malik et al., 2025), and HelpSteer3 (Wang

- et al., 2026). For downstream policy evaluation, we use IFEval (Frick et al., 2025), InfoBench (Qin

- et al., 2024), IFBench (Peng et al., 2025), Arena-Hard (Li et al., 2024b), AlpacaEval2 (Dubois et al.,

- 2024), Creative Writing Benchmark v3 (Paech, 2025), and WildBench (Lin et al., 2025). Evaluation construction and metric details are deferred to Appendix A.2.

Rubric-RM

7.50

RM-R1

Skywork

7.25

Rubric-ARROW

Meanoraclescore

Oracle

7.00

6.75

6.50

6.25

6.00

5.75

1 4 8 16 32 64 128

Best-of-N

- Figure 1: Best-of-N selection quality on WildBench: each method selects one response from the first N sampled candidates, and we report the mean oracle score. The Oracle curve marks the per-prompt best candidate (upper bound); shaded regions show 95% bootstrap confidence intervals over prompts. Failure to track the Oracle curve

- at larger N indicates reward-model overoptimization. Table 1: Comparison of judge and reward models across benchmarks. RewardBench2 reports its Precise IF and Focus dimensions; the API pipelines use GPT-4.1-Mini as the rubric generator and Gemini-2.5-Flash-Lite as the judge. Best results in bold.

RewardBench IF Evaluation Benchmarks RM-Bench RewardBench2

HelpSteer3 Avg. Chat Chat Hard FollowBench PPE-IFEval InfoBench IFBench Chat Precise IF Focus

Black-box LLMs (For reference only)

Claude-3.5-Sonnet 96.4 74.0 – 58.0 – – 62.5 38.8 87.0 – Gemini-2.5-Flash 95.0 83.3 86.0 75.0 85.6 69.3 78.5 57.5 84.1 70.6 78.5 API Prompting (Rubric+Judge pairwise) 79.6 79.2 83.2 61.0 82.2 66.2 67.9 42.5 79.6 71.4 71.3 API Prompting (Rubric+Judge pointwise) 51.7 61.5 60.5 26.4 44.8 33.6 37.8 14.8 50.4 39.2 42.1 API Prompting (direct Judge) 89.6 71.2 81.7 59.2 72.9 60.4 67.2 13.2 63.4 70.3 64.9

Larger White-box LLMs (For reference only)

RM-R1-14B (Qwen-2.5-Inst) 73.5 79.8 84.0 59.0 85.5 60.8 73.2 23.8 84.6 74.8 69.9 RM-R1-14B (DeepSeek-Dist) 90.3 78.9 89.9 61.2 82.4 59.0 71.4 30.6 79.0 74.6 71.7 RM-R1-32B (Qwen-2.5-Inst) 95.3 80.3 84.9 60.4 86.1 60.4 75.3 33.1 84.2 72.9 73.3 RM-R1-32B (DeepSeek-Dist) 95.3 83.1 89.2 63.2 85.0 58.6 74.2 36.9 79.2 75.6 74.0 RRM-32B 94.7 81.1 85.7 60.2 84.4 60.8 73.9 34.4 83.6 75.4 73.4

White-box Judge/Reward LLMs

RM-R1-7B (Qwen-2.5-Inst) 83.0 70.0 56.3 55.2 71.3 55.2 64.2 20.6 76.2 65.2 61.7 RM-R1-7B (DeepSeek-Dist) 85.3 67.3 69.7 51.0 70.3 56.5 62.2 13.8 55.4 62.6 59.4 RRM-7B 77.7 69.5 65.5 51.0 68.2 53.2 59.9 10.0 60.4 62.4 57.8 JudgeLRM-7B 92.1 56.1 79.8 46.0 62.7 47.5 55.4 9.4 29.1 60.2 53.8

Rubric-based Methods

Qwen-3-8B (Rubric+Judge pairwise) 73.9 63.6 63.0 53.8 74.6 55.6 64.2 21.9 56.6 61.8 58.9 Qwen-3-8B (Rubric+Judge pointwise) 69.3 66.4 57.5 52.6 69.9 50.2 61.2 27.0 62.9 55.5 57.3 Rubric-RM 88.2 74.1 76.1 67.0 80.8 65.4 65.7 34.4 82.2 67.0 70.1 Rubric-RM-voting@5 89.9 75.4 81.5 70.8 83.8 67.1 67.0 40.0 86.5 67.5 73.0 Rubric-ARROW w/o RL 87.4 71.0 77.8 69.5 75.9 65.5 64.8 40.3 73.3 67.5 69.3 Rubric-ARROW w/o RL-voting@5 87.7 73.3 84.0 72.2 79.7 68.5 66.9 43.1 77.0 67.5 72.0 RIFL 89.6 73.3 81.2 73.3 78.4 69.2 64.8 42.3 76.7 67.7 71.7 RIFL-voting@5 89.7 74.1 84.0 75.8 80.1 69.8 64.3 41.9 77.2 67.7 72.5 Rubric-ARROW 89.1 79.2 78.2 72.6 82.1 68.2 67.8 42.9 80.8 70.2 73.1 Rubric-ARROW-voting@5 90.8 81.8 79.8 76.0 84.0 73.2 68.6 45.0 84.9 72.0 75.6

###### 6.2. Performance of Rubric-ARROW

- Table 1 shows that Rubric-ARROW provides a strong white-box pointwise reward model for openended evaluation. Among white-box methods, Rubric-ARROW achieves the best average performance, improving over the strongest rubric baseline Rubric-RM from 70.1 to 73.1, and further to 75.6 with voting@5. The gains are most pronounced on instruction-following and chat-oriented benchmarks, where

- Table 2: Ablation on the length penalty. Best results in bold.

RewardBench IF Evaluation Benchmarks RM-Bench RewardBench2

HelpSteer3 Avg. Chat Chat Hard FollowBench PPE-IFEval InfoBench IFBench Chat Precise IF Focus

Rubric-ARROW w/o length penalty 87.5 75.6 79.7 69.6 81.5 69.0 66.1 43.9 80.6 69.3 72.3 Rubric-ARROW w/o length penalty-voting@5 90.8 78.5 81.5 73.2 86.7 71.6 68.7 38.1 82.6 71.2 74.3 Rubric-ARROW 89.1 79.2 78.2 72.6 82.1 68.2 67.8 42.9 80.8 70.2 73.1 Rubric-ARROW-voting@5 90.8 81.8 79.8 76.0 84.0 73.2 68.6 45.0 84.9 72.0 75.6

- Table 3: Ablation study on probability-based scoring on InFoBench (Qin et al., 2024) and IFBench (Peng et al.,

- 2025) Dataset Method Tie → Correct Correct Preserved Wrong → Correct

Probability 98 / 140 305 / 312 5 / 36 Random 63 / 140 160 / 312 12 / 36

InfoBench

Probability 68 / 115 234 / 255 13 / 74 Random 67 / 115 130 / 255 38 / 74

IFBench

Rubric-ARROW-voting@5 achieves the best results on Chat Hard, PPE-IFEval, InfoBench, and IFBench. The improvement over the no-RL variant further confirms the benefit of our alternating RL procedure, indicating that jointly optimizing the rubric generator and judge yields more discriminative reward signals. Notably, Rubric-ARROW-voting@5 also outperforms prompting-based judges, suggesting that learned rubric-conditioned evaluation is more reliable than direct prompting.

We next study robustness under best-of-N selection on 100 WildBench prompts from instruction-following and chat categories, using frontier-LLM scores under the dataset rubric as the oracle reference. As shown in Fig. 1, Rubric-RM and Skywork begin to degrade after N = 8, suggesting sensitivity to reward overoptimization. In contrast, Rubric-ARROW continues to closely track the oracle upper bound, indicating that its rewards remain well aligned when used for offline selection.

###### 6.3. Ablation Study

We perform two ablation studies to better understand where the improvements of Rubric-ARROW come from. First, we remove the length penalty to examine whether controlling rubric verbosity is important for reliable rubric-conditioned judging. Second, we evaluate probability-based scoring against a random-probability control to verify whether the model-derived confidence scores provide meaningful tie-breaking signals.

Length penalty. Removing the length penalty consistently hurts performance, reducing the average score from 73.1 to 72.3 and voting@5 from 75.6 to 74.3 (Table 2). The drop is most visible on stricter instruction-following benchmarks such as Chat Hard and PPE-IFEval, suggesting that length regularization helps prevent overly verbose or redundant rubrics and improves reward accuracy.

Probability-based scoring. Table 3 shows that probability-based scoring breaks ties without sacrificing correctness. On InfoBench and IFBench, it resolves 98/140 and 68/115 tied cases while preserving most originally correct predictions (305/312 and 234/255). In contrast, a random-probability control resolves a similar number of ties but corrupts many correct decisions, preserving only 160/312 and 130/255. Thus, the improvement comes from meaningful judge confidence rather than arbitrary tie-breaking.

- Table 4: Trained-policy comparison on IFEval and InfoBench. Asterisks (⋆) denote results taken from Viswanathan et al. (2026), Liu et al. (2025a); underlined results were reproduced by us. Best in bold.

IFEval (Prompt) IFEval (Inst.) IFEval InfoBench Loose Strict Loose Strict AVG AVG

Model

GPT-4 (0314)⋆ 79.3 76.9 85.4 83.6 81.3 87.3 AutoIF (Dong et al., 2025) 56.9 47.1 67.0 57.6 57.2 80.6 UltraIF (An et al., 2025) 75.4 71.3 83.0 79.4 77.3 80.7 RAIF (Qin et al., 2026) – – – – 70.1 82.7

Qwen2.5-7B-Instruct⋆ 75.0 72.5 81.8 79.9 77.3 78.1 (76.0)

+ SFT (Distilled)⋆ 66.8 64.1 75.3 72.8 69.8 72.5 + DPO (via Skywork)⋆ 75.7 68.0 83.2 78.5 76.0 82.0 + DPO (via ArmoRM)⋆ 73.8 70.2 81.7 78.3 76.0 83.5 + DPO (via Ultrafbk.)⋆ 71.5 69.1 79.9 77.7 74.6 80.0 + DPO (via AI Judge)⋆ 73.0 68.9 80.9 77.8 75.2 76.1 + DPO (via RLCF)⋆ 77.3 72.6 84.1 80.3 78.6 84.1 (81.5)

+ IterDPO (via RLCF) 78.2 74.3 84.5 81.1 79.5 81.8 + DPO (via Rubric-RM)⋆ 78.2 73.9 84.5 81.2 79.5 83.0 + IterDPO (via Rubric-RM) 77.6 74.1 84.3 81.7 79.4 83.3 +DPO (via Rubric-ARROWw/o RL) 76.0 72.8 82.9 80.3 78.0 82.3 +IterDPO (via Rubric-ARROWw/o RL) 76.7 74.1 83.2 81.2 79.4 82.4 +DPO (via RIFL) 77.5 74.1 84.3 81.8 79.4 82.8 +IterDPO (via RIFL) 78.0 74.9 84.2 81.5 79.7 84.1

+ DPO (via Rubric-ARROW) 79.7 75.6 85.3 82.1 80.7 83.6 + IterDPO (via Rubric-ARROW) 79.9 74.5 85.4 81.4 80.3 83.7

- 6.4. Performance of offline RL We investigate whether the gains of Rubric-ARROW generalize to downstream offline policy learning.

Instruction-Following Evaluation. Table 4 and Fig. 2 show that policies trained with Rubric-ARROWderived rewards yield consistent instruction-following gains. On IFEval, DPO via Rubric-ARROW lifts the average score to 80.7, outperforming both Rubric-RM and the Rubric-ARROW variant without RL, while remaining competitive on InfoBench (83.6 with DPO, 83.7 with IterDPO). The advantage is more pronounced on IFBench (Fig. 2): Rubric-ARROW reaches 37.4 with DPO and 36.7 with IterDPO, outperforming Rubric-RM under both settings.

Human preference alignment evaluation. Rubric-ARROW-trained rewards also transfer to preference-

alignment benchmarks (Table 5, Table 6). On Arena-Hard and AlpacaEval, DPO via Rubric-ARROW reaches an average of 50.7 and IterDPO further improves it to 53.0, achieving the best results on Arena-Hard (both vanilla and style-controlled) and on AlpacaEval length-controlled. On WildBench, IterDPO via Rubric-ARROW reaches the best macro score of 55.2 (53.6 with DPO), surpassing IterDPO via Rubric-RM by 2.2% relative.

Creative writing evaluation. On the Creative Writing Benchmark v3 (Fig. 3), policies trained with Rubric-ARROW lead all compared methods: DPO via Rubric-ARROW reaches 39.8 and IterDPO further improves the score to 40.5. Rubric-ARROW-based optimization also outperforms creativewriting baselines such as RaR, Rubric-based RL, and RuscaRL, as well as Rubric-RM-based DPO and IterDPO. This indicates that Rubric-ARROW-derived rewards generalize beyond instruction-following and preference benchmarks to open-ended generation.

- Table 5: Trained-policy comparison on Arena-Hard and AlpacaEval under vanilla and style/length-controlled settings. Asterisks (⋆) denote results from Viswanathan et al. (2026), Rezaei et al. (2025), Liu et al. (2025a). Best in bold.

Arena-Hard AlpacaEval

Model

AVG Vanilla Style-Con Vanilla Length-Con

GPT-4 (0314)⋆ 50.0 50.0 22.1 35.3 39.4 UltraIF (An et al., 2025) 31.4 – – – –

Qwen2.5-7B-Instruct⋆ 51.3 42.8 33.5 36.2 41.0 + SFT (Distilled)⋆ 32.6 29.2 36.1 33.3 32.8 + DPO (via Skywork)⋆ 55.1 50.3 44.8 41.5 47.9 + DPO (via ArmoRM)⋆ 50.8 46.4 37.6 38.1 43.2 + DPO (via Ultrafbk.)⋆ 52.8 47.9 33.7 38.7 43.3 + DPO (via AI Judge)⋆ 51.0 44.4 28.8 33.4 39.4 + DPO (via RLCF)⋆ 54.6 48.4 36.2 37.1 44.1 + IterDPO (via RLCF) 51.1 54.6 38.9 39.2 46.0 + DPO (via Rubric-RM)⋆ 52.9 53.1 47.0 41.3 48.6 + IterDPO (via Rubric-RM) 56.3 56.7 50.1 42.0 51.3 + RL (via OnlineRubrics)⋆ 56.5 – 55.0 30.4 –

+ DPO (via Rubric-ARROWw/o RL) 52.6 54.3 40.7 42.1 47.4 + IterDPO (via Rubric-ARROWw/o RL) 55.8 56.9 42.7 42.1 49.4 + DPO (via RIFL) 52.4 53.4 43.2 41.1 47.5 + IterDPO (via RIFL) 55.2 55.7 45.6 42.4 49.7

+ DPO (via Rubric-ARROW) 53.5 54.3 49.1 45.7 50.7 + IterDPO (via Rubric-ARROW) 57.5 57.7 51.4 45.5 53.0

40

37.4

37.4 36.7

35.7

33.7

33.7

35

32.0

30

28.2

28.2

Score

25

22.8 22.4

20

15

10

Qwen2.5 RLMT (DPO)

RLMT (PPO)

RLCF (DPO)

Rubric-RM (DPO)

RIFL (DPO)

Rubric-ARROW (DPO)

RLCF (IterDPO)

Rubric-RM (IterDPO)

RIFL (IterDPO)

Rubric-ARROW (IterDPO)

- Figure 2: Trained-policy comparison on IFBench. Baseline results (except Rubric-RM IterDPO) are from OpenRubrics Liu et al. (2025a); two baselines are omitted for readability (see Fig. 4 in Appendix A for the full version).

Qwen2.5 RaR Rubric-based RL-S

RuscaRL Rubric-RM (DPO)

RIFL (DPO)

Rubric-ARROW (DPO)

Rubric-RM (IterDPO)

RIFL (IterDPO)

Rubric-ARROW (IterDPO)

- 35

- 36

- 37

- 38

- 39

- 40

- 41

Score

37.4

38.8

38.3

38.6

37.5

38.5

39.8

38.8 38.6

40.5

- Figure 3: Trained-policy comparison on Creative Writing Benchmark v3. Baseline results (except Rubric-RM) are from RuscaRL Zhou et al. (2025); two baselines are omitted for readability (see Fig. 5 in Appendix A for the full version).

- Table 6: Trained-policy comparison on WildBench, with task-specific and macro WB scores. Asterisks (⋆) denote results from Wang et al. (2025), Xu et al. (2026). Best in bold.

Method Creative Planning Math Info seeking Coding WB Score

Claude-3.5-Sonnet (20240620)⋆ 55.6 55.6 50.2 55.5 56.5 54.7 GPT-4-turbo (20240409)⋆ 58.7 56.2 51.0 57.2 55.1 55.2 GPT-4o-mini (20240718)⋆ 60.1 58.2 54.0 57.4 57.2 57.1

Qwen2.5-7B-Instruct⋆ 50.1 51.8 47.1 50.7 45.0 48.7 + DRIFT⋆ 52.5 53.2 50.6 52.4 50.3 51.7 + SPIN⋆ 43.3 45.5 41.6 46.3 39.1 42.9 + IterDPO⋆ (via OpenAssistant) 46.8 48.6 44.5 48.0 44.3 46.3 + DPO⋆ (via RLCF) 51.4 52.7 49.0 51.3 48.8 50.5 + IterDPO⋆ (via RLCF) 51.9 52.6 47.8 51.4 46.5 49.7 + DPO (via Rubric-RM) 54.8 55.5 51.5 54.1 52.9 53.6 + IterDPO (via Rubric-RM) 57.0 56.2 50.6 54.9 52.8 54.0 + DPO (via Rubric-ARROWw/o RL) 53.3 53.9 46.8 54.4 51.6 51.8 + IterDPO (via Rubric-ARROWw/o RL) 54.5 55.7 51.2 54.6 52.8 53.2 + DPO (via RIFL) 55.0 55.2 50.7 54.0 51.0 52.6 + IterDPO (via RIFL) 55.1 57.0 52.8 55.1 52.5 54.0

+ DPO (via Rubric-ARROW) 54.9 55.8 52.2 54.8 51.4 53.6 + IterDPO (via Rubric-ARROW) 56.4 57.9 52.6 56.6 53.3 55.2

- Table 7: Comparison of online RL method with different alignment strategies applied to Qwen2.5-7B-Instruct on instruction following and preference alignment benchmarks. Best results are in bold.

IFEval (Prompt) IFEval (Inst.) AlpacaEval

Method

AVG Loose Strict Loose Strict Vanilla Length

Qwen2.5-7B-Instruct 75.0 72.5 81.8 79.9 33.5 36.2 56.1 +GRPO (RM-R1) 76.7 73.6 83.2 80.2 53.2 42.7 63.2 +GRPO (Rubric-ARROW w/o RL) 76.9 74.5 83.5 81.2 51.4 44.9 63.6 +GRPO (RIFL) 76.7 73.8 83.1 80.6 47.8 42.4 61.9 +GRPO (Skywork) 76.0 72.3 83.1 80.3 50.5 45.6 63.0 +GRPO (Rubric-ARROW) 79.3 76.2 84.9 82.5 50.9 49.3 65.4

###### 6.5. Performance of online RL

We evaluate Rubric-ARROW in an online RL setting by directly optimizing Qwen2.5-7B-Instruct with GRPO using different reward models. As shown in Table 7, GRPO via Rubric-ARROW delivers the best overall average, improving from 56.1 for the base model to 65.4. It also outperforms reward baselines such as RM-R1, RIFL, and Skywork. Compared with the Rubric-ARROW variant without RL, Rubric-ARROW further improves the average score from 63.6 to 65.4. These results indicate that Rubric-ARROW provides a stronger and more reliable reward signal for online policy optimization.

###### 6.6. Efficiency Comparison

We evaluate inference efficiency on 100 samples with vLLM (Table 8): Rubric-ARROW-8B processes them in 28.35s, an order of magnitude faster than reasoning-based reward models (RRM-7B and RM-R1-7B/14B, all >170s) and substantially faster than Rubric-RM-8B (105.12s). Although JudgeLRM7B is slightly faster (25.71s), it produces only a direct judgment without rubric-conditioned signals. Rubric-ARROW thus preserves interpretability at a fraction of the reasoning-based inference cost.

- Table 8: Computing speed on 100 samples (vLLM). Results with “⋆” were taken from (Xu et al., 2026). Method Compute Time (s)

JudgeLRM-7B⋆ 25.71 RRM-7B⋆ 203.40 RM-R1-7B (Qwen-2.5-Inst)⋆ 260.37 RM-R1-7B (DeepSeek-Dist)⋆ 170.76 RM-R1-14B (Qwen-2.5-Inst)⋆ 322.79 RM-R1-14B (DeepSeek-Dist)⋆ 382.02 Rubric-RM-8B⋆ 105.12

Rubric-ARROW-8B 28.35

- 6.7. Case Study

We qualitatively analyze failures of a baseline reward model on a challenging example. Table 12 in Appendix A presents a comparison instance about the difference between birding and bird watching. RIFL produces several surface-level rubrics (e.g., conciseness, clear organization, neutral tone, and whether a direct comparison is made), and its judge marks the inaccurate Resp B as True on most of them; the response wins on aggregate even though the judge separately flags its factual inaccuracy. However, Rubric-ARROW produces a focused set of hard-rule rubrics centered on accurately addressing the question, and the judge correctly marks Resp A as True and Resp B as False on the central directcomparison rubric, yielding the right preference.

#### 7. Conclusion

We propose Rubric-ARROW, an alternating framework that trains a pointwise rubric reward model for non-verifiable domains. After a brief SFT warm-up, its alternating RL stage couples a probability-based scoring rule that reduces ties with phase-specific preference-based rewards and alternating GRPO to jointly train the rubric generator and the rubric-conditioned judge using only pairwise preference data, with no further frontier-LLM annotation. Theoretical analysis establishes preference consistency, variance reduction from opposite-side averaging, and phase-wise convergence guarantees for alternating GRPO. Empirically, Rubric-ARROW attains the best average across reward-modeling benchmarks and yields the best downstream policies under both offline (DPO, IterDPO) and online (GRPO) training.

#### References

Kaikai An, Li Sheng, Ganqu Cui, Shuzheng Si, Ning Ding, Yu Cheng, and Baobao Chang. Ultraif: Advancing instruction following from the wild. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 18722–18737, 2025.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng He. Judgelrm: Large reasoning models as a judge. arXiv preprint arXiv:2504.00050, 2025a.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, et al. Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387, 2025b.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback. 2023.

Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren Zhou. Self-play with execution feedback: Improving instruction-following capabilities of large language models. In International Conference on Learning Representations, volume 2025, pages 39286–39313, 2025.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. How to evaluate reward models for rlhf. In International Conference on Learning Representations, volume 2025, pages 18128–18163, 2025.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.

Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Reward reasoning models. Advances in Neural Information Processing Systems, 38:150477–150510, 2026.

Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, and Chris Kedzie. Llm-rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13806–13834, 2024.

Yun He, Wenzhe Li, Hejia Zhang, Songlin Li, Karishma Mandyam, Sopan Khosla, Yuanhao Xiong, Nanshu Wang, Xiaoliang Peng, Beibin Li, et al. Advancedif: Rubric-based benchmarking and reinforcement learning for advancing llm instruction following. arXiv preprint arXiv:2511.10507, 2025.

Ilgee Hong, Changlong Yu, Liang Qiu, Weixiang Yan, Zhenghao Xu, Haoming Jiang, Qingru Zhang, Qin Lu, Xin Liu, Chao Zhang, et al. Think-rm: Enabling long-horizon reasoning in generative reward models. Advances in Neural Information Processing Systems, 38:174227–174251, 2026.

Ruipeng Jia, Yunyi Yang, Yuxin Wu, Yongbo Gai, Siyuan Tao, Mengyu Zhou, Jianhe Lin, Xiaoxi Jiang, and Guanjun Jiang. Open rubric system: Scaling reinforcement learning with pairwise adaptive rubric. arXiv preprint arXiv:2602.14069, 2026.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. Followbench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, 2024.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1755–1797, 2025.

Yukyung Lee, Joonghoon Kim, Jaehee Kim, Hyowon Cho, Jaewook Kang, Pilsung Kang, and Najoung Kim. Checkeval: A reliable llm-as-a-judge framework for evaluating text generation using checklists. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 15782–15809, 2025.

Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al. From generation to judgment: Opportunities and challenges of llm-as-a-judge. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 2757–2791, 2025.

Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. Llms-asjudges: a comprehensive survey on llm-based evaluation methods. arXiv preprint arXiv:2412.05579, 2024a.

Sunzhu Li, Jiale Zhao, Miteto Wei, Huimin Ren, Yang Zhou, Jingwen Yang, Shunyu Liu, Kaike Zhang, and Wei Chen. Rubrichub: A comprehensive and highly discriminative rubric dataset via automated coarse-to-fine generation. arXiv preprint arXiv:2601.08430, 2026.

Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939, 2024b.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. Wildbench: Benchmarking llms with challenging tasks from real users in the wild. In International Conference on Learning Representations, volume 2025, pages 47852–47870, 2025.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. Openrubrics: Towards scalable synthetic rubric generation for reward modeling and llm alignment. arXiv preprint arXiv:2510.07743, 2025a.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 2511–2522, 2023.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. Rm-bench: Benchmarking reward models of language models with subtlety and style. In International Conference on Learning Representations, volume 2025, pages 44323–44355, 2025b.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. Generative reward models. arXiv preprint arXiv:2410.12832, 2024.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A Smith, Hannaneh Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Samuel J Paech. Eq-bench creative writing benchmark v3, 2025. Tianjun Pan, Xuan Lin, Wenyan Yang, Qianyu He, Shisong Chen, Licai Qi, Wanqing Xu, Hongwei Feng,

Bo Xu, and Yanghua Xiao. Rubriceval: A rubric-level meta-evaluation benchmark for llm judges in instruction following. arXiv preprint arXiv:2603.25133, 2026.

Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Bin Xu, Lei Hou, and Juanzi Li. Agentic reward modeling: Integrating human preferences with verifiable correctness signals for reliable reward systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15934–15949, 2025.

Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, and Dong Yu. Infobench: Evaluating instruction following ability in large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13025–13048, 2024.

Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, and Xing Sun. Incentivizing reasoning for advanced instruction-following of large language models. Advances in Neural Information Processing Systems, 38:108337–108401, 2026.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

MohammadHossein Rezaei, Robert Vacareanu, Zihao Wang, Clinton Wang, Bing Liu, Yunzhong He, and Afra Feyza Akyürek. Online rubrics elicitation from pairwise comparisons. arXiv preprint arXiv:2510.07284, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Hwanjun Song, Hang Su, Igor Shalyminov, Jason Cai, and Saab Mansour. Finesure: Fine-grained summarization evaluation using llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 906–922, 2024.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.

Vijay Viswanathan, Yanchao Sun, Xiang Kong, Meng Cao, Graham Neubig, and Sherry Wu. Checklists are better than reward models for aligning language models. Advances in Neural Information Processing Systems, 38:114728–114754, 2026.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10582–10592, 2024.

Yifan Wang, Bolian Li, Junlin Wu, Zhaoxuan Tan, Zheli Liu, Ruqi Zhang, Ananth Grama, and Qingkai Zeng. Drift: Learning from abundant user dissatisfaction in real-world preference learning. arXiv preprint arXiv:2510.02341, 2025.

Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, and Oleksii Kuchaiev. Helpsteer3-preference: Open human-annotated preference data across diverse tasks and languages. Advances in Neural Information Processing Systems, 38, 2026.

Lipeng Xie, Sen Huang, Zhuo Zhang, Anni Zou, Yunpeng Zhai, Dingchao Ren, Kezun Zhang, Haoyuan Hu, Boyin Liu, Haoran Chen, et al. Auto-rubric: Learning from implicit weights to explicit rubrics for reward modeling. arXiv preprint arXiv:2510.17314, 2025.

Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, and Hongkun Yu. Incentivizing agentic reasoning in llm judges via tool-integrated reinforcement learning. arXiv preprint arXiv:2510.23038, 2025.

Ran Xu, Tianci Liu, Zihan Dong, Tony Yu, Ilgee Hong, Carl Yang, Linjun Zhang, Tao Zhao, and Haoyu Wang. Alternating reinforcement learning for rubric-based reward modeling in non-verifiable llm post-training. arXiv preprint arXiv:2602.01511, 2026.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shuangshuang Ying, Yunwen Li, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Xeron Du, Tianyu Zheng, Yichi Zhang, et al. Beyond correctness: Evaluating subjective writing preferences across cultures. arXiv preprint arXiv:2510.14616, 2025.

Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang, Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, Melanie Kambadur, Dhruv Mahajan, and Rui Hou. Self-generated critiques boost reward modeling for language models. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas

Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11499–11514, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.573. URL https:// aclanthology.org/2025.naacl-long.573/.

Junkai Zhang, Zihao Wang, Lin Gui, Swarnashree Mysore Sathyendra, Jaehwan Jeong, Victor Veitch, Wei Wang, Yunzhong He, Bing Liu, and Lifeng Jin. Chasing the tail: Effective rubric-based reward modeling for large language model post-training. arXiv preprint arXiv:2509.21500, 2025.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, et al. Swift: a scalable lightweight infrastructure for fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 29733–29735, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 3: system demonstrations), pages 400–410, 2024.

Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Kongcheng Zhang, Jiale Zhao, Jingwen Yang, Yihe Zhou, Jianwei Lv, Tongya Zheng, et al. Breaking the exploration bottleneck: Rubric-scaffolded reinforcement learning for general llm reasoning. arXiv preprint arXiv:2508.16949, 2025.

#### A. Experiment

###### A.1. Implementation Details

Tables 9 and 10 summarize the hyperparameters used for Rubric-ARROW and policy model training, respectively. In rubric-based scoring, we assign a weight of 3 to each Hard Rule and a weight of 1 to each Principle when computing the final score. We train the GRPO models using the ms-swift library2 (Zhao et al., 2025), and conduct DPO and IterDPO training with LLaMA-Factory3 (Zheng et al., 2024). In total, Rubric-ARROW is trained for three alternating RL iterations. The inference-time sampling parameters are reported in Table 11. For baseline methods, we use the sampling configurations specified in their official implementations or original papers.

###### A.2. Evaluation Details

Pairwise construction for FollowBench and InfoBench. For these two benchmarks, we convert the original single-response evaluation into a pairwise comparison setting: for each prompt, two responses are sampled from a fixed source model (Qwen-3-8B or Qwen-3-14B), and constraint violations are identified using each benchmark’s official verifier.

- 2https://github.com/modelscope/ms-swift
- 3https://github.com/hiyouga/LLaMA-Factory

- Table 9: Hyper-parameters used in Rubric-ARROW training. Module Parameter Value Module Parameter Value

#generations 6

#generations 6

Cutoff Length 512 Cutoff Length 1024 Batch Size 64 Batch Size 32

Optimizer AdamW Optimizer AdamW

Learning Rate 1e-6 Learning Rate 1e-6 Temperature 1.0 Temperature 1.0 #iterations 2 #iterations 2

Rubric Generator

Judge

Epochs 1 Epochs 1

ϵhigh 0.28 ϵhigh 0.28

ϵlow 0.2 ϵlow 0.2

β 0.001 β 0.001

Splits and metrics. We follow each benchmark’s official splits and scoring rules and report accuracy, win rate, or the benchmark-specific metric.

###### A.3. Baselines

Reward-model evaluation baselines. We compare Rubric-ARROW with state-of-the-art white-box generative judge and reward models, including RM-R1 (Chen et al., 2025b), RRM (Guo et al., 2026), and JudgeLRM (Chen et al., 2025a). We also include black-box references for context: Claude-3.5-Sonnet4, Gemini-2.5-Flash Comanici et al. (2025), and API-based rubric+judge pipelines (pairwise, pointwise, and direct-judge variants, with GPT-4.1-Mini5 as the rubric generator and Gemini-2.5-Flash-Lite as the judge). To isolate the effect of learned rubric-conditioned pointwise scoring, we further compare with prompted Qwen-3-8B rubric+judge pipelines (pairwise and pointwise) and with trained rubric-based baselines Rubric-RM (Liu et al., 2025a), Rubric-ARROW w/o RL, and RIFL (He et al., 2025).

Downstream policy training baselines. For downstream policy training, we provide training signals for Qwen2.5-7B-Instruct (Yang et al., 2024) using different reward models. Skywork (Liu et al., 2024) and ArmoRM (Wang et al., 2024) are scalar Bradley–Terry reward models trained on pairwise human preferences; for Skywork we use the Skywork/Skywork-Reward-V2-Llama-3.1-8B checkpoint. UltraFeedback (Cui et al., 2023) provides multi-aspect AI feedback annotations used as a preference dataset. RLCF (Viswanathan et al., 2026) is a checklist-based reward signal that converts explicit constraints into a scalar reward. We also compare against rubric-based reward models Rubric-RM, Rubric-ARROW w/o RL, and RIFL.

- 4https://www.anthropic.com/news/claude-3-5-sonnet
- 5https://openai.com/index/gpt-4-1/

### Table 10: Hyper-parameters used in policy model training. Method Parameter Value Method Parameter Value

Cutoff Length 2048

#generations 6

Batch Size 64 Cutoff Length 2048

Optimizer AdamW Batch Size 64 Learning Rate 8e-7 Optimizer AdamW

Epochs 1 Learning Rate 5e-7 beta 0.1 Temperature 1.0

DPO

GRPO

SFT mixing weight 0.3 #iterations 2 / / Epochs 1

/ / ϵhigh 0.28

/ / ϵlow 0.2

/ / β 0.001

##### Table 11: Sampling parameters used in Rubric-ARROW inference. Module Parameter Value Module Parameter Value

Maximum Tokens 512

Maximum Tokens 1024

Temperature 0.0 Temperature 1.0

Rubric Generator

Top-P / Top-P 0.95

Judge

Top-K / Top-K -1

Enable-thinking False Enable-thinking False

- A.4. Length-Aware Rubric Reward

The rubric-generator reward Ri(,rm) in Eq. (6) combines preference consistency with a length-aware bonus. Concretely, let

Ci = {m ∈ {1, . . . , n} : qi(s1i,m − s2i,m) > 0} (10) denote the set of preference-consistent rubric rollouts for pair i, and let

Ki,m = |ri,m| (11) denote the number of rubric items in rollout m. When Ci ̸= ∅, we further define

∑︁

1

Ki,m. (12)

Ki,m, K¯i =

Kimin = min m∈Ci

|Ci|

m∈Ci

The rubric-generator reward used in practice is

⎧ ⎪⎨

1.1, if m ∈ Ci, Ki,m = Kimin, K¯i ≥ 5,

Ri(,rm) =

(13)

+1, if m ∈ Ci, −1, otherwise.

⎪⎩

If Ci = ∅, all rubric rollouts receive reward −1. This length-aware bonus rewards the shortest rubric set among those that correctly recover the human preference, while the condition K¯i ≥ 5 prevents the generator from being rewarded for producing overly short rubrics.

###### A.5. Full Version of Policy Model Performance on IFBench and Creative Writing Benchmark v3

Figs. 4 and 5 report the full versions of the IFBench and Creative Writing v3 policy-model comparisons abbreviated in the main text, including the two baselines that were omitted from the main-body figures for readability.

40

37.4

36.7 37.4 36.7

35.7 35.7

33.7

33.7

35

32.0

30

28.2

28.2

Score

25

22.8 22.4

20

15

10

Qwen2.5-7B -Instruct

RLMT (DPO)

RLMT (PPO)

RLCF (DPO)

Rubric-RM (DPO)

Rubric-ARROW w/o RL (DPO)

RIFL (DPO)

Rubric-ARROW (DPO)

RLCF (IterDPO)

Rubric-RM (IterDPO)

Rubric-ARROW w/o RL (IterDPO)

RIFL (IterDPO)

Rubric-ARROW (IterDPO)

Figure 4: Comparison of trained policy models on IFBench. Results of baselines except Rubric-RM (IterDPO) are from OpenRubrics Liu et al. (2025a).

- 35

- 36

- 37

- 38

- 39

- 40

- 41

40.5

39.8

39.4

38.8

38.8

38.6

38.6

38.5

38.3

38.3

Score

37.5

37.4

Qwen2.5-7B -Instruct

RaR Rubric-based RL-S

RuscaRL Rubric-RM (DPO)

Rubric-ARROW w/o RL (DPO)

RIFL (DPO)

Rubric-ARROW (DPO)

Rubric-RM (IterDPO)

Rubric-ARROW w/o RL (IterDPO)

RIFL (IterDPO)

Rubric-ARROW (IterDPO)

Figure 5: Comparison of trained policy models on Create Writing Benchmark v3. Results of baselines except Rubric-RM are from RuscaRL Zhou et al. (2025).

###### A.6. Case Study

- Table 12 expands the case study from the main text, showing the prompt, the two candidate responses, and the full rubric and judge output for both RIFL and Rubric-ARROW.

- B. Proofs of Theoretical Results This appendix provides the technical assumptions and full proofs omitted from the main text.

- B.1. Auxiliary setup For the judge-side results, define

- 1

- 2

- 1

- 2

(︀

(︀

S¯n+ > S−)︀

)︀

, (14)

S+ > S¯n−

Jn =

+

P

P

where S¯n+ and S¯n− are the averages of n i.i.d. copies of S+ and S−, respectively. For the optimization results, fix a phase b ∈ {j,r} and hold the other module θ−b constant throughout the phase. The phase-specific population objective is

Jb(θb; θ−b) = {︃Jj(θj; θr), b = j, Jr(θr; θj), b = r.

(15)

Let LG,b(θb; θ−b) denote the one-step on-policy GRPO surrogate for phase b, and let the GRPO iterates satisfy

θbt+1 = θbt + ηtgt(b), t = 0,1, . . . , T − 1, (16) where gt(b) is the stochastic gradient estimator for the active phase. We use the following assumptions for Theorems 5.3 and 5.4.

- A1. Phase-wise smoothness and upper boundedness. For each phase b ∈ {j,r}, LG,b(θb; θ−b) is Lb-smooth in θb when θ−b is fixed, and LG,b(θb; θ−b) ≤ L⋆G,b.
- A2. Unbiased phase-wise gradient estimator. At each step, E[gt(b) | θbt] = ∇θbLG,b(θbt; θ−b). (17)
- A3. Bounded conditional variance. There exists a constant σb2 such that

E[︂

⃦gt(b) − ∇θbLG,b(θbt; θ−b)⃦

2

| θbt]︂ ≤

σb2 Gb

, (18) where Gj = 2n for the judge phase and Gr = n for the generator phase.

- A4. Step size. We use

ηt =

η0 √T

, η0 ≤

1 Lb

. (19)

- A5. Phase-wise surrogate-to-true gradient gap. For each phase b ∈ {j,r}, there exist nonnegative constants Bgrp(b) , Bstale(b) , and Bclip(b) such that

E⃦∇θbJb(θb; θ−b) − ∇θbLG,b(θb; θ−b)⃦2

sup

θb

Bgrp(b) Gb

+ Bstale(b) + Bclip(b) . (20)

≤

- B.2. Proof of Theorem 5.1 Proof. We first analyze the term P(S+ > S¯n−). Since

S+ − S¯n− = (µ+ − µ−) +

(︀

ξ+ − ξ¯n−

)︀

= ∆ + Zn−, (21) where ξ¯n− := n1

∑︀n

m=1 ξm− and Zn− := ξ+ − ξ¯n−, we have

P(S+ > S¯n−) = P

(︀

Zn− > −∆

)︀

. (22)

Because ξ+ and each ξm− are independent and symmetric about 0, the average ξ¯n− is symmetric about 0, and therefore Zn− = ξ+ − ξ¯n− is also symmetric about 0. Since Zn− is continuous, its median is 0, which implies

P(Zn− > 0) =

- 1

- 2

. (23) Moreover, the map c  → P(Zn− > c) is strictly decreasing in c. Therefore,

P

(︀

Zn− > −∆

)︀

⎧ ⎪⎨

⎪⎩

> 12, ∆ > 0, = 12, ∆ = 0, < 12, ∆ < 0.

(24)

The second term is analogous. Writing

S¯n+ − S− = ∆ + Zn+, Zn+ := ξ¯n+ − ξ−, (25) we again obtain that Zn+ is continuous and symmetric about 0, hence

P(S¯n+ > S−)

⎧ ⎪⎨

⎪⎩

> 12, ∆ > 0, = 12, ∆ = 0, < 12, ∆ < 0.

(26)

Averaging Equations (24) and (26) inside Equation (14) yields

Jn

⎧ ⎪⎨

⎪⎩

> 21, ∆ > 0, = 12, ∆ = 0, < 12, ∆ < 0.

(27)

This proves the claim.

| |
|---|

- B.3. Proof of Theorem 5.2 Proof. Because Gaussian distributions are closed under averaging and subtraction, we have

(︀

)︀

, (28) with

ξ+ − ξ¯n−

###### S+ − S¯n− = ∆ +

)︂. (29)

###### ξ+ − ξ¯n− ∼ N(︂0, σ+2 +

σ−2 n

Therefore,

⎞ ⎠. (30)

⎛ ⎝ ∆

P(S+ > S¯n−) = Φ

√︁σ+2 + σ−2 /n

Similarly,

(︀

)︀

, (31) with

ξ¯n+ − ξ−

###### S¯n+ − S− = ∆ +

+ σ−2 )︂, (32) which implies

###### ξ¯n+ − ξ− ∼ N(︂0,

σ+2 n

⎛ ⎝ ∆

⎞ ⎠. (33)

P(S¯n+ > S−) = Φ

√︁σ−2 + σ+2 /n

Combining the two branches gives

⎞ ⎠

⎛ ⎝ ∆

[︃Φ

- 1

- 2

###### Jn =

√︁σ+2 + σ−2 /n

(34)

⎛ ⎝ ∆

⎞ ⎠

]︃.

###### + Φ

√︁σ−2 + σ+2 /n

For the baseline single pairwise comparison, note that

###### S+ − S− ∼ N(︀

###### )︀

, (35) so

∆, σ+2 + σ−2

⎛ ⎝ ∆

⎞ ⎠. (36)

Jpair = Φ

√︁σ+2 + σ−2

If n > 1, then

σ−2 n

σ+2 n

< σ−2 + σ+2 . (37)

###### σ+2 +

###### < σ+2 + σ−2 , σ−2 +

Since Φ is strictly increasing and ∆ > 0, each argument in Equation (34) is strictly larger than the argument defining Jpair, hence Jn > Jpair.

Finally, if σ+ = σ− := σ, then both terms in Equation (34) coincide and equal

Φ(︂

)︂, (38)

∆ σ√1 + 1/n

which is strictly increasing in n whenever ∆ > 0.

| |
|---|

###### B.4. Proof of Theorem 5.3

Proof. Fix a phase b ∈ {j,r} and suppress the frozen variable θ−b for readability. Since LG,b is Lb-smooth in θb, we have

LG,b(θbt+1) ≥ LG,b(θbt) + ηt⟨∇θbLG,b(θbt), gt(b)⟩

(39)

Lbηt2 2

2

⃦gt(b)⃦

.

−

Conditioning on θbt and using Equation (17),

[︀⟨∇θbLG,b(θbt), gt(b)⟩ | θbt

]︀

= ⃦∇θbLG,b(θbt)⃦2. (40) Moreover, by Equation (18),

E

[︀

]︀

2

= ⃦∇θbLG,b(θbt)⃦2

⃦gt(b)⃦

| θbt

E

[︀

]︀

2

⃦gt(b) − ∇θbLG,b(θbt)⃦

| θbt

(41)

+ E

σb2 Gb

≤ ⃦∇θbLG,b(θbt)⃦2 +

.

Substituting these identities into Equation (39) gives

[︀

###### ]︀

LG,b(θbt+1) | θbt

E

≥ LG,b(θbt)

+ ηt(︁1 − Lb2ηt)︁

(42)

###### ⃦∇θbLG,b(θbt)⃦2 −

Lbηt2σb2 2Gb

.

Because ηt ≤ 1/Lb, we have 1 − Lb2ηt ≥ 12. Taking full expectation and summing over t = 0, . . . , T − 1,

###### ∑︁T−1

- 1

- 2

ηt E⃦∇θbLG,b(θbt)⃦2 ≤ L⋆G,b − LG,b(θb0)

t=0

(43)

###### ∑︁T−1

Lbσb2 2Gb

ηt2.

+

t=0

Under Equation (19),

∑︁T−1

###### ∑︁T−1

√T,

ηt2 = η02. (44)

ηt = η0

t=0

t=0

√T yields

Dividing Equation (43) by 21η0

(︀

)︀

∑︁T−1

L⋆G,b − LG,b(θb0)

2

1 T

E⃦∇θbLG,b(θbt)⃦2 ≤

###### √T

η0

(45)

t=0

Lbη0σb2 Gb√T

.

+

which implies the stated O(T−1/2) rate.

| |
|---|

- B.5. Proof of Theorem 5.4 Proof. At each step t, decompose the phase-wise true gradient as

∇θbJb(θbt) = ∇θbLG,b(θbt)

(46)

+ (︁∇θbJb(θbt) − ∇θbLG,b(θbt))︁.

Using ∥u + v∥2 ≤ 2∥u∥2 + 2∥v∥2, we obtain

⃦∇θbJb(θbt)⃦2 ≤ 2⃦∇θbLG,b(θbt)⃦2

(47)

+ 2⃦∇θbJb(θbt) − ∇θbLG,b(θbt)⃦2.

Take expectations and average over t = 0, . . . , T − 1:

∑︁T−1

∑︁T−1

2 T

1 T

E⃦∇θbJb(θbt)⃦2 ≤

E⃦∇θbLG,b(θbt)⃦2

t=0

t=0

(48)

∑︁T−1

2 T

E⃦∇θbJb(θbt) − ∇θbLG,b(θbt)⃦2.

+

t=0

The first term is controlled by Equation (45). For the second term, apply the gradient-gap assumption in Equation (20) uniformly over the iterates:

∑︁T−1

2 T

E⃦∇θbJb(θbt) − ∇θbLG,b(θbt)⃦2

t=0

2Bgrp(b) Gb

+ 2Bstale(b) + 2Bclip(b) .

≤

Substituting Equations (45) and (49) into Equation (48) yields

(49)

∑︁T−1

1 T

E⃦∇θbJb(θbt)⃦2

t=0

(︀

)︀

L⋆G,b − LG,b(θb0)

2Lbη0σb2 Gb√T

4

√T

≤

+

η0

2Bgrp(b) Gb

+ 2Bstale(b) + 2Bclip(b) .

+

(50)

This proves the claimed O(T−1/2) + O(G−1

b ) stationary-neighborhood bound for the phase-specific true objective. Since Gj = 2n and Gr = n, the rollout-dependent term is O(n−1) in both phases.

| |
|---|

#### C. Prompts

We report the prompts used in our experiments in this section. For baseline methods, we follow the prompts provided by their official implementations and original papers.

###### Prompt for Rubric Generation (Rubric-ARROW)

"Your task is to extract a set of rubric-style instructions from a user's request.\n" "These rubrics will be used as evaluation criteria to check if a response fully meets the request.\n" "Every rubric item must be a universal principle. If any rubric still contains topic-specific references (e.g., names, places, myths, numbers, historical facts), it is automatically invalid.\n" "\n" "- **Two Distinct Categories:**\n" " - [Hard Rule]: Derived strictly from explicit requirements stated in the <request> (format, length, structure, forbidden/required elements, etc.).\n" " - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria (e.g., clarity, correctness, sound reasoning, pedagogy).\n" "\n" "- **Comprehensiveness:**\n" " The rubric must cover all critical aspects implied by the request and examples, including explicit requirements and implicit quality standards.\n" "\n" "- **Conciseness & Uniqueness:**\n" " Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria must be merged into a single rubric. Wording must be precise and free of repetition.\n" "\n" "- **Format Requirements:**\n" " - Use a numbered list.\n" " - Each item starts with \"The response\" phrased in third person.\n" " - Append [Hard Rule] or [Principle] at the end of each item.\n" " - Do not include reasoning, explanations, or examples in the final output—only the rubrics.\n" "\n" "Here is the request:\n" "{prompt}\n" "\n" "Please generate the rubrics for the above request."

###### Prompt for Judge Generation (Rubric-ARROW)

Your job is to look at a conversation and a set of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation <<conversation>>

# Rubric item <<rubric_item>>

# Instructions Return a json object. For each rubric item i (starting from 1), keys must be exactly "explanation_i" and "criteria_met_i" for each i and it includes two top-level fields in the JSON object:

- - The "explanation_i" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- - The "criteria_met_i" field should be a boolean indicating (true/false) whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- - One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

# Final Output Format (a single JSON object, not an array) {

- "explanation_1": "...",

- "criteria_met_1": true/false,

"explanation_2": "...",

- "criteria_met_2": true/false,

... repeat this pattern for every rubric item i in order (i = 1, 2, 3, ...)

} # Final instruction Return just the json object. Do not include any other text in the response.

Table 12: Case study with rubric-level error highlighting. The baseline marks several surface-level criteria as true for the incorrect response, while Rubric-ARROW aligns each judge decision with the corresponding rubric and rejects the misleading comparison.

Case Study on RewardBench Chat Hard Prompt: What is the difference between birding and bird watching?

- Resp A: Birding is more active and committed, often involving documenting birds, while bird watching is a more casual or passive activity.
- Resp B: Birding and bird watching both involve observing birds, but birding is done with the eyes, while bird watching is done with the ears. RIFL Rubric Judge on Resp B Resp A Resp B

The response must directly compare birding and bird watching. [Hard Rule]

“The reply gives a direct comparison/contrast between birding and bird watching (one uses eyes, the other uses ears), so it does compare the concepts, albeit incorrectly.”

###### True True

The response should provide clear and accurate definitions for both concepts. [Principle]

“The assistant did not provide clear, accurate definitions for either concept; birding and bird watching are both defined incorrectly and imprecisely.”

False False

The response should identify key distinctions between birding and bird watching. [Principle]

“The stated distinctions are inaccurate and not the key distinctions usually cited, such as commitment, equipment, or identification/logging goals.”

True False

The response should use appropriate terminology. [Principle]

“Terminology is imprecise and inappropriate (‘eyes’ vs. ‘ears’); correct terms would include equipment, goals, habitat, or level of commitment.”

False False

The response should be clearly organized. [Principle]

“The response is organized simply, with a comparison followed by a contrast, and guides the reader clearly despite being content incorrect.”

###### False True

The response should be concise and avoid unnecessary information. [Principle]

“The answer is concise and contains no unnecessary information.” False True

The response should maintain an objective and neutral tone. [Principle]

###### “Tone is objective and neutral throughout.” False True

The response should be factually accurate. [Principle]

“The content is factually inaccurate regarding the distinctions between birding and bird watching.”

False False

Rubric-ARROW Rubric Judge on Resp B A B

The response must directly address the user’s question about the difference between birding and bird watching. [Hard Rule]

“The reply mentions both are about observing birds but does not accurately explain the real semantic distinction. The key difference in meaning is not correctly captured.”

###### True False

The response must provide clear and accurate definitions of both birding and bird watching. [Principle]

“The response lacks clear definitions for each term; it gives trivial distinctions rather than concise definitions of what birding and bird watching typically entail in practice.”

False False

The response must focus on the primary distinctions between the two terms, such as activity level, scope, goals, record-keeping, expertise, or equipment. [Principle]

“The stated methods (eyes vs. ears) do not reflect the primary objectives or typical scope differences. It fails to focus on the actual primary distinctions such as activity level, year-round involvement, and context.”

False False

