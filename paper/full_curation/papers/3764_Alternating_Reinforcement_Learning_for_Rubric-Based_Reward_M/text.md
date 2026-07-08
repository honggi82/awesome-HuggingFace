# arXiv:2602.01511v2[cs.CL]11Feb2026

## Alternating Reinforcement Learning for Rubric-Based Reward Modeling in Non-Verifiable LLM Post-Training

Ran Xu1,* Tianci Liu2,* Zihan Dong3 Tony Yu4 Ilgee Hong4 Carl Yang1 Linjun Zhang3 Tuo Zhao4 Haoyu Wang5

1Emory University 2Purdue University 3Rutgers University 4Georgia Institute of Technology 5University at Albany

Standard reward models typically predict scalar scores that fail to capture the multifaceted nature of response quality in non-verifiable domains, such as creative writing or open-ended instruction following. To address this limitation, we propose Rubric-ARM, a framework that jointly optimizes a rubric generator and a judge using reinforcement learning from preference feedback. Unlike existing methods that rely on static rubrics or disjoint training pipelines, our approach treats rubric generation as a latent action learned to maximize judgment accuracy. We introduce an alternating optimization strategy to mitigate the non-stationarity of simultaneous updates, providing theoretical analysis that demonstrates how this schedule reduces gradient variance during training. Extensive experiments show that RubricARM achieves strong performance among baselines on multiple benchmarks and significantly improves downstream policy alignment in both offline and online reinforcement learning settings.

Keywords: Rubrics-as-Rewards, Reward Modeling, LLM Alignment, Synthetic Data

Date: 2026-02-13

Model Weights & Checkpoints: https://huggingface.co/collections/OpenRubrics/rubricarm

Contact: ran.xu@emory.edu; liu3351@purdue.edu; hwang28@albany.edu

#### 1. Introduction

Reward modeling serves as the compass for aligning large language models (LLMs) with human intents, typically by generating a scalar score or preference label to predict human preferences (Stiennon et al., 2020, Wang et al., 2024a). However, in complex non-verifiable domain, such as creative writing or open-ended instruction following, these scalar or pairwise judgments often fail to capture the multifaceted nature of response quality (Ying et al., 2025). To address this limitation, recent advancements have shifted toward rubric-based reward modeling, where models explicitly generate structured criteria to ground their judgments (Gunjal et al., 2026, Liu et al., 2025a, Pathak et al., 2025). By decomposing evaluation into interpretable dimensions, rubric-based models offer transparency and improve generalization across prompt-specific evaluation axes.

Central to rubric-based evaluation is the availability of high-quality rubrics. To ensure rubric quality, earlier work has primarily relied on human-authored rubrics, which are expensive to produce and difficult to scale to large datasets (Arora et al., 2025). More recent approaches seek to automate rubric construction using LLMs (Viswanathan et al., 2025, Gunjal et al., 2026); however, these methods are

∗

These authors contributed equally to this work, order was determined randomly (by rolling a die).

largely prompting-based and rely on fixed, frozen models for both rubric generation and response quality judgment. Consequently, they do not update the model’s intrinsic capabilities to the target domain or the underlying preference distribution, limiting their ability to generate in-domain, preference-aligned rubrics. Moreover, even when learning-based components are introduced (Liu et al., 2025a, Rezaei et al., 2025), the rubric generator and the judge are treated as separate modules and trained independently rather than jointly optimized. This decoupled training pipeline prevents deeper integration between rubric construction and judgment, leading to suboptimal evaluation signals. Designing effective rubricbased reward models are still challenging.

In this work, we propose Rubric-ARM, an end-to-end framework that jointly optimizes the rubric generator and the judge via alternating reinforcement learning (RL), enabling the two components to co-evolve and mutually reinforce one another during training. We formulate rubrics as latent actions that guide the reward model in recovering the underlying preference signal, and posit that improved rubric generation directly leads to more accurate preference predictions. To ensure stable joint optimization, Rubric-ARM employs an alternating training strategy that decouples the learning dynamics while preserving a shared objective. Training alternates between (i) optimizing the reward model with a fixed rubric generator to align with target preference labels, and (ii) optimizing the rubric generator with a fixed reward model to produce discriminative rubrics that maximize prediction accuracy.

A key challenge of the alternating RL is the instability caused by simultaneous updates to both components. Our analysis reveals that early-stage exploration by the rubric generator can dominate the learning dynamics. To mitigate this, we first stabilize the reward model under fixed rubrics before optimizing the rubric generator. This alternating schedule reduces variance and ensures robust optimization.

Our contributions can be summarized as follows:

- • We develop Rubric-ARM, a rubric-based reward model to produce high-quality rubrics and precise judgments. To the best of our knowledge, this is the first approach that jointly optimizes rubric and judging via RL.
- • We introduce an alternating RL training algorithm that couples the rubric generator and judge through a shared correctness objective, enabling mutual improvement while stabilizing optimization.
- • We evaluate Rubric-ARM across diverse alignment settings (9 reward modeling and 6 policy benchmarks). Rubric-ARM outperforms strong reasoning-based judges and prior rubric-based reward models, achieving a +4.7% average gain on reward-modeling benchmarks, and consistently improves downstream policy post-training when used as the reward signal.

#### 2. Related Works

LLM-based Reward and Judge Models. While Zheng et al. (2023) established the foundational utility of LLM-based judges. Subsequent research expanded the scope of reasoning to include chain-of-thoughts (Zhang et al., 2025), self-critiques (Ankner et al., 2024, Yu et al., 2025b, Mahan et al., 2024) or plan evaluations strategically (Saha et al., 2025). Liu et al. (2025c) explore inference-time reasoning for generative reward models. Recent studies (Chen et al., 2025, 2026, Whitehouse et al., 2026, Guo et al.,

- 2025, Hong et al., 2025, Xu et al., 2026) leverage online RL to directly incentivize detailed reasoning, aiming to mitigate bias and enhance the accuracy of pointwise and pairwise scoring. Rubrics-based Reward Models. Recently, rubric-based approaches have emerged as a promising

- direction for LLM evaluation (Arora et al., 2025, Hashemi et al., 2024, Pathak et al., 2025, Akyürek et al., 2025), alignment (Viswanathan et al., 2025, Zhang et al., 2026), and reasoning (Gunjal et al.,
- 2026, Zhou et al., 2025, Huang et al., 2025). However, a unique challenge lies in generating high-quality rubrics at scale. To address this, Li et al. (2026), Liu et al. (2025a), Xie et al. (2025) extract rubrics from pairwise comparison signals, while Rezaei et al. (2025), Zhang et al. (2026), Shao et al. (2025) dynamically generate rubrics by leveraging policy model outputs in an online setting.

#### 3. Preliminaries

We study rubric-based reward modeling in non-verifiable domains, where response quality cannot be directly validated against ground truth. The rubric-based reward model contains two parts, namely rubric generator and judge. The key components of Rubric-ARM are described as follows.

Rubrics. We define a rubric as a structured set of evaluation criteria conditioned on a prompt. Formally, let x denote a prompt, a rubric r(x) = {ci}ik=1 consists of k criteria, where each ci specifies a distinct aspect of response quality (e.g., factual correctness, tone, or presentation).

For training rubric-based reward models in non-verifiable domains, a pairwise preference dataset is given as 𝒟 = {(xi, y(i1), y(i2), oi⋆)}iN=1, where x is a prompt, y(1) and y(2) are two candidate responses, and o⋆ ∈ {0,1} indicates which response is preferred (e.g., o⋆ = 1 means y(1) ≻ y(2)). Formally, the rubric generator πr generates a rubric r from the prompt as

r ∼ πr(⋅ ∣ x; θr), (1)

while a judge πj predicts a preference o with the reasoning chain c conditioned on the prompt, responses, and rubric as

(c, o) ∼ πj(⋅ ∣ x, y(1), y(2),r; θj). (2) Learning Objective. We define the preference-correctness reward

R(o, o⋆) = I[o = o⋆], (3) where I[o = o⋆] represents if the binary prediction extracted from o aligns with ground truth o⋆. Denote θr, θj as the parameter for πr and πj respectively, our goal is to learn (θr, θj) that maximize expected preference correctness under generated rubrics:

[R(o, o⋆)]. (4)

max

E

E

E

r∼πr(⋅∣x;θr)

(x,y(1),y(2),o⋆)∼𝒟

(c,o)∼πj(⋅∣x,y(1),y(2),r;θj)

θr,θj

Since both r (text) and c, o (discrete decision with reasoning) are sampled actions, we optimize eq. (4) with RL.

#### 4. Rubric-ARM: Alternating RL for Rubric Generation and Judging

In non-verifiable domains, supervision is limited to pairwise preference feedback and rubrics are not directly observed. Simultaneously updating the rubric generator πr and the judge πj leads to nonstationary learning targets and unstable optimization. As shown in Figure 1, Rubric-ARM addresses this challenge using an alternating RL scheme that decouples the updates of two components.

Input

### Rubric-ARM

Prompt

Rubric Generator 𝝅𝒓

AlternatingRL

[Figure 1]

- Response A Rubrics

Preference Prediction

[Figure 2]

[Figure 3]

- Response B

[Figure 4]

Judge 𝝅𝒋

Output

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

###### A < B

A B

Figure 1: The overall framework for Rubric-ARM.

###### 4.1. Stage I: SFT Warmup

We equip both πj and πr with basic rubric generation and judging capabilities via leveraging open-source datasets. Following the prior work (Liu et al., 2025a), we fine-tune on synthetic rubrics and judge trajectories derived from open-source datasets including UltraFeedback (Cui et al., 2024), SkyWork (Liu

- et al., 2024), Magpie (Xu et al., 2025b), and Synthetic Instruction Following (Lambert et al., 2025a).

Both πr(r ∣ x; θr) and πj(c, o ∣ x, y(1), y(2),r; θj) are trained with the standard next-token prediction objective.

###### 4.2. Stage II: Alternating Reinforcement Learning

Stage I (SFT) warm-starts the rubric generator πr and judge πj by imitating synthetic rubric generation and judging trajectories, but optimizes the two components independently and does not directly target preference correctness. We therefore optimize both components using alternating reinforcement learning. Specifically, training switches between (i) improving the judge with a fixed rubric generator and (ii) improving the rubric generator with a fixed judge, providing each component with a clearer learning signal while preserving the same end objective R(o, o⋆).

- (i) RL for Judge πj with the current πr. With the rubric generator parameters θr held fixed, we update θj to improve preference correctness under rubrics sampled from πr:

[I[o = o⋆]]. (5)

Jj(θj; θr) = E

max

###### E

###### E

r∼πr(⋅∣x;θr)

(c,o)∼πj(⋅∣x,y(1),y(2),r;θj)

(x,y(1),y(2),o⋆)∼𝒟

θj

This phase trains the judge to produce rubric-conditioned evaluations that recover the dataset preference. Since πr(⋅ ∣ x; θr) is fixed during judge updates, we cache rubrics to reduce sampling cost and stabilize optimization. For each training instance (xi, y(i1), y(i2), oi⋆), we sample a rubric ri ∼ πr(⋅ ∣ xi; θr) once and reuse it for multiple judge optimization steps, yielding the Monte Carlo estimate:

Jj(θj; θr) ≈ E

[I[o = oi⋆]]. (6)

###### E

(xi,y(i1),y(i2),oi⋆)∼𝒟,ri

(c,o)∼πj(⋅∣xi,y(i1),y(i2),ri;θj)

In practice, we use a shaped reward that augments the final correctness signal Racc = I[o = oi⋆] with format-based reward Rfmt that enforces valid judging trajectories (i.e., addressing each rubric criterion with per-criterion explanations, followed by an overall justification and a final decision). The final reward for the judge πj is Rj = Racc + Rfmt.

- (ii) RL for Rubric Generator πr with the current πj. With the judge parameters θj fixed, we update θr to prefer rubrics that lead the current judge to make correct decisions. Concretely, we maximize the preference correctness under rubrics drawn from πr as:

Jr(θr; θj) = E

[I[o = o⋆]]. (7)

max

###### E

###### E

r∼πr(⋅∣x;θr)

(c,o)∼πj(⋅∣x,y(1),y(2),r;θj)

(x,y(1),y(2),o⋆)∼𝒟

θr

Intuitively, πr learns to generate criteria that are discriminative for the given prompt and usable by the judge to recover the dataset preference. In practice, we approximate the expectation with a single rollout by greedy decoding (t = 0), i.e., we generate one judging trajectory (c, o) per rubric and use the Monte Carlo estimate

Rr = I[o = o⋆]. (8) Optimization (alternating RL). Rubric-ARM alternates between optimizing Eq. 5 and 7. At iteration t, we run:

rit ∼ πr(⋅ ∣ xi; θrt) ∀(xi, y(i1), y(i2), oi⋆) ∈ 𝒟, (9) θjt+1 ← GRPO θjt ; {rit},𝒟 , (10) θrt+1 ← GRPO(θrt ; θjt+1,𝒟). (11)

Here we cache one rubric per instance during judge updates (since πr is fixed in that phase). In each phase, GRPO (Shao et al. (2024), details in Appendix A) updates only the active policy while keeping the other fixed. Notably, we alternate training by updating the judge before the rubric generator in each cycle. In Sec. 5, we provide theoretical analysis proving the benefits of this ordering.

Connection to EM Algorithm. Our alternating optimization can be viewed as a generalized EM procedure (Dempster et al., 1977) with rubrics r as latent variables. For each preference instance

(x, y(1), y(2), o⋆), the judge defines a conditional model pθj(o⋆ ∣ x, y(1), y(2),r), while the rubric generator πr(r ∣ x; θr) acts as an amortized variational distribution over the latent rubric (Agrawal and Domke, 2021). With πr fixed, updating πj maximizes the expected correctness (or log-likelihood) under sampled rubrics, analogous to the M-step. With πj fixed, updating πr increases probability mass on rubrics that make the current judge more likely to recover o⋆, analogous to an amortized E-step. Because rubrics are high-dimensional discrete text sequences, we use stochastic policy-gradient updates rather than exact posterior inference, yielding a stochastic EM-style coordinate ascent scheme.

###### 4.3. Policy Model Post-training with Rubric-ARM

We use the trained rubric generator πr(⋅ ∣ q; θr) and judge πj(⋅ ∣ q,⋅,⋅,r; θj) to provide preference supervision for post-training a policy model πϕ(a ∣ q), where q denotes the prompt and a denotes a

candidateand predictsresponse.a preferenceFor anylabelpair of responses (a, b), Rubric-ARM samples a rubric r ∼ πr(⋅ ∣ q; θr) ô = Judgeθ

(q, a, b,r) ∈ {0,1}, (12)

j

where ô = 0 indicates a ≻ b and ô = 1 indicates b ≻ a. Preference Optimization with Rubric-ARM. Given a prompt q, we sample two rollouts from the current policy,

a1, a2 ∼ πϕ(⋅ ∣ q), (13) and use Rubric-ARM to label which one is preferred via Eq. (12) and retain examples where the predictions are consistent for both orders. We then update πϕ with the standard DPO objective (Rafailov

- et al., 2023) relative to a fixed reference policy πref. For iterative DPO (Xiong et al., 2024, Pang et al.,

- 2024), we repeat (i) sampling rollouts, (ii) labeling them with Rubric-ARM, and (iii) applying DPO updates for multiple rounds.

Online RL with Rubric-ARM. Following recent works on using pairwise judges to provide reward signals (Xu et al., 2025a), we also consider online RL where Rubric-ARM provides rewards for optimizing πϕ. For each prompt q, we adopt the ReMax-style baseline construction (Li et al., 2024) by first generating a deterministic reference response via greedy decoding,

a(0) = Greedy(πϕ(⋅ ∣ q)) (t = 0), (14) and then sample K additional rollouts,

{a(k)}kK=1 ∼ πϕ(⋅ ∣ q). (15) To mitigate positional bias, we query the judge in both orders under the same rubric r. Let ô→(k) ∈ {0,1} denote the judge outcome for (q, a(k), a(0),r) and ô←(k) ∈ {0,1} for the swapped order (q, a(0), a(k),r). We define the final reward for response a(k) as

Rϕ(q, a(k)) =

- 1

- 2 (I(ô→(k) = 0) + I(ô←(k) = 1)). (16)

#### 5. Theoretical Analysis

We analyze the gradient variance to justify our training schedule. We compare two phases: Strategy A (Judge Warmup), where we optimize the judge with pre-generated, reused rubrics; and Strategy B (Rubric Generator Training), where we optimize the rubric generator against a fixed judge.

log πj(o ∣ c,r) be the score functions. Let p(r) ∶= P(o = o∗ ∣ c,r) be the judge’s correctness probability given a rubric. We define the gradient variance as Var(ĝ) ∶= E∥ĝ∥2 − ∥E[ĝ]∥2.

Setup. Let ur(r) ∶= ∂θ∂

log πr(r ∣ x) and uj(o ∣r) ∶= ∂θ∂

r

j

###### 5.1. Variance Decomposition

We first examine Strategy A. By freezing the rubric r¯ (reuse) during judge updates, we eliminate inter-rubric variance.

- Proposition 5.1 (Judge Variance under Strategy A). Conditioned on a reused rubric r¯, the variance of the judge’s gradient estimator ĝA is solely determined by the judge’s binary classification uncertainty:

Var(ĝA ∣r¯) = p(r¯)(1 − p(r¯))∥uj(o∗ ∣r¯)∥2. (17)

- Proposition 5.2 (Generator Variance under Strategy B). The total variance of the generator’s gradient estimator ĝB decomposes into:

Var(ĝB) = E

[p(r)(1 − p(r))∥ur(r)∥2]

+ Varr(p(r)ur(r))

(18)

r

(II) Cross-Rubric Inconsistency

(I) Multiplicative Reward Noise

Interpretation. Term (I) represents the judge’s Aleatoric uncertainty amplified by the high-dimensional generator gradient ∥ur∥2. Term (II) captures the optimization difficulty when different rubrics yield different expected rewards p(r), causing the gradient direction to oscillate.

###### 5.2. Variance Domination in Early Training

We now derive the variance gap. Instead of assuming trivial gradient dominance, we postulate a condition linking the generator’s exploration intensity to its gradient magnitude.

Assumption 5.3 (Exploration-Gradient Sufficiency). We assume that during early training, the generator’s gradient norm is sufficient relative to the judge’s, satisfying the following exploration-dependent lower bound:

√ 1 − p(r) 1 − p(r) + C1p(r)

∥ur∥ ∥uj∥

, (19)

>

iswheredefinedp representsas: C1 ∶=theVarjudge’sr(p(r)correctnessur(r))/Erprobability[p(r)2∥ur(r(analyzed)∥2]. pointwise or in expectation), and C1 ∈ (0,1) Remark 5.4. The condition in Assumption 5.3 is mild and physically justified. Active exploration (C1 > 0) introduces a positive buffer, making the required gradient-norm ratio on the RHS strictly less than 1 and thus avoiding the need for the generator’s gradient to strictly dominate. Moreover, the judge and generator both produce comparable-length sequences over the same vocabulary (checks/prediction vs. rubrics), so their gradient norms are typically of the same order; the exploration buffer is enough to absorb small mismatches and satisfy the condition in practice.

Theorem 5.5 (Strict Variance Domination). Under Assumption 5.3, the gradient variance of Strategy B strictly dominates the expected conditional variance of Strategy A:

Var(ĝB) > E

[Var(ĝA ∣r¯)]. (20)

r¯

This inequality establishes that the structural instability driven by exploration (quantified by C1) is the governing factor in the variance landscape, overriding differences in gradient magnitudes.

Remark 5.6 (Implication for Training Stability). The variance gap derived in Theorem 5.5 justifies the proposed training schedule (We first train the judge, then train the rubric generator, and subsequently perform alternating training following this sequence.) by highlighting a critical trade-off in Signal-to-Noise Ratio (SNR). The strictly higher variance in Strategy B implies that generator updates are dominated by exploration stochasticity rather than the true gradient direction, risking optimization instability. In contrast, Strategy A acts as a variance reduction mechanism: by fixing the rubric, it effectively sets the exploration coefficient C1 → 0 locally, isolating the judge from structural noise and providing a stable target for effective learning.

#### 6. Experiment

###### 6.1. Datasets and Experiment Settings

Training data. We train the two components of Rubric-ARM, the rubric generator and the judge, on the general-domain portions of OpenRubrics (Liu et al., 2025a). The dataset is split equally into non-overlapping parts, and each rubric-judge alternating round is run on a single part. During training judge, we randomly shuffle the order of response candidates to be evaluated; as shown in App. D.2, this practice greatly helps reduce position bias in reward modeling.

Backbone and variants. Both the rubric generator and the judge are fine-tuned from Qwen-3-8B (Team, 2025). At inference time, Rubric-ARM follows the two-stage rubric-judging process, as detailed in Sec. 3. We also report ensemble results voting@5, by aggregating five independent judges via majority voting.

Baselines. For reward-model evaluation, we follow Liu et al. (2025a) and compare Rubric-ARM against strong same-scale white-box judges, including JudgeLRM (Chen et al., 2025), RRM (Guo et al., 2025), RM-R1 (Chen et al., 2026), and Rubric-RM (Liu et al., 2025a) (SFT-only rubric generator +

judge). We also report judges using black-box APIs when available. To isolate the benefit of rubric-aware training, we include a training-free baseline, Qwen-3-8B (Rubric+Judge) (Yang et al., 2025), which directly generates rubrics and judgments via prompting. For policy training, we use RubricARM as the reward model to fine-tune Qwen2.5-7B-Instruct (Qwen et al., 2025) and compare against Skywork (Liu et al., 2024), ArmoRM (Wang et al., 2024b), UltraFeedback (Cui et al., 2024), RLCF/AI Judge (Viswanathan et al., 2025), OnlineRubrics (Rezaei et al., 2025), and Rubric-RM (Liu et al., 2025a).

Evaluation benchmarks and metrics. We evaluate Rubric-ARM as a pairwise reward model on widely used alignment benchmarks: RewardBench (Chat/Chat-Hard) (Lambert et al., 2025b), RM-Bench (Liu et al., 2025b), PPE-IFEval (Frick et al., 2024), FollowBench (Jiang et al., 2024), InfoBench (Qin et al., 2024), IFBench (Peng et al., 2025), RewardBench2 (Precise-IF/Focus) (Malik et al., 2025), Arena-Hard (Chiang et al., 2024), AlpacaEval 2 (Dubois et al., 2025), Creative Writing Benchmark v3 (Paech, 2025), WildBench (Lin et al., 2024), and WritingPreferenceBench (Ying et al., 2025). For FollowBench and InfoBench, we convert the original single-response setup to pairwise evaluation by sampling two responses from the same model (Qwen-3-8B/14B) and using the benchmark’s verifier to identify constraint violations. We follow each benchmark’s official splits and scoring rules, reporting accuracy, win-rate, or the benchmark-specific metric.

###### 6.2. Performance of Rubric-ARM

- Table 1 compares Rubric-ARM against a broad set of judge/reward models. Rubric-ARM achieves the best average performance among all white-box methods, improving Rubric-RM from 70.1 to 74.8, and reaching 76.2 with voting@5. These gains are consistent across both instruction-following and preference-style benchmarks, supporting our key contribution: Rubric-ARM learns more discriminative rubrics and a more reliable rubric-conditioned judge through RL. Notably, Rubric-ARM also substantially outperforms API-based judges (e.g., 76.2 vs. 71.3 for Rubric+Judge API and 64.9 for direct Judge API), indicating that explicit rubric-conditioned learning yields a stronger and more stable evaluation signal than black-box judging.

- Table 1: Comparison of different judge and reward models across multiple benchmarks. RewardBench2 reports results on Precise IF, and Focus dimensions. Rubric API uses GPT-4.1-Mini, and Judge API uses Gemini-2.5-FlashLite. Best results are highlighted in bold.

RewardBench IF Evaluation Benchmarks RM-Bench RewardBench2

HelpSteer3 Avg. Chat Chat Hard FollowBench PPE-IFEval InfoBench IFBench Chat Precise IF Focus

Black-box LLMs (For reference only)

Claude-3.5-Sonnet 96.4 74.0 – 58.0 – – 62.5 38.8 87.0 – Gemini-2.5-Flash 95.0 83.3 86.0 75.0 85.6 69.3 78.5 57.5 84.1 70.6 78.5 API (Rubric+Judge) 79.6 79.2 83.2 61.0 82.2 66.2 67.9 42.5 79.6 71.4 71.3 API (direct Judge) 89.6 71.2 81.7 59.2 72.9 60.4 67.2 13.2 63.4 70.3 64.9

Larger White-box LLMs (For reference only)

RM-R1-14B (Qwen-2.5-Inst) 73.5 79.8 84.0 59.0 85.5 60.8 73.2 23.8 84.6 74.8 69.9 RM-R1-14B (DeepSeek-Dist) 90.3 78.9 89.9 61.2 82.4 59.0 71.4 30.6 79.0 74.6 71.7 RM-R1-32B (Qwen-2.5-Inst) 95.3 80.3 84.9 60.4 86.1 60.4 75.3 33.1 84.2 72.9 73.3 RM-R1-32B (DeepSeek-Dist) 95.3 83.1 89.2 63.2 85.0 58.6 74.2 36.9 79.2 75.6 74.0 RRM-32B 94.7 81.1 85.7 60.2 84.4 60.8 73.9 34.4 83.6 75.4 73.4

White-box Judge/Reward LLMs

RM-R1-7B (Qwen-2.5-Inst) 83.0 70.0 56.3 55.2 71.3 55.2 64.2 20.6 76.2 65.2 61.7 RM-R1-7B (DeepSeek-Dist) 85.3 67.3 69.7 51.0 70.3 56.5 62.2 13.8 55.4 62.6 59.4 RRM-7B 77.7 69.5 65.5 51.0 68.2 53.2 59.9 10.0 60.4 62.4 57.8 JudgeLRM-7B 92.1 56.1 79.8 46.0 62.7 47.5 55.4 9.4 29.1 60.2 53.8

Rubric-based Methods

Qwen-3-8B (Rubric+Judge) 73.9 63.6 63.0 53.8 74.6 55.6 64.2 21.9 56.6 61.8 58.9 Rubric-RM 88.2 74.1 76.1 67.0 80.8 65.4 65.7 34.4 82.2 67.0 70.1 Rubric-RM-voting@5 89.9 75.4 81.5 70.8 83.8 67.1 67.0 40.0 86.5 67.5 73.0 Rubric-ARM 89.4 79.6 85.7 70.8 86.1 65.9 69.2 41.9 89.4 69.8 74.8 Rubric-ARM-voting@5 90.3 80.7 87.4 72.0 87.7 67.1 69.1 46.2 90.3 71.1 76.2

63.2

64

61.0

59.8 60.3

59

57.5

57.4

56.6

Score

53.1

54

49

46.8

44.7

44

Claude-4 o4-mini Gemini Skywork

Skywork -Gemma

RRM RM-R1

RM-R1 -Qwen2.5

Rubric -RM

Rubric -ARM

-DS

-Llama

Figure 2: Performance of different judge and reward models on WritingPreferenceBench.

We further assess generalization on WritingPreferenceBench (Ying et al., 2025), shown in Fig. 2 (detail results are shown in Table 12), which serves as an out-of-distribution benchmark since none of the compared reward/judge models are trained on this domain. Despite this distribution shift, RubricARM remains strong and achieves the best overall score among all methods (63.2), outperforming Rubric-RM (60.3) and strong reasoning reward models such as RM-R1-Qwen2.5-7B (59.8). The improvements are broad across diverse writing genres (e.g., Functional, Promotional, Non-Fiction, and Poetry), suggesting that Rubric-ARM learns rubrics that capture transferable criteria beyond the training domains, thereby providing a robust reward signal with improved OOD generalization.

- Table 2: Ablation study about the effectiveness of the format reward and the order of judge optimization and rubric generator. Best results are highlighted in bold.

RewardBench IF Evaluation Benchmarks RM-Bench RewardBench2

HelpSteer3 Avg. Chat Chat Hard FollowBench PPE-IFEval InfoBench IFBench Chat Precise IF Focus

Rubric-ARM switch opt 93.2 76.3 85.9 67.3 84.1 64.6 69.5 24.4 86.1 71.8 72.4 Rubric-ARM switch opt-voting@5 94.0 76.5 89.1 67.8 85.0 64.6 69.8 39.4 90.1 72.4 74.9 Rubric-ARM w/o format 89.8 78.7 87.1 69.2 86.1 64.3 69.5 25.6 84.8 70.8 72.6 Rubric-ARM w/o format-voting@5 91.5 78.5 88.2 70.2 87.7 65.1 69.7 43.8 88.9 71.1 75.5 Rubric-ARM 89.4 79.6 85.7 70.8 86.1 65.9 69.2 41.9 89.4 69.8 74.8 Rubric-ARM-voting@5 90.3 80.7 87.4 72.0 87.7 67.1 69.1 46.2 90.3 71.1 76.2

- Table 3: Comparison of trained policy models with different reward models on a format-based constrained instruction-following benchmark (IFEval) and an open-ended benchmark (InfoBench). Baseline results with "⋆" are from Viswanathan et al. (2025), Liu et al. (2025a). Results with underlines are reproduced by us using official checkpoints and evaluation scripts. Best scores are in bold.

IFEval (Prompt) IFEval (Inst.) IFEval InfoBench Loose Strict Loose Strict AVG AVG

Model

GPT-4 (0314)⋆ 79.3 76.9 85.4 83.6 81.3 87.3 AutoIF (Dong et al., 2025) 56.9 47.1 67.0 57.6 57.2 80.6 UltraIF (An et al., 2025) 75.4 71.3 83.0 79.4 77.3 80.7 RAIF (Qin et al., 2025) – – – – 70.1 82.7

Qwen2.5-7B-Instruct⋆ 75.0 72.5 81.8 79.9 77.3 78.1 (76.0)

+ SFT (Distilled)⋆ 66.8 64.1 75.3 72.8 69.8 72.5 + DPO (via Skywork)⋆ 75.7 68.0 83.2 78.5 76.0 82.0 + DPO (via ArmoRM)⋆ 73.8 70.2 81.7 78.3 76.0 83.5 + DPO (via Ultrafbk.)⋆ 71.5 69.1 79.9 77.7 74.6 80.0 + DPO (via AI Judge)⋆ 73.0 68.9 80.9 77.8 75.2 76.1 + DPO (via RLCF)⋆ 77.3 72.6 84.1 80.3 78.6 84.1 (81.5)

+ IterDPO (via RLCF) 78.2 74.3 84.5 81.1 79.5 81.8 + DPO (via Rubric-RM)⋆ 78.2 73.9 84.5 81.2 79.5 83.0 + IterDPO (via Rubric-RM) 77.6 74.1 84.3 81.7 79.4 83.3

+ DPO (via Rubric-ARM) 78.7 76.0 84.7 82.5 80.4 83.7 + IterDPO (via Rubric-ARM) 79.3 75.1 86.0 82.9 80.8 85.0

###### 6.3. Ablation Study

- Table 2 reports two ablation studies that examine (i) the optimization order between the judge and the rubric generator, and (ii) the contribution of the format reward. Unless stated otherwise, all settings are kept identical to Rubric-ARM.

Optimization order. Our default schedule updates the judge first, then the rubric generator, and alternates thereafter. Swapping this order (switch opt) consistently hurts performance: the average drops from 74.8→72.4 (−2.4) without voting and from 76.2→74.9 (−1.3) with voting@5, with especially large regressions on strict instruction-following metrics (e.g., RewardBench2-Precise IF: 41.9→24.4). This suggests that a stronger judge provides a less noisy learning signal for rubric optimization.

Format reward. Removing the format reward (w/o format) also degrades results: 74.8→72.6 (−2.2) without voting and 76.2→75.5 (−0.7) with voting@5. The largest gains appear on structure-sensitive metrics (e.g., RewardBench2-Precise IF: +16.3), indicating that Rfmt helps prevent degenerate judging behaviors (e.g., missing criteria checks) and improves rubric adherence.

- Table 4: Comparison of different strategies applied to Qwen2.5-7B-Instruct on Arena-Hard and AlpacaEval. Results are reported for vanilla models and style/length-controlled settings. Baseline results with "⋆" are from Viswanathan et al. (2025), Rezaei et al. (2025), Liu et al. (2025a). Best results are in bold.

Arena-Hard AlpacaEval

Model

AVG Vanilla Style-Con Vanilla Length-Con

GPT-4 (0314)⋆ 50.0 50.0 22.1 35.3 39.4 UltraIF (An et al., 2025) 31.4 – – – –

Qwen2.5-7B-Instruct⋆ 51.3 42.8 33.5 36.2 41.0 + SFT (Distilled)⋆ 32.6 29.2 36.1 33.3 32.8 + DPO (via Skywork)⋆ 55.1 50.3 44.8 41.5 47.9 + DPO (via ArmoRM)⋆ 50.8 46.4 37.6 38.1 43.2 + DPO (via Ultrafbk.)⋆ 52.8 47.9 33.7 38.7 43.3 + DPO (via AI Judge)⋆ 51.0 44.4 28.8 33.4 39.4 + DPO (via RLCF)⋆ 54.6 48.4 36.2 37.1 44.1 + IterDPO (via RLCF) 51.1 54.6 38.9 39.2 46.0 + DPO (via Rubric-RM)⋆ 52.9 53.1 47.0 41.3 48.6 + IterDPO (via Rubric-RM) 56.3 56.7 50.1 42.0 51.3 + RL (via OnlineRubrics)⋆ 56.5 – 55.0 30.4 –

+ DPO (via Rubric-ARM) 57.8 59.5 47.1 42.5 51.7 + IterDPO (via Rubric-ARM) 58.8 58.9 52.0 44.0 53.4

- 6.4. Performance of offline RL-based Policy Models We evaluate whether the benefit of Rubric-ARM transfers to downstream offline policy learning.

Instruction-Following Evaluation. Table 3 and Fig. 3 show that policies optimized with Rubric-ARMtrained rewards consistently achieve the strongest instruction-following performance. On IFEval, DPO with Rubric-ARM improves the overall average to 80.4, and iterative DPO further raises it to 80.8 (best), with particularly strong gains on instruction-level constraints. The advantage also transfers to the open-ended InfoBench benchmark, where Rubric-ARM reaches 83.7 with DPO and 85.0 with iterative DPO (best). Compared to iterative baselines, Rubric-ARM remains consistently stronger: on IFBench (Fig. 3), RLCF improves from 28.2 to 32.0 with IterDPO, while Rubric-ARM achieves 35.4 with IterDPO; similarly, iterative Rubric-RM reaches 33.7, still below Rubric-ARM. Overall, these results indicate that Rubric-ARM provides a more precise reward signal, and that iterative optimization amplifies the gains over both one-shot DPO and iterative baselines.

Human Preference Alignment Evaluation. Table 4 and Table 5 show that Rubric-ARM-trained rewards consistently yield stronger preference alignment across both controlled and open-domain evaluations. On Arena-Hard and AlpacaEval (Table 4), DPO with Rubric-ARM achieves the best overall average (51.7), and IterDPO further improves it to 53.4 (best). On WildBench (Table 5), RubricARM again yields the strongest macro score: DPO via Rubric-ARM reaches 53.7, while IterDPO via Rubric-ARM achieves 55.7 (best), improving over IterDPO with Rubric-RM (54.0) by 1.7%, indicating improved preference-aligned helpfulness on broad, real-world tasks.

Creative Writing. We further evaluate whether Rubric-ARM-based rewards benefit open-ended generation on the Creative Writing Benchmark v3 (Fig. 4). Policies trained with Rubric-ARM outperform baselines: DPO using Rubric-ARM achieves 39.0, and IterDPO further improves to 39.3 (best). Notably,

- Table 5: Comparison of different alignment strategies applied to Qwen2.5-7B-Instruct on WildBench. Results are reported for task-specific scores and task macro WB score. Baseline results with "⋆" are from Wang et al. (2025). Best results are in bold.

###### Method Creative Planning Math Info seeking Coding WB Score

Claude-3.5-Sonnet (20240620)⋆ 55.6 55.6 50.2 55.5 56.5 54.7 GPT-4-turbo (20240409)⋆ 58.7 56.2 51.0 57.2 55.1 55.2 GPT-4o-mini (20240718)⋆ 60.1 58.2 54.0 57.4 57.2 57.1

Qwen2.5-7B-Instruct⋆ 50.1 51.8 47.1 50.7 45.0 48.7 +DRIFT⋆ 52.5 53.2 50.6 52.4 50.3 51.7 +SPIN⋆ 43.3 45.5 41.6 46.3 39.1 42.9 +IterDPO⋆ (via OpenAssistant) 46.8 48.6 44.5 48.0 44.3 46.3 +DPO (via RLCF) 51.4 52.7 49.0 51.3 48.8 50.5 +IterDPO (via RLCF) 51.9 52.6 47.8 51.4 46.5 49.7 +DPO (via Rubric-RM) 54.8 55.5 51.5 54.1 52.9 53.6 +IterDPO (viaRubric-RM) 57.0 56.2 50.6 54.9 52.8 54.0

+DPO (via Rubric-ARM) 55.2 55.6 49.5 56.0 53.1 53.7 +IterDPO (via Rubric-ARM) 57.3 57.2 53.3 56.2 55.2 55.7

35.4

35.0

35

33.7

33.7

32.0

30

28.2

28.2

Score

25

22.8 22.4

20

15

Qwen2.5-7B -Instruct

RLMT (DPO)

RLMT (PPO)

RLCF (DPO)

Rubric-RM (DPO)

Rubric-ARM (DPO)

RLCF (IterDPO)

Rubric-RM (IterDPO)

Rubric-ARM (IterDPO)

- Figure 3: Comparison of trained policy models on IFBench. Results of baselines except Rubric-RM (IterDPO) are from OpenRubrics Liu et al. (2025a).

Qwen2.5-7B -Instruct

RaR Rubric-based RL-S

RuscaRL Rubric-RM (DPO)

Rubric-ARM (DPO)

Rubric-RM (IterDPO)

Rubric-ARM (IterDPO)

- 36

- 37

- 38

- 39

Score

37.4

38.8

38.3

38.6

37.5

39.0

38.8

39.3

- Figure 4: Comparison of trained policy models on Create Writing Benchmark v3. Results of baselines except Rubric-RM are from RuscaRL (Zhou et al., 2025).

Rubric-ARM-based optimization also surpasses strong creative-writing baselines such as RaR (38.8) and RuscaRL (38.6), suggesting that rewards learned by Rubric-ARM generalize well to subjective, non-verifiable generation tasks beyond standard instruction following and preference alignment.

- Table 6: Comparison of online RL method with different alignment strategies applied to Qwen2.5-7B-Instruct on instruction following and preference alignment benchmarks. Best results are in bold.

Method

IFEval (Prompt) IFEval (Inst.)

IFBench

AlpacaEval

AVG Loose Strict Loose Strict Vanilla Length

Qwen2.5-7B-Instruct 75.0 72.5 81.8 79.9 28.2 33.5 36.2 46.8 +GRPO (RM-R1) 76.7 73.6 83.2 80.2 30.6 53.2 42.7 52.3 +GRPO (Rubric-ARM) 79.3 76.2 85.3 83.0 34.8 56.2 44.8 55.4

- Table 7: Computing speed on 100 samples (vLLM). Results with “⋆” were taken from Liu et al. (2025a).

80

Compute Time (s)

70

IFEval

AlpacaEval

IFBench

AVG

60

Score

JudgeLRM-7B⋆ 25.71 RRM-7B⋆ 203.40 RM-R1-7B (Qwen-2.5-Inst)⋆ 260.37 RM-R1-7B (DeepSeek-Dist)⋆ 170.76 RM-R1-14B (Qwen-2.5-Inst)⋆ 322.79 RM-R1-14B (DeepSeek-Dist)⋆ 382.02 Rubric-RM-8B 105.12

50

40

30

0 1 2 3

Iteration

Rubric-ARM-8B 33.50

Figure 5: Performance of iterative DPO with Rubric-ARM across three iterations.

###### 6.5. Performance of online RL-based Policy Models

We evaluate Rubric-ARM in an online RL setting by directly optimizing Qwen2.5-7B-Instruct with GRPO (Sec. 4.3) using different reward models. As shown in Table 6, GRPO with Rubric-ARM-trained rewards substantially improves both instruction following and preference alignment compared to the base model and a strong reward baseline RM-R1. Specifically, Qwen2.5-7B-Instruct achieves an average score of 46.8, while GRPO with RM-R1 increases it to 52.3. Replacing the reward with RubricARM yields the best overall performance, reaching 55.4 on average. The gains are consistent across instruction-following and human-preference alignment metrics, which indicates that Rubric-ARM provides a more effective online learning signal for GRPO.

###### 6.6. Effect of Iterative Policy Optimization

Fig. 5 evaluates iterative DPO with Rubric-ARM over three optimization iterations. Overall, the average performance increases monotonically across iterations, indicating that iteratively refining the policy with Rubric-ARM-based supervision yields progressively better alignment. These results suggest that Rubric-ARM provides a sufficiently stable signal to support multi-round offline optimization without performance degradation.

- Table 8: Case study with error highlighting. Baselines either lose in the reasoning process, or favor the longer response but miss the explicit constraint about thumb war, while Rubric-ARM enforces hard rules before evaluating principles.

Case Study on RewardBench Chat Hard Prompt Wars involve armed conflicts between nations, states, groups, or individuals, resulting in the loss of life,

property damage, displacement of populations, and numerous other types of suffering. Is a thumb war violent? I care about both physical and psychological violence.

- Resp A (snippet) “War is inherently violent ...direct physical violence ...psychological violence ...” (“war”; not “thumb war”)
- Resp B (snippet) “ ...not considered a violent activity but rather a fun and harmless pastime ...friendly competition ...” Label Resp B.

RRM-7B “ ...Assistant 1 ...doesn’t directly answer ...But wait ...the instruction doesn’t specify only thumb war

...it’s more about war in general ...I’m a bit confused ...(No Decision)”

JudgeLRM “Assistant 1 provided a detailed and accurate response ...Assistant 2’s answer is inaccurate ...(Choose

A)” Rubric-ARM

Rubric “1. The response must directly address whether a thumb war constitutes violence. [Hard Rule] 2. The response must consider both physical and psychological aspects of violence. [Hard Rule] 3. The response should accurately define and distinguish relevant concepts to ensure clarity. [Principle] 4. The response should present a balanced analysis by acknowledging multiple perspectives or contexts. [Principle] 5. The response should support its conclusions with logical reasoning or evidence. [Principle] 6. The response should maintain a respectful and empathetic tone. [Principle] ”

Judge “The response must directly address whether a thumb war constitutes violence ...Response A fails ...not

addressing thumb wars at all. Response B satisfies both .... (Choose B)”

###### 6.7. Efficiency Comparison

We conclude with an inference-cost analysis and case studies. Table 7 reports wall-clock time on 100 RewardBench2 prompts. Despite using two Qwen-3-8B modules (rubric generator + judge), Rubric-ARM runs in 33.50s, faster than most reasoning-based and rubric-based baselines. While JudgeLRM is slightly faster, it does not provide the explicit, interpretable rubric-conditioned signals that Rubric-ARM is designed for downstream policy optimization. Overall, our rubric-judge design replaces long chain-of-thought with short rubric generation and lightweight judging, yielding strong efficiency. Rubric-ARM is also faster than Rubric-RM, which typically generates longer rubric lists and incurs higher overhead.

###### 6.8. Case Study

We qualitatively analyze failures of baseline reward models on challenging examples. Table 8 shows a RewardBench Chat-Hard instance about thumb war: reasoning-based models (e.g., RRM-7B and JudgeLRM) are distracted by “war” and incorrectly prefer an armed-conflict response. In contrast, Rubric-ARM generates and enforces a rubric with an explicit hard rule about thumb war, leading to the correct preference. We provide additional IFBench examples in App. D.3, where Rubric-ARM reliably extracts hard constraints and judges correctly while Rubric-RM fails.

#### 7. Conclusion

In this work, we propose Rubric-ARM, a novel framework for reward modeling in non-verifiable LLM post-training. Treating rubric generation as a latent action, we jointly optimize a generator and a judge via alternating reinforcement learning. To ensure stability, we employ an alternating update schedule, a design theoretically grounded in our gradient-variance analysis. Empirically, Rubric-ARM achieves 4.7% gains across diverse benchmarks and robust out-of-distribution generalization. It also delivers superior supervision for policy alignment in both offline and online RL settings, showing Rubric-ARM offers a more reliable reward signal than static approaches.

#### References

Abhinav Agrawal and Justin Domke. Amortized variational inference for simple hierarchical models. Advances in Neural Information Processing Systems, 34:21388–21399, 2021.

Afra Feyza Akyürek, Advait Gosai, Chen Bo Calvin Zhang, Vipul Gupta, Jaehwan Jeong, Anisha Gunjal, Tahseen Rabbani, Maria Mazzone, David Randolph, Mohammad Mahmoudi Meymand, et al. Prbench: Large-scale expert rubrics for evaluating high-stakes professional reasoning. arXiv preprint arXiv:2511.11562, 2025.

Kaikai An, Li Sheng, Ganqu Cui, Shuzheng Si, Ning Ding, Yu Cheng, and Baobao Chang. UltraIF: Advancing instruction following from the wild. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 18711–18726, Suzhou, China, November 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025.emnlp-main.945/.

Zachary Ankner, Mansheej Paul, Brandon Cui, Jonathan Daniel Chang, and Prithviraj Ammanabrolu. Critique-out-loud reward models. In Pluralistic Alignment Workshop at NeurIPS 2024, 2024. URL https://openreview.net/forum?id=CljYUvIlRW.

Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng He. Judgelrm: Large reasoning models as a judge. arXiv preprint arXiv:2504.00050, 2025.

Xiusi Chen, Gaotang Li, Ziqi Wang, Bowen Jin, Cheng Qian, Yu Wang, Hongru Wang, Yu Zhang, Denghui Zhang, Tong Zhang, et al. RM-r1: Reward modeling as reasoning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=1ZqJ6jj75q.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating llms by human preference, 2024. URL https://arxiv.org/abs/ 2403.04132.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models

with scaled AI feedback. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=BOorDpKHiJ.

Arthur P Dempster, Nan M Laird, and Donald B Rubin. Maximum likelihood from incomplete data via the em algorithm. Journal of the royal statistical society: series B (methodological), 39(1):1–22, 1977.

Guanting Dong, Keming Lu, Chengpeng Li, Tingyu Xia, Bowen Yu, Chang Zhou, and Jingren Zhou. Self-play with execution feedback: Improving instruction-following capabilities of large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=cRR0oDFEBC.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. Length-controlled alpacaeval: A simple way to debias automatic evaluators, 2025. URL https://arxiv.org/abs/2404.04475.

Evan Frick, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios N. Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. How to evaluate reward models for rlhf, 2024. URL https://arxiv.org/abs/2410.14872.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= c1bTcrDmt4.

Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. Reward reasoning models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems,

###### 2025. URL https://openreview.net/forum?id=V8Kbz7l2cr.

Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin Van Durme, and Chris Kedzie. Llm-rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. arXiv preprint arXiv:2501.00274, 2024.

Ilgee Hong, Changlong Yu, Liang Qiu, Weixiang Yan, Zhenghao Xu, Haoming Jiang, Qingru Zhang, Qin Lu, Xin Liu, Chao Zhang, and Tuo Zhao. Think-RM: Enabling long-horizon reasoning in generative reward models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=UfQAFbP6xq.

Zenan Huang, Yihong Zhuang, Guoshan Lu, Zeyu Qin, Haokai Xu, Tianyu Zhao, Ru Peng, Jiaqi Hu, Zhanming Shen, Xiaomeng Hu, et al. Reinforcement learning with rubric anchors. arXiv preprint arXiv:2508.12790, 2025.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. FollowBench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4667–4688, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-long.257/.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James Validad Miranda, Alisa Liu, et al. Tulu 3: Pushing frontiers in open language model post-training. In Second Conference on Language Modeling, 2025a. URL https://openreview. net/forum?id=i1uGbfHHpH.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi. RewardBench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1755–1797. Association for Computational Linguistics, 2025b.

Sunzhu Li, Jiale Zhao, Miteto Wei, Huimin Ren, Yang Zhou, Jingwen Yang, Shunyu Liu, Kaike Zhang, and Wei Chen. Rubrichub: A comprehensive and highly discriminative rubric dataset via automated coarse-to-fine generation. arXiv preprint arXiv:2601.08430, 2026.

Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. In International Conference on Machine Learning, pages 29128–29163. PMLR, 2024.

Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Faeze Brahman, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. Wildbench: Benchmarking llms with challenging tasks from real users in the wild, 2024. URL https://arxiv.org/abs/2406.04770.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024.

Tianci Liu, Ran Xu, Tony Yu, Ilgee Hong, Carl Yang, Tuo Zhao, and Haoyu Wang. Openrubrics: Towards scalable synthetic rubric generation for reward modeling and llm alignment. arXiv preprint arXiv:2510.07743, 2025a.

Yantao Liu, Zijun Yao, Rui Min, Yixin Cao, Lei Hou, and Juanzi Li. RM-bench: Benchmarking reward models of language models with subtlety and style. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id=QEHrmQPBdd.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025c.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. Generative reward models. arXiv preprint arXiv:2410.12832, 2024.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A Smith, Hannaneh Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937, 2025.

###### Samuel J Paech. Eq-bench creative writing benchmark v3. https://github.com/EQ-bench/ creative-writing-bench, 2025.

Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason E Weston. Iterative reasoning preference optimization. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=4XIKfvNYvx.

Aditya Pathak, Rachit Gandhi, Vaibhav Uttam, Arnav Ramamoorthy, Pratyush Ghosh, Aaryan Raj Jindal, Shreyash Verma, Aditya Mittal, Aashna Ased, Chirag Khatri, et al. Rubric is all you need: Enhancing llm-based code evaluation with question-specific rubrics. arXiv preprint arXiv:2503.23989, 2025.

Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Bin Xu, Lei Hou, and Juanzi Li. Agentic reward modeling: Integrating human preferences with verifiable correctness signals for reliable reward systems. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15934–15949, Vienna, Austria, July 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025.acl-long.775/.

Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, and Dong Yu. InFoBench: Evaluating instruction following ability in large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 13025–13048, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.findings-acl.772/.

Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, and Xing Sun. Incentivizing reasoning for advanced instruction-following of large language models. arXiv preprint arXiv:2506.01413, 2025.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report, 2025. URL https: //arxiv.org/abs/2412.15115.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

MohammadHossein Rezaei, Robert Vacareanu, Zihao Wang, Clinton Wang, Bing Liu, Yunzhong He, and Afra Feyza Akyürek. Online rubrics elicitation from pairwise comparisons. arXiv preprint arXiv:2510.07284, 2025.

Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason E Weston, and Tianlu Wang. Learning to plan & reason for evaluation with thinking-LLM-as-a-judge. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=PNRznmmWP7.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G Finlayson, David Sontag, et al. Dr tulu: Reinforcement learning with evolving rubrics for deep research. arXiv preprint arXiv:2511.19399, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. Judging the judges: A systematic study of position bias in llm-as-a-judge. In Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 292–314, 2025.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Vijay Viswanathan, Yanchao Sun, Xiang Kong, Meng Cao, Graham Neubig, and Tongshuang Wu. Checklists are better than reward models for aligning language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/ forum?id=RPRqKhjrr6.

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, et al. Secrets of rlhf in large language models part ii: Reward modeling. arXiv preprint arXiv:2401.06080, 2024a.

Haoxiang Wang, Wei Xiong, Tengyang Xie, Han Zhao, and Tong Zhang. Interpretable preferences via multi-objective reward modeling and mixture-of-experts, 2024b. URL https://arxiv.org/abs/ 2406.12845.

Yifan Wang, Bolian Li, Junlin Wu, Zhaoxuan Tan, Zheli Liu, Ruqi Zhang, Ananth Grama, and Qingkai Zeng. Drift: Learning from abundant user dissatisfaction in real-world preference learning. arXiv preprint arXiv:2510.02341, 2025.

Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, and Swarnadeep Saha. J1: Incentivizing thinking in LLM-as-a-judge via reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=dnJEHl6DI1.

Lipeng Xie, Sen Huang, Zhuo Zhang, Anni Zou, Yunpeng Zhai, Dingchao Ren, Kezun Zhang, Haoyuan Hu, Boyin Liu, Haoran Chen, et al. Auto-rubric: Learning to extract generalizable criteria for reward modeling. arXiv preprint arXiv:2510.17314, 2025.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=c1AKcA6ry1.

Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, and Hongkun Yu. Incentivizing agentic reasoning in llm judges via tool-integrated reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=AXNRILww9c.

Wenyuan Xu, Xiaochen Zuo, Chao Xin, Yu Yue, Lin Yan, and Yonghui Wu. A unified pairwise framework for rlhf: Bridging generative reward modeling and policy optimization. arXiv preprint arXiv:2504.04950, 2025a.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview. net/forum?id=Pnk7vMbznK.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shuangshuang Ying, Yunwen Li, Xingwei Qu, Xin Li, Sheng Jin, Minghao Liu, Zhoufutu Wen, Xeron Du, Tianyu Zheng, Yichi Zhang, et al. Beyond correctness: Evaluating subjective writing preferences across cultures. arXiv preprint arXiv:2510.14616, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https://openreview.net/forum?id=2a36EMSSTp.

Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang, Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, Melanie Kambadur, Dhruv Mahajan, and Rui Hou. Self-generated critiques boost reward modeling for language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11499–11514, 2025b.

Junkai Zhang, Zihao Wang, Lin Gui, Swarnashree Mysore Sathyendra, Jaehwan Jeong, Victor Veitch, Wei Wang, Yunzhong He, Bing Liu, and Lifeng Jin. Chasing the tail: Effective rubric-based reward modeling for large language model post-training. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=pBjy4ek2QV.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Ccwp4tFEtE.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. Swift:a scalable lightweight infrastructure for fine-tuning, 2024. URL https://arxiv.org/abs/2408.05517.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Yixin Cao, Yang Feng, and Deyi Xiong, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 400–410, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.acl-demos.38/.

Yang Zhou, Sunzhu Li, Shunyu Liu, Wenkai Fang, Kongcheng Zhang, Jiale Zhao, Jingwen Yang, Yihe Zhou, Jianwei Lv, Tongya Zheng, et al. Breaking the exploration bottleneck: Rubric-scaffolded reinforcement learning for general llm reasoning. arXiv preprint arXiv:2508.16949, 2025.

#### A. Details for Group Relative Policy Optimization (GRPO)

GRPO (Shao et al., 2024) is an actor-only policy optimization method that reduces variance by using the within-prompt average reward as a baseline. Concretely, for each prompt q, GRPO samples a group of responses O = {o1, o2, . . . , oG} from the old policy πθold(⋅ ∣ q), computes a group-normalized advantage Âi,t for each token, and then performs a PPO-style clipped update. Following Yu et al. (2025a), we upweight informative prompts using a larger clipping threshold εhigh.

∣oi∣

G

min(ρi,t(θ) Âi,t, clip(ρi,t(θ),1 − εlow,1 + εhigh) Âi,t) − β DKL[πθ ∥ πref]],

𝒥GRPO(θ) = E

[

1 G

1 ∣oi∣

∑

∑

q∼P(Q),O∼πθold(⋅∣q)

t=1

i=1

where ρi,t(θ) = ππθ(oi,t∣q,oi,<t)

θold(oi,t∣q,oi,<t) is the token-level importance ratio.

##### B. Detailed Theoretical Derivations In this section, we provide the complete proofs for the variance analysis presented in Section 5.

- B.1. Preliminaries Recall the definitions:

- • Reward: R(o) = I[o = o⋆].
- • Judge Correctness: p(r) = πj(o∗ ∣ c,r).
- • Generator Score: ur(r) = ∂θ∂

r

θr log πr(r ∣ x).

- • Judge Score: uj(o ∣r) = ∂θ∂

j

log πj(o ∣ c,r). We utilize the vector form of the Law of Total Variance: Lemma B.1. For random vectors X and Y, Var(Y) = EX[Var(Y ∣ X)] + VarX(E[Y ∣ X]).

- B.2. Proof of Proposition 5.1 (Strategy A)

- Proof. In Strategy A, the rubric r¯ is fixed. The gradient estimator is ĝA = R(o)uj(o ∣r¯), where o ∼ πj(⋅∣r¯).

Since r¯ is fixed, uj(o ∣r¯) takes two values: uj(o∗ ∣r¯) (when correct) and uj(¬o∗ ∣r¯) (when wrong). Considering the term associated with the reward R(o), the variable is a scaled Bernoulli. Conditioned on r¯:

- • With probability p(r¯), o = o∗, so ĝA = 1 ⋅ uj(o∗ ∣r¯).
- • With probability 1 − p(r¯), o ≠ o∗, so ĝA = 0 (since R = 0).

Let v ∶= uj(o∗ ∣r¯). The first moment is:

E[ĝA ∣r¯] = p(r¯)v + (1 − p(r¯)) ⋅ 0 = p(r¯)v.

The second moment is:

E[∥ĝA∥2 ∣r¯] = p(r¯)∥v∥2 + (1 − p(r¯)) ⋅ 0 = p(r¯)∥v∥2. Thus, the variance is:

Var(ĝA ∣r¯) = E∥ĝA∥2 − ∥E[ĝA]∥2 = p(r¯)∥v∥2 − ∥p(r¯)v∥2

= (p(r¯) − p(r¯)2)∥v∥2

= p(r¯)(1 − p(r¯))∥uj(o∗ ∣r¯)∥2.

| |
|---|

###### B.3. Proof of Proposition 5.2 (Strategy B)

- Proof. In Strategy B, we update θr. The estimator is ĝB = R(o)ur(r), where r ∼ πr and o ∼ πj(⋅∣r). We apply Lemma B.1 conditioning on r.

- Step 1: Conditional Variance (Inner Term). Conditioned on r, ur(r) is a constant vector. The randomness comes only from R(o).

Var(ĝB ∣r) = Varo∣r(R(o)ur(r)) = ∥ur(r)∥2Varo∣r(R(o)). Since R(o)∣r ∼ Bernoulli(p(r)), its variance is p(r)(1 − p(r)). Thus:

Var(ĝB ∣r) = p(r)(1 − p(r))∥ur(r)∥2.

- Step 2: Conditional Expectation (Outer Term).

E[ĝB ∣r] = E

o∣r[R(o)]ur(r) = p(r)ur(r).

- Step 3: Total Variance Decomposition. By applying the Law of Total Variance (Lemma B.1), we express the total variance as the sum of the expected conditional variance and the variance of the conditional expectation:

Var(ĝB) = E

[Var(ĝB ∣r)] + Varr(E[ĝB ∣r]).

r

Substituting the results derived in Step 1 and Step 2 into the equation above yields the final decomposition:

[p(r)(1 − p(r))∥ur(r)∥2] + Varr(p(r)ur(r)). This concludes the proof.

Var(ĝB) = E

r

| |
|---|

- B.4. Proof of Theorem 5.5 Proof. We analyze the sign of the variance difference ∆ = Var(ĝB) − Er¯[Var(ĝA ∣r¯)].

- 1. Variance Difference Expansion. Substituting the expressions from Propositions 5.1 and 5.2:

∆ = E

r

[p(r)(1 − p(r))∥ur(r)∥2] + Varr(p(r)ur(r))

VB

−E

r

[p(r)(1 − p(r))∥uj(o∗ ∣r)∥2]

VA

= E

r

[p(r)(1 − p(r))(∥ur(r)∥2 − ∥uj(o∗ ∣r)∥2)] + Varr(p(r)ur(r)).

- 2. Incorporating the Exploration Coefficient. Using the definition of C1 from Assumption 5.3, we substitute Varr(p(r)ur(r)) = C1 Er[p(r)2∥ur(r)∥2]:

∆ = E

r

[p(r)(1 − p(r))(∥ur(r)∥2 − ∥uj(o∗ ∣r)∥2) + C1p(r)2∥ur(r)∥2].

- 3. Verification of Positivity. To show ∆ > 0, we analyze the term inside the expectation (the integrand). We split the expression into multiple lines to isolate the quadratic components:

Integrand = (p(r) − p(r)2)∥ur(r)∥2 − (p(r) − p(r)2)∥uj(o∗ ∣r)∥2 + C1p(r)2∥ur(r)∥2

= p(r)[(1 − p(r))∥ur(r)∥2 − (1 − p(r))∥uj(o∗ ∣r)∥2 + C1p(r)∥ur(r)∥2]

= p(r)[∥ur(r)∥2(1 − p(r) + C1p(r)) − ∥uj(o∗ ∣r)∥2(1 − p(r))]. We now invoke the inequality from Assumption 5.3:

√ 1 − p(r) 1 − p(r) + C1p(r)

∥ur(r)∥ ∥uj(o∗ ∣r)∥

>

.

Squaring both sides and rearranging:

∥ur(r)∥2(1 − p(r) + C1p(r)) > ∥uj(o∗ ∣r)∥2(1 − p(r)).

This implies that the term inside the square brackets is strictly positive. Since p(r) ∈ (0,1), the entire integrand is strictly positive for any r. Therefore, the expectation is strictly positive:

[Var(ĝA ∣r¯)]. This concludes the proof.

∆ > 0 ⟹ Var(ĝB) > E

r¯

| |
|---|

#### C. Implementation Details

- Table 9 and Table 10 show the hyperparameters used in Rubric-ARM and policy model training. We implement the GRPO training based on ms-swift1 library (Zhao et al., 2024) and implement DPO and

1https://github.com/modelscope/ms-swift

IterDPO based on LLaMA-Factory2 (Zheng et al., 2024). We totally conduct 3 iterations for Rubric-ARM alternating RL training. Additionally, the sampling parameters used in inference are summarized in Table 11. We used the same sampling parameters as their official implementations and papers for baseline methods.

- Table 9: Hyper-parameters used in Rubric-ARM training. Module Parameter Value Module Parameter Value

Rubric Generator

#generations 6

Judge

#generations 7

Cutoff Length 512 Cutoff Length 1024

Batch Size 288 Batch Size 224

Optimizer AdamW Optimizer AdamW

Learning Rate 1e-6 Learning Rate 1e-6 Temperature 1.0 Temperature 1.0 #iterations 2 #iterations 2

Epochs 1 Epochs 1

ϵhigh 0.28 ϵhigh 0.28

ϵlow 0.2 ϵlow 0.2

β 0.001 β 0.001

- Table 10: Hyper-parameters used in policy model training. Method Parameter Value Method Parameter Value

Cutoff Length 2048

#generations 6

Batch Size 64 Cutoff Length 2048

Optimizer AdamW Batch Size 288 Learning Rate 8e-7 Optimizer AdamW

Epochs 1 Learning Rate 5e-7 beta 0.1 Temperature 1.0

DPO

GRPO

SFT mixing weight 0.2 #iterations 2 / / Epochs 1

/ / ϵhigh 0.28

/ / ϵlow 0.2

/ / β 0.001

#### D. Additional Experimental Results

- D.1. Performance on WritingPreferenceBench We present the performance on WritingPreferenceBench in Table 12.

2https://github.com/hiyouga/LlamaFactory

- Table 11: Sampling parameters used in Rubric-ARM inference. Module Parameter Value Module Parameter Value

Rubric Generator

Maximum Tokens 1024

Judge

Maximum Tokens 4096

Temperature 0.0 Temperature 1.0 Top-P / Top-P 1.0

Top-K / Top-K -1

Enable-thinking False Enable-thinking False

- Table 12: Comparison of different judge and reward models on WritingPreferenceBench. Best results are highlighted in bold.

Func. Promo. Non-Fic. Fiction Funny Poetry Script Role AVG LLM as Judge (black-box model)

Claude-4-Opus-thinking 65.7 64.3 64.1 60.1 54.2 64.0 43.5 51.7 61.0 OpenAI-o4-mini 58.3 58.6 60.9 55.5 53.2 68.0 30.4 55.2 56.6 Gemini-2.5-Flash 59.1 57.7 62.5 59.8 52.2 56.0 34.8 51.7 57.5

White-box Reward Models

Skywork-Llama-3.1-8B 53.6 56.3 60.6 49.0 52.2 56.0 65.2 41.4 53.1 Skywork-Gemma-2-27B 49.0 53.9 59.6 33.9 55.1 36.0 21.7 51.7 46.8 RM-R1-DeepSeek-Qwen-7B 62.5 55.1 59.2 55.4 58.0 56.0 65.2 41.4 57.4 RM-R1-Qwen2.5-7B 67.0 57.2 53.9 60.0 54.6 72.0 47.8 65.5 59.8 RRM-7B 50.0 35.3 50.0 49.5 38.5 36.4 45.5 53.8 44.7

Rubric-based Models

Rubric-RM 58.3 58.5 57.9 58.3 58.0 76.0 47.8 55.2 60.3 Rubric-ARM 67.8 63.1 65.8 60.9 61.0 80.0 47.8 55.2 63.2

###### D.2. Position Bias Analysis

In this section, we study position bias in pairwise judge and reward models, where the predicted preference may depend on the relative order of the two responses (Shi et al., 2025). We evaluate three settings: (1) keeping the response order fixed as in the original dataset, (2) flipping the order for all instances, and (3) randomly flipping the order on a per-instance basis. Table 13 reports results on RewardBench and the IF evaluation benchmarks. Overall, baseline methods exhibit non-trivial position bias. For RRM-7B, changing the order leads to a 46.2-point difference on PPE-IFEval (75.8 vs. 29.6). Likewise, for RM-R1-7B (Qwen-2.5-Inst), flipping the order changes InfoBench by 11.9 points (81.8 vs. 69.9). For RM-R1-7B (DeepSeek-Dist), the order sensitivity remains substantial, with a

- 9.9-point difference on InfoBench (78.3 vs. 68.4) and a 9.3-point difference on FollowBench (79.0 vs. 69.7). In contrast, our Rubric-ARM remains consistently stable across different orderings, suggesting substantially reduced position bias and more robust evaluation. This design choice is aligned with our RL training design, where we randomize the response order when collecting reward signals, which further mitigates position bias in downstream policy optimization.

- Table 13: Position bias analysis for different judge and reward models. Rubric-ARM shows much lower sensitivity to the ordering of response pairs.

RewardBench IF Evaluation Benchmarks

Avg. Variation

Chat Chat Hard FollowBench PPE-IFEval InfoBench IFBench White-box Judge/Reward LLM: RRM-7B Mixed Ord 77.7 69.5 65.5 51.0 68.2 53.2

- Fixed Ord-1 73.9 61.6 53.8 29.6 62.3 30.2
- Fixed Ord-2 82.1 72.1 64.7 75.8 74.2 74.2 Variation 8.2 10.5 11.7 46.2 11.9 44.0 22.08 White-box Judge/Reward LLM: RM-R1-7B (Qwen-2.5-Inst) Mixed Ord 83.0 70.0 56.3 55.2 71.3 55.2

- Fixed Ord-1 82.1 63.4 57.1 54.8 81.8 53.8
- Fixed Ord-2 82.4 71.1 56.3 50.4 69.9 54.1

- Variation 0.9 7.7 0.8 4.8 11.9 1.4 4.58 White-box Judge/Reward LLM: RM-R1-7B (DeepSeek-Dist) Mixed Ord 85.3 67.3 69.7 51.0 70.3 56.5

- Fixed Ord-1 87.1 67.3 79.0 52.8 78.3 53.2
- Fixed Ord-2 82.7 69.5 70.6 54.7 68.4 60.6 Variation 4.4 2.2 9.3 3.7 9.9 7.4 6.15 Rubric-based Method: Rubric-RM

- Mixed Ord 88.2 74.1 76.1 67.0 80.8 65.4

- Fixed Ord-1 87.4 74.6 79.8 70.8 80.9 66.4
- Fixed Ord-2 88.7 73.5 75.6 67.2 78.5 64.4

Variation 1.3 1.1 4.2 3.8 2.4 2.0 2.47 Rubric-based Method: Rubric-ARM (Ours)

- Mixed Ord 89.4 79.6 85.7 70.8 86.1 65.9

- Fixed Ord-1 89.9 79.4 84.9 71.8 86.1 65.3
- Fixed Ord-2 88.4 80.3 85.7 71.0 87.9 66.9 Variation 1.5 0.9 0.8 1.0 1.8 1.6 1.27

###### D.3. Additional Case Study

In this section we compare Rubric-ARM with Rubric-RM, another rubric-based RM trained with SFT, on a randomly chosen example from IFBench. The case specifies keywords and paragraph length. Results are shown in Table 14. In this IFBench example, which requires specific keywords and exactly two paragraphs, the baseline Rubric-RM suffers from a judging hallucination, incorrectly claiming that a valid response is split into three paragraphs. Rubric-ARM, on the contrary, accurately extracts these hard constraints and identifies the missing open-source keyword in the negative sample, while correctly verifying the structure of the positive one.

- Table 14: Case study with error highlighting. In the baseline Rubric-RM, the rubric output confused the judge and led to an incorrect prediction.

Case Study on IFBench Prompt What is Nextcloud, why should I use it, and how does it compare to other cloud storage solutions? Please

provide the response with a tone of excitement and enthusiasm. Your response should contain exactly 2 paragraphs and include the keywords ‘cloud storage’ and ‘open-source’.

- Resp A (snippet) “Nextcloud is an incredibly exciting and innovative cloud storage solution ...As an open-source platform

...Imagine having your very own private cloud ...” (2 paragraphs)

- Resp B (snippet) “Nextcloud is an incredibly powerful and flexible solution ...While services like Google Drive and Dropbox are great ...” (2 paragraphs; missing “open-source”)

Label Resp A. Rubric-RM

Rubric 1. The response must define what Nextcloud is. [Hard Rule] 2. The response must explain why the user should use Nextcloud. [Hard Rule] 3. The response must compare Nextcloud to other cloud storage solutions. [Hard Rule] 4. The response must use a tone of excitement and enthusiasm. [Hard Rule] 5. The response must consist of exactly two paragraphs. [Hard Rule] 6. The response must include the keywords ‘cloud storage’ and ‘open-source’. [Hard Rule] 7. The response should present information in a clear, logical structure that guides the reader through definition, benefits, and comparison. [Principle] 8. The response should use vivid, engaging language to maintain reader interest. [Principle] 9. The response should employ precise and varied vocabulary to convey ideas effectively. [Principle] 10. The response should provide specific, relevant details to substantiate its claims. [Principle] 11. The response should maintain coherence and flow, ensuring each paragraph builds on the previous one. [Principle] 12. The response should balance conciseness with sufficient elaboration to cover all required points. [Principle]

- Judge “...Response A is split into three paragraphs, not exactly two ...Response A fails the gatekeeper criterion (paragraph count) ...(Choose B)

Rubric-ARM

Rubric 1. The response must address what Nextcloud is, why it should be used, and how it compares to other cloud storage solutions. [Hard Rule] 2. The response must maintain an enthusiastic and excited tone throughout. [Hard Rule] 3. The response must consist of exactly two paragraphs. [Hard Rule] 4. The response must include the keywords ‘cloud storage’ and ‘open-source’. [Hard Rule] 5. The response should present information in a clear and logical structure, grouping related ideas coherently. [Principle] 6. The response should use vivid and engaging language to sustain reader interest. [Principle] 7. The response should provide relevant comparisons or contrasts to contextualize the subject. [Principle] 8. The response should maintain conciseness and avoid unnecessary repetition. [Principle]

- Judge “...Response B mentions ‘cloud’ and ‘open’ but not the exact keyword ‘open-source,’ so fails. ...Response A fulfills all hard rules, including the precise keywords ...(Choose A)”

#### E. Prompts

We present the prompts we used in this section. For baseline methods, we adopted the prompts from their official implementations and papers.

Prompt for Rubric Generation (Rubric-ARM)

Your task is to extract a set of rubric-style instructions from a user's request. These rubrics will be used as evaluation criteria to check if a response fully meets the request. Every rubric item must be a universal principle. If any rubric still contains topic-specific

references (e.g., names, places, myths, numbers, historical facts), it is automatically

invalid.

- - **Two Distinct Categories:**
- - [Hard Rule]: Derived strictly from explicit requirements stated in the <request> (format, length, structure, forbidden/required elements, etc.).
- - [Principle]: Derived by abstracting any concrete cues into domain-agnostic quality criteria ( e.g., clarity, correctness, sound reasoning, pedagogy).
- - **Comprehensiveness:** The rubric must cover all critical aspects implied by the request and examples, including

explicit requirements and implicit quality standards.

- - **Conciseness & Uniqueness:** Each rubric must capture a distinct evaluation criterion. Overlapping or redundant criteria

must be merged into a single rubric. Wording must be precise and free of repetition.

- - **Format Requirements:**
- - Use a numbered list.
- - Each item starts with "The response" phrased in third person.
- - Append [Hard Rule] or [Principle] at the end of each item.
- - Do not include reasoning, explanations, or examples in the final outputâĂŤonly the rubrics.

Here is the request: {prompt}

Please generate the rubrics for the above request.

Prompt for Judge Generation (Rubric-ARM)

You are a fair and impartial judge. Your task is to evaluate 'Response A' and 'Response B' based on a given instruction and a rubric. You will conduct this evaluation in distinct phases as outlined below.

- ### Phase 1: Compliance Check Instructions First, identify the single most important, objective 'Gatekeeper Criterion' from the rubric.

- - **A rule is objective (and likely a Gatekeeper) if it can be verified without opinion. Key examples are: word/paragraph limits, required output format (e.g., JSON validity), required/ forbidden sections, or forbidden content.**
- - **Conversely, a rule is subjective if it requires interpretation or qualitative judgment. Subjective rules about quality are NOT Gatekeepers. Examples include criteria like "be creative," "write clearly," "be engaging," or "use a professional tone."**

- ### Phase 2: Analyze Each Response Next, for each Gatekeeper Criterion and all other criteria in the rubric, evaluate each response

item by item.

- ### Phase 3: Final Judgment Instructions Based on the results from the previous phases, determine the winner using these simple rules.

Provide a final justification explaining your decision first and then give your decision.

--### REQUIRED OUTPUT FORMAT You must follow this exact output format below.

--- Compliance Check --Identified Gatekeeper Criterion: <e.g., Criterion 1: Must be under 50 words.>

--- Analysis ---

- **Response A:**

- - Criterion 1 [Hard Rule]: Justification: <...>
- - Criterion 2 [Hard Rule]: Justification: <...>
- - Criterion 3 [Principle]: Justification: <...>
- - ... (and so on for all other criteria)

- **Response B:**

- - Criterion 1 [Hard Rule]: Justification: <...>
- - Criterion 2 [Hard Rule]: Justification: <...>
- - Criterion 3 [Principle]: Justification: <...>
- - ... (and so on for all other criteria)
- --- Final Judgment --Justification: <...> Winner: <Response A / Response B>

Task to Evaluate: Instruction: {instruction}

Rubric: {rubric}

- Response A:

- {response_a}

Response B:

- {response_b}

