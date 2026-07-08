### arXiv:2510.21978v2[cs.LG]18Jun2026

# Beyond Reasoning Gains: Mitigating General-Capability Forgetting in Large Reasoning Models

Hoang Phan1,2,∗, Xianjun Yang1, Yuanshun Yao1, Jingyu Zhang1,3,∗, Shengjie Bi1, Xiaocheng Tang1, Madian Khabsa1, Lijuan Liu1, Deren Lei1

1Meta Superintelligence Labs, 2New York University, 3Johns Hopkins University

∗Work done at Meta

Reinforcement learning with verifiable rewards (RLVR) has delivered impressive gains in mathematical and multimodal reasoning and has become a standard post-training paradigm for contemporary language and vision-language models. However, the RLVR recipe introduces a significant risk of capability regression, in which models forget foundational skills after prolonged training without employing regularization strategies. We empirically confirm this concern, observing that opensource reasoning models suffer performance degradation on core capabilities such as perception and faithfulness. While imposing regularization terms like KL divergence can help prevent deviation from the base model, these terms are computed on the current task and therefore do not guarantee preservation of broader knowledge. Meanwhile, commonly used experience replay across heterogeneous domains makes it nontrivial to decide how much training emphasis each objective should receive. To address this, we propose RECAP—a replay strategy with dynamic objective reweighting for general knowledge preservation. Our reweighting mechanism adapts online using short-horizon signals of convergence and instability, shifting the post-training focus away from saturated objectives and toward underperforming or volatile ones. Our method is end-to-end and readily applicable to existing RLVR pipelines without training additional models or heavy tuning. Extensive experiments on benchmarks using Qwen2.5-VL-3B and Qwen2.5-VL-7B demonstrate the effectiveness of our method, which not only preserves general capabilities but also improves reasoning by enabling more flexible trade-offs among in-task rewards.

Date: June 19, 2026 Correspondence: Hoang Phan hvp2011@nyu.edu, Deren Lei deren@meta.com.

#### 1 Introduction

Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated remarkable generalpurpose capabilities (Achiam et al., 2023; Yang et al., 2023), yet strengthening their proficiency in complex reasoning remains a key frontier of research. Reinforcement Learning with Verifiable Rewards (RLVR) (Shao et al., 2024), an extension of Reinforcement Learning from Human Feedback (RLHF) (Ziegler et al., 2019; Ouyang et al., 2022), has emerged as a powerful paradigm for this purpose. By providing explicit reward signals such as exact-match correctness, format adherence, and answer brevity, RLVR has been applied to instruction following, STEM problem solving, code generation, and logical reasoning (Lightman et al., 2023; Peng et al., 2025), resulting in large performance gains on benchmark scores, leading to headlines that language models can “learn to reason” (Guo & DeepSeek-AI, 2025).

Despite strong headline gains, RLVR exhibits recurring failure modes, prompting questions about whether current pipelines genuinely expand reasoning abilities (Shojaee et al., 2025). For example, exploration and diversity collapse occur when on-policy finetuning overly narrows the policy distribution—raising Pass@1 while reducing Pass@k and solution-path diversity (Chen et al., 2026; Dang et al., 2025). Likewise, outcomeonly rewards introduce sparse credit assignment and instability, and not every task can be naturally cast as a reinforcement-learning problem (e.g., translation, summarization, or captioning). In addition, strict answer formats and format-sensitive graders may conflate genuine reasoning improvements with mere format

###### MiMo-VL-7B

###### Qwen2.5-VL-3B

###### Qwen2.5VL-7B

A-OKVQA

A-OKVQA

A-OKVQA

100%

100%

100%

75%

75%

75%

R-Bench-Dis

AesBench

R-Bench-Dis

AesBench

R-Bench-Dis

AesBench

50%

50%

50%

25%

25%

25%

OCRBench

VStar

OCRBench

VStar

OCRBench

VStar

VisOnly

VisOnly

VisOnly

MiMo-VL-7B-SFT

Qwen2.5-VL-3B

Qwen2.5-VL-7B

MiMo-VL-7B-RL

VLAA-Thinker-Qwen2.5VL-3B

VLAA-Thinker-Qwen2.5VL-7B

Qwen2.5VL-3b-RLCS

WeThink-Qwen2.5VL-7B

SpaceThinker-Q

Figure 1 General-capability comparison of base VLMs (blue) and their reasoning-tuned variants (green/purple/red) on six representative non-reasoning benchmarks (higher is better). A-OKVQA (knowledge-based VQA), AesBench (Huang et al., 2024) (image aesthetics), VStar (Wu & Xie, 2023) (spatio-temporal reasoning), VisOnly (Kamoi et al., 2025) (vision-only recognition aggregate), OCRBench (Liu et al., 2024b) (text recognition), and R-Bench-Dis (Li et al., 2025b) (distribution-shift robustness). Across both Qwen2.5-VL families, reasoning-finetuned models generally underperform their base models on perception and robustness tasks, whereas MiMo-VL-7B-RL remains close to its SFT baseline.

compliance, even introducing evaluation artifacts (Petrov et al., 2025). Recent studies report that many RL-trained models underperform even their base models in standardized evaluations, where formatting-rewardonly baselines can degrade original performance even more severely (Prabhudesai et al., 2025). This suggests that the format reward may be underoptimized, yet optimizing it can also cause forgetting of mathematical capabilities (Chandak et al., 2025).

Another critical yet underexplored issue in RLVR is that optimizing for a narrow set of targeted rewards can lead to regression in general capabilities acquired during pretraining. Although models become proficient in following formatting requirements and solving reasoning tasks, they simultaneously exhibit increased hallucinations (Jaech et al., 2024; Yao et al., 2025b) and become more vulnerable to jailbreak attacks (Lou et al., 2025; Yao et al., 2025a). These results suggest that reasoning-oriented post-training can improve reasoning while trading off non-target competencies (e.g., perception, safety, factual grounding), especially when RL training is prolonged without explicit regularization (Liu et al., 2025a).

###### IoU on LISA (out-of-domain)

To examine forgetting from reasoning-focused finetuning, we begin by probing the general abilities of open-source reasoning models beyond their target reasoning tasks. As shown in Fig. 1, models finetuned for chain-of-thought or RL reasoning frequently lag behind their base counterparts on perception and robustness. For example, we observe a consistent drop on VisOnlyQA across both Qwen2.5-VL families, while MiMo-VL-7B-RL performs competitively relative to its base model on those non-reasoning tasks. We hypothesize that this is due to its specialized finetuning strategy, which employs mixed on-policy reinforcement learning to improve the model’s capabilities across multiple axes beyond math and reasoning, according to the MiMo-VL-7B technical report (Team et al., 2025). However, the detailed framework and the sampling or reweighting strategy are not disclosed. These patterns support our central claim: optimizing for reasoning rewards can erode non-reasoning capabilities, motivating a continual learning method to preserve general skills during reasoning-oriented post-training.

67.2

66.9

66.5

65.9 65.7

66

65.1

64

IoU

62

61.0

60.2

59.6

60

RECAP

58.1

57.6

Reasoning

58

0 100 200 300 400 500

Steps

Figure 2 Performance comparison between finetuning solely on math and with RECAP. Our model not only preserves base-model performance but also improves it (↑ 2%), whereas the reasoning-only model quickly falls behind the base model after 100 iterations.

Our initial experiments with Qwen2.5-VL-7B show that training solely on reasoning rewards degrades performance on general capabilities; for example, performance drops by 7% on the

Uniform objective weights Relative priority between objectives

Step 1: Replay general data

Step 2: Inspect per-objective convergence behavior

[Figure 1]

[Figure 2]

Reasoning domains

[Figure 3]

Geometry, Algebra, Diagrams, Charts: Format reward Accuracy reward

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

|final loss|
|---|

[Figure 8]

[Figure 9]

General domains

Object Detection, Segmentation, OCR: Next-token prediction loss Format reward IoU reward

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 3 Overview of RECAP. Along with the target reasoning task, we sample data from general domains to maintain that knowledge during finetuning. Initially, the objectives of interest are weighted uniformly to optimize the main model. After a few iterations, we record the convergence behavior of individual objectives. Based on this behavior, we adjust the focus to prevent any objective from dominating and assign less weight to saturated ones.

perception task, as shown in Figure 2. To address this, we propose gathering general-capability data and integrating it into RLVR via an online weighting mechanism. However, due to cross-domain heterogeneity, it is nontrivial to decide how much weight to assign to each loss term or reward. We then measure how different reward signals evolve during RLVR training in Section 4 and find that some rewards converge up to three times faster than others, suggesting that faster-converging rewards should not be emphasized later in training once the model has already learned the corresponding skills.

Motivated by the above observations, we propose Replay-Enhanced CApability Preservation (RECAP)—a principled replay-based training strategy that mixes general data back into the RL objective and then dynamically reweights objectives based on their convergence rate and instability. As shown in Figure 3, RECAP computes the relative priorities of the objectives of interest by inspecting their convergence behavior to reweight the final loss. Experiments demonstrate that our method not only preserves general capabilities but also improves reasoning performance by allowing flexible trade-offs among reward types. Our contributions are as follows:

- • We systematically re-evaluate open-source reasoning models and show that reasoning-focused finetuning consistently regresses general capabilities. This motivates replaying general-capability data during RLVR to preserve pretraining knowledge. We further show that objectives exhibit distinct convergence behavior, making commonly used, manually tuned reweighting schemes suboptimal in such scenarios.
- • We introduce RECAP, a plug-in scheduler that replays general-capability data during RLVR and dynamically reweights both general and main-task objectives. RECAP naturally down-weights saturated format signals and refocuses capacity on harder, high-variance objectives. The method is end-to-end, magnitude-agnostic, requires no auxiliary models, and can be integrated into RLVR pipelines with negligible overhead.
- • In our experiments, RECAP preserves or improves general capabilities while matching or exceeding the reasoning performance of reasoning-only finetuning. It consistently outperforms strong continualfinetuning baselines and is competitive with specialized open models while using less compute. Empirically, we also find that replaying general data yields shorter, more concise rationales without compromising reasoning ability.

#### 2 Related Work

Foundation models and post-training. Large transformer models pretrained on broad corpora serve as general-purpose backbones with strong abilities and wide transfer across domains (Brown et al., 2020; Touvron

- et al., 2023). Post-training adapts these backbones to downstream tasks via (i) supervised finetuning, from early ULMFiT (Howard & Ruder, 2018) to instruction tuning in FLAN (Wei et al., 2021) or Flan-T5 (Chung
- et al., 2024); (ii) reinforcement learning from human or AI feedback, typically combining preference modeling with policy optimization; and (iii) direct preference optimization objectives that bypass explicit reward models. For reasoning, reinforcement learning with verifiable rewards has become a common recipe: verifiers

or rule-based checkers score final solutions in math and related domains, often within PPO-style pipelines (Guo & DeepSeek-AI, 2025; Liu et al., 2025b). Process-based neural reward models provide supervision for intermediate progress (Setlur et al., 2025) rather than only the final output. However, PRMs can induce reward hacking: agents learn to exploit the appearance of a correct process rather than achieving the intended outcome (Wang et al., 2025b; Shao et al., 2024).

Catastrophic forgetting in continual learning and post-training. Catastrophic forgetting describes performance regression on previously acquired skills when adapting to new data (McCloskey & Cohen, 1989; French, 1999). Early work in this vein introduced regularization-based mitigations such as EWC (Kirkpatrick et al., 2017), SI (Zenke et al., 2017), and MAS (Aljundi et al., 2018) that prevent excessive changes to important parameters. Functional approaches like LwF (Li & Hoiem, 2016) constrain outputs via distillation (Hinton et al., 2015), while replay with small episodic memories (Rebuffi et al., 2017; Lopez-Paz & Ranzato, 2017; Rolnick et al., 2019; Buzzega et al., 2020) provides consistently strong baselines across settings. In RLHF post-training, a KL penalty toward the reference policy is commonly used to stabilize updates and curb over-optimization (Ouyang et al., 2022). In addition to standard KL-regularization approaches, InstructGPT (Ouyang et al., 2022) interleaves pretraining gradients with RLHF updates to reduce drift relative to KL-only regularization (Zheng et al., 2023). Concurrent works (Zhang et al., 2025; Fu et al., 2025) integrate verified rollouts to stabilize learning or penalize discrepancies on augmented training data (Wang et al., 2025d). However, these methods do not guarantee preservation of performance in non-target domains. Other approaches tackle forgetting by incorporating mixed, verifiable reward suites (Team et al., 2025) or introducing reflection or re-attention mechanisms under RL objectives (Chu et al., 2025). In addition, recent reasoning-focused RL pipelines often reduce or remove KL to encourage exploration (Hu et al., 2025a; Hao et al., 2025), potentially exacerbating forgetting.

#### 3 Background

This section first provides the essential background on standard post-training for large language models, covering (i) supervised finetuning (SFT), (ii) RL-based alignment from preferences, (iii) reinforcement learning with verifiable rewards, and (iv) GRPO for long-form reasoning. Our approach builds on recent progress in reasoning-centric LLMs—exemplified by DeepSeek-R1 (Guo & DeepSeek-AI, 2025) and other contemporary models (Ji et al., 2025; Yu et al., 2025; Seed et al., 2025; Wang et al., 2025a). We adopt GRPO as our primary RL algorithm because it effectively reduces memory and compute overhead relative to standard PPO.

Supervised finetuning (SFT). Let an LLM with parameters θ induce a conditional policy πθ(· | x) over responses y to a prompt x. SFT optimizes the negative log-likelihood on instruction–response pairs D =

{(x(i),y(i))}Ni=1:

LSFT(θ) = −

N

log πθ y(i)| x(i) .

i=1

SFT has been central to transferring general-purpose LMs to downstream instruction following and broad zero-shot generalization (e.g., ULMFiT (Howard & Ruder, 2018); instruction-tuned FLAN (Wei et al., 2021)), and it typically provides the initialization for subsequent preference- or reward-based alignment.

RL-based post-training. Reinforcement learning from human feedback (RLHF) fits a reward model rϕ(x,y) from pairwise human preferences (Ziegler et al., 2019; Rafailov et al., 2023; Lambert, 2025), commonly using a Bradley–Terry likelihood (Bradley & Terry, 1952), and then maximizes reward while regularizing toward πref (often with a KL penalty) via policy optimization such as PPO (Schulman et al., 2017):

Ex∼µ, y∼π

θ(·|x) rϕ(x,y) − β Ex∼µ DKL πθ(· | x)∥πref(· | x) .

max

θ

This pipeline improves helpfulness and harmlessness while retaining base-model competence; see early LMpreference work and InstructGPT (Ouyang et al., 2022) for canonical formulations, and PPO for the underlying stable policy-gradient updates (Stiennon et al., 2020).

Reinforcement learning with verifiable rewards (RLVR). In settings with programmatic or automatic verifiers (e.g., exact-match answers, execution-based checks, or constraint checkers), RLVR replaces learned human-preference rewards with verifiable signals r(x,y) ∈ [0,1]. This reduces labeler noise and can better target reasoning fidelity by rewarding demonstrably correct steps or outcomes, while retaining the same KL-regularized RL form (Lightman et al., 2023).

GRPO for long-form reasoning. Group Relative Policy Optimization (GRPO) is a PPO-style algorithm tailored for LLM reasoning that forgoes a learned critic and instead computes advantages from groupnormalized sequence rewards. For each prompt x, sample a group of G rollouts O = {oi}Gi=1 from a frozen rollout policy πθ

. Let Ri be the verifiable sequence-level reward (e.g., exact-match correctness), and define the group-normalized advantage Ai = Ri − mean(R) /std(R). With token-wise importance ratio

old

πθ(oi,t | x,oi,<t) πθ

ri,t(θ) =

,

(oi,t | x,oi,<t)

old

GRPO maximizes the clipped surrogate plus KL regularization:

  1

 .

|oi|

G

1 |oi|

JGRPO(θ) = E

min ri,t(θ) Ai, clip ri,t(θ),1 − ϵ,1 + ϵ Ai − β DKL πθ ∥ πref

G

t=1

i=1

By normalizing across a group of responses for each prompt, GRPO stabilizes updates without a critic, which is preferable when long chain-of-thought rewards are sparse and verifier-based. Empirically, GRPO has been shown to boost mathematical-reasoning performance in open models (Shao et al., 2024).

General ability degradation. Unlike traditional continual-learning studies, modern post-training pipelines must jointly consider gains in reasoning and retention of general abilities acquired during pretraining (e.g., perception, grounding, instruction following, safety). Let G = {G1,...,GM} denote a suite of general-ability tasks and R = {R1,...,RL} a suite of reasoning datasets.

#### 4 Our Method: Replay-Enhanced CApability Preservation (RECAP)

We address forgetting in RLVR by (i) replaying general-capability data alongside reasoning data, and (ii) dynamically reweighting objectives online using local estimates of progress and instability for individual objectives. Below, we present how RECAP governs loss coefficients and shifts the optimization away from saturated objectives toward underperforming or volatile ones—without changing the underlying RL algorithm.

Setting. In the context of supervised learning, let D = {Dn}Nn=1 be N domains and ℓ(n,kt) (θ) the mini-batch loss of objective k ∈ {1,...,K} on domain n at iteration t for parameters θ. Note that K ≥ N because some

tasks use more than one reward or objective. The model parameters θ are thus optimized by minimizing the average loss across objectives:

N

1 N

L(kt) =

ℓ(n,kt) (θ).

n=1

Our framework acts on {L(kt)}Kk=1 regardless of whether each Lk arises from an RL reward surrogate or a supervised learning loss term.

Per-objective convergence rate and stability of convergence. Due to the unstable nature of RL training, we cannot rely solely on the per-step objective or reward value to compute the reweighting coefficients. Instead, for each objective k, we measure the convergence rate over a sliding window of length 2 × W by computing the current window average and the previous window average:

t−W

t

1 W

1 W

µ(kt) =

L(ks)

, µ˜(kt) =

L(ks)

s=t−W+1

s=t−2W+1

estimated current loss value

estimated old loss value

Reward

Convergence Rate

Inverse Signal To Noise

3.5

| |format acc iou ntp<br><br>| | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1.0

1.4

3.0

1.2

0.8

1.0

2.5

0.6

0.8

score

2.0

0.6

0.4

1.5

0.4

0.2

0.2

1.0

0.0

0.0

0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4

0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4

0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4

#tokens 1e6

#tokens 1e6

1e6

- Figure 4 Different rewards exhibit different convergence behavior. While the format reward is easy to optimize and initially has the highest convergence rate, it quickly saturates and thus yields a near-unity convergence rate (c ∼ 1) and low instability (i ∼ 0) after 50 steps. By contrast, the reasoning accuracy fluctuates the most, thereby steering the optimization toward the corresponding objective. IoU and ntp denote the IoU reward and next-token-prediction accuracy during training. The result is obtained in the first setting in our experiments below.

###### and the instability (coefficient of variation) in the same window:

σk(t) =

t

1 2W − 1

L(ks) − µ(kt) 2.

s=t−2W+1

|convergence rate c(kt) = µ˜(kt)/µ(kt)|
|---|

Based on these measures, we form two signals: (i) the

captures how quickly the loss improves, while (ii) the

|inverse signal-to-noise ratio i(kt) = σk(t)/(µ(kt) + µ˜(kt))|
|---|

captures loss instability.

Intuitively, c(kt) > 1 indicates recent improvement (loss dropping relative to the previous window), while c(kt)≈1 signals saturation. The term i(kt) is larger when the objective is noisy or unstable.

Relative priority between domains. We convert these signals into normalized coefficients via a temperaturecontrolled softmax. Given temperature T > 0, we define the priority score of the k-th objective as s(kt) and compute the coefficients λ for reweighting objectives.

s(kt) = c(kt) + i(kt), λ(kt) =

K exp s(kt)/T K i=1 exp s(it)/T

. (1)

The prefactor K preserves average scale so that K1 k λ(kt)=1. Lower T sharpens priorities while higher T approaches uniform mixing. We set T =5 by default in our experiments.

Overall training objective. At step t, we minimize the following weighted objective:

L(t)(θ) =

K

1 K

λ(kt) L(kt).

k=1

Optimizing θ with ∇θL(t) steers learning toward objectives that are both slow to converge (high ck) and fluctuating (high ik), while leaving well-learned, stable objectives with lower weight. The scheme reduces to the standard equal weighting as T → ∞ (i.e., uniformly sample from all domains without loss reweighting).

- Figure 4 illustrates the insight behind our proposed method, in which we finetune the model on a reasoning dataset (tracked by accuracy) while replaying perception data (IoU) and an SFT dataset for general-capability retention. Early in training, the format signal is easy to optimize and saturates quickly, so its c falls toward 1 and i toward 0, reducing its priority. After 100 steps, many signals plateau (convergence rate c≈1) while still differing in stability. Among them, the reasoning reward fluctuates the most (∼ 0.3), yielding higher i and thus higher λ. At this point, the model has learned to answer according to the predefined template; thus, the corresponding signal-to-noise ratio of the formatting reward is ≈ 0. This motivates combining both

progress (c) and instability (i) in Equation 1, as they complement each other. Tuning the trade-off between these two terms offers finer-grained control and can improve performance. However, for simplicity, we take their unweighted sum s = c + i, which performs consistently well in our experiments.

Entropy-regularized interpretation of RECAP reweighting. We can show that the RECAP reweighting rule arises as the closed-form solution of an entropy-regularized priority allocation problem. At iteration t, let:

s(t) = s(1t),...,s(Kt) ∈ RK

denote the per-objective priority scores. In RECAP, we define s(kt) = c(kt) + i(kt), where c(kt) measures the recent convergence behavior of objective k, and i(kt) captures its instability. Rather than manually assigning weights to those objectives, we interpret the weights as an allocation vector:

p ∈ ∆K := p ∈ RK≥0 :

K

pk = 1 .

k=1

Given a temperature parameter T > 0, the entropy-regularized priority allocation objective has the following formula:

K

Φt(p) = ⟨p,s(t)⟩ + TH(p), where H(p) = −

k=1

pk log pk,

We consider the entropy-regularized priority allocation problem: max

Φt(p).

p∈∆K

The linear term assigns higher mass to objectives with larger RECAP priority scores, while the entropy term prevents degenerate allocation to a single objective and encourages smoother mixtures. The temperature T controls this trade-off: smaller T yields sharper allocations, whereas larger T approaches uniform weighting. The following result characterizes the solution.

Theorem 1 (Entropy-regularized priority allocation). For any T > 0 and any score vector s(t) ∈ RK, this optimization problem has a unique maximizer p(t),⋆ ∈ ∆K, given by

exp s(kt)/T

p(kt),⋆ =

, k = 1,...,K.

K j=1 exp s(jt)/T

The proof is deferred to Appendix A.1. This theorem shows that the softmax form used in RECAP is not merely a heuristic normalization, it is an entropy-regularized allocation problem over objectives.

#### 5 Experiments

##### 5.1 Experimental Settings

To evaluate our proposed approach, we conduct experiments on two base models, Qwen2.5-VL-3B and Qwen2.5-VL-7B, across two complementary setups at different scales. RLVR-Only Setting: This smaller setup follows the experimental configuration of Liang et al. (2025), focusing on domain-specific RLVR, which serves as an upper-bound baseline for static data-mixture approaches and as our closest baseline. The Qwen2.5-VL-3B model is trained until data from a particular domain is exhausted. Since Liang et al. (2025) enforces binary (0–1) rewards across all domains and thus cannot directly handle non-RL tasks, we further extend our evaluation to a Hybrid Setting: To bridge RL and supervised training paradigms, we finetune Qwen2.5-VL-7B under a larger mixed-objective regime that combines RLVR with SFT-style training. Specifically, we use

ThinkLite-VL-70k (Wang et al., 2025c) while jointly replaying perception-oriented datasets such as RefCOCO (Kazemzadeh et al., 2014) and LLaVA-OneVision OCR (Li et al., 2024).

To isolate the effect of replay and dynamic reweighting and for ease of implementation, we uniformly sample across data sources by default and reweight only the objectives of interest. Unless otherwise noted, we disable the reference-KL penalty to disentangle the effectiveness of regularization approaches (Li & Hoiem, 2017) from that of our replay mechanism. We also include a comparison with this regularization-based approach in our list of established baselines for continual learning below:

- • Reasoning-only: We train only on the target reasoning task with fixed reward weights and no replay. This baseline represents standard task-specific RLVR finetuning without any explicit mechanism for preserving general capabilities.
- • PropMix: We include general-capability data during finetuning and sample domains in proportion to their source-dataset sizes. Objective losses are not reweighted.
- • Uniform: We sample data uniformly across domains while keeping all objective weights fixed.
- • Coreset: We replay a size-limited subset of general-capability data, set to half of the reasoning-data volume in our setup, following standard coreset-style replay methods (Rebuffi et al., 2017; Chaudhry et al., 2019).
- • LwF: We sample data uniformly across domains and add KL regularization with coefficient β = 0.01. We refer to this variant as LwF because it follows the principle of Learning without Forgetting (Li & Hoiem, 2017) by constraining the updated model to remain close to a reference policy.

For context, we also include representative open-source vision-language models specializing in reasoning that are derived from the corresponding base models in each experiment. We list them here for easier benchmarking and do not aim to outperform them, as those models often undergo many complex training pipelines. Models are evaluated with LMMS-Eval (Zhang et al., 2024a).

##### 5.2 Experimental Results

According to Table 1, RL training on the reasoning domain consistently improves base-model performance on both reasoning and perception benchmarks. On SCIENCEQA in particular, RL lifts the Qwen2.5-VL-3B score from 6 to 60. On this benchmark, our proposed method even outperforms the comparison open-source reasoning models. We consider MoDoMoDo the upper-bound approach among static data-mixture approaches because (i) it has access to target-task performance during finetuning, which requires rerunning experiments if new target tasks are introduced, and (ii) it trains multiple proxy models of the same size as the baseline models to learn test performance as a function of the mixing ratio, which is computationally expensive,

- Table 1 Benchmark results in the RLVR-only setting. We report accuracy scores on six benchmarks, where the MoDoMoDo baseline is trained to maximize performance. For this table only, we use the rule-based evaluator on the MathVista dataset instead of "gpt-3.5-turbo" to align with Liang et al. (2025).

Model SAT ScienceQA MathVista (mini) ChartQA InfoVQA MMMU Open-source reasoning baselines

VLAA-Thinker-3B 49.38 14.63 30.4 45.84 30.81 32.22 MM-R1-MGT-PerceReason 50.83 34.21 33.4 44.88 61.42 40.22 Ocean_R1_3B_Instruct 59.49 68.72 38.7 54.00 38.02 40.89 Qwen2.5VL-3b-RLCS 24.12 21.32 17.2 3.32 10.86 27.11 vision-grpo-qwen-2.5-vl-3b 50.57 4.17 32.4 67.80 58.29 37.22 Qwen2.5-VL-3B-Instruct-GRPO-deepmath 34.70 45.27 32.3 70.24 49.75 39.11

Qwen2.5-VL-3B and our variants

Base model 43.98 6.20 23.6 43.88 32.02 38.67 Uniform 44.55 64.85 32.4 69.68 58.30 39.44 MoDoMoDo 49.95 65.74 32.2 70.40 59.88 39.11 RECAP 55.19 71.59 33.2 70.40 60.78 42.44

especially in the context of reinforcement learning. Even after selecting an “optimal” mixture, the method still depends on hand-tuned reward weights (e.g., doubling accuracy and IoU relative to formatting rewards). These trade-off coefficients are also set differently in prior work without clarification, which limits generality.

- Table 2 Benchmark results in large hybrid setting. We report accuracy scores (higher is better) on nine perception and reasoning benchmarks. Rows above the break are open-source reasoning models with different backbones; the lower block compares variants finetuned from the same Qwen2.5-VL-7B base model. Bold = best; underline = second best within the Qwen2.5-VL-7B family.

Model LISA MMMU-PRO AI2D MathVista MathVision MathVerse MMBench VizWiz OCRBenchv2 Open-source reasoning baselines

VLAA-Thinker-7B 63.14 26.30 75.45 63.90 11.18 29.87 75.95 47.57 40.23 Vision-R1-7B 47.30 26.76 0.00 61.80 18.75 23.32 69.46 53.12 24.63 OpenVLThinker-7B 42.73 21.79 59.94 59.10 5.59 19.26 71.53 52.89 28.30

Qwen2.5-VL-7B and our variants

Base model 65.13 25.55 67.62 61.70 9.54 26.29 71.82 50.82 39.49 LwF 65.08 29.59 73.93 63.90 18.42 33.98 73.11 53.12 39.56 PropMix 66.80 31.39 75.32 63.40 21.05 34.75 73.54 57.05 37.60 Uniform 65.18 31.91 76.43 65.60 22.13 36.07 75.34 54.05 38.06 Coreset 64.82 31.91 79.92 66.90 23.36 37.58 78.09 63.76 35.49 Reasoning-only 57.58 33.87 74.97 65.50 24.87 40.74 77.84 62.45 38.55 RECAP 67.24 34.15 78.21 66.70 25.11 40.83 78.52 61.97 39.72

Table 2 reports the performance of models finetuned from Qwen2.5-VL-7B across multiple benchmarks in the hybrid setting, showcasing a more general scenario than RLVR-only training. Overall, RECAP achieves the best or runner-up performance across datasets, except for VizWiz, where it still improves base-model performance by more than 10%. Compared with the base model, naively finetuning on the reasoning domain causes significant forgetting, especially on tasks that do not require extensive thinking. For example, finetuning on ThinkLite-VL-70k reduces segmentation ability from 65.13 to 57.58 on LISA. Meanwhile, replaying general data helps preserve performance across all baselines on this dataset (≈ 64). Compared with uniform sampling, LwF achieves similar scene-understanding performance while obtaining lower scores on the reasoning benchmark (e.g., 29.59 vs 31.91 on MMMU-PRO). Similar behavior is observed in (Wang et al., 2025d; Hu et al., 2025a), where this term is removed for more plasticity. Among all baselines, our proposed method obtains the highest segmentation score, boosting base-model performance by more than 2%. This improvement highlights the impact of loss reweighting over the uniform baseline.

Format Reward Earlier In Training

Format Reward

Accuracy Reward

0.8

|baseline ours<br><br>| | | | | |
|---|---|---|---|---|---|
| | | | | | |

0.12

0.4

0.7

0.10

0.6

0.3

0.5

0.08

score

0.4

0.2

0.06

0.3

0.04

0.2

0.1

0.1

0.02

0.0

0.0

0.00

0 100 200 300 400 500

0 10 20 30 40 50 60

0 100 200 300 400 500

step

step

step

- Figure 5 Evolution of format and accuracy rewards on the reasoning domain during training: Training curves for the format reward over full training (left), an early-training zoom (middle), and the accuracy reward (right). While the uniform baseline is better at maximizing the format reward, it falls behind our proposed method later in terms of accuracy, as we prioritize correct solutions over formatting once the model can follow the predefined template. Curves are smoothed with an exponential moving average for readability.

##### 5.3 Ablation Studies

We conduct an ablation study on the large hybrid setting by comparing the accuracy and formatting rewards of our proposed method with the uniform baseline to isolate the effect of our reweighting. The uniform

baseline employs identical hyperparameters, including the data-sampling and model-training pipelines, yet differs from ours only in the loss reweighting mechanism (λk = 1/K). In Figure 5, we present the curves for the format and accuracy rewards during training. In the early phase, the baseline’s format reward increases faster—consistent with format being a low-variance, easy-to-optimize signal—yet a crossover soon appears and our method surpasses it as training progresses (step 40). In contrast, for the accuracy reward, our method opens a growing lead over time (right). This behavior aligns with our scheduler: once the format objective shows fast convergence and low instability, its weight is down-regulated and capacity is reallocated to harder, higher-variance objectives (e.g., accuracy), avoiding over-optimization of formatting while improving task correctness.

We also empirically find that using the same format reward for different domains is suboptimal. We start by examining the approach from Liang et al. (2025), which employs the same thinking reward on every domain, including scene-understanding tasks. In Figure 6, we plot the response length on the segmentation task during training and find that the Qwen2.5-VL model rapidly trims its chain-of-thought and answers the question directly later in training. This behavior suggests that explicit reasoning is unnecessary for such perception tasks and that encouraging long rationales can even be detrimental. We also include qualitative examples in the appendix to show how the model gradually suppresses its reasoning trace during training. Motivated by this observation, in our broader setting, we keep answer-format rewards for perception domains (no thinking) and reserve thinking rewards for tasks that truly benefit from step-by-step reasoning.

Thinking length over time

1200

Thinkinglength(characters)

1000

800

600

400

200

0

0 1000 2000 3000 4000 5000

Sample index

- Figure 6 Thinking length on a segmentation task during finetuning. When a uniform “thinking reward” is applied to all domains, the model quickly learns that long chains of thought are unnecessary for segmentation. The average response length drops from several hundred characters at the start of training to tens (often near zero) later on.

0 1000 2000 3000 4000 5000

Sample index

0

200

400

600

800

1000

1200

Thinkinglength(characters)

Thinking Length Over Time

reasoning-only (67.3) ours (27.3)

Figure 7 Thinking length on a reasoning task during finetuning. We compare a model trained reasoning-only (blue) against our RECAP method (orange). Highlighted curves show the running average thinking length per example, where our method generates only ∼ 27.3 words per question, compared with the baseline (∼ 67.3).

- Figure 7 tracks the length of the generated thinking segment on the reasoning task throughout training. When trained only on reasoning data, the model maintains long chains of thought with high variability. In contrast, mixing general-capability replay with dynamic objective reweighting progressively reduces thinking length and stabilizes variance, converging to concise rationales (60% reduction, 67 → 27 words on average) while preserving accuracy. This shorter reasoning directly improves inference efficiency—fewer generated tokens reduce latency and compute cost—without sacrificing problem-solving quality.

#### 6 Conclusion

In this paper, we investigate forgetting in recent reasoning-focused vision-language models and find that these models exhibit clear forgetting of general knowledge obtained during pretraining. Motivated by this, we propose an approach that replays general data during finetuning and uses a plug-in method to reweight objectives without incurring the additional cost of training external models. On reasoning benchmarks, our proposed method not only preserves general knowledge but also improves target reasoning performance by properly reweighting the rewards for those tasks.

#### Limitations

Our framework RECAP is generic and extends naturally beyond RLVR and SFT to preference- and alignmentbased objectives (Rafailov et al., 2023; Garg et al., 2025; Hong et al., 2024; Ethayarajh et al., 2024) and process reward models (Lightman et al., 2024; Setlur et al., 2024). However, due to constraints on the training datasets available for this work, our empirical evaluation focuses on RLVR and standard SFT settings. We expect RECAP to yield similar gains over uniform or manually tuned baselines with any heterogeneous objective sets, but we leave a comprehensive evaluation across non-RL objectives to future work. In practice, applying our scheduler to non-RL losses does not require an expensive search over coefficients or per-objective normalization due to its magnitude-agnostic nature.

#### Acknowledgments

We would like to thank Riham Mansour and Misha Bilenko for their steadfast leadership and support. We would also like to thank the Meta legal and policy team for their review and approval of this research.

#### References

Josh Achiam, Steven Adler, Sandhini Agarwal, and et al. GPT-4 technical report. arXiv:2303.08774, 2023. Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware

synapses: Learning what (not) to forget. In ECCV, pp. 139–154, 2018. doi: 10.1007/978-3-030-01219-9_9.

Thomas Borsani, Andrea Rosani, Giuseppe Nicosia, and Giuseppe Di Fatta. Gradient similarity surgery in multi-task deep learning. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 95–111. Springer, 2025.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: A strong, simple baseline. In NeurIPS, volume 33, pp. 15920–15930, 2020.

Nikhil Chandak, Shashwat Goel, and Ameya Prabhu. Incorrect baseline evaluations call into question recent llm-rl claims. https://safe-lip-9a8.notion.site/ Incorrect-Baseline-Evaluations-Call-into-Question-Recent-LLM-RL-Claims-2012f1fbf0ee8094ab8ded1953c15a37? pvs=4, 2025. Notion Blog.

Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with A-GEM. In ICLR, 2019.

Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, et al. Reasoning models don’t always say what they think. arXiv preprint arXiv:2505.05410, 2025.

Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In Proceedings of the 35th International Conference on Machine Learning, volume 80, pp. 794–803, 2018.

Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In Advances in Neural Information Processing Systems, volume 33, pp. 2039–2050, 2020.

Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? volume 38, pp. 57654–57689, 2026.

Xu Chu, Xinrong Chen, Guanyu Wang, Zhijie Tan, Kui Huang, Wenyu Lv, Tong Mo, and Weiping Li. Qwen look again: Guiding vision-language reasoning models to re-attention visual information. arXiv:2505.23558, 2025. Introduces BRPO with reflection and visual re-attention to reduce hallucinations.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal of Machine Learning Research, 25(70):1–53, 2024.

Roberto Cipolla, Yarin Gal, and Alex Kendall. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7482–7491, 2018.

Xingyu Dang, Christina Baek, J Zico Kolter, and Aditi Raghunathan. Assessing diversity collapse in reasoning. In Scaling Self-Improving Foundation Models without Human Supervision, 2025.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Daxiang Dong, Hua Wu, Wei He, Dianhai Yu, and Haifeng Wang. Multi-task learning for multiple language translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1723–1732, 2015.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 11198–11201, 2024.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.

Robert M. French. Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3(4):128–135,

1999. doi: 10.1016/S1364-6613(99)01294-2.

Ling Fu, Biao Yang, Zhebin Kuang, Jiajun Song, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, Mingxin Huang, Zhang Li, Guozhi Tang, Bin Shan, Chunhui Lin, Qi Liu, Binghong Wu, Hao Feng, Hao Liu, Can Huang, Jingqun Tang, Wei Chen, Lianwen Jin, Yuliang Liu, and Xiang Bai. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. arXiv preprint arXiv:2501.00321, 2024.

Yuqian Fu, Tinghong Chen, Jiajun Chai, Xihuai Wang, Songjun Tu, Guojun Yin, Wei Lin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. Srft: A single-stage method with supervised and reinforcement fine-tuning for reasoning. arXiv preprint arXiv:2506.19767, 2025.

Shivank Garg, Ayush Singh, Shweta Singh, and Paras Chopra. Ipo: Your language model is secretly a preference classifier. arXiv preprint arXiv:2502.16182, 2025.

Daya Guo and DeepSeek-AI. Deepseek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature,

2025. doi: 10.1038/s41586-025-09422-z. Yihang Guo, Tianyuan Yu, Liang Bai, Yanming Guo, Yirun Ruan, William Li, and Weishi Zheng. Revisit the imbalance optimization in multi-task learning: An experimental analysis. arXiv preprint arXiv:2509.23915, 2025.

Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

Zhezheng Hao, Hong Wang, Haoyang Liu, Jian Luo, Jiarui Yu, Hande Dong, Qiang Lin, Can Wang, and Jiawei Chen. Rethinking entropy interventions in rlvr: An entropy change perspective. arXiv preprint arXiv:2510.10150, 2025.

Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement

learning that matters. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015. Jiwoo Hong, Noah Lee, and James Thorne. Orpo: Monolithic preference optimization without reference model. arXiv

preprint arXiv:2403.07691, 2024. Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290,

- 2025a.

Zhiyuan Hu, Yibo Wang, Hanze Dong, Yuhui Xu, Amrita Saha, Caiming Xiong, Bryan Hooi, and Junnan Li. Beyond’aha!’: Toward systematic meta-abilities alignment in large reasoning models. arXiv preprint arXiv:2505.10554,

- 2025b.

Yipo Huang, Quan Yuan, Xiangfei Sheng, Zhichao Yang, Haoning Wu, Pengfei Chen, Yuzhe Yang, Leida Li, and Weisi Lin. Aesbench: An expert benchmark for multimodal large language models on image aesthetics perception. arXiv preprint arXiv:2401.08276, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Yunjie Ji, Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, and Xiangang Li. Am-thinking-v1: Advancing the frontier of reasoning at 32b scale. arXiv preprint arXiv:2505.08311, 2025.

Ryo Kamoi, Yusen Zhang, Sarkar Snigdha Sarathi Das, Ranran Haoran Zhang, and Rui Zhang. Visonlyqa: Large vision language models still struggle with visual perception of geometric information. In Conference on Language Modeling (COLM), 2025.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Alessandro Moschitti, Bo Pang, and Walter Daelemans (eds.), Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 787–798, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1086.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European conference on computer vision, pp. 235–251. Springer, 2016.

Diederik P Kingma. Adam: A method for stochastic optimization. The third International Conference on Learning Representations, 2015.

James Kirkpatrick, Razvan Pascanu, and Neil et al. Rabinowitz. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017. doi: 10.1073/pnas.1611835114.

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024a.

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation

via large language model. pp. 9579–9589, 2024b. Nathan Lambert. Reinforcement learning from human feedback. arXiv preprint arXiv:2504.12501, 2025. Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and

Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. Bo Li, Kaichen Zhang, and Andrés Marafioti. Multimodal open r1. https://github.com/EvolvingLMMs-Lab/

open-r1-multimodal, 2025a. Accessed: 2025-02-08.

Chunyi Li, Jianbo Zhang, Zicheng Zhang, Haoning Wu, Yuan Tian, Wei Sun, Guo Lu, Xiongkuo Min, Xiaohong Liu, Weisi Lin, et al. R-bench: Are your large multimodal model robust to real-world corruptions? IEEE Journal of Selected Topics in Signal Processing, 2025b.

Zhizhong Li and Derek Hoiem. Learning without forgetting. In ECCV, pp. 614–629, 2016. doi: 10.1007/ 978-3-319-46493-0_37.

Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

Yiqing Liang, Jielin Qiu, Wenhao Ding, Zuxin Liu, James Tompkin, Mengdi Xu, Mengzhou Xia, Zhengzhong Tu, Laixi Shi, and Jiacheng Zhu. Modomodo: Multi-domain data mixtures for multimodal llm reinforcement learning, 2025.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024.

Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor W Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. Transactions on Machine Learning Research, pp. 2835–8856, 2022.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Advances in Neural Information Processing Systems, volume 34, pp. 18878–18890, 2021a.

Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu. Famo: Fast adaptive multitask optimization. In Advances in Neural Information Processing Systems, volume 36, pp. 57226–57243, 2023.

Haotian Liu et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision (ECCV), 2024a.

Liyang Liu, Yi Li, Zhanghui Kuang, Jing-Hao Xue, Yimin Chen, Wenming Yang, Qingmin Liao, and Wayne Zhang. Towards impartial multi-task learning. In International Conference on Learning Representations, 2021b.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025a.

Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1871–1880, 2019.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024b.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Zifan Liu et al. Measuring multimodal mathematical reasoning with the math-vision dataset. In Advances in Neural Information Processing Systems (NeurIPS), 2024c.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In NeurIPS, pp.

6467–6476, 2017. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. Xinyue Lou, You Li, Jinan Xu, Xiangyu Shi, Chi Chen, and Kaiyu Huang. Think in safety: Unveiling and mitigating

safety alignment collapse in multimodal large reasoning model. arXiv preprint arXiv:2505.06538, 2025.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022a.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022b.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

U.-V. Marti and H. Bunke. The IAM-database: an english sentence database for offline handwriting recognition. International Journal on Document Analysis and Recognition, 5(1):39–46, 2002.

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of ACL, 2022.

Minesh Mathew, Viraj Bagal, Rubèn Pérez Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In WACV, 2022.

Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of Learning and Motivation, volume 24, pp. 109–165. Academic Press, 1989. doi: 10.1016/ S0079-7421(08)60536-8.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Greg Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training. In International Conference on Learning Representations (ICLR), 2018.

Ankush Mishra, Karteek Alahari, and C. V. Jawahar. Scene text recognition using higher order language priors. In British Machine Vision Conference (BMVC), 2012.

Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. In Proceedings of the 39th International Conference on Machine Learning (ICML), 2022.

Long Ouyang, Jeff Wu, Xu Jiang, and et al. Training language models to follow instructions with human feedback. In NeurIPS, 2022.

Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, and Juanzi Li. Verif: Verification engineering for reinforcement learning in instruction following. arXiv preprint arXiv:2506.09942, 2025.

Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev, Mislav Balunović, Nikola Jovanović, and Martin Vechev. Proof or bluff? evaluating llms on 2025 usa math olympiad. arXiv preprint arXiv:2503.21934, 2025.

Hoang Phan, Ngoc Tran, Trung Le, Toan Tran, Nhat Ho, and Dinh Phung. Stochastic multiple target sampling gradient descent. Advances in neural information processing systems, 35:22643–22655, 2022.

Hoang Phan, Lam Tran, Quyen Tran, Ngoc Tran, Tuan Truong, Qi Lei, Nhat Ho, Dinh Phung, and Trung Le. Beyond losses reweighting: Empowering multi-task learning via the generalization perspective. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2440–2450, 2025.

Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. Maximizing confidence alone improves reasoning. arXiv preprint arXiv:2505.22660, 2025.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. Sat: Spatial aptitude training for multimodal language models.

- arXiv preprint arXiv:2412.07755, 3, 2024a.

Arijit Ray, Jiafei Duan, Reuben Tan, Dina Bashkirova, Rose Hendrix, Kiana Ehsani, Aniruddha Kembhavi, Bryan A Plummer, Ranjay Krishna, Kuo-Hao Zeng, et al. Sat: Spatial aptitude training for multimodal language models.

- arXiv preprint arXiv:2412.07755, 3, 2024b.

Sylvestre Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. iCaRL: Incremental classifier and representation learning. In CVPR, pp. 5533–5542, 2017. doi: 10.1109/CVPR.2017.587.

David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Greg Wayne. Experience replay for continual learning. In NeurIPS, pp. 350–360, 2019.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, et al. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. arXiv preprint arXiv:2504.13914, 2025.

Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, and Anton Konushin. Independent component alignment for multi-task learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20083–20093, 2023.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for llm reasoning. arXiv preprint arXiv:2410.08146, 2024.

Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. Rewarding progress: Scaling automated process verifiers for LLM reasoning. In The Thirteenth International Conference on Learning Representations, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity. arXiv preprint arXiv:2506.06941, 2025.

Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: A dataset for image captioning with reading comprehension. In European Conference on Computer Vision (ECCV), 2020.

A. Singh et al. TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text. arXiv preprint arXiv:2105.05486, 2021.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.

Core Team, Zihao Yue, Zhenru Lin, Yifan Song, Weikun Wang, Shuhuai Ren, Shuhao Gu, Shicheng Li, Peidian Li, Liang Zhao, Lei Li, Kainan Bao, Hao Tian, Hailin Zhang, Gang Wang, Dawei Zhu, Cici, Chenhong He, Bowen Ye, Bowen Shen, Zihan Zhang, Zihan Jiang, Zhixian Zheng, Zhichao Song, Zhenbo Luo, Yue Yu, Yudong Wang, Yuanyuan Tian, Yu Tu, Yihan Yan, Yi Huang, Xu Wang, Xinzhe Xu, Xingchen Song, Xing Zhang, Xing Yong, Xin Zhang, Xiangwei Deng, Wenyu Yang, Wenhan Ma, Weiwei Lv, Weiji Zhuang, Wei Liu, Sirui Deng, Shuo Liu, Shimao Chen, Shihua Yu, Shaohui Liu, Shande Wang, Rui Ma, Qiantong Wang, Peng Wang, Nuo Chen, Menghang Zhu, Kangyang Zhou, Kang Zhou, Kai Fang, Jun Shi, Jinhao Dong, Jiebao Xiao, Jiaming Xu, Huaqiu Liu, Hongshen Xu, Heng Qu, Haochen Zhao, Hanglong Lv, Guoan Wang, Duo Zhang, Dong Zhang, Di Zhang, Chong Ma, Chang Liu, Can Cai, and Bingquan Xia. Mimo-vl technical report, 2025.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025a.

Shibo Wang and Przemyslaw Kanwar. Bfloat16: The secret to high performance on cloud tpus. Google Cloud Blog, 2019. URL https://cloud.google.com/blog/products/ai-machine-learning/ bfloat16-the-secret-to-high-performance-on-cloud-tpus.

Teng Wang, Zhangyi Jiang, Zhenqi He, Shenyang Tong, Wenhan Yang, Yanan Zheng, Zeyu Li, Zifan He, and Hailei Gong. Towards hierarchical multi-step reward models for enhanced reasoning in large language models. arXiv preprint arXiv:2503.13551, 2025b.

Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025c.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025d.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 2023.

Yang Yao, Xuan Tong, Ruofan Wang, Yixu Wang, Lujundong Li, Liang Liu, Yan Teng, and Yingchun Wang. A mousetrap: Fooling large reasoning models for jailbreak with chain of iterative chaos. arXiv preprint arXiv:2502.15806, 2025a.

Zijun Yao, Yantao Liu, Yanxu Chen, Jianhui Chen, Junfeng Fang, Lei Hou, Juanzi Li, and Tat-Seng Chua. Are reasoning models more prone to hallucination? arXiv preprint arXiv:2505.23646, 2025b.

En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yana Wei, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, pp. 5824–5836, 2020.

Ye Yuan, Xiao Liu, Wondimu Dikubab, Hui Liu, Zhilong Ji, Zhongqin Wu, and Xiang Bai. Syntax-aware network for handwritten mathematical expression recognition, 2022.

Xin Yue et al. Mmmu: A massive multi-discipline multimodal understanding benchmark for expert agi. arXiv preprint arXiv:2311.16502, 2023.

Xin Yue et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017. doi: 10.1073/pnas.1611835114. Supplementary/alternate venue listings exist.

Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, and Guorui Zhou. Rlep: Reinforcement learning with experience replay for llm reasoning. arXiv:2507.07451, 2025. Experience replay improves stability and final accuracy on AIME/AMC.

Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024a.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? pp. 169–186, 2024b.

Ruxin Zheng, Yifei Li, Xiang Li, Chong Chen, et al. Secrets of rlhf in large language models part i: Ppo. arXiv:2307.04964,

2023. Analyzes PPO variants and shows PPO-ptx mitigates knowledge decline. Daniel M. Ziegler, Nisan Stiennon, Jeff Wu, and et al. Fine-tuning language models from human preferences. arXiv:1909.08593, 2019.

## Appendix

#### A Experimental setup

In this section, we provide detailed statistics of the training datasets used in our experiments, along with implementation details and additional experimental results.

##### A.1 Proof of Entropy-Regularized Priority Allocation

We provide the proof that the RECAP reweighting rule is the unique solution of an entropy-regularized priority allocation problem.

- Lemma 1 (Existence of an optimizer). For any T > 0 and any score vector s(t) ∈ RK, the optimization problem in Equation 4 admits at least one maximizer p⋆ ∈ ∆K. Proof. The feasible set ∆K is compact. The map

p  → ⟨p,s(t)⟩

is continuous, and the entropy H(p) is continuous on ∆K under the convention 0log 0 = 0. Therefore, Φt is continuous on a compact set. By the extreme value theorem, Φt attains its maximum on ∆K. Hence, there exists at least one maximizer p⋆ ∈ ∆K.

| |
|---|

- Lemma 2 (Interior optimality). Any maximizer p⋆ of Equation 4 lies in the interior of the simplex, i.e., p⋆k > 0, k = 1,...,K.

Proof. Assume for contradiction that p⋆ is a maximizer and that p⋆j = 0 for some j. Since Kk=1 p⋆k = 1, there exists some index i ̸= j such that p⋆i > 0. For ε ∈ (0,p⋆i ), define the feasible perturbation

p(ε) = p⋆ + εej − εei, where ei and ej denote the standard basis vectors. Then p(ε) ∈ ∆K. The change in objective value is

Φt(p(ε)) − Φt(p⋆) = ε s(jt) − s(it) + T H(p(ε)) − H(p⋆) . Only coordinates i and j change, so the entropy difference is

H(p(ε)) − H(p⋆) = −εlog ε − (p⋆i − ε)log(p⋆i − ε) + p⋆i log p⋆i . Using a first-order expansion around p⋆i > 0, we have

−(p⋆i − ε)log(p⋆i − ε) + p⋆i log p⋆i = ε(log p⋆i + 1) + O(ε2). Therefore,

H(p(ε)) − H(p⋆) = −εlog ε + ε(log p⋆i + 1) + O(ε2). Substituting this into the objective difference gives

Φt(p(ε)) − Φt(p⋆) = −Tεlog ε + ε s(jt) − s(it) + T(log p⋆i + 1) + O(ε2).

As ε → 0+, the term −εlog ε is positive and dominates all linear O(ε) terms. Since T > 0, it follows that for sufficiently small ε > 0,

Φt(p(ε)) > Φt(p⋆),

which contradicts the optimality of p⋆. Hence, no coordinate of p⋆ can be zero, and therefore p⋆k > 0 for all k.

| |
|---|

- Proposition 1 (First-order optimality conditions). Let p⋆ be a maximizer of Equation 4. Then there exists a scalar ν ∈ R such that, for every k = 1,...,K,

s(kt) − T(log p⋆k + 1) + ν = 0.

Proof. By Lemma Equation 2, any maximizer p⋆ lies in the interior of the simplex. Hence, the non-negativity constraints are inactive at p⋆, and we only need to enforce the equality constraint Kk=1 pk = 1. The Lagrangian is

K

K

K

pks(kt) − T

L(p,ν) =

pk − 1 .

pk log pk + ν

k=1

k=1

k=1

Stationarity at p⋆ requires

∂L ∂pk

(p⋆,ν) = 0, k = 1,...,K. Since

∂ ∂pk

(−pk log pk) = −(log pk + 1), we obtain

s(kt) − T(log p⋆k + 1) + ν = 0,

which proves the claim. Theorem 2 (Closed-form solution of entropy-regularized priority allocation). For any T > 0, the optimization problem in Equation 4 has a unique maximizer p(t),⋆ ∈ ∆K, given by

| |
|---|

exp s(kt)/T

p(kt),⋆ =

, k = 1,...,K.

K j=1 exp s(jt)/T

Proof. Existence follows from Lemma 1. By Lemma 2 and Proposition 1, any maximizer p⋆ satisfies

s(kt) − T(log p⋆k + 1) + ν = 0 for some scalar ν ∈ R. Rearranging gives

s(kt) + ν

log p⋆k =

T − 1. Exponentiating both sides yields

s(kt) T

ν T − 1 .

p⋆k = exp

exp

The factor exp(ν/T − 1) is independent of k. Therefore,

s(kt) T

p⋆k ∝ exp

.

Enforcing the simplex constraint Kk=1 p⋆k = 1 gives

p⋆k =

which proves the closed form. It remains to show uniqueness. The map

exp s(kt)/T

,

K j=1 exp s(jt)/T

p  → ⟨p,s(t)⟩

is linear, and the entropy map p  → H(p) is strictly concave on the interior of ∆K. Since T > 0, Φt is strictly concave on the interior of ∆K. By Lemma 2, every maximizer lies in the interior. Therefore, there cannot be two distinct maximizers, and the maximizer is unique.

| |
|---|

- Proposition 2 (RECAP reweighting rule). Define the RECAP loss weights by

λ(kt) := Kp(kt),⋆. Then

K exp s(kt)/T

λ(kt) =

, k = 1,...,K,

K j=1 exp s(jt)/T

and the weights satisfy the mean-normalization property

K

1 K

λ(kt) = 1.

k=1

Proof. Substituting the closed-form optimizer from Theorem 2 into the definition λ(kt) := Kp(kt),⋆ gives

exp s(kt)/T

λ(kt) = K

K j=1 exp s(jt)/T

K exp s(kt)/T

=

.

K j=1 exp s(jt)/T

Moreover,

K

1 K

1 K

λ(kt) =

k=1

K

K

Kp(kt),⋆ =

p(kt),⋆ = 1,

k=1

k=1

where the last equality follows from p(t),⋆ ∈ ∆K. This proves the proposition.

| |
|---|

##### A.2 Data statistics and implementation details

- Table 3 reports the full statistics of the training corpora used in our experiments. For LLaVA-OneVision-OCR, we extract OCR-focused subsets from the official LLaVA-OneVision release (Li et al., 2024): IIIT5K (Mishra et al., 2012), HME100K (Yuan et al., 2022), IAM (Marti & Bunke, 2002), TextCaps (Sidorov et al., 2020), and TextOCR (Singh et al., 2021), alongside release-provided synthetic/curated subsets (rendered_text, k12_printing, chrome_writing). These images are resized so that the longer side is ≤ 512px while preserving aspect ratio to mitigate out-of-memory errors without altering task semantics.

- Table 3 Data statistics for each data source. We present the original volume of data (# samples).

Dataset Domain Answer Type Rewards/Objectives # samples

RefCOCO (Kazemzadeh et al., 2014) Referring Expression Comprehension 2D Bounding Box IoU, Answer Format 321327 LLaVA-OneVision-OCR (Li et al., 2024) Scene Text-Centric Visual Question Answering Natural Language Next Token Prediction 66468 ThinkLite-VL-70k (Wang et al., 2025c) Math Reasoning & Natural Image/Chart Understanding Natural Language Acc, Thinking Format 69997 LISA-train (Lai et al., 2024b) Referring Expression 2D Bounding Box IoU, Thinking Format 1326 GeoQAV (Li et al., 2025a) Math Visual Question Answering Multiple Choice Acc, Thinking Format 1969 SAT-train (Ray et al., 2024b) Spatial Visual Question Answering Natural Language Acc, Thinking Format 15000 ScienceQA-train (Lu et al., 2022b) Science Visual Question Answering Multiple Choice Acc, Thinking Format 6218

We optimize with GRPO and SFT losses using AdamW (Loshchilov & Hutter, 2017) (β1=0.9, β2=0.999, ε=10−8). The learning rate follows a linear schedule: 10% warm-up to ηmax=1×10−6, then linear decay to 0. Window size W and temperature T are set to 10 and 5.0, respectively, in our experiments. Due to the large computational requirements of RL training, we find that setting T = 5 and α = 0.5 works reasonably well in the RLVR-only setting. For simplicity, we keep this configuration for the hybrid setup and do not perform additional hyperparameter tuning in the large-scale setting. All runs use bfloat16 precision (Wang & Kanwar, 2019; Micikevicius et al., 2018) and FlashAttention kernels (Dao et al., 2022) for memory- and throughput-efficient attention. We enable thinking mode on reasoning tasks by enforcing structured traces (i.e.,

wrapping thoughts in <think>...</think>), which has been shown to improve reasoning and transparency (Hu et al., 2025b; Xie et al., 2025; Chen et al., 2025).

For the larger hybrid setting, each model is trained for 500 steps on 8 GPUs using data parallelism, with a per-device batch size of 1, 2 gradient accumulation steps (effective batch size 16), and 4 rollouts per prompt (64 rollouts per optimizer step). We evaluate our models on a broad suite of widely used VLM benchmarks spanning general multimodal understanding, visual reasoning, math-in-vision, OCR, and accessibility: LISA (Lai et al., 2024a), MMMU-Pro (Yue et al., 2024), AI2D (Kembhavi et al., 2016), MathVista (Lu et al., 2023), MathVision (Liu et al., 2024c), MathVerse (Zhang et al., 2024b), MMBench (Liu et al., 2024a), VizWiz (Gurari et al., 2018), and OCRBench v2 (Fu et al., 2024). For the smaller setup, we also follow Liang et al. (2025) and train on 8 GPUs using data parallelism, with a per-device batch size of 2 and 4 rollouts per prompt, and then evaluate the models on SAT (Ray et al., 2024a), ScienceQA (Lu et al., 2022a), MathVista (Lu et al., 2023), ChartQA (Masry et al., 2022), InfoVQA (Mathew et al., 2022), and MMMU (Yue et al., 2023). Our evaluation protocol closely follows LMMS-Eval (Zhang et al., 2024a) and VLMEvalKit (Duan et al., 2024).

##### Evaluation prompt

Non-Thinking: {Question} Output the answer in <answer> </answer> tags. Thinking: {Question} Output the thinking process in <think> </think> tags and the final answer (option) in <answer> </answer> tags.

We provide the description of RECAP in Algorithm 1 and its pseudocode in Algorithm 2. To use it, one first computes the task losses, calls update to update the task weighting, and then obtains the weighted loss via get_weighted_loss to perform standard backpropagation. For typical settings (e.g., Qwen-3B/7B, K < 10 objectives, window size W = 10), our method introduces only Θ(KW) extra scalar operations and Θ(KW) memory, which is negligible compared to the Θ 1011 − Θ 1012 FLOPs per step of the underlying model; in practice, we observed no measurable slowdown.

- Algorithm 1 Replay-Enhanced CApability Preservation (RECAP).

Require: Base parameters θ(0); domain list {Dn}Nn=1 comprising general domains {D1G,...,DMG } and reasoning

domains {D1R,...,DLR}; objectives {Lk}Kk=1; window size W; temperature T; total iterations Tmax

- 1: Initialize λ(0)k ← 1 for all k
- 2: Initialize loss history buffers Bk of length 2W for each objective k
- 3: for t = 1 to Tmax do
- 4: Sample mini-batches from reasoning and replay data on each domain Dn
- 5: Compute per-domain, per-objective losses ℓ(n,kt) (θ(t))
- 6: Compute per-objective averaged losses

L(kt) ←

1 N

N

n=1

ℓ(n,kt) (θ(t)), ∀k

- 7: for k = 1 to K do
- 8: Push L(kt) into buffer Bk (FIFO)
- 9: end for
- 10: if t ≥ 2W then
- 11: for k = 1 to K do
- 12: Compute current-window mean: µ(kt) ← W1 ts=t−W+1 L(ks) and previous-window mean µ˜(kt) ← 1

W

t−W s=t−2W+1 L(ks)

- 13: Compute the instability:

σk(t) =

1 2W − 1

t

s=t−2W+1

L(ks) − µ(kt) 2

- 14: Compute the convergence rate c(kt) ← µ˜

(t) k

µ(kt)

, the inverse signal-to-noise ratio i(kt) ← σ

(t) k

µ(kt)+˜µ(kt)

and the relative priority between domains:

s(kt) ← c(kt) + i(kt)

- 15: end for
- 16: Calculate softmax weights:

λ(kt) ←

K exp s(kt)/T K j=1 exp s(jt)/T

, ∀k

- 17: else
- 18: λ(kt) ← 1 for all k
- 19: end if
- 20: Compute final objective:

L(t)(θ(t)) ←

1 K

K

k=1

λ(kt)L(kt)

- 21: Update parameters: θ(t+1) ← θ(t) − η∇θL(t)(θ(t))
- 22: end for

- Algorithm 2 Implementation of our proposed method in PyTorch-like pseudocode class RECAP: def __init__(self, num_objectives, window_size, T=5.0):

# num_objectives (K,) number of objectives (rewards / losses) # window_size W, length of each averaging window self.K = num_objectives self.W = window_size self.T = T

# loss_history[k] stores a list of recent scalar losses for objective k self.loss_history = [deque(maxlen=2 * self.W) for _ in range(num_objectives)]

# current weights λ (no grad, treated as buffer) self.lambdas = torch.ones(self.K) self.step = 0

def get_weighted_loss(self, losses): # losses (K,) tensor of per-objective losses L(kt) # returns L(t) = K1 k λ(kt)L(kt) weights = self.lambdas.detach() loss = (weights * losses).mean() return loss

def update(self, losses): # losses (K,) tensor of current per-objective losses (no grad needed) self.step += 1

# append current losses into history (FIFO of length at most 2W) for k in range(self.K):

self.loss_history[k].append(losses[k].detach()) # if not enough history, keep uniform mixing

if self.step < 2 * self.W: self.lambdas = torch.ones(self.K) return

# compute per-objective signals c(kt) and i(kt)

c = torch.zeros(self.K) i = torch.zeros(self.K) for k in range(self.K):

hist = torch.stack(self.loss_history[k]) recent = hist[-self.W:] # current window old = hist[-2*self.W:-self.W] # previous window mu = recent.mean() mu_old = old.mean() sigma = hist.std(unbiased=True)

# c(kt) = µ˜(kt)/µ(kt) c[k] = mu_old / mu

# i(kt) = σk(t)/(µ(kt) + µ˜(kt)) i[k] = sigma / (mu + mu_old)

# priority scores s(kt) = c(kt) + i(kt) s = c + i

# temperature-controlled softmax, normalized so K1 k λk = 1 w = torch.softmax(s / self.T, dim=-1) self.lambdas = self.K * w.detach()

- A.3 Prompts used in our experiments The prompt used for training our model is shown in Figure 8.

| |Example of question| |
|---|---|---|
| | | |

|Question: Julie wrote down how many cups of lemonade she sold in the past 5 days. What is the range of the numbers?<br><br>Prompt: Julie wrote down how many cups of lemonade she sold in the past 5 days. What is the range of the numbers? Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.|
|---|

[Figure 17]

Image Prompt

<think> To find the range of the numbers, we need to follow these steps:

- 1. Identify the highest number in the list.
- 2. Identify the lowest number in the list.

3. Subtract the lowest number from the highest number. The numbers from the table are: 82, 96, 82, 93, 88.

- Step 1: Identify the highest number. The highest number is 96.
- Step 2: Identify the lowest number. The lowest number is 82.

Step 3: Subtract the lowest number from the highest number. 96 - 82 = 14

Therefore, the range of the numbers is 14. </think> <answer>14</answer>

Answer

Ground truth: 14

- Figure 8 Example conversation for finetuning. We ask the model to produce its reasoning chain within <think> . . . </think> and the final answer in <answer>. . . </answer> tags.

- A.4 Qualitative examples
- Figure 9 presents responses from models trained either with reasoning data only or with additional general data.

Question:

Which of these oceans does the prime meridian intersect? Choices:

[Figure 18]

- (A) the Pacific Ocean
- (B) the Indian Ocean
- (C) the Atlantic Ocean

<think>The prime meridian is the line of longitude that runs through the Royal Observatory in Greenwich, England. It intersects the Atlantic Ocean, the South Atlantic Ocean, and the Southern Ocean (north of Antarctica) but not the Pacific Ocean, Indian Ocean, or Arctic Ocean, which are not on the given map or intersected by the prime meridian.</think>

Reasoning only:

<answer>the Atlantic Ocean</answer>

image

<think>the Arctic Ocean</think> Ours: <answer>the Arctic Ocean</answer>

Question:

If you select a marble without looking, how likely is it that you will pick a black one? Choices:

[Figure 19]

- (A) unlikely
- (B) probable
- (C) impossible
- (D) certain

<think>There are four marbles in total: three blue marbles and one black marble. The probability of picking a black marble is the number of black marbles divided by the total number of marbles, which is 1/4 or 25%. This probability is less than 50%, so it is not probable, certain, or impossible. Therefore, the likelihood is unlikely.</think> <answer>unlikely</answer>

Reasoning only:

image

<think>There is only one black marble out of four marbles, so the probability of Ours: picking it is low.</think> <answer>unlikely</answer>

[Figure 20]

Question: Is the dotted line a line of symmetry?

think>Considering the image, the dotted line does not divide the popsicle into two mirror images. The popsicle has a symmetrical shape, but the line of symmetry must divide it into identical halves. The dotted line does not do this, as it passes through the front of the popsicle and does not reflect the other side in a symmetrical manner. Therefore, the dotted line is not a line of symmetry.</think>

Reasoning only:

<answer>no</answer>

<think>no</think> Ours: <answer>no</answer>

image

[Figure 21]

Question:

A concession stand worker at the movie theater looked up how many bags of popcorn were sold in the past 5 days. What is the range of the numbers?

<think>The range is the difference between the highest and lowest numbers. The highest number of bags sold is 7 (Saturday) and the lowest is 1 (Friday). So, the range is 7 - 1 = 6.</think> <answer>6</answer>

Reasoning only:

<think>7 - 1 = 6</think> Ours: <answer>6</answer>

image

- Figure 9 Generated answers from the reasoning-only finetuned model and our model. While the reasoning-only finetuned model generates long reasoning chains for simple visual questions, our model produces more concise answers, especially on non-mathematical tasks.

##### A.5 Evolution of rollouts during training with the thinking reward

Below, we show generated responses from the model at different points during training, where we observe the finetuned model gradually shortening its chain of thought on image-understanding tasks to provide final answers directly. This motivates our use of non-thinking rewards on tasks that do not require reasoning, instead of employing the same thinking-format reward for every task (Liang et al., 2025).

###### Example 1: rollout # 0

Prompt: Please provide the bounding box coordinate of the region this sentence describes: flicking tail. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows: <answer>[x1, y1, x2, y2]</answer> Answer: To determine the bounding box coordinates for the zebra that is flicking its tail, we need to follow these steps:

- 1. Identify the zebra in the image that is flicking its tail.
- 2. Observe the position and direction of the zebra’s tail.
- 3. Define a bounding box that encapsulates the entire zebra, keeping in mind that the tail should be included in the box. Upon inspection:

- - The zebra on the right is flicking its tail, as evident from the blurred tail motion and the position of the tail.
- - The bounding box should encompass the entire zebra, including its tail. <answer>[291, 41, 432, 315]</answer>

###### Example 2: rollout # 500

Prompt: Please provide the bounding box coordinate of the region this sentence describes: second from right. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows: <answer>[x1, y1, x2, y2]</answer>

Answer: <think>To determine the bounding box for the second from right, I first identify the rightmost object, which is the one on the far right, identified by its position and bounding box. The second from right is the one to the left of the rightmost object. By visual inspection, the bounding box for the second from right should start after the farthest object and extend leftward.</think>

<answer>[237, 72, 374, 328]</answer>

Example 3: rollout # 1000

Prompt: Please provide the bounding box coordinate of the region this sentence describes: boat blocked by cow stomach. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows: <answer>[x1, y1, x2, y2]</answer> Answer: <think> The sentence refers to the boat that is partially covered by the cow’s stomach. This boat seems to be the one with the green and white colors.</think> <answer>[327,21,631,280]</answer>

Example 4: rollout # 5000

Prompt: Please provide the bounding box coordinate of the region this sentence describes: guy on left with backpack on. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows: <answer>[x1, y1, x2, y2]</answer> Answer: <think>[114,53,236,378]</think> <answer>[114,53,236,378]</answer>

##### A.6 Reward dynamics during training

Given the multi-objective nature of the problem, one might apply existing methods in the multi-task learning literature (Guo et al., 2025) to reweight different objectives and rewards. In practice, this is difficult for two reasons. First, computing per-objective gradients is prohibitively expensive at LLM scale, especially under reinforcement learning. Second, on-policy RL signals are high-variance and non-stationary (Henderson et al., 2018), making per-iteration statistics unreliable indicators of task progress. As shown in Figure 10, all rewards fluctuate substantially within their [0,1] range, with the standard deviation of the total reward peaking near 0.9 around step ∼ 20. Thus, we propose a method that uses a sliding window, which provides a more robust proxy for understanding convergence behavior.

Next Token Prediction

Answer Format Reward

Thinking Format Reward

1.0

1.0

0.9

0.9

0.8

0.8

0.8

0.6

0.7

0.7

Value

0.6

0.6

0.4

0.5

0.5

0.2

0.4

Raw

Raw

Raw

Running average (w=10) Running average (w=25)

Running average (w=10) Running average (w=25)

Running average (w=10) Running average (w=25)

0.4

0.3

0.0

0 50 100 150 200 250 300

0 50 100 150 200 250 300 350

0 50 100 150 200 250 300

IoU Reward

Accuracy Reward

Total Reward STD

1.0

1.0

0.8

0.8

0.8

| | |
|---|---|
|Raw| |

0.6

0.6

0.6

Value

0.4

0.4

0.4

0.2

0.2

Raw

0.2

Running average (w=10) Running average (w=25)

Running average (w=10) Running average (w=25)

0.0

0.0

0 50 100 150 200 250 300 350 Step

0 50 100 150 200 250 300 Step

0 100 200 300 400 500 Step

- Figure 10 Reward dynamics and variability during RLVR training. Per-step rewards (light traces) and sliding-window means (dark curves) for six metrics: Next-Token Prediction, Answer-Format, Thinking-Format, IoU, Accuracy, and the Total-Reward Standard Deviation (lower-right). Asynchronous convergence and high variance motivate short-horizon statistics for dynamic objective reweighting rather than per-iteration magnitudes.

We also conduct an ablation on the effect of the window size W by increasing it from 10 (our default throughout the experiments) to larger values, up to 25, as shown in Figure 10. Since we train Qwen2.5-VL-7B for 500 iterations, setting W = 25 delays the onset of dynamic reweighting by 50 iterations, according to Algorithms 1 and 2, because our method requires 2W steps of history. By the time reweighting becomes active, some rewards have already entered a near-converged regime, which reduces the usefulness of the convergence-rate term and makes the scheme rely mostly on the instability term. We therefore choose W = 10 as a reasonable compromise between sensitivity and robustness: it accumulates enough information while remaining responsive to the current state of training.

##### A.7 Reward values at the end of training

###### In Figure 11, we plot the coefficients of the five objectives used in the hybrid setup. From these coefficients, we can rank the objectives by how strongly our method focuses on each one, from low to high: format rewards, the IoU reward, next-token prediction (on the OCR task), and reasoning accuracy.

Per-step weights for each objective

0.30

format_answer

format_think

sft

0.28

iou

accuracy

0.26

()RECAPweighttk

0.24

0.22

0.20

0.18

0.16

400 420 440 460 480 500 Training step

###### Figure 11 Evolution of per-objective coefficients. In the last 100 iterations, the coefficient for each objective is relatively consistent, with format rewards receiving the lowest focus while the supervised finetuning objective and accuracy rewards are emphasized due to their instability.

Similar to what we observe in the main paper, Figure 12 shows the final performance of our model and the uniform baseline. Results show near-parity on thinking formatting (<think></think> <answer> </answer>) and direct answer reward (<answer> </answer>) but consistent improvements in reasoning score, IoU, and mean token accuracy (+2.01, +1.11, and +1.40 points, respectively). This aligns with our design goal: once format signals saturate, we down-weight them and shift capacity to harder, higher-variance objectives, improving accuracy while maintaining output format.

Ours vs. Baseline across Metrics

100

Baseline

95.36 95.79

95.69 95.76

Ours

95

90

84.84

83.44

85

score

80

75.15

74.04

75

70

65.97

63.96

65

60

ThinkFormat AnswerFormat ReasoningAcc. IoU MeanTokenAcc.

###### Figure 12 Final performance across metrics. We compare a uniform baseline with our dynamic reweighting. The gains on correctness-oriented metrics indicate that reallocating weight away from saturated format rewards toward harder objectives yields better solutions without sacrificing adherence to templates.

Following the illustrative setup of Navon et al. (2022), we consider a synthetic two-task problem with a shared parameter vector and two scalar objectives. The corresponding Pareto front can be computed analytically and is shown in gray in Figure 13. We benchmark our method against established loss-magnitude-based methods (first row) and gradient-based multi-task learning methods (second row). To mimic the unstable nature of RL training, we inject noise into the first objective, which induces substantial fluctuations for competing methods, whereas our approach remains stable and closely tracks the Pareto front. Additional runtime comparisons in the appendix highlight that our method also achieves favorable wall-clock efficiency, a crucial advantage for large-scale RL training. The two tasks L1(x) and L2(x) are defined on x = (x1,x2)⊤ ∈ R2,

L1(x) = f1(x)g1(x) + f2(x)h1(x) + 3ϵ L2(x) = f1(x)g2(x) + f2(x)h2(x),

where ϵ ∼ N(0,1), and the functions are given by:

- f1(x) = max tanh(0.5x2),0
- f2(x) = max tanh(−0.5x2),0

- g1(x) = log max |0.5(−x1 − 7) − tanh(−x2)|,0.000005 + 6
- g2(x) = log max |0.5(−x1 + 3) − tanh(−x2) + 2|,0.000005 + 6

- h1(x) = (−x1 + 7)2 + 0.1(−x1 − 8)2 /10 − 20
- h2(x) = (−x1 − 7)2 + 0.1(−x1 − 8)2 /10 − 20.

We use five different starting points {(−8.5,7.5),(0,0),(9.0,9.0),(−7.5,−0.5),(9.0,−1.0)}. Those points are optimized with Adam (Kingma, 2015) with a learning rate of 1e-2 for 10000 iterations. To justify our proposed reweighting mechanism, we compare against established multi-task learning techniques, including loss-balancing methods such as UW (Cipolla et al., 2018), DWA (Liu et al., 2019), GradNorm (Chen et al., 2018), RGW (Lin et al., 2022), and FAMO (Liu et al., 2023), as well as gradient-based methods: PCGrad (Yu et al., 2020), CAGrad (Liu et al., 2021a), GradDrop (Chen et al., 2020), MGDA (Dong et al., 2015), IMTL (Liu et al., 2021b), MT-SGD (Phan et al., 2022), Nash-MTL (Navon et al., 2022), FS-MTL (Phan et al., 2025), Aligned-MTL (Senushkin et al., 2023), and SAM-GS (Borsani et al., 2025). Their convergence behavior is presented in Figure 13, where RECAP improves across all initialized solutions compared with other MTL methods. While gradient-based MTL methods such as CAGrad and Nash-MTL do not depend on the initial solutions, they suffer from slow convergence and higher per-step computational cost compared to RECAP.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

(a) First objective (b) Second objective (c) Scale-invariant (d) Uncertainty Weighting

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

(e) Linear scalarization (f) Random loss reweighting (g) Dynamic Weight Average (h) FAMO

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

(i) Grad Drop (j) MGDA (k) MT-SGD (l) PCGRAD

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

(m) CAGRAD (n) IMTL (o) Nash-MTL (p) FairGrad

[Figure 38]

[Figure 39]

[Figure 40]

(q) AlignedMTL (r) SAMGS (s) RECAP

- Figure 13 A modified illustrative two-task example from (Navon et al., 2022) showing the convergence of comparison methods from different initialization points (black dots •). Each optimization trajectory is colored from yellow to red. The bold gray line represents the Pareto front. Overall, introducing noise in the first objective causes instability in many MTL methods. Methods that leverage per-step loss-magnitude statistics, such as FAMO, DWA, and UW, exhibit considerable instability during convergence from different30 initializations.

###### Table 4 Evaluation results on NYUv2 scene understanding. Test performance for three tasks: semantic segmentation, depth estimation, and surface normals. We highlight the best loss-magnitude-based MTL method in bold and the best gradient-based MTL method with an underscore.

Segmentation Depth Surface Normal Complexity

Angle Distance ↓ Within t◦ ↑ ∆m% ↓ Mean Median 11.25 22.5 30

mIoU ↑ Pix Acc ↑ Abs Err ↓ Rel Err ↓

STL 38.30 63.76 0.6754 0.2780 25.01 19.21 30.14 57.20 69.15

LS 39.29 65.33 0.5493 0.2263 28.15 23.96 22.09 47.50 61.08 5.59 SI 38.45 64.27 0.5354 0.2201 27.60 23.37 22.53 48.57 62.32 4.39

RLW 37.17 63.77 0.5759 0.2410 28.27 24.18 22.26 47.05 60.62 7.78

Θ(1) DWA 39.11 65.31 0.5510 0.2285 27.61 23.18 24.17 50.18 62.39 3.57 UW 36.87 63.17 0.5446 0.2260 27.04 22.61 23.54 49.05 63.65 4.05 Ours 41.26 66.79 0.5303 0.2203 27.11 22.23 24.64 50.88 64.02 0.77

GradNorm 20.09 64.64 0.7200 0.2800 24.83 18.86 30.8 57.94 69.73 7.22 MGDA 30.47 59.90 0.6070 0.2555 24.88 19.45 29.18 56.88 69.36 1.38 PCGrad 38.06 64.64 0.5550 0.2325 27.41 22.80 23.86 49.83 63.14 3.97

Θ(K) GradDrop 39.39 65.12 0.5455 0.2279 27.48 22.96 23.38 49.44 62.87 3.58 CAGrad 39.79 65.49 0.5486 0.2250 26.31 21.58 25.61 52.36 65.58 0.20 IMTL-G 39.35 65.60 0.5426 0.2256 26.02 21.19 26.2 53.13 66.24 −0.76 FS-MTL 40.42 65.61 0.5389 0.2121∗ 25.03 19.75 28.90 56.19 68.72 −4.77∗

Nash-MTL 40.13 65.93 0.5261 0.2171 25.26 20.08 28.4 55.47 68.15 −4.04

- Table 4 reports the performance of different MTL methods on the real-scene understanding benchmark, which includes one segmentation task and two pixel-level regression tasks. Overall, our method nearly matches the single-task baselines (∆m% ↓≈ 0) while being roughly 3× more efficient in both runtime and memory, and it consistently outperforms all other loss-reweighting methods across all metrics (except Angle Distance Mean, where it is competitive with Uncertainty Weighting). Notably, our approach even surpasses several established gradient-based methods, such as GradNorm, MGDA, PCGRAD, and GradDrop, while remaining roughly three times faster than gradient-based alternatives. We also observe a clear Pareto trade-off: although NashMTL achieves the highest overall relative improvement in ∆m% ↓, it lags behind GradNorm and MGDA on the surface-normal task, whereas these methods incur substantial performance drops on segmentation and depth estimation.

- Figure 14 plots the loss curves for three different objectives, showing stable optimization across all of them. In contrast, our RL rewards are much sparser than in this SFT setting, and the training curves in Figures 4 and 10 exhibit substantially higher fluctuations. This motivates a more robust loss-reweighting mechanism, as relying solely on instantaneous per-step loss values is not sufficiently representative of the underlying learning dynamics or objective progress.

Train Semantic Loss

Train Normal Loss

Train Depth Loss

0.9

2.00

0.35

1.75

0.8

1.50

0.7

0.30

1.25

loss

loss

loss

0.6

0.25

1.00

0.5

0.20

0.75

0.4

0.50

0.3

0.15

0.25

0.2

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

0 25 50 75 100 125 150 175 200

step

step

step

- Figure 14 Loss curves during training on NYUv2. Compared with the training curves in our experiments (e.g., Figure 10), these curves are much smoother and more stable, making per-step statistics informative signals of learning progress.

###### The running-time comparison in Figure 15 shows that, although effective in some scenarios, gradient-based

Runtime per method

300

Runtime(seconds)

250

200

150

100

50

0

ours stl ls uwscaleinvls rlwdwafamopcgradmgdagraddropcagrad imtlnashmtlfairgradalignedmtlsamgs

- Figure 15 Running time of different MTL methods. While robust to noise in some scenarios, gradient-based methods (denoted by blue) often incur significant overhead (≈ K times, because they compute per-objective gradients) compared with loss-magnitude-based methods (denoted by orange).

MTL methods require storing and computing all task gradients, incurring Θ(K) space and time overhead, where K is the number of objectives. In our illustrative setup with K = 3, this already makes these methods about three times slower (∼ 300s vs. ∼ 100s) than single-task baselines and other loss-reweighting approaches. In our main RLVR experiments, we have four domains with two objectives per domain (K = 8), which would make gradient-manipulation methods roughly 8× slower than standard training. For this reason, we focus on loss-reweighting mechanisms, which avoid such substantial computational overhead.

[Figure 41]

(a) α = 0 (b) α = 0.25 (c) α = 0.5

[Figure 42]

[Figure 43]

[Figure 44]

(d) α = 0.75 (e) α = 1.

[Figure 45]

- Figure 16 Ablation on the trade-off α. Using only the convergence rate (α = 1) or only the inverse signal-to-noise ratio (α = 0) leads to unstable learning for the second and first objectives, respectively.

###### We conduct ablation studies on the temperature hyperparameter T and the trade-off α between the convergence rate and the inverse signal-to-noise ratio: s(kt) = αc(kt)+(1−α)i(kt). From Figure 16, we observe that intermediate values such as α = 0.5 or 0.75 strike a good balance between the two terms and yield noticeably more stable

convergence across all initializations.

For the temperature, setting T too low makes training unstable: as shown in Figure 17a, the trajectories exhibit strong fluctuations near the Pareto front. Conversely, setting T to a high value (e.g., T = 30 in Figure 17d) also harms convergence: for the two initializations farthest from the Pareto front, optimization requires many more steps to approach the front (the trajectories remain red for longer).

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

(a) T = 0.1 (b) T = 1.0 (c) T = 10.0 (d) T = 30.0

- Figure 17 Ablation studies on the temperature T. T > 1 acts as a regularization to avoid extreme reweighting (one domain dominates others) and stabilizes the training.

Due to the large computational requirements of RL training, we find that setting T = 5 and α = 0.5 works reasonably well in the RLVR-only setting. For simplicity, we keep this configuration for the hybrid setup and do not perform additional hyperparameter tuning in the large-scale setting. Table 5 reports the results when varying the trade-off α, the temperature T, and the window size W. Although up-weighting the instability term can increase the weight assigned to the accuracy reward because this term fluctuates substantially, it comes at the cost of sacrificing essential perception skills. For example, α = 0.25 improves performance on SAT and ScienceQA by 0.1% and 0.6%, respectively, but reduces ChartQA and InfoVQA performance by 3%. Similar to our illustrative example, decreasing the temperature induces higher variation across tasks—for instance, it yields the highest score on MathVista while reducing SAT performance by 3.2%.

- Table 5 Benchmark performance in RLVR-only setting. Ablation results when varying the temperature and convergence rate-instability trade-off.

###### Model SAT ScienceQA MathVista (mini) ChartQA InfoVQA MMMU

|MoDoMoDo<br><br>|50.0 65.7 32.2 70.4 59.9 39.1|
|---|---|
|RECAP<br><br>α = 0.50, T = 1.0, W=10 α = 0.75, T = 5.0, W=10 α = 0.25, T = 5.0, W=10 α = 0.50, T = 5.0, W=50|55.2 71.6 33.2 70.4 60.8 42.4 52.0 71.6 33.4 68.1 58.5 40.4<br><br>54.4 71.2 32.9 70.0 60.7 41.0<br><br>55.3 72.2 33.7 66.1 56.8 39.3<br><br><br>51.9 70.5 32.9 69.9 59.7 40.8|

