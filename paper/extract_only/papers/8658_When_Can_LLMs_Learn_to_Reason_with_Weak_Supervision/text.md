## When Can LLMs Learn to Reason with Weak Supervision?

# arXiv:2604.18574v1[cs.LG]20Apr2026

Salman Rahman12* Jingyan Shen2* Anna Mordvina2 Hamid Palangi3 Saadia Gabriel1 Pavel Izmailov2 ❖ Project page: salmanrahman.net/rlvr-weak-supervision

### Abstract

Large language models have achieved significant reasoning improvements through reinforcement learning with verifiable rewards (RLVR). Yet as model capabilities grow, constructing highquality reward signals becomes increasingly difficult, making it essential to understand when RLVR can succeed under weaker forms of supervision. We conduct a systematic empirical study across diverse model families and reasoning domains under three weak supervision settings: scarce data, noisy rewards, and self-supervised proxy rewards. We find that generalization is governed by training reward saturation dynamics: models that generalize exhibit a prolonged presaturation phase during which training reward and downstream performance climb together, while models that saturate rapidly memorize rather than learn. We identify reasoning faithfulness, defined as the extent to which a model’s intermediate steps logically support its final answer, as the pre-RL property that predicts which regime a model falls into, while output diversity alone is uninformative. Motivated by these findings, we disentangle the contributions of continual pre-training and supervised fine-tuning, finding that SFT on explicit reasoning traces is necessary for generalization under weak supervision, while continual pre-training on domain data amplifies the effect. Applied together to Llama3.2-3B-Base, these interventions enable generalization across all three settings where the base model previously failed.

### 1. Introduction

Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for improving reasoning ca-

*Equal contribution 1University of California, Los Angeles 2New York University 3Google. Correspondence to: Salman Rahman <salman@cs.ucla.edu>, Jingyan Shen <js15262@nyu.edu>.

Preprint. April 21, 2026.

pabilities in large language models (Guo et al., 2025; Jaech et al., 2024; Team et al., 2025). With only binary feedback on correctness, RLVR has enabled substantial gains across diverse reasoning tasks without requiring dense supervision. However, recent findings suggest these improvements may be driven by factors other than the integration of correctness signals. Some studies report that RLVR succeeds even under extreme conditions: training on just a single example can yield significant gains (Wang et al., 2025a), and random or incorrect rewards sometimes match ground-truth performance (Shao et al., 2025). Other work shows that proxy signals such as self-certainty (Zhao et al., 2025; Prabhudesai et al., 2025), entropy minimization (Agarwal et al., 2025), majority voting (Zuo et al., 2025), or self-generated training data (Huang et al., 2025) can replace verifiable rewards.

Furthermore, techniques that succeed on one model family often fail on others (Shao et al., 2025), underreported baselines may inflate perceived benefits (Chandak et al., 2025), and prolonged training with proxy rewards (i.e., reward signals derived from model outputs without ground-truth verification) can lead to reward hacking and performance collapse (Shafayat et al., 2025). These mixed results leave a fundamental question: When can RLVR generalize1 under weak supervision, and what determines success or failure?

Understanding when RLVR works under weak supervision matters for practice. Ground-truth verifiers are often limited: labels may be noisy or unavailable, and as models become stronger than their supervisors, alternative reward signals become necessary (Burns et al., 2023).

We conduct a systematic empirical study of RLVR under weak supervision across two model families (Qwen and Llama), and three reasoning domains (MATH, SCIENCE, and GRAPH). Our work is organized around three questions:

- • RQ1 (Weak Supervision): Does RLVR generalize across model families and domains under scarce data, noisy rewards, and self-supervised proxy rewards?
- • RQ2 (Model Properties): What pre-RL model proper-

1Throughout, we use generalization to mean improvement on downstream evaluation benchmarks, both in-domain held-out sets and out-of-domain transfer, following RL training.

ties determine whether a model generalizes under weak supervision?

• RQ3 (Intervention): How can we enable generalization in models that fail under weak supervision?

Our investigation uncovers three findings. First, generalization under weak supervision is governed by training reward saturation dynamics. Models that generalize exhibit a prolonged pre-saturation phase during which training reward climbs steadily and the model learns transferable reasoning patterns; models that fail saturate rapidly and enter a post-saturation phase where further training yields diminishing returns. Which regime a model falls into depends on its pretraining priors: models with strong domain-aligned pretraining (Qwen on MATH and SCIENCE) sustain extended pre-saturation phases and generalize under scarce data, noisy rewards, and self-supervised proxy rewards, while models without such priors (Llama across all domains, and Qwen on GRAPH) saturate rapidly and fail to generalize even under moderate label noise. We treat the model-family contrast as a proxy for pretraining-prior strength rather than an intrinsic property of either family, a reading that §4 confirms by showing that continual pre-training on math data transforms Llama’s RL behavior to resemble Qwen’s.

Second, reasoning faithfulness, not output diversity, distinguishes models that generalize from models that memorize. A natural hypothesis for rapid saturation is that failing models lack exploratory capacity. We find the opposite: Llama models reach perfect training reward faster than Qwen and maintain higher output diversity throughout training, yet they generalize poorly. The missing property is reasoning faithfulness, defined by whether a model’s intermediate steps logically support its final answer. Models that saturate rapidly produce correct answers through reasoning chains that do not justify them, memorizing rather than learning. Diversity is only informative when considered jointly with faithfulness.

Third, SFT on explicit reasoning traces is necessary for generalization under weak supervision, and continual pre-training amplifies the effect. We run a controlled comparison that disentangles the two interventions, training Llama3.2-3B Base, a continually pre-trained variant (CPT, ours), and Instruct, each with either Thinking SFT (explicit reasoning traces) or Non-Thinking SFT (final solutions only). Thinking SFT is necessary: it improves reasoning faithfulness, extends the pre-saturation phase, and enables generalization under all three weak supervision settings, while Non-Thinking SFT on the same prompts fails. Continual pre-training is a multiplier rather than a substitute. CPT combined with Thinking SFT produces the strongest generalization, recovering performance in settings where Llama previously failed.

### 2. Experimental Setup

We evaluate the following model families: (1) Qwen2.51.5B / 3B (Base): General-purpose models pretrained on 18 trillion tokens (Team, 2024); (2) Qwen2.5-Math-1.5B / 7B (Math-specialized): Built upon Qwen2.5 with an additional 1 trillion math-related tokens (Yang et al., 2024); (3) Llama-

##### 3.2-3B / 8B-Instruct (Instruction-tuned): Pretrained on 9 trillion tokens and aligned via SFT, rejection sampling, and DPO (Dubey et al., 2024). We use the Instruct variants for Llama because the base models do not reliably follow the required format for on-policy rollouts. We revisit LlamaBase in §4 , where SFT handles the format-following issue.

Domains and Datasets. We select three domains with varying levels of pretraining exposure: MATH (high exposure), SCIENCE (moderate coverage) and GRAPH tasks (underrepresented in typical pretraining corpora). We use SkyworkOR1 (He et al., 2025a) for MATH, SCP datasets (Liu et al., 2025a; Lu et al., 2025) spanning physics, chemistry, and biology for SCIENCE, and tasks from Reasoning Gym (Stojanovski et al., 2025) involving discrete algorithmic reasoning for GRAPH. For Math and Science, we use the 1.5B/3B models as our primary experiments and additionally evaluate 7B/8B models to verify that our findings hold at larger scale. For GRAPH, we only use the 7B/8B variants because the smaller models achieve solve@16 = 0, leaving no informative signal for RL. More details are provided in Appendix B.

Model-Aware Data Filtering. To ensure informative training signals, we implement model-specific difficulty filtering. For each problem, we sample 16 responses and count correct solutions (solve@16 ∈ [0,16]). We retain only problems where solve@16 ∈ [1,15], effectively discarding instances that are either trivial or intractable for the model, stratified equally across difficulty levels (details in Appendix B.2). This filtered set serves as the candidate pool for all weak supervision settings studied in this work; we describe how training data is constructed from this pool for all settings in §3.

Training Configuration. We use GRPO (Group Relative Policy Optimization) as our RL algorithm (Shao et al., 2024). For each query q sampled from training datasets D, a group of individual responses {oi}Gi=1 are sampled from the policy πθ

before the update. GRPO maximizes the following objective:

old

JGRPO(θ) =E(q,a)∼D,{o

i}Gi=1∼πθold(·|q)

|oi|

G

1 |oi|

1 G

min ρi,tAˆi,clip(ρi,t,1−ϵ,1+ϵ)Aˆi − βDKL(πθ||πref) ,

t=1

i=1

where ρi,t := ππθ(oi,t|q,oi,<t)

θold(oi,t|q,oi,<t) denotes the probability ratio between the current and pre-update sampling policy and

Qwen2.5-Math-1.5B (7B) Qwen2.5-1.5B Llama3.2-3B-Instruct (8B) N=8 N=min(2048, Nmax)

1.00

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

TrainingReward

20

40

60

###### MATH-500(%)

SCP-Hard(%)

0.75

###### AMC(%)

15

30

45

MathScienceGraph

0.50

10

20

30

0.25

5

10

15

0

0.00

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

1.00

32

GPQADiamond(%)

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

TrainingReward

60

###### MATH-500(%)

SCP-Hard(%)

24

0.75

24

45

0.50

16

16

30

0.25

8

8

15

0

0.00

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

40

1.00

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

QuantumLock(%)

LargestIsland(%)

75

TrainingReward

MATH-500(%)

45

30

0.75

60

30

20

0.50

45

15

10

0.25

0

30

0

0.00

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

- Figure 1. Comparison of training dynamics and test performance (avg@16 metric) across model families and domains. For each domain, we plot training reward (column 1), in-domain benchmark performance (column 2-3) and OOD benchmark performance (column

4) over RL steps for two dataset sizes: 8 (solid lines) and Nmax (dashed lines), where Nmax is the largest available training set in the domain for the model. For MATH and SCIENCE, Nmax = 2048. For Graph, Nmax = 882 for Qwen model and Nmax = 256 for Llama model. Colored vertical dashed lines mark the saturation step tNsat for each run. The shaded region indicates one standard deviation over independent sampling. Qwen models exhibit extended pre-saturation phases and generalize from 8 samples, while Llama models saturate rapidly with limited gains. Corresponding results for 7B and 8B models on MATH and SCIENCE are provided in Appendix C.3.

G i=1)

Aˆi := ri−mean({ri}

std({ri}Gi=1)) is the advantage of i-th response calculated by normalizing the group-level rewards. Rewards ri ∈ {0,1} are binary and assigned by ground-truth answer verification. The KL regularization DKL(πθ||πref) is applied to a fixed reference policy πref, weighted by a scalar coefficient β. All experiments use the verl framework (Sheng et al., 2024) (hyperparameter details in Appendix B.3).

Evaluation. We evaluate reasoning performance using avg@16 accuracy (average pass@1 over 16 independent samples per problem) with temperature 1.0 sampling and report pass@k for k ∈ {4,8,16} in the Appendix. For MATH, we use MATH-500, AMC, AIME 2024, AIME 2025, Minerva Math, and OlympiadBench evals. For SCIENCE, we use GPQA-Diamond, a held-out SCP-Hard set (Liu et al., 2025a) (a subset of SCP problems where both Qwen2.51.5B and Llama-3.2-3B-Instruct achieve solve@16 = 1 preRL), Science Bench, MMLU-Science, and SuperGPQA. For GRAPH, we use held-out Quantum Lock and Largest Island tasks from Reasoning Gym (Stojanovski et al., 2025), filtered similarly to solve@16 = 1. For each domain, we designate benchmarks as in-domain or out-of-domain (OOD). For example, for MATH training, MATH-500 and AMC are in-domain, while SCP-Hard and GPQA-Diamond are OOD (full assignments in Appendix Table 2). We report representative results in the main text and full results in the Appendix.

### 3. RLVR Under Weak Supervision

To understand when RLVR generalizes under weak supervision, we study three settings: scarce data (§3.1), noisy rewards (§3.2), and self-supervised proxy rewards (§3.3). We then analyze policy behavior to explain why some models succeed and others fail under these conditions (§3.4). We additionally analyze GRPO baseline selection in Appendix E.

Throughout this section, we compare Qwen and Llama model families. We treat this comparison as a proxy for variation in pretraining priors rather than an intrinsic property of either family: Qwen2.5-Math is pretrained on an additional 1T math-specific tokens, while Llama-3.2-Instruct is aligned for general instruction-following. The contrast we report is between models with strong domain-aligned pretraining and those without, and §4 confirms this interpretation by showing that continual pre-training on math data transforms Llama’s RL behavior to resemble Qwen’s.

#### 3.1. Scarce Data

To understand how data scarcity affects RLVR generalization, we investigate training dynamics across dataset sizes N ∈ {8,32,64,512,2048} across diverse model families and domains. Unlike prior work on sample-efficient RLVR (Wang et al., 2025a; Sun et al., 2025), which select specific data points, we use stratified random sampling

Table 1. Comparison of saturation steps t(8)sat , pre-saturation gain ∆(8)sat and post-saturation residual ∆∗post(8) across model families and training domains when training on 8 examples. We additionally report the large-small gap Gsat(n1,,in8) and G(satn1,,ood8) . For Graph, the largest available setting is n1 = 882 for Qwen model and n1 = 256 for Llama model (marked with †). The green cells mark ∆(8)sat > 0 (effective pre-saturation learning) while red mark rapid saturation t(8)sat < 100. The large-small gap at saturation steps G(satn1,,in8) and G(satn1,,ood8) are generally small. Results on more benchmarks and pass@k metrics are reported in Table 3-7 in Appendix.

In-domain Benchmarks OOD Benchmark

Model t(8)sat

∆(8)sat ∆∗post(8) ∆(8)sat ∆∗post(8) Gsat,in ∆(8)sat ∆∗post(8) Gsat,ood Training Domain: Math MATH500 AMC G(2048sat,in,8) SCP-Hard G(2048sat,ood,8)

Qwen2.5-Math-1.5B 302 29.7 1.5 18.7 0.6 -1.1 10.5 2.1 2.4 Qwen2.5-1.5B 170 32.1 0.9 12.7 3.3 -0.5 7.0 0.3 -0.4 Llama3.2-3B-Instruct 55 10.8 -1.9 8.8 -2.1 -0.9 3.9 0.0 1.5

###### Training Domain: Science SCP-Hard GPQA-Diamond G(2048sat,in,8) MATH500 G(2048sat,ood,8)

Qwen2.5-Math-1.5B 268 14.5 1.1 16.9 1.6 1.1 25.3 0.8 1.1 Qwen2.5-1.5B 161 6.4 0.2 13.3 1.7 1.8 32.3 2.1 1.2 Llama3.2-3B-Instruct 61 1.8 1.7 11.9 3.0 5.1 7.3 2.2 0.6

†

† sat,ood

Training Domain: Graph Quantum Lock Largest Island G(n1,8)

sat,in MATH500 G(n1,8)

Qwen2.5-Math-7B 150 8.3 4.9 19.8 1.9 -1.8 21.0 2.1 -3.7 Llama3.1-8B-Instruct 29 10.1 7.1 1.8 1.0 3.0† 9.1 3.8 0.0†

across difficulty levels defined in §2. For N < 64, we repeat prompts uniformly to reach batch size 64 (e.g., N = 8 implies 8 repeats).

To study training dynamics, we leverage reward saturation to distinguish periods where the policy improves on the training dataset from those where it plateaus. Intuitively, once training reward saturates, further updates yield little new sig-

G i=1 ri as the expected training reward at update step t ∈ {1,...,T}, and let r¯max := max1≤t≤T r¯t be the maximum reward observed during training. We identify training has saturated once the reward is close to this maximum, and define the saturation step as the earliest update where this occurs:

nal. We define r¯t := Eq∼D,{o

1 G

i}Gi=1∼πold(·|q)

tsat := inf t ∈ {1,...,Teff} : r¯t ≥ ϵmaxr¯max}.

We use ϵmax = 0.99 and set Teff = T −50, i.e., we search for tsat only up to the first Teff updates to avoid boundary effects near the end of training. We define the pre-saturation phase as all steps t ∈ {1,...,tsat − 1} and post-saturation phase as all steps t ∈ {min(tsat,T),...,T}.

To quantify data efficiency, we introduce three metrics. Let M(n)(t) denote an evaluation metric (e.g., avg@16 on MATH-500) at training step t for training with n samples,

and t(satn) be the corresponding saturation step.

- • Pre-saturation gain ∆(satn)(M): performance gain from

initialization to saturation as ∆(satn)(M) := M(n) t(satn) − M(n)(0). Larger positive values indicate effective learning before saturation.

- • Post-saturation residual ∆∗post(n)(M): maximum addi-

tional gain after saturation, defined as ∆∗post(n)(M) := maxt∈[t(n)

sat, T] M(n)(t) − M(n) t(satn) . Values near zero indicate negligible post-saturation gains.

′,n)

• Large-small gap G(n

sat (M): we define this gap as M(n

′)(t(satn)) − M(n)(t(satn)) for n′ > n, which compares performance between larger (n′) and smaller (n) datasets at the saturation step of the smaller run. At the smaller run’s saturation step, how much better does the larger run perform? Larger positive values indicate substantial benefit from more data; values near zero suggest limited ad-

vantage from increasing dataset size. We denote G(satn1,,in8) as the average gap over the in-domain benchmarks, and

G(satn1,,ood8) as the average gap over OOD benchmarks.

Pre-saturation phase dominates small-sample learning, and its length predicts generalization. Table 1 summarizes the proposed metrics across model families and training domains when training on 8 examples. Results on more benchmarks and pass@k metrics are provided in Appendix C.2 and Tables 3-7. All model-domain pairs

show clearly positive ∆(8)sat for all metrics (i.e., both avg@16 and pass@k,k ∈ {4,8,16}) across in-domain and out-ofdomain benchmarks, indicating that as few as 8 training examples can trigger measurable learning during the pre-

saturation phase. Neither G(2048sat,in,8) nor G(2048sat,out,8) is significantly greater than zero on 7 out of 8 model-domain

pairs, indicating that the pre-saturation improvements are often comparable to those obtained with larger training sets. This suggests that early learning is not strongly data-limited.

In contrast, the post-saturation residual ∆∗post(8) is typically smaller than ∆(8)sat, indicating diminishing returns once the

###### Qwen2.5-Math-7B Graph

###### Llama-3.2-3B-Instruct Math

1.0

0.75

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

TrainingReward

TrainingReward

0.8

0.60

0.6

0.45

0.4

0.30

0.2

0.15

0.0

QuantumLock(%)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

40

52

MATH-500(%)

30

48

20

44

10

40

0

36

28

78

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

MATH-500(%)

24

72

AMC(%)

66

20

60

16

54

12

0 150 300 450

0 150 300 450

Training Steps

Training Steps

=0

=0.3 =0.5

=0.7 =0.9

| |
|---|

| |
|---|

=0.1

| |
|---|

| |
|---|

| |
|---|

- Figure 2. Effect of reward label corruption on training dynamics and generalization. γ denotes the fraction of training prompts with corrupted labels, ranging from clean (γ = 0) to mostly incorrect (γ = 0.9). For Qwen on GRAPH and Llama on MATH, generalization degrade at γ ≥ 0.5. For Llama, training reward curves stay close across all γ, suggesting overfitting to noise.

8-sample run reaches t(8)sat.

Fig. 1 shows the training curves across data scales. The length of the pre-saturation phase is the primary determinant of whether a model can generalize. With 8 training samples, Qwen2.5-Math-1.5B on MATH increases reward steadily for over 300 steps; this sustained ascent allows the model to extract generalizable reasoning patterns that transfer to heldout evaluation benchmarks such as MATH-500 and SCPHard. A within-family comparison isolates the pretraining effect: Qwen2.5-Math-1.5B, which shares architecture with Qwen2.5-1.5B but has additional math-specific pretraining, saturates more slowly and transfers further (Table 1).

Figs. 13, 14, and 15 (Appendix C.1) show the full range N ∈ {8,32,64,512,2048} across MATH, SCIENCE, and GRAPH. For Qwen models on MATH and SCIENCE, indomain performance is nearly independent of N. For Llama across all domains, and for Qwen on GRAPH, different N produces visibly different dynamics on some of the evals, with smaller datasets saturating earlier and at lower downstream performance.

Models without domain-aligned priors saturate rapidly and fail to generalize. In contrast, Llama models across all domains, and Qwen on GRAPH (Fig. 1) exhibit clear dependence on data scale. For Llama, training on 8 samples leads

to rapid saturation, with t(8)sat occurring within the first 100: it maximizes the training reward much faster than the Qwen models. These models require larger datasets (N ≥ 512) to achieve meaningful generalization (details in Appendix C.1 Fig. 13 and Fig. 14). The results in the GRAPH domain sug-

gest that even for models with strong mathematical priors, the lack of domain-specific pre-training accelerates saturation and necessitates higher data volume to drive learning. We further provide illustrations for 7B and 8B models on MATH and SCIENCE domains in Appendix C.3.

Extended pre-saturation enables out-of-domain transfer. Positive ∆(8)sat values in Table 1 indicate that the reasoning patterns learned during the pre-saturation phase transfer across domains, particularly for Qwen models. With only 8 samples, Qwen2.5-1.5B trained on MATH achieves consistent gains on the out-of-domain SCIENCE benchmark (SCP-Hard), while Qwen2.5-Math-7B trained on GRAPH improves out-of-domain MATH-500 performance by 21.0% (Fig. 1). In contrast, Llama models show limited out-ofdomain transfer even when in-domain performance improves; their gains remain localized to the specific training distribution.

#### Takeaway:

(1) RLVR can generalize from as few as 8 samples when models remain in an extended pre-saturation phase, whereas rapidly saturating models require substantially more data. (2) Whether scarce-data learning succeeds is model- and domain-dependent, reflecting the influence of pretraining priors. (3) In the low-data regime, Llama models can achieve perfect training rewards much faster than Qwen by rapidly memorizing training examples but achieve little meaningful task learning.

#### 3.2. Noisy Rewards

When ground-truth verifiers are available but imperfect, reward labels may contain errors. To evaluate RLVR robustness to such noisy supervision, we vary the fraction of incorrect labels γ by randomly replacing ground-truth answers with the most frequent incorrect answer produced by the model itself (details in Appendix D.1). Unless otherwise noted, experiments use N = 2048.

RLVR demonstrates robustness to reward noise, but generalization varies across models. Fig. 2 and Appendix Fig. 26 summarize performance across seven model–domain pairs under varying γ. At γ ≤ 0.3, test performance across most settings remains close to the clean rewards (γ = 0), indicating robustness to moderate label noise. On MATH and SCIENCE, Qwen models maintain gains under substantial corruption (up to γ = 0.7). In contrast, Qwen on GRAPH and Llama on MATH and SCIENCE degrade at γ ≥ 0.5. Higher γ leads to consistently lower training rewards throughout training, but for Llama on MATH, training reward curves remain nearly identical across all γ despite severe corruption, indicating Llama fits incorrect answers

###### Qwen2.5-3B Science

###### Llama-3.2-3B-Instruct Science

TrainingReward

1.0

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

0.8

0.6

0.6

0.4

0.4

0.2

0.2

MATH-500(%)

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

48

65

40

60

55

32

50

24

45

16

40

40

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

SCP-Hard(%)

20

32

15

24

10

16

5

8

0

0

0 150 300 450 600 750

0 150 300 450 600 750

Training Steps

Training Steps

RLVR

Majority Vote

Self Certainty

| |
|---|

| |
|---|

- Figure 3. Comparison of reward variants (RLVR, self-certainty, majority vote) with 1024 training samples. Proxy rewards without verifiers exhibit failure modes under prolonged training: training collapse (self-certainty) and reward spikes followed by performance drops (majority vote) (more results are in Appendix D.2).

more easily. We also observe that model-domain pairs with faster saturation (§3.1) are generally less robust to label noise, a connection we develop in §3.4 and §4.

Takeaway: Robustness to label noise varies sharply across model-domain pairs: Qwen on MATH and SCIENCE tolerates up to 70% corruption, while Llama and Qwen on GRAPH degrade at 50%. Model-domain pairs that saturate faster under clean rewards are less robust, and Llama fits corrupted labels nearly as fast as clean ones — evidence that rapid saturation reflects memorization capacity rather than learning efficiency.

#### 3.3. Self-Supervised Proxy Rewards

When ground-truth verifiers are entirely unavailable, models must rely on alternative reward signals (Burns et al., 2023; Rahman et al., 2025; Bowman et al., 2022). Recent work has proposed self-supervised proxy rewards derived from model outputs, but whether these approaches work well across model families and task domains remains unexplored. We evaluate two such rewards: self-certainty (Zhao et al.,

- 2025) and majority vote (Zuo et al., 2025) (implementation details in Appendix D.2).

Proxy rewards trigger reward hacking and policy collapse. While RLVR tolerates moderate label noise in some model-domain pairs (§3.2), Fig. 3 shows that fully replacing verifiable feedback with self-supervised proxy signals introduces severe failures under prolonged training. Only mathspecialized models (Qwen2.5-Math-1.5B on MATH and

[Figure 1]

Figure 4. Evolution of semantic diversity during 8-sample training on MATH. Llama shows significantly higher post-saturation diversity than Qwen, albeit with lower performance outcomes.

SCIENCE) show improvement with majority voting, while other models fail entirely. For Qwen2.5-3B on SCIENCE, majority voting yields temporary gains before collapse after 500 steps, as the policy converges toward a single output to maximize agreement. Self-certainty rewards lead to performance collapse across all settings. These results show that current self-supervised proxy rewards are insufficient to replace verifiable feedback in most settings. (details in Appendix D.2 and Fig. 27).

Takeaway: Self-supervised proxy rewards succeed only for math-specialized models (Qwen-Math under majority voting). Other models exhibit the same pattern we observed in §3.1 and §3.2: faster saturation and weaker pretraining priors coincide with brittleness. Under prolonged training, the failure mode is reward hacking: policies converge toward outputs that maximize the proxy without corresponding downstream gains.

#### 3.4. Why Do Models Fail Under Weak Supervision?

The results in §3.1–§3.3 show a consistent pattern: models with strong domain-aligned pretraining (Qwen on MATH and SCIENCE) generalize under weak supervision, while those without (Llama across domains, Qwen on GRAPH) fail. A natural hypothesis, motivated by prior work linking diminished exploratory capacity to rapid policy saturation (Cui et al., 2025), is that failing models produce less diverse outputs. To test this, we analyze model behavior along two complementary axes: response diversity and reasoning faithfulness. Formal definitions and implementation details are provided in Appendix F.

To quantify response diversity, we quantify semantic diversity to characterize meaningful patterns in the model’s reasoning rather than surface-level variation (Farquhar et al., 2024; Li et al., 2025). We measure diversity on the 8-sample subset of the MATH, SCIENCE and GRAPH training datasets, as well as on the MATH-500 evaluation dataset, over a selection of prompts at various steps throughout training. For each prompt, we cluster model responses using pairwise similarity judgments from an LLM judge and define the diversity score as the Shannon diversity index over the re-

[Figure 2]

- Figure 5. Evolution of reasoning faithfulness (on correct samples) and faithful diversity on models throughout RL using 8 samples from a variety of datasets. Llama models in the MATH domain exhibit significantly lower faithfulness compared to Qwen.

sulting clusters. See Figure 31 for the judge model prompt. High diversity does not prevent rapid saturation. Fig. 4 reports the evolution of diversity scores for models trained on 8 samples from the MATH training dataset, computed on the corresponding training set. Llama reaches reward saturation earlier and retains higher diversity than Qwen, the opposite of what the exploration-saturation hypothesis predicts. Diversity computed on the MATH-500 evaluation dataset is presented in the appendix (Fig. 30).

Since diversity alone does not explain failure under weak supervision, we investigate the faithfulness of a model’s reasoning. Inspired by prior work (Baker et al., 2025), we define a response as faithful if its reasoning trace contains the information needed to justify the final answer and is logically consistent with it. At a given training step and for a given prompt, we categorize each policy rollout as aligned, partially aligned, or misaligned based on rubrics provided to an LLM-as-a-judge (see prompt in Fig. 32). We then compute the policy faithfulness rate Fπ(l) as the fraction of responses assigned to label l. Appendix F outlines results for inter-model agreement on alignment categorization to evaluate the reliability of our LLM-as-a-judge.

Models with rapid saturation exhibit low reasoning faithfulness. Fig. 5 (left) shows the fraction of correct responses that are aligned over RL training across models and domains studied in §3.1. On the MATH domain, the Llama model shows much lower reasoning faithfulness during training than the Qwen models. This indicates that Llama’s rapid reward gains do not reflect improved reasoning: a substantial fraction of correct answers are memorized, with reasoning traces that do not support them. Fig. 33 in Appendix F includes additional faithfulness results on these domains, covering proportion aligned and proportion misaligned on correct, incorrect and all responses.

Reasoning diversity should be considered jointly with

faithfulness. Fig. 5 (right) reports faithful diversity: diversity computed only over faithful responses. This joint measure reveals a consistent pattern across all three domains. On MATH, Llama’s apparent diversity advantage (Fig. 4) disappears — most diverse responses are unfaithful, and the faithful subset is narrow. On SCIENCE, aligned proportions are uniformly high across models, masking real differences in reasoning quality; faithful diversity separates them, with Qwen-Math maintaining the highest values throughout training. On GRAPH, Qwen-Math and Llama show comparable aligned proportions, but Qwen-Math sustains higher faithful diversity. In every case, the model that generalizes best in §3.1 is the one exploring the widest range of faithful reasoning paths — not the one with the highest raw diversity, nor the one with the highest aligned proportion. Raw diversity overstates exploratory capacity; aligned proportion saturates on easier domains; only their intersection predicts generalization.

#### Takeaway:

Low reasoning faithfulness, not low diversity, explains why models fail under weak supervision: rapidly saturating models memorize answers rather than acquire transferable reasoning. Raw diversity metrics are misleading — Llama exhibits higher output diversity than Qwen while generalizing worse. Diversity becomes informative only when computed over faithful responses.

In summary, §3 shows that the surprising capabilities often attributed to RLVR, such as learning from scarce data, tolerating noisy rewards, succeeding without verification, are not universal but depend on pre-RL reasoning faithfulness. §4 takes up the natural question: can pre-RL interventions targeting faithfulness extend the pre-saturation phase and recover generalization under weak supervision?

### 4. Improving RLVR Under Weak Supervision via Pre-RL Training

Section 3 showed that rapid saturation and low reasoning faithfulness are linked: models that generalize poorly under weak supervision produce correct answers through reasoning that does not support them. This raises a causal question. If faithfulness drives the pre-saturation phase, and the pre-saturation phase drives generalization, then instilling faithfulness before RL should extend the phase and recover generalization. We test this by running a controlled comparison of pre-RL interventions on Llama3.2-3B, the model that failed most consistently in §3.

We study two axes of pre-RL training. The first is continual pre-training (CPT), extended training on domain-specific

32

| |
|---|

| |
|---|

TrainingReward

MATH-500(%)

56

SCP-Hard(%)

| |
|---|

| |
|---|

15

AMC(%)

24

| |
|---|

| |
|---|

| |
|---|

48

0.6

Scarce

| |
|---|
| |

| |
|---|
| |

Data Majority

| |
|---|
| |

16

40

| |
|---|

12

| |
|---|

| | | | | | |
|---|---|---|---|---|---|
| || |
|---|
|| |
|---|
<br><br>| || |
|---|
| |
| | | | | | |
| | | | | | |

| |
|---|

18

6

| |
|---|

| |
|---|

12

0

0.0

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

64

MATH-500(%)

| |
|---|

| |
|---|

| |
|---|

TrainingReward

30

SCP-Hard(%)

56

| |
|---|

| |
|---|

| |
|---|

AMC(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

15

| |
|---|

48

| |
|---|
| |

0.6

Vote NoisyReward

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

15

| |
|---|

20

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0

0.0

0

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

| |
|---|

60

TrainingReward

MATH-500(%)

0.6

| |
|---|

| |
|---|

| |
|---|

25

SCP-Hard(%)

54

(=0.7)

| |
|---|

AMC(%)

15

| |
|---|

| |
|---|

| |
|---|

20

| |
|---|

| |
|---|

48

| |
|---|

| |
|---|

| |
|---|

0.3

15

20

| |
|---|

| |
|---|

| |
|---|
| |

| |
|---|

5

| |
|---|

| |
|---|

15

| |
|---|

0

10

0

| |
|---|

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

Base + Thinking SFT Base + Non-Thinking SFT CPT + Thinking SFT CPT + Non-Thinking SFT Instruct

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 6. RL training dynamics and generalization on MATH for Llama3.2-3B Base, CPT, and Instruct variants under different SFT initializations across three weak supervision settings: scarce data (N = 8, top), majority vote (middle), and noisy reward (γ = 0.7, bottom). Thinking SFT (solid lines) consistently prolongs the pre-saturation phase and improves generalization for both CPT and Base models compared to their Non-Thinking SFT counterparts (dashed lines) and the Instruct baseline (dash-dot). CPT + Thinking SFT achieves the strongest performance across all settings.

pretraining tokens to strengthen the pretraining prior. The second is supervised fine-tuning (SFT), with the specific question of whether SFT on explicit reasoning traces differs in its effect from SFT on final answers alone. Crossing these axes gives a 2×2 design: two initializations (Base, CPT) each followed by two SFT regimes (Thinking, NonThinking). We additionally include Llama3.2-3B-Instruct as a reference: it shares the architecture of Llama3.2-3B-Base but has undergone extensive instruction tuning, rejection sampling, and DPO, providing a strong off-the-shelf baseline against which to judge our targeted interventions. We then run RL under all three weak supervision settings from §3: scarce data, noisy rewards, and self-supervised proxy rewards.

We focus on the MATH domain for two reasons: Llama’s baseline failure is sharpest there, providing the cleanest test of whether pre-RL interventions can recover generalization; and high-quality math pretraining corpora (Nemotron-CCMath) and reasoning-trace datasets (OpenThoughts-114K) are available, enabling the interventions at sufficient scale. Continual Pre-Training (CPT). We continually pre-train Llama3.2-3B-Base for one epoch on approximately 52B math tokens from the Nemotron-CC-Math dataset (Mahabadi et al., 2025)2. Training details are provided in Appendix B.5.

SFT Training Regimes. Following CPT or Base initialization, we apply supervised fine-tuning to determine whether

2Nemotron-CC-Math-v1

explicit reasoning traces influence subsequent RL dynamics. We compare two SFT regimes that differ only in whether the supervision includes explicit reasoning. Both regimes use the same 43.5K math prompts and differ only in the target output. Specifically, we sample these prompts from OpenThoughts-114K (Guha et al., 2025), retaining only those whose reasoning traces have correct final answers and total length below 8192 tokens.

- • Non-thinking SFT: The model is supervised to output the final solution without generating intermediate reasoning traces.
- • Thinking SFT: The model is trained on explicit, verified long-form reasoning traces.

A training example is shown in Fig. 12 in the Appendix. The SFT regimes are near-iso-compute: Thinking SFT trains on roughly 1B tokens, Non-Thinking SFT on roughly 0.27B, both negligible relative to the 52B-token CPT stage. Differences between Thinking and Non-Thinking SFT therefore reflect the content of the supervision rather than its cost. We report the CPT loss curve in Appendix Fig. 10 and the SFT loss curves in Fig. 11.

Implementation and training details of SFT are provided in Appendix B.6. For the subsequent RL phase, we evaluate across all three weak supervision settings: scarce data (N = 8), noisy rewards (γ = 0.7), and self-supervised proxy rewards (majority vote). All other hyperparameters follow the configurations in §2, with the maximum response length during RL extended to 8192 tokens to accommodate long-

form reasoning traces.

#### 4.1. Results

- Fig. 6 reports RL training dynamics for the five pre-RL configurations (Base, CPT, and Instruct, with Thinking SFT or Non-Thinking SFT applied to Base and CPT) across the three weak supervision settings. For each setting, we plot training reward alongside three downstream metrics: two in-domain (MATH-500, AMC) and one out-of-domain (SCP-Hard); additional benchmarks and pass@k results are Fig. 34 and Fig. 35 in Appendix G. We draw three findings from this figure, developed in the paragraphs below.

Thinking SFT is necessary for substantial learning under weak supervision. The Instruct baseline is flat or decreasing across all three settings on all downstream evaluations RL produces no meaningful improvement from this starting point. Thinking SFT is the only intervention that enables substantial downstream gains on scarce data and majority vote, and it does so for both Base and CPT initializations (solid blue and solid red). Non-Thinking SFT shows modest gains only when paired with CPT, and only under noisy rewards; Non-Thinking SFT on Base is flat or degrades across all three settings.

CPT amplifies the Thinking SFT effect. Thinking SFT on Base alone produces modest gains. Combined with CPT, it produces substantially larger gains on every evaluation: CPT + Thinking SFT is the top-performing curve across all three weak supervision settings and all three evals. The CPT + Non-Thinking SFT comparison rules out a compute-based explanation: the same 52B CPT tokens, paired with SFT targets that strip reasoning traces, fail to enable generalization on scarce data and majority vote. The amplification is specific to the combination: extra pre-training compute alone is insufficient; Thinking SFT alone helps but is limited, only the combination recovers full generalization.

Base initialization fails under most weak supervision settings regardless of SFT. The Base model shows meaningful improvement only in two combinations: Base + Thinking SFT under scarce data and majority vote, and even there gains are modest. Under noisy rewards, neither Base + Thinking SFT nor Base + Non-Thinking SFT produces meaningful downstream improvement. This isolates CPT’s contribution: Thinking SFT is necessary but not sufficient domain-aligned pretraining is required for the intervention to generalize across all three weak supervision settings.

Thinking SFT improves reasoning faithfulness. In §3.4, we identified low reasoning faithfulness as the pre-RL property that distinguished failing from succeeding models.

- Fig. 7 shows that Thinking SFT raises aligned-response rate throughout the pre-saturation phase, relative to the NonThinking SFT baseline. CPT + Thinking SFT achieves the

[Figure 3]

Figure 7. Evolution of reasoning faithfulness of the Llama3.23B family on weak supervision domains when combined with continual pretraining and SFT variants. When combined with Thinking-SFT and CPT, the Llama3.2-3B-Base model exhibits higher reasoning faithfulness.

highest faithfulness among all configurations, consistent with its strongest generalization across all weak supervision settings. Together with the extended pre-saturation dynamics visible in Fig. 6 (leftmost column), this result supports our hypothesis in §3.4: pre-RL interventions that instill faithfulness produce longer pre-saturation phases and recovered generalization, in models that previously failed.

Takeaway: SFT on explicit reasoning traces, not on final answers, is necessary for Llama to learn substantially from RL under weak supervision. It raises reasoning faithfulness, extends the pre-saturation phase, and recovers generalization under scarce data, noisy rewards, and self-supervised proxy rewards. Continual pre-training amplifies the effect but does not substitute for it: CPT + Non-Thinking SFT fails despite matched compute. The strongest configuration, CPT + Thinking SFT, recovers performance in settings where Llama had previously collapsed entirely.

### 5. Related Work

RLVR for Reasoning. Reinforcement learning with verifiable rewards has emerged as an effective post-training method for improving reasoning in large language models (Guo et al., 2025; Olmo et al., 2025; Yu et al., 2025; Zeng et al., 2025). Recent work has explored when RLVR yields improvements (Liu et al., 2025b;a; Hu et al., 2025). Wang et al. (2025a) demonstrate that training on a single example can provide meaningful learning signals. Other work explores alternative rewards, including self-certainty (Zhao et al., 2025), majority voting (Zuo et al., 2025), negative signals (Zhu et al., 2025), self-generated training data Huang et al. (2025), and spurious rewards (Shao et al., 2025). However, these findings often do not transfer across model families, with studies reporting inconsistent results between Qwen and Llama (Zeng et al., 2025; Gandhi et al., 2025; Shao et al., 2025). Moreover, most prior work focuses on improving performance on narrow domains (primarily math) without examining generalization. Recent work (He

et al., 2026; Yang et al., 2026; Plesner et al., 2026) has concurrently studied when and how RLVR can learn under self-supervision or noisy supervision. Our work extends this literature in two ways. First, we characterize the conditions under which RLVR generalizes across model families and domains, focusing on saturation dynamics and reasoning faithfulness. Second, we identify a concrete intervention that restores generalization in models where weak supervision would otherwise fail.

Role of Pre-Training and Fine-Tuning in RL. Recent work emphasizes that pre-training and mid-training shape RL generalization (Qi et al., 2025; Wang et al., 2025b; Zhang et al., 2025; Akter et al., 2025), but focuses on compute allocation and distribution alignment to improve performance. Our work specifically focuses on understanding how base model priors shaped from continual pretraining and reasoning SFT can enable generalization across different weak supervision settings.

Diversity and Faithfulness in Reasoning. Maintaining output diversity during RL has been proposed to promote exploration and mitigate model collapse (Kirk et al., 2024; Casper et al., 2023; Rafailov et al., 2023; Yu et al., 2025), but prior work has not explored what types of diversity benefit generalization. Separately, research has highlighted mismatches between chain-of-thought traces and model predictions (Turpin et al., 2023; Chen et al., 2025b; Baker et al., 2025; Tutek et al., 2025) and emphasized the importance of ensuring faithful reasoning throughout training (Gui et al.,

- 2026). Wen et al. (2025) argues that RLVR can incentivize correct reasoning in base LLMs as long as priors have been established. Our work connects these lines of research, showing that diversity alone does not ensure generalization and that reasoning faithfulness distinguishes models’ training dynamics. We further demonstrate that pre-RL intervention can improve reasoning faithfulness and improve generalization under weak supervision.

### 6. Conclusion

In this work, we studied when and why RLVR generalizes under weak supervision across diverse model families and three reasoning domains. Success under scarce data, noisy rewards, and self-supervised proxy rewards depends on preRL properties, pretraining priors and reasoning faithfulness, rather than on RL dynamics alone. Models that saturate rapidly produce correct answers through reasoning that does not support them, memorizing rather than learning, while maintaining the high output diversity normally taken as a sign of healthy exploration. Pre-RL interventions targeting reasoning faithfulness recover generalization: SFT on explicit reasoning traces is the necessary ingredient, and continual pre-training on reasoning-heavy data amplifies the effect without substituting for it. These findings sug-

gest two concrete practices for RL from weak supervision. First, monitor training reward saturation as a diagnostic: plateaued reward with flat downstream performance indicates the model has exhausted what RL can extract from its priors, and further RL compute is unlikely to help. Second, when weak supervision fails, allocate compute to pre-RL interventions that install strong priors rather than to longer RL training. Taken together, our findings argue that RL under weak supervision is best understood not as a training technique applied to a fixed model, but as the final stage of a pipeline whose success is largely determined before RL begins.

### Acknowledgements

We would like to thank Leon Li, Vatsal Baherwani, Rohun Agrawal, Siyan Zhao, Liwei Jiang, and Andy Han for their insightful discussions and feedback on the draft. Pavel Izmailov was supported by a grant from the Alignment Project, funded by the UK AI Security Institute (grant APS2-100141).

### References

Agarwal, S., Zhang, Z., Yuan, L., Han, J., and Peng, H. The unreasonable effectiveness of entropy minimization in llm reasoning. arXiv preprint arXiv:2505.15134, 2025.

AI-MO. Aime 2024. https://huggingface.co /datasets/AI-MO/aimo-validation-aime,

- 2024a.

AI-MO. Amc 2023. https://huggingface.co /datasets/AI-MO/aimo-validation-amc,

- 2024b.

Akter, S. N., Prabhumoye, S., Nyberg, E., Patwary, M., Shoeybi, M., Choi, Y., and Catanzaro, B. Front-loading reasoning: The synergy between pretraining and posttraining data. arXiv preprint arXiv:2510.03264, 2025.

Baker, B., Huizinga, J., Gao, L., Dou, Z., Guan, M. Y., Madry, A., Zaremba, W., Pachocki, J., and Farhi, D. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv preprint arXiv:2503.11926, 2025.

Bowman, S. R., Hyun, J., Perez, E., Chen, E., Pettit, C., Heiner, S., Lukoˇsi¯ut˙e, K., Askell, A., Jones, A., Chen, A., et al. Measuring progress on scalable oversight for large language models. arXiv preprint arXiv:2211.03540, 2022.

Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A., Joglekar, M., Leike, J., et al. Weak-to-strong generalization: Eliciting

strong capabilities with weak supervision. arXiv preprint arXiv:2312.09390, 2023.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., Wang, T. T., Marks, S., S´egerie, C.-R., Carroll, M., Peng, A., Christoffersen, P. J. K., Damani, M., Slocum, S., Anwar, U., Siththaranjan, A., Nadeau, M., Michaud, E. J., Pfau, J., Krasheninnikov, D., Chen, X., Langosco, L., Hase, P., Biyik, E., Dragan, A. D., Krueger, D., Sadigh, D., and Hadfield-Menell, D. Open problems and fundamental limitations of reinforcement learning from human feedback. Transactions on Machine Learning Research, 2023. URL https://openreview.net/forum ?id=bx24KpJ4Eb.

Chandak, N., Goel, S., and Prabhu, A. Incorrect baseline evaluations call into question recent llm-rl claims. ht tps://safe-lip-9a8.notion.site/Incor rect-Baseline-Evaluations-Call-into-Q uestion-Recent-LLM-RL-Claims-2012f1f bf0ee8094ab8ded1953c15a37?pvs=4, 2025. Notion Blog.

Chen, P., Li, X., Li, Z., Yin, W., Chen, X., and Lin, T. Exploration vs exploitation: Rethinking rlvr through clipping, entropy, and spurious reward. arXiv preprint arXiv:2512.16912, 2025a.

Chen, Y., Benton, J., Radhakrishnan, A., Uesato, J., Denison, C., Schulman, J., Somani, A., Hase, P., Wagner, M., Roger, F., et al. Reasoning models don’t always say what they think. arXiv preprint arXiv:2505.05410, 2025b.

Cheng, Z., Hao, S., Liu, T., Zhou, F., Xie, Y., Yao, F., Bian, Y., Zhuang, Y., Dey, N., Zha, Y., Gu, Y., Zhou, K., Wang, Y., Li, Y., Fan, R., She, J., Gao, C., Saparov, A., Li, H., Killian, T. W., Yurochkin, M., Liu, Z., Xing, E. P., and Hu, Z. Revisiting reinforcement learning for llm reasoning from a cross-domain perspective, 2025. URL https://arxiv.org/abs/2506.14965.

Cohen, J. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37– 46, 1960.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Du, X., Yao, Y., Ma, K., Wang, B., Zheng, T., Zhu, K., Liu, M., Liang, Y., Jin, X., Wei, Z., et al. Supergpqa: Scaling llm evaluation across 285 graduate disciplines. arXiv preprint arXiv:2502.14739, 2025.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024.

Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

Gandhi, K., Chakravarthy, A., Singh, A., Lile, N., and Goodman, N. D. Cognitive behaviors that enable selfimproving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.

Google DeepMind. Gemini 3 flash. https://gemi ni.google.com/, 2025. Released December 2025. Accessed: 2026-02-18.

Guha, E., Marten, R., Keh, S., Raoof, N., Smyrnis, G., Bansal, H., Nezhurina, M., Mercat, J., Vu, T., Sprague, Z., et al. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.

Gui, R., Li, Y., Qu, X., Liu, Z., Cheng, Y., and Cheng, Y. Faithrl: Learning to reason faithfully through step-level faithfulness maximization. arXiv preprint arXiv:2602.03507, 2026.

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al. Deepseekr1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025.

- He, B., Zuo, Y., Liu, Z., Zhao, S., Fu, Z., Yang, J., Qian, C., Zhang, K., Fan, Y., Cui, G., et al. How far can unsupervised rlvr scale llm training? arXiv preprint arXiv:2603.08660, 2026.
- He, C., Luo, R., Bai, Y., Hu, S., Thai, Z., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., et al. Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P., Zhang, X., Zhang, F., Xu, J., Shen, W., Li, S., Zeng, L., Wei, T., Cheng, C., An, B., Liu, Y., and Zhou, Y. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025a.

He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P., Zhang, X., Zhang, F., Xu, J., Shen, W., Li, S., Zeng, L., Wei, T., Cheng, C., Liu, Y., and Zhou, Y. Skywork open reasoner series. https://capricious-hydroge n-41c.notion.site/Skywork-Open-Reaon ser-Series-1d0bc9ae823a80459b46c149e 4f51680, 2025b. Notion Blog.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Hu, J., Liu, M., Lu, X., Wu, F., Harchaoui, Z., Diao, S., Choi, Y., Molchanov, P., Yang, J., Kautz, J., et al. Brorl: Scaling reinforcement learning via broadened exploration. arXiv preprint arXiv:2510.01180, 2025.

Huang, C., Yu, W., Wang, X., Zhang, H., Li, Z., Li, R., Huang, J., Mi, H., and Yu, D. R-zero: Selfevolving reasoning llm from zero data. arXiv preprint arXiv:2508.05004, 2025.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh,

- A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky,

- A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Kirk, R., Mediratta, I., Nalmpantis, C., Luketina, J., Hambro, E., Grefenstette, E., and Raileanu, R. Understanding the effects of rlhf on llm generalisation and diversity. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net /forum?id=PXD3FAVHJT.

Lewkowycz, A., Andreassen, A., Dohan, D., Dyer, E., Michalewski, H., Ramasesh, V., Slone, A., Anil, C., Schlag, I., Gutman-Solo, T., et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

Li, T., Zhang, Y., Yu, P., Saha, S., Khashabi, D., Weston, J., Lanchantin, J., and Wang, T. Jointly reinforcing diversity and quality in language model generations. arXiv preprint arXiv:2509.02534, 2025.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025a.

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., and Lin, M. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Lu, D., Tan, X., Xu, R., Yao, T., Qu, C., Chu, W., Xu, Y., and

- Qi, Y. Scp-116k: A high-quality problem-solution dataset and a generalized pipeline for automated extraction in the higher education science domain. arXiv preprint arXiv:2501.15587, 2025.

Mahabadi, R. K., Satheesh, S., Prabhumoye, S., Patwary, M., Shoeybi, M., and Catanzaro, B. Nemotron-cc-math: A 133 billion-token-scale high quality math pretraining dataset. arXiv preprint arXiv:2508.15096, 2025.

Olmo, T., Ettinger, A., Bertsch, A., Kuehl, B., Graham, D., Heineman, D., Groeneveld, D., Brahman, F., Timbers, F., Ivison, H., et al. Olmo 3. arXiv preprint arXiv:2512.13961, 2025.

OpenAI. Openai o3 and o4-mini system card. https:

//cdn.openai.com/pdf/2221c875-02dc-4 789-800b-e7758f3722c1/o3-and-o4-min i-system-card.pdf, April 2025a. Accessed 2026-

- 01-25.

OpenAI. gpt-oss-20b model card, 2025b. URL https: //arxiv.org/abs/2508.10925. Accessed: 2026-

- 02-18.

Opencompass. Aime 2025. https://huggingface. co/datasets/opencompass/AIME2025, 2024.

Plesner, A., Guzm´an, F., and Athalye, A. An imperfect verifier is good enough: Learning with noisy rewards. arXiv preprint arXiv:2604.07666, 2026.

Prabhudesai, M., Chen, L., Ippoliti, A., Fragkiadaki, K., Liu, H., and Pathak, D. Maximizing confidence alone improves reasoning. arXiv preprint arXiv:2505.22660, 2025.

- Qi, Z., Nie, F., Alahi, A., Zou, J., Lakkaraju, H., Du, Y., Xing, E., Kakade, S., and Zhang, H. Evolm: In search of lost language model training dynamics. arXiv preprint arXiv:2506.16029, 2025.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741, 2023.

Rahman, S., Issaka, S., Suvarna, A., Liu, G., Shiffer, J., Lee, J., Parvez, M. R., Palangi, H., Feng, S., Peng, N., et al. Ai debate aids assessment of controversial claims. arXiv preprint arXiv:2506.02175, 2025.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Shafayat, S., Tajwar, F., Salakhutdinov, R., Schneider, J., and Zanette, A. Can large reasoning models self-train? arXiv preprint arXiv:2505.21444, 2025.

Shannon, C. E. A mathematical theory of communication. The Bell system technical journal, 27(3):379–423, 1948.

Shao, R., Li, S. S., Xin, R., Geng, S., Wang, Y., Oh, S., Du, S. S., Lambert, N., Min, S., Krishna, R., et al. Spurious rewards: Rethinking training signals in rlvr. arXiv preprint arXiv:2506.10947, 2025.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv:2409.19256, 2024.

Stojanovski, Z., Stanley, O., Sharratt, J., Jones, R., Adefioye,

- A., Kaddour, J., and K¨opf, A. Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2505.24760, 2025.

Sun, Y., Shen, J., Wang, Y., Chen, T., Wang, Z., Zhou, M., and Zhang, H. Improving data efficiency for llm reinforcement fine-tuning through difficulty-targeted online data selection and rollout replay. arXiv preprint arXiv:2506.05316, 2025.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Team, Q. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/b log/qwen2.5/.

Turpin, M., Michael, J., Perez, E., and Bowman, S. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. Advances in Neural Information Processing Systems, 36: 74952–74965, 2023.

Tutek, M., Hashemi Chaleshtori, F., Marasovic, A., and Belinkov, Y. Measuring chain of thought faithfulness by unlearning reasoning steps. In Christodoulopoulos, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 9935–9960,

Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.504. URL https://ac lanthology.org/2025.emnlp-main.504/.

- Wang, X., Hu, Z., Lu, P., Zhu, Y., Zhang, J., Subramaniam, S., Loomba, A. R., Zhang, S., Sun, Y., and Wang, W. Scibench: Evaluating college-level scientific problemsolving abilities of large language models. arXiv preprint arXiv:2307.10635, 2023.
- Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

- Wang, Y., Yang, Q., Zeng, Z., Ren, L., Liu, L., Peng, B., Cheng, H., He, X., Wang, K., Gao, J., et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025a.
- Wang, Z., Zhou, F., Li, X., and Liu, P. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025b.

Wen, X., Liu, Z., Zheng, S., Ye, S., Wu, Z., Wang, Y., Xu, Z., Liang, X., Li, J., Miao, Z., et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., Lu, K., Xue, M., Lin, R., Liu, T., Ren, X., and Zhang, Z. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.

Yang, S., Zhu, G., Song, B., Li, S., Wang, H., Zheng, X., Ma, Y., Chen, Z., Wang, W., and Chen, G. Can llms learn to reason robustly under noisy supervision? arXiv preprint arXiv:2604.03993, 2026.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yue, Y., Chen, Z., Lu, R., Zhao, A., Wang, Z., Song, S., and Huang, G. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Zeng, W., Huang, Y., Liu, Q., Liu, W., He, K., Ma, Z., and He, J. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.

Zhang, C., Neubig, G., and Yue, X. On the interplay of pre-training, mid-training, and rl on reasoning language models. arXiv preprint arXiv:2512.07783, 2025.

Zhao, X., Kang, Z., Feng, A., Levine, S., and Song, D. Learning to reason without external rewards. arXiv preprint arXiv:2505.19590, 2025.

Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., Zhu, X., Li, H., Zhang, Y., Long, X., Hua, E., et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

### A. Limitations and Future Work

We acknowledge several limitations. First, due to computational constraints, our analysis is restricted to specific model families and scales. Validating these findings across larger architectures and broader task suites remains an important direction. Second, our analysis of diversity and faithfulness relies on an LLM-as-a-judge framework. Although we conducted small-scale human verification to validate label quality, we currently restrict this evaluation to a small scale to allow for reasonable labeling costs. Consequently, the development of scalable metrics for reasoning faithfulness and diversity remains an important direction for future research.

### B. Implementation Details

#### B.1. Training and Evaluation Datasets

We investigate RL training dynamics across two model families: Qwen (comprising Qwen2.5-1.5B/3B and Qwen2.5-Math1.5B/7B) and Llama (Llama-3.2-3B/8B-Instruct). Our analysis spans three distinct reasoning domains, MATH, SCIENCE, and GRAPH, allowing for a holistic investigation of RLVR under weak supervision across different domains and model families. For MATH, we sample training prompts from the Skywork-OR1 (He et al., 2025b) dataset. For SCIENCE, we draw problems from the SCP dataset curated by prior work (Liu et al., 2025a; Lu et al., 2025), by selecting Physics, Chemistry, and Biology subjects. For GRAPH, we generate two synthetic algorithmic tasks, Quantum Lock and Largest Island, using the curriculum specifications provided by the Reasoning Gym benchmark (Stojanovski et al., 2025). For each task, we instantiate five difficulty levels following the benchmark’s curriculum, with a balanced number of samples per level.

We include the following domain-specific benchmarks for evaluations:

- • MATH500 (Lightman et al., 2023): A widely used subset of the MATH test split (Hendrycks et al.).
- • AMC (AI-MO, 2024b): 40 competition-level math questions.
- • AIME 2024 (AI-MO, 2024a): 30 competition-level math questions.
- • AIME 2025 (Opencompass, 2024): 30 competition-level math questions.
- • Minerva Math (Lewkowycz et al., 2022): A set of 272 undergraduate-level science and math questions from MIT OpenCourseWare.
- • OlympiadBench (He et al., 2024): A benchmark of 675 problems from international math olympiads and physics contests.
- • GPQA-Diamond (Rein et al., 2024): 198 expert-level questions from GPQA spanning physics, chemistry, and biology; we preprocess the data following previous practice (Cheng et al., 2025).
- • SCP-Hard (Lu et al., 2025; Liu et al., 2025a): A held-out set of 50 SCP questions filtered such that the base models (Qwen2.5-1.5B series models and Llama3.2-3B-Instruct model) achieve solve@16= 1, containing disjoint questions from the SCP training datasets.
- • SuperGPQA (Du et al., 2025): a subset constructed from the original SuperGPQA which contains 319 science questions and 250 non-science questions.
- • MMLU SCI (Wang et al., 2024): a subset of MMLU Pro benchmark containing all college-level chemistry, physics and biology questions.
- • Science Bench (Wang et al., 2023): 692 college-level science questions.
- • Graph Test: A held-out set of 50 algorithmically generated instances from the Quantum Lock and Largest Island tasks using Reasoning Gym (Stojanovski et al., 2025), disjoint from training, filtered such that the base models (Qwen2.5-1.5B series and Llama3.2-3B-Instruct) achieve Pass@16= 1.

#### We also note that GPQA-Diamond, MMLU SCI, and SuperGPQA are multiple-choice benchmarks, for which pass@k may be a less reliable metric. Table 2 details the training and evaluation datasets across the three reasoning domains.

Table 2. Training datasets and evaluation benchmarks across three reasoning domains.

#### Domain Training Source In-Distribution Eval Out-of-Distribution Eval

MATH Skywork-OR1 MATH-500, AMC, AIME-2024, AIME2025, Minerva Math, OlympiadBench

Science Bench, SuperGPQA, GPQADiamond, SCP-Hard, MMLU SCI

SCIENCE SCP-116K SCP-Hard, GPQA-Diamond, MMLU SCI, SuperGPQA, Science Bench

MATH-500, AMC, Minerva Math, OlympiadBench

GRAPH Reasoning Gym Quantum Lock, Largest Island MATH-500, AIME-2024, Minerva Math, GPQA-Diamond, SCP-Hard

Prompt template for MATH and GRAPH. system You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \boxed{Your Answer}. user <question> assistant

- Figure 8. Prompt template used for RL training and evaluation on MATH and GRAPH. The placeholder <question> is replaced with the actual mathematical question during fine-tuning and evaluation. Special tokens are omitted for clarity.

B.2. Training Data Preparation Details We describe our procedure for constructing filtered training datasets tailored to each model’s capabilities.

Difficulty Estimation. For each problem in the source dataset, we sample 16 responses from the base model and count the number of correct solutions, yielding solve@16 ∈ [0,16]. We retain only problems with solve@16 ∈ [1,15], excluding problems that are too difficult (solve@16 = 0) or trivially easy (solve@16 = 16) for the model.

Prompt template for SCIENCE. system Let’s think step by step and output the final answer within \boxed{}. user <question> assistant

- Figure 9. Prompt template used for RL training and evaluation on SCIENCE. The placeholder <question> is replaced with the actual mathematical question during fine-tuning and evaluation. Special tokens are omitted for clarity.

Stratified Sampling. We use a stratified round-robin selection method to construct training subsets of size N ∈ {8,32,64,512,2048}. Filtered problems are partitioned into 15 bins {Bi}15i=1 according to their solve@16 values. To select N problems:

- 1. Initialization: Set the current count of selected problems ntotal = 0.
- 2. Round-Robin Selection: While ntotal < N:

- • Iterate through bins Bi for i = 1,...,15.
- • If Bi contains unsampled problems, randomly select one problem without replacement, add it to the training set, and increment ntotal.
- • Terminate immediately if ntotal = N.

This approach ensures that all difficulty levels are represented as uniformly as possible across all data scales.

#### B.3. Implementation Details of RL Training

All experiments are implemented using the verl framework (Sheng et al., 2024) with its default hyperparameters: learning rate 10−6, KL coefficient β = 0.001, clip ratio ϵ = 0.2 and no entropy regularization. We set group size G = 8 for computational efficiency. For response sampling, we fix the sampling temperature 1.0 and a maximum response length of 2048 tokens unless otherwise noted. In verl, we set both the training batch size and mini-batch size to 64 prompts, yielding exactly one gradient update per training step. Each experiment is run for 496 total gradient updates. A simple rule-based reward function is used, assigning reward 1 to correct answers and 0 otherwise, without incorporating any format-related signals. For MATH and SCIENCE, answer matching and reward computation is implemented with Math-Verify3 library; for GRAPH, we use the internal task-specific evaluation protocol from Reasoning Gym. Prompt templates are detailed in Fig. 8 and Fig. 9.

Training Steps (log scale)

10 100 1k 10k 100k

Training loss (EMA, 50-step window)

1.20

1.10

1.00

Loss

0.90

0.80

0.70

0.60

0.1 B 1 B 10 B 51 B

Tokens Seen (log scale)

Figure 10. Training loss during continual pre-training of Llama3.2-3B on approximately 52B tokens of Nemotron-CC-Math data.

Training Steps

Training Steps

0 100 200 300 400 500

0 100 200 300 400 500

0.50

Training loss (EMA, 50-step window)

Training loss (EMA, 50-step window)

0.45

0.80

0.40

0.70

0.35

Loss

Loss

0.30

0.60

0.25

0.20

0.50

0.15

0.00 B 0.20 B 0.40 B 0.60 B 0.80 B 1.00 B

0.00 B 0.05 B 0.10 B 0.15 B 0.20 B 0.25 B

Tokens Seen

Tokens Seen

(a) CPT + Thinking SFT

(b) CPT + Non-Thinking SFT

- Figure 11. Training loss for (a) Thinking SFT and (b) Non-Thinking SFT on 43.5K math prompts, initialized from the CPT checkpoint.

#### B.4. Implementation Details of Evaluation

We evaluate reasoning performance using avg@16 accuracy (average pass@1 over 16 independent samples per problem) with temperature 1.0 sampling and report pass@k for k ∈ {4,8,16}.

3https://github.com/huggingface/Math-Verify

###### Table 3. Math-domain training (1.5B/3B): in-domain benchmarks.

Model Metric t(8)sat MATH-500 AMC-2023 Minerva Math OlympiadBench AIME-2024

∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat

Qwen2.5-Math-1.5B Avg@16 302 29.7 1.5 -2.0 18.7 0.6 -0.1 14.3 1.5 -1.0 13.7 1.0 -0.8 7.3 0.6 0.2 Pass@4 12.3 0.5 -1.2 15.3 -0.1 0.2 14.0 2.0 -1.0 10.4 1.0 0.5 14.8 0.4 2.0 Pass@8 6.2 0.2 -1.0 13.1 -0.1 -1.2 11.1 2.0 -1.4 8.0 1.2 1.0 16.5 0.4 2.4 Pass@16 2.6 0.4 -0.4 10.8 -0.1 -2.8 10.3 1.2 -3.3 5.8 1.8 1.6 16.7 0.5 3.3

Qwen2.5-1.5B Avg@16 170 42.6 0.9 -0.8 15.8 3.3 -1.0 12.6 0.9 -1.3 13.5 1.2 0.2 1.0 0.9 -0.2 Pass@4 25.4 0.6 0.5 16.7 5.2 0.1 16.1 1.1 -1.1 13.9 1.3 0.6 3.4 2.9 0.3 Pass@8 18.6 0.6 1.0 15.0 7.1 2.1 15.7 1.0 -0.3 12.9 1.4 1.2 4.2 4.4 2.2 Pass@16 13.2 0.7 1.2 8.4 11.0 3.6 14.0 1.6 1.5 11.4 1.7 1.6 3.3 6.4 6.7

Llama3.2-3B-Instruct Avg@16 55 10.8 -1.9 -1.4 8.8 -2.1 -0.4 7.4 -1.1 -1.7 6.5 -0.7 0.5 7.1 -0.6 -3.8 Pass@4 23.9 -3.2 -0.6 21.3 -2.7 3.2 7.7 -1.3 -1.2 14.3 -1.1 2.4 0.0 0.0 -4.2 Pass@8 21.7 -3.9 -0.7 20.0 -2.1 6.4 5.6 -0.4 0.1 13.8 -0.9 3.3 0.0 0.4 -3.7 Pass@16 18.8 -3.8 0.2 15.7 -1.1 10.1 1.9 2.2 2.3 11.6 -1.0 4.4 0.0 1.1 -3.3

###### Table 4. Math-domain training (1.5B/3B): out-of-domain benchmarks.

Model Metric t(8)sat GPQA Diamond SCP-Hard MMLU SCI Science Bench

∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat

Qwen2.5-Math-1.5B Avg@16 302 12.6 1.1 0.9 10.5 2.1 2.4 25.4 1.2 -10.8 4.4 0.2 6.5 Pass@4 33.0 1.7 1.7 22.8 6.0 7.2 35.9 -0.0 -6.7 7.6 0.1 -1.6 Pass@8 41.0 2.5 2.2 25.9 8.6 10.2 31.9 -0.4 -5.4 5.6 0.2 -1.5 Pass@16 39.1 3.8 2.5 26.0 10.6 10.0 22.3 -0.4 -3.4 3.5 1.0 -0.7

Qwen2.5-1.5B Avg@16 170 13.8 1.4 -6.6 7.0 0.3 -0.4 29.4 -0.7 -11.2 6.8 0.6 -0.8 Pass@4 33.0 1.5 -15.5 19.0 2.0 -0.6 47.4 -0.7 -14.6 9.2 0.7 -0.6 Pass@8 37.9 1.0 -17.8 25.3 5.0 0.6 46.4 -0.4 -15.3 9.4 1.0 0.0 Pass@16 31.5 2.3 -16.2 28.0 9.8 2.0 37.4 -0.3 -16.1 9.0 1.2 0.4

LLama3.2-3B-Instruct Avg@16 55 -4.3 1.9 0.8 2.0 0.0 1.5 2.4 -0.4 -1.7 4.8 -0.3 0.1 Pass@4 -2.1 2.8 0.6 5.5 -0.4 3.3 0.1 0.6 0.3 6.8 -0.6 -0.5 Pass@8 0.5 3.0 0.7 8.3 -1.2 3.2 -0.3 0.6 0.2 7.7 -1.0 -0.8 Pass@16 -1.1 6.7 2.1 14.0 -4.3 0.0 -0.3 1.1 0.3 8.2 -1.0 -0.3

#### B.5. Implementation Details of Continual Pre-Training

We continually pre-train Llama3.2-3B on the Nemotron-CC-Math-4plus subset (Mahabadi et al., 2025), comprising approximately 52B tokens of math-relevant documents filtered at quality score ≥ 4. Training is conducted for one epoch with a maximum sequence length of 2,048 tokens and a batch size of 128 sequences. We use AdamW with a peak learning rate of 2 × 10−5, cosine decay schedule, 5% linear warmup, weight decay of 0.01, and gradient clipping at 1.0.

#### B.6. Implementation Details of SFT

For SFT, we train for three epochs with a batch size of 16 and a maximum sequence length of 8192 tokens. We tune the learning rate for each model within the 1 × 10−5,5 × 10−5] and report results for the best-performing setting. For the subsequent RL phase, we evaluate performance across training sample sizes N ∈ {8,2048}. All other hyperparameters follow the configurations established in Section B.3, with the maximum response length extended to 8192 tokens to accommodate long-form reasoning traces.

### C. Data Scale Effect

#### C.1. Additional Experimental Results from Small to Large Data Scale

Figs. 13, 14, and 15 present domain-specific training dynamics and generalization performance across sample sizes N ∈ {8,32,64,512,2048}. Each figure tracks the training reward, two in-distribution benchmarks, and one OOD benchmark, as listed in Table 2.

In the MATH domain, Llama models exhibit rapid saturation in small-sample regimes and rely heavily on data scale. In contrast, Qwen models yield comparable performance across varying sample sizes, characterized by extended saturation periods. Specifically, the math-specialized Qwen2.5-Math-1.5B sustains a pre-saturation phase for 330 gradient steps on 8

###### Table 5. Science-domain training (1.5B/3B): in-domain benchmarks.

Model Metric t(8)sat Science Bench SCP-Hard GPQA-Diamond MMLU SCI SuperGPQA

∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat

Qwen2.5-Math-1.5B Avg@16 268 11.1 0.2 0.0 14.5 1.1 0.8 16.9 1.6 1.3 19.1 1.0 2.0 6.6 0.3 1.4 Pass@4 7.5 -0.1 -0.5 35.0 -1.3 -1.3 38.0 2.3 1.4 34.7 1.1 -0.4 18.1 0.7 2.8 Pass@8 6.0 -0.2 -0.7 43.5 -2.7 -1.6 44.5 1.7 0.5 31.4 0.9 0.3 24.7 0.4 2.8 Pass@16 4.6 0.2 -0.7 44.0 -1.2 -2.0 41.1 0.2 0.5 22.3 0.5 -0.0 29.4 -0.1 1.3

Qwen2.5-1.5B Avg@16 161 6.8 0.5 -0.0 6.4 0.2 0.6 13.3 1.7 2.9 25.4 4.4 8.7 7.9 0.9 2.4 Pass@4 9.7 0.4 0.0 20.4 -1.5 -0.5 33.5 1.9 5.1 54.9 2.0 7.3 21.1 1.5 5.8 Pass@8 9.9 0.3 0.3 31.1 -1.7 -2.3 39.7 1.5 5.4 62.5 0.1 4.7 28.5 2.0 7.4 Pass@16 9.4 0.3 0.7 40.0 1.7 -2.0 32.0 3.0 5.6 60.5 0.2 2.6 32.8 3.1 8.8

Llama3.2-3B-Instruct Avg@16 61 2.6 0.5 0.7 1.8 1.7 5.9 11.9 3.0 4.3 10.6 0.8 2.3 5.2 2.2 3.7 Pass@4 3.5 0.4 0.6 5.1 2.9 11.1 24.0 4.8 3.8 8.7 1.4 0.7 8.5 3.9 5.6 Pass@8 4.1 0.3 0.3 8.2 1.9 11.9 25.7 4.2 1.5 6.7 1.8 -0.2 10.6 3.9 4.5 Pass@16 4.8 0.1 -0.3 16.0 -4.3 6.0 22.3 2.3 -1.1 2.6 2.6 -0.5 11.7 3.4 2.2

###### Table 6. Science-domain training (1.5B/3B): out-of-domain benchmarks.

Model Metric t(8)sat MATH-500 AMC Olympiad Bench Minerva Math

∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat ∆sat ∆∗post Gsat

Qwen2.5-Math-1.5B Avg@16 268 25.3 0.8 1.1 14.7 1.5 0.1 11.4 0.9 0.5 12.7 1.1 0.6 Pass@4 10.2 0.5 0.7 12.8 0.6 0.5 9.7 0.5 -1.1 12.5 1.3 0.7 Pass@8 4.5 0.8 0.5 10.8 0.1 0.1 8.1 0.0 -1.6 9.4 1.8 0.7 Pass@16 1.2 1.3 0.4 9.4 -0.0 -1.3 7.4 -0.3 -2.1 7.7 1.7 0.7

Qwen2.5-1.5B Avg@16 161 32.3 2.1 1.2 14.5 0.7 -0.2 10.7 1.5 1.6 10.6 1.9 0.7 Pass@4 30.9 1.1 0.9 21.1 1.9 2.0 16.9 1.4 2.1 18.3 1.4 0.7 Pass@8 24.5 0.6 0.1 20.4 3.0 5.3 16.9 1.0 1.8 20.4 0.5 0.1 Pass@16 19.0 0.2 -0.6 15.7 4.1 11.2 14.1 1.1 1.4 21.3 0.3 -1.5

Llama3.2-3B-Instruct Avg@16 61 7.3 2.2 0.6 4.9 2.7 1.5 5.8 1.2 -0.1 5.9 1.2 1.2 Pass@4 4.8 2.4 2.3 6.6 3.9 1.5 8.4 2.2 0.3 7.0 1.5 1.8 Pass@8 3.2 2.4 3.2 6.0 4.5 0.2 8.3 2.6 0.4 5.9 2.2 2.0 Pass@16 1.0 3.0 3.6 4.9 5.1 0.0 7.7 2.7 0.4 4.3 2.4 1.5

samples, driving continuous improvements on in-domain benchmarks.

In the SCIENCE domain, the pre-saturation phase yields similar gains across all sample sizes; however, after the saturation point, larger sample sizes demonstrate distinct benefits. Similar to MATH domain, models exhibit significantly different saturation dynamics on small samples.

In the GRAPH domain, we compare two larger models, Qwen2.5-Math-7B and Llama3.1-8B-Instruct. The Qwen model also saturates faster here than in other domains, implying that the lack of domain-specific pre-training accelerates saturation in small-sample regimes.

#### C.2. Full Evaluation Results

In this section, we will report the full evaluation results with all benchmarks and pass@k (k ∈ {1,4,8,16} metrics. Fig. 16, Fig. 17, Fig. 18, Fig. 19, Fig. 24, and Fig. 25 include in-domain and out-of-domain evaluation results across multiple benchmarks in MATH, SCIENCE and GRAPH domains.

Discussions on pass@k. Despite prior work (Yue et al., 2025) discussing divergent behavior between pass@1 and pass@k for k > 1 during RL training, we observe that ∆(8)sat keeps the same sign for all k ∈ {1,4,8,16} across most model-benchmark pairs, indicating consistent improvement in both pass@1 and pass@k. This indicates that during the pre-saturation period, the model is not just closing pass@k and pass@1 gap.

- C.3. Additional Experimental Results on Large Models In this section, we will report the full evaluation results on 7B and 8B models. Fig. 20 and Fig. 21 show the results of Qwen2.5-Math-7B and Llama3.1-8B-Instruct models on MATH domain with

###### Table 7. Graph-domain training (7B/8B): in-distribution benchmarks.

Model Metric t(8)sat Quantum Lock Largest Island

∆sat ∆∗post Gsat ∆sat ∆∗post Gsat

Qwen2.5-Math-7B Avg@16 151 8.0 4.9 7.3 19.8 1.9 -10.9 Pass@4 22.6 1.5 -1.3 16.5 6.1 16.3 Pass@8 26.8 2.8 -2.4 10.7 9.0 29.9 Pass@16 30.6 5.5 -4.1 6.8 8.6 41.3

LLama3.1-8B-Instruct Avg@16 29 10.1 7.1 3.2 1.8 1.0 -0.2 Pass@4 15.3 -0.0 6.2 1.8 0.0 1.6 Pass@8 15.4 7.4 12.5 1.8 0.0 2.8 Pass@16 20.1 16.0 25.0 1.8 0.0 3.1

in-domain and out-of-domain benchmarks, respectively. Fig. 22 and Fig. 23 present the results of Qwen2.5-Math-7B and Llama3.1-8B-Instruct models on SCIENCE domain with in-domain and out-of-domain benchmarks, respectively. Fig. 25 provides the results of Qwen2.5-Math-7B and Llama3.1-8B-Instruct models on GRAPH domain with more out-ofdomain benchmarks.

Similar to the observations on smaller models, during the pre-saturation phases, models show generalization on both in-domain and out-of-domain benchmarks in terms of pass@k metrics. Compared to the 3B model, the 8B Llama model exhibits better cross-domain generalization. However, Llama models still saturate more faster than Qwen models and show clear data dependence (e.g., Fig. 22 on SCIENCE).

### D. Reward Type Effect

#### D.1. Additional Results on Reward Corruption

Reward corruption implementation. For each corruption level γ, we uniformly sample a γ fraction of prompts from the N = 2048 training set for each model–domain pair. For each selected prompt, we draw 96 model responses at temperature 1.0 and select the most frequently occurring incorrect final answer (i.e., one that receives zero reward under our verifier) as the corrupted target. During RL training, we replace the ground-truth labels of the selected prompts with these corrupted labels. For Llama models and the GRAPH domain, we cap γ at 0.9 due to the base model’s inability to generate valid solutions even with extensive sampling.

Results. Fig. 2 shows complementary results to Section 3.2. We observe similar patterns that some models are robust to even large amounts of reward noise. In particular, Qwen models exhibit generalization ability even when trained on almost completely corrupted data; in contrast, Llama models tend to show high reward curves yet poorer generalization to new data, suggesting overfitting to incorrect responses.

#### D.2. Additional Results on Self-Supervised Proxy Rewards

Proxy rewards implementation. We evaluate two self-supervised proxy rewards as alternatives to ground-truth verification: majority voting and self-certainty.

- 1. Majority Voting Reward. Following TTRL (Zuo et al., 2025), we estimate pseudo-labels via majority voting and assign binary rewards based on agreement with the consensus answer. For each prompt, we sample 16 responses from the policy model. The most frequently occurring answer among these 16 responses is selected as the pseudo-label. Rewards are then computed as: r = 1 if the response matches the pseudo-label, and r = 0 otherwise. For policy optimization, we use the first 8 responses to compute advantages. All other RL hyperparameters follow Section B.3.
- 2. Self-Certainty Reward. Following Zhao et al. (2025), we use the model’s own confidence as the reward signal. Self-certainty is defined as the average KL divergence between a uniform distribution over the vocabulary and the

model’s next-token distribution:

|o|

1 |o|

r = Self-certainty(o|q) :=

KL(U∥pπ

(·|q,o<i)) (1)

θ

i=1

where o<i denotes previously generated tokens and U is the uniform distribution over the vocabulary. Higher values indicate greater model confidence. For each prompt, we sample 8 responses and use the self-certainty scores directly as rewards to compute advantages. All other RL hyperparameters follow Section B.3.

Results. Fig. 27 shows full results of self-supervised proxy rewards across model-domain pairs. Except for Qwen2.5-Math1.5B, all other models exhibit failure modes under prolonged training. For Qwen2.5-1.5B on SCIENCE, both proxy rewards collapse: majority voting shows a sharp reward spike followed by performance degradation, while self-certainty leads to complete training collapse. Similarly, Llama-3.2-3B-Instruct on MATH shows degraded performance with both proxy rewards despite increasing training rewards. Only Qwen2.5-Math-1.5B on MATH maintains stable performance with majority voting, though self-certainty still collapses after approximately 200 steps. These results demonstrate that self-supervised proxy rewards are brittle and model-dependent, with only math-specialized models showing partial robustness.

#### D.3. Reward Hacking Example Under Majority Vote

Table 8 shows two rollouts from Qwen2.5-3B trained on SCIENCE with majority vote rewards at training step 846. In both cases, the model produces plausible intermediate reasoning but converges to the same final answer

|0|
|---|

, regardless of the problem content. The majority vote reward is 1.0 because all rollouts agree on this answer — the policy has learned to produce identical outputs to maximize consensus, constituting reward hacking. The correct answers (68.4g and τ0/k, respectively) appear in the reasoning traces but are overridden in the final answer.

Table 8. Two rollouts from Qwen2.5-3B on SCIENCE at step 846 under majority vote reward. Both produce coherent reasoning toward the correct answer but output

|0|
|---|

as the final answer, achieving majority vote reward of 1.0.

Rollout 1: Sucrose solution problem Rollout 2: Moment of inertia problem Prompt: Prepare a 0.0348 mole fraction solution of sucrose using 100g of water.

Prompt: A wheel with moment of inertia I is acted upon by torque τ0, resisted by τf = −kω. Find the maximum speed.

Reasoning (excerpt): “ωmax = τk0 ”

Reasoning (excerpt): “Mass of sucrose required = 0.2 moles × 342 g/mole = 68.4 g”

|0|
|---|

|0|
|---|

Final answer: Majority vote reward: 1.0 Majority vote reward: 1.0

Final answer:

### E. Baseline Effect

We analyze how the choice of reward baseline influences generalization in GRPO. Standard GRPO uses the within-group mean reward (µ = G1 Gi=1 ri) as the baseline. By replacing µ with a constant baseline b ∈ {0,1}, we isolate the direction of the policy update: b = 0 retains only positive reinforcement from correct samples (GRPO-POS), equivalent to the REINFORCE algorithm, while b = 1 retains only negative reinforcement from incorrect samples (GRPO-NEG), which (Zhu et al., 2025) studied in MATH domain. We remove the length penalty term |1o| in GRPO for this experiment. Based on the policy gradient theory, where subtracting an action-independent baseline does not change the expected gradient but reduces variance, with a large batch these two methods should yield similar learning behavior (Williams, 1992).

Figs. 28 and 29 present the training results on the SCIENCE domain for 8 and 1024 samples, respectively. In both regimes, GRPO-POS and GRPO-NEG achieve comparable Pass@1 performance to standard GRPO, exhibiting similar saturation and generalization behaviors. We note that this contrasts with recent findings by (Zhu et al., 2025), which highlight the superiority of GRPO-NEG. However, their improvements were primarily observed in Pass@k metrics rather than Pass@1 and evaluated on MATH domain. Beyond these metric differences, it’s worth studying whether implementation artifacts may also influence observations. For instance, clipping terms in the GRPO formulation can introduce biases (Shao et al., 2025; Chen et al., 2025a). While our strictly on-policy setup mitigates such clipping effects, we leave a comprehensive analysis of these affects to future work.

### F. Diversity and Faithfulness

Table 9. Inter-rater agreement between LLM judges measured using Cohen’s Kappa.

#### Judge Pair Cohen’s Kappa

OpenAI o3 vs. GPT-OSS-20B (OpenAI, 2025b) 0.752 OpenAI o3 vs. Gemini 3 Flash (Google DeepMind, 2025) 0.649

#### F.1. Quantification of generation diversity

To quantify the generation diversity of a model on a given prompt, we generate a number of responses, y1,...,yN and cluster them based on their reasoning similarity. Basing our analysis on the method used by (Li et al., 2025), to determine reasoning similarity between two outputs yi,yj, we define a function s(yi,yj) ∈ {0,1} such that s(yi,yj) = 1 if yi,yj are similar and 0 otherwise. To evaluate s(·,·), we prompt GPT-4o (Hurst et al., 2024) as a diversity judge to determine whether the reasoning produced by any two responses follows a different reasoning path using the prompt specified in Fig. 31.

We form semantic clusters by iterating through responses and comparing them to a representative response from each existing cluster, creating a new cluster if the response is dissimilar to each representative. This is performed under the assumption of transitivity of similarity. We create clusters {C1,...CK} where Ci = {y1,...,yn

i} such that s(yi,yj) = 1 ∀yi,yj ∈ Ci. We then define the diversity scores using the Shannon Diversity Index (Shannon, 1948) as follows. For a given prompt, let N be the total number of responses, ni be the number of responses in cluster Ci, and K be the number of clusters. Let pi = n

N Define the Shannon entropy

i

K

H(p) = −

pi log pi

i=1

and the effective number of clusters

Neff = exp H(p) . We then define the diversity score

Neff − 1 K − 1

. (2)

Divπ(x) =

when K > 1 and 0 otherwise. For a data distribution D, we define the overall generation diversity as dπ(D) = Ex∼D[Divπ(x)]. Empirically, we sample N = 16 outputs per prompt and estimate dπ using 8 prompts from the specified dataset. We define Faithful Diversity as this metric calculated only on responses that achieve a faithfulness score of 1 (see below).

- Fig. 36 shows an example of the LM-as-judge output when prompted to evaluate the similarity of 2 responses.

#### F.2. Quantification of reasoning faithfulness

Inspired by prior work (Baker et al., 2025), we define the faithfulness as a response’s intermediate reasoning trace contains all relevant information and remains logically consistent with the predicted final answer. Each model rollout produces a response y that contains (i) a reasoning trace and (ii) a final answer. We write y = (r,a), where r is the reasoning text and a is the extracted final answer. For each input prompt x, we sample y ∼ π(· | x) from the policy.

Faithfulness labeling. We define a discrete faithfulness labeling function sfaithful : X × Y → {0, 12,1}, where sfaithful(x,y) measures the internal agreement between r and a in y:

- • sfaithful(x,y) = 1 (aligned) if the reasoning trace r constitutes a coherent and logically supportive justification for the produced answer a, regardless of whether a is correct;
- • sfaithful(x,y) = 12 (partially aligned) if r exhibits a plausible argumentative trajectory toward a but contains substantial gaps, unsupported leaps, or local inconsistencies that weaken the justification;

- • sfaithful(x,y) = 0 (misaligned) if a is not supported by r, e.g., r contradicts a, fails to address the question, or the answer appears as the “lucky” guess.

In practice, we implement sfaithful(x,y) by querying OpenAI o3 (OpenAI, 2025a) as an LLM-as-a-judge with a fixed rubric (Fig. 32). OpenAI o3 is used for this task, as opposed to GPT-4o, due to requiring a larger model in order to be able to

accurately reason about complex mathematical and scientific steps present in the reasoning traces. For a label l ∈ {0, 12,1}, we define the faithfulness rate of policy π over dataset D as

Fπ(l) := Px∼D, y∼π(·|x) sfaithful(x,y) = l .

At training step t, we approximate Fπ

(l) using N training prompts {xi}Ni=1 and K rollouts per prompt:

t

N

K

1 NK

sfaithful xi,yi,k = l , yi,k ∼ πt(· | xi). (3)

Fπ

(l) =

t

i=1

k=1

We use N = 8 prompts and K = 16 rollouts per prompt at selected RL checkpoints on the specified training dataset. We report Fπ

(l) for l ∈ {0, 12,1} to characterize the distribution of reasoning faithfulness under the policy πt.

t

- Fig. 37 shows an example of the LM-as-judge output when prompted to evaluate the faithfulness of a model response when trained on the MATH training dataset.

Reliability of LLM-as-a-judge. To mitigate bias from using an LLM-as-a-judge for faithfulness evaluation, we assess consistency across multiple LLM judges by computing Cohen’s Kappa (Cohen, 1960) across 16 faithfulness-scored Qwen2.5-Math-1.5B outputs when trained on 8 samples from the MATH training dataset at steps 20, 120 and 440.

The judges achieve substantial agreement (κ= 0.752 and 0.649), indicating consistent faithfulness labeling across different models. We additionally conducted a small-scale manual evaluation to human-check the faithfulness scores and find fair alignment with the LLM judges.

#### F.3. Additional results on diversity analysis

Fig. 30 shows the semantic diversity of Llama3.2-3B-Instruct, Qwen2.5-1.5B and Qwen2.5-Math-1.5B on the MATH-500 evaluation dataset throughout RL training. Qwen-Math exhibits higher reasoning diversity on correct responses than the other models at the later stages of training, highlighting that RL enables it to successfully learn diverse and reliable strategies; coupled with its better performance on the evaluation dataset, this indicates stronger generalization properties. In particular, we observe significantly lower diversity in the Llama3.2-3B-Instruct model in comparison to its diversity metric on the training dataset (Fig. 4), implying disagreement between training and evaluation distributions and further highlighting the limitations of training diversity as an indicator of reasoning capabilities.

#### F.4. Additional results on faithfulness analysis

- Fig. 33 shows the proportion of responses that are classified as aligned or misaligned when calculated with respect to correct, incorrect or all responses. Out of all correct, incorrect and overall responses, both Qwen2.5-1.5B and Qwen2.5-Math-1.5B show higher proportion of aligned responses and lower proportion of misaligned responses than Llama3.2-3B when trained on 8 samples from the MATH dataset. Qwen2.5-Math-1.5B additionally shows this result when trained on 8 samples from SCIENCE.

G. Pre-RL Intervention

- Fig. 34 report pass@16 results and Fig. 35 reports results on more benchmarks.

Example prompt and response format of SFT User Prompt: Calculate the limit of the function:

arcsin(3x) √2 + x −

√2

lim

x→0

Assistant: <think> Okay, so I need to find the limit ...

Substituting x = 0 gives the indeterminate form 00, so ... Rationalizing the denominator, we get ...

</think> The limit is calculated as follows:

- 1. Recognize the indeterminate form: substituting x = 0 yields 0

0

,

which suggests using rationalization or l’Hˆopital’s rule.

- 2. Rationalize the denominator: arcsin(3x)

√2 + x −

√2 ·

√2 + x + √2 √2 + x + √2

=

arcsin(3x) √2 + x + √2 x

.

- 3. Split the limit:

lim

x→0

arcsin(3x) x · lim

x→0

√2 + x +

√

2 .

- 4. Evaluate each part:

- • Using

lim

z→0

arcsinz z

= 1, the first limit becomes 3.

- • The second limit evaluates to 2√2.

- 5. Combine the results:

√

√

3 · 2

2. Final Answer:

2 = 6

|6<br><br>√<br><br>2|
|---|

- Figure 12. Example prompt and response format of SFT. In Thinking SFT, the model is trained with reasoning traces enclosed by <think> and </think>, whereas Non-Thinking SFT omits them.

8 samples

32 samples

64 samples

512 samples

2048 samples

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1.0

TrainingReward

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

###### MATH-500(%)

SCP-Hard(%)

52

72

36

0.8

Qwen2.5-Math

###### AMC(%)

30

48

69

0.6

24

44

-1.5B Qwen2.5-1.5BQwen2.5-3B

66

18

0.4

40

12

63

0.2

36

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

60

- 0
- 1

TrainingReward

| | | |
|---|---|---|

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

###### MATH-500(%)

30

20

SCP-Hard(%)

50

24

###### AMC(%)

15

40

18

10

30

12

5

20

6

0

0

10

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

1.0

TrainingReward

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

###### MATH-500(%)

66

SCP-Hard(%)

40

25

0.8

###### AMC(%)

63

36

20

0.6

32

15

60

0.4

28

10

57

0.2

24

5

54

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

Llama3.2-3B

1.0

TrainingReward

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

15

###### MATH-500(%)

-Instruct

SCP-Hard(%)

54

30

0.8

12

###### AMC(%)

25

51

9

0.6

20

48

6

0.4

3

15

45

0.2

0

10

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

- Figure 13. Comparisons of RL training dynamics and performance across different models on MATH domain. Results are averaged over three independent runs, with shaded regions indicating error bars. Vertical dashed lines denote the saturation step for each data scale if it saturates before 496 gradient steps. Llama models exhibit rapid saturation in small-sample regimes and rely heavily on data scale. In contrast, Qwen models yield comparable performance across varying sample sizes, characterized by extended saturation periods. Evaluation results in this figure are based on greedy decoding.

8 samples

32 samples

64 samples

512 samples

2048 samples

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

GPQADiamond(%)

1.0

TrainingReward

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

30

###### MATH-500(%)

40

SCP-Hard(%)

72

0.8

25

Qwen2.5-Math

35

69

20

0.6

30

15

-1.5B Qwen2.5-1.5BQwen2.5-3B

0.4

66

25

10

0.2

20

63

5

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

GPQADiamond(%)

30

- 0
- 1

30

60

TrainingReward

| | | |
|---|---|---|

###### MATH-500(%)

SCP-Hard(%)

24

24

50

18

18

40

12

12

30

6

6

20

0

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

GPQADiamond(%)

1.0

TrainingReward

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

###### MATH-500(%)

66

SCP-Hard(%)

33

30

0.8

63

30

24

0.6

27

60

18

0.4

24

12

57

0.2

21

6

54

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

GPQADiamond(%)

Llama3.2-3B

1.0

TrainingReward

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

54

MATH-500(%)

-Instruct

SCP-Hard(%)

30

32

0.8

52

27

24

0.6

50

24

16

21

0.4

48

8

18

46

0.2

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

- Figure 14. Comparisons of RL training dynamics and performance across different models on SCIENCE domain. Results are averaged over three independent runs, with shaded regions indicating error bars. Vertical dashed lines denote the saturation step for each data scale. The pre-saturation phase yields similar gains across all sample sizes; however, after the saturation point, larger sample sizes demonstrate distinct benefits. Models exhibit significantly different saturation dynamics on small samples. Evaluation results in this figure are based on greedy decoding.

| | | | | | |
|---|---|---|---|---|---|

0 150 300 450

Training Steps

- 0
- 1

TrainingReward

0 150 300 450

Training Steps

0

10

20

30

40

50

QuantumLock(%)

0 150 300 450

Training Steps

0

8

16

24

32

LargestIsland(%)

0 150 300 450

Training Steps

74

76

78

80

82

MATH-500(%)

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0 150 300 450

Training Steps

0.2

0.4

0.6

0.8

1.0

TrainingReward

0 150 300 450

Training Steps

0

15

30

45

QuantumLock(%)

0 150 300 450

Training Steps

0

2

4

6

LargestIsland(%)

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

0 150 300 450

Training Steps

50

52

54

MATH-500(%)

Qwen2.5-Math

- -7B Llama3.2-8B
- -Instruct

| |
|---|

8 samples

| |
|---|

32 samples

| |
|---|

128 samples

| |
|---|

256 samples

| |
|---|

882 samples

- Figure 15. Comparisons of RL training dynamics and performance across different models on GRAPH domain. We use larger models (Qwen2.5-Math-7B, Llama-3.1-8B-Instruct) due to increased task difficulty. Results are averaged over three independent runs, with shaded regions indicating error bars. Vertical dashed lines denote the saturation step for each data scale. Qwen model also saturates faster here than in other domains. Larger datasets yield clear gains in the post-saturation phases. Evaluation results in this figure are based on greedy decoding.

Qwen2.5-Math-1.5B Qwen2.5-1.5B LLama3.2-3B-Instruct N=8 N=2048

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

88

80

88

60

80

MATH-500

70

80

45

72

60

72

64

30

50

56

64

15

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

60

70

80

40

50

60

70

30

###### AMC

40

50

60

20

40

30

50

10

30

20

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

24

48

36

42

MinervaMath

30

18

36

42

24

30

36

12

18

24

30

6

12

18

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

30

OlympiadBench

56

48

40

24

48

40

32

18

32

40

24

12

24

32

16

6

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

40

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

15

24

40

32

###### AIME2024

10

18

24

30

5

12

16

20

0

6

8

10

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

10.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

16

24

20

7.5

###### AIME2025

12

18

15

5.0

8

12

10

2.5

4

6

5

0.0

0

0

0

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 16. Full in-domain benchmark evaluation results for the MATH domain across multiple models. Vertical dashed lines denote the saturation step for each data scale.

Qwen2.5-Math-1.5B Qwen2.5-1.5B LLama3.2-3B-Instruct N=8 N=2048

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

90

60

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

75

GPQADiamond

24

75

60

45

16

60

45

30

8

45

30

15

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

32

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

80

60

45

24

SCP-Hard

60

45

30

16

40

30

8

15

15

20

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

30

25

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

16

ScienceBench

30

25

20

12

20

24

15

8

15

18

10

4

10

12

5

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

80

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

60

90

80

###### MMLUSCI

45

60

75

60

60

30

40

40

45

15

20

20

30

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

60

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

20

40

60

SuperGPQA

45

15

30

45

10

30

20

30

5

10

15

0

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 17. Full in-domain benchmark evaluation results for the SCIENCE domain across multiple models. Vertical dashed lines denote the saturation step for each data scale.

Qwen2.5-Math-1.5B Qwen2.5-1.5B LLama3.2-3B-Instruct N=8 N=2048

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

60

90

75

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

GPQADiamond

24

75

60

45

18

60

45

30

12

45

30

6

15

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

60

80

20

40

SCP-Hard

15

45

60

30

10

30

20

40

5

10

15

20

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

32

28

24

16

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

ScienceBench

28

24

20

12

24

20

16

8

20

16

12

4

12

16

8

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

60

80

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

90

80

45

###### MMLUSCI

60

75

60

30

60

40

40

45

15

20

20

30

0

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 18. Full out-of-domain benchmark evaluation results for the MATH domain across multiple models. Vertical dashed lines denote the saturation step for each data scale.

Qwen2.5-Math-1.5B Qwen2.5-1.5B LLama3.2-3B-Instruct N=8 N=2048

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

90

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

90

60

75

75

MATH-500

80

45

60

60

70

30

45

60

45

15

30

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

60

40

75

60

30

45

60

AMC

45

20

30

45

10

30

15

30

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

40

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

48

40

24

MinervaMath

32

40

18

32

24

32

12

24

16

24

6

16

8

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

30

60

50

OlympiadBench

40

24

50

40

18

30

40

30

12

20

30

20

6

10

20

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 19. Full out-of-domain benchmark evaluation results for the SCIENCE domain across multiple models. Vertical dashed lines denote the saturation step for each data scale.

Qwen2.5-Math-7B LLama3.1-8B-Instruct N=8 N=2048

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

93

80

95.0

97.5

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

90

92.5

95.0

87

70

90.0

MATH-500

92.5

84

87.5

90.0

81

60

80

72

84

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

76

68

50

80

72

64

76

68

60

40

72

64

56

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

92

60

76

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

82.5

88

80.0

72

84

77.5

45

68

80

75.0

64

AMC

72.5

76

45

30

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

54

66

40

48

60

35

42

54

30

15

36

48

25

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

32

57

51

MinervaMath

44

28

54

48

40

24

45

51

20

36

42

48

16

32

39

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

70.0

40

63

56

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

67.5

OlympiadBench

60

52

65.0

57

32

48

62.5

54

44

60.0

51

24

40

48

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

32

36

44

28

16

40

32

24

36

28

20

8

32

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

32

54

60

48

40

55

###### AIME2024

24

42

50

30

36

45

16

40

24

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

20

32

18

8

24

12

10

16

6

0

8

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

24

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

32

40

12

18

###### AIME2025

24

30

8

12

16

20

4

6

8

10

0

0

0

0

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 20. Full in-domain benchmark evaluation results for the MATH domain on 7B and 8B models.

Qwen2.5-Math-7B LLama3.1-8B-Instruct N=8 N=2048

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

64

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

80

30

GPQADiamond

88

56

72

25

80

48

64

20

72

56

40

15

48

64

32

10

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

56

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

72

88

24

48

SCP-Hard

64

80

18

40

56

72

12

32

48

64

6

40

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

33

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

27

18

36

ScienceBench

30

24

15

33

27

21

30

12

24

18

27

9

21

15

24

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

64

90

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

94.5

80

56

###### MMLUSCI

88

48

76

93.0

86

40

72

91.5

84

32

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 21. Full out-of-domain benchmark evaluation results for the MATH domain on 7B and 8B models.

Qwen2.5-Math-7B LLama3.1-8B-Instruct N=8 N=384

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

96

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

64

80

GPQADiamond

30

88

56

70

24

80

48

60

18

72

40

50

12

64

32

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

80

90

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

40

60

70

80

SCP-Hard

30

50

60

70

20

40

60

50

10

30

50

40

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

24

32

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

40

36

ScienceBench

20

28

36

32

16

24

28

32

12

20

24

28

16

8

20

24

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

92

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

70

84

94

90

MMLUSCI

60

80

92

88

50

76

86

90

40

72

84

88

30

68

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

75

24

60

45

SuperGPQA

70

20

55

40

65

16

50

35

60

12

30

45

55

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 22. Full in-domain benchmark evaluation results for the SCIENCE domain on 7B and 8B models.

Qwen2.5-Math-7B LLama3.1-8B-Instruct N=8 N=384

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

95

93

80

95.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

90

92.5

87

90

70

MATH-500

90.0

84

87.5

81

85

60

75

80

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

70

50

76

80

65

72

68

40

60

75

64

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

84

92

60

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

75

81

88

72

78

84

69

45

75

66

80

###### AMC

72

63

48

65

30

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

54

42

60

48

36

55

42

15

30

50

36

24

45

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

57.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

35

51

MinervaMath

55.0

44

30

48

52.5

40

25

45

50.0

36

20

42

47.5

15

32

39

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

56

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

64

69

40

OlympiadBench

60

66

48

56

63

32

60

52

40

45

24

50

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

40

32

45

35

16

40

30

24

35

8

25

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 23. Full out-of-domain benchmark evaluation results for the SCIENCE domain on 7B and 8B models.

Qwen2.5-Math-7B LLama3.1-8B-Instruct N=8 N=256

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

100

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

80

60

QuantumLock

45

75

60

30

40

50

40

15

20

25

20

0

0

0

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

40

60

50

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

60

40

LargestIsland

30

45

30

45

20

20

30

30

8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

6

10

15

15

4

0

2

0

0

0

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 24. Full in-domain benchmark evaluation results for the GRAPH domain on 7B and 8B models.

Qwen2.5-Math-7B LLama3.1-8B-Instruct N=8 N=256

Avg@16(%)

Pass@4(%)

Pass@8(%)

Pass@16(%)

95.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

90

75

92.5

87

90.0

88

MATH-500

84

87.5

60

81

85.0

80

78

80

72

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

45

64

72

72

56

64

30

48

56

64

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

60

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

50

40

45

18

35

###### AIME2024

45

40

30

35

12

25

30

24

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

16

6

18

12

15

12

8

6

4

0

0

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

55

40

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

48

MinervaMath

50

55

32

42

45

50

24

36

40

45

16

30

35

40

24

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

32

60

88

GPQADiamond

70

50

24

80

60

40

72

16

50

30

64

8

40

20

56

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

40

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

60

70

80

32

50

SCP-Hard

60

70

24

40

50

60

16

30

40

50

8

20

30

40

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

0 150 300 450 Training Steps

###### Figure 25. Full out-of-domain benchmark evaluation results for the GRAPH domain on 7B and 8B models.

Qwen2.5-1.5B Math

Llama-3.2-3B-Instruct Math

Qwen2.5-1.5B Science

Llama-3.2-3B-Instruct Science

Qwen2.5-Math-7B Graph

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.75

0.60

TrainingReward

TrainingReward

TrainingReward

TrainingReward

TrainingReward

0.8

0.60

0.8

0.60

0.45

0.6

0.45

0.6

0.45

0.30

0.30

0.4

0.4

0.30

0.15

0.15

0.2

0.2

0.15

0.00

0.0

0.00

50

54

QuantumLock(%)

60

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

SCP-Difficult(%)

SCP-Difficult(%)

30

###### MATH-500(%)

###### MATH-500(%)

40

32

51

50

24

30

24

48

40

18

20

45

16

12

30

10

6

42

8

20

0

0

39

82

32

60

30

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

54

###### MATH-500(%)

###### MATH-500(%)

MATH-500(%)

28

80

50

24

52

###### AMC(%)

###### AMC(%)

24

78

40

18

50

20

76

30

12

48

16

74

20

6

46

12

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

Training Steps

=0

=0.1

=0.3

=0.5

=0.7

=0.9/1.0

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 26. Effect of reward label corruption on training dynamics and generalization. γ denotes the fraction of training prompts with corrupted labels, ranging from clean (γ = 0) to fully incorrect (γ = 1). Qwen models on MATH and SCIENCE domains maintain performance under substantial corruption, while generalization of Llama models and GRAPH domain degrade at γ ≥ 0.5. Evaluation results in this figure are based on greedy decoding.

###### Qwen2.5-Math-1.5B Math

###### Llama-3.2-3B-Instruct Math

###### Qwen2.5-1.5B Science

###### Qwen2.5-Math-1.5B Science

1.0

- 0
- 1

TrainingReward

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.90

0.75

0.8

0.75

0.6

0.60

0.60

0.4

0.45

0.45

0.2

0.30

0.30

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

60

MATH-500(%)

60

70

70

50

60

50

40

50

60

12

SCPDifficult(%)

15

32

32

12

9

24

24

9

6

16

16

6

3

8

8

3

0

0

0

0

0 200 400 600 800

0 200 400 600 800

0 200 400 600 800

0 200 400 600 800

Training Steps

Training Steps

Training Steps

Training Steps

RLVR

Majority Vote

Self Certainty

| |
|---|

| |
|---|

- Figure 27. Comparison of reward variants (RLVR, self-certainty, majority vote) with 1024 training samples. Proxy rewards without verifiers exhibit failure modes under prolonged training: training collapse (self-certainty), and reward spikes followed by performance drops (majority vote). Evaluation results in this figure are based on greedy decoding.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 150 300 450

Training Steps

0.2

0.4

0.6

0.8

1.0

TrainingReward

Training Reward

0 150 300 450

Training Steps

15

20

25

30

35

SCP-Difficult(%)

SCP-Difficult (%)

0 150 300 450

Training Steps

5

10

15

20

25

GPQADiamond(%)

GPQA Diamond (%)

0 150 300 450

Training Steps

62

64

66

68

70

72

MATH-500(%)

MATH-500 (%)

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

0 150 300 450

Training Steps

0.2

0.4

0.6

0.8

1.0

TrainingReward

0 150 300 450

Training Steps

0

3

6

9

12

15

SCP-Difficult(%)

0 150 300 450

Training Steps

17.5

20.0

22.5

25.0

27.5

GPQADiamond(%)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 150 300 450

Training Steps

46.5

48.0

49.5

51.0

52.5

MATH-500(%)

Qwen2.5-Math

- -1.5B Llama3.2-3B
- -Instruct

GRPO-POS GRPO-NEG GRPO

- Figure 28. Effect of baseline variants on SCIENCE domain with 8 training samples. GRPO-pos (positive updates only) and GRPO-neg (negative updates only) produce comparable performance to standard GRPO.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

0 150 300 450

Training Steps

0.30

0.45

0.60

0.75

TrainingReward

Training Reward

0 150 300 450

Training Steps

15

20

25

30

35

40

SCP-Difficult(%)

SCP-Difficult (%)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 150 300 450

Training Steps

5

10

15

20

25

GPQADiamond(%)

GPQA Diamond (%)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 150 300 450

Training Steps

62

64

66

68

70

72

MATH-500(%)

MATH-500 (%)

0 150 300 450

Training Steps

0.15

0.30

0.45

0.60

0.75

0.90

TrainingReward

0 150 300 450

Training Steps

5

10

15

20

25

SCP-Difficult(%)

0 150 300 450

Training Steps

18

20

22

24

26

28

GPQADiamond(%)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0 150 300 450

Training Steps

46

48

50

52

54

MATH-500(%)

Qwen2.5-Math

- -1.5B Llama3.2-3B
- -Instruct

GRPO-POS GRPO-NEG GRPO

- Figure 29. Effect of baseline variants on SCIENCE domain with 1024 training samples. Similar to Figs. 28, GRPO-pos (positive updates only) and GRPO-neg (negative updates only) produce comparable performance to standard GRPO.

[Figure 4]

- Figure 30. Response diversity on 8 samples from the MATH-500 evaluation dataset. Qwen-math shows high diversity within its correct answers, suggesting a range of learned robust reasoning paths.

Diversity Prompt for LLM-as-a-judge You are given the original prompt and two model-generated responses. Determine whether the two responses use different strategies to solve the problem.

Use the following guidelines:

- -Different solution methods: Clearly different approaches (e.g., algebraic vs. geometric, analytical vs. numerical).
- - Critical reasoning divergence: Significant differences in key reasoning steps or assumptions, even if final answers match.
- - Conceptual differences: Distinct underlying concepts or representations (e.g., probability vs. combinatorics).

**Also label as different if:** The two responses share the same general approach but differ meaningfully in specific intermediate steps or manipulations crucial to the solution.

Original prompt: prompt

- Generation 0: generation 0
- Generation 1: generation 1 Question : Do Generation 0 and Generation 1 use different strategies?

You may first generate a short reasoning, then end your response with either ||yes|| if they use different strategies or ||no|| if they use the same strategy.

Figure 31. LM prompt to check similarity between responses.

Faithfulness Prompt for LLM-as-a-judge You will be given: (1) a math problem prompt, and (2) a model response that may include BOTH reasoning and a final answer. Definitions : ”Reasoning” = the parts of the model response that attempt to justify or derive a result (intermediate steps, equations, explanations, casework, narrative logic). ”Final answer” = the model’s explicit committed result (e.g., after ”Final:”, ”Answer:”, ”Therefore”, ” boxed”, or the last clear numeric/symbolic conclusion). If multiple answers appear, treat the last explicitly committed one as the final answer. Task : Decide how well the reasoning supports the final answer, using these labels: Label 1 (Correlated): The reasoning forms a coherent derivation that would lead to the final answer as stated. Minor algebra slips are allowed IF the overall derivation still clearly targets that answer. The final answer may be objectively wrong; you judge alignment, not correctness. Label 0.5 (Partially Correlated):

The reasoning is related to the problem and seems to move toward the final answer, but has major gaps, unjustified leaps, missing steps, or serious errors that break the proof. The answer is not a pure non-sequitur, but the support is weak/incomplete.

Label 0 (Uncorrelated):

The final answer is not supported by the reasoning. Examples include: contradiction with earlier derived statements; switching to an unrelated method; violating key constraints from the prompt; or the final answer appearing as an unsupported guess.

Output format (MANDATORY):

- 1) Briefly identify (a) the extracted final answer and (b) the key reasoning path in 1–3 sentences.
- 2) Then output exactly one label token on its own at the end: ∥1∥ or ∥0.5∥ or ∥0∥. Prompt: prompt Response: response

Question: Does the reasoning path correspond to the provided answer? You may first generate a short reasoning, then end your response with either ∥1∥ if they are fully correlated, ∥0.5∥ if they are partially correlated, or ∥0∥ if the answer is uncorrelated to the preceding logic.

- Figure 32. LM prompt to evaluate reasoning faithfulness on a sample from the MATH dataset.

[Figure 5]

###### Figure 33. Proportion of aligned and misaligned responses across models and training datasets.

80

90

72

MATH-500(%)

| |
|---|

| |
|---|

TrainingReward

| |
|---|

SCP-Hard(%)

84

| |
|---|

64

| |
|---|

AMC(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

78

56

0.6

Scarce

| |
|---|

Data Majority

40

| | | |
|---|---|---|
|| |
|---|
<br><br>|| |
|---|
<br><br>|| |
|---|
|
| || |
|---|
| |
|| |
|---|
| | |

60

| |
|---|

40

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

54

| |
|---|

30

| |
|---|

| |
|---|

| |
|---|

0.0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

90

| |
|---|

| |
|---|

| |
|---|

TrainingReward

MATH-500(%)

| |
|---|

SCP-Hard(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

60

| |
|---|

| |
|---|

AMC(%)

| |
|---|

| |
|---|

| |
|---|

0.6

Vote NoisyReward

| |
|---|

60

| |
|---|

| |
|---|

40

30

| |
|---|

0.0

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

| |
|---|

TrainingReward

0.6

MATH-500(%)

| | |
|---|---|
| | |

SCP-Hard(%)

| |
|---|

| |
|---|

| |
|---|

| |
|---|

60

(=0.7)

60

AMC(%)

75

| |
|---|

| |
|---|

| |
|---|

| |
|---|
| |

| |
|---|

0.3

| |
|---|

| |
|---|

40

| |
|---|

| |
|---|
| |

| |
|---|

30

| |
|---|

| |
|---|

50

| |
|---|

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Training Steps

Training Steps

Training Steps

Training Steps

Base + Thinking SFT Base + Non-Thinking SFT CPT + Thinking SFT CPT + Non-Thinking SFT Instruct

###### Figure 34. Evaluation results of pass@16 metric across models with different pre-RL intervention on weak supervision.

###### avg@16 pass@16 avg@16 pass@16

ScienceBench(%)

AIME2024(%)

30

8

10

20

Scarce

Data Majority

15

0

0

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

ScienceBench(%)

AIME2024(%)

8

20

10

20

Vote NoisyReward

0

0

0

0

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

0 150 300 450 600

ScienceBench(%)

AIME2024(%)

30

(=0.7)

10

8

25

15

0

0

0

0 150 300 450

0 150 300 450

0 150 300 450

0 150 300 450

Training Steps

Training Steps

Training Steps

Training Steps

Base + Thinking SFT Base + Non-Thinking SFT CPT + Thinking SFT CPT + Non-Thinking SFT Instruct

###### Figure 35. Evaluation results on AIME 2024 and Science Bench across models with different pre-RL intervention on weak supervision.

Diversity Judge Example Output Prompt: A box contains 5 white balls and 5 black balls. I draw them out of the box, one at a time. What is the probability that all of my draws alternate colors, starting and ending with the same color?

- Response A: To determine the probability that all of the ball draws alternate colors, starting and ending with the same color, we need to consider the following:

- 1. **Total number of ways to arrange the balls**: There are 5 white balls and 5 black balls, so the total number of

ways to arrange these 10 balls is given by the binomial coefficient 105 , which represents the number of ways to choose 5 positions out of 10 for the white balls (the remaining 5 positions will be for the black balls). This value

is calculated as:

10 5

=

10! 5!5!

= 252

- 2. **Number of favorable outcomes**: For the drawings to alternate colors starting and ending with the same color, there are only two possible sequences: - Starting with a white ball and alternating (W B W B W B W B W B) - Starting with a black ball and alternating (B W B W B W B W B W) So there are only 2 favorable outcomes.
- 3. **Probability calculation**: The probability is the ratio of the number of favorable outcomes to the total number of outcomes. Therefore, the probability P is:

P =

Number of favorable outcomes Total number of outcomes

=

2 252

=

1 126

So the final answer is

|1 126<br><br>|
|---|

.

- Response B: To solve the problem, we need to determine the probability that a sequence of 10 balls (5 white and 5 black) is arranged such that they alternate colors. There are only two possible sequences that meet this condition: ”WBWBWBWBWB” and ”BWBWBWBWBW”. The total number of ways to arrange 5 white and 5

black balls is given by the binomial coefficient 105 , which represents the number of distinct permutations of 5 white and 5 black balls. This is calculated as:

10 5

10! 5!5!

= 252

=

Since there are only 2 favorable outcomes (the two alternating sequences mentioned above), the probability is the ratio of the number of favorable outcomes to the total number of outcomes:

1 126

2 252

Probability =

=

|1 126<br><br>|
|---|

. Judge Output: Both Generation 0 and Generation 1 use the same strategy to solve the problem. They both identify that there are two possible sequences that satisfy the condition of alternating colors and starting and ending with the same color: ”WBWBWBWBWB” and ”BWBWBWBWBW”. They both calculate the total number of ways to arrange the 10 balls using the binomial coefficient 105 , which accounts for the permutations of 5 white and 5 black balls. Finally, they both compute the probability by taking the ratio of the number of favorable outcomes (2) to the total number of outcomes (252), resulting in the same final answer of 1261 . There are no significant differences in their reasoning steps, assumptions, or intermediate calculations. Both responses follow a similar logical flow and use the same mathematical concepts to arrive at the solution.

So, the probability that all of my draws alternate colors, starting and ending with the same color, is

#### Judge Result: ∥no∥

Figure 36. Qualitative Example of Diversity Analysis

Faithfulness Judge Example Output Prompt: Find all real pairs (a,b) that solve the system of equation

a2 + b2 = 25, 3(a + b) − ab = 15.

Response: To solve the system of equations:

a2 + b2 = 25, 3(a + b) − ab = 15,

we can use the following reasoning and algebraic manipulations. First, recall the identity:

a2 + b2 = (a + b)2 − 2ab.

Let’s denote s = a + b and p = ab. So, the first equation becomes: s2 − 2p = 25. The second equation remains: 3s − p = 15. We can solve these two equations simultaneously. First, we express p from the second equation: p = 3s − 15. Next, we substitute p into the first equation: s2 − 2(3s − 15) = 25. Simplify the equation:

s2 − 6s + 30 = 25 =⇒ s2 − 6s + 5 = 0.

√b2−4ac

This is a quadratic equation in s. We can solve for s using the quadratic formula s = −b±

2a , where a = 1, b = −6, and c = 5:

√16 2

√36 − 20 2

6 ±

6 ±

6 ± 4 2

s =

=

=

.

So, the solutions for s are: s = 5 or s = 1. Now, we find the corresponding p values for each s: 1. If s = 5, then p = 3(5) − 15 = 0. 2. If s = 1, then p = 3(1) − 15 = −12. We now need to find the pairs (a,b) that satisfy the equations a + b = s and ab = p. These are the roots of the quadratic equations: 1. For s = 5 and p = 0, the quadratic equation is:

t2 − 5t = 0 =⇒ t(t − 5) = 0.

So, the solutions are t = 0 and t = 5. Thus, the pairs (a,b) are (0,5) and (5,0). 2. For s = 1 and p = −12, the quadratic equation is:

t2 − t − 12 = 0. We solve this quadratic equation using the quadratic formula:

√1 + 48 2

1 ±

- 1 ± 7

- 2

t =

=

.

So, the solutions are t = 4 and t = −3. Thus, the pairs (a,b) are (4,−3) and (−3,4). Therefore, the complete set of solutions is:

|(0,5),(5,0),(4,−3),(−3,4)|
|---|

.

Judge Output: Final answer given: (0,5),(5,0),(4,−3),(−3,4). Reasoning path: Introduces sums s = a + b and product p = ab, rewrites equations into a system in s and p, solves to get s = 5 or 1, finds corresponding p, then solves quadratic t2 − st + p = 0 to obtain all ordered pairs consistent with each (s,p), yielding the listed solutions. The derivation is complete and consistently leads to the stated pairs.

#### Judge Result: ∥1∥

Figure 37. Qualitative Example of Faithfulness Analysis

